# Prefill Is Compute-Bound. Decode Is Memory-Bound. Why Your GPU Shouldn't Do Both.

**Source:** https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/
**Author:** Gokul Chandra Purnachandra Reddy
**Published:** April 15, 2026 — 16-minute read
**Level:** L1 — Production-grounded analysis; compute vs memory arithmetic; decision framework
**Why here:** The most complete L1 article on PD disaggregation. Opens with a real production failure (64 H100 GPUs, tensor cores at 92% during prefill → 28% during decode), provides quantified hardware utilisation numbers from InfoQ and academic papers, walks through the full KV cache cost calculation in Python, and closes with a 5-check deployment decision framework. Cites Perplexity, Meta, LinkedIn, Mistral, and NVIDIA Dynamo as production adopters.

---

## The Core Observation

> "A Llama 70B model running inference on an H100 GPU hits 92% compute utilization during prefill. Thirty milliseconds later, during decode, that same GPU drops to 30%. The hardware hasn't changed. The model weights are identical. The arithmetic intensity of the workload fell by 5x between one phase and the next."

| Phase | GPU utilisation (H100) | Arithmetic intensity | Bottleneck |
|---|---|---|---|
| Prefill | 90–95% | 200–400 ops/byte | Tensor cores (compute) |
| Decode | 20–40% | 60–80 ops/byte | HBM bandwidth (memory) |

**Source:** InfoQ technical analysis, September 2025. Also cited: SPAD paper (UT Austin) — reducing memory bandwidth by 40% increased prefill latency only 17% (because prefill doesn't use bandwidth); reducing compute by 50% increased decode latency only 22% (because decode doesn't use compute).

---

## What Monolithic Serving Costs You

**Immediate problem — interference:** When a new prefill request enters the batch, active decode requests stall. Prefill is compute-heavy and takes longer per step. Users watching streaming responses see text pause mid-sentence while the GPU processes someone else's prompt.

**Slower-burning problem — utilisation waste:**
- A typical generation produces 200–500 output tokens.
- Each decode step: 10–30ms. A 300-token response: 3–9 seconds decode + ~200ms prefill.
- The GPU runs decode for 90%+ of wall-clock time at 30% compute utilisation.
- You're paying for H100-level compute and only using it 10% of request duration.

**Chunked prefill (vLLM) smooths the worst spikes** but doesn't solve the utilisation mismatch — the GPU still does both jobs.

---

## Splitting the Inference Path

Disaggregated inference: separate pools, connected by a fast network.

**Three components:**

1. **KV-aware router** — routes requests to available prefill workers; after prefill completes, routes the KV cache to a decode worker. Tracks which decode workers hold which caches (enables prefix caching across requests sharing system prompts).

2. **Prefill pool** — GPUs optimised for high-throughput matrix multiplication. Process prompts, build KV caches, hand off. Never generate output tokens. Need high FLOPS, not massive HBM.

3. **Decode pool** — GPUs optimised for memory bandwidth. Receive KV caches, generate tokens autoregressively. Large batch sizes amortise HBM reads across many concurrent requests.

**Independent scaling:** prompt-heavy workload → add prefill workers. Many concurrent users with long responses → add decode workers.

---

## The KV Cache Transfer Tax

The KV cache must move from prefill GPU to decode GPU. It is not small.

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes=2):
    """Returns KV-cache size in bytes for a single request."""
    per_token = n_layers * n_kv_heads * head_dim * 2 * dtype_bytes  # 2 for K and V
    return per_token * seq_len

# Llama-3.1-70B with GQA (80 layers, 8 KV heads, 128 dims, FP16)
cache = kv_cache_bytes(n_layers=80, n_kv_heads=8, head_dim=128, seq_len=4096)
# Per token: 80 × 8 × 128 × 2 × 2 = 327,680 bytes
# Full 4K-token prompt: 1.34 GB

# Llama-3.1-8B (smaller model)
cache_8b = kv_cache_bytes(n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096)
# 0.54 GB
```

**Transfer time at different bandwidths:**

| Network | Bandwidth | Transfer time (1.34 GB) |
|---|---|---|
| 100 Gbps (standard RDMA / EFA) | 12.5 GB/s | **107 ms** |
| 400 Gbps (high-end RDMA) | 50 GB/s | **27 ms** |

**Is the transfer worth it?** DistServe P99 latency measurements show 2×+ improvement with disaggregation because eliminating prefill-decode interference removes the tail latency spikes that queuing causes. **The 27–107ms transfer cost replaces 200–500ms of queuing delay.**

**Perplexity's optimisation:** Layer-pipelined KV transfer — decode worker starts processing early layers while later layers are still in transit over RDMA (libfabric + cuMem). This reduces effective TTFT below the raw bandwidth calculation.

---

## Production Stack (as of early 2026)

**Request lifecycle for Llama-70B (4K-token prompt, 300-output-token response):**
```
Router receives request
→ Prefill worker: 150–400ms (compute-bound, 4K tokens)
→ RDMA transfer: 27–107ms (1.34 GB KV cache)
→ Decode worker: 3–9 seconds (300 tokens, memory-bound)
→ Output stream to client
```
Decode dominates wall time (90%+). Prefill + transfer is a small fraction.

**Production adopters (cited in article):**
- **Perplexity** — 435M queries/month, RDMA-based disaggregated serving
- **Meta, LinkedIn, Mistral** — vLLM with disaggregated serving in production
- **NVIDIA** — Dynamo (GTC 2025) built specifically for this pattern
- **SGLang** — 96 H100 GPUs (24 prefill + 72 decode), 52,300 input tokens/sec

**Infrastructure options:**
- **vLLM** — built-in `kv_transfer_config` JSON flag
- **NVIDIA Dynamo** — datacenter-scale orchestration (vLLM/SGLang/TensorRT-LLM backends)
- **llm-d** (Red Hat + IBM Research) — Kubernetes CRDs for disaggregated serving

---

## When Disaggregation Makes Things Worse

> "Not every workload benefits. The BentoML inference handbook reports 20–30% performance degradation when disaggregation is applied to workloads that don't need it."

**Avoid disaggregation when:**
- Median prompt < 512 tokens and generation < 100 tokens — KV transfer overhead exceeds savings.
- Prefix cache hit rate > 80% — the KV cache is already local on the decode worker from previous turns.
- < 16 GPUs — scheduling overhead of two pools exceeds utilisation gains.
- No RDMA — TCP increases transfer latency, potentially negating the benefit.

---

## The 5-Check Deployment Decision Framework

Before refactoring your serving stack:

1. **Measure prefill-to-decode time ratio.** If decode < 70% of wall-clock time, disaggregation has smaller payoff. If decode > 85%, you're wasting tensor cores most of the day.

2. **Calculate KV cache transfer size** using the formula above. If > 500 MB/request and network < 100 Gbps, transfer latency eats into TTFT budget.

3. **Check prefix cache hit rate.** Hit rates > 80% reduce the value of a separate prefill pool.

4. **Count your GPUs.** Below 16 GPUs, scheduling overhead typically exceeds gain. Above 32 with sustained traffic, cost savings compound.

5. **Audit your network.** RDMA-capable NICs (EFA on AWS, ConnectX on bare metal)? If limited to TCP, disaggregation can still work for long-context workloads but with higher transfer latency.

If checks 1, 4, and 5 are all favorable → disaggregation will almost certainly reduce per-token cost.

---

## Cost Arithmetic

- **Prefill workers**: need high FLOPS, not massive HBM → H100 SXM is well-suited; H200 is overkill.
- **Decode workers**: need high HBM bandwidth + large capacity for batching → H200 (larger HBM3e) is actually *better* than H100 for decode.
- **SPAD paper estimate**: prefill chip with 40% less HBM bandwidth loses only 17% prefill performance; decode chip with 50% less compute loses only 22% decode performance.
- **InfoQ cluster-level analysis**: 15–40% total infrastructure cost reduction from disaggregation.

---

## Key Takeaways for Layer 19

- The 5× arithmetic intensity drop between prefill and decode is not an approximation — it is a structural property of autoregressive generation that cannot be eliminated by scheduling.
- The 90%+ compute utilisation during prefill and 20–40% during decode are measured numbers, not estimates.
- The KV transfer overhead (27–107ms at RDMA bandwidth) is smaller than the queuing delay it replaces (200–500ms at production P99) — this is why disaggregation improves latency despite adding a network hop.
- Disaggregation is a **latency and cost architecture**, not a throughput architecture. Total tokens/GPU-hour is roughly the same; what changes is which GPUs you pay for and whether they are used efficiently.
- The 5-check framework is the correct first step before any infrastructure change.
