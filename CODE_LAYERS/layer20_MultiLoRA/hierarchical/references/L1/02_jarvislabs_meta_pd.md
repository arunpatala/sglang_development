# Disaggregated Prefill-Decode: The Architecture Behind Meta's LLM Serving

**Source:** https://jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode
**Author:** Vishnu Subramanian (Jarvis Labs), January 29, 2026 — 11-minute read
**Level:** L1 — Practitioner research notes; architecture diagrams; KV size math; network table
**Why here:** Written as a practitioner's research notes from studying the vLLM router repository. Provides the most concrete KV cache size calculation of any L1 article, a full network bandwidth comparison table showing which interconnects are required, Mermaid flowcharts of the architecture, and a per-GPU-type optimization table. Confirms that Meta, LinkedIn, Mistral, and HuggingFace are running vLLM with disaggregated serving in production.

---

## The Problem: Simple Scaling Stops Working

Standard deployment: pick vLLM or SGLang, choose a model, spin it up. Add workers and load balancing as traffic grows. Performance improves — but not as much as expected. The reason is the fundamental asymmetry of LLM inference.

---

## The Two Phases and Their Mismatch

### Prefill (Parallel)
- Processes all input tokens simultaneously
- Massive matrix multiplications across all input tokens
- **Compute-bound**: tensor cores are maxed out
- Bottleneck: GPU tensor core throughput
- Impact metric: Time to First Token (TTFT)

### Decode (Sequential)
- Generates tokens one at a time — each token depends on the previous
- Reads KV cache + model weights repeatedly; tiny compute per step
- **Memory-bound**: GPU waits for memory reads, not for compute
- Bottleneck: HBM bandwidth
- Impact metric: Inter-Token Latency (ITL)

| Aspect | Prefill | Decode |
|---|---|---|
| Token processing | Parallel (all at once) | Sequential (one by one) |
| GPU bottleneck | Compute-bound | Memory-bound |
| Latency impact | TTFT | ITL |
| Batch efficiency | High | Low |

---

## The Disaggregated Architecture

```
Incoming Request → Router
Router → Prefill Workers (compute-optimized)
Prefill Workers → [KV Cache Transfer] → Decode Workers (memory-optimized)
Decode Workers → Response Stream
```

### Three Steps

**Step 1: Prefill**
- Request arrives at router
- Router sends to a prefill worker
- Prefill worker processes entire prompt and builds KV cache

**Step 2: Handoff**
- Router notifies an available decode worker
- Decode worker connects directly to the prefill worker
- KV cache transfers from prefill worker to decode worker

**Step 3: Decode**
- Decode worker generates tokens sequentially using the transferred KV cache
- Streams response back to client

---

## Why This Works: Independent Scaling

```
Scenario A: Short prompts, long responses
  1–2 Prefill Workers + 4–5 Decode Workers

Scenario B: Long prompts, short responses
  4–5 Prefill Workers + 1–2 Decode Workers
```

Scale each pool based on your actual workload — don't add capacity for both phases together.

### Per-Worker Optimization Table

| Optimization | Prefill Worker | Decode Worker |
|---|---|---|
| Batch size | Large batches | Small batches |
| GPU type | High compute (H100) | High memory bandwidth |
| Scheduling | Batch similar-length prompts | Continuous batching |
| Memory | Temporary KV cache (per-request) | Persistent KV cache (long-lived) |

### Real-World Performance Results

| System | Key Result | Source |
|---|---|---|
| DistServe | 7.4× more requests served, 12.6× better SLO compliance | OSDI 2024 |
| Splitwise | 2.35× throughput same cost; or 1.4× throughput at 20% lower cost | Microsoft Research |
| Mooncake | 525% throughput improvement with KV cache disaggregation | Moonshot AI |
| SGLang + DeepSeek-R1 | 52.3k input tokens/s, 22.3k output tokens/s on 96 H100s | SGLang Blog |

**Production adopters:** Meta, LinkedIn, Mistral, HuggingFace (vLLM-based); NVIDIA Dynamo (GTC 2025).

---

## The Hard Part: KV Cache Transfer Math

### KV Cache Size Calculation (Llama-3.1-70B)

```
KV cache per token:
├── 80 layers
├── 8 KV heads per layer
├── 128 dimensions per head
├── 2 (K and V)
├── 2 bytes (FP16)
└── = 80 × 8 × 128 × 2 × 2 = 327,680 bytes per token

For a 4K-token prompt:
└── 4,096 × 327,680 = 1.34 GB of KV cache
```

1.34 GB must transfer fast enough that the decode worker isn't sitting idle.

### Network Bandwidth Requirements

For TTFT < 500ms with 200ms prefill time, you have ~300ms for KV transfer:
```
Required bandwidth = 1.34 GB / 0.3 seconds ≈ 4.5 GB/s minimum
```

| Network Type | Bandwidth | Transfer Time (1.34 GB) |
|---|---|---|
| 1 GbE | 125 MB/s | **10.7 seconds** — completely unusable |
| 10 GbE | 1.25 GB/s | **1.07 seconds** — too slow |
| 25 GbE | 3.125 GB/s | **0.43 seconds** — borderline |
| 100 GbE | 12.5 GB/s | **0.11 seconds** — acceptable |
| InfiniBand HDR | 25 GB/s | **54 ms** — good |
| NVLink (within node) | 600 GB/s | **2.2 ms** — ideal |

**Takeaway: Internet connections won't work. You need co-located workers on the same rack with InfiniBand or NVLink.**

---

## Implementation: vLLM Router

Meta's implementation in the [vLLM router](https://github.com/vllm-project/router), building on SGLang's router work:

- **Smart request routing**: classifies requests by input token count, expected output length, current worker utilisation.
- **Direct worker communication**: after prefill, the decode worker connects directly to the prefill worker for KV transfer — bypassing the router to minimise latency.
- **Health monitoring**: tracks GPU utilisation, KV cache memory usage, queue depths, transfer throughput.

---

## When to Use (and When Not To)

### Good Use Cases
- High-throughput production serving: many concurrent requests, need consistent latency
- Variable prompt lengths: mix of short and long prompts, summarisation, RAG
- Strict SLAs: requirements on TTFT and ITL that standard serving can't meet

### Not Worth the Complexity For
- Development and testing: low request volume → standard vLLM is fine
- Small models: KV cache is small enough that transfer isn't a bottleneck
- Batch processing: TTFT doesn't matter, just throughput

---

## Infrastructure Requirement

This pattern requires:
- **Co-located workers**: same rack, same datacenter zone
- **High-speed interconnect**: InfiniBand, NVLink, or at minimum 100 GbE
- **Shared memory architecture** (within node): for faster intra-node transfers
- **Coordinated scheduling**: workers must be aware of each other

Current GPU cloud platforms (including JarvisLabs at article time) assume isolated workers — KV transfer over the internet is too slow. The industry is moving toward interconnected worker pool infrastructure.

---

## Key Takeaways for Layer 19

- The KV cache size formula (`n_layers × n_kv_heads × head_dim × 2 × 2 × seq_len`) is the fundamental sizing tool — run it for your specific model before evaluating disaggregation.
- **NVLink (2.2ms)** vs **InfiniBand (54ms)** vs **100 GbE (110ms)**: the interconnect choice determines TTFT floor.
- Independent scaling is the primary economic argument: scale prefill workers for prompt-heavy workloads, decode workers for response-heavy workloads — not both as a unit.
- The 5-step how-it-works (prefill → handoff → decode) is the correct mental model for understanding SGLang's `--disaggregation-mode prefill/decode` flags.
- Meta, LinkedIn, Mistral, and HuggingFace are already running this in production — it is not experimental.
