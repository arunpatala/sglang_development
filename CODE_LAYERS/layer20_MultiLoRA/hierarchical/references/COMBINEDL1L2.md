# Prefill-Decode Disaggregation — From Why to How

**Level:** L1 + L2 — Concept, measurement, architecture, and deployment. No paper math; no source code internals.

**What this file is:** A single coherent blog synthesising all L1 and L2 source material into a progressive narrative. Sections build from "what are prefill and decode?" to "how do I launch a disaggregated SGLang cluster?" Sections above L2 (DistServe formal goodput derivation, Mooncake RDMA internals, SARATHI pipeline analysis, TaiChi SLO regimes, vLLM connector architecture, expert parallelism dispatch internals) are deliberately left out.

**Sources synthesised:**
- L1/01 — TDS: Prefill Is Compute-Bound, Decode Is Memory-Bound (April 2026)
- L1/02 — Jarvis Labs: The Architecture Behind Meta's LLM Serving (January 2026)
- L1/03 — Jackson MZ: Practical Guide to LLM PD Disaggregation
- L1/04 — NADDOD: Understanding PD Disaggregation (August 2025)
- L1/05 — BentoML LLM Inference Handbook
- L1/06 — LearnCodeCamp: Prefill, Decode, TTFT, and ITL
- L1/07 — LLM Inference Performance Estimator (interactive roofline tool)
- L1/08 — NVIDIA Spotlight: Perplexity AI at 435M Queries/Month
- L1/09 — Hao Zhang: CMU LLM Systems Lecture Slides (DistServe PI)
- L2/01 — SGLang PD Disaggregation Official Documentation
- L2/02 — LMSYS Blog: DeepSeek on 96 H100 GPUs (May 2025)

**Omitted (above L2):** DistServe paper formal analysis, Splitwise production trace model, Mooncake multi-NIC RDMA topology selection, NIXL UCX/GDS internals, SARATHI chunked-prefill pipeline mechanics, TaiChi SLO-regime switching logic, vLLM connector architecture, expert parallelism dispatch patterns.

---

## Section Plan

| § | Title | Sources | Reading time |
|---|-------|---------|------|
| 1 | [The Two Phases of Every LLM Request](#1-the-two-phases-of-every-llm-request) | L1/03, L1/06, L1/04 | 3 min |
| 2 | [The Metrics That Matter: TTFT and ITL](#2-the-metrics-that-matter-ttft-and-itl) | L1/06, L1/09 | 3 min |
| 3 | [Why One GPU Can't Do Both Well](#3-why-one-gpu-cant-do-both-well) | L1/01, L1/07, L1/06 | 4 min |
| 4 | [The Hidden Cost of Running Both Phases Together](#4-the-hidden-cost-of-running-both-phases-together) | L1/01, L1/04, L1/05, L1/09 | 4 min |
| 5 | [Goodput: The Metric That Actually Matters](#5-goodput-the-metric-that-actually-matters) | L1/09 | 2 min |
| 6 | [How Disaggregation Works](#6-how-disaggregation-works) | L1/02, L1/03, L1/04, L1/09 | 4 min |
| 7 | [The KV Cache Transfer Tax](#7-the-kv-cache-transfer-tax) | L1/01, L1/02 | 4 min |
| 8 | [The Handshake: How SGLang Moves the KV Cache](#8-the-handshake-how-sglang-moves-the-kv-cache) | L2/01, L2/02 | 4 min |
| 9 | [Launching a Disaggregated Cluster](#9-launching-a-disaggregated-cluster) | L2/01 | 4 min |
| 10 | [Production Evidence at Scale](#10-production-evidence-at-scale) | L1/08, L2/02, L1/01, L1/02 | 4 min |
| 11 | [When Disaggregation Makes Things Worse](#11-when-disaggregation-makes-things-worse) | L1/01, L1/05 | 3 min |
| 12 | [The Decision Framework and Cost Arithmetic](#12-the-decision-framework-and-cost-arithmetic) | L1/01, L1/02 | 3 min |

**Total reading time:** ~42 minutes

---

## 1. The Two Phases of Every LLM Request

When you send a message to an LLM, the model doesn't process it in a single continuous operation. Every request — every prompt, every conversation turn — passes through exactly two distinct phases before you see any output.

### Phase 1: Prefill (Prompt Processing)

When a prompt arrives, the model processes the **entire input sequence all at once**, in parallel. Every token in the prompt is processed simultaneously.

For each token, in each attention layer, the model computes three vectors: a Query (Q), a Key (K), and a Value (V). The K and V vectors for every input token are stored in GPU memory — this stored structure is the **KV cache**. It is the model's working memory for the current request.

At the end of prefill, the KV cache is complete and the model has not yet produced a single output token. Prefill is pure setup: building the context the model needs to generate.

> "To make sure inference is not compute- and bandwidth-bound at the same time, we want to separate them." — Jackson MZ

### Phase 2: Decode (Token Generation)

After prefill, the model enters **decode mode**: generating output tokens one at a time, sequentially. Each decode step:

1. Takes the current sequence (prompt + all tokens generated so far)
2. Predicts the next token
3. Appends that token to the sequence
4. Computes K and V for the new token and adds them to the KV cache
5. Repeats until a stop condition is reached

Because each new token depends on all previous ones, decode **cannot be parallelised across output tokens**. You must generate them one by one. This is the autoregressive constraint — a fundamental property of how transformer language models work.

**Why the KV cache is essential for decode:** Without it, every decode step would require re-computing attention over the entire growing sequence from scratch — O(n²) cost that grows with every token generated. With the cache, each decode step only computes attention for the single new token and attends to the cached past — roughly O(n) total cost across all decode steps. The KV cache is what makes decode tractable.

| Phase | What it does | How it processes tokens | Output |
|---|---|---|---|
| Prefill | Processes the input prompt | In parallel — all tokens at once | KV cache |
| Decode | Generates output tokens | Sequentially — one token per step | Output text |

---

## 2. The Metrics That Matter: TTFT and ITL

Two user-facing metrics capture the performance of LLM inference — and they map directly to the two phases.

### Time to First Token (TTFT)

TTFT is the elapsed time from **when the request is submitted** to **when the first output token appears** in the response stream.

- TTFT ≈ prefill time + the compute time for the first decode step.
- Long prompts → higher TTFT (more input tokens to process in prefill).
- This is the metric users notice most: the "thinking" pause before any text appears.

### Inter-Token Latency (ITL) — also called TPOT

ITL (Inter-Token Latency) or TPOT (Time Per Output Token) is the **average time between consecutive output tokens** once generation has started.

- ITL is dominated entirely by the decode phase.
- Lower ITL = text streams in faster = the response feels snappier and more continuous.
- Typical values on high-end hardware: 20–100ms per token (10–50 tokens per second).

**Together, they define user experience:**

| Metric | What it captures | Dominated by | User impact |
|---|---|---|---|
| TTFT | Time until response starts | Prefill | Perceived "hang" before reply |
| ITL / TPOT | Streaming speed | Decode | Smooth vs stuttering text stream |

A system with low TTFT feels responsive. A system with low ITL feels fast. You want both — and they are controlled by different hardware bottlenecks.

---

## 3. Why One GPU Can't Do Both Well

This is the root of the problem. Prefill and decode don't just have different characteristics — they have **opposite hardware requirements**.

### Prefill Is Compute-Bound

During prefill, the model performs large matrix multiplications over the full prompt length. For a 4,096-token prompt, this means computing a full S×S attention matrix per head per layer — at S=4,096, that's 16.7 million multiply-add operations per head per layer.

The bottleneck is the GPU's tensor cores — the hardware is "embarrassingly parallel" and they are saturated. Memory bandwidth is barely used because the computation is so dense relative to the data being moved.

**Measured numbers on H100 SXM (InfoQ analysis, September 2025):**
- GPU compute utilisation during prefill: **90–95%**
- Arithmetic intensity: **200–400 FLOP/byte**

### Decode Is Memory-Bound

During decode, each step generates exactly one token. The matrix multiplications are tiny (shape [1 × d] against the full KV cache). There is very little computation per step. But the model must **load the entire KV cache from HBM** on every single decode step.

The bottleneck shifts from the tensor cores to memory bandwidth — the model is waiting for data to arrive from HBM, not for compute to finish.

**Measured numbers on H100 SXM:**
- GPU compute utilisation during decode: **20–40%**
- Arithmetic intensity: **60–80 FLOP/byte** (compared to 200–400 during prefill)

**The roofline formula:**

```
Prefill arithmetic intensity ≈ SeqLen FLOP/byte
(scales with prompt length — longer prompts are more compute-bound)

Decode arithmetic intensity ≈ 1 FLOP/byte
(constant regardless of sequence length — always memory-bound)
```

This is the key formula. `Decode AI ≈ 1 FLOP/byte` means decode will always sit below the memory bandwidth ceiling on the roofline plot, regardless of how many tensor cores the GPU has. FLOPS are irrelevant for decode. What matters is how fast the GPU can move data from HBM to the compute units.

**The practical consequence:** A Llama-3.1-70B model on an H100 hits 92% compute utilisation during prefill. Thirty milliseconds later, during decode, the same GPU drops to 30%. Same hardware, same model weights, same kernel — the arithmetic intensity of the workload fell by 5× between one phase and the next.

> "The hardware hasn't changed. The model weights are identical. The arithmetic intensity of the workload fell by 5× between one phase and the next." — TDS, April 2026

You are paying for an H100 and getting meaningful work from perhaps 20–30% of it for 90% of every request's lifetime.

---

## 4. The Hidden Cost of Running Both Phases Together

The standard deployment is **monolithic serving**: a single pool of GPUs handles both prefill and decode. The scheduler interleaves incoming prefill requests with ongoing decode steps in the same batch.

This creates two distinct problems.

### Problem 1: Real-Time Interference

When a new prefill request enters the batch, it occupies the GPU's tensor cores. Decode steps from other requests must wait. Because prefill is compute-heavy and takes longer per iteration, active decode requests experience a **latency spike** — users watching a streaming response see the text pause mid-sentence while the GPU processes someone else's prompt.

The interference runs both directions:
- **Prefill → decode interference:** a large prefill entering the batch increases ITL for all active decode requests.
- **Decode → prefill interference:** when the running queue of decode requests is large, the scheduler deprioritises new prefill requests to protect existing ITL — increasing TTFT for new arrivals.

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." — BentoML

The key insight: TTFT and ITL are controlled by different hardware bottlenecks. Optimising one phase on shared hardware forces a compromise on the other. There is no scheduling policy that can eliminate this tension on a single GPU pool.

### Problem 2: Sustained Utilisation Waste

The interference problem appears per-request. The utilisation waste is a constant, structural overhead.

A typical generation produces 200–500 output tokens. Each decode step takes 10–30ms. A 300-token response spends roughly **3–9 seconds in decode** and perhaps **200ms in prefill**. The GPU runs decode for 90%+ of wall-clock time — and during that 90%, compute utilisation sits at 20–40%. The tensor cores are idle, waiting for memory bandwidth.

You provisioned H100-class compute to handle prefill peaks, and then spent the vast majority of every request's lifetime doing work that doesn't need those tensor cores.

### The Scheduling Workaround and Its Limits

**Chunked prefill** (also called iterative prefill, popularised by SARATHI) partially addresses the interference problem. Instead of processing an entire long prompt in one massive batch slot, the engine breaks it into smaller chunks and interleaves them with decode steps. This caps the ITL spike any single prefill can cause.

But chunked prefill does not solve the utilisation problem. The GPU is still performing compute-heavy work (prefill chunks) and memory-heavy work (decode steps) in alternation — on the same hardware. The arithmetic intensity still oscillates between 200–400 FLOP/byte and 1 FLOP/byte within the same scheduling window. The fundamental mismatch remains.

---

## 5. Goodput: The Metric That Actually Matters

Raw throughput — requests served per second — hides the real cost of colocation. A system can serve many requests per second while violating the latency SLOs that users actually care about.

**Goodput**, introduced formally in the DistServe paper and explained in Hao Zhang's CMU lecture:

> "Goodput = number of requests that meet **both** the TTFT SLO **and** the TPOT SLO per unit time."

A request counts toward goodput only if it satisfies both:
- TTFT ≤ TTFT_SLO (the response started fast enough)
- TPOT ≤ TPOT_SLO (each token streamed fast enough)

If either condition is violated, the request doesn't count — even if it completed successfully.

**The SLO space:**

```
TPOT SLO axis (ms/token)
    │   ┌──────────┐
    │   │  GOOD    │  ← requests that count toward goodput
    │   │REQUESTS  │
    ├───┤          │
    │   └──────────┘
    └──────────────────── TTFT SLO axis (ms)
                     ↑
              only requests in this box count
```

In a monolithic system under load, the TTFT and TPOT SLOs pull against each other. Accepting more requests improves raw throughput but pushes requests outside the SLO box. Under strict SLO constraints, goodput peaks well below raw throughput capacity — and a disaggregated system, by eliminating the interference, fits many more requests inside the box.

**DistServe measured goodput improvements of 4–7× over vLLM under strict SLO constraints** (TTFT < 2s, TPOT < 100ms) across OPT-13B, OPT-66B, and LLaMA-2-70B. These are not raw throughput numbers; they are counts of SLO-satisfying requests. Without SLOs, the improvement is smaller — goodput under constraints is where disaggregation truly earns its complexity.

---

## 6. How Disaggregation Works

The idea is direct: if prefill and decode have opposite hardware requirements and interfere when colocated, **run them on separate hardware pools**.

> "Disaggregated inference splits these two phases onto separate hardware pools, each sized for what it actually does." — TDS, April 2026

### Three Components

**Component 1: The Router**

A KV-aware router sits in front of both pools as the single entry point for clients. Its job is to:
1. Receive incoming requests from clients
2. Route each request to an available prefill worker
3. After prefill completes, route the resulting KV cache to an available decode worker
4. Track which decode workers hold which KV caches (enabling prefix cache reuse across requests sharing system prompts)

The router is what allows clients to send requests without knowing which pool handles which phase.

**Component 2: The Prefill Pool**

A set of GPUs optimised for high-throughput matrix multiplication. Prefill workers:
- Process incoming prompts
- Build the KV cache for each request
- Write the KV cache to the network for transfer to a decode worker
- Never generate output tokens

Because prefill is compute-bound, these GPUs benefit from high FLOPS. They don't need massive HBM capacity — they produce KV caches temporarily and immediately transfer them. Multiple prompts can be batched together, amortising the matrix computation across many requests simultaneously.

**Component 3: The Decode Pool**

A set of GPUs optimised for memory bandwidth and HBM capacity. Decode workers:
- Receive KV caches from prefill workers
- Generate output tokens autoregressively using those caches
- Maintain KV caches for the duration of each generation

Because decode is memory-bound, these GPUs benefit from high HBM bandwidth and large HBM capacity (to hold KV caches for many concurrent users). Large batch sizes amortise HBM reads across many concurrent decode requests, pushing utilisation higher.

**The full flow for one request:**

```
1. Client request arrives at the Router
2. Router → Prefill Worker
3. Prefill Worker: processes prompt, builds KV cache
4. Prefill Worker → [KV Cache transfer over RDMA/NVLink] → Decode Worker
5. Decode Worker: generates tokens autoregressively
6. Decode Worker → Output stream → Client
```

### Independent Scaling: The Economic Argument

The primary economic benefit of disaggregation is **pool-level independent scaling**:

```
Workload A: Short prompts, long outputs
  → 1–2 Prefill Workers + 4–5 Decode Workers
  (little prefill work; lots of decode work)

Workload B: Long prompts, short outputs
  → 4–5 Prefill Workers + 1–2 Decode Workers
  (expensive prefill; quick decode)
```

In monolithic serving, scaling means adding identical GPU nodes — each capable of both prefill and decode, each provisioned for the worst case of both. With disaggregation, you add only what you need: more prefill capacity or more decode capacity, separately. You stop paying for H100-level compute during decode, and for H200-level memory bandwidth during prefill.

| Optimisation | Prefill Worker | Decode Worker |
|---|---|---|
| Optimal batch size | Large (many prompts at once) | Small but many concurrent |
| GPU type | High FLOPS (H100 SXM) | High HBM bandwidth (H200) |
| Scheduling | Batch similar-length prompts | Continuous batching |
| KV cache lifetime | Temporary — transferred immediately | Persistent — held for full generation |
| Optimal parallelism | Tensor parallelism (reduces latency) | Pipeline parallelism (increases throughput) |

---

## 7. The KV Cache Transfer Tax

Disaggregation is not free. The KV cache produced during prefill must move from the prefill GPU to the decode GPU over the network. This is the **KV cache transfer tax**, and it is the primary cost of disaggregation.

### How Big Is It?

The size depends on three model parameters and the prompt length:

```
KV cache per token = n_layers × n_kv_heads × head_dim × 2 (K and V) × bytes_per_element

For Llama-3.1-70B (GQA):
  80 layers × 8 KV heads × 128 dims × 2 × 2 bytes (FP16) = 327,680 bytes per token

For a 4K-token prompt:
  4,096 × 327,680 = 1.34 GB
```

**Python formula (works for any model):**

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes=2):
    """KV cache size in bytes for a single request."""
    per_token = n_layers * n_kv_heads * head_dim * 2 * dtype_bytes  # 2 for K and V
    return per_token * seq_len

# Llama-3.1-70B, 4K-token prompt
kv_70b = kv_cache_bytes(n_layers=80, n_kv_heads=8, head_dim=128, seq_len=4096)
# → 1.34 GB

# Llama-3.1-8B, 4K-token prompt
kv_8b = kv_cache_bytes(n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096)
# → 0.54 GB
```

Run this formula for your model before evaluating whether disaggregation is feasible on your network.

### How Long Does It Take?

The transfer time determines how much the KV transfer adds to TTFT. The decode worker cannot start generating tokens until the KV cache arrives.

For a TTFT budget of 500ms with 200ms prefill time, you have ~300ms for the transfer:
```
Required bandwidth = 1.34 GB / 0.3s ≈ 4.5 GB/s minimum
```

| Network Type | Bandwidth | Transfer Time (1.34 GB) | Verdict |
|---|---|---|---|
| 1 GbE | 125 MB/s | **10.7 seconds** | Completely unusable |
| 10 GbE | 1.25 GB/s | **1.07 seconds** | Too slow |
| 25 GbE | 3.125 GB/s | **0.43 seconds** | Borderline |
| 100 GbE | 12.5 GB/s | **0.11 seconds** | Acceptable |
| InfiniBand HDR | 25 GB/s | **54 ms** | Good |
| NVLink (within node) | 600 GB/s | **2.2 ms** | Ideal |

**The takeaway:** standard internet connections are unusable. You need co-located workers on the same rack with InfiniBand or NVLink.

### Is the Transfer Worth It?

The KV transfer overhead (27–107ms at RDMA speeds) is paid on every request. The benefit it replaces — queuing delay from prefill-decode interference — is not a constant; it grows with load.

DistServe P99 latency measurements show 2×+ improvement with disaggregation. This is because the interference problem creates **tail latency spikes**: when a large prefill enters the batch, active decode requests can stall for 200–500ms. The 27–107ms transfer cost replaces these 200–500ms stall events.

**Perplexity's further optimisation — layer-pipelined transfer:** Rather than waiting for the entire KV cache to arrive before decode begins, Perplexity's implementation transfers the cache layer by layer. The decode worker starts processing layer 0's attention while layers 1–79 are still in transit. This reduces the effective TTFT addition well below the raw bandwidth calculation.

---

## 8. The Handshake: How SGLang Moves the KV Cache

The most important implementation detail in PD disaggregation is the **handshake protocol**: the sequence of steps through which a prefill server and a decode server coordinate to transfer the KV cache safely and efficiently.

SGLang's design (as implemented for the 96 H100 DeepSeek deployment):

```
Step 1: Decode Server pre-allocates KV cache pages for the incoming request
         → sends KV page indices to the Prefill Server

Step 2: Prefill Server receives the KV page indices
         → runs the model forward pass (prefill)
         → writes KV cache directly into the Decode Server's
           pre-allocated pages (via RDMA / Mooncake or NIXL)
         → sends completion notification

Step 3: Decode Server receives the KV cache
         → begins autoregressive generation
         → streams output tokens back to client
```

**The critical design choice:** the Decode Server pre-allocates **before** the Prefill Server begins. This prevents a race condition where the Decode Server might not have space when the Prefill Server finishes. The allocation happens first, then prefill runs, then the transfer fills the pre-allocated pages.

**Two transfer backend options in SGLang:**

- **Mooncake** (recommended for production): RDMA-based, optimised for InfiniBand and NVLink. Uses GPUDirect RDMA to transfer KV cache data directly between GPU HBM on different nodes without CPU involvement.
- **NIXL** (network-agnostic): NVIDIA Inference Xfer Library, supports UCX and LIBFABRIC backends. Works with any high-speed interconnect that UCX supports.

---

## 9. Launching a Disaggregated Cluster

SGLang uses a single `launch_server` command with a `--disaggregation-mode` flag to convert a standard inference server into a phase-specific worker.

### Minimal Setup: Single Node, Mooncake Backend

```bash
# 1. Install the transfer engine
uv pip install mooncake-transfer-engine

# 2. Start the prefill worker (uses GPU 0)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# 3. Start the decode worker (uses GPU 1 on the same node)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 \
  --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# 4. Start the router (the client-facing entry point)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

Clients send requests to port 8000 (the router). The router dispatches to the prefill worker and the decode worker; clients never address either worker directly.

### Minimal Setup: Single Node, NIXL Backend

```bash
# 1. Install NIXL
pip install nixl

# 2. Start prefill worker with NIXL
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-transfer-backend nixl

# 3. Start decode worker with NIXL
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 \
  --disaggregation-transfer-backend nixl

# 4. Start the router (same as above)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### Key CLI Flags Reference

| Flag | Worker | What it does |
|---|---|---|
| `--disaggregation-mode prefill` | Prefill | This server handles prefill only; never generates tokens |
| `--disaggregation-mode decode` | Decode | This server handles decode only; requires KV cache from prefill |
| `--disaggregation-transfer-backend mooncake` | Both | Use Mooncake RDMA engine for KV transfer |
| `--disaggregation-transfer-backend nixl` | Both | Use NIXL (UCX/LIBFABRIC) for KV transfer |
| `--disaggregation-ib-device mlx5_roce0` | Both | Which RDMA NIC to use for KV transfer |
| `--pd-disaggregation` | Router | Enable PD-aware routing in the router |
| `--max-running-requests 128` | Decode | Cap concurrent decode requests to avoid OOM |

### Environment Variables for Tuning

**Prefill server:**

| Variable | Purpose | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Parallel transfer queues (allows simultaneous KV transfers to multiple decode instances) | `4` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | Seconds to wait for decode server to send KV page indices | `300` |
| `SGLANG_DISAGGREGATION_THREAD_POOL_SIZE` | Worker threads for KV transfer per TP rank | Auto (based on CPU count) |

**Decode server:**

| Variable | Purpose | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | How often to health-check prefill servers (sec) | `5.0` |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | Consecutive failures before marking prefill offline | `2` |
| `SGLANG_DISAGGREGATION_WAITING_TIMEOUT` | Seconds to wait for KV cache arrival after initialization | `300` |

**For NVL72 deployments (NVLink transport):**

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

**For heterogeneous TP (prefill TP ≠ decode TP):**

```bash
# Enable GPU staging buffer for better throughput when TP sizes differ
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64   # per-worker, prefill side
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096    # ring buffer, decode side
```

> **Note:** The GPU staging buffer (2–5× throughput improvement at high concurrency) is only for non-MLA models (GQA/MHA). Do **not** enable for DeepSeek-V2/V3, which use MLA architecture.

### What the Router Does

The `sglang_router` with `--pd-disaggregation` is the client-facing entry point. It:
- Routes each incoming request to a prefill worker
- Waits for prefill completion
- Dispatches the KV cache handoff to an available decode worker
- Streams the decode output back to the client

Clients always target the router. They never interact with prefill or decode workers directly.

---

## 10. Production Evidence at Scale

Disaggregated serving is not a research prototype or an engineering curiosity. As of early 2026, it is running in production at multiple companies handling hundreds of millions of queries per month.

### Perplexity AI: 435 Million Search Queries per Month

Perplexity AI — an AI-powered search engine — handles 435+ million queries per month using NVIDIA H100 SXM GPUs and Kubernetes, serving 20+ AI models simultaneously (Llama-3.1 8B, 70B, 405B, embedding models, and classifier models).

The Perplexity inference team's stated production outcome:

> "In terms of inference serving middleware, the team is actively collaborating with the NVIDIA Triton engineering team to deploy disaggregating serving, a groundbreaking technique that separates the prefill and decode inference phases of an LLM workflow onto separate NVIDIA GPUs. This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token. Additionally, this technique gives Perplexity the flexibility to use **different NVIDIA GPU products for each inference phase** given its specific hardware resource requirements." — NVIDIA Spotlight, December 2024

**What this confirms:**
1. Disaggregated serving is in active production deployment — not evaluation — at a 435M-query/month service.
2. It "significantly boosts overall system throughput while meeting SLAs."
3. Hardware heterogeneity (different GPU SKUs for prefill and decode) is a real production benefit already being leveraged.

Cost impact: by hosting the Related-Questions feature (a single product feature) on internal GPUs using optimised serving, Perplexity estimates **approximately $1 million annually saved** compared to third-party API costs.

### SGLang + DeepSeek-V3 on 96 H100 GPUs

The SGLang team (LMSYS) published the most detailed large-scale PD disaggregation deployment in May 2025: DeepSeek-V3 (671B parameters, MoE architecture) deployed across 12 nodes × 8 H100 GPUs.

**Deployment configuration:**
- **Total cluster**: 96 H100 GPUs across 12 nodes
- **Prefill pool**: 3 nodes (24 GPUs) — 1:3 prefill-to-decode ratio
- **Decode pool**: 9 nodes (72 GPUs)
- **Model**: DeepSeek-V3 — 671B parameters, 256 experts, MLA attention
- **KV transfer**: Mooncake + InfiniBand

**Results:**

| Metric | Value |
|---|---|
| Input throughput per node | **52,300 tokens/second** |
| Output throughput per node | **22,300 tokens/second** |
| Input sequence length | 2,000 tokens |

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team, May 2025

**Why PD disaggregation was mandatory for this workload — not optional:** DeepSeek-V3 uses Mixture-of-Experts (MoE) with 256 experts, only 8 activated per token. The expert parallelism (EP) communication library (DeepEP) uses different dispatch patterns for prefill ("normal" mode: high-throughput for large batches) and decode ("low-latency" mode: minimises dispatch latency for single-token batches). These patterns are incompatible on the same GPU workers. PD disaggregation is the only way to run each phase with its optimal dispatch pattern. For MoE models at this scale, disaggregation is not an optimisation — it is a correctness requirement.

**The 3:9 prefill-to-decode GPU ratio** is the production tuning point for DeepSeek-V3 at this scale. Decode is more resource-hungry than prefill because it runs continuously (every token, one by one) while prefill is bursty (large batch, then done).

### Additional Production Adopters

| Company | Stack | Scale |
|---|---|---|
| Perplexity AI | NVIDIA Dynamo Triton + TensorRT-LLM | 435M queries/month |
| Meta | vLLM with disaggregated serving | Production |
| LinkedIn | vLLM with disaggregated serving | Production |
| Mistral | vLLM with disaggregated serving | Production |
| HuggingFace | vLLM with disaggregated serving | Production |
| Moonshot AI (Kimi) | SGLang + Mooncake | Production (Kimi serving) |

NVIDIA built its entire Dynamo inference framework (announced GTC 2025) around disaggregated serving as the default architecture for enterprise deployments.

---

## 11. When Disaggregation Makes Things Worse

The 96 H100 result and the Perplexity case study are real, but they come with conditions. Disaggregation is not universally beneficial — applied to the wrong workload, it can **degrade performance by 20–30%**.

> "If your workload is too small, or your GPU setup isn't tuned for this approach, performance can drop by 20–30% in our tests." — BentoML

### Three Failure Modes

**Failure Mode 1: Short prompts + short outputs — transfer overhead exceeds savings**

If your median prompt is 256 tokens and output is 50 tokens:
- KV cache for 256 tokens (Llama-70B): ~83 MB
- Transfer time at 100 GbE: ~7ms
- Prefill time for 256 tokens: ~20ms
- Decode time for 50 tokens: ~500ms

The interference problem barely exists for a 20ms prefill. The routing overhead, network transfer, and coordination between servers adds more latency than the interference ever would. Local prefill on the decode worker takes fewer total milliseconds.

**Failure Mode 2: High prefix cache hit rate — local is faster**

For multi-turn conversations where the decode worker already holds the KV cache from the previous turn, incremental prefill is tiny (only the new tokens, not the full context). Sending the few new tokens' KV cache from a prefill worker over the network wastes bandwidth on data that's essentially already local.

For workloads with >80% prefix cache hit rates, the decode worker should run prefill locally for those requests — the network round-trip always adds latency that the cache hit savings can't justify.

**Failure Mode 3: Too few GPUs — scheduling overhead dominates**

With fewer than ~16 GPUs total, maintaining two separate pools creates operational complexity without sufficient throughput to amortise it. The scheduling overhead of the two-pool architecture — health checks, handshake coordination, queue management — exceeds the utilisation gains from separating phases.

### The Anti-Patterns

| Scenario | What to use instead |
|---|---|
| Development and testing | Standard vLLM or SGLang in monolithic mode |
| Small models (< 13B) | KV cache is small; transfer overhead may exceed interference savings |
| Batch-only workloads | TTFT doesn't matter; throughput is the only metric; no disaggregation needed |
| Short prompts + short outputs | Local prefill on decode worker is faster |
| < 16 total GPUs | Too small for the overhead to pay off |
| No RDMA (TCP only) | Transfer latency neutralises the benefit for typical prompt sizes |

---

## 12. The Decision Framework and Cost Arithmetic

### The 5-Check Framework

Before refactoring your serving stack, run through these checks:

**Check 1 — Measure your prefill-to-decode time ratio**
Run your current deployment under production load and record what fraction of wall-clock GPU time is spent in each phase. If decode accounts for less than 70% of request duration, disaggregation has a smaller utilisation payoff. If decode exceeds 85% of wall time, you are paying for idle tensor cores most of the day.

**Check 2 — Calculate your KV cache transfer size**
Use the formula above. If it exceeds 500 MB per request and your network is under 100 Gbps, the transfer latency will consume your TTFT budget. Run this for your actual model and median prompt length, not a theoretical worst case.

**Check 3 — Check your prefix cache hit rate**
If your prefix cache hit rate is above 80%, a large fraction of requests can skip the prefill pool entirely and run prefill locally on the decode worker. The value of a separate prefill pool decreases with hit rate.

**Check 4 — Count your GPUs**
Below ~16 GPUs, scheduling overhead typically exceeds utilisation gain. Above 32 GPUs with sustained traffic, the cost savings from right-sized hardware start to compound.

**Check 5 — Audit your network**
Do your nodes have RDMA-capable NICs (EFA on AWS, ConnectX on bare metal)? At what bandwidth? If you are limited to TCP, disaggregation can still work for long-context workloads, but effective transfer bandwidth will be lower than the InfiniBand numbers.

**Decision:** If checks 1, 4, and 5 all come back favorable → disaggregation will almost certainly reduce your per-token serving cost.

### Cost Arithmetic

**Choosing GPU types per pool:**

| | Prefill Pool | Decode Pool |
|---|---|---|
| **Bottleneck** | Compute (tensor cores) | Memory bandwidth (HBM) |
| **Best GPU** | H100 SXM (high FLOPS) | H200 (larger HBM3e, higher BW) |
| **Why H200 is overkill for prefill** | Paying for HBM capacity that prefill doesn't use | — |
| **Why H200 is better for decode** | — | Larger HBM fits more concurrent KV caches; higher BW reduces decode latency |

A prefill chip with 40% less HBM bandwidth loses only 17% of prefill performance (SPAD paper, UT Austin) — because prefill doesn't use that bandwidth. A decode chip with 50% less compute loses only 22% of decode performance — because decode doesn't use that compute. The silicon savings from removing unused capability are substantial.

**Cluster-level savings:**

- **InfoQ analysis**: 15–40% total infrastructure cost reduction from disaggregation at cluster scale.
- This comes from: not overprovisioning hardware, eliminating idle tensor cores during decode, and being able to add decode workers without buying unused prefill capacity.

**The adoption curve:**

Eighteen months separated the DistServe paper (OSDI 2024) from production deployment at Perplexity, Meta, LinkedIn, Mistral, and NVIDIA's own Dynamo framework. That is fast, even by ML infrastructure standards. Systems teams that have already sized for disaggregation and designed their network infrastructure for InfiniBand are paying less per token than teams that haven't.

---

## Key Quotes

> "To make sure inference is not compute- and bandwidth-bound at the same time, we want to separate them." — Jackson MZ

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." — BentoML

> "A Llama 70B model running inference on an H100 GPU hits 92% compute utilization during prefill. Thirty milliseconds later, during decode, that same GPU drops to 30%. The hardware hasn't changed. The model weights are identical." — TDS, April 2026

> "Goodput = number of requests that meet both the TTFT SLO and the TPOT SLO per unit time." — Hao Zhang, CMU LLM Systems

> "This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token. Additionally, this technique gives Perplexity the flexibility to use different NVIDIA GPU products for each inference phase." — NVIDIA Spotlight on Perplexity, December 2024

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team, May 2025

> "If your workload is too small, or your GPU setup isn't tuned for this approach, performance can drop by 20–30% in our tests." — BentoML

---

## What Is Left Out and Why

### Left out: DistServe formal goodput derivation

The DistServe paper (L3/01) formalises goodput with a mathematical model of SLO satisfaction under load, derives optimal placement policies, and introduces the interference quantification framework. This is L3 material — it requires engaging with the queuing model and the formal argmax over goodput. The goodput concept is explained here at L1 level; the formal derivation belongs at L3.

### Left out: Splitwise hardware simulation

Splitwise (Microsoft Research, ISCA 2024) characterises prefill vs decode hardware requirements from production traces and proposes purpose-built prefill/decode silicon. The simulation results and hardware design proposals are L3 material; the hardware heterogeneity insight (prefill benefits from FLOPS, decode from HBM bandwidth) is included here.

### Left out: Mooncake RDMA internals

The Mooncake Transfer Engine paper describes multi-NIC pooling, topology-aware path selection, and GPUDirect RDMA memory registration. These implementation details of the transfer layer are L3 material. The transfer latency numbers (what bandwidth Mooncake achieves) are included; the mechanism producing them is not.

### Left out: SARATHI chunked-prefill pipeline mechanics

SARATHI's analysis of pipeline bubbles, the chunking policy, and the formal comparison with disaggregation are L4 material. Chunked prefill is introduced here as the "scheduling workaround with limits"; its internal mechanics are not.

### Left out: TaiChi SLO-regime switching

TaiChi (arXiv, August 2025) proposes a hybrid system that dynamically switches between aggregation and disaggregation based on which SLO regime the current workload falls in, achieving 77% goodput improvement in some configurations. The SLO-regime analysis and switching policy are L4 material.

### Left out: vLLM connector architecture and expert parallelism dispatch patterns

The vLLM `BaseKVConnector`, `KVLookupBufferBase`, and `--kv-transfer-config` connector types (PyNcclConnector, MooncakeConnector, etc.) are L4 implementation material. Similarly, DeepEP normal vs low-latency dispatch modes, EPLB configuration, and multi-token prediction simulation are L3+ material.
