# Prefill-Decode Disaggregation — Systems Design, Transfer Engines, and the SLO Trade-off

**Level:** L3 + L4 — Superset of `COMBINEDL1L2.md`. Adds formal goodput treatment, paper results, the SARATHI baseline, Mooncake/NIXL internals, Dynamo's 4-plane architecture, TaiChi's SLO-regime framework, and the vLLM connector abstraction. All L1+L2 concepts are retained and expanded.

**What this file is:** A single coherent blog synthesising all L3 and L4 source material into a progressive narrative, building on the L1+L2 foundation. Every section from `COMBINEDL1L2.md` is present; most are extended with quantified measurements, formal definitions, or implementation detail drawn from the papers.

**Sources synthesised:**
- L1+L2 (all — retained and expanded from `COMBINEDL1L2.md`)
- L3/01 — DistServe (USENIX OSDI 2024): goodput framing, interference quantification, 7.4× result
- L3/02 — Splitwise (ISCA 2024, Best Paper): production traces, hardware heterogeneity, 1.4×/2.35× cost results
- L3/03 — Mooncake (USENIX FAST 2025): Transfer Engine, GPUDirect RDMA, multi-NIC pooling, topology-aware selection
- L3/04 — NVIDIA Dynamo + NIXL: 4-plane architecture, KV-aware routing, NIXL Transfer Agent, async API
- L4/01 — SARATHI (USENIX OSDI 2024): chunked prefill mechanics, pipeline bubbles, aggregation baseline limits
- L4/02 — TaiChi (arXiv August 2025): SLO-regime framework, hybrid switching, 77% goodput improvement
- L4/03 — vLLM connector architecture: 6 connectors, `BaseKVConnector`, `KVLookupBufferBase`, `drop_select`

**Omitted (moved to `OMITTEDL3L4.md`):** SplitwiseSim detailed CLI, NIXLBench/KVBench commands, TaiChi PPD three-way disaggregation, Mooncake prediction-based early rejection algorithm, vLLM directory structure listing, DeepEP/EPLB MoE dispatch internals, `KVTransferConfig` full JSON schema.

---

## Section Plan

| § | Title | Primary sources | Reading time |
|---|-------|----------------|------|
| 1 | [The Two Phases and Their Structural Asymmetry](#1-the-two-phases-and-their-structural-asymmetry) | L1/06, L3/02, L1/07 | 4 min |
| 2 | [The Metrics: TTFT, ITL, and Goodput](#2-the-metrics-ttft-itl-and-goodput) | L1/09, L3/01 | 4 min |
| 3 | [The Roofline Model: Why Each Phase Needs Different Hardware](#3-the-roofline-model-why-each-phase-needs-different-hardware) | L1/01, L1/07, L3/02 | 4 min |
| 4 | [Interference in Monolithic Serving: Quantification](#4-interference-in-monolithic-serving-quantification) | L3/01, L4/01, L1/05 | 4 min |
| 5 | [SARATHI: The Aggregation Baseline and Its Limits](#5-sarathi-the-aggregation-baseline-and-its-limits) | L4/01 | 5 min |
| 6 | [DistServe: Disaggregation for Goodput](#6-distserve-disaggregation-for-goodput) | L3/01 | 4 min |
| 7 | [Splitwise: Production Traces and Hardware Heterogeneity](#7-splitwise-production-traces-and-hardware-heterogeneity) | L3/02 | 4 min |
| 8 | [How Disaggregation Works: Architecture and Independent Scaling](#8-how-disaggregation-works-architecture-and-independent-scaling) | L1/02, L1/03, L1/09 | 3 min |
| 9 | [The KV Cache Transfer Tax: Size, Cost, and Network Requirements](#9-the-kv-cache-transfer-tax-size-cost-and-network-requirements) | L1/01, L1/02, L3/02 | 4 min |
| 10 | [Mooncake: Transfer Engine Internals](#10-mooncake-transfer-engine-internals) | L3/03 | 5 min |
| 11 | [NVIDIA Dynamo: 4-Plane Architecture for Enterprise Scale](#11-nvidia-dynamo-4-plane-architecture-for-enterprise-scale) | L3/04 | 5 min |
| 12 | [NIXL: The Vendor-Agnostic KV Transfer Library](#12-nixl-the-vendor-agnostic-kv-transfer-library) | L3/04 | 4 min |
| 13 | [The Aggregation-vs-Disaggregation Decision: TaiChi's SLO-Regime Framework](#13-the-aggregation-vs-disaggregation-decision-taichis-slo-regime-framework) | L4/02 | 5 min |
| 14 | [The KV Transfer Abstraction: vLLM's Connector Architecture](#14-the-kv-transfer-abstraction-vllms-connector-architecture) | L4/03 | 5 min |
| 15 | [Launching a Disaggregated Cluster with SGLang](#15-launching-a-disaggregated-cluster-with-sglang) | L2/01 | 4 min |
| 16 | [Production Evidence at Scale](#16-production-evidence-at-scale) | L1/08, L2/02, L1/01, L3/03 | 4 min |
| 17 | [Hardware Heterogeneity and the Cost Arithmetic](#17-hardware-heterogeneity-and-the-cost-arithmetic) | L3/02, L1/01 | 3 min |
| 18 | [The Decision Framework: 5 Checks Before Refactoring](#18-the-decision-framework-5-checks-before-refactoring) | L1/01, L4/02 | 3 min |

**Total reading time:** ~74 minutes

---

## 1. The Two Phases and Their Structural Asymmetry

Every LLM request passes through exactly two phases before any output reaches the user.

### Prefill (Prompt Processing)

When a prompt arrives, the model processes the **entire input sequence in parallel** — all tokens simultaneously. For each token, in each attention layer, the model computes Query (Q), Key (K), and Value (V) vectors. The K and V vectors for every input token are stored in GPU memory as the **KV cache** — the model's working memory for this request. Prefill ends with a complete KV cache and no output tokens yet produced.

### Decode (Token Generation)

After prefill, the model enters **autoregressive decode mode**: generating one output token per step, sequentially. Each step reads the entire KV cache, predicts the next token, appends its K and V to the cache, and repeats. Because each token depends on all previous ones, decode **cannot be parallelised across output tokens** — it is structurally sequential.

**Why the KV cache exists:** Without it, every decode step would recompute attention over the growing sequence from scratch — O(n²) cost. With the cache, each decode step is O(n) total across the full generation.

### The Fundamental Asymmetry (Quantified from Production)

Splitwise characterised both phases from **real Azure production traces** across two LLM serving services:

| Characteristic | Prefill | Decode |
|---|---|---|
| Compute profile | Compute-intensive (saturates FLOP throughput) | Memory-bandwidth-bound (loads KV per step) |
| GPU utilisation | High — matrix multiplications across full batch | Low — autoregressive, one token per step |
| Memory pressure | Low (transient activations only) | High (KV cache grows with sequence length) |
| Power draw | High (compute-bound) | Lower (memory-bandwidth-bound) |
| Duration | Short (milliseconds to seconds) | Long (seconds to minutes) |

**The Splitwise key finding**: token generation phases **do not require the compute capability of the latest GPUs**. Even with continuous batching across many requests, decode-phase arithmetic intensity is too low to saturate high-FLOP GPUs.

---

## 2. The Metrics: TTFT, ITL, and Goodput

Three metrics define LLM serving quality. Two are user-facing; one is the correct system-level target.

### Time to First Token (TTFT)

TTFT is the elapsed time from request submission to the first output token. TTFT ≈ prefill time + first decode step. Long prompts → higher TTFT. This is what users perceive as the model "thinking" before responding.

### Inter-Token Latency (ITL) / TPOT

ITL (also called TPOT — Time Per Output Token) is the average time between consecutive output tokens once generation begins. ITL is dominated by the decode phase. Typical values on high-end hardware: 20–100ms per token (10–50 tokens/second).

| Metric | Dominated by | User impact |
|---|---|---|
| TTFT | Prefill | "Hang" before first word |
| ITL / TPOT | Decode | Smooth vs stuttering stream |

### Goodput: The Correct Optimisation Target

Raw throughput — requests per second — hides the real cost of colocation. A system can serve many requests per second while violating the latency SLOs that users actually experience.

**Goodput**, formalised in DistServe:

> "Goodput = number of requests that meet **both** the TTFT SLO **and** the TPOT SLO per unit time."

A request counts toward goodput only if:
- TTFT ≤ TTFT_SLO (response started fast enough), **AND**
- TPOT ≤ TPOT_SLO (each token streamed fast enough)

Both conditions must hold simultaneously.

**The SLO space:**

```
TPOT SLO axis (ms/token)
    │   ┌──────────┐
    │   │  GOOD    │  ← goodput counts only these
    │   │REQUESTS  │
    ├───┤          │
    │   └──────────┘
    └──────────────────── TTFT SLO axis (ms)
```

In a collocated system under load, TTFT and TPOT SLOs pull against each other. Accepting more requests improves raw throughput but pushes requests outside the SLO box. Disaggregation eliminates the interference that creates this tension, fitting many more requests inside the box.

**Why goodput matters for DistServe's results**: the 7.4× improvement is measured in goodput under strict SLO constraints — not raw throughput. Without SLOs, the improvement is smaller. Measuring with SLOs tells the story users actually experience.

---

## 3. The Roofline Model: Why Each Phase Needs Different Hardware

The roofline model maps workloads onto the performance space defined by two hardware ceilings: peak FLOPS and peak memory bandwidth.

**Arithmetic intensity** (AI) = FLOPs per byte of memory accessed. Operations to the left of the "ridge point" are memory-bandwidth-bound; operations to the right are compute-bound.

### Prefill on the Roofline

During prefill, the dominant operations are large matrix multiplications (Q @ Kᵀ and Attention @ V) across the full prompt length S. For a 4,096-token prompt:

```
Prefill arithmetic intensity ≈ SeqLen FLOP/byte

For S = 4,096:
  AI ≈ 200–400 FLOP/byte (measured on H100 SXM)
  GPU utilisation: 90–95%
```

Prefill sits **far to the right** of the ridge point on the roofline — firmly compute-bound. The HBM bus is barely taxed.

**Scaling law:** longer prompts → higher arithmetic intensity → more compute-bound. Very long prompts benefit most from a dedicated prefill pool.

### Decode on the Roofline

During decode, each step generates one token. The matrix operations are tiny (shape [1 × d] against cached KV of length n). But the model must load the **entire KV cache from HBM** on every step:

```
Decode arithmetic intensity ≈ 1 FLOP/byte (constant, independent of sequence length)
  GPU utilisation: 20–40% (measured on H100 SXM)
```

Decode sits **near the origin** of the roofline — always memory-bandwidth-bound, regardless of how many tensor cores the GPU has. Adding FLOPS doesn't help decode. Only higher HBM bandwidth helps.

### The SPAD Validation (UT Austin)

The SPAD paper validated these sensitivities experimentally:
- **Reducing memory bandwidth by 40%** → prefill latency increased only **17%** (prefill doesn't use bandwidth)
- **Reducing compute capacity by 50%** → decode latency increased only **22%** (decode doesn't use compute)

This confirms that the two phases are in fundamentally different hardware regimes — not adjacent on the same workload spectrum. Trying to optimise a single GPU for both simultaneously always produces a compromise.

### The Interactive Roofline Tool

The LLM Inference Performance Estimator (https://joursbleu.github.io/llm-perf-model/) makes this visual. Select any model + GPU combination and the roofline plot shows exactly where prefill and decode land. The 5× arithmetic intensity drop is visible with any long prompt.

---

## 4. Interference in Monolithic Serving: Quantification

When prefill and decode share the same GPU pool, interference is **structural** — it cannot be eliminated by scheduling, only reduced.

### The Interference Mechanism

When a new prefill request enters the batch:
1. The GPU's tensor cores are saturated by the prefill computation for hundreds of milliseconds.
2. All active decode steps wait for the prefill to yield.
3. Decode requests experience an **ITL spike** proportional to prefill batch size and prompt length.
4. Users watching streaming responses see text pause mid-sentence.

**The reverse**: when many decode requests fill the running queue, new prefill requests queue longer — TTFT increases even when the server is not at capacity.

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." — BentoML

### Quantified: DistServe's Interference Measurements

DistServe measured the interference effect directly:

- **TPOT variance**: collocated systems show **3–5× higher TPOT variance** vs disaggregated systems under production SLO constraints, even with chunked prefill (SARATHI).
- **ITL spike correlation**: decode ITL spikes with prefill batch size — a clear positive correlation observed in Hao Zhang's CMU lecture slide measurements.
- **TTFT impact**: as the running decode queue fills, new prefill requests queue longer → TTFT increases even below full server capacity.

### The Resource Coupling Problem

Interference is one problem. Resource coupling is a second, separate structural problem that scheduling cannot solve:

1. **FLOPS vs HBM bandwidth**: a single GPU cannot simultaneously be compute-optimal (for prefill) and memory-bandwidth-optimal (for decode). The hardware that maximises prefill performance is overprovisioned for decode and vice versa.

2. **Parallelism mismatch**: the tensor parallelism (TP) degree that minimises per-request prefill latency (lower TP → fewer all-reduce calls → lower TTFT per request) is different from the TP that maximises decode throughput. With a collocated system, you must choose one TP configuration for both.

3. **Over-provisioning trap**: to meet both TTFT and TPOT SLOs simultaneously, a collocated system must over-provision GPUs — paying for hardware that runs at low utilisation to handle peak demands in both dimensions simultaneously.

---

## 5. SARATHI: The Aggregation Baseline and Its Limits

SARATHI (OSDI 2024) defines the **aggregation-side state of the art** — the best that careful scheduling can achieve without separating phases. Every disaggregation paper compares against SARATHI. Understanding it calibrates the value of full disaggregation.

### The Problem SARATHI Addresses

Standard inference: a single long prefill monopolises the GPU for hundreds of milliseconds. All decode requests are blocked — **head-of-line blocking**.

### Chunked Prefill Mechanics

Instead of processing a large prefill in one step, SARATHI splits it into **fixed-size chunks** interleaved with decode steps:

```
Standard:  [PPPPPPPPPPPP | D D D D D D D D D]
            ← 12 prefill tokens → ← decode →
            One prefill blocks all decodes for entire duration.

SARATHI:   [PPPP | D D D D] → [PPPP | D D D D] → [PPPP | D D D D]
           Chunk 1 + decodes   Chunk 2 + decodes   Chunk 3 + decodes
```

**Decode-maximal batching**: each batch = one prefill chunk (saturates GPU FLOP throughput for the chunk duration) + as many decode requests as fit in remaining KV memory. Decode requests "piggyback" — their incremental compute per step is an order of magnitude lower than prefill.

### Pipeline Bubble Reduction

In pipeline-parallel inference, mixed-duration microbatches create idle pipeline stages ("bubbles"). Chunked prefill creates uniformly-sized microbatches:
- **6.29× pipeline bubble reduction** on GPT-3 (175B) with PP=8
- **1.91× end-to-end throughput improvement** from the bubble reduction alone

### Results

| Model | Metric | SARATHI improvement |
|---|---|---|
| LLaMA-13B (A6000) | Decode throughput | **Up to 10×** |
| LLaMA-13B (A6000) | End-to-end throughput | **Up to 1.33×** |
| LLaMA-33B (A100) | Decode throughput | **4.25×** |
| LLaMA-33B (A100) | End-to-end throughput | **1.25×** |

### What SARATHI Cannot Fix

| Problem | SARATHI | PD Disaggregation |
|---|---|---|
| Head-of-line blocking | **Reduced** (chunk duration, not full prefill) | **Eliminated** (prefill on separate GPU) |
| Decode TPOT increase during prefill | **Reduced but non-zero** (decode waits for each chunk) | **Zero** (decode GPU never runs prefill) |
| Resource coupling | **Unchanged** (same GPU, same TP config) | **Decoupled** (independent TP per phase) |
| Hardware optimisation per phase | Impossible | Different GPU SKUs per pool |
| TPOT variance | **Still variable** (chunk scheduling adds jitter) | **Stable** (decode GPU is dedicated) |

**The key limit**: even a 1-token prefill chunk requires exclusive GPU access for one forward step. At very high prefill rates (long prompts, many concurrent requests), residual interference remains significant. DistServe measures **3–5× higher TPOT variance** with chunked prefill vs disaggregated systems.

**SARATHI in SGLang**: the prefill server uses `--chunked-prefill-size` to limit individual step duration, ensuring no single large prompt delays KV transfer indefinitely. Chunked prefill and disaggregation are **complementary**, not competing.

```bibtex
@inproceedings{agrawal2024sarathi,
  title     = {{SARATHI}: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills},
  author    = {Amey Agrawal and Ashish Panwar and Jayashree Mohan
               and Nipun Kwatra and Bhargav S. Gulavani and Ramachandran Ramjee},
  booktitle = {USENIX OSDI 2024},
  year      = {2024},
  url       = {https://arxiv.org/abs/2308.16369}
}
```

---

## 6. DistServe: Disaggregation for Goodput

DistServe (OSDI 2024) is the foundational paper that defined prefill-decode disaggregation as a system-design principle and introduced **goodput** as the correct optimisation target. Every subsequent system (Splitwise, Mooncake, Dynamo, SGLang PD) cites DistServe.

### Core Design Choices

**Independent resource allocation**: each phase is provisioned for its own SLO:
- TTFT SLO tight → scale prefill pool (more GPUs or more instances)
- TPOT SLO tight → scale decode pool
- Both tight → scale both independently

**Per-phase parallelism strategy**:
- Prefill pool: TP optimised for compute throughput (e.g., TP=4, high FLOP utilisation, fewer all-reduce calls)
- Decode pool: TP optimised for memory bandwidth (e.g., TP=8, maximise HBM reads per step across shards)

These are **different configurations** — something impossible in a collocated system where one TP configuration serves both phases.

**Bandwidth-aware placement**: DistServe analyses the serving cluster's bandwidth topology. When prefill and decode workers are on the same physical rack (high NVLink/InfiniBand BW), KV transfer cost is amortised. When cross-pod, DistServe reduces the TP that governs KV cache size.

### Experimental Results (OPT-13B, OPT-66B, OPT-175B)

Evaluated across chatbot, document summarisation, and coding workloads. Baseline: vLLM with continuous batching.

| Metric | DistServe vs vLLM |
|---|---|
| Max requests served at SLO (goodput) | **7.4× more** |
| Tightest SLO achievable at same request rate | **12.6× tighter** |
| Requests within SLO (>90% guarantee) | Maintained across all workloads |

**Why 7.4×**: the prefill pool handles 100% compute-bound workloads with its optimal TP config; the decode pool handles 100% memory-bandwidth-bound workloads with its optimal config. Neither is compromised by the other's requirements. The improvement is entirely from removing the coupling constraint.

### DistServe → SGLang Concept Mapping

| DistServe concept | SGLang equivalent |
|---|---|
| Prefill instance (KV producer) | `--disaggregation-mode prefill` server |
| Decode instance (KV consumer) | `--disaggregation-mode decode` server |
| KV cache transfer | Mooncake RDMA or NIXL |
| Router / orchestration layer | `sglang_router.launch_router --pd-disaggregation` |
| Per-phase resource allocation | Separate `--tp-size`, `--dp-size` per mode |
| Bandwidth-aware placement | `--disaggregation-ib-device` NIC selection |
| Goodput optimisation | `--max-running-requests` on decode server |

```bibtex
@inproceedings{zhong2024distserve,
  title     = {DistServe: Disaggregating Prefill and Decoding for
               Goodput-optimized Large Language Model Serving},
  author    = {Yinmin Zhong and Shengyu Liu and Junda Chen and Jianbo Hu
               and Yibo Zhu and Xuanzhe Liu and Xin Jin and Hao Zhang},
  booktitle = {USENIX OSDI 2024},
  year      = {2024},
  url       = {https://arxiv.org/abs/2401.09670}
}
```

---

## 7. Splitwise: Production Traces and Hardware Heterogeneity

Splitwise (ISCA 2024, Best Paper) is the independent co-discovery of PD disaggregation from Microsoft Research, published simultaneously with DistServe. Its unique contribution: grounding the analysis in **real production traces from two Azure LLM services**, and introducing the insight that **different hardware SKUs** may be optimal per phase.

### Production Trace Findings

Splitwise's Azure trace data validates two things that DistServe's synthetic benchmarks assumed:
1. The prefill-decode phase asymmetry is a **consistent production phenomenon**, not an artifact of controlled experiments.
2. Token generation consistently **underutilises GPU compute** even with continuous batching — making latest-gen high-FLOP GPUs overkill for decode.

> "Token generation phases do not require the compute capability of the latest GPUs and can be run on lower-power, lower-cost hardware with equivalent quality." — Splitwise

### Hardware Heterogeneity: The Cost Insight

Splitwise's novel design dimension: **run each phase on hardware tailored to its bottleneck**.

| Phase | Optimal hardware | Reason |
|---|---|---|
| Prefill | Latest GPUs (H100, B200) | Highest FLOP throughput per dollar |
| Decode | Older or memory-optimised GPUs (A100, H20) | High HBM bandwidth per dollar; FLOP budget not the constraint |

**Measured cost/throughput results** (H100 prefill + A100 decode vs all-H100):

| Scenario | Result |
|---|---|
| Same cost budget | **2.35× more throughput** |
| Same throughput target | **1.4× more throughput at 20% lower cost** |

This insight directly influences production deployments — some cloud providers run prefill on current-gen GPUs and decode on previous-generation hardware, reducing per-token serving cost with no quality loss.

### KV State Transfer Amortisation Threshold

Splitwise provides the analytical threshold for when disaggregation is net-positive:

**For prompts longer than ~500 tokens generating more than ~50 output tokens**, disaggregation is always net-positive in latency. Below this threshold, the KV transfer overhead (even at InfiniBand speeds) exceeds the interference-elimination gain.

This is consistent with the L1+L2 "5-check framework": check for median prompt < 512 tokens or output < 100 tokens before deploying.

### Splitwise → vLLM Lineage

Splitwise includes **GitHub PR #2809** — the first public implementation of inter-instance KV transfer in vLLM. This PR became the direct ancestor of vLLM's `vllm/distributed/kv_transfer/` module, and every vLLM disaggregation connector is an evolution of this prototype.

```bibtex
@inproceedings{patel2024splitwise,
  title     = {Splitwise: Efficient Generative LLM Inference Using Phase Splitting},
  author    = {Pratyush Patel and Esha Choukse and Chaojie Zhang and Aashaka Shah
               and {\'I}{\~n}igo Goiri and Saeed Maleki and Ricardo Bianchini},
  booktitle = {ISCA 2024},
  year      = {2024},
  url       = {https://arxiv.org/abs/2311.18677}
}
```

---

## 8. How Disaggregation Works: Architecture and Independent Scaling

Disaggregated inference runs prefill and decode on **separate GPU pools**, connected by a high-speed network. The architecture has three components.

**Component 1 — The Router**: single client-facing entry point. Routes requests to available prefill workers; after prefill, routes the KV cache to a decode worker. Tracks which decode workers hold which caches.

**Component 2 — The Prefill Pool**: GPUs optimised for high-throughput matrix multiplication. Process prompts, build KV caches, hand them off. Never generate output tokens. Can batch many prompts together simultaneously.

**Component 3 — The Decode Pool**: GPUs optimised for HBM bandwidth and capacity. Receive KV caches, run autoregressive decode. Maintain KV caches for the full duration of generation. Large batch sizes amortise HBM reads across many concurrent users.

**The full request flow:**

```
1. Client request → Router
2. Router → Prefill Worker (processes prompt, builds KV cache)
3. Prefill Worker → [KV Cache transfer over RDMA/NVLink] → Decode Worker
4. Decode Worker → autoregressive token generation → Output stream → Client
```

**Independent scaling by workload:**

```
Short prompts, long outputs:   1–2 Prefill Workers + 4–5 Decode Workers
Long prompts, short outputs:   4–5 Prefill Workers + 1–2 Decode Workers
MoE (DeepSeek-V3, 96 H100s):  3 prefill nodes (24 GPUs) + 9 decode nodes (72 GPUs)
```

Scale what you need. Don't add capacity for both phases as a unit.

| Optimisation | Prefill Worker | Decode Worker |
|---|---|---|
| Batch size | Large batches | Small but many concurrent |
| GPU type | High FLOPS (H100) | High HBM bandwidth (H200) |
| Scheduling | Batch similar-length prompts | Continuous batching |
| KV cache lifetime | Temporary — transferred immediately | Persistent — held for full generation |
| Optimal parallelism | Tensor parallelism (reduces TTFT) | Pipeline parallelism (increases throughput) |

---

## 9. The KV Cache Transfer Tax: Size, Cost, and Network Requirements

The KV cache produced during prefill must move from the prefill GPU to the decode GPU. This is the **transfer tax** — the primary cost of disaggregation.

### Sizing the Cache

```
KV cache per token = n_layers × n_kv_heads × head_dim × 2 (K and V) × bytes_per_element
```

```python
def kv_cache_bytes(n_layers, n_kv_heads, head_dim, seq_len, dtype_bytes=2):
    per_token = n_layers * n_kv_heads * head_dim * 2 * dtype_bytes
    return per_token * seq_len

# Llama-3.1-70B (GQA), 4K-token prompt
kv_70b = kv_cache_bytes(n_layers=80, n_kv_heads=8, head_dim=128, seq_len=4096)
# Per token: 80 × 8 × 128 × 2 × 2 = 327,680 bytes
# Full prompt: 1.34 GB

# Llama-3.1-8B, 4K-token prompt
kv_8b = kv_cache_bytes(n_layers=32, n_kv_heads=8, head_dim=128, seq_len=4096)
# 0.54 GB
```

For LLaMA-70B at 128K context: the cache is ~40 GB per request.

### Network Requirements and Transfer Times

For a TTFT budget of 500ms with 200ms prefill time, you have ~300ms for transfer:

```
Required bandwidth = 1.34 GB / 0.3s ≈ 4.5 GB/s minimum (4K-token prompt)
```

| Network Type | Bandwidth | Transfer Time (1.34 GB) | Verdict |
|---|---|---|---|
| 1 GbE | 125 MB/s | 10.7 seconds | Unusable |
| 10 GbE | 1.25 GB/s | 1.07 seconds | Too slow |
| 25 GbE | 3.125 GB/s | 0.43 seconds | Borderline |
| 100 GbE | 12.5 GB/s | **0.11 seconds** | Acceptable |
| InfiniBand HDR | 25 GB/s | **54 ms** | Good |
| NVLink (within node) | 600 GB/s | **2.2 ms** | Ideal |

**The transfer cost vs interference cost**: DistServe P99 latency shows 2×+ improvement with disaggregation. The 27–107ms transfer cost replaces 200–500ms prefill-induced decode stalls. The trade is favourable for all workloads above Splitwise's amortisation threshold (~500 input tokens, ~50 output tokens).

**Layer-pipelined transfer (Perplexity's optimisation)**: transfer KV cache layer by layer. The decode worker starts processing layer 0's attention while layers 1–79 are still in transit. Effective TTFT addition falls below the raw bandwidth calculation.

---

## 10. Mooncake: Transfer Engine Internals

Mooncake (FAST 2025) is both the production serving platform for Kimi (Moonshot AI's LLM service) and the open-source KV transfer engine integrated into SGLang, vLLM v1, TensorRT-LLM, and LMDeploy. Understanding its internals explains what is actually happening when SGLang uses `--disaggregation-ib-device` and what `SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK` enables.

### The Two-Level Architecture

Mooncake has two disaggregation levels:
1. **Phase disaggregation**: separate prefill and decode clusters (same as DistServe/Splitwise)
2. **Cache disaggregation**: a distributed KV cache pool from underutilised CPU DRAM and SSD across the GPU cluster — sitting between the two clusters and enabling cross-node prefix caching

SGLang uses the Transfer Engine for (1). SGLang's HiCache uses the Mooncake Store (the KV cache pool) as a storage backend for (2).

### GPUDirect RDMA: The Key to Network-Rate Transfer

Without GPUDirect, the KV transfer path crosses the CPU twice:

```
Prefill GPU VRAM → CPU RAM (PCIe) → NIC → network → NIC → CPU RAM (PCIe) → Decode GPU VRAM
```

With GPUDirect RDMA, the CPU is entirely bypassed:

```
Prefill GPU VRAM → NIC → network → NIC → Decode GPU VRAM
```

This eliminates two PCIe crossings. For 40 GB of KV data (LLaMA3-70B at 128K context), GPUDirect RDMA reduces transfer time from ~15 seconds (CPU-mediated path) to the NIC line-rate limit — a >100× improvement.

### Topology-Aware NIC Selection

Modern inference servers have multiple CPU sockets, DRAM banks, GPUs, and RDMA NICs. Not all NIC–GPU paths are equal: a NIC on the same PCIe root complex as the GPU provides full bandwidth; a NIC behind a PCIe/UPI bridge provides half or less.

Mooncake's topology-aware path selection:
1. On startup, each server generates a **topology matrix**: GPU-NIC affinity, NUMA distances, PCIe bandwidth per path.
2. The matrix is broadcast to all cluster members.
3. For each KV transfer, the Transfer Engine selects the NIC(s) with the **highest-bandwidth path** to the source/destination GPU.

Skipping this (naive NIC selection) can halve effective transfer bandwidth — a common deployment error.

### Multi-NIC Pooling

A single RDMA NIC on an H100 server provides ~200 Gbps. Servers typically have 4–8 NICs. Mooncake uses **multiple RDMA NICs simultaneously** for a single transfer:

- Large transfers (>1 GB) are internally split into slices.
- Each slice is assigned to a different NIC based on topology affinity.
- All slices submitted in parallel; completed independently.
- If one NIC becomes congested, the Transfer Engine retries on other NICs.

For 8-NIC servers, effective aggregate bandwidth: up to ~1.6 Tbps per node — sufficient for even very large KV caches.

### NVLink Transport for NVL72

The NVIDIA GB200 NVL72 rack interconnects 72 GPUs with rack-scale NVLink (MNNVL), providing ~900 GB/s aggregate bandwidth — ~10× higher than InfiniBand per port.

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

For within-rack KV transfers on NVL72, NVLink bypasses InfiniBand entirely and reduces transfer time by ~10×.

### Supported Protocols

| Protocol | Use case |
|---|---|
| RDMA InfiniBand / RoCEv2 | Inter-node GPU-to-GPU (GPUDirect, highest BW) |
| NVLink / MNNVL | Intra-rack (NVL72), bypasses InfiniBand |
| NVMe-oF | Storage → GPU (cold KV cache loading from disk) |
| TCP | Fallback; used for auxiliary metadata even when RDMA is available |
| CXL / shared memory | Emerging rack-scale shared DRAM path |

### Production Results

- **Kimi service**: 75% more requests handled vs baseline; up to **525% throughput increase** in long-context scenarios under SLO adherence.
- **Kimi K2** (128 H200 GPUs, July 2025): **224,000 tokens/second prefill**, **288,000 tokens/second decode**.

```bibtex
@inproceedings{moonshot2025mooncake,
  title     = {Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving},
  author    = {{Moonshot AI} and {MadSys Group, Tsinghua University}},
  booktitle = {USENIX FAST 2025},
  year      = {2025},
  url       = {https://arxiv.org/abs/2407.00079}
}
```

---

## 11. NVIDIA Dynamo: 4-Plane Architecture for Enterprise Scale

NVIDIA Dynamo (announced GTC 2025) is an open-source inference serving framework designed around disaggregated serving as a first-class primitive. It is backend-agnostic (vLLM, SGLang, TensorRT-LLM) and demonstrates what a **full production orchestration layer** needs beyond just a transfer engine.

Dynamo's central thesis: disaggregated serving requires not just a transfer engine but a **separate plane for each concern** — requests, control, discovery, and events.

### The 4-Plane Architecture

**Request Plane** (the data path):
```
Client → Frontend → Router → Prefill worker
                               ↓ (KV metadata)
                     Router → Decode worker
                               ↓ (token stream)
               Frontend ← Client
```

**Control Plane (Planner)** — dynamic pool sizing based on live SLO signals:
- Monitors TTFT P95 violations → scale prefill pool
- Monitors TPOT violations → scale decode pool
- Uses a feedback control loop with configurable reaction time and cooldown
- SGLang's current disaggregation has no equivalent; scaling is manual

**Discovery Plane** — etcd lease-based worker registration:
- Workers publish endpoint, role (prefill/decode), and load state at startup
- Lease TTL: 10 seconds. Worker death → lease expires → router removes it automatically
- New workers serve within one lease period — no manual router reconfiguration
- SGLang uses a static `--prefill` / `--decode` URL list; dynamic discovery requires manual restart

**Event Plane** — asynchronous state propagation:
- KV cache hit/miss signals: prefill workers publish prefix cache state; router subscribes
- SLO violation alerts: monitoring publishes; Planner subscribes
- Worker availability changes: discovery plane publishes; router subscribes

The event plane is what makes KV-aware routing scalable at large numbers of workers — the router updates its routing table via events rather than polling each worker continuously.

### KV-Aware Routing: Dynamo's Core Innovation

Unlike round-robin, Dynamo's Router tracks **KV cache state** across prefill workers:
- Each prefill worker maintains a local KV cache (similar to SGLang's RadixCache)
- The Router tracks which token prefixes are cached on which workers
- For each incoming request, the Router routes to the prefill worker with the **highest prefix overlap** — skipping recomputation of already-cached context

**Impact**: in a test with 100K requests to DeepSeek R1-Distilled Llama-70B FP8 on 2 H100 nodes, KV-aware routing measurably reduces TTFT by routing repeat prefixes to workers that already have the KV cached.

SGLang's router currently uses round-robin. KV-aware prefix routing is planned.

### Benchmark Result

| Metric | Result |
|---|---|
| Requests served (DeepSeek-R1, GB200 NVL72) | Up to **30× more requests** vs collocated |

The 30× improvement combines: disaggregation + NVLink-speed transfer + KV-aware routing + dynamic scaling.

---

## 12. NIXL: The Vendor-Agnostic KV Transfer Library

NIXL (NVIDIA Inference Xfer Library) is the open-source KV transfer library used by Dynamo and supported natively by SGLang (`--disaggregation-transfer-backend nixl`) and vLLM (`NixlConnector`). Where Mooncake is a production system with RDMA-specific optimisations, NIXL is a **vendor-agnostic abstraction** that works with any high-speed interconnect.

### The Transfer Agent

Each server process runs one NIXL **Transfer Agent**, managing three components:

**1. Memory Section** — unified view of all registered memory types:
- GPU VRAM (registered for GPUDirect RDMA)
- CPU DRAM (pinned)
- Local NVMe
- Remote storage (NVMe-oF or S3)

All exposed through the same buffer-list API — the caller doesn't need to know where the data physically lives.

**2. Transfer Backend Interface** — pluggable transports (selected automatically by source/destination memory type):
- UCX (RDMA InfiniBand/RoCEv2) — for inter-node GPU VRAM transfers
- NVIDIA Magnum IO GPUDirect Storage (GDS) — for NVMe → GPU transfers
- TCP — fallback when RDMA is unavailable
- NVLink — for within-rack GB200 NVL72

**3. Metadata Handler** — exchanges registration metadata between agents via etcd. Metadata is cached to avoid per-transfer round-trips.

### Three Use Cases in One Library

| Use case | Path |
|---|---|
| Disaggregation | KV blocks from prefill GPU VRAM to decode GPU VRAM (RDMA zero-copy) |
| Long-context KV loading | KV cache from NVMe/object storage to GPU VRAM (GDS path) |
| Expert parallelism | MoE all-to-all expert activations across GPUs (NVLink/RDMA) |

### Async Transfer API (Enables Compute-Transfer Overlap)

```python
# Submit non-blocking write
handle = agent.transfer_submit_write(src_addr, dst_addr, size)

# Check completion without blocking
status = agent.transfer_check_status(handle)
```

The async model allows compute and transfer to overlap. The decode server can begin processing layers that have already been transferred while later layers are still in transit — equivalent to Perplexity's layer-pipelined KV transfer strategy, implemented at the library level.

### Using NIXL in SGLang

```bash
# Start prefill worker with NIXL backend
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend nixl \
  --port 30000

# Switch to LIBFABRIC if UCX is unavailable
export SGLANG_DISAGGREGATION_NIXL_BACKEND=LIBFABRIC
```

Available NIXL backends: `UCX` (default, RDMA InfiniBand/RoCEv2), `LIBFABRIC`.

---

## 13. The Aggregation-vs-Disaggregation Decision: TaiChi's SLO-Regime Framework

TaiChi (arXiv, August 2025) provides the most rigorous answer to the question every production team needs to answer: "is PD disaggregation always better than aggregation?" The answer is **no — the optimal choice depends on which SLO is binding**.

### The Framework

TaiChi shows empirically and theoretically that the two approaches occupy different positions in the SLO space:

**When PD Aggregation is Optimal — tight TTFT + relaxed TPOT** (e.g., chatbots that must start responding quickly):
- All GPUs contribute to prefill simultaneously → lower TTFT
- TPOT interference violations are tolerable
- Aggregation achieves higher GPU utilisation by batching prefill and decode together
- Disaggregation would hurt TTFT: fewer total GPUs handle prefill (only the prefill pool)

**When PD Disaggregation is Optimal — tight TPOT + relaxed TTFT** (e.g., code generators streaming long outputs smoothly):
- Dedicated decode pool is never interrupted by prefill → stable TPOT
- Higher TTFT is acceptable (users wait longer for first token but then see smooth output)
- Aggregation causes prefill interference to spike TPOT, violating the strict TPOT SLO

**The Balanced SLO Problem — tight TTFT + tight TPOT** (e.g., agentic workflows needing both):
- PD aggregation: TPOT violations from interference
- PD disaggregation: TTFT violations from fewer GPUs handling prefill
- Neither approach is optimal; the system cannot satisfy both at the same request rate

### TaiChi's Solution: Dynamic Hybrid Switching

TaiChi assigns workers to **dynamic categories** that can be reassigned in real time:

- **P-heavy workers**: primarily process prefill batches (disaggregation-mode prefill)
- **D-heavy workers**: primarily process decode batches (disaggregation-mode decode)

In aggregation mode: all workers process mixed P+D batches (SARATHI-style).
In disaggregation mode: fixed split, KV transferred between pools.
In **hybrid mode**: the P-heavy / D-heavy split is dynamic, adjusted by an SLO monitoring loop:

```
Monitor TTFT violations → too many → shift some D-heavy workers → P-heavy
Monitor TPOT violations → too many → shift some P-heavy workers → D-heavy
At equilibrium: minimum workers of each type to satisfy both SLOs
```

### Results

Implemented on vLLM; evaluated on DeepSeek-R1, Llama-70B:

| Metric | TaiChi vs state-of-the-art |
|---|---|
| Goodput improvement | **Up to 77% over SOTA** |
| TTFT reduction vs pure PD disaggregation | **Up to 13.2×** (when TTFT SLO is tight) |
| TPOT reduction vs pure PD aggregation | **Up to 1.69×** (when TPOT SLO is tight) |

### The SLO Decision Matrix

| Workload type | TTFT constraint | TPOT constraint | Recommended approach |
|---|---|---|---|
| Chatbot (responsiveness) | Tight | Relaxed | PD Aggregation (SARATHI-style) |
| Code streamer (smooth output) | Relaxed | Tight | **PD Disaggregation** |
| Agentic AI (both matter) | Tight | Tight | TaiChi hybrid or large disagg cluster |
| Batch inference (offline) | Relaxed | Relaxed | Aggregation (maximise throughput) |
| RAG with long prompts | Moderate | Tight | **PD Disaggregation** |
| Multi-turn chat | Moderate | Moderate | TaiChi or cache-aware aggregation |
| MoE models (DeepSeek-V3) | — | — | **PD Disaggregation** (mandatory, not optional) |

**SGLang's current design**: pure disaggregation — optimal for TPOT-tight workloads. For TTFT-tight workloads (latency-sensitive chatbots), the trade-off deserves evaluation before deploying.

```bibtex
@article{wang2025taichi,
  title   = {Prefill-Decode Aggregation or Disaggregation? Unifying Both for
             Goodput-Optimized LLM Serving},
  author  = {Chao Wang and others},
  journal = {arXiv preprint arXiv:2508.01989},
  year    = {2025},
  url     = {https://arxiv.org/abs/2508.01989}
}
```

---

## 14. The KV Transfer Abstraction: vLLM's Connector Architecture

vLLM's disaggregated prefilling (building on Splitwise's PR #2809) provides the clearest open-source **KV transfer abstraction specification**. It defines what any inference framework needs to expose for disaggregation: six concrete connectors covering every major transfer protocol, and two abstract interfaces specifying the send/receive and ownership semantics.

> **Critical note from vLLM docs:** "Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." — This is the canonical one-sentence statement of what disaggregation optimises.

### The 6 Supported Connectors

| Connector | Transport | When to use |
|---|---|---|
| **NixlConnector** | RDMA InfiniBand/RoCEv2 via UCX | Default high-performance; production clusters with RDMA NICs |
| **MooncakeConnector** | RDMA, NVLink, TCP via Mooncake TE | Multi-NIC pooling + topology-aware path selection |
| **P2pNcclConnector** | NCCL P2P (PCIe or NVLink) | Clusters without RDMA NICs; requires proxy process |
| **LMCacheConnectorV1** | NIXL + LMCache storage | Cross-engine KV sharing + persistent storage unified |
| **ExampleConnector** | Reference implementation | Starting template for custom connectors |
| **MultiConnector** | Chains multiple connectors | RDMA → NCCL fallback; NIXL + LMCache combined |

### BaseKVConnector: The Core Abstraction

```python
class BaseKVConnector(ABC):
    def send_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches, hidden_or_intermediate_states
    ) -> None: ...

    def recv_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches
    ) -> Tuple[torch.Tensor, bool]: ...
```

- **`send_kv_caches`**: called on the **prefill instance** after the forward pass. Writes KV tensors to the transfer buffer.
- **`recv_kv_caches`**: called on the **decode instance** before its forward pass. Returns `(hidden_states, bypass_model_exec)` — if `bypass_model_exec=True`, the decode instance skips its own prefill computation entirely (it already has the KV from the transfer).

### KVLookupBufferBase: The Ownership Protocol

```python
class KVLookupBufferBase(ABC):
    def insert(self, input_tokens, roi, key, value, hidden) -> None: ...
    def drop_select(self, input_tokens, roi) -> Tuple[...]: ...
```

- **`insert`**: prefill instance inserts KV cache into the buffer.
- **`drop_select`**: decode instance **atomically** retrieves and removes the KV cache matching its request. The "drop" ensures **each KV cache is consumed by exactly one decode instance** — preventing double consumption or races.

This ownership protocol is the correct formulation for any KV transfer buffer — the same pattern SGLang's handshake implements at the framework level.

### The `kv_role` Field

The most important configuration dimension:
- `kv_producer`: this instance is the prefill worker (sends KV)
- `kv_consumer`: this instance is the decode worker (receives KV)
- `kv_both`: runs both phases — for testing or multi-turn scenarios where previous decode state feeds the next prefill

### vLLM vs SGLang Comparison

| Aspect | vLLM | SGLang |
|---|---|---|
| Transfer abstraction | `BaseKVConnector` (6 implementations) | `DisaggTransferBackend` (Mooncake, NIXL) |
| Configuration | JSON via `--kv-transfer-config` | CLI flags `--disaggregation-mode`, `--disaggregation-ib-device` |
| Router | External (Ray Serve, Dynamo, custom) | Built-in `sglang_router` |
| MoE support | General MoE | DeepEP + EPLB integration |
| Multi-connector chaining | MultiConnector | Not yet |
| Cache-aware routing | Not built in | Round-robin (prefix-aware planned) |

---

## 15. Launching a Disaggregated Cluster with SGLang

### Minimal Setup: Single Node (Mooncake Backend)

```bash
# 1. Install
uv pip install mooncake-transfer-engine

# 2. Prefill worker (GPU 0)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode prefill \
  --port 30000 \
  --disaggregation-ib-device mlx5_roce0

# 3. Decode worker (GPU 1)
python -m sglang.launch_server \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --disaggregation-mode decode \
  --port 30001 --base-gpu-id 1 \
  --disaggregation-ib-device mlx5_roce0

# 4. Router (client-facing)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 \
  --decode http://127.0.0.1:30001 \
  --host 0.0.0.0 --port 8000
```

### Multi-Node Setup: DeepSeek-V3 on 12 Nodes

```bash
# Prefill node 0 (of 2 prefill nodes)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode prefill \
  --disaggregation-ib-device ${device_name} \
  --host ${local_ip} --port 30000 \
  --dist-init-addr ${prefill_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep \
  --mem-fraction-static 0.8

# Decode node 0 (of 9 decode nodes)
python -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V3-0324 \
  --disaggregation-mode decode \
  --disaggregation-ib-device ${device_name} \
  --host ${local_ip} --port 30001 \
  --dist-init-addr ${decode_master_ip}:5000 \
  --nnodes 2 --node-rank 0 \
  --tp-size 16 --dp-size 8 \
  --enable-dp-attention --moe-a2a-backend deepep \
  --mem-fraction-static 0.8 \
  --max-running-requests 128
```

### Key CLI Flags

| Flag | Effect |
|---|---|
| `--disaggregation-mode prefill/decode` | Converts server into a phase-specific worker |
| `--disaggregation-transfer-backend mooncake/nixl` | Selects KV transfer engine |
| `--disaggregation-ib-device mlx5_roce0` | Specifies RDMA NIC for KV transfer |
| `--max-running-requests 128` | Caps concurrent decode requests (prevents OOM) |
| `--moe-a2a-backend deepep` | Activates DeepEP for MoE expert parallelism |
| `--enable-dp-attention` | Required alongside DeepEP for DeepSeek models |

### The Handshake Protocol

SGLang's pre-allocation handshake ensures correctness:

```
Step 1: Decode Server pre-allocates KV cache pages
         → sends KV page indices to Prefill Server

Step 2: Prefill Server receives KV indices
         → runs model forward pass
         → writes KV cache into Decode Server's pre-allocated pages (RDMA)
         → sends completion notification

Step 3: Decode Server receives KV cache
         → begins autoregressive generation
         → streams output tokens to client
```

The **decode pre-allocates before prefill runs** — preventing the race condition where the decode server has no space when prefill tries to write.

### Critical Environment Variables

**Prefill server:**

| Variable | Purpose | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_QUEUE_SIZE` | Parallel transfer queues (concurrent KV transfers to multiple decode instances) | `4` |
| `SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT` | Seconds to wait for decode to send KV page indices | `300` |

**Decode server:**

| Variable | Purpose | Default |
|---|---|---|
| `SGLANG_DISAGGREGATION_HEARTBEAT_INTERVAL` | Health-check interval to prefill servers (sec) | `5.0` |
| `SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE` | Consecutive failures before marking prefill offline | `2` |

**NVL72 NVLink transport:**

```bash
export SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK
export MC_FORCE_MNNVL=True
```

**Heterogeneous TP staging buffer** (when prefill TP ≠ decode TP; non-MLA models only):

```bash
export SGLANG_DISAGG_STAGING_BUFFER=1
export SGLANG_DISAGG_STAGING_BUFFER_SIZE_MB=64   # prefill side per-worker
export SGLANG_DISAGG_STAGING_POOL_SIZE_MB=4096    # decode side ring buffer
# Result: 2–5× throughput improvement at high concurrency
# Note: Do NOT use for DeepSeek-V2/V3 (MLA architecture)
```

---

## 16. Production Evidence at Scale

### Perplexity AI: 435M Queries/Month (Dec 2024)

Perplexity AI actively deploys disaggregated serving in production at 435M+ search queries/month, serving 20+ AI models simultaneously on H100 SXM GPUs.

> "This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token. Additionally, this technique gives Perplexity the flexibility to use **different NVIDIA GPU products for each inference phase** given its specific hardware resource requirements." — NVIDIA Spotlight, December 2024

Key facts:
- Disaggregated serving in **active production** (not evaluation)
- Hardware heterogeneity leveraged as a production benefit
- ITL P99 impact: power-of-two routing choices → **lowest** P99; round-robin → **highest** (confirmed disaggregation advantage)
- Cost savings: ~$1M/year on the Related-Questions feature alone

### SGLang + DeepSeek-V3 on 96 H100 GPUs (May 2025)

The LMSYS/SGLang team deployed DeepSeek-V3 (671B MoE) across 12 nodes × 8 H100 GPUs:

| Configuration | Value |
|---|---|
| Hardware | 96 H100 GPUs (12 nodes × 8) |
| Prefill pool | 3 nodes (24 GPUs) |
| Decode pool | 9 nodes (72 GPUs) — 3:9 ratio |
| KV transfer | Mooncake + InfiniBand |
| Input throughput | **52,300 tokens/second per node** |
| Output throughput | **22,300 tokens/second per node** |

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team

**Why PD disaggregation was mandatory** (not optional) for this deployment: DeepSeek-V3's expert parallelism library (DeepEP) uses incompatible dispatch patterns for prefill ("normal" mode for large batches) and decode ("low-latency" mode for single-token batches). These patterns cannot coexist on the same GPU workers. Disaggregation is the only way to give each phase its optimal dispatch mode.

### Moonshot AI — Kimi K2 (Jul 2025)

| Configuration | Value |
|---|---|
| Hardware | 128 H200 GPUs |
| Transfer engine | Mooncake Transfer Engine + InfiniBand |
| Prefill throughput | **224,000 tokens/second** |
| Decode throughput | **288,000 tokens/second** |

### Additional Production Adopters

| Company | Stack |
|---|---|
| Meta | vLLM disaggregated serving |
| LinkedIn | vLLM disaggregated serving |
| Mistral | vLLM disaggregated serving |
| HuggingFace | vLLM disaggregated serving |
| Moonshot AI (Kimi) | SGLang + Mooncake Transfer Engine |
| NVIDIA (Dynamo) | Enterprise orchestration layer for disaggregated serving |

---

## 17. Hardware Heterogeneity and the Cost Arithmetic

### Per-Phase Hardware Profiles

| | Prefill Pool | Decode Pool |
|---|---|---|
| Bottleneck | Compute (tensor cores) | Memory bandwidth (HBM) |
| Needs | High FLOPS | High HBM bandwidth + large HBM capacity |
| Best GPU | H100 SXM (high FLOPS) | H200 (larger HBM3e, higher BW) |
| Why H200 is overkill for prefill | Paying for HBM capacity and BW that prefill doesn't use | — |
| Why H200 is better for decode | — | Larger HBM fits more concurrent KV caches; higher BW reduces ITL |

**SPAD paper validation (UT Austin):**
- Prefill chip with **40% less HBM bandwidth** → only **17% prefill latency increase** (prefill doesn't use bandwidth)
- Decode chip with **50% less compute** → only **22% decode latency increase** (decode doesn't use compute)

This validates using different hardware for each phase: the silicon cost savings from removing unused capability are real.

### Cluster-Level Cost Reduction

**Splitwise numbers** (H100 prefill + A100 decode vs all-H100):
- Same cost budget: **2.35× more throughput**
- Same throughput target: **1.4× higher throughput at 20% lower cost**

**InfoQ cluster-level analysis**: **15–40% total infrastructure cost reduction** from disaggregation at cluster scale. This comes from: not overprovisioning hardware for both phases, eliminating idle tensor cores during decode, and scaling each pool independently.

### The Adoption Timeline

DistServe was published at OSDI 2024. Within 18 months, disaggregated serving was running in production at Perplexity, Meta, LinkedIn, Mistral, and Moonshot AI, and NVIDIA built an entire framework (Dynamo) around it. Teams that sized for disaggregation and designed InfiniBand network topology accordingly are paying measurably less per token than teams that have not.

---

## 18. The Decision Framework: 5 Checks Before Refactoring

Before refactoring your serving stack — or before disagreeing with disaggregation — run these 5 checks. Apply TaiChi's SLO-regime framework first (§13) to confirm disaggregation is the right architecture for your SLO constraint, then use these checks to validate feasibility.

**Check 1 — Measure your decode/prefill time ratio**
What fraction of wall-clock GPU time is decode vs prefill? If decode < 70%: smaller payoff. If decode > 85%: you are paying for idle tensor cores most of the day.

**Check 2 — Calculate your KV cache transfer size**
Use the formula in §9 with your actual model and median prompt length. If > 500 MB/request and network < 100 Gbps: transfer latency eats into TTFT budget.

**Check 3 — Check your prefix cache hit rate**
If prefix cache hit rate > 80%, the decode worker already holds most of the KV from previous turns. A separate prefill pool adds a network round-trip for data that's essentially local. Consider local prefill on the decode worker for high-hit-rate requests.

**Check 4 — Count your GPUs**
Below ~16 GPUs: scheduling overhead typically exceeds utilisation gain. Above 32 GPUs with sustained traffic: cost savings compound.

**Check 5 — Audit your network**
RDMA-capable NICs? (EFA on AWS, ConnectX on bare metal). InfiniBand vs 100 GbE? Run KVBench (NIXL) against your actual cluster topology to measure effective transfer bandwidth before committing.

**Decision**: if checks 1, 4, and 5 are all favorable, disaggregation reduces per-token cost. If Check 3 is the concern (high prefix hit rate), evaluate a hybrid where high-hit requests run local prefill on decode workers and only long, cold prompts go through the prefill pool.

### When Not to Disaggregate

| Scenario | What to use instead |
|---|---|
| Development / testing | Standard vLLM or SGLang, monolithic |
| Short prompts + short outputs (< 512 tokens in, < 100 out) | Monolithic + SARATHI chunked prefill |
| Batch-only workloads (TTFT irrelevant) | Aggregation (maximise throughput) |
| < 16 total GPUs | Too small for two-pool overhead |
| No RDMA, TCP only | Works for long-context, marginal for typical workloads |
| Tight TTFT + relaxed TPOT (chatbots) | Consider aggregation per TaiChi's SLO framework |

---

## Key Quotes

> "To make sure inference is not compute- and bandwidth-bound at the same time, we want to separate them." — Jackson MZ

> "Since prefill primarily determines the TTFT and decode impacts ITL, collocating them makes it difficult to optimize both metrics simultaneously." — BentoML

> "A Llama 70B model running inference on an H100 GPU hits 92% compute utilization during prefill. Thirty milliseconds later, during decode, that same GPU drops to 30%." — TDS, April 2026

> "Goodput = number of requests that meet both the TTFT SLO and the TPOT SLO per unit time." — Hao Zhang, DistServe / CMU LLM Systems

> "Token generation phases do not require the compute capability of the latest GPUs and can be run on lower-power, lower-cost hardware with equivalent quality." — Splitwise (ISCA 2024)

> "Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." — vLLM documentation

> "This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token." — NVIDIA Spotlight on Perplexity

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." — SGLang Team, May 2025

---

## What Is Left Out and Why

Sections moved to `OMITTEDL3L4.md`:

### SplitwiseSim CLI Details

SplitwiseSim is a discrete-event simulator for offline cluster design evaluation. The simulator inputs, configuration format, and example trace files are L4+/tool-level detail. The simulator's purpose and scope are described in §7; the CLI is omitted.

### NIXLBench and KVBench Commands

NIXLBench sweeps block sizes and batch sizes; KVBench auto-calculates KV sizes for known models and generates NIXLBench commands. These tools are mentioned in §12 and §18 (Check 5). The command-line details belong in an operator runbook, not a conceptual blog.

### TaiChi PPD Three-Way Disaggregation

TaiChi also proposes PPD disaggregation — a third worker type (Prompt worker) for context history KV loading in multi-turn conversations. This extension is theoretically motivated but not implemented in any production framework. Omitted as it adds complexity without actionable guidance for current SGLang deployments.

### Mooncake Prediction-Based Early Rejection

Mooncake's scheduler uses a prediction model to reject requests early (under heavy load) if they cannot be served within SLO. SGLang's current disaggregation does not implement this. Mentioned in §10; mechanism omitted.

### DeepEP/EPLB MoE Dispatch Internals

The DeepEP "normal" vs "low-latency" dispatch mode conflict and EPLB expert load balancing algorithm are mentioned as the reason MoE models require disaggregation (§16). The dispatch mode mechanics and EPLB configuration details are omitted.

### vLLM `KVTransferConfig` Full JSON Schema

The complete `KVTransferConfig` dataclass with all fields, types, and defaults is omitted. The `kv_role` field (the most important) is described in §14; the full schema is in `L4/03_vllm_disagg_connector.md`.

### Dynamo Control Plane Feedback Loop Parameters

The Planner's reaction time, cooldown configuration, and YAML deployment format are enterprise deployment specifics beyond the conceptual architecture described in §11.
