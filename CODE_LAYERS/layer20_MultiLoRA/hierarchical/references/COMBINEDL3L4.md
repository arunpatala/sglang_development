# Multi-LoRA Inference — Mathematics, Systems, and Production Architecture

**Level:** L3 + L4 — Superset of `COMBINEDL1L2.md`. Adds formal LoRA mathematics from the original paper, Punica SGMV kernel internals, S-LoRA Unified Paging, dLoRA's dynamic merge/unmerge decision algorithm, CaraServe's cold-start elimination, Loquetier's unified fine-tuning+serving framework, ServerlessLoRA and Predictive-LoRA's serverless architecture, and InfiniLoRA's disaggregated LoRA design for MoE models. All L1+L2 concepts are retained and expanded.

**What this file is:** A single coherent article synthesising all L3 and L4 source material into a progressive narrative building on the L1+L2 foundation. Every section from `COMBINEDL1L2.md` is present; most are extended with quantified measurements, formal definitions, or implementation detail drawn from the papers.

**Sources synthesised:**
- L1+L2 (all — retained and expanded from `COMBINEDL1L2.md`)
- L3/01 — LoRA: Low-Rank Adaptation of Large Language Models (Hu et al., ICLR 2022): rank decomposition proof, scaling analysis, empirical rank-deficiency study
- L3/02 — Punica: Multi-Tenant LoRA Serving (Chen et al., MLSys 2024): SGMV kernel design, BGMV variant, 12× throughput result
- L3/03 — S-LoRA: Serving Thousands of Concurrent LoRA Adapters (Sheng et al., MLSys 2024): Unified Paging, tensor parallelism strategy, 4× throughput, 2000+ concurrent adapters
- L4/01 — dLoRA: Dynamically Orchestrating Requests and Adapters (Wu et al., OSDI 2024): credit-based merge/unmerge, request-adapter co-migration, 1.8× latency over S-LoRA
- L4/02 — CaraServe: CPU-Assisted and Rank-Aware LoRA Serving (Li et al., 2024): CPU-assisted prefill, rank-aware scheduling, 99% SLO attainment
- L4/03 — Loquetier: Virtualized Multi-LoRA Framework (Zhang et al., NeurIPS 2025): unified fine-tuning+serving, Virtualized Module abstraction, 3× inference throughput
- L4/04 — ServerlessLoRA (Sui et al., 2025): backbone sharing, pre-loading, contention-aware batching, 86% TTFT reduction
- L4/05 — Predictive-LoRA (Tang et al., 2025): LSTM traffic predictor, page-based memory management, 1.52× throughput over S-LoRA
- L4/06 — InfiniLoRA (Chen et al., 2026): disaggregated LoRA server, GPU-initiated RDMA, MoE-aware design, 3.05× request rate

**Omitted (moved to `OMITTEDL3L4.md`):** Punica SGMV CUDA thread block assignment code, S-LoRA queuing model formal proof, dLoRA convergence analysis, CaraServe CPU profiler threshold derivation, Loquetier full kernel source, Predictive-LoRA LSTM training loop, InfiniLoRA SLO provisioning solver, DoRA/rsLoRA/PiSSA/EVA variant derivations.

---

## Section Plan

| § | Title | Primary sources | Reading time |
|---|-------|----------------|------|
| 1 | [The Fine-Tuning Explosion and the Serving Crisis](#1-the-fine-tuning-explosion-and-the-serving-crisis) | L1/01, L1/02, L1/03, L4/04 | 4 min |
| 2 | [LoRA Mathematics: The Low-Rank Hypothesis](#2-lora-mathematics-the-low-rank-hypothesis) | L2/03, L3/01 | 6 min |
| 3 | [The Serving Problem: One Base Model, One Thousand Adapters](#3-the-serving-problem-one-base-model-one-thousand-adapters) | L1/02, L1/03, L3/01 | 4 min |
| 4 | [The SGMV Kernel: Batching Across Different Adapters](#4-the-sgmv-kernel-batching-across-different-adapters) | L1/03, L3/02 | 6 min |
| 5 | [Three Pillars of Production Multi-Adapter Serving](#5-three-pillars-of-production-multi-adapter-serving) | L1/03, L3/03 | 5 min |
| 6 | [The Live Benchmark: 50 Adapters at the Cost of One](#6-the-live-benchmark-50-adapters-at-the-cost-of-one) | L1/02, L3/02 | 3 min |
| 7 | [S-LoRA: Scaling to Thousands of Concurrent Adapters](#7-s-lora-scaling-to-thousands-of-concurrent-adapters) | L3/03 | 6 min |
| 8 | [dLoRA: Dynamic Merge/Unmerge and Request Migration](#8-dlora-dynamic-mergeunmerge-and-request-migration) | L4/01 | 6 min |
| 9 | [CaraServe: CPU-Assisted Cold-Start Elimination](#9-caraserve-cpu-assisted-cold-start-elimination) | L4/02 | 5 min |
| 10 | [Serving LoRA Adapters in SGLang](#10-serving-lora-adapters-in-sglang) | L2/01, L3/03 | 5 min |
| 11 | [Serving LoRA Adapters in vLLM](#11-serving-lora-adapters-in-vllm) | L2/02 | 4 min |
| 12 | [The HuggingFace PEFT Checkpoint Format](#12-the-huggingface-peft-checkpoint-format) | L2/03, L3/01 | 4 min |
| 13 | [Advanced Production Features](#13-advanced-production-features) | L1/03, L2/02, L4/03 | 4 min |
| 14 | [The KV Cache vs Adapter VRAM Trade-off](#14-the-kv-cache-vs-adapter-vram-trade-off) | L1/02, L1/03, L3/03 | 4 min |
| 15 | [Serverless LoRA: Backbone Sharing and Predictive Loading](#15-serverless-lora-backbone-sharing-and-predictive-loading) | L4/04, L4/05 | 5 min |
| 16 | [Loquetier: Unified Fine-Tuning and Serving](#16-loquetier-unified-fine-tuning-and-serving) | L4/03 | 4 min |
| 17 | [InfiniLoRA: Disaggregated LoRA for MoE Models](#17-infinilora-disaggregated-lora-for-moe-models) | L4/06 | 5 min |
| 18 | [Decision Framework: When Does Multi-LoRA Serving Pay Off?](#18-decision-framework-when-does-multi-lora-serving-pay-off) | L1/01, L1/02, L4/01 | 4 min |

**Total reading time:** ~84 minutes

---

## 1. The Fine-Tuning Explosion and the Serving Crisis

Fine-tuning large language models has never been easier. Datasets are accessible, tooling is mature, and the PEFT ecosystem has made the process inexpensive. A motivated ML team can produce dozens of task-specific fine-tuned models in a single sprint.

The problem arrives the moment those models go to production.

A 7-billion-parameter model in half-precision (FP16) occupies approximately **14 gigabytes of GPU VRAM** just for its static weights — before inference, before KV cache, before anything. An enterprise SaaS provider offering custom-tuned models for each of 1,000 customers would, in the traditional deployment model, require 1,000 GPU instances. At $2–$5 per H100 hour, that is millions of dollars monthly in GPU infrastructure — with most instances sitting idle at any given time because most customers are not querying at the same moment.

The industry's response to the training cost was Parameter-Efficient Fine-Tuning. LoRA and its variants reduced the trainable parameter count to under 1% of the model. A full LLaMA-3.1-8B fine-tune produces 16 GB of weights; a LoRA adapter fine-tune produces 50–200 MB of adapter weights. The 8B parameters of the base model are never touched.

But LoRA solved the training cost without solving the serving cost. Until recently, most inference frameworks required adapters to be **merged** into the base model before serving — effectively recreating the full-weight model for each adapter. You gained on training storage and training compute, and then spent it all back on inference infrastructure.

> "Fine-tuning large language models is easier than ever. Serving them efficiently? That's where LoRAX steps in." — Sai Mudhiganti, 2025

The serverless deployment model amplifies this problem further. ServerlessLoRA (2025) quantified it directly: a naive serverless deployment with 10 concurrent LoRA functions loads the base model 10 separate times — one per function — consuming 140 GB of VRAM for a single 14 GB base model, 99% of which is redundant duplication. The serving crisis is not merely a throughput problem; it is a resource allocation problem.

This is the serving crisis: not a shortage of fine-tuned models, but the inability to serve many of them cheaply from shared hardware. The seven years from the LoRA paper (2021) to InfiniLoRA (2026) represent the research community's systematic attack on this problem.

---

## 2. LoRA Mathematics: The Low-Rank Hypothesis

To understand multi-adapter serving deeply, you need the formal mathematical foundation from the original LoRA paper (Hu et al., ICLR 2022). The ideas in §§ 3–17 all reduce to manipulations of these few equations.

### The Core Hypothesis

Full fine-tuning updates all model weights: `W = W₀ + ΔW`. The trainable delta `ΔW` lives in a high-dimensional space (`ℝ^{d×k}`). The LoRA hypothesis is:

> **The intrinsic dimensionality of the adaptation `ΔW` is very low.** There exists a low-rank matrix `BA` where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, and `r ≪ min(d,k)` such that `ΔW ≈ BA`.

Instead of updating `W` directly, LoRA freezes `W₀` and trains only `A` and `B`. The number of trainable parameters drops from `d×k` to `(d+k)×r`. For a typical 4096×4096 weight matrix with rank 8, that is `(4096+4096)×8 = 65,536` vs `4096×4096 = 16,777,216` — a **256× reduction**.

### The Forward Pass

For a linear layer with weight `W₀ ∈ ℝ^{d×k}` and input `x ∈ ℝ^{k}`:

```
h = W₀x + ΔWx
  = W₀x + BAx · (α/r)
```

Where:
- `W₀`: frozen pre-trained weight matrix
- `A ∈ ℝ^{r×k}`: "down-projection" — maps input to rank-r space
- `B ∈ ℝ^{d×r}`: "up-projection" — maps rank-r space back to output
- `α`: scaling hyperparameter
- `r`: rank, typically 1–64

**In transposed notation (PyTorch `F.linear(x, W)` convention):**

```python
h = x @ W₀.T + (x @ A.T) @ B.T * (alpha / r)
```

This is exactly what `LoRAAdapter.apply()` in Layer 20 computes.

### Weight Initialization

The initialization strategy is non-trivial and deliberate:

- **A initialized with Kaiming-uniform (random)** — ensures gradients flow from the first training step
- **B initialized to zeros** — so `BA = 0` at initialization, meaning the pre-trained model behavior is preserved at the start of training

This means LoRA fine-tuning begins from the exact pre-trained behavior. The model starts with no perturbation and gradually learns the correction.

### Scaling: Why `alpha/r`?

The scaling factor `alpha/r` ensures the adaptation magnitude is comparable across different ranks:

- If `r` doubles (double the parameters), the scale is halved to compensate
- `alpha` is kept fixed as a hyperparameter (often set to `r` for simplicity, or `2r` for stronger adaptation)
- When `alpha = r`, the effective scaling is 1.0

**For `phh/Qwen3-0.6B-TLDR-Lora` (used in Layer 20):** `r = 8`, `lora_alpha = 32`, effective scaling = `32/8 = 4.0`.

### Which Layers to Target?

LoRA can be applied to any weight matrix. The paper's empirical finding: **targeting more matrices with a smaller rank outperforms targeting fewer matrices with larger rank**, given the same parameter budget.

| Target | Parameters | Typical use |
|---|---|---|
| `q_proj`, `v_proj` | Minimal | Paper recommendation baseline |
| All attention (`q`, `k`, `v`, `o`) | Balanced | Full attention coverage |
| All attention + MLP (`gate`, `up`, `down`) | Full | Best quality, more memory |

Modern adapters like `phh/Qwen3-0.6B-TLDR-Lora` target all 7 projection modules at rank-8. This is why `LoRAAdapter._load_weights()` handles keys for all 7 module types.

### Merge vs. Separate: The Serving Decision

For a single fixed adapter, LoRA can be **merged** into the base weights:

```
W = W₀ + BA · (alpha/r)
```

After merging, the forward pass is exactly `Wx` — zero additional overhead, one GEMM per layer, identical cost to the base model. However, merging "bakes in" one adapter and destroys the base model's separability.

**Layer 20 deliberately avoids merging** to enable mixed-batch serving. This introduces the masked delta computation overhead:

```python
output = base_output + (x @ A.T) @ B.T * scaling * lora_mask
```

For pure single-adapter deployments with no base-model traffic, merging is strictly optimal. For mixed batches (some tokens want base, some want LoRA), the unmerged approach is required.

### The Rank-Deficiency Investigation

The paper includes an empirical study showing that the optimal fine-tuning subspace is genuinely low-rank:

- Pre-training subspace and fine-tuning subspace have high overlap
- The adaptation direction is well-captured by rank-1 to rank-4 matrices for most NLP tasks
- This is why small `r` values (4, 8) work well in practice

**Empirical results from the paper (GPT-3 175B):**

| Method | Trainable params | Memory reduction | Performance vs full FT |
|---|---|---|---|
| Full fine-tuning | 175B | — | baseline |
| Adapter layers | ~1.1M | 3× | comparable |
| LoRA (r=4) | ~4.7M | **3×** | **on-par or better** |
| LoRA (r=8) | ~9.4M | **3×** | **on-par or better** |

The rank-deficiency observation motivates adaptive rank allocation (EVA in PEFT, dLoRA's adapter-specific rank optimization), which is why the research does not simply standardize on rank-64 for everything.

---

## 3. The Serving Problem: One Base Model, One Thousand Adapters

With the LoRA math understood, the serving problem becomes precise.

### Approach 1: Merge Weights (Naïve)

Merge each adapter into its own copy of the base model, serve independently. This produces correct results but completely undoes LoRA's storage savings: 14 GB per adapter, 1,000 adapters = 14 TB of VRAM across 1,000 GPU instances.

Furthermore, merged serving kills batch efficiency: requests for different adapters cannot be batched together.

### Approach 2: Separate Computation (Efficient)

```
h = W₀ · x  +  B · A · x · (α/r)
  = (base contribution)  +  (adapter contribution)
```

- **Base contribution:** computed once for the entire batch — all tokens, regardless of adapter
- **Adapter contribution:** computed per-adapter, applied per-token

The base model is held once regardless of adapter count. 14 GB loaded once. The forward pass for `W₀ · x` is shared across all requests in the batch.

> "There are two methods to compute h using LoRA... approach 2) seems to be more suitable for a multi-task/tenancy scenario, as we can compute the left term once, the right term for every adapter n, and sum them as needed." — João Moura (AWS), 2024

### The Compute Asymmetry

Why is the adapter contribution cheap relative to the base model?

For LLaMA-7B with a 4096-token context:
- Base model attention: `32 layers × 4096 × 4096 × 32 ≈ 536M` multiply-adds per layer pass
- LoRA delta (rank-8): `2 × 4096 × 8 = 65,536` multiply-adds per layer
- **LoRA overhead: 0.012% of base model computation**

SGMV's task is to compute that 0.012% efficiently without the overhead of separate kernel launches for each adapter. Once solved, the adapter count becomes irrelevant to throughput.

### Why the Adapter Computation Is the Hard Part

A naive implementation iterates over adapters and launches a separate matrix multiply for each group of tokens sharing that adapter. With 50 adapters and diverse access patterns, the GPU launches 50 separate matrix multiplications per layer — each tiny, each with kernel scheduling overhead. GPU utilisation collapses.

This is precisely the problem Punica solved with its SGMV kernel.

---

## 4. The SGMV Kernel: Batching Across Different Adapters

Punica (Chen et al., MLSys 2024) introduced the Segmented Gather Matrix-Vector Multiplication (SGMV) kernel — the core computational primitive that all production multi-LoRA systems build on.

### The Batching Problem

For a batch with tokens using different adapters:
```
Token 0: adapter (A₁, B₁)  →  delta₀ = x₀ @ A₁.T @ B₁.T * scale₁
Token 1: adapter (A₂, B₂)  →  delta₁ = x₁ @ A₂.T @ B₂.T * scale₂
Token 2: adapter (A₁, B₁)  →  delta₂ = x₂ @ A₁.T @ B₁.T * scale₁
Token 3: (base only)        →  delta₃ = 0
```

Standard cuBLAS GEMMs are optimized for large batches of identical operations. The per-adapter loop approach launches `O(num_adapters)` separate kernels — each kernel has startup overhead (~5–10μs on modern GPUs), memory transfer cost, and poor occupancy for small groups.

### SGMV Algorithm

**Segmented Gather Matrix-Vector Multiplication** executes all adapter deltas in a single kernel launch:

1. **Gather:** For each token `i`, identify which adapter weight matrices `(Aₙ, Bₙ)` to use
2. **Segment:** Group tokens by adapter to maximize data locality (tokens 0 and 2 above are in the same segment for adapter 1)
3. **Multiply:** Execute down-projection `x @ Aₙᵀ` and up-projection `h @ Bₙᵀ` for all segments in one fused pass using custom CUDA thread block scheduling

**Key CUDA insight:** Standard cuBLAS GEMMs are optimized for large uniform batches. SGMV uses a custom kernel that assigns CUDA thread blocks to segments, executes standard GEMM within each segment, handles variable-length segments gracefully, and avoids per-kernel-launch overhead by fusing all segments into one launch.

| Approach | Kernel launches | GPU utilization |
|---|---|---|
| Naive (per-adapter loop) | O(num_adapters) | poor for many adapters |
| SGMV | 1 | near-optimal |

### Benchmark: 12× Throughput, +2ms Per Token

On a single A100 80GB GPU with LLaMA-7B:

> "Punica achieves 12× higher throughput in serving multiple LoRA models compared to state-of-the-art LLM serving systems while only adding 2ms latency per token." — Punica (MLSys 2024)

The 12× throughput comes from:
1. Base model GEMM computed once per batch (not per-adapter)
2. SGMV eliminates per-adapter kernel launch overhead

The 2ms latency overhead is **nearly constant regardless of the number of adapters**. You pay 2ms extra per token for the SGMV computation; the actual adapter count in the batch doesn't change that cost significantly.

### SGMV vs BGMV

Later work introduced **BGMV (Batched Gather Matrix-Vector Multiplication)**, optimized for the decode phase:

| Kernel | Optimized for | Used by |
|---|---|---|
| SGMV | Prefill (batch of tokens, long sequences) | LoRAX, SGLang Triton backend |
| BGMV | Decode (one token per request) | vLLM |

During decode, each request generates exactly one token — making the "batch" a collection of single-token requests from many concurrent sequences. BGMV is specifically designed for this access pattern, where batch dimension is large but sequence length per request is 1.

### The Masked Alternative (Layer 20)

Layer 20 implements a simpler — but correct — alternative to SGMV:

```python
# Layer 20 approach
out = base_output + delta * lora_mask   # mask ∈ {0.0, 1.0} per token
```

Comparison:

| Layer 20 | Punica SGMV |
|---|---|
| Computes delta for ALL tokens | Computes delta only for LoRA tokens |
| `out = base_out + delta * lora_mask` | `out = base_out + sgmv_delta` |
| O(n_tokens × r × d) extra FLOPs | O(n_lora_tokens × r × d) extra FLOPs |
| ~10 lines of Python | ~500 lines of custom CUDA |

For a batch of 100 tokens where 10 need a LoRA adapter, the mask approach wastes 90% of the extra FLOPs on zero-multiplied results. SGMV skips the computation entirely for base-model tokens.

The masked approach is the correct choice for Layer 20's single-adapter pedagogical implementation. SGMV is the production upgrade for multi-adapter batching at scale.

---

## 5. Three Pillars of Production Multi-Adapter Serving

With SGMV solving the compute problem, the remaining challenges are operational: memory management for large adapter pools and scheduling across concurrent adapters.

LoRAX (Predibase's open-source inference server) assembled all components into a production-ready architecture resting on three pillars.

### Pillar 1: Dynamic Adapter Loading

Rather than loading all adapters at server startup, LoRAX loads adapters just-in-time:

1. Server starts with only the base model in VRAM
2. Request arrives specifying `adapter_id`
3. If adapter is in VRAM: immediate processing
4. If not: fetch from HuggingFace Hub, S3, or local disk; queue the request; process once loaded

> "The time taken to load an adapter is a function of its size. Since LoRA adapters are typically between 10MB and 200MB, the load time is measured in hundreds of milliseconds." — Neel Shah (Towards AI), 2026

### Pillar 2: Tiered Weight Caching

A three-tier cache hierarchy manages adapter weight lifetime:

| Tier | Storage | Policy | Speed |
|---|---|---|---|
| **Hot** | GPU VRAM | Currently active | Immediate |
| **Warm** | Host DRAM | LRU eviction from GPU | ~10ms reload |
| **Cold** | NVMe / S3 | Full adapter catalog | ~100ms–seconds |

When GPU VRAM is full and a new adapter is needed, the LRU adapter is evicted to DRAM. A warm start (DRAM → VRAM) is near-instantaneous compared to a cold start (network download → VRAM).

This hierarchy enables a server to hold, in principle, thousands of adapters — as many as fit on disk. S-LoRA's Unified Paging (§ 7) formalises this as a rigorous memory management system.

### Pillar 3: Continuous Multi-Adapter Batching

A heterogeneous scheduler extends continuous batching to work across multiple adapters:

1. A fair scheduler marks a subset of adapters as "active" at any given time
2. Requests from active adapters are drained from their queues and combined into a single heterogeneous batch
3. SGMV computes all adapter deltas in one pass, with a mathematical mask ensuring each token uses the correct adapter's weights
4. After a configurable time window, the scheduler rotates: a different set of adapters becomes active

> "Requests from active adapters are drained from their queues and combined into a single batch. A mathematical mask is applied during the computation of activations to ensure that each input sequence is processed by the correct adapter weights." — Neel Shah (Towards AI), 2026

The mask in this description is the production version of Layer 20's `lora_mask` approach — extended to N adapters and powered by SGMV instead of dense matmuls.

---

## 6. The Live Benchmark: 50 Adapters at the Cost of One

The theoretical claims from Punica are compelling. The most concrete empirical evidence comes from an end-to-end benchmark by João Moura (AWS) on Amazon SageMaker, January 2024.

### Setup

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Adapter:** `vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k` (rank-8)
- **Replication:** 50 copies uploaded under different S3 prefixes (simulating 50 distinct adapters)
- **Hardware:** AWS `ml.g5.xlarge` — one NVIDIA A10G 24 GB GPU
- **Concurrency:** 20 parallel clients, 300 total requests

### Result

```
Single Adapter:
  Total Time: 42.34s  |  Average Latency: 2.80s  |  Throughput: 7.09 req/s

50 Adapters (random access):
  Total Time: 42.60s  |  Average Latency: 2.82s  |  Throughput: 7.04 req/s
```

**Virtually identical performance.** Serving 50 different adapters randomly costs almost the same as serving one adapter repeatedly.

> "We would effectively now be serving 50 different models on a single A10G." — João Moura, 2024

### Why It Works: The FLOPs Arithmetic

For Mistral-7B with a 4096-token context:
- Base model GEMM: ~536M multiply-adds per layer pass (dominates)
- LoRA delta (rank-8): 65,536 multiply-adds per layer (0.012% overhead)

SGMV's task is to compute that 0.012% correctly per token without per-adapter kernel overhead. Once it succeeds, the adapter count becomes irrelevant to throughput. The Punica 12× claim is confirmed end-to-end on production hardware.

---

## 7. S-LoRA: Scaling to Thousands of Concurrent Adapters

Punica demonstrated multi-LoRA serving without throughput degradation but assumed all adapters fit in GPU VRAM simultaneously. For large adapter pools, this is infeasible.

**S-LoRA (Sheng et al., MLSys 2024)** addresses three problems Punica didn't:
1. **Memory fragmentation** — adapters have different ranks, causing holes when loaded/evicted
2. **KV cache contention** — adapters and KV cache compete for the same GPU VRAM
3. **Cold-start latency** — loading an adapter from CPU takes time; need careful scheduling

### Unified Paging

The central S-LoRA innovation treats adapter weights and KV cache entries as **first-class citizens in a shared paged memory pool**.

**Analogy to PagedAttention (vLLM):** vLLM manages KV cache in fixed-size pages, eliminating fragmentation from variable-length sequences. S-LoRA extends this:

- **KV cache pages:** each page holds `page_size × num_heads × head_dim` elements
- **Adapter pages:** each page holds a fixed-size chunk of adapter A or B matrix

Both live in the **same pool**. When full, LRU adapter pages are evicted to make room.

```
GPU VRAM:
┌──────────────────────────────┐
│  Base Model Weights (frozen)  │  ← static, never evicted
│                              │
│  Unified Memory Pool         │  ← dynamic
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
│  │KV₀ │ │A₁  │ │KV₁ │ │B₁  │ │  ← interleaved KV and adapter pages
│  └────┘ └────┘ └────┘ └────┘ │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐ │
│  │B₂  │ │KV₂ │ │A₂  │ │A₃  │ │
│  └────┘ └────┘ └────┘ └────┘ │
└──────────────────────────────┘
```

**Why unified?** Separating the pool would require over-provisioning both halves to avoid OOM. A unified pool allows dynamic rebalancing: during heavy prefill, adapters can be evicted; during decode-heavy phases, adapter pages claim more space.

### Adapter Loading and Eviction

S-LoRA maintains three pools:
- **GPU pool:** currently active adapters (pages in VRAM)
- **CPU pool:** recently evicted adapters (pages in host memory)
- **Disk/remote:** full adapter catalog

**Loading sequence for a cold adapter:**
1. Check GPU pool → immediate use
2. Check CPU pool → schedule H2D transfer; add request to waiting queue
3. Check disk/remote → disk read then H2D transfer

**Eviction policy:** LRU across adapter pages (not whole adapters). Individual pages can be evicted, allowing partial adapters to remain in VRAM if only some layers are needed.

### Tensor Parallelism for LoRA

S-LoRA defines the canonical strategy for sharding LoRA weights across GPUs:

**Column-parallel modules** (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`):
- Base weight `W₀` is column-sharded
- LoRA B matrix is column-sharded: each GPU holds corresponding B columns
- LoRA A matrix is **replicated** on all GPUs (A is small: `r × in_dim`)

**Row-parallel modules** (`o_proj`, `down_proj`):
- Base weight `W₀` is row-sharded
- LoRA A matrix is row-sharded: each GPU holds corresponding A rows
- LoRA B matrix is **replicated** on all GPUs

This ensures each GPU can compute its portion of the LoRA delta independently, with the same all-reduce cost as the base model's TP communication. SGLang's LoRA tensor parallelism is directly based on this strategy.

### Benchmarks

On A100 80GB GPU with LLaMA-7B:

| System | Max concurrent adapters | Throughput |
|---|---|---|
| HuggingFace PEFT | 1 | 1× |
| vLLM (naive) | ~16 (VRAM limited) | 1× |
| S-LoRA | **2000+** | **4×** |

The "several orders of magnitude" adapter count comes from moving adapter storage to main memory (CPU DRAM, typically 256GB+) and only loading active adapters to GPU.

### Layer 20 as the Minimal S-LoRA Special Case

| Feature | Layer 20 | S-LoRA |
|---|---|---|
| Adapters in pool | 1 (static) | Thousands |
| Memory pool | Single GPU allocation | Unified Paging (GPU + CPU + disk) |
| Eviction policy | None needed | LRU page eviction |
| Tensor parallelism | Not supported | Column/row shard strategy |
| Adapter loading | At startup | JIT from CPU/disk |
| Kernel | Float mask (dense) | SGMV/BGMV (sparse) |

The next layer after Layer 20 would implement S-LoRA's Unified Paging to support multiple adapters — that is the full multi-LoRA implementation described in `sglang_multi_lora_implementation.md`.

---

## 8. dLoRA: Dynamic Merge/Unmerge and Request Migration

S-LoRA always keeps adapters separate (unmerged). dLoRA (Wu et al., OSDI 2024) shows this is suboptimal for workloads with **skewed adapter distributions**.

### The Merge vs. Unmerge Trade-off

**Merged serving:**
```
W = W₀ + BA · scaling
h = Wx   # single GEMM, zero overhead
```
- Zero overhead: one GEMM instead of two
- Optimal for batches where all requests use the same adapter
- Must merge/unmerge when adapter changes (cost proportional to layers × parameter size)
- Only efficient when one adapter dominates traffic

**Unmerged serving (S-LoRA / SGMV):**
```
h = W₀x + BAx · scaling  # two GEMMs, SGMV for batching
```
- Supports mixed batches: base model + LoRA in same batch
- Supports multiple adapters in same batch
- Always pays SGMV overhead even on "hot" adapters with concentrated traffic

**dLoRA's insight:** In real workloads, adapter request distribution is often highly skewed. Some adapters are "hot" (hundreds of requests/minute), others are "cold" (occasional). For hot adapters, merging and dedicating a worker replica is more efficient than paying SGMV overhead on every request.

### The Credit-Based Batching Algorithm

```python
# Simplified credit-based logic
credits[adapter_id] += 1   # per arriving request
credits[adapter_id] -= decay_rate  # per time unit

if credits[adapter_id] > MERGE_THRESHOLD:
    # High traffic → merge and dedicate a replica
    merge(adapter_id, base_model)
    route_all_requests_for(adapter_id) → merged_replica

elif credits[adapter_id] < UNMERGE_THRESHOLD:
    # Traffic dropped → unmerge, return to shared SGMV serving
    unmerge(adapter_id, base_model)
    route_requests_via_sgmv()
```

This adapts to workload shifts in real time. A credit accumulates when requests arrive and decays when they don't — analogous to a leaky bucket algorithm. The hysteresis (separate MERGE and UNMERGE thresholds) prevents thrashing between modes.

### Request-Adapter Co-Migration

**Problem:** With multiple worker replicas (each holding a copy of the base model), varying output lengths cause replicas to become load-imbalanced over time.

**Solution:** dLoRA implements **request-adapter co-migration** — moving both the in-progress KV cache and adapter weights from an overloaded replica to an underloaded one.

Migration cost:
- KV cache: proportional to sequence length (can be large)
- Adapter weights: small (typically 10–200MB)
- Total: dominated by KV cache for long sequences

dLoRA includes a migration decision function that triggers only when the expected latency reduction exceeds the migration cost, preventing thrashing.

```
                   Request Router
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   [Replica 0]    [Replica 1]    [Replica 2]
   base + A₁,B₁   base (SGMV)   base + A₂,B₂
   (merged)        (unmerged)     (merged)
          │             │             │
          └─────────────┼─────────────┘
                        ▼
                 Co-Migration Manager
```

### Results

| Aspect | S-LoRA | dLoRA |
|---|---|---|
| Serving mode | Always unmerged + SGMV | Dynamic: merge or unmerge based on traffic |
| Load balancing | Static routing | Dynamic request-adapter co-migration |
| Average latency | Higher | **1.8× lower** |
| Throughput vs vLLM | 4× | **57.9×** |

### Relationship to Layer 20

Layer 20 hardcodes "always unmerged" — which is exactly what dLoRA would choose for a single static adapter with mixed base/LoRA traffic. dLoRA generalizes this to N adapters with dynamic switching:

```
Layer 20: lora_mask per token (0.0 or 1.0)  →  static unmerged serving
dLoRA:    credits per adapter               →  dynamic merge decision
```

---

## 9. CaraServe: CPU-Assisted Cold-Start Elimination

S-LoRA and Punica queue requests while adapters are loading from CPU to GPU. For large adapter pools where cold starts are frequent, this queuing creates substantial TTFT overhead.

**CaraServe (Li et al., 2024)** solves the cold-start problem with a two-pronged approach: CPU-assisted prefill and rank-aware scheduling.

### The Cold-Start Problem

Without CaraServe, the timeline for a request with a cold (not in GPU VRAM) adapter is:

```
Time →
|──────────────────|──────────────────|──────────────────|
| Adapter loading  | GPU prefill      | GPU decode       |
| (CPU→GPU xfer)   | (starts AFTER    |                  |
| 100-500ms        |  loading done)   |                  |

TTFT = adapter_load_time + prefill_time  ← cold start penalty
```

For large adapters or slow PCIe bandwidth, adapter loading can dominate TTFT.

### CPU-Assisted Prefill

CaraServe's approach overlaps adapter loading with prefill execution:

```
Time →
|──────────────────|──────────────────|
| Adapter loading  | GPU prefill      |  GPU decode...
| (CPU→GPU xfer)   |                  |
|──────────────────|                  |
| CPU prefill      |──────────────────|
| (concurrent      | switch to GPU    |
|  with load)      |                  |

TTFT = max(adapter_load_time, cpu_prefill_time)  ← reduced
```

The CPU runs prefill on its own compute while the GPU transfers weights. Because CPU prefill is slower (~10× slower than GPU), this overlap is beneficial only when `adapter_load_time > gpu_prefill_time`. CaraServe's profiler determines this threshold per adapter rank and prompt length.

**Implementation challenges and solutions:**
1. **CPU is ~10× slower** → only use CPU prefill when adapter load time exceeds cpu_prefill_time threshold
2. **KV state must transfer CPU→GPU** → pre-allocate pinned (page-locked) memory for fast DMA
3. **Synchronization** → GPU decode starts when both adapter is loaded AND KV prefill data is available

### Rank-Aware Scheduling

LoRA adapters have heterogeneous ranks (r = 4, 8, 16, 64, 128). High-rank adapters require more compute time AND more loading time. A rank-naive scheduler may schedule a rank-4 adapter request while a rank-128 adapter request (with the same SLO deadline) misses its deadline for lack of lead time.

```python
def compute_priority(request):
    adapter_rank = get_rank(request.adapter_id)
    slo_deadline = request.arrival_time + request.ttft_slo
    time_remaining = slo_deadline - current_time
    compute_cost_estimate = estimate_cost(adapter_rank, request.prompt_length)
    urgency = compute_cost_estimate / time_remaining
    return urgency  # higher urgency → scheduled first
```

### Results

On A100 GPU with LLaMA-7B:

| Metric | S-LoRA | CaraServe |
|---|---|---|
| Average latency speedup | 1× | **1.4×** |
| SLO attainment (strict TTFT) | 85% | **99%** |
| Cold-start handling | Request queued | CPU-assisted immediate start |

The 99% SLO attainment for a system serving thousands of adapters (where cold starts are frequent) is the headline result.

### Memory Architecture Comparison

| System | Adapter storage | Cold start handling | KV cache management |
|---|---|---|---|
| Layer 20 | GPU only (1 adapter) | None (no cold starts) | Not paged |
| S-LoRA | CPU→GPU on demand | Request queued during load | Unified Paging |
| CaraServe | CPU→GPU on demand | CPU-assisted prefill | Pinned memory |
| dLoRA | GPU pool (replica per merged) | Migration-aware | Standard paged |

**Layer 20's position:** Zero cold-start problem — the single adapter is loaded at startup and permanently resident in GPU VRAM. CaraServe becomes relevant when extending Layer 20 to a dynamic adapter pool.

---

## 10. Serving LoRA Adapters in SGLang

SGLang's LoRA serving system incorporates both S-LoRA and Punica techniques, providing a production-ready multi-adapter serving stack built directly on the same framework as Layer 20's minimal implementation.

### Enabling LoRA: The Key Arguments

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    --max-loras-per-batch 2 \
    --lora-eviction-policy lru
```

| Argument | What it controls |
|---|---|
| `--enable-lora` | Enable LoRA support (auto-set if `--lora-paths` is provided) |
| `--lora-paths` | Adapters to load at startup; format: `name=path_or_hf_id` |
| `--max-loras-per-batch` | Max adapters active in GPU pool simultaneously. Default: 8 |
| `--max-loaded-loras` | Cap on adapters in CPU memory (≥ `max-loras-per-batch`) |
| `--lora-eviction-policy` | `lru` (default) or `fifo` |
| `--lora-backend` | `triton` or `csgmv` (Chunked SGMV) |
| `--max-lora-rank` | Max rank for GPU buffer reservation (needed for dynamic loading) |
| `--lora-target-modules` | Projection modules to enable LoRA on; `all` for all supported |

The `--max-loras-per-batch` and `--lora-eviction-policy lru` arguments directly implement S-LoRA's Unified Paging with LRU eviction.

### Serving Single vs Multiple Adapters

**Single adapter, per-request selection:**
```python
json_data = {
    "text": [
        "List 3 countries and their capitals.",  # uses lora0
        "List 3 countries and their capitals.",  # uses base model
    ],
    "lora_path": ["lora0", None],  # None → base model
}
```

**Multiple adapters in the same batch:**
```python
json_data = {
    "text": ["List 3 countries.", "Write a Python function."],
    "lora_path": ["lora0", "lora1"],  # different adapter per sequence
}
```

**OpenAI-compatible API:**
```python
# Use adapter: model="base_model_name:adapter_name"
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct:lora0",
    messages=[{"role": "user", "content": "List 3 countries"}]
)
```

### Dynamic Loading and Unloading at Runtime

```python
# Load a new adapter
requests.post(f"http://localhost:{port}/load_lora_adapter", json={
    "lora_name": "new_fine_tune",
    "lora_path": "huggingface_user/my-new-adapter",
})

# Unload an adapter
requests.post(f"http://localhost:{port}/unload_lora_adapter", json={
    "lora_name": "old_adapter",
})
```

**When planning dynamic loading:** specify `--max-lora-rank` and `--lora-target-modules` at launch. Without these, SGLang infers the maximums from `--lora-paths` at startup and may reject later-loaded adapters with larger ranks.

### Memory Architecture: What `max-loras-per-batch` Controls

```
GPU VRAM allocation:
  Base model weights:       fixed (e.g., 16 GB for LLaMA-8B)
  KV cache:                 configurable (largest remaining)
  LoRA GPU pool:            max_loras_per_batch × max_lora_rank × module_dims × 2
```

Layer 20 is the degenerate case: `max_loras_per_batch = 1`, never evicted, loaded once at startup.

### Tensor Parallelism (S-LoRA Strategy)

SGLang follows S-LoRA's tensor parallelism strategy:
- Column-parallel modules: LoRA B column-sharded; LoRA A replicated
- Row-parallel modules: LoRA A row-sharded; LoRA B replicated

Same all-reduce cost as the base model's TP communication — LoRA adds no extra inter-GPU communication.

---

## 11. Serving LoRA Adapters in vLLM

vLLM's multi-LoRA implementation uses a different API design — request-level `LoRARequest` objects for offline inference, and the `model` field for online serving.

### Offline Inference

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)

outputs = llm.generate(
    prompts=["Write a SQL query to answer: ..."],
    sampling_params=SamplingParams(temperature=0, max_tokens=256),
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
)
```

### Online Serving

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

```bash
curl http://localhost:8000/v1/completions \
  -d '{"model": "sql-lora", "prompt": "Write a SQL query...", "max_tokens": 256}'
```

### LoRAResolver Plugins (Fully Automatic Loading)

vLLM supports plugins that automatically resolve and load adapters on demand — no explicit load call needed:

```python
class S3LoRAResolver(LoRAResolver):
    async def resolve_lora(self, base_model_name, lora_name):
        local_path = self.download_from_s3(lora_name)
        return LoRARequest(lora_name=lora_name, lora_path=local_path,
                           lora_int_id=abs(hash(lora_name)))

LoRAResolverRegistry.register_resolver("s3_resolver", S3LoRAResolver())
```

This is the production realisation of LoRAX's "dynamic adapter loading" pillar as an extensible plugin.

### In-Place Adapter Reloading

For Reinforcement Learning pipelines where adapters are updated continuously:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -d '{"lora_name": "rl-adapter", "lora_path": "/path/to/updated/adapter", "load_inplace": true}'
```

`load_inplace=true` replaces weights in-place without restarting the server.

---

## 12. The HuggingFace PEFT Checkpoint Format

Every adapter on HuggingFace Hub — including `phh/Qwen3-0.6B-TLDR-Lora` used in Layer 20 — follows the PEFT checkpoint format.

### Files in a PEFT Adapter Checkpoint

```
adapter_config.json
adapter_model.safetensors   (sometimes adapter_model.bin for older checkpoints)
README.md                   (optional)
```

### `adapter_config.json`

```json
{
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

Key fields:
- `r`: rank (controls adapter size and expressiveness — see § 2 for why `r=8` is typically sufficient)
- `lora_alpha`: the scaling numerator; effective scale = `lora_alpha / r`
- `target_modules`: which projection layers have LoRA applied

### `adapter_model.safetensors` — Weight Tensor Keys

```
base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight  →  [r, in_dim]
base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight  →  [out_dim, r]
base_model.model.model.layers.{layer_idx}.mlp.gate_proj.lora_A.weight     →  [r, in_dim]
...
```

Structure:
- `base_model.model.` — PEFT prefix (always present)
- `model.layers.{layer_idx}.` — architecture path
- `{component}.{module_name}.` — e.g., `self_attn.q_proj.`
- `lora_A.weight` or `lora_B.weight`

`LoRAAdapter._load_weights()` splits each key on `.`, finds `lora_A` or `lora_B`, extracts the layer index and module name, and stores the tensor in a nested dict `{layer_idx: {module_name: tensor}}`.

### Loading for Inference (PEFT)

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
peft_model = PeftModel.from_pretrained(base_model, "phh/Qwen3-0.6B-TLDR-Lora")
peft_model.eval()
outputs = peft_model(**inputs)  # adapter applied automatically
```

This is the ground truth that `verify_lora.py` in Layer 20 compares against.

### Merging for Zero-Overhead Inference

```python
merged_model = peft_model.merge_and_unload()
# Zero inference overhead — one GEMM, not two
# Cannot be unmerged; destroys base model separability
```

Merging is appropriate for a fixed single-adapter deployment. It is inappropriate for multi-adapter serving (Layer 20, LoRAX, SGLang production) because it destroys the base model's separability from the adapter.

### Target Module Coverage Impact on `LoRAAdapter`

| Coverage | Configuration | Layer 20 impact |
|---|---|---|
| Minimal | `["q_proj", "v_proj"]` | `has_layer()` returns False for 5 modules |
| Attention | `["q_proj", "k_proj", "v_proj", "o_proj"]` | 3 MLP modules not applied |
| Full | all 7 | `_load_weights()` handles all 7 key patterns |

`phh/Qwen3-0.6B-TLDR-Lora` uses full coverage — why `_load_weights()` must handle all 7 module types.

---

## 13. Advanced Production Features

### Structured Generation (JSON Mode)

LLMs fine-tuned for extraction need outputs to be structurally valid JSON. LoRAX integrates Outlines to enforce this: during token sampling, structured generation constrains the probability distribution so only tokens that keep the response on a valid JSON parse path have nonzero probability.

| Mode | Content accuracy | Structural validity |
|---|---|---|
| Base model | 50% | 90% |
| LoRA adapter | 71% | 92% |
| Adapter + structured generation | **80%** | **99.9%** |

### Lookahead LoRA: Adapter-Based Speculative Decoding

Standard speculative decoding requires a separate draft model. LoRAX's Lookahead LoRA trains the adapter to simultaneously predict the next token (standard) and predict the following 2–3 tokens (draft). The adapter itself becomes the speculative decoder.

Reported throughput improvement: **2–3× over standard LoRA adapters** — making it valuable for latency-sensitive applications where ITL is critical.

### Loquetier: Unified Fine-Tuning and Serving

Production ML systems must simultaneously train new adapters (fine-tuning on new data) and serve existing adapters (inference on live traffic). Loquetier (NeurIPS 2025) addresses this gap.

**The Virtualized Module abstraction:**

```python
class VirtualizedModule:
    def __init__(self, base_layer):
        self.base_layer = base_layer      # shared, frozen
        self.adapters = {}                # adapter_id → (A, B, mode)

    def forward(self, x, adapter_id=None, mode="serve"):
        base_out = self.base_layer(x)
        if adapter_id is None:
            return base_out
        A, B, _ = self.adapters[adapter_id]
        if mode == "serve":
            return base_out + self._serve_delta(x, A, B)
        else:  # mode == "train"
            return base_out + self._train_delta(x, A, B)  # with gradients
```

**Fused kernel:** A single kernel handles both fine-tuning and inference batches:
```
Mixed batch [train_tokens, serve_tokens]
              ↓
     Single fused kernel:
     - Computes base model output for all tokens
     - For train_tokens: computes LoRA delta + gradient checkpoints
     - For serve_tokens: computes LoRA delta (no gradient tracking)
```

This is Layer 20's float mask concept extended to two token types: training and serving.

**Results (A100 80GB):**

| Setting | Baseline | Loquetier | Improvement |
|---|---|---|---|
| Inference only | SOTA co-serving | Loquetier | **3.0× throughput** |
| Unified (training + serving) | PEFT | Loquetier | **46.4× SLO attainment** |

### Dynamic Adapter Management in vLLM

For RL pipelines where adapters are updated continuously:
1. **LoRAResolver** loads new adapters automatically when their names first appear in requests
2. **In-place reload** (`load_inplace=true`) swaps adapter weights without changing the adapter name

Together: a training loop pushes new checkpoints and the serving layer picks them up without downtime.

---

## 14. The KV Cache vs Adapter VRAM Trade-off

Multi-adapter serving introduces a VRAM trade-off that does not exist for single-model serving: every adapter slot in the GPU pool consumes memory that would otherwise be available for the KV cache.

### The Competition

On an A10G 24 GB GPU serving Mistral-7B:
```
Base model weights: ~14 GB
Available for KV + adapters: ~10 GB
```

With each rank-8 adapter occupying ~50 MB for a 7B model:

| Active adapters | Adapter pool | KV cache budget | Max batch size (approx.) |
|---|---|---|---|
| 1 | ~50 MB | ~9.95 GB | ~100 requests |
| 8 | ~400 MB | ~9.6 GB | ~96 requests |
| 32 | ~1.6 GB | ~8.4 GB | ~84 requests |
| 128 | ~6.4 GB | ~3.6 GB | ~36 requests |

At 128 active adapters, you have 64% of the original KV cache budget.

> "Loading more adapters to GPU for concurrent execution means you will have less DRAM available to store the KV cache... the optimal configuration will be defined by the specific requirements of your workload." — João Moura, 2024

### S-LoRA's Unified Paging Solution

S-LoRA's key insight: rather than pre-allocating fixed pool sizes for adapters and KV separately, treat them as equals in a single paged pool and rebalance dynamically:

- During heavy prefill (lots of KV needed): evict adapter pages
- During decode-heavy phases (KV cache stable): adapter pages claim more space

This reduces fragmentation and over-provisioning compared to a split-pool design.

### The Break-Even Point

- **Skewed access** (80% of requests use 5% of adapters): keep `max_loras_per_batch` at 8–16. Hot adapters are always resident; cold adapters load on demand; large KV cache handles high volume.
- **Uniform access** (each adapter gets equal traffic): larger pool sizes make sense; every adapter has a reasonable chance of being in the next batch.
- **Layer 20's single-adapter case**: the pool always has exactly one adapter. Full remaining VRAM goes to KV cache. No trade-off exists.

---

## 15. Serverless LoRA: Backbone Sharing and Predictive Loading

Serverless deployment models (AWS Lambda with GPUs, Replicate, Modal) are increasingly popular for LLM inference. Two papers address the specific challenges of LoRA in serverless environments.

### ServerlessLoRA: Three Problems, Three Solutions

ServerlessLoRA (Sui et al., 2025) identifies that existing serverless LLM systems fail catastrophically with LoRA due to three specific problems:

**Problem 1: 99% weight redundancy**
```
Function 1: [base_model (14GB)] + [adapter_1 (50MB)]  = 14.05 GB
Function 2: [base_model (14GB)] + [adapter_2 (50MB)]  = 14.05 GB
...
Function N: [base_model (14GB)] + [adapter_N (50MB)]  = 14.05 GB

Total VRAM: N × 14.05 GB  ← completely unnecessary duplication
```

**Problem 2: Artifact loading latency**
Cold start = `base_model_load_time + adapter_load_time`. Both must complete before first token.

**Problem 3: Magnified resource contention**
Multiple LoRA functions on the same GPU → peak memory usage overlaps during prefill → OOM or throttling.

**Solution 1: Secure backbone sharing**
```
               Shared Base Model (14GB)
                        │
            ┌───────────┼───────────┐
            ▼           ▼           ▼
  [Function 1]   [Function 2]   [Function 3]
  adapter_1      adapter_2      adapter_3
  (50MB)         (50MB)         (50MB)
  (isolated)     (isolated)     (isolated)
```

VRAM savings: from `N × 14.05 GB` to `14 GB + N × 0.05 GB ≈ 14.5 GB for N=10`. The key insight shared with Layer 20: **the base model is the expensive artifact; adapters are cheap**.

**Solution 2: Pre-loading**
Adapters are pre-loaded during the function's warm-up phase (before the first request arrives), using access-pattern prediction to identify hot adapters.

**Solution 3: Contention-aware batching and offloading**
During traffic bursts, inactive adapters are offloaded from VRAM to CPU; prefill execution is staggered to reduce peak VRAM pressure.

**Results (A100 80GB, LLaMA-7B, 50 adapters, industrial workload trace):**

| Metric | SOTA serverless | ServerlessLoRA | Improvement |
|---|---|---|---|
| TTFT (mean) | 820ms | **115ms** | **86% reduction** |
| Cost per million tokens | $12.5 | **$1.4** | **89% reduction** |
| Throughput | 1× | **3.2×** | — |

### Predictive-LoRA: LSTM Traffic Prediction + Page-Based Memory

Predictive-LoRA (Tang et al., 2025) addresses two problems ServerlessLoRA didn't fully solve: reactive loading (loading after requests arrive) and fragmentation from variable-rank adapters.

**The LSTM traffic predictor:**

P-LoRA uses a **lightweight LSTM** neural network to predict future adapter demand:
- Input: per-adapter request rates, cumulative counts, time-of-day features, recent access history
- Output: predicted request rate for each adapter in the next time window
- Model size: few hundred parameters (~1ms inference on CPU)
- Trained continuously via online learning

Why LSTM over simpler predictors?

| Predictor | Can model | Limitation |
|---|---|---|
| LRU (recency) | Temporal locality | Cannot predict future demand |
| Frequency (LFU) | Access frequency | Cannot adapt to changing patterns |
| Moving average | Short-term trends | Cannot capture long-term cycles |
| **LSTM** | **Long-range dependencies, cyclical patterns** | Slight training overhead |

The LSTM captures patterns like "adapter_A is popular 9am–5pm on weekdays" — the kind of temporal structure that simpler predictors miss.

**Page-based adapter memory management:**

```python
PAGE_SIZE = 64MB  # fixed, eliminates fragmentation

class AdapterMemoryPool:
    def allocate(self, adapter_id, adapter_size_bytes):
        n_pages = math.ceil(adapter_size_bytes / PAGE_SIZE)
        if len(self.free_pages) < n_pages:
            self.evict_lru_adapter()
        page_ids = [self.free_pages.popleft() for _ in range(n_pages)]
        self.adapter_page_map[adapter_id] = page_ids
```

Fixed-size pages mean any page can be used for any adapter — no fragmentation regardless of rank heterogeneity. P-LoRA maintains >87% VRAM utilization even with rank-4 to rank-128 adapters in the same pool.

**Results (Azure Functions trace):**

| System | TTFT | Throughput | VRAM utilization |
|---|---|---|---|
| S-LoRA | 1× (baseline) | 1× | 60–70% |
| P-LoRA | **0.65× (35% lower)** | **1.52×** | **87%+** |

---

## 16. Loquetier: Unified Fine-Tuning and Serving

Prior multi-LoRA work (Punica, S-LoRA, dLoRA) focused exclusively on inference serving. Real production deployments also need to fine-tune new adapters on fresh user data — continuously — while serving existing adapters to live users without interruption.

Running both workloads creates resource contention:

```
GPU Resources:
┌─────────────────────────────────────────────────┐
│  Training: forward + backward + optimizer step  │  ← large memory
│  Serving: prefill + decode × multiple adapters  │  ← time-sensitive
└─────────────────────────────────────────────────┘
         ↑ both compete for VRAM and compute
```

**Loquetier (NeurIPS 2025)** addresses this gap with two components:

### The Virtualized Module (VM)

The VM wraps each LoRA-targeted base layer, supporting multiple adapters in both training and serving mode simultaneously:

- Base layer is **never duplicated** regardless of how many adapters are active
- Each adapter has its own `(A, B)` pair with independent gradient state
- Training and serving adapters can coexist in the pool
- The VM separates PEFT bookkeeping from the underlying base layer computation

### Fused Kernel for Training + Serving

**Naive approach (prior systems):**
```
Training batch  → separate kernel → loss + gradients
Serving batch   → separate kernel → output logits
```
Two kernel launches per forward pass, redundant base model computation.

**Loquetier's approach:**
```
Mixed batch [train_tokens, serve_tokens]
              ↓
     Single fused kernel handles both:
     - train_tokens: LoRA delta + gradient checkpoints
     - serve_tokens: LoRA delta only (no gradient tracking)
     - Masking per token type (conceptually like Layer 20's lora_mask, extended to two modes)
```

**Evaluation (A100 80GB):**

| Setting | Baseline | Loquetier | Improvement |
|---|---|---|---|
| Inference only | SOTA co-serving | Loquetier | **3.0× throughput** |
| Training only | PEFT | Loquetier | ~1.5× throughput |
| Unified (training + serving) | PEFT | Loquetier | **46.4× SLO attainment** |

Loquetier represents the direction Layer 20 would evolve toward in an active learning pipeline — where new adapters are trained continuously as users interact with the system.

---

## 17. InfiniLoRA: Disaggregated LoRA for MoE Models

InfiniLoRA (Chen et al., April 2026) is the most architecturally ambitious paper in the survey. It addresses a problem that all prior work ignored: **Mixture-of-Experts (MoE) models dramatically increase LoRA memory cost**, making existing coupled LoRA serving designs poorly scalable.

### Why MoE Changes Everything

**Dense models (e.g., LLaMA-7B, Qwen3-0.6B):**
- LoRA adds `r × d_model` parameters per targeted layer
- For LLaMA-7B with rank-8: ~4M extra parameters (0.06% of model)
- LoRA memory is negligible relative to base model

**MoE models (e.g., DeepSeek-V2, Mixtral, Qwen3-MoE):**
- Each token routes to a subset of "expert" FFN layers
- There are many experts (e.g., 64 experts, 2–4 activated per token)
- LoRA applied to all expert layers: `n_experts × r × d_expert` parameters
- For Mixtral-8×7B with rank-8: ~50M extra parameters per layer (significant!)

**LoRA memory scales with number of experts.** With 64 experts and 32 layers, LoRA weights for a single adapter can be comparable to the base model itself. The assumption of "adapters are small" breaks down completely for MoE.

### The Coupled LoRA Problem (All Prior Work)

In all prior systems (S-LoRA, Punica, dLoRA), LoRA runs on the same GPU as the base model:

```
Base Model GPU:
┌─────────────────────────────────┐
│ Base model weights (frozen)     │  ← dominant VRAM
│ KV cache                        │  ← grows with batch
│ LoRA A/B weights (per adapter)  │  ← small for dense, LARGE for MoE
│ LoRA computation (SGMV)         │  ← overlaps with base model FLOPs
└─────────────────────────────────┘
```

For MoE: LoRA weights compete with KV cache → reduced batch size → lower throughput. LoRA compute delays base model decode → tail latency inflation.

### InfiniLoRA's Disaggregated Architecture

InfiniLoRA separates LoRA execution onto a dedicated **LoRA Server**:

```
                    Incoming Requests
                           │
                    ┌──────▼──────┐
                    │   Scheduler  │
                    └──────┬──────┘
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    [Base Model GPU 0]  [Base GPU 1]  [Base GPU 2]
    (prefill + decode)  (...)         (...)
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │ LoRA deltas requested
                             ▼
                   [LoRA Server Cluster]
                   ┌─────────────────────────────┐
                   │  LoRA GPU 0  │  LoRA GPU 1   │
                   │  adapter_1   │  adapter_2    │
                   │  adapter_3   │  ...          │
                   └─────────────────────────────┘
```

The base model GPUs send intermediate activations to the LoRA Server; it computes and returns the deltas.

### Critical-Path Optimizations

**1. GPU-Initiated Communication:**
Traditional systems use the CPU to orchestrate GPU-to-GPU transfers. InfiniLoRA uses NCCL GPU-initiated RDMA: base model GPU sends activations and receives deltas directly, bypassing the CPU entirely. Reduces communication overhead from O(100μs) to O(10μs).

**2. Parallelism-Aware LoRA Execution:**
The LoRA Server is TP-aware: if the base model uses 4-way TP, the LoRA Server aggregates 4 activation shards, computes the delta, and sends back — matching the base model's parallelism degree.

**3. Hardware-Specialized LoRA Kernels:**
- A matrix multiply (`x @ A.T`, down-projection): bandwidth-bound → HBM-optimized hardware
- B matrix multiply (`h @ B.T`, up-projection): compute-bound → tensor core-optimized hardware

InfiniLoRA routes different parts of the LoRA computation to the appropriate hardware within the LoRA Server cluster.

### SLO-Driven Provisioning

An offline capacity planner takes:
- Expected request rate per adapter
- SLO requirement (P99 TTFT)
- Hardware costs (GPU hours)

And outputs how many LoRA Server GPUs to provision, how to distribute adapters, and when to scale up/down. This enables elastic scaling of LoRA capacity independently of base model capacity.

### Results

On H100 cluster with DeepSeek-V2 (MoE model):

| Metric | S-LoRA (coupled) | InfiniLoRA (disaggregated) |
|---|---|---|
| Serviceable request rate | 1× | **3.05×** |
| SLO attainment | 46% | **100%** |
| VRAM for base model GPUs | 90% occupied | 64% occupied (more KV cache) |

The 3.05× throughput comes from: (1) base model GPUs no longer constrained by LoRA VRAM → larger KV cache → larger batch, (2) LoRA computation overlaps with base model decode via async communication.

### The Disaggregation Evolution

InfiniLoRA extends PD Disaggregation (Layer 19) to the LoRA dimension:

| Disaggregated component | Paper |
|---|---|
| Prefill vs. Decode (separate GPU pools) | DistServe (OSDI 2024), Mooncake |
| KV cache vs. Compute | Mooncake (FAST 2025) |
| **LoRA vs. Base model** | **InfiniLoRA (2026)** |

The trend: specialized compute + memory components for each part of the inference pipeline.

**Layer 20 position:** The simplest possible "coupled" design. InfiniLoRA is the state-of-the-art "disaggregated" design. The progression from Layer 20 → full production LoRA serving → InfiniLoRA mirrors the progression from simple batch inference → PD disaggregation in the base model layer.

---

## 18. Decision Framework: When Does Multi-LoRA Serving Pay Off?

Combining insights from all papers surveyed:

### The 5-Check Framework

**Check 1 — How many adapters do you actually need?**

If fewer than 4 adapters with predictable access patterns, consider separate dedicated servers with merged weights. Multi-adapter infrastructure pays off at 8+ adapters; it becomes clearly worthwhile at 50+.

**Check 2 — What is your adapter access distribution?**

Collect a day of production traffic and plot adapter request frequency:
- **Highly skewed** (top 10 adapters handle 95%): use dLoRA's insight — merge the hot adapters into dedicated replicas, serve the long tail via SGMV
- **Uniform**: larger GPU pool with S-LoRA-style Unified Paging
- **Unknown distribution**: start with LRU eviction at `max_loras_per_batch=8`

**Check 3 — What is your KV cache budget?**

```python
kv_per_token = n_layers × n_kv_heads × head_dim × 2 × dtype_bytes
kv_per_request = kv_per_token × avg_sequence_length
kv_budget = gpu_vram - model_weights - adapter_pool_size
max_concurrent_requests ≈ kv_budget / kv_per_request
```

If adapter pool reduces `max_concurrent_requests` below target, reduce `max_loras_per_batch` or provision a larger GPU.

**Check 4 — Do you need dynamic loading or can you pre-load at startup?**

- Pre-loading (like Layer 20, extended to N via `--lora-paths`): works when catalog is fixed and fits in CPU DRAM
- Dynamic loading: needed when adapters are created continuously, or catalog is too large for full pre-loading
- Predictive pre-loading (P-LoRA): optimal for serverless with predictable access patterns

**Check 5 — Are you using a MoE base model?**

If yes, standard coupled LoRA designs (S-LoRA, dLoRA) degrade because LoRA weights scale with the number of experts. Consider InfiniLoRA-style disaggregation where LoRA Server scales independently.

### The Hardware Decision Table

| Scenario | Recommendation |
|---|---|
| 1 adapter, static | Merge into base model; zero overhead; standard serving |
| 2–4 adapters, stable | Pre-load at startup with `--lora-paths`; minimal pool |
| 5–100 adapters, stable catalog | SGLang or vLLM with `max_loras_per_batch=8–16`; LRU |
| 100+ adapters, dynamic catalog | Dynamic loading via REST API or LoRAResolver; tier to CPU/disk |
| Skewed access (hot adapters) | dLoRA-style merge for hot adapters; SGMV for cold |
| RL training pipeline | vLLM with in-place reload (`load_inplace=true`) |
| Serverless / per-request isolation | ServerlessLoRA backbone sharing; P-LoRA predictive loading |
| MoE base model | InfiniLoRA disaggregated LoRA server |
| Training + serving simultaneously | Loquetier Virtualized Module + fused kernel |

### The Economic Case

For 1,000 customer-specific fine-tunes of a 7B model:

| Architecture | GPU instances | Monthly cost (H100 at $3/hr) |
|---|---|---|
| Dedicated per adapter | 1,000 | ~$2.16M/month |
| Multi-LoRA (10 servers, 100 concurrent/server) | 10–20 | ~$22K–$43K/month |
| Savings | — | **~98% cost reduction** |

The savings assume 24/7 serving with comparable request rates. Real workloads have uneven traffic; actual savings come from not provisioning for peak on each adapter independently.

> "The transition from 'One Model per GPU' to '1000 Adapters per GPU' is more than just a hardware optimisation; it is a fundamental shift in how we architect AI systems." — Neel Shah (Towards AI), 2026

---

## Key Quotes

> "Fine-tuning large language models is easier than ever. Serving them efficiently? That's where LoRAX steps in." — Sai Mudhiganti, July 2025

> "Despite the allure of batching without merging, it is not always beneficial to unmerge, especially when the types of requests are skewed." — dLoRA (OSDI 2024)

> "Punica achieves 12× higher throughput in serving multiple LoRA models compared to state-of-the-art LLM serving systems while only adding 2ms latency per token." — Punica (MLSys 2024)

> "S-LoRA can improve the throughput by up to 4 times and increase the number of served adapters by several orders of magnitude." — S-LoRA (MLSys 2024)

> "CaraServe can speed up the average request serving latency by up to 1.4× and achieve SLO attainment of up to 99%." — CaraServe (2024)

> "InfiniLoRA achieves an average 3.05× increase in serviceable request rate under strict latency SLOs." — InfiniLoRA (2026)

> "The transition from 'One Model per GPU' to '1000 Adapters per GPU' is more than just a hardware optimisation; it is a fundamental shift in how we architect AI systems." — Neel Shah (Towards AI), March 2026

---

## What Sits in `OMITTEDL3L4.md`

The following material was deliberately excluded to keep this article focused. Full content is in `OMITTEDL3L4.md`.

### Punica SGMV CUDA thread block assignment
Section 4 describes what SGMV does; the Punica paper contains the actual CUDA implementation: thread block-to-segment mapping, gather operations for non-contiguous adapter pages, handling variable-rank adapters in the same kernel launch.

### S-LoRA Unified Paging formal analysis
Section 7 describes Unified Paging conceptually; S-LoRA provides the formal analysis: proof that Unified Paging eliminates fragmentation, the queuing model for adapter loading latency, tensor parallelism sharding derivation.

### dLoRA convergence analysis
Section 8 describes the credit algorithm; the paper provides convergence analysis for the hysteresis thresholds (MERGE_THRESHOLD, UNMERGE_THRESHOLD) and the formal migration cost model.

### CaraServe CPU profiler threshold derivation
Section 9 describes the CPU-assisted prefill concept; the paper provides the profiler that determines per-adapter thresholds as a function of adapter rank and prompt length.

### Predictive-LoRA LSTM training loop
Section 15 describes the LSTM predictor; the paper provides the online learning update rule, the feature normalization scheme, and the hyperparameter sensitivity study.

### InfiniLoRA SLO provisioning solver
Section 17 describes the capacity planner conceptually; the paper provides the optimization problem formulation and the solver that handles multi-adapter, multi-SLO provisioning under hardware cost constraints.

### DoRA, rsLoRA, PiSSA, EVA variant derivations
These LoRA variants (referenced in `OMITTEDL1L2.md`) involve mathematical modifications to the core LoRA formula. Full derivations belong to L3/L4 treatment but are separate from the serving systems focus of this article.
