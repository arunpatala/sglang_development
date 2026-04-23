# InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models

**Source:** https://arxiv.org/abs/2604.07173
**Paper PDF:** https://arxiv.org/pdf/2604.07173
**Authors:** Hongyu Chen and others
**Submitted:** April 8, 2026
**Level:** L4 — Cutting-edge research; disaggregated LoRA execution for MoE and large model architectures
**Why here:** InfiniLoRA represents the frontier of multi-LoRA serving research (April 2026). It addresses a problem that all prior work ignored: **Mixture-of-Experts (MoE) models dramatically increase LoRA memory cost**, making existing coupled LoRA serving designs (where LoRA runs on the same GPU as the base model) poorly scalable. InfiniLoRA decouples LoRA execution entirely into a separate LoRA Server, enabling disaggregated scaling. This is the most sophisticated architecture in the survey, and represents the direction that production systems are heading as MoE models (DeepSeek-V2, Qwen3-MoE, Mixtral) become dominant.

**BibTeX:**
```bibtex
@article{chen2026infinilora,
  title  = {{InfiniLoRA}: Disaggregated Multi-{LoRA} Serving for Large Language Models},
  author = {Hongyu Chen and others},
  journal = {arXiv preprint arXiv:2604.07173},
  year   = {2026},
  url    = {https://arxiv.org/abs/2604.07173}
}
```

---

## Abstract

LoRA enables efficient customization of LLMs and is widely used in multi-tenant and multi-task serving. However, **emerging model architectures such as MoE significantly increase LoRA memory cost**, making existing coupled LoRA serving designs poorly scalable and prone to tail-latency inflation. We present InfiniLoRA, a disaggregated LoRA serving system that decouples LoRA execution from base-model inference.

InfiniLoRA introduces a **shared LoRA Server** with:
- Parallelism-aware execution
- SLO-driven provisioning
- Critical-path optimizations: GPU-initiated communication + hardware-specialized LoRA kernels

Results: InfiniLoRA achieves an average **3.05× increase in serviceable request rate** under strict latency SLOs, and improves the percentage of LoRA adapters satisfying the SLO requirement by **54.0%**.

---

## Why MoE Changes Everything

### Dense models (e.g., LLaMA-7B, Qwen3-0.6B)

- Each token passes through all layers sequentially
- LoRA adds `r × d_model` parameters per targeted layer
- For LLaMA-7B with rank-8: ~4M extra parameters (0.06% of model)
- LoRA memory is negligible relative to base model

### MoE models (e.g., DeepSeek-V2, Mixtral, Qwen3-MoE)

- Each token is routed to a subset of "expert" FFN layers
- There are many experts (e.g., 64 experts, only 2-4 activated per token)
- LoRA applied to all expert layers: `n_experts × r × d_expert` parameters
- For Mixtral-8×7B with rank-8: ~50M extra parameters per layer (significant!)

**LoRA memory scales with number of experts**, not just layer count. With 64 experts and 32 layers, LoRA weights for a single adapter can be comparable to the base model itself.

---

## The Coupled LoRA Problem

In all prior systems (S-LoRA, Punica, dLoRA), LoRA computation runs **on the same GPU as the base model inference**:

```
Base Model GPU:
┌─────────────────────────────────┐
│ Base model weights (frozen)     │  ← dominant VRAM usage
│ KV cache                        │  ← grows with batch
│ LoRA A/B weights (per adapter)  │  ← small for dense, LARGE for MoE
│ LoRA computation (SGMV)         │  ← overlaps with base model FLOPs
└─────────────────────────────────┘
```

For MoE models:
- LoRA weights compete with KV cache for VRAM
- Reducing max batch size to fit LoRA → lower throughput
- Tail latency inflates when LoRA compute delays base model decode

---

## InfiniLoRA's Disaggregated Architecture

InfiniLoRA separates LoRA execution onto a dedicated **LoRA Server**:

```
                    Incoming Requests
                           │
                    ┌──────▼──────┐
                    │   Scheduler  │
                    └──────┬──────┘
                           │ routes requests
              ┌────────────┼────────────┐
              ▼            ▼            ▼
    [Base Model GPU 0]  [Base Model GPU 1]  [Base Model GPU 2]
    (prefill + decode)  (prefill + decode)  (prefill + decode)
         │                   │                   │
         └───────────────────┼───────────────────┘
                             │  LoRA deltas requested
                             ▼
                   [LoRA Server Cluster]
                   ┌─────────────────────────────┐
                   │  LoRA GPU 0  │  LoRA GPU 1   │
                   │  adapter_1   │  adapter_2    │
                   │  adapter_3   │  adapter_4    │
                   │  ...         │  ...          │
                   └─────────────────────────────┘
```

The LoRA Server holds all adapter A/B weights. When the base model GPUs need LoRA deltas, they send the intermediate activations to the LoRA Server and receive back the deltas.

---

## Critical-Path Optimizations

### 1. GPU-Initiated Communication

In traditional disaggregated systems, the CPU orchestrates GPU-to-GPU communication (activation transfer to LoRA Server, delta transfer back). This introduces CPU overhead on the critical path.

InfiniLoRA uses **NCCL GPU-initiated communication**: the base model GPU sends activations and receives deltas directly via GPU-to-GPU RDMA, bypassing the CPU entirely. This reduces the communication overhead from O(100μs) to O(10μs).

### 2. Parallelism-Aware LoRA Execution

The LoRA Server uses tensor parallelism internally (analogous to S-LoRA's TP strategy) and is aware of the base model's tensor parallel degree:

- If the base model uses 4-way TP, the base model GPUs send 4 shards of activations
- The LoRA Server aggregates the shards, computes the delta, and sends back
- Alternatively, the LoRA Server can be distributed with matching TP degree

### 3. Hardware-Specialized LoRA Kernels

Different operations have different optimal hardware:
- **A matrix multiply** (`x @ A.T`, down-projection to rank-r): bandwidth-bound, better on HBM-optimized hardware
- **B matrix multiply** (`h @ B.T`, up-projection from rank-r to d_model): compute-bound, better on tensor core-optimized hardware

InfiniLoRA routes different parts of the LoRA computation to the appropriate hardware within the LoRA Server cluster.

---

## SLO-Driven Provisioning

InfiniLoRA includes an offline capacity planner:

Given:
- Expected request rate per adapter
- SLO requirement (P99 TTFT)
- Hardware costs (GPU hours)

The planner outputs:
- How many LoRA Server GPUs to provision
- How to distribute adapters across LoRA Server GPUs
- When to scale the LoRA Server up/down

This enables elastic scaling of the LoRA serving capacity independently of the base model capacity.

---

## Evaluation

On H100 cluster with DeepSeek-V2 (MoE model):

| Metric | S-LoRA (coupled) | InfiniLoRA (disaggregated) |
|---|---|---|
| Serviceable request rate | 1× | **3.05×** |
| SLO attainment | 46% | **100%** |
| VRAM for base model GPUs | 90% occupied | 64% occupied (more KV cache) |

The 3.05× throughput improvement comes from:
1. Base model GPUs no longer constrained by LoRA VRAM
2. Larger KV cache → larger batch size → better GPU utilization
3. LoRA computation overlaps with base model decode via async communication

---

## The Disaggregation Evolution

InfiniLoRA represents the natural extension of PD Disaggregation (Layer 19) to the LoRA serving dimension:

| Dimension | Disaggregated component | Paper |
|---|---|---|
| Prefill vs. Decode | Separate GPU pools | DistServe (OSDI 2024), Mooncake |
| LoRA vs. Base model | Separate LoRA Server | **InfiniLoRA (2026)** |
| KV cache vs. Compute | Separate KV cache server | Mooncake (FAST 2025) |

The trend is clear: specialized compute + memory components for each part of the inference pipeline.

---

## Relevance to Layer 20

InfiniLoRA is the most architecturally distant from Layer 20, but represents the direction production systems are evolving toward:

| Feature | Layer 20 | InfiniLoRA |
|---|---|---|
| LoRA placement | Same GPU as base model | Dedicated LoRA Server cluster |
| Communication | None (shared memory) | GPU-initiated RDMA |
| Scaling | Cannot scale | LoRA cluster scales independently |
| MoE support | Not tested | Core design target |
| Parallelism | None | TP-aware LoRA sharding |

Layer 20 is the simplest possible version of the "coupled" design. InfiniLoRA is the state-of-the-art "disaggregated" design. The progression from Layer 20 → full production LoRA serving → InfiniLoRA-style disaggregation mirrors the progression from simple batch inference → PD disaggregation in the base model layer.
