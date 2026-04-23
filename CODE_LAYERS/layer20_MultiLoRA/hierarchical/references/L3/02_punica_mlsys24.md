# Punica: Multi-Tenant LoRA Serving

**Source:** https://arxiv.org/abs/2310.18547
**Paper PDF:** https://arxiv.org/pdf/2310.18547
**MLSys page:** https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html
**Code:** https://github.com/punica-ai/punica
**Authors:** Lequn Chen, Zihao Ye (University of Washington); Yongji Wu, Danyang Zhuo (Duke University); Luis Ceze, Arvind Krishnamurthy (University of Washington)
**Submitted:** October 28, 2023
**Venue:** MLSys 2024
**Level:** L3 — Foundational systems paper; defines the SGMV kernel powering all production multi-LoRA systems
**Why here:** Punica introduced the **Segmented Gather Matrix-Vector Multiplication (SGMV)** kernel, which is the core computational primitive for efficient multi-LoRA serving. It is cited and used by SGLang, vLLM, and LoRAX. Understanding Punica is essential for understanding why Layer 20's float-mask approach is less efficient than production systems, and what the production alternative looks like.

**BibTeX:**
```bibtex
@inproceedings{chen2024punica,
  title  = {Punica: Multi-Tenant {LoRA} Serving},
  author = {Lequn Chen and Zihao Ye and Yongji Wu and Danyang Zhuo
            and Luis Ceze and Arvind Krishnamurthy},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year   = {2024},
  url    = {https://arxiv.org/abs/2310.18547}
}
```

---

## Abstract

Low-rank adaptation (LoRA) has become an important and popular method to adapt pre-trained models to specific domains. We present Punica, a system to serve multiple LoRA models in a shared GPU cluster. Punica contains a new CUDA kernel design that allows batching of GPU operations for different LoRA models. This allows a GPU to hold only a single copy of the underlying pre-trained model when serving multiple, different LoRA models, significantly enhancing GPU efficiency in terms of both memory and computation. Our scheduler consolidates multi-tenant LoRA serving workloads in a shared GPU cluster. With a fixed-sized GPU cluster, our evaluations show that **Punica achieves 12× higher throughput** in serving multiple LoRA models compared to state-of-the-art LLM serving systems while only adding **2ms latency per token**.

---

## The Core Problem: Batching Across Different Adapters

For a batch with tokens using different adapters:
- Token 0: adapter A₁, B₁
- Token 1: adapter A₂, B₂  
- Token 2: adapter A₁, B₁ (same as token 0)

The base model forward pass is trivially batched: `X · W₀ᵀ` where `X = [x₀, x₁, x₂]`.

The LoRA delta, however, must be applied per-adapter:
```
delta₀ = x₀ · A₁ᵀ · B₁ᵀ · scaling₁
delta₁ = x₁ · A₂ᵀ · B₂ᵀ · scaling₂
delta₂ = x₂ · A₁ᵀ · B₁ᵀ · scaling₁
```

A naive implementation launches a separate GEMM kernel for each group of tokens sharing an adapter. Each kernel invocation has overhead (memory transfers, GPU scheduling). With many small groups (diverse adapters), GPU utilization collapses.

---

## The SGMV Kernel

**Segmented Gather Matrix-Vector Multiplication (SGMV)** computes the LoRA delta for a heterogeneous batch in a single fused kernel.

### Algorithm intuition

1. **Gather:** For each token `i` in the batch, identify which adapter weight matrices `(Aₙ, Bₙ)` to use.
2. **Segment:** Group tokens by adapter to maximize data locality.
3. **Multiply:** Execute the down-projection `x @ Aₙᵀ` and up-projection `h @ Bₙᵀ` in a single pass using custom CUDA thread block scheduling.

### Key CUDA implementation insight

Standard cuBLAS GEMMs are optimized for large batches of identical operations. SGMV uses a custom kernel that:
- Assigns CUDA thread blocks to segments (groups of tokens sharing an adapter)
- Within each segment, executes standard GEMM operations
- Handles variable-length segments gracefully
- Avoids per-kernel-launch overhead by fusing all segments into one launch

### Complexity

| Approach | Kernel launches | Data transfers | GPU utilization |
|---|---|---|---|
| Naive (per-adapter loop) | O(num_adapters) | redundant loads | poor for many adapters |
| SGMV | 1 | optimized | near-optimal |

---

## Memory Architecture

Punica stores:
- **Base model:** single copy in GPU VRAM (shared across all adapters)
- **All adapter A and B matrices:** loaded into a contiguous GPU buffer

The buffer grows linearly with the number of served adapters. Punica does not implement eviction — for large adapter pools, S-LoRA's Unified Paging (see L3/03) extends Punica with dynamic eviction.

---

## Scheduler Design

Punica's scheduler manages the mapping of incoming requests to GPU execution:

1. **Request arrives** with an `adapter_id`
2. **Check GPU buffer:** if adapter weights are present, immediately ready
3. **If not present:** fetch from CPU/disk, add to GPU buffer
4. **Batch:** collect requests across different adapters into a single batch
5. **Execute:** one SGMV kernel invocation computes deltas for all adapters simultaneously

The scheduler also handles the cluster level: distributing requests across multiple GPUs, balancing load, and managing inter-GPU communication for adapter weight distribution.

---

## Benchmarks

On a single A100 80GB GPU with LLaMA-7B:

| System | Throughput (requests/sec) | Latency overhead |
|---|---|---|
| vLLM (naive multi-model) | 1× (baseline) | — |
| Punica | **12×** | +2ms/token |

The 12× throughput is achieved because:
- Base model GEMM is computed only once per batch (not per-adapter)
- SGMV kernel handles all adapter deltas in one launch
- Latency overhead (+2ms/token) is nearly constant regardless of number of adapters

---

## BGMV vs SGMV

Later work (including vLLM) introduced **BGMV (Batched Gather Matrix-Vector Multiplication)**, optimized for the decode phase:

| Kernel | Optimized for | Used in |
|---|---|---|
| SGMV | Prefill (long sequences, batch of tokens) | LoRAX, SGLang Triton backend |
| BGMV | Decode (single token per request) | vLLM |

Layer 20 uses neither — it uses a **float mask approach** which is simpler but less efficient:

```python
# Layer 20 approach
out = base_out + delta * lora_mask  # always computes delta even for base-only tokens

# SGMV approach
# only computes delta for tokens that actually need it
# groups tokens by adapter to maximize memory bandwidth utilization
```

---

## Adoption

Punica's SGMV kernel has been adopted by:
- **LoRAX** (Predibase) — the paper is cited in their blog post and code
- **SGLang** — `lora_backend=triton` and `lora_backend=csgmv` both build on SGMV concepts
- **vLLM** — `punica_wrapper` module in `vllm/lora/ops/`

---

## Relevance to Layer 20

Punica defines the production alternative to Layer 20's masked approach:

| Layer 20 | Punica Production |
|---|---|
| `delta = (x @ A.T) @ B.T * scaling` for all tokens | SGMV: only for tokens with a LoRA adapter |
| `out = base_out + delta * lora_mask` | `out = base_out + sgmv_delta` |
| 2 extra matmuls per layer, per token | 2 extra matmuls only for LoRA tokens |
| O(n_tokens × r × d) extra FLOPs | O(n_lora_tokens × r × d) extra FLOPs |
| Simple Python code, ~10 lines | Custom CUDA kernel, ~500 lines |

For a batch of 100 tokens where 10 need a LoRA adapter, the mask approach wastes 90% of the extra FLOPs on zero-multiplied results. SGMV skips the computation entirely for base-model tokens.
