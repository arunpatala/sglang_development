# S-LoRA: Serving Thousands of Concurrent LoRA Adapters

**Source:** https://arxiv.org/abs/2311.03285
**Paper PDF:** https://arxiv.org/pdf/2311.03285
**Code:** https://github.com/S-LoRA/S-LoRA
**Authors:** Ying Sheng, Shiyi Lin, Joseph E. Gonzalez, Ion Stoica, Lianmin Zheng (UC Berkeley)
**Submitted:** November 6, 2023 (v1), revised June 5, 2024 (v3)
**Venue:** MLSys 2024
**Level:** L3 — Foundational systems paper; defines Unified Paging and tensor parallel LoRA
**Why here:** S-LoRA extends Punica's computational approach with a memory management system that enables *thousands* of concurrent adapters (vs Punica's small fixed set). S-LoRA's Unified Paging is the direct ancestor of SGLang's `lora_eviction_policy` and `max_loras_per_batch` arguments. Understanding S-LoRA is essential for understanding the memory management gap between Layer 20 (zero management) and SGLang production (full dynamic pool with eviction).

**BibTeX:**
```bibtex
@inproceedings{sheng2024slora,
  title  = {S-{LoRA}: Serving Thousands of Concurrent {LoRA} Adapters},
  author = {Ying Sheng and Shiyi Lin and Joseph E. Gonzalez and Ion Stoica and Lianmin Zheng},
  booktitle = {Proceedings of Machine Learning and Systems (MLSys)},
  year   = {2024},
  url    = {https://arxiv.org/abs/2311.03285}
}
```

---

## Abstract

The "pretrain-then-finetune" paradigm is commonly adopted in the deployment of large language models. Low-Rank Adaptation (LoRA) is a popular approach that adapts a base model to multiple tasks by adding lightweight trainable adapters. We present S-LoRA, a system designed for the scalable serving of many LoRA adapters. S-LoRA stores all adapters in the main memory and fetches the adapters used by the currently running queries to the GPU memory. To efficiently use the GPU memory and reduce fragmentation, S-LoRA proposes **Unified Paging**. Unified Paging uses a unified memory pool to manage dynamic adapter weights with different ranks and KV cache tensors with varying sequence lengths. Additionally, S-LoRA employs a novel tensor parallelism strategy and highly optimized custom CUDA kernels for heterogeneous batching of LoRA computation. Collectively, these features enable S-LoRA to serve thousands of LoRA adapters on a single GPU or across multiple GPUs with a small overhead. Compared to state-of-the-art libraries such as HuggingFace PEFT and vLLM (with naive support of LoRA serving), **S-LoRA can improve the throughput by up to 4 times and increase the number of served adapters by several orders of magnitude**.

---

## The Scalability Problem Punica Didn't Solve

Punica demonstrated that multi-LoRA serving is possible without throughput degradation. However, Punica assumed all adapters fit in GPU VRAM simultaneously. For large adapter pools (thousands of adapters), this is infeasible.

**S-LoRA addresses:**
1. **Memory fragmentation** — adapters have different ranks (and thus different sizes); naively allocating each creates fragmentation
2. **KV cache contention** — adapters and KV cache compete for the same GPU VRAM; need unified management
3. **Cold start** — loading an adapter from CPU to GPU takes time; need careful scheduling to hide latency

---

## Unified Paging

The central innovation of S-LoRA is treating adapter weights and KV cache entries as **first-class citizens in a shared paged memory pool**.

### Analogy to PagedAttention (vLLM)

vLLM's PagedAttention manages KV cache entries in fixed-size pages, eliminating fragmentation from variable-length sequences. S-LoRA extends this idea:

- **KV cache pages:** each page holds `page_size × num_heads × head_dim` elements
- **Adapter pages:** each page holds a fixed-size chunk of adapter A or B matrix

Both types of pages live in the **same memory pool**. When the pool is full, S-LoRA evicts the least-recently-used (LRU) adapter pages to make room for new KV cache entries or new adapters.

### Why unified?

Separating the pool would require over-provisioning both halves to avoid OOM in worst-case scenarios. A unified pool allows dynamic rebalancing: during heavy prefill (lots of KV cache needed), adapters can be evicted; during decode-heavy phases (KV cache stable, many new adapters needed), adapter pages can claim more space.

---

## Memory Layout

For a batch using `k` different adapters with ranks `r₁, r₂, ..., rₖ`:

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

Adapter pages are **non-contiguous** in VRAM. The SGMV/BGMV kernels are modified to support non-contiguous adapter weight access via gather operations.

---

## Adapter Loading and Eviction

S-LoRA's scheduler maintains:
- **GPU pool:** currently active adapters (pages in VRAM)
- **CPU pool:** recently evicted adapters (pages in host memory)
- **Disk/remote:** full adapter catalog

**Loading sequence for an adapter on a new request:**
1. Check GPU pool → if present, use immediately
2. Check CPU pool → if present, schedule H2D transfer; add request to waiting queue
3. Check disk/remote → if present, schedule disk read then H2D transfer

**Eviction policy:** LRU across adapter pages (not whole adapters — individual pages can be evicted, allowing partial adapters to remain in VRAM if only some layers are needed).

---

## Tensor Parallelism for LoRA

S-LoRA defines how to shard LoRA weights across multiple GPUs:

### Column-parallel modules (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`)

- Base weight `W₀` is column-sharded: each GPU holds columns `[col_start:col_end]`
- LoRA B matrix is column-sharded: each GPU holds corresponding B columns
- LoRA A matrix is replicated on all GPUs (A is typically small: `r × in_dim`)

### Row-parallel modules (`o_proj`, `down_proj`)

- Base weight `W₀` is row-sharded: each GPU holds rows `[row_start:row_end]`
- LoRA A matrix is row-sharded: each GPU holds corresponding A rows
- LoRA B matrix is replicated on all GPUs

This ensures that each GPU can compute its portion of the LoRA delta independently, with a single all-reduce for the A matrix contribution in row-parallel modules (same cost as the base model's all-reduce).

This strategy is now implemented in SGLang's LoRA tensor parallelism.

---

## Custom CUDA Kernels

S-LoRA modifies and extends Punica's SGMV kernel to support:
- **Non-contiguous adapter pages** in the unified memory pool
- **Variable-rank adapters** in the same batch
- **Decode-phase optimization** (single-token batches from many adapters)

---

## Benchmarks

On A100 80GB GPU with LLaMA-7B:

| System | Max concurrent adapters | Throughput |
|---|---|---|
| HuggingFace PEFT | 1 | 1× |
| vLLM (naive) | ~16 (VRAM limited) | 1× |
| S-LoRA | **2000+** | **4×** |

The "several orders of magnitude" claim for adapter count comes from moving adapter storage to main memory (CPU DRAM, typically 256GB+) and only loading active adapters to GPU.

---

## Comparison: Punica vs S-LoRA

| Feature | Punica | S-LoRA |
|---|---|---|
| Memory management | All adapters in GPU VRAM | Unified Paging (GPU + CPU + disk) |
| Max adapters | Limited by VRAM | Limited by disk |
| Memory fragmentation | Yes (different ranks) | No (unified paged pool) |
| Tensor parallelism | Not addressed | Full TP strategy |
| Eviction policy | None | LRU page eviction |

---

## SGLang's Implementation of S-LoRA Concepts

SGLang's `--lora-eviction-policy lru` and `--max-loras-per-batch` directly implement S-LoRA's Unified Paging with LRU eviction:

```bash
python3 -m sglang.launch_server \
    --model-path base_model \
    --lora-paths adapter1 adapter2 adapter3 ... \
    --max-loras-per-batch 8 \          # max adapters in GPU pool simultaneously
    --lora-eviction-policy lru \       # LRU eviction when pool is full
    --max-loaded-loras 32              # max adapters in CPU memory
```

---

## Relevance to Layer 20

Layer 20 is S-LoRA with exactly 1 adapter and no memory management:

| Feature | Layer 20 | S-LoRA |
|---|---|---|
| Adapters in pool | 1 (static) | Thousands |
| Memory pool | Single GPU allocation | Unified Paging (GPU + CPU + disk) |
| Eviction policy | None needed | LRU page eviction |
| Tensor parallelism | Not supported | Column/row shard strategy |
| Adapter loading | At startup | JIT from CPU/disk |
| Kernel | Float mask (dense compute) | SGMV/BGMV (sparse compute) |

The next layer after Layer 20 would implement S-LoRA's Unified Paging to support multiple adapters — that is the full multi-LoRA implementation described in `sglang_multi_lora_implementation.md`.
