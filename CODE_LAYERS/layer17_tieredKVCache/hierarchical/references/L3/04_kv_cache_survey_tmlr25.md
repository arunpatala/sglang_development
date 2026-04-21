# A Survey on Large Language Model Acceleration based on KV Cache Management

**Source:** https://arxiv.org/abs/2412.19442
**Paper PDF:** https://arxiv.org/pdf/2412.19442
**Venue:** arXiv (submitted December 27, 2024; revised July 30, 2025)
**Authors:** Haoyang Li et al. (TreeAI Lab)
**GitHub (curated paper list):** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
**Level:** L3 — Comprehensive survey; orientation for the research space
**Why here:** This survey categorizes 200+ KV cache management papers into a clear taxonomy (token-level, model-level, system-level). It situates HiCache in the broader research landscape: as a **system-level, lossless, multi-tier prefix cache**. Reading this before the L4 papers helps understand which research threads led to HiCache's design decisions vs which are orthogonal optimizations. Layer 17's `REFERENCES.md` high-level section draws from this taxonomy.

**BibTeX:**
```bibtex
@article{li2024kvcachesurvey,
  title = {A Survey on Large Language Model Acceleration based on {KV} Cache Management},
  author = {Haoyang Li and others},
  journal = {arXiv preprint arXiv:2412.19442},
  year = {2024},
  url = {https://arxiv.org/abs/2412.19442}
}
```

---

## Taxonomy

The survey organizes KV cache optimizations into three levels:

### Token-level Optimizations
Reduce the number of KV entries that need to be stored or computed:
- **KV cache selection** — drop less important tokens from cache (e.g., H2O, StreamingLLM)
- **Budget allocation** — adaptive per-layer budget (e.g., PyramidKV)
- **KV merging** — merge similar KV entries (e.g., KVMerger)
- **KV quantization** — lower precision for cached entries (e.g., KIVI, KVQuant)
- **Low-rank decomposition** — compress KV matrices

> HiCache is **not** in this category — it stores full-precision KV caches. Token-level methods trade accuracy for memory; HiCache is lossless.

### Model-level Optimizations
Architectural changes to reduce KV cache size or improve reuse:
- **Multi-Query Attention (MQA)** — share K/V across heads
- **Grouped Query Attention (GQA)** — share K/V across groups of heads
- **Multi-head Latent Attention (MLA)** — DeepSeek's low-rank projection approach

> HiCache supports both MHA and MLA via `MHATokenToKVPoolHost` and `MLATokenToKVPoolHost`. MLA has a dedicated optimization (single-rank write-back).

### System-level Optimizations
Focus on **how** KV caches are stored, moved, and scheduled:
- **Memory management** — paged memory (PagedAttention), radix tree (RadixAttention)
- **Scheduling** — preemption policies, disaggregated prefill/decode (PD disaggregation)
- **Hardware-aware designs** — CPU offloading, NVMe storage, RDMA transfers
- **Multi-tier systems** — hierarchical caches spanning GPU/CPU/storage

> **HiCache belongs here** — system-level, multi-tier, lossless prefix cache built on RadixAttention.

---

## Where HiCache Fits in the Taxonomy

```
System-level Optimizations
└── Multi-Tier KV Cache Systems
    ├── GPU + CPU tier (L1+L2)           → HiCache Tier 1+2
    ├── CPU + persistent storage tier    → HiCache Tier 2+3
    ├── Cross-instance sharing via L3    → HiCache shared backends
    └── Prefix reuse via RadixTree       → HiCache's HiRadixTree
```

HiCache is a **lossless, prefix-based, multi-tier system-level KV cache manager**. It does not compress, quantize, or drop KV entries — it focuses entirely on **where** caches live and **how quickly** they can be restored.

---

## Key Papers in the System-level, Multi-tier Thread

The survey identifies these as the primary multi-tier KV cache systems:

| Paper | Venue | Contribution |
|---|---|---|
| PagedAttention (vLLM) | SOSP 2023 | OS-style paged GPU KV memory management |
| RadixAttention (SGLang) | arXiv 2023 | Prefix tree for cross-request KV reuse |
| CachedAttention | ATC 2024 | Three-tier hierarchy with layer-wise loading (→ see L3/02) |
| InfiniGen | OSDI 2024 | Selective prefetch of important KV from CPU (→ see L4/01) |
| IMPRESS | FAST 2025 | Importance-informed three-tier with storage focus (→ see L4/02) |
| LMCache | arXiv 2025 | Middleware KV cache layer, cross-engine sharing (→ see L3/03) |
| HiCache (SGLang) | Blog 2025 | Production multi-tier with pluggable backends (→ see L2/01) |

---

## Token-Level Methods: The Other Direction

While HiCache focuses on **where** to store KV caches (lossless), the survey documents many methods for **which** KV entries to keep (lossy). These are orthogonal to HiCache:

- **H2O** (Heavy Hitter Oracle): keep the top-k most recently/frequently attended KV entries
- **StreamingLLM**: keep attention sinks (first few tokens) + recent sliding window
- **PyramidKV**: allocate more KV budget to lower layers (which have more uniform attention patterns)
- **KIVI**: quantize KV to 2-bit to fit 4× more in GPU VRAM

> These methods can be combined with HiCache: quantize what's in GPU VRAM while offloading the rest to CPU/storage.

---

## Quantization in Context

The survey covers KV quantization extensively. Key data point: KV quantization typically uses INT8 or INT4. This affects:
- **Storage size**: 2–4× reduction
- **Accuracy**: small but measurable degradation on long-context tasks
- **Transfer bandwidth**: proportionally reduced PCIe transfer time

HiCache stores **FP16/BF16** (full precision). If combined with KV quantization, HiCache's host pool could store 2–4× more tokens for the same memory budget.

---

## Key Takeaways for Layer 17

- HiCache is a **system-level, lossless, multi-tier prefix cache** — it does not touch the KV values themselves.
- **Token-level methods** (H2O, StreamingLLM, KIVI) are **orthogonal** to HiCache and can be combined with it.
- **Model-level methods** (GQA, MLA) change the KV shape; HiCache handles this via separate host pool implementations.
- The multi-tier thread (CachedAttention → InfiniGen → IMPRESS → HiCache) is the direct academic lineage.
- The curated paper list at [github.com/TreeAI-Lab/Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) is the best starting point for further research.
