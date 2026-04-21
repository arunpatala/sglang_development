# IMPRESS: An Importance-Informed Multi-Tier Prefix KV Storage System for Large Language Model Inference

**Source:** https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress
**Paper PDF:** https://www.usenix.org/system/files/fast25-chen-weijian.pdf
**Venue:** USENIX FAST '25 (File and Storage Technologies, February 2025, Santa Clara, CA)
**Authors:** Weijian Chen, Shuibing He, Haoyang Qu, Ruidong Zhang, Siling Yang, Ping Chen (Zhejiang University); Yi Zheng, Baoxing Huai (Huawei Cloud); Gang Chen (Zhejiang University)
**Level:** L4 — Advanced research paper; storage-systems perspective on tiered KV cache
**Why here:** IMPRESS is the most systems-focused paper in the tiered KV cache space, published at FAST (the top file and storage systems venue). It tackles the specific problem that HiCache's `--hicache-storage-backend file` and disk backends face: **disk I/O latency is high, so loading all prefix KV from SSD does not always reduce TTFT**. IMPRESS's importance-informed selection and per-tier KV reordering are research-level techniques that inform HiCache's L3 prefetch policy design.

**BibTeX:**
```bibtex
@inproceedings{305216,
  author = {Weijian Chen and Shuibing He and Haoyang Qu and Ruidong Zhang and Siling Yang
            and Ping Chen and Yi Zheng and Baoxing Huai and Gang Chen},
  title = {{IMPRESS}: An {Importance-Informed} {Multi-Tier} Prefix {KV} Storage System
           for Large Language Model Inference},
  booktitle = {23rd USENIX Conference on File and Storage Technologies (FAST 25)},
  year = {2025},
  pages = {187--201},
  url = {https://www.usenix.org/conference/fast25/presentation/chen-weijian-impress},
  publisher = {USENIX Association},
  month = feb
}
```

---

## Problem

Modern LLM applications often **prepend long contexts** before user queries (system prompts, RAG documents, tool definitions, conversation history) to improve output quality. These contexts frequently repeat — partially or fully — across multiple queries. Existing systems store and reuse their prefix KVs to reduce redundant computation.

**The disk I/O problem**: when CPU memory is insufficient to hold all prefix KVs, they must be stored on disk. However:
- **Disk I/O latency is high** (NVMe: ~100 µs; network storage: >1 ms)
- **Loading all prefix KVs from disk does not always reduce TTFT** — I/O time can exceed the computation savings
- **Naïve loading of all KV pages wastes I/O bandwidth** on pages that contribute little to attention accuracy

---

## IMPRESS's Core Insight: Importance-Informed I/O

**Key observation**: attention scores across different attention heads show significant similarity in which tokens are "important." A single importance score per token can represent all heads with low error.

This enables an **I/O-efficient important KV identification algorithm** that:
1. Computes importance scores with much less overhead than full attention
2. Reorders KV pages by importance on disk (important pages stored contiguously)
3. Loads only the top-k important pages during inference, skipping the rest

---

## System Architecture

### Three-tier hierarchy

```
GPU VRAM     [L1, fastest]
    ↕  PCIe DMA
CPU DRAM     [L2, medium]
    ↕  NVMe / Storage I/O
SSD / Disk   [L3, slowest but largest]
```

### Four key components

#### 1. Importance-informed KV Identification
- Cross-head attention similarity: importance scores are consistent across heads
- Run a lightweight approximation of attention to get per-token importance scores
- Cost: significantly lower than full attention computation

#### 2. KV Reordering on Disk
- After computing importance scores, **physically reorder** KV pages on disk by importance
- Important pages are stored contiguously → can be loaded with a single large sequential I/O
- Unimportant pages stored separately → can be skipped entirely

#### 3. Importance-informed Cache Management
- Between GPU, CPU, and disk tiers: evict unimportant pages first
- Retain important pages in faster tiers longer (importance-weighted LRU)
- Prefetch only important pages from disk to CPU when a prefix cache hit is detected

#### 4. Caching Accuracy Guarantee
- By selecting only the top-k% of KV pages by importance, IMPRESS maintains **comparable inference accuracy** to loading all KV pages
- The importance threshold is configurable; higher = less I/O but slightly lower accuracy

---

## Results

- **TTFT reduced by up to 2.8×** compared to state-of-the-art systems
- Maintains **comparable inference accuracy** (within 0.5–2% on benchmarks)
- Significant improvement specifically for **disk-tier** scenarios where I/O latency is the bottleneck
- Evaluated on: LLaMA-2-7B, LLaMA-2-13B, Mistral-7B; datasets: LongBench, GSM8K

---

## Connection to HiCache

| IMPRESS Concept | HiCache Equivalent |
|---|---|
| Three-tier GPU/CPU/SSD hierarchy | HiCache L1/L2/L3 |
| Importance score per token | Hit count (`_inc_hit_count()`) as a proxy for importance |
| Load only important pages from disk | `write_through_selective` + prefetch threshold |
| KV reordering for I/O efficiency | `page_first` memory layout for contiguous page I/O |
| Per-tier eviction by importance | LRU eviction in `HiRadixCache.evict()` (hit-count weighted) |
| Prefetch trigger on cache hit detection | HiCache's L3 query → prefetch pipeline in `HiCacheController` |

**Key difference**: IMPRESS applies **content-based importance** (which tokens matter for attention); HiCache uses **access-frequency-based importance** (which prefixes are accessed repeatedly). IMPRESS's approach is lossy; HiCache's is lossless.

**Research direction**: applying IMPRESS-style importance scoring to HiCache's L3 prefetch decisions — loading only important token KV pages from storage instead of all pages for a prefix — is an open problem.

---

## Why FAST Matters

FAST is the premiere storage systems conference. IMPRESS appearing there signals that:
- The storage systems community recognizes LLM KV cache management as a **storage problem**, not just a ML problem
- Future hardware-storage co-design (NVMe-oF, GPU Direct Storage, CXL memory) is expected to address KV cache I/O bottlenecks
- Storage-tier optimization is as important as GPU-tier optimization for long-context inference

This aligns with HiCache's pluggable backend design: as new storage technologies emerge (CXL, GDS, persistent memory), new backends can be added without changing the HiCache architecture.

---

## Key Takeaways for Layer 17

- **Disk I/O latency dominates** for L3 backends — loading all prefix KV from NVMe can be slower than recomputing.
- **Importance-informed selection** (IMPRESS) is a way to get the benefit of large storage capacity without the full I/O cost.
- HiCache's `write_through_selective` and prefetch threshold approximate this by using access frequency as a proxy for importance.
- The `page_first` layout in HiCache directly addresses IMPRESS's insight about contiguous page I/O.
- IMPRESS's results (2.8× TTFT reduction) are for disk-tier scenarios — the most relevant comparison for HiCache with `--hicache-storage-backend file`.
- CXL memory and GPU Direct Storage are the next-generation hardware that will reshape these tradeoffs.
