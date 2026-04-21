# Cost-Efficient Large Language Model Serving for Multi-turn Conversations with CachedAttention

**Source:** https://www.usenix.org/conference/atc24/presentation/gao-bin-cost
**Paper PDF:** https://www.usenix.org/system/files/atc24-gao-bin.pdf
**Venue:** USENIX ATC '24 (USENIX Annual Technical Conference, July 2024, Santa Clara, CA)
**Authors:** Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic (National University of Singapore); Junbo Deng, Xingkun Yang, Zhou Yu, Pengfei Zuo (Huawei Cloud)
**Level:** L3 — Academic paper; seminal multi-tier KV cache system
**Why here:** CachedAttention is the first published system to establish the exact three-tier KV cache hierarchy (GPU → CPU RAM → SSD/storage) that HiCache implements. Its core techniques — layer-wise pre-loading, asynchronous saving, and scheduler-aware fetching — are all present in HiCache. Reading this paper reveals why each design decision in HiCache exists and what the alternative designs are. Layer 17's `02_three_tier_architecture.md` and `03_host_pool.md` are essentially engineering-level elaborations of what this paper proposes at the research level.

**BibTeX:**
```bibtex
@inproceedings{298501,
  author = {Bin Gao and Zhuomin He and Puru Sharma and Qingxuan Kang and Djordje Jevdjic
            and Junbo Deng and Xingkun Yang and Zhou Yu and Pengfei Zuo},
  title = {{Cost-Efficient} Large Language Model Serving for Multi-turn Conversations
           with {CachedAttention}},
  booktitle = {2024 USENIX Annual Technical Conference (USENIX ATC 24)},
  year = {2024},
  pages = {111--126},
  url = {https://www.usenix.org/conference/atc24/presentation/gao-bin-cost},
  publisher = {USENIX Association},
  month = jul
}
```

---

## Problem

Existing LLM serving engines are **inefficient for multi-turn conversations** because they repeatedly compute the KV caches of historical tokens. Each new turn must recompute all previous context, incurring high costs even though the tokens are identical to the previous turn.

**Root cause**: LLM serving engines treat each request as stateless, discarding KV caches after each generation. For multi-turn conversations, this means re-prefilling the entire conversation history on every turn.

---

## Proposed Solution: CachedAttention

CachedAttention maintains a **hierarchical KV caching system** that leverages cost-effective memory/storage mediums to save KV caches for **all requests**, not just active ones.

### Three-tier hierarchy

```
GPU VRAM  [fastest, smallest]
    ↕  DMA (pinned memory, PCIe)
CPU DRAM  [medium speed, larger]
    ↕  Storage I/O
SSD / Persistent Storage  [slowest, largest]
```

### Four core techniques

#### 1. Layer-wise Pre-loading
When a request's KV cache resides in CPU/SSD, load it **one layer at a time** concurrently with GPU computation:
- Load layer N+1 from CPU while GPU computes attention for layer N
- Effectively hides the transfer latency behind compute
- Requires pinned (page-locked) CPU memory for DMA

> This is exactly what HiCache implements as "Compute-Transfer Overlap" in `load_to_device_per_layer()`.

#### 2. Asynchronous Saving
After GPU generates new KV cache tokens, save them to CPU/SSD **asynchronously**:
- Does not block the next inference step
- Uses a dedicated I/O thread and CUDA stream
- Prioritizes KV caches of requests likely to be accessed again

> HiCache's `backup_from_device_all_layer()` and `HiCacheController.write()` implement this.

#### 3. Scheduler-aware Fetching
The KV cache placement is guided by the **inference job scheduler's hints**:
- Before a conversation's next turn is scheduled, begin pre-fetching its KV cache from SSD → CPU → GPU
- The scheduler knows which sessions are "active" and likely to return
- Eviction prioritizes sessions unlikely to generate another turn

> HiCache's hit-count tracking (`_inc_hit_count()`) and `write_through_selective` write policy implement the priority aspect of this.

#### 4. KV Cache Invalidation Avoidance (Positional Encoding Decoupling)
Standard transformers embed positional information directly into Q/K/V projections. When conversation history grows past the context window, cached KV entries become **invalid** because positions shift.

CachedAttention **decouples positional encoding** from the saved KV caches, allowing cached entries to remain valid even after truncation.

> Note: HiCache does not implement this optimization — it relies on token-exact prefix matching. This is a known limitation for sessions that exceed the context window.

---

## Results

- TTFT reduced by up to **87%** for multi-turn conversations
- Prompt prefilling throughput improved by up to **7.8×**
- End-to-end inference cost reduced by up to **70%**

Test setup: 5-turn conversations, GPT-2-XL baseline, compared to no-cache baseline and GPU-only KV cache.

---

## Key Differences from HiCache

| Aspect | CachedAttention | SGLang HiCache |
|---|---|---|
| Scope | Multi-turn conversations only | Any prefix reuse (system prompts, RAG, coding agents) |
| Cache key | Session ID + turn number | RadixTree token prefix |
| Positional encoding | Decoupled (allows truncation) | Coupled to token prefix (token-exact match) |
| Storage backends | Generic (SSD assumed) | Pluggable: file, Mooncake, 3FS, NIXL, AIBrix, dynamic |
| Write policies | Not configurable | `write_back`, `write_through`, `write_through_selective` |
| Integration | Research prototype | Production system in SGLang |
| Tensor parallelism | Not addressed | `all_reduce` synchronization for TP |

---

## Key Takeaways for Layer 17

- **The three-tier hierarchy is not new** — CachedAttention established it in a peer-reviewed system in 2024.
- **Layer-wise pre-loading** is the core technique for hiding L2→L1 transfer latency; HiCache calls this "compute-transfer overlap."
- **Asynchronous saving** is why HiCache's write operations don't stall the inference scheduler.
- **Scheduler-aware placement** motivates HiCache's hit-count tracking and `write_through_selective` policy.
- CachedAttention's limitation (positional encoding invalidation) explains why HiCache focuses on **prefix-exact** reuse rather than session-level reuse.
