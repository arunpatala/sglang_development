# L3 References: Tiered KV Cache

**Level:** L3 — Technical / design-focused

**Reader profile:** Wants to understand the design decisions behind HiCache: why layer-wise loading, why pinned memory, what the storage backend interface looks like, how prefetch policies work. Has read the L2 blog posts and wants more depth.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_hicache_design.md` | SGLang Docs | Complete architectural reference: HiRadixTree design, workflow (local match → prefetch → write-back), data transfer optimizations, all config parameters with descriptions. |
| 02 | `02_cachedattention_atc24.md` | USENIX ATC 2024 | CachedAttention paper — the academic precedent with identical three-tier design (GPU→CPU→SSD); layer-wise pre-loading, async save, scheduler-aware prefetch; 87% TTFT reduction. |
| 03 | `03_lmcache_arxiv25.md` | arXiv Oct 2025 | LMCache paper — the alternative to HiCache in SGLang (`--enable-lmcache`); middleware vs embedded approach; supports CPU/disk/Redis/S3; 15× throughput on multi-round QA. |
| 04 | `04_kv_cache_survey_tmlr25.md` | TMLR May 2025 | Comprehensive 200+ paper survey; token/model/system taxonomy; situates HiCache as a system-level, multi-tier, lossless prefix cache. |

---

## Recommended reading order

**Fast path (45 min):** 01 → 02
- 01 for the authoritative SGLang design document.
- 02 for the academic work that established the three-tier pattern; confirms HiCache's choices are well-founded.

**Thorough path (90 min):** 01 → 02 → 03 → 04
- 03 to understand what LMCache does differently and why SGLang offers both.
- 04 for orientation in the research landscape — useful before reading L4 papers.

---

## How these map to Layer 17

| Layer 17 lesson | Most relevant L3 reference |
|---|---|
| `02_three_tier_architecture.md` — HiRadixTree, node state machine | 01 (HiRadixTree design section), 02 (CachedAttention architecture) |
| `03_host_pool.md` — layer-wise overlapping, compute-transfer overlap | 01 (CPU-to-GPU transfer optimizations), 02 (layer-wise pre-loading) |
| `04_storage_backend.md` — prefetch policies, write-back policies | 01 (prefetch from L3 section, write-back section) |
| `05_observability.md` — what to monitor | 04 (survey taxonomy helps frame which metrics matter) |
| `06_configuration.md` — LMCache vs HiCache | 03 (LMCache architecture and tradeoffs) |
