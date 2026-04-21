# L2 References: Tiered KV Cache

**Level:** L2 — Practitioner / deployment perspective

**Reader profile:** Knows LLM inference basics (prefill, decode, KV cache). Has run SGLang or vLLM. Wants to understand why GPU VRAM runs out, what CPU offloading does, and how to enable HiCache. Satisfied when they can write a working launch command and understand what each flag does.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_hicache_blog.md` | LMSYS Blog (Sep 2025) | The SGLang HiCache launch post: motivation, design summary, 6× throughput benchmark, production launch commands with 3FS and Mooncake backends. |
| 02 | `02_vllm_kv_offloading_blog.md` | vLLM Blog (Jan 2026) | vLLM's CPU offloading feature introduced in v0.11.0/v0.12.0; DMA vs custom kernel comparison; physical block-size table for common models; throughput benchmarks. Direct comparison point for HiCache design choices. |

---

## Recommended reading order

**Fast path (20 min):** 01
- The LMSYS blog post is the most direct answer to "what is HiCache and why does it matter." Read this first.

**Thorough path (45 min):** 01 → 02
- 02 shows how vLLM tackled the same problem; seeing the differences (DMA vs GPU-assisted kernels, page layout change) deepens understanding of HiCache's design choices.

---

## How these map to Layer 17

| Layer 17 lesson | Most relevant L2 reference |
|---|---|
| `01_eviction_problem.md` — why GPU eviction is costly | 01 (community testimonials on TTFT impact) |
| `02_three_tier_architecture.md` — three-tier overview | 01 (HiRadixTree diagram, architecture summary) |
| `03_host_pool.md` — GPU↔CPU DMA | 02 (DMA vs custom kernel analysis, block-size table) |
| `06_configuration.md` — launch commands | 01 (canonical production launch recipes for 3FS and Mooncake) |

---

## Common L2 limits to name for readers

These articles **do not explain:**
- How `HiRadixCache.evict()` works internally or how the write-through threshold is triggered.
- The `PoolTransfer` / `HiCacheStorage` interface and how to write a custom backend.
- Why pinned memory is faster than pageable for GPU DMA.
- The `RadixCacheMetricsCollector` Prometheus metrics and how to interpret them.

Those live in L3 (design docs) and L4 (papers) references.
