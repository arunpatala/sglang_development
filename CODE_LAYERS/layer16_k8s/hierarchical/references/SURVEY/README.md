# SURVEY References: LLM Router / Gateway

**Level:** L4 / Survey — Foundational academic papers and comprehensive surveys.

**Reader profile:** Wants to understand the theoretical underpinnings of the routing algorithms and how Layer 15's design fits into the broader research literature.

---

## Files in this directory

> Files marked **[summary]** are hand-written concept summaries. Files marked **[full paper]** are the complete paper text converted from PDF via `pymupdf4llm`.

| # | File | Type | Source | Best for |
|---|------|------|--------|----------|
| 01 | `01_mitzenmacher2001_power_of_two_choices.md` | summary | IEEE TPDS 2001 | The mathematical proof that d=2 choices gives exponential improvement over d=1 (random). Foundation for `LeastLoadPolicy`. |
| 02 | `02_srivatsa2024_preble_iclr2025.md` | summary | ICLR 2025 | Concept summary of Preble's E2 algorithm and its mapping to Layer 15. |
| 03 | `03_preble_iclr2025_full.md` | full paper | arXiv:2407.00023 | Complete Preble paper — E2 algorithm, global radix tree, 1.5×–14.5× latency improvement. |
| 04 | `04_intelligent_router_ibm_full.md` | full paper | arXiv:2408.13510 | Complete IBM Intelligent Router paper — round-robin baseline, RL routing, Join-Shortest-Queue. |
| 05 | `05_skywalker_eurosys2026_full.md` | full paper | arXiv:2505.24095 | Complete SkyWalker paper — cross-region load balancing, SGLang Router as single-region baseline. |
| 06 | `06_survey_llm_inference_2025_full.md` | full paper | arXiv:2506.21901 | Complete LLM Inference Systems survey — routing section covers Round Robin, PoT, Preble, PD disaggregation. |
| 07 | `07_sglang_radixattention_full.md` | full paper | arXiv:2312.07104 | Complete SGLang paper — RadixAttention, tree-based KV cache reuse, motivation for prefix-aware routing. |

---

## Recommended reading order

**Theory first:** 01 → 03
- 01: mathematical foundation of power-of-two choices.
- 03: full Preble paper — the distributed prefix caching algorithm built on top (E2 algorithm).

**Broader context:** 06 → 04 → 05 → 07
- 06: survey placing all routing strategies in the full inference stack.
- 04: IBM empirical study — round-robin baseline and RL-based alternatives.
- 05: SkyWalker — cross-region extension and SGLang Router benchmarks.
- 07: SGLang/RadixAttention — why per-engine prefix caching alone isn't enough at cluster scale.

---

## How these map to Layer 15

| Layer 15 | Survey reference |
|---|---|
| `LeastLoadPolicy._pick_worker` | 01 (Mitzenmacher 2001: d=2 is the sweet spot) |
| `PrefixCacheAwarePolicy` cache-affinity routing | 03 (Preble E2: exploitation phase) |
| `PrefixCacheAwarePolicy` load-balance guard | 03 (Preble E2: exploration phase) |
| `cache_threshold` parameter | 03 (E2 minimum match ratio) |
| `balance_abs_threshold` parameter | 03 (E2 load divergence threshold) |
| `RadixTrie` per worker | 03 (Preble global radix tree, simplified to per-worker) |
| `RoundRobinPolicy` baseline | 04 (IBM §A.1.5 — round-robin as empirical baseline) |
| `LeastLoadPolicy` = Join-Shortest-Queue | 04 (IBM §A.2.1 — JSQ is the Layer 15 analog) |
| SGLang Router = production reference | 05 (SkyWalker: SGLang Router used as single-region baseline) |
| KV cache reuse motivation | 07 (SGLang RadixAttention — why prefix affinity matters) |

---

## Key Papers Not Downloaded (but referenced)

| Paper | Why relevant | Where cited |
|---|---|---|
| Mitzenmacher et al. 2001 (PoT Survey) | Broader survey of two-choice techniques | REFERENCES.md |
| Ray Serve PrefixCacheAffinityRouter | Cluster-level prefix routing, `imbalanced_threshold` | L3/01 |
