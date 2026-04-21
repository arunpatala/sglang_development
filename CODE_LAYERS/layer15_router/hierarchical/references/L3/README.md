# L3 References: LLM Router / Gateway

**Level:** L3 — Mechanism level (pseudocode, invariants, configuration)

**Reader profile:** Has read the lesson files and wants to understand the design decisions. Comfortable with Python and distributed systems concepts. Wants runnable examples and parameter tables before looking at the SGLang Rust source.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_ray_serve_prefix_cache_affinity_router.md` | Ray Docs 2.54.0 | Best short technical description of the two-guard policy design. Full parameter reference. Runnable Python config. |
| 02 | `02_sglang_router_cli_reference.md` | SGLang Docs | Policy names, parameter defaults, deployment modes. Essential companion to `config.yml`. |
| 03 | `03_intelligent_router_llm_workloads.md` | arXiv 2408.13510 | Empirical study: establishes round-robin as baseline, Join-Shortest-Queue as `LeastLoadPolicy`, quantifies routing improvement. |

---

## Recommended reading order

**Fast path (30 min):** 01 → 02
- 01 for the mechanism: two-guard policy with parameter mapping to Layer 15.
- 02 for the production CLI: how the same parameters are configured in SGLang.

**Thorough path (60 min):** 02 → 01 → 03
- 02 first to orient around production parameter names.
- 01 for the detailed mechanism.
- 03 for empirical validation of why three policies are needed.

---

## How these map to Layer 15

| Layer 15 `router.py` | Most relevant L3 reference |
|---------------------|---------------------------|
| `PrefixCacheAwarePolicy._pick_worker` | 01 (three-tier strategy: load check → prefix match → fallback) |
| `config.yml` `cache_threshold` | 01 (`match_rate_threshold`), 02 (`--cache-threshold`) |
| `config.yml` `balance_abs_threshold` | 01 (`imbalanced_threshold`), 02 (`--balance-abs-threshold`) |
| `config.yml` `policy: round_robin` | 02 (`--policy round_robin`) |
| `config.yml` `policy: least_load` | 02 (`--policy power_of_two`) |
| `config.yml` `policy: prefix_cache_aware` | 02 (`--policy cache_aware`) |
| `Worker.in_flight` | 03 (Join Shortest Queue tracks outstanding request count) |

---

## Common L3 limits to name for readers

These articles **do not explain**:
- The Rust implementation of `RadixTree` in `sgl-model-gateway/src/policies/cache_aware.rs`.
- Circuit breaker three-state machine (Closed → Open → HalfOpen).
- Prometheus metrics format and naming conventions.
- Kubernetes service discovery via label selectors.

Those live in L4 (production docs and papers) and L5 (source code).
