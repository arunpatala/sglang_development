# L4 References: LLM Router / Gateway

**Level:** L4 — Production + systems (real stacks, benchmarks, tradeoffs)

**Reader profile:** Has read all lesson files and L3 references. Comfortable reading production documentation and academic papers. Wants to understand the full production system before looking at source code.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_model_gateway_docs.md` | SGLang Docs | Primary production reference. All features beyond Layer 15 scope: gRPC, PD disaggregation, circuit breaker, Prometheus, Kubernetes. Source code mapping to `router.py`. |
| 02 | `02_preble_distributed_prompt_scheduling.md` | ICLR 2025 | Theoretical and empirical foundation for `PrefixCacheAwarePolicy`. E2 algorithm = Layer 15's cache-threshold/balance-threshold design. |
| 03 | `03_skywalker_cross_region_load_balancer.md` | EUROSYS 2026 | Cross-region extension. Benchmark set matches Layer 15's three policies. SGLang Router (= Layer 15) is the recognized single-region baseline. |
| 04 | `04_survey_llm_inference_systems.md` | arXiv 2025 | Survey. §5.2.2 positions Round Robin, PoT, Preble, and PD disaggregation in the inference stack. |

---

## Recommended reading order

**Fast path (45 min):** 02 → 01
- 02 for the theoretical foundation (E2 algorithm) and experimental validation.
- 01 for the production reference (all features, source code mapping).

**Thorough path (90 min):** 02 → 01 → 03 → 04
- 03 for the cross-region perspective and benchmark context.
- 04 for the academic survey framing.

---

## How these map to Layer 15

| Layer 15 design choice | Most relevant L4 reference |
|------------------------|---------------------------|
| `cache_threshold: 0.5` default | 02 (Preble E2 min-match ratio), 01 (`--cache-threshold 0.3` SGLang default) |
| `balance_abs_threshold: 32` default | 02 (Preble load gap trigger), 01 (`--balance-abs-threshold 64` SGLang default) |
| Three-policy design | 03 (SkyWalker benchmark: RR + LL + cache_aware = complete baseline set) |
| `_health_loop` simplicity | 01 (circuit breaker three-state = what was deliberately omitted) |
| `lesson/09_whats_next.md` PD disaggregation | 01 (PD launch mode), 02 (Preble multi-GPU), 03 (KV cache transfer) |
| RadixTrie text-level matching | 02 (Preble global radix tree at token level; text-level is the router-side approximation) |

---

## Features beyond Layer 15 scope (to name for advanced readers)

| Feature | Where to read |
|---|---|
| Circuit breaker (3-state machine) | 01 (SGLang docs: `--cb-*` parameters) |
| Retries with exponential backoff | 01 (SGLang docs: `--retry-*` parameters) |
| Prometheus metrics | 01 (SGLang docs: 40+ metrics, Prometheus port) |
| OpenTelemetry tracing | 01 (SGLang docs: distributed tracing section) |
| gRPC routing | 01 (SGLang docs: gRPC launch mode) |
| PD disaggregation | 01 (SGLang docs: `--pd-disaggregation`) |
| Kubernetes service discovery | 01 (SGLang docs: `--service-discovery`) |
| Hot entry replication | 02 (Preble: replication of popular prefixes across GPUs) |
| Cross-region KV transfer | 03 (SkyWalker: selective KV block migration) |
