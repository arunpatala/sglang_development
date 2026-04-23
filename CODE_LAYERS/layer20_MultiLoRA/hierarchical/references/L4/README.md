# L4 References: Prefill-Decode (PD) Disaggregation

**Level:** L4 — Advanced / research papers

**Reader profile:** Wants to understand the broader research context: what came before disaggregation (chunked prefill), whether disaggregation is always the right choice, and how the academic community is pushing the boundaries of disaggregated serving. Comfortable reading systems and ML papers.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sarathi_osdi24.md` | OSDI 2024 | SARATHI/Sarathi-Serve — chunked prefill as the aggregation-side alternative to disaggregation. Shows that carefully-scheduled collocated serving reduces (but cannot eliminate) interference. The control condition against which disaggregation gains are measured. |
| 02 | `02_taichi_agg_vs_disagg.md` | arXiv Aug 2025 | TaiChi — the most rigorous comparison of PD aggregation vs disaggregation. Shows each excels under different SLO regimes and proposes a unified hybrid system achieving 77% goodput improvement over SOTA. |
| 03 | `03_vllm_disagg_connector.md` | vLLM Docs + GitHub | vLLM disaggregated prefilling connector architecture — 6 connector types (NIXL, Mooncake, NCCL, LMCache, ExampleConnector, MultiConnector), the `kv_transfer/` module, Connector/LookupBuffer abstraction. |

---

## Recommended reading order

**Fast path (60 min):** 02 → 01
- 02 (TaiChi) first: the SLO-regime analysis is the clearest framework for deciding when to use disaggregation vs aggregation. Read this before deploying.
- 01 (SARATHI): understand chunked prefill as the incumbent technique that disaggregation must outperform. Also used as `--chunked-prefill-size` in SGLang's own prefill servers.

**Thorough path (2–3 hours):** 01 → 02 → 03
- 03 (vLLM connector architecture): for engineers who need to implement or integrate a custom KV transfer mechanism.

---

## How these map to Layer 19

| Layer 19 lesson | Most relevant L4 reference |
|---|---|
| `01_the_interference_problem.md` — why collocated systems use chunked prefill as a mitigation | 01 (SARATHI: chunked prefill reduces but doesn't eliminate interference) |
| `02_pd_architecture.md` — when to use disaggregation vs aggregation | 02 (TaiChi: SLO-regime analysis, TTFT-tight vs TPOT-tight) |
| `03_kv_transfer.md` — vLLM connector abstractions | 03 (Connector/LookupBuffer, 6 connector types, `--kv-transfer-config` JSON) |
| `06_tradeoffs_and_limits.md` — when disaggregation hurts | 02 (TaiChi: under tight TTFT + relaxed TPOT, aggregation is better) |
