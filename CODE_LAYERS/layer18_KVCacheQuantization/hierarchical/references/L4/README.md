# L4 References: KV Cache Quantization

**Level:** L4 — Advanced / research and survey papers

**Reader profile:** Wants to understand the KV cache quantization landscape at a research level: how methods are taxonomized, how they compare on controlled benchmarks, and where the field is heading. Comfortable reading systems papers.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_longctx_compression_bench_24.md` | arXiv Jul 2024 | Comprehensive benchmark of 10+ KV compression methods across 7 task categories; reveals complementary failure modes of quantization vs token dropping; the definitive controlled comparison |
| 02 | `02_kv_cache_management_survey_24.md` | arXiv Dec 2024 | 200+ paper survey; KV cache quantization is the "token-level" subcategory; situates FP8/INT8/2-bit in relation to token dropping, merging, low-rank decomposition, system-level tiering |
| 03 | `03_llm_inference_optimization_survey_24.md` | arXiv Aug 2024 | Broad LLM inference optimization taxonomy; places KV cache quantization in context of activation quantization (dynamic, input-dependent) vs weight quantization (static) |

---

## Recommended reading order

**Fast path (60 min):** 02 (survey taxonomy section on token-level quantization)
- Gets you the full landscape in one document with comparative analysis.

**Thorough path (3–4 hours):** 01 → 02 → 03
- 01: controlled benchmarks — see actual numbers for quantization vs other compression approaches.
- 02: full taxonomy — understand where every paper fits.
- 03: broader context — KV cache quantization as a special case of activation quantization.

---

## How these map to Layer 18

| Layer 18 topic | Most relevant L4 reference |
|---|---|
| KV quant vs token dropping tradeoffs | 01 (benchmark across 7 task types) |
| Where FP8/INT8 fits in the landscape | 02 (token-level quantization section) |
| Why KV quant is harder than weight quant | 03 (activation quantization: dynamic range varies by input) |
| Research directions beyond FP8 | 02 (2-bit methods like KIVI, KVQuant in the survey) |
