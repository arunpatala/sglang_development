# L2 References: KV Cache Quantization

**Level:** L2 — Practitioner / deployment perspective

**Reader profile:** Has run SGLang or vLLM. Wants to enable FP8 KV cache today, understand what `--kv-cache-dtype fp8_e4m3` does, and know whether they need a calibration file. Satisfied when they can write a launch command and interpret the accuracy warning.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_vllm_kv_quant_docs.md` | vLLM Blog (Jan 2026) | vLLM's CPU KV offloading connector, DMA analysis, block layout — shows how vLLM and SGLang solve the same problems with different approaches; the physical block size table for common models explains FP8 transfer efficiency |
| 02 | `02_kivi_paper.md` | arXiv Feb 2024 | KIVI — the foundational paper explaining WHY K needs per-channel quant and V needs per-token quant; 2-bit results; the key distribution insight behind all serious KV quantization work |

---

## Recommended reading order

**Fast path (15 min):** 02 (KIVI abstract + key findings table)
- Establishes the core insight: K and V have different distributions and need different quantization strategies.

**Thorough path (45 min):** 01 → 02
- 01: deployment context — how the physical memory layout affects quantization efficiency (larger contiguous blocks = better FP8 throughput).
- 02: the statistical justification for asymmetric quantization.

---

## How these map to Layer 18

| Layer 18 topic | Most relevant L2 reference |
|---|---|
| Why FP8 KV cache matters | 02 (2.6× memory reduction, 3.47× throughput) |
| K vs V distribution difference | 02 (per-channel K, per-token V) |
| Physical block layout and DMA | 01 (fragmented vs contiguous block sizes) |
| Scale calibration | 01 (DMA efficiency varies with block size; same logic applies to scale granularity) |
