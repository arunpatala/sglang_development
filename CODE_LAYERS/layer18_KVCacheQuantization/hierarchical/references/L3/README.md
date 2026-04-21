# L3 References: KV Cache Quantization

**Level:** L3 — Technical / design-focused

**Reader profile:** Wants to understand how KV quantization is implemented in both SGLang and vLLM, what the scale calibration workflow looks like, and what the research papers say about sub-4-bit quantization. Has read the L2 intro and wants implementation depth.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_kivi_arxiv24.md` | arXiv Feb 2024 | KIVI — asymmetric 2-bit; per-channel K, per-token V; distribution analysis; 2.35–3.47× throughput; hardware-friendly no-tuning implementation |
| 02 | `02_kvquant_arxiv24.md` | arXiv Jan 2024 | KVQuant — sub-4-bit; pre-RoPE K quantization; non-uniform per-layer datatypes; per-vector outlier isolation; 1M token context on single A100 |
| 03 | `03_zipcache_arxiv24.md` | arXiv May 2024 | ZipCache — mixed precision with salient token identification; channel-separable tokenwise quant; FlashAttention-compatible; 4.98× compression, 56.9% decode latency reduction |
| 04 | `04_sageattention2_icml25.md` | ICML 2025 | SageAttention2 — INT4 QK + FP8 PV compute quantization (not just storage); 3× faster than FlashAttention2; orthogonal to KV storage quant, composable |

---

## Recommended reading order

**Fast path (60 min):** 01 → 02
- 01: The foundational distribution insight (K per-channel, V per-token). Everything else builds on this.
- 02: What happens when you push to sub-4-bit — pre-RoPE quant is the key technique.

**Thorough path (2–3 hours):** 01 → 02 → 03 → 04
- 03: Mixed-precision approach — not all tokens need to be quantized equally.
- 04: Compute quantization (attention matmul) — the next frontier beyond storage quantization.

---

## How these map to Layer 18

| Layer 18 topic | Most relevant L3 reference |
|---|---|
| K vs V distribution difference | 01 (KIVI: channel-wise outliers in K, smoother distribution in V) |
| Scale granularity choices | 01 (per-token), 02 (per-channel, per-vector) |
| Pre-RoPE quantization | 02 (KVQuant section 3.2) |
| Mixed-precision KV | 03 (salient token identification) |
| vLLM q_scale / prob_scale | 04 (quantizing the compute path, not just storage) |
| SGLang 2-bit future direction | 01 and 02 (research foundations for sub-FP8) |
