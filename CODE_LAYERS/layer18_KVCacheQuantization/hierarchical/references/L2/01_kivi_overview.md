# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

**Source:** https://arxiv.org/abs/2402.02750
**Paper PDF:** https://arxiv.org/pdf/2402.02750
**Venue:** NeurIPS 2024
**Authors:** Jiayi Yuan, Haoxuan You, Tianhao Zhao, Yiyang Cai, Guohao Li, Siddharth Garg, Zhangyang Wang (UCSD, NYU, MBZUAI)
**GitHub:** https://github.com/jy-yuan/KIVI
**Level:** L2 — Foundational insight, accessible abstract
**Why here (L2):** KIVI's core insight — that K and V tensors have different statistical distributions and need asymmetric quantization strategies — is the single most important concept for understanding why FP8 KV quantization sometimes hurts accuracy more than expected. The abstract alone (2-minute read) completely changes how you think about KV quantization design.

**BibTeX:**
```bibtex
@inproceedings{yuan2024kivi,
  title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
  author={Jiayi Yuan and Haoxuan You and Tianhao Zhao and Yiyang Cai
          and Guohao Li and Siddharth Garg and Zhangyang Wang},
  booktitle={Advances in Neural Information Processing Systems (NeurIPS)},
  year={2024},
  url={https://arxiv.org/abs/2402.02750}
}
```

---

## The Core Problem

KV cache grows linearly with **batch size × context length**. For larger batch sizes and longer context lengths, KV cache becomes the new bottleneck in both speed and memory. Loading KV cache from memory also causes the GPU compute units to be **idle** — a utilization problem, not just a memory problem.

**Straightforward solution:** quantize KV cache to fewer bits. But existing work lacked understanding of the KV element distribution, leading to approaches that either hurt accuracy significantly or required tuning.

---

## The Core Insight: Asymmetric Distribution

KIVI's key contribution is a comprehensive study of the **element distribution** in KV caches of popular LLMs:

### Key (K) cache distribution

- **Outliers appear along the channel dimension** — specific channels (feature dimensions) consistently produce large values across all tokens.
- These per-channel outliers have stable statistics across different tokens and inputs.
- → **Must quantize K per-channel**: group elements along the channel dimension and quantize each channel separately with its own scale and zero-point.
- Quantizing K per-token would place outlier channels in the same quantization bin as normal channels → extreme clipping.

### Value (V) cache distribution

- **Smoother distribution** — values vary across tokens but are more uniform within each token.
- No persistent per-channel outliers.
- → **Can quantize V per-token**: group elements along the token dimension.
- Per-token quantization is also simpler to implement and more cache-friendly.

This asymmetry (K per-channel, V per-token) is why the method is called **Asymmetric** 2-bit quantization.

---

## Quantization Scheme

```
K cache: [num_tokens, num_heads, head_dim]
  → quantize along head_dim (channel) dimension
  → one (scale, zero) per (token, head) channel group
  → group_size elements share one quantization range

V cache: [num_tokens, num_heads, head_dim]
  → quantize along num_tokens (token) dimension
  → one (scale, zero) per (head, channel) token group
  → group_size tokens share one quantization range
```

Both K and V use a **residual** scheme: the last `group_size` tokens are kept at FP16 (the "sink" tokens) because they haven't accumulated enough history for the channel statistics to stabilize.

---

## Results

| Metric | KIVI-2 (2-bit) vs BF16 |
|---|---|
| Peak memory reduction | **2.6×** (including model weights) |
| Batch size increase | **up to 4×** for same VRAM |
| Throughput improvement | **2.35–3.47×** on real LLM inference |
| Quality (LLaMA, Falcon, Mistral) | Near-identical — "almost the same quality" |

Benchmarks: LongBench, NeedleBench, MMLU, and other standard evaluations.

---

## Why This Matters for SGLang/vLLM FP8 KV

Both SGLang (`--kv-cache-dtype fp8_e4m3`) and vLLM (`fp8_e4m3`) use **per-tensor FP8** for K and V — a single scale per attention layer. This is simpler and faster but ignores KIVI's insight:

- **Per-tensor FP8 K**: one scale for the entire K tensor. Channel outliers get quantized with the same scale as normal channels → outlier channels dominate the scale → normal channels lose precision.
- **KIVI-2 K**: per-channel scale → outlier channels get their own scale → normal channels retain precision.

This explains why FP8 KV with scale=1.0 (no calibration) performs worse: the scale is too large for normal channels. With calibrated checkpoint scales (from `llm-compressor`), the per-tensor scale is chosen to minimize the MSE across the full tensor, which partially compensates.

vLLM's `fp8_per_token_head` mode (dynamic per-token-head scales) is a step toward KIVI's insight — adapting the scale dynamically per token rather than using a fixed per-tensor scale.

---

## Key Takeaways for Layer 18

- **K needs per-channel quantization** (channel outliers); **V needs per-token quantization** (token-wise variation).
- Per-tensor FP8 (current production approach) is a simplification of the optimal approach — it works well enough with calibration, but sub-optimal.
- 2-bit quantization is achievable with near-lossless quality if the granularity is right — this is the research direction that could push memory savings beyond the 2× of FP8.
- The `group_size` parameter in KIVI (how many elements share a scale) is the main accuracy/memory tradeoff knob.
- SGLang and vLLM do not yet implement per-channel K quantization; this remains a research gap in production systems.
