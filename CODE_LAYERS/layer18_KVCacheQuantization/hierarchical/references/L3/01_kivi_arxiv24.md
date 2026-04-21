# KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

**Source:** https://arxiv.org/abs/2402.02750
**Paper PDF:** https://arxiv.org/pdf/2402.02750
**Venue:** NeurIPS 2024
**Authors:** Jiayi Yuan, Haoxuan You, Tianhao Zhao, Yiyang Cai, Guohao Li, Siddharth Garg, Zhangyang Wang
**GitHub:** https://github.com/jy-yuan/KIVI
**Level:** L3 — Technical research paper; foundational KV distribution analysis
**Why here:** KIVI is the first paper to systematically study the element distribution of KV caches across popular LLMs and derive a principled quantization strategy from that analysis. It establishes the canonical result that **K requires per-channel quantization** and **V requires per-token quantization** — the foundation that all subsequent KV quantization work (KVQuant, ZipCache, QJL, WKVQuant) builds on.

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

## Problem

KV cache is the dominant memory consumer for long sequences and large batches. At large batch sizes:
- KV cache exceeds model weight memory
- GPU compute waits for KV data to load (memory-bandwidth bound, not compute-bound)
- Larger batch = worse utilization per token

Quantization reduces total bytes taken by KV cache. But naive quantization causes significant accuracy degradation. **Why?** Existing work hadn't studied *why* KV cache is hard to quantize.

---

## The Distribution Study

KIVI analyzes the tensor statistics of K and V caches across LLaMA, Falcon, and Mistral models on diverse tasks. Three key findings:

### Finding 1: Channel-wise outliers in K

The **Key** cache exhibits **persistent outlier channels** — specific feature dimensions (channels in the `head_dim` axis) that consistently produce values much larger than the median across all tokens.

```
K[token, head, :] visualization:
channel 0:  [0.1, 0.2, 0.1, ...]  — normal
channel 7:  [8.2, 8.5, 8.1, ...]  — outlier (consistent across tokens)
channel 64: [0.3, 0.2, 0.4, ...]  — normal
```

These outlier channels appear at **the same positions across different tokens and inputs** — they are a structural property of the model, not random noise.

**Implication:** Per-tensor or per-token quantization of K is forced to accommodate these outliers by widening the quantization range, causing normal channels to lose precision. **Per-channel quantization** gives each channel its own range → outlier channels quantize within their range, normal channels retain full precision.

### Finding 2: Token-wise variation in V

The **Value** cache shows a fundamentally different pattern. Values within a single token's V vector are relatively uniform (no persistent per-channel outliers), but the overall magnitude varies across tokens.

**Implication:** **Per-token quantization** of V is appropriate — each token gets its own scale and zero-point.

### Finding 3: 2-bit is achievable with the right granularity

At INT2 with the asymmetric strategy (K per-channel, V per-token), KIVI achieves near-identical quality to BF16. At naive INT2 (per-tensor), quality degrades significantly.

---

## The KIVI Algorithm

### Two-phase KV quantization

**Phase 1 — Streaming residual (last `group_size` tokens at FP16):**
The last `group_size` tokens in the KV cache are kept at full FP16 precision. This is because statistics for newly computed K/V values haven't stabilized yet. Once more tokens arrive, these "residual" tokens are quantized and merged into the quantized cache.

**Phase 2 — Quantized history:**
All tokens older than the residual window are stored at INT2.

```
Cache layout:
[token 0, 1, ..., T-g-1] → INT2 quantized
[token T-g, ..., T-1]    → FP16 residual (group_size tokens)
```

### Quantization granularity

```
K quantization:
  scale = (max(K[:, channel]) - min(K[:, channel])) / (2^bits - 1)
  zero  = -min(K[:, channel]) / scale
  → one (scale, zero) per channel per head
  → group elements along head_dim dimension

V quantization:
  scale = (max(V[token, :]) - min(V[token, :])) / (2^bits - 1)
  zero  = -min(V[token, :]) / scale
  → one (scale, zero) per token per head
  → group elements along head_dim dimension for token groups
```

### Hardware-friendly implementation

- No backpropagation or gradient computation needed
- No tuning on calibration dataset
- Custom CUDA kernels for dequantization at attention time
- Compatible with existing attention implementations

---

## Results

### Memory savings

| Setting | Peak GPU Memory |
|---|---|
| BF16 KV cache | Baseline |
| KIVI-2 (2-bit) | **2.6× reduction** (including model weights) |
| Batch size increase | **up to 4× larger batch** for same memory |

### Throughput improvements

| Model | Throughput vs BF16 |
|---|---|
| LLaMA-2-7B | **2.35× higher** |
| LLaMA-2-13B | **3.47× higher** |
| Falcon-7B | **~2.5× higher** |

### Quality retention

KIVI-2 achieves "almost the same quality" as BF16 on:
- LongBench (document QA, summarization, few-shot)
- NeedleBench (long-context retrieval)
- MMLU (5-shot, multiple choice)
- HumanEval (code generation)

---

## Connection to SGLang/vLLM FP8 KV

| Aspect | KIVI | SGLang FP8 | vLLM fp8_per_token_head |
|---|---|---|---|
| K scale granularity | Per-channel | Per-tensor | Per-token-head |
| V scale granularity | Per-token | Per-tensor | Per-token-head |
| Bits | 2 | 8 | 8 |
| Memory saving | 8× vs BF16 | 2× vs BF16 | 2× vs BF16 |
| Scale source | Computed online | Calibrated checkpoint | Computed online (dynamic) |
| Tuning required | No | Yes (for best accuracy) | No |
| Production ready | Research | Yes | Yes |

vLLM's `fp8_per_token_head` is philosophically closer to KIVI (dynamic scales, no calibration) than SGLang's per-tensor FP8 (static calibrated scales). Neither implements KIVI's exact per-channel K / per-token V scheme at production scale yet.

---

## Key Takeaways for Layer 18

- **The fundamental insight**: K has channel outliers (persistent across tokens) → per-channel quant. V has token variation → per-token quant.
- Per-tensor FP8 is a **simplification** that works when the scale is calibrated — it implicitly handles the outlier problem by choosing a scale that accommodates the outlier channel.
- 2-bit quantization is feasible at production quality — but requires the right granularity. The missing piece is hardware-efficient kernels for 2-bit dequant that match FP8 speeds.
- The `group_size` parameter (how many tokens are kept as FP16 residual) is the main latency/accuracy tradeoff knob.
- KIVI-2 outperforms many token-dropping methods at the same memory budget because it preserves **all** tokens (just at lower precision).
