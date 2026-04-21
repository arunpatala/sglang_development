# ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification

**Source:** https://arxiv.org/abs/2405.14256
**Paper PDF:** https://arxiv.org/pdf/2405.14256
**Venue:** arXiv May 2024; ECCV 2024
**Authors:** Yefei He, Luoming Zhang, Weijia Wu, Jing Liu, Hong Zhou, Bohan Zhuang
**Level:** L3 — Mixed-precision quantization; salient token selection
**Why here:** ZipCache bridges KV quantization and token dropping: it identifies "salient" tokens that are critical for attention accuracy (keep at full precision) and aggressively quantizes the rest. This mixed-precision approach represents the frontier between pure quantization (all tokens same bits) and pure token dropping (remove less important tokens entirely). Its FlashAttention-compatible saliency metric is particularly practical.

**BibTeX:**
```bibtex
@article{he2024zipcache,
  title={ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification},
  author={Yefei He and Luoming Zhang and Weijia Wu and Jing Liu and Hong Zhou and Bohan Zhuang},
  journal={arXiv preprint arXiv:2405.14256},
  year={2024},
  url={https://arxiv.org/abs/2405.14256}
}
```

---

## Problem

Existing KV cache quantization methods quantize **all tokens uniformly** at the same precision. But in practice:
- Some tokens receive much higher attention weight than others ("salient" tokens)
- Quantizing salient tokens aggressively causes large accuracy degradation
- Non-salient tokens can be quantized heavily with minimal impact

Prior methods that try to identify salient tokens (for keeping at full precision) have **inaccurate saliency metrics** that fail at high compression ratios because they require computing full attention scores — which is incompatible with FlashAttention.

---

## ZipCache's Two Main Contributions

### Contribution 1: Channel-Separable Tokenwise Quantization

A memory-efficient quantization scheme that reduces the overhead of quantization parameters compared to fine-grained groupwise quantization:

```
Standard groupwise quantization:
  → one (scale, zero) per group of G elements
  → overhead: 2 × (total_elements / G) parameters

Channel-separable tokenwise:
  → separate scale per (token × channel) = per-element-of-outer-product
  → but uses separable channel and token scales: scale_total = scale_channel × scale_token
  → overhead: num_tokens + head_dim parameters (vs num_tokens × head_dim / G for groupwise)
```

For typical sizes (head_dim=128, seq_len=8192, G=64):
- Groupwise overhead: 2 × 8192 × 128 / 64 = 32,768 parameters
- Channel-separable: 8192 + 128 = 8,320 parameters

**~4× reduction in scale storage overhead** while maintaining per-element accuracy.

### Contribution 2: Normalized Attention Score as Saliency Metric

**The key insight:** An attention token is "salient" if its attention weight is significantly higher than the mean. ZipCache uses **normalized attention scores** that account for the causal mask structure:

```python
# Standard attention score:
score[i, j] = softmax(Q_i × K_j^T / sqrt(d))

# Normalized score (ZipCache):
# The lower-triangular structure of causal attention means each token
# has a different number of "competitors" for attention.
# Normalize by the expected attention if uniform:
normalized_score[i, j] = score[i, j] × (i + 1)  # adjust for position
```

This normalization makes saliency scores **comparable across token positions** — early tokens naturally receive more attention due to causal masking, so without normalization they appear falsely "salient."

### FlashAttention compatibility

The saliency metric is **decoupled from the full attention scores** — it can be approximated using a small prefix of the attention matrix, not requiring the full `O(seq_len²)` computation. This makes it compatible with memory-efficient FlashAttention implementations.

---

## Mixed-Precision Cache Layout

```
For each token, compute normalized attention score:
  if score > threshold (salient):
    keep token in FP16 (full precision)
  else (non-salient):
    quantize token aggressively (INT2 or INT4)
```

The threshold is adaptive — it keeps approximately `k%` of tokens at full precision, where `k` is a user-chosen budget.

---

## Results

### On Mistral-7B, GSM8k dataset

| Compression ratio | Accuracy drop |
|---|---|
| **4.98×** compression | **0.38%** accuracy drop |

### On LLaMA3-8B, input length 4096

| Metric | ZipCache vs baseline |
|---|---|
| Prefill latency | **37.3% reduction** |
| Decode latency | **56.9% reduction** |
| GPU memory | **19.8% reduction** |

The decode latency reduction is large because fewer bytes need to be loaded per decode step — smaller cache → faster memory bandwidth → faster decode.

---

## Comparison with Uniform Quantization

| Method | All tokens same bits? | Saliency-aware? | FlashAttn compatible? |
|---|---|---|---|
| FP8 KV (SGLang/vLLM) | Yes (FP8) | No | Yes |
| KIVI | Yes (INT2) | No | Partially |
| KVQuant | Yes (INT3) | No (per-layer) | No (custom kernel) |
| **ZipCache** | No (mixed) | **Yes** | **Yes** |
| H2O / StreamingLLM | No (drop) | Yes (drop salient's history) | Yes |

ZipCache occupies a unique position: it's not "drop everything below threshold" nor "quantize everything to N bits" — it's "quantize non-salient aggressively, keep salient full-precision."

---

## Connection to SGLang/vLLM

**ZipCache is not implemented in SGLang or vLLM** — it's a research system. But its ideas influence production:

1. The **normalized saliency score** idea could be applied within SGLang's scheduler to decide which pages to write-through to HiCache's L3 tier vs evict — a cross-layer optimization between KV quantization and tiered storage.

2. The **channel-separable tokenwise quantization** reduces scale overhead — relevant for vLLM's `fp8_per_token_head` mode where scale tensors add meaningful memory overhead.

3. The **FlashAttention-compatible saliency** approach shows that mixed-precision quantization doesn't have to sacrifice attention backend compatibility.

---

## Key Takeaways for Layer 18

- **Mixed-precision quantization** (salient=FP16, non-salient=INT2) achieves better accuracy/compression tradeoff than uniform quantization.
- The **normalized attention score** is a better saliency metric than raw attention weights — accounts for causal mask structure.
- **56.9% decode latency reduction** shows that decode is highly memory-bandwidth bound — KV quantization directly reduces decode latency by reducing bytes loaded per step.
- ZipCache's mixed approach is more complex to implement but potentially more powerful than uniform FP8 quantization for the same memory budget.
- The key gap: production systems don't yet implement token-level adaptive quantization granularity.
