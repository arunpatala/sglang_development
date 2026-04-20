# QuantSpec: Self-Speculative Decoding with Hierarchical Quantized KV Cache

**Source:** https://arxiv.org/html/2502.10424v1
**Authors:** Rishabh Tiwari, Haocheng Xi, Aditya Tomar, Coleman Hooper, et al. (UC Berkeley, Apple, CMU)
**Venue:** ICML (Machine Learning)
**Level:** L4 — Research paper on quantized KV cache for self-speculative decoding; >90% acceptance rate; up to 2.49× speedup; hierarchical 4-bit KV design; custom CUDA kernels
**Why here:** Addresses the same long-context problem as MagicDec but from a different angle — instead of sparse KV (MagicDec), it uses **quantized KV** (4-bit). Shows that quantized KV achieves higher acceptance rates than sparse KV at similar memory budgets. The double full-precision buffer technique is a production engineering insight about how to handle KV rollback (the Layer 14 `_kv_rewind()` problem) elegantly with quantized caches.

---

## The problem QuantSpec solves

In long-context inference (4K–128K tokens), the **KV cache is the primary bottleneck** — not model weights:

- Each decode step loads the full KV cache from HBM
- KV cache size grows linearly with sequence length × batch size
- At 128K context on a single GPU, the KV cache can be 60+ GB

Traditional speculative decoding with a separate small draft model fails here because:
- Small models have poor long-context understanding → low acceptance rates
- Two KV caches (draft + target) require even more VRAM than one

**QuantSpec's solution:** Use the **target model itself** as the draft model, but with a 4-bit quantized KV cache. The target model runs twice per iteration — once with cheap 4-bit KV (draft phase) and once with full INT8 KV (verify phase).

---

## The hierarchical quantization design

The key innovation: INT8 KV cache (target quality) and INT4 KV cache (draft quality) **share the same memory** via a hierarchical decomposition.

### Mathematical foundation

Any INT8 value can be decomposed into two INT4 values:
```
C_INT8 = 2^4 × C_U_INT4 + C_L_INT4

where:
C_U_INT4 = upper 4 bits (quantized with asymmetric round-to-nearest)
C_L_INT4 = lower 4 bits (quantized from quantization error of upper 4 bits)
```

This allows:
- **Draft model** uses only `C_U_INT4` — half the memory, slightly lower precision
- **Target model** uses `C_U_INT4 + C_L_INT4` = `C_INT8` — full INT8 precision

**No separate draft KV cache needed.** The draft simply reads the upper half of each stored KV pair. This eliminates the memory duplication of traditional two-model speculative decoding.

### How quantization is applied

- **Key cache:** Quantized along the **channel axis** (per-head, per-channel group)
- **Value cache:** Quantized along the **token axis** (per-token group)
- **Group size G:** Set equal to the head dimension (reduces overhead)
- **Both use asymmetric quantization** with scale and zero-point per group

### Verification

Perplexity on Llama-2-7B shows INT8 KV is nearly identical to FP16 KV:

| KV precision | WikiText-2 PPL | C4 PPL |
|-------------|---------------|--------|
| FP16 baseline | 6.4595 | 7.2617 |
| INT8 (QuantSpec target) | 6.4696 | 7.2620 |

< 0.002 perplexity difference. Quality is preserved.

---

## The double full-precision buffer (the KV rollback solution)

This is the most engineering-relevant innovation in the paper for Layer 14 readers.

### The challenge

Quantization interacts badly with speculative decoding's KV rollback:
- Speculative decoding may reject draft tokens → those KV entries must be discarded (`_kv_rewind()` in Layer 14)
- Naively storing the most recent tokens in quantized format and rolling back requires repeated quantize/dequantize operations → expensive

Additionally:
- Per-token axis quantization requires accumulating a full group of tokens before quantizing → you can't quantize token by token efficiently

### The solution: double full-precision buffer of size 2G

Split the recent token cache into two halves:
- **CF1** (full-precision, size G): always populated; provides high acceptance rate for the most recent G tokens
- **CF2** (full-precision, size G): accumulates newly generated tokens during the current round of draft speculation

**Lifecycle:**
1. **Prefill:** Quantize all but the most recent 2G tokens into the INT4 hierarchical cache. Keep the most recent 2G in full precision (CF1 and CF2).
2. **Draft phase:** Draft model uses INT4 hierarchical KV + CF1 full-precision for recent tokens
3. **Verify phase:** Target model uses INT8 (from INT4 upper + lower) + CF1 + newly generated CF2 tokens
4. **On rejection:** Simply discard the rejected tokens from CF2 — no dequantize needed
5. **On CF2 full:** Move CF2 → CF1, quantize old CF1 → INT4 hierarchical cache, reset CF2

This design achieves:
1. **High acceptance rate** — the most recent G tokens are always in full precision (no quantization error near the decision boundary)
2. **Low overhead** — quantization happens every G steps, not every token
3. **Clean rollback** — rejected tokens are always in CF2 (full precision), so discarding them is just a pointer reset

---

## Performance results

All benchmarks at batch=1 (single-request inference):

### Llama-2-7B-32K-Instruct, PG19 dataset

| Context | Method | Acceptance Rate | Speedup vs AR |
|---------|--------|----------------|--------------|
| 4K | StreamingLLM | 88.87% | 1.13× |
| 4K | SnapKV | 93.59% | 1.17× |
| 4K | **QuantSpec** | 92.46% | **1.35×** |
| 32K | StreamingLLM | 91.19% | 1.84× |
| 32K | SnapKV | 72.54% | 1.63× |
| 32K | **QuantSpec** | 91.16% | **2.08×** |

### LWM-Text-Chat-128k, ∞BENCH Summary

| Context | Method | Acceptance Rate | Speedup vs AR |
|---------|--------|----------------|--------------|
| 128K | StreamingLLM | — | OOM |
| 128K | SnapKV | — | OOM |
| 128K | **QuantSpec** | 94.31% | **2.49×** |

**Key advantage over sparse KV:** At 128K context, sparse KV methods (StreamingLLM, SnapKV) run out of memory. QuantSpec fits because the INT4 hierarchical KV cache is only 50% of the INT8 baseline.

### Memory reduction

| Method | Memory vs full FP16 KV |
|--------|------------------------|
| QuantSpec | ~1.3× reduction (relative to baselines) |
| INT4 hierarchical vs FP16 | ~4× reduction |

---

## Custom CUDA kernels

QuantSpec implements custom CUDA attention kernels for the hierarchical quantized KV:

- **Draft attention kernel:** Loads only `C_U_INT4` (4-bit), dequantizes on-the-fly to FP16 for computation
- **Target attention kernel:** Loads both `C_U_INT4` and `C_L_INT4`, reconstructs INT8, then dequantizes to FP16
- **Speed:** Up to **2.88× faster than FP16 FlashAttention** at 4-bit precision

The CUDA kernels are the bridge from the paper to production deployment — without them, loading INT4 and dequantizing on-the-fly would be slower than just using FP16.

---

## QuantSpec vs MagicDec (comparison)

| Aspect | MagicDec (L4/07) | QuantSpec (this) |
|--------|-----------------|-----------------|
| Draft model | Target model + sparse KV | Target model + quantized KV |
| Memory savings | Sparse attention → fewer KV entries | INT4 quantization → smaller entries |
| Acceptance rate | ~90% | **>90%** (often higher) |
| OOM at 128K | Sometimes (sparse KV still large) | No (INT4 is 4× smaller) |
| Quality degradation | Yes (sparse attention misses tokens) | Minimal (INT8 target ≈ FP16) |
| CUDA kernels required | No (sparse attention is standard) | Yes (custom INT4 attention) |
| Combines with sparse KV | N/A | Yes (the paper suggests this as future work) |

Both methods address the same problem (KV cache as the long-context bottleneck) via different means. QuantSpec has better acceptance rates and handles 128K context; MagicDec has a simpler implementation.

---

## How this maps to Layer 14

| QuantSpec concept | Layer 14 equivalent |
|-----------------|---------------------|
| Hierarchical INT4/INT8 KV | Extension of `KVPool` page format from FP16 to quantized |
| `C_U_INT4` for draft / `C_INT8` for target | Extension of `DraftReq.draft_kv_pool` — shared memory with the target pool |
| Double full-precision buffer CF1/CF2 | Extension of `_kv_rewind()` — instead of freeing pages, maintain a FP buffer |
| Move CF2 → CF1 every G steps | Corresponds to when `_kv_rewind()` would be called after a verification round |
| No separate draft model | Replaces `DraftModelRunner` entirely — the target model serves as both draft and verifier |
| Custom CUDA attention kernel | Beyond Layer 14's scope — this is an L5 contribution task |

**The central contrast:** Layer 14 uses full-precision dual KV pools for draft and target. QuantSpec uses a single hierarchical quantized pool that the same model can access at two levels of precision. This is a fundamentally different memory architecture that eliminates the `kv_memory_fraction` tradeoff entirely.

---

## Limits of this paper (for book context)

- **Batch=1 only** — all experiments are single-request inference; large-batch QuantSpec behavior not evaluated
- **Accuracy impact** — INT4 quantization introduces errors; for extreme contexts (>100K), precision loss can matter
- **Custom CUDA kernels required** — not plug-and-play with standard vLLM or SGLang without kernel integration
- **Combines with sparse KV** — but the combination is left as future work; not yet implemented
- **Training-free** — QuantSpec requires no additional training; but also gets no benefit from model-specific fine-tuning (unlike EAGLE's trained draft head)
