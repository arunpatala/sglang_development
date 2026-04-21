# 02 — Float8 and the Quantization Taxonomy

## What This Section Covers

Section 01 established why KV cache quantization is necessary. This section explains the mechanics: how float8 numbers are encoded, how they differ from int8 and bf16, and the taxonomy of choices every quantization scheme must make — precision, scale granularity, and scale source. These choices determine the accuracy/memory tradeoff and explain the architectural differences between SGLang's and vLLM's implementations.

---

## How Float8 Is Encoded

IEEE float8 uses the same structure as IEEE float16 and float32, just with fewer bits:

```
BF16  (16 bits): sign(1) | exponent(8) | mantissa(7)   → range ±3.39×10^38
FP8 e4m3 (8 bits): sign(1) | exponent(4) | mantissa(3)  → range ±448
FP8 e5m2 (8 bits): sign(1) | exponent(5) | mantissa(2)  → range ±57344
INT8  (8 bits):  sign(1) | magnitude(7)                → range ±127 (uniform steps)
```

The exponent bits determine the **dynamic range** (largest representable magnitude). The mantissa bits determine the **precision** (how many distinct values between two powers of 2).

### FP8 e4m3 vs FP8 e5m2

```
fp8_e4m3: 4 exp bits → range ±448      — more precise, narrower range
fp8_e5m2: 5 exp bits → range ±57344    — less precise, wider range
```

For KV cache storage, **`fp8_e4m3` is preferred** because:
- KV activations generally fall within ±448 when scaled correctly
- More mantissa bits (3 vs 2) means more distinct representable values → smaller rounding error
- `fp8_e5m2` was designed for gradient descent (large dynamic range needed), not for activations

### Why FP8 is better than INT8 for KV cache

INT8 quantizes linearly with uniform steps of 1/127 (for symmetric) or 1/255 (for asymmetric). This is fine when the distribution is roughly uniform. KV activations, however, have **outliers** — certain channels or tokens produce values much larger than the typical value. INT8's uniform grid wastes resolution: the step size is set by the maximum value, leaving normal values with only a few distinguishable levels.

FP8's floating-point encoding uses non-uniform spacing: values near zero are represented finely, values near the maximum are represented coarsely. This matches the rough log-normal distribution of activation magnitudes better than a uniform grid.

---

## The Three Dimensions of the Quantization Design Space

Every quantization scheme involves three independent choices:

### Dimension 1: Precision (how many bits)

```
FP8 (1 byte/value):   fp8_e4m3, fp8_e5m2
INT8 (1 byte/value):  uniform, symmetric/asymmetric
FP4 (0.5 byte/value): fp4_e2m1 (experimental)
INT4 (0.5 byte/value): KIVI, GPTQ-style
INT2 (0.25 byte/value): KIVI research
```

Fewer bits = more memory saving but larger rounding error. FP8 is the sweet spot for current hardware (H100 has native FP8 tensor cores).

### Dimension 2: Scale granularity (how many values share one scale)

The scale converts between the FP8 range (±448) and the BF16 activation range. A coarser scale must accommodate more diverse values; a finer scale adapts to local statistics.

```
Per-tensor:        one scale per (layer, K or V)
                   e.g., one k_scale for all 8 heads × all tokens × head_dim=128

Per-channel:       one scale per feature dimension
                   e.g., 128 k_scales (one per element of head_dim)
                   Adapts to outlier channels independently

Per-token:         one scale per token position (across all heads)
                   Adapts to token-wise amplitude variation

Per-token-head:    one scale per (token, head)
                   Finest production granularity; used in vLLM dynamic quantization

Per-vector (2D):   one scale per group of 16-32 consecutive elements
                   KVQuant's approach; finest granularity, highest overhead
```

Finer granularity always improves accuracy but adds memory for the scale tensors:

| Granularity | Scales per layer (Llama-70B, 2048 tokens, 8 KV heads) | Scale memory overhead |
|---|---|---|
| Per-tensor | 2 (one k_scale, one v_scale) | ~0% |
| Per-channel | 2 × 128 = 256 | ~0.1% of KV memory |
| Per-token-head | 2 × 2048 × 8 = 32,768 | ~4% of KV memory |
| Per-vector (g=16) | 2 × 2048 × 8 × 8 = 262,144 | ~32% (stored as FP32) |

### Dimension 3: Scale source (when is the scale computed)

```
Static / calibrated:  scale computed offline on a representative dataset,
                      stored in a JSON file, loaded at model startup.
                      Pros: zero runtime overhead; highest quality if calibrated well
                      Cons: requires a calibration step; scale may be wrong for OOD inputs

Dynamic:              scale computed at inference time, per batch or per token/head.
                      Pros: always accurate for the current input; no calibration needed
                      Cons: adds a kernel call or fused logic to every cache write

Fixed (scale=1.0):    no scaling. Works if activations happen to fit in ±448.
                      Pros: zero overhead
                      Cons: saturation for models with larger KV activations → accuracy risk
```

---

## The K vs V Distribution Asymmetry (Why KIVI Works)

The KIVI paper (NeurIPS 2024) ran a comprehensive study of KV activation statistics across Llama, Falcon, and Mistral models. Two critical findings:

**Finding 1: K has channel-wise outliers.**

Key tensors have specific feature dimensions (channels in `head_dim`) that are consistently 10–100× larger than the median value — across all token positions and all requests. This is not random; the same few channels are the outliers regardless of input.

```
K distribution (schematic):
Channel 3:  values mostly in [-2, +2] across all tokens
Channel 47: values mostly in [-80, +80] across all tokens  ← outlier channel
Channel 103: values mostly in [-1, +1] across all tokens
...
```

A per-tensor scale must be set to accommodate channel 47's ±80 range, which means channel 3's ±2 values are represented with a step size of 80/448 ≈ 0.18 — very coarse. With a **per-channel scale**, channel 47 gets scale 80/448 and channel 3 gets scale 2/448 = 0.004, representing both ranges maximally.

**Finding 2: V has token-wise variation.**

Value tensors do not have persistent channel outliers. Instead, different tokens have different overall magnitudes — some tokens produce large V vectors, others small. This calls for a **per-token scale** rather than per-channel.

**Implication for production engines:** both SGLang and vLLM currently use per-tensor scales (one per attention layer) — the coarsest granularity. This is the pragmatic starting point (simplest, no scale storage overhead, fastest), but it means leaving accuracy on the table. KIVI demonstrates that per-channel K + per-token V quantization achieves near-BF16 accuracy even at **2 bits** — something per-tensor FP8 cannot match.

---

## Pre-RoPE vs Post-RoPE Quantization

Rotary Positional Embedding (RoPE) rotates the Key vector by a position-dependent angle:

```
K_rotated = RoPE(K, position)    # multiply by rotation matrix
```

RoPE scrambles the channel structure of K: after rotation, a given channel index no longer corresponds to the same semantic feature dimension — the rotation has mixed channels together. The **per-channel outlier structure** of K is broken by RoPE.

Quantizing K **before** RoPE (pre-RoPE) preserves the per-channel statistics and allows much better calibration. KVQuant's paper demonstrates this achieves sub-4-bit K quantization with < 0.1 perplexity loss.

**Current production status:** both SGLang and vLLM cache **post-RoPE K**. Implementing pre-RoPE quantization requires splitting the RoPE computation and the KV cache write — a non-trivial change to the attention backend. This is an open gap between the research literature and production implementations.

---

## The Full Quantization Design Space

```
                      PER-TENSOR  PER-CHANNEL  PER-TOKEN  PER-TOKEN-HEAD  PER-VECTOR
                      ─────────────────────────────────────────────────────────────►
                                                                      (finer = better)
FP8 (1 byte)          SGLang      -            -          vLLM           -
                      static      -            -          dynamic        -

INT8 (1 byte)         -           -            -          vLLM           -
                                                           dynamic        -

FP4 (0.5 byte)        SGLang(exp) -            -          vLLM(WIP)      -

INT4 (0.5 byte)       -           KIVI K       KIVI V     -              KVQuant

INT2 (0.25 byte)      -           KIVI K       KIVI V     -              -
```

Production engines today occupy the top-left corner (FP8 per-tensor). Research has demonstrated the bottom-right quadrant (INT2-4 with per-channel/per-vector scales) is achievable with negligible accuracy loss. The gap is in production-ready CUDA kernels and integration into the inference stack.

---

## Why PyTorch Has No `float8` `index_put_`

A practical engineering note that explains a design choice in SGLang's implementation:

PyTorch's `index_put_()` (scatter operation) is not implemented for `torch.float8_e4m3fn` or `torch.float8_e5m2` tensors as of PyTorch 2.x. This is because FP8 tensor core support was added after the scatter kernel library was written, and filling the gap requires non-trivial kernel work.

SGLang's workaround (visible in `memory_pool.py:662`): allocate the KV pool with `store_dtype = torch.uint8`, cast the FP8 tensor to uint8 view for scatter writes, and re-cast to FP8 view for attention reads. The bits are identical — only the Python/CUDA type descriptor changes.

This is a temporary workaround that will be removed once upstream PyTorch adds FP8 scatter support.

---

## Summary

- FP8 uses a floating-point (non-uniform) encoding that handles activation outliers better than INT8's uniform grid
- `fp8_e4m3` is preferred for KV cache: more mantissa bits, narrower but sufficient range
- Three design dimensions: **precision** (how many bits), **scale granularity** (how many values share a scale), **scale source** (static calibrated vs dynamic)
- K tensors have channel-wise outliers → benefit from per-channel scales; V tensors have token-wise variation → benefit from per-token scales; production engines use per-tensor for simplicity
- Pre-RoPE K quantization is more principled but not yet in production; current engines cache post-RoPE K
- SGLang uses per-tensor static FP8; vLLM adds per-token-head dynamic FP8 and is moving toward NVFP4

**Next section:** how SGLang implements FP8 KV — the complete write and read path through the codebase.
