# 05 — Research Frontiers: Sub-FP8 and Compute Quantization

## What This Section Covers

Sections 03 and 04 covered the production state of KV cache quantization — FP8 per-tensor (SGLang) and FP8 per-token-head (vLLM). This section examines four research systems that go further: KIVI (2-bit INT K/V with asymmetric granularity), KVQuant (sub-4-bit with pre-RoPE quantization), SageAttention2 (INT4/FP8 attention compute), and ZipCache (salient-token mixed precision). These papers reveal what is fundamentally possible and where production engines will be in 2–3 years.

---

## KIVI: 2-Bit KV Cache Quantization

**Paper:** "KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache" (NeurIPS 2024, arXiv 2402.02750)
**Authors:** Zirui Liu, Jiayi Yuan, Hongye Jin, et al.
**Code:** [https://github.com/jy-yuan/KIVI](https://github.com/jy-yuan/KIVI)

### The Core Insight: K and V Need Different Granularity

Section 02 introduced the K/V distribution asymmetry. KIVI is the paper that systematically demonstrated it:

**K tensors:** Plot `K[head, token, :]` as a heatmap across token positions. You see **vertical stripes** — the same feature dimensions (channels in `head_dim`) are consistently large across all tokens. This is the channel-wise outlier structure.

**V tensors:** Plot `V[head, token, :]` the same way. You see **horizontal stripes** — different tokens have different overall magnitudes, but within each token the distribution is relatively uniform across channels. This is token-wise variation.

This asymmetry means:
- K should be quantized **per-channel** (one scale per element of `head_dim`)
- V should be quantized **per-token** (one scale per token across all channels)

Using a single per-tensor scale for both, as production engines do, is the wrong granularity for both.

### The Algorithm: Quantize and Residual

KIVI keeps a **residual buffer** of the most recent `r` tokens in full FP16, and quantizes the rest to INT2 with asymmetric per-channel (K) or per-token (V) scales:

```
Active context = [old tokens (quantized, INT2)] + [recent tokens (residual, FP16)]
                  ← quantized_len tokens →     ← r tokens (r is small, e.g. 128) →
```

When the residual buffer exceeds `r` tokens, the oldest residual tokens are quantized to INT2 and moved to the quantized portion. During attention, the full K/V is reconstructed by:

```python
K_full = dequantize(K_int2, per_channel_scale) concat K_residual_fp16
V_full = dequantize(V_int2, per_token_scale) concat V_residual_fp16
```

This handles the observation that very recent tokens (in the residual) are most likely to be attended to with high weights — keeping them at full precision reduces output error while the distant past can tolerate 2-bit quantization.

### Results

Tested on Llama-2-7B/13B/70B, Falcon-7B, Mistral-7B:
- **Memory:** 2.6× reduction in peak GPU memory (KV cache + model weights combined)
- **Throughput:** 2.35–3.47× improvement in token throughput at the same context length
- **Accuracy:** < 0.5% degradation on MMLU, LongBench, HumanEval, GSM8K vs BF16

At 2 bits with the right granularity, KIVI achieves accuracy comparable to FP8 per-tensor — demonstrating that the bottleneck in FP8 per-tensor is the granularity, not the bit width.

### What This Means for Production

KIVI requires custom CUDA kernels for:
1. INT2 grouped dequantization (per-channel for K, per-token for V)
2. Mixed-precision attention (INT2 dequant + FP16 residual concat before QKVO)

These kernels are not in SGLang or vLLM mainline. KIVI's open-source implementation patches the `transformers` library directly — it is not a drop-in for production serving stacks. Production adoption requires integrating the dequant kernels into FlashInfer or CUTLASS.

---

## KVQuant: Sub-4-Bit with Pre-RoPE Quantization

**Paper:** "KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization" (NeurIPS 2024, arXiv 2401.18079)
**Authors:** Coleman Hooper, Sehoon Kim, et al. (UC Berkeley, ICSI)

### Pre-RoPE K Quantization

Section 02 noted that RoPE scrambles K's channel structure. KVQuant quantizes K **before** RoPE is applied to it:

```
Standard flow:
  K_raw → RoPE(K_raw, pos) → K_rotated → store in KV cache (post-RoPE)

KVQuant flow:
  K_raw → quantize(K_raw, per_channel_scale) → store pre-RoPE K in KV cache
  at attention time: dequantize → apply RoPE → compute Q @ K^T
```

This preserves the channel-wise outlier structure through quantization, enabling accurate per-channel scales at sub-4-bit depths. The cost is recomputing RoPE at attention time — one rotation matrix multiply per attention call — but this is much cheaper than the dequantization of the full KV cache.

### Non-Uniform Per-Layer Datatypes

Different transformer layers have different KV activation distributions. KVQuant calibrates the optimal quantization grid per layer:

- Layers near the middle of the network (layers 10–30 in Llama-7B) have more Gaussian-like distributions → INT3 or INT4 works well
- Early and late layers have heavier tails → need FP8 or slightly more bits
- KVQuant assigns the optimal number of bits per layer based on calibration sensitivity

This is similar to mixed-precision weight quantization (e.g., GPTQ with per-layer sensitivity) applied to KV cache.

### Per-Vector Dense + Sparse Quantization

For very long contexts (> 1M tokens), even INT4 per-channel K is not enough: extremely rare outlier values (> 3σ from the mean) cause clipping. KVQuant's solution:

1. Identify outlier elements (|value| > threshold, typically 0.1% of elements)
2. Store outliers separately in a sparse FP16 tensor
3. Quantize the remaining 99.9% of values to INT3 per-vector

The attention kernel computes:
```
K_reconstructed = dequant(K_int3, per_vector_scale) + K_outlier_sparse_fp16
```

The sparse tensor is tiny (0.1% of elements) and its FP16 cost is small. The dense body is at INT3 with high accuracy.

### Results

- < 0.1 perplexity degradation on Wikitext-2 at **3 bits** (vs BF16)
- **1 million token context** on a single A100-80GB using 3-bit K + 4-bit V
- Throughput improvement: ~3× at 128K context vs BF16 (from 4× smaller KV cache)

### The Pre-RoPE Gap in Production

Both SGLang and vLLM cache **post-RoPE K**. Implementing pre-RoPE quantization requires:
1. Splitting `apply_rotary_pos_emb(q, k)` into `(q_rotated, k_pre_rope) = split(...)` — k goes to cache before rotation, k_rotated is used for the current position's contribution
2. At attention time: dequantize K_pre_rope from cache, apply RoPE at the stored positions, then compute Q @ K^T

This is a non-trivial change: the KV cache write happens in `set_kv_buffer()` but RoPE is applied in the model's attention module before the KV write. Separating them requires changing the attention module's forward pass signature. Neither SGLang nor vLLM has done this yet.

---

## SageAttention2: INT4/FP8 Attention Compute

**Paper:** "SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization" (ICML 2025, arXiv 2411.10958)
**Authors:** Jintao Zhang, Haofeng Huang, et al. (Tsinghua University)
**Code:** [https://github.com/thu-ml/SageAttention](https://github.com/thu-ml/SageAttention)

### The Distinction: Storage vs Compute Quantization

This is the critical distinction between KV cache quantization (Sections 01–04) and SageAttention2:

- **KV cache quantization** (this layer): stores K and V in reduced precision → saves memory, reduces memory bandwidth at decode time, affects what is written to and read from HBM
- **Attention compute quantization** (SageAttention2): runs the attention matrix multiplications (Q@K^T and P@V) in reduced precision → saves FLOPS on the tensor cores, does NOT affect how KV tensors are stored in HBM

Both are orthogonal: you can have BF16 KV storage with FP8 attention compute, or FP8 KV storage with BF16 attention compute, or combine both.

### The Algorithm

SageAttention2 quantizes the four operands of attention:

```
Prefill attention:
  Q → INT4 (per-thread, 16 elements per thread tile)
  K → INT4 (per-thread)
  Q_INT4 @ K_INT4^T → attn_logits (accumulated in INT32, rescaled to BF16)
  softmax(attn_logits) → P_BF16

  P_BF16 → FP8 (per-thread)
  V_BF16 → FP8 (per-thread)
  P_FP8 @ V_FP8 → output (accumulated in FP32, rescaled to BF16)
```

### Q Outlier Smoothing

Q tensors also have channel-wise outliers. Naive INT4 quantization of Q clips these outliers, causing large errors in the attention logits. SageAttention2's solution:

```
Q_smooth = Q / channel_scale   # per-channel scale to flatten outlier channels
Q_INT4 = round(Q_smooth / tile_scale)   # then quantize per thread-tile
# K_INT4 must absorb the channel_scale: K_scaled = K * channel_scale
# Q_smooth @ K_scaled^T = Q @ K^T    (mathematically equivalent)
```

This is per-channel smoothing (related to LLM.int8() and SmoothQuant for weight quantization) applied to Q activations at attention time.

### Results

- ~3× faster than FlashAttention2 on RTX 4090 (consumer Ampere GPU)
- Matches FlashAttention3 FP8 throughput on H100 Hopper with **better accuracy** (FP8 attention compute vs INT4/FP8 mixed)
- End-to-end model perplexity degradation: < 0.01 bits/char on PG19 at INT4 QK + FP8 PV

### Composability with KV Cache Quantization

The full attention quantization stack:
```
[KV stored in FP8 in HBM]
       ↓ (load from HBM)
[K_fp8, V_fp8 loaded at decode tile]
       ↓ (SageAttention2 kernel)
[K_fp8 → INT4, V_fp8 → FP8 (passed through already)] Q → INT4
       ↓
[INT4 @ INT4 → INT32 → BF16 logits → P → FP8 @ FP8 V → BF16 output]
```

Combined: FP8 KV storage (halved memory) + SageAttention2 INT4 compute (2× compute throughput) = maximum efficiency attention.

The vLLM `q_scale`/`prob_scale` additions (Section 04) are the production-in-progress version of this fully quantized compute path.

---

## ZipCache: Salient Token Mixed Precision

**Paper:** "ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification" (ECCV 2024, arXiv 2405.14256)
**Authors:** Yefei He, Luoming Zhang, et al.

### The Core Insight: Not All Tokens Are Equal for Quantization

Some tokens receive disproportionately high attention weights — "salient" tokens (often the system prompt beginning, separator tokens, and rare domain-specific tokens). Quantizing these to INT2 or INT4 causes visible output quality degradation because their attention contribution is large. Non-salient tokens can be aggressively quantized with minimal impact.

ZipCache's approach:
1. Identify salient tokens using a FlashAttention-compatible saliency metric: tokens whose normalized attention scores exceed a threshold `τ`
2. Keep salient tokens in FP16
3. Quantize non-salient tokens to INT4 with per-token scales

### Channel-Separable Tokenwise Quantization

Rather than per-vector scales (one per group of 16 elements), ZipCache uses **channel-separable tokenwise** quantization:

```
K quantization:
  - Channel scale: s_c[d] = max_tokens(|K[:, :, d]|) / 127   (calibrated once per d)
  - Token scale: s_t[token] = max_d(|K[token, :, :] / s_c|) / 127  (computed at inference)
  - K_int8[token, head, d] = round(K[token, head, d] / s_c[d] / s_t[token])

Reconstruction:
  K_bf16 = K_int8 * s_c[d] * s_t[token]
```

This is effectively INT8 with 2D factored scales — more accurate than per-token alone, but with lower scale storage overhead than per-vector.

### FlashAttention Compatibility

A challenge with attention-based token importance scores: computing them requires running attention first — a chicken-and-egg problem. ZipCache uses the attention output norm as a proxy for token importance:

```
saliency_score[token] = ||attn_output[token]||_2 / sqrt(d_model)
```

This can be computed without running the full attention softmax, and is computed on the fly during the prefill pass.

### Results

- 4.98× effective compression (mixed INT4/FP16) vs BF16
- Only 0.38% accuracy drop on Mistral-7B / GSM8K
- 56.9% decode latency reduction vs BF16 (from reduced HBM loads)
- Outperforms uniform INT4 quantization by 1.2–2.1% on reasoning tasks (saliency-aware mix preserves critical tokens)

---

## The Research → Production Gap

| Technique | Bit-width | Key innovation | Production status |
|---|---|---|---|
| FP8 per-tensor (current) | 8-bit | Simple, hardware-native | Deployed in SGLang, vLLM |
| FP8 per-token-head (current) | 8-bit | Dynamic scales | Deployed in vLLM |
| KIVI | 2-bit | Per-channel K, per-token V | Research prototype (transformers patch) |
| KVQuant | 3–4-bit | Pre-RoPE, non-uniform, sparse outliers | Research prototype (custom CUDA) |
| ZipCache | 4-bit+FP16 | Saliency-aware mixed precision | Research prototype |
| SageAttention2 | 4-bit compute | INT4 QK + FP8 PV matmuls | Library (not in serving stacks) |
| NVFP4 (vLLM) | 4-bit | Hardware-native Blackwell | WIP (Blackwell only) |

The shared obstacles preventing research techniques from entering production:
1. **CUDA kernel availability**: custom INT2/INT3 dequantization kernels are not in FlashInfer, CUTLASS, or cuBLAS; writing them requires deep GPU expertise
2. **Scale calibration infrastructure**: per-channel or pre-RoPE calibration requires changes to how quantization toolkits (llm-compressor, AutoAWQ) interact with the serving runtime
3. **Pre-RoPE cache format change**: incompatible with the existing post-RoPE cache layout used by all current attention backends
4. **Sparse tensor overhead**: managing a sparse FP16 tensor alongside a dense INT3 tensor requires changes to the paged memory allocator

---

## Summary

- **KIVI** (NeurIPS 2024): 2-bit INT KV with per-channel K + per-token V granularity; 2.6× memory, 3× throughput, near-BF16 accuracy; demonstrates the scale granularity matters more than bit-width
- **KVQuant** (NeurIPS 2024): sub-4-bit via pre-RoPE quantization of K, non-uniform per-layer dtypes, and sparse outlier handling; 1M token contexts on A100-80GB at 3-bit; pre-RoPE requires breaking the post-RoPE assumption in all current attention backends
- **SageAttention2** (ICML 2025): INT4/FP8 *compute* quantization (not storage); 3× faster attention on RTX 4090; orthogonal to KV storage quantization, composable for maximum efficiency
- **ZipCache** (ECCV 2024): salient-token-aware mixed FP16/INT4 precision; 4.98× compression with 56.9% decode latency reduction and minimal accuracy loss; FlashAttention-compatible saliency metric
- Production gap: all research techniques require custom CUDA kernels and calibration infrastructure not yet in SGLang or vLLM mainline

**Next section:** practical configuration guide — how to enable FP8 KV in SGLang, when to use calibrated vs dynamic scales, and how KV quantization interacts with HiCache and model weight quantization.
