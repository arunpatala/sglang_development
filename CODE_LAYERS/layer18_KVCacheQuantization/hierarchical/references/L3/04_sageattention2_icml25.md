# SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization

**Source:** https://arxiv.org/abs/2411.10958
**Paper PDF:** https://arxiv.org/pdf/2411.10958
**Venue:** ICML 2025
**Authors:** Jintao Zhang, Haofeng Huang, Pengle Zhang, Jia Wei, Jun Zhu, Jianfei Chen (Tsinghua University)
**GitHub:** https://github.com/thu-ml/SageAttention
**Level:** L3 — Attention compute quantization (orthogonal to KV storage quantization)
**Why here:** SageAttention2 quantizes **the attention computation itself** — not just what gets stored in the KV cache. It represents the direction vLLM is heading with `q_scale` and `prob_scale`: fully quantized attention where Q×K^T and P×V matmuls run in INT4/FP8. Understanding this distinction (compute quantization vs storage quantization) is essential for understanding where the field is heading.

**BibTeX:**
```bibtex
@inproceedings{zhang2025sageattention2,
  title={SageAttention2: Efficient Attention with Thorough Outlier Smoothing
         and Per-thread INT4 Quantization},
  author={Jintao Zhang and Haofeng Huang and Pengle Zhang and Jia Wei
          and Jun Zhu and Jianfei Chen},
  booktitle={Proceedings of the 42nd International Conference on Machine Learning (ICML)},
  year={2025},
  url={https://arxiv.org/abs/2411.10958}
}
```

---

## The Two Different Quantization Problems

It's important to separate two problems:

**Problem 1: KV Storage Quantization** (Layer 18's main topic)
```
After computing K and V: cast to FP8/INT8 and store
At attention time: reload and dequantize or use FP8-native attention
Goal: reduce GPU memory bandwidth consumption and VRAM footprint
```

**Problem 2: Attention Compute Quantization** (SageAttention2's topic)
```
While computing attention:
  Q × K^T matmul in INT4 → softmax → P × V matmul in FP8
Goal: use lower-precision CUDA tensor cores for higher FLOP throughput
```

These are **orthogonal**: you can have:
- FP16 K/V storage + FP16 compute (baseline)
- FP8 K/V storage + FP16 compute (SGLang/vLLM default FP8 path)
- FP16 K/V storage + INT4/FP8 compute (SageAttention2)
- FP8 K/V storage + INT4/FP8 compute (fully quantized attention — the future)

---

## SageAttention2 Architecture

### Matrix assignments

```
Q × K^T:  both Q and K quantized to INT4  → INT4×INT4 matmul (fastest)
P (softmax output) × V: P quantized to FP8, V in FP8  → FP8×FP8 matmul
```

### Three precision techniques

#### 1. Thread-level INT4 quantization for Q and K

Standard per-tensor or per-token INT4 causes significant accuracy loss due to outliers. SageAttention2 uses **per-thread** (warp-level) quantization:

```
Each thread block handles a small sub-matrix of Q/K
→ per-thread scale = max(|Q_thread|) or max(|K_thread|)
→ quantize only the elements in that thread's scope
→ much smaller quantization groups → finer-grained scale
```

Thread-level quantization reduces effective quantization error because each scale covers far fewer elements (typically 16–64 elements per scale) than token-level or tensor-level.

#### 2. Q smoothing for INT4 accuracy

Q activations often have outlier channels (similar to K in the KV storage domain). SageAttention2 applies a **smooth transformation** to Q before quantization:

```python
# Smooth Q by redistributing outlier magnitudes:
smooth_Q = Q / smooth_scale[channel]  # per-channel scaling
INT4_Q = quantize_int4(smooth_Q)
# At attention time, absorb smooth_scale into descale factor
```

This is analogous to KIVI's per-channel treatment of K — both recognize that channel outliers break per-token quantization.

#### 3. Two-level accumulation for FP8 PV

FP8 accumulation suffers from precision loss in long sums. SageAttention2 uses a two-level scheme:

```
Level 1: Accumulate partial P×V products in FP8 within each warp
Level 2: Accumulate warp results in FP32 before final output
```

This matches the accuracy of FP16 accumulation while using FP8 arithmetic for the majority of operations.

---

## Performance Results

### Throughput vs FlashAttention2 (OPS = operations per second)

| GPU | Method | Relative OPS |
|---|---|---|
| RTX 4090 | FlashAttention2 | 1× |
| RTX 4090 | xformers | 0.7× |
| RTX 4090 | **SageAttention2** | **~3×** |
| H100 (Hopper) | FlashAttention3 (fp8) | 1× |
| H100 (Hopper) | **SageAttention2** | **~1× (matches FA3 fp8)** |

### Accuracy

"Comprehensive experiments confirm that our approach incurs negligible end-to-end metrics loss across diverse models, including those for language, image, and video generation."

Tested on: language models (LLaMA family), image generation (DiT-based), video generation (CogVideo, Wan).

---

## Connection to vLLM's q_scale and prob_scale

vLLM's `BaseKVCacheMethod` adds `q_scale` and `prob_scale` to the attention layer:
- `q_scale`: scale factor for quantizing Q before the Q×K^T matmul
- `prob_scale`: scale factor for quantizing the softmax output P before the P×V matmul

This is exactly the SageAttention2 compute path:

```python
# vLLM conceptual flow for fully-quantized attention:
Q_fp8 = quantize(Q, q_scale)                    # using q_scale
attn_weights = Q_fp8 @ K_fp8.T / sqrt(d)        # FP8 × FP8
P = softmax(attn_weights)
P_fp8 = quantize(P, prob_scale)                  # using prob_scale
output = P_fp8 @ V_fp8                           # FP8 × FP8
```

The `prob_scale` in vLLM corresponds to SageAttention2's PV quantization. SGLang does not have these compute-path scales yet.

---

## Composability with KV Storage Quantization

SageAttention2 is **fully composable** with KV storage quantization:

```
Full pipeline:
1. Compute K, V at BF16
2. Store K, V as FP8 (storage quantization)
3. At decode step: load K_fp8, V_fp8
4. Use SageAttention2 compute path: INT4(Q) × INT4(K_fp8) → PV in FP8
5. Output in BF16
```

Memory saving (from step 2) + compute speedup (from step 4) are independent benefits.

---

## Key Takeaways for Layer 18

- **Attention compute quantization** (SageAttention2) is distinct from and composable with **KV storage quantization** (FP8 KV cache).
- vLLM's `q_scale` and `prob_scale` are the production path toward SageAttention2-style fully-quantized attention.
- SGLang lacks `q_scale` / `prob_scale` — this is a gap compared to vLLM.
- The **per-thread** (thread-level) quantization granularity in SageAttention2 is finer than per-token and enables INT4 QK matmuls without unacceptable accuracy loss.
- **3× faster than FlashAttention2** on RTX 4090 — this shows there's substantial compute capacity being left on the table with FP16 attention.
- The future of production attention: FP8 KV storage + INT4/FP8 compute quantization, with SageAttention2 or equivalent as the compute backend.
