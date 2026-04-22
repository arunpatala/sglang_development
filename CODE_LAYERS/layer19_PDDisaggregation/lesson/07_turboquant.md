# 07 — TurboQuant: Online Vector Quantization for KV Cache

## What This Section Covers

Sections 03–04 covered FP8 per-tensor and per-token-head quantization — production methods that achieve 2× KV memory reduction. Section 05 covered research methods (KIVI, KVQuant) that go further but require calibration or are not yet in serving stacks. This section covers **TurboQuant** (Google Research, ICLR 2026) — a calibration-free, theoretically-grounded method that achieves 3–5× compression and landed in vLLM mainline in April 2026. It bridges the gap between FP8 (production, 2×) and research (4–5×, not yet deployed).

---

## The Core Problem TurboQuant Solves

FP8 per-tensor quantization applies a single scale per attention layer. That scale must accommodate the largest activation value across all tokens and all channels — inflating it and compressing small values into a narrow band of representable values. KIVI showed that K tensors have channel-wise outliers and V tensors have token-wise variation, so per-tensor scale is the wrong granularity for both.

All of these approaches share a deeper problem: **they are data-dependent**. Either they require calibration data (per-tensor FP8 with `--quantization-param-path`) or they compute per-token statistics dynamically (vLLM `fp8_per_token_head`). Neither approach has a formal proof of how close they are to the theoretical best achievable distortion.

TurboQuant's contribution: a **data-oblivious** algorithm with a proof that it is within a constant factor (≈2.7×) of the Shannon information-theoretic lower bound on distortion — for any input, at any bit-width.

---

## The Algorithm: Two Stages

TurboQuant = **PolarQuant** (main compression) + **QJL residual** (inner product bias correction)

### Stage 1: TurboQuant_MSE — Optimal MSE Quantization

**Goal:** compress a d-dimensional KV vector to b bits per coordinate, minimizing mean-squared reconstruction error.

**Step 1 — Random rotation:**
```
y = Π · x
```
where `Π ∈ ℝ^{d×d}` is a random rotation matrix (generated via QR decomposition of a random Gaussian matrix). This rotation is fixed once at model load — all KV vectors for a given layer use the same `Π`.

**Why rotation helps:** After random rotation, each coordinate of `y` follows a **Beta distribution** (converges to Gaussian in high dimensions). More importantly, distinct coordinates become nearly independent. This is the key: independent coordinates can be quantized independently with scalar quantizers — no cross-channel interactions needed.

Without rotation, KV vectors have structured outliers (channel-wise for K, token-wise for V) that make scalar quantization suboptimal. Rotation spreads the outlier energy uniformly across all coordinates, making each coordinate equally easy to quantize.

**Step 2 — Lloyd-Max scalar quantization:**

The Beta distribution of each rotated coordinate is known analytically. Solve the 1D k-means problem for this distribution to find optimal centroids `c_1, ..., c_{2^b}`:

```
minimize Σ_i ∫_{bucket_i} |x - c_i|² · f_Beta(x) dx
```

This is computed once offline for each bit-width b = 1, 2, 3, 4 and stored as a fixed codebook. At inference, each coordinate of `y` is quantized to the nearest centroid index — a simple lookup.

**Step 3 — Encode:**
```python
# For each KV vector x (shape: head_dim,):
y = Pi @ x                          # rotate: (d,)
idx = nearest_centroid(y, codebook) # scalar quantize: (d,) int indices at b bits each
store(pack(idx))                    # pack b-bit indices into bytes
```

**Step 4 — Decode:**
```python
idx = unpack(stored_bytes)          # recover b-bit indices
y_hat = codebook[idx]               # centroid lookup: (d,)
x_hat = Pi.T @ y_hat               # inverse rotation (Pi is orthogonal, so Pi^{-1} = Pi^T)
```

**Distortion guarantee:**
```
MSE(TurboQuant_MSE) ≤ (√3·π/2) · (1/4^b)   for any bit-width b
Lower bound (any quantizer):         ≥ 1/4^b
```
TurboQuant is within a factor of `√3·π/2 ≈ 2.7` of the optimal — and this factor shrinks at lower bit-widths (at 1-bit it is 1.45×, essentially optimal).

---

### Stage 2: QJL — Residual Correction for Unbiased Inner Products

Attention computes `Q @ K^T`. KV cache quantization affects K — so quantization errors in K directly affect attention scores. The TurboQuant_MSE quantizer is optimal for MSE, but it introduces **bias in inner product estimation**:

```
E[⟨Q, DeQuant(Quant(K))⟩] = (2/π) · ⟨Q, K⟩   at 1-bit MSE quantization
```

The bias factor `2/π ≈ 0.637` shrinks as bit-width increases but doesn't vanish until high bit-widths.

**QJL fix:** Compute the residual `r = x - DeQuant(Quant(x))` (the reconstruction error from Stage 1). Apply a 1-bit Quantized Johnson-Lindenstrauss transform:

```
QJL(r) = sign(S · r)    where S ∈ ℝ^{d×d} is a fixed random Gaussian matrix
```

Store the 1-bit signs plus the scalar `||r||_2`. Decode:
```
x_hat_qjl = (√(π/2) / d) · ||r||_2 · S^T · sign(S·r)
```

The final estimate is `x_hat_mse + x_hat_qjl` — proven unbiased:
```
E[⟨Q, x_hat_mse + x_hat_qjl⟩] = ⟨Q, K⟩   exactly (unbiased)
```

**QJL budget:** At total budget b bits/coordinate, use (b-1) bits for Stage 1 MSE and 1 bit for Stage 2 QJL residual. So `tq3` = 2-bit MSE + 1-bit QJL, `tq4` = 3-bit MSE + 1-bit QJL.

**Important: vLLM's production implementation drops QJL.** Community testing by 5+ independent groups found that QJL amplifies variance through the softmax, hurting attention quality in practice despite the theoretical unbiasedness guarantee. All vLLM presets use Stage 1 only with **Norm Correction (NC)** instead — described in the implementation section.

---

## How TurboQuant Differs From All Prior Methods

| Property | FP8 per-tensor | KIVI (2-bit) | TurboQuant |
|---|---|---|---|
| Calibration required | Yes (or scale=1.0 risk) | No | No |
| Data-oblivious | No | No | **Yes** |
| Theoretical optimality proof | None | None | **Within 2.7× of Shannon lower bound** |
| Distribution assumption | None (uses empirical max) | K=channel-wise, V=token-wise | **Any distribution (rotation neutralizes it)** |
| Hardware requirement | FP8 silicon (H100+) | Any GPU | **Any GPU (INT storage)** |
| Compression | 2× | 4× | 3–5× |

The rotation is the key insight: it converts a hard data-dependent quantization problem (different tokens have different outlier structures) into a data-independent one (all rotated vectors have the same Beta distribution regardless of input).

---

## vLLM Implementation (PR #38479, merged April 15 2026)

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│  Store path (Triton kernels, runs on aux_stream)            │
│  K → WHT rotation → Lloyd-Max quantize → bit-pack ──┐      │
│  V → uniform quantize → bit-pack ────────────────────┤→ KV  │
│                                                      │  cache│
│  Prefill path                                        │      │
│  Q, K, V (raw) → flash_attn_varlen_func → output    │      │
│  (no KV compression needed for prefill attention)    │      │
│                                                      │      │
│  Decode path (Triton split-KV kernel)                │      │
│  KV cache → unpack → dequant → Q·K scores ──→ output│      │
└─────────────────────────────────────────────────────────────┘
```

### Key Engineering Decisions

#### WHT Instead of QR Rotation

The paper uses QR decomposition of a random Gaussian matrix to generate `Π`. vLLM replaced this with a **Walsh-Hadamard Transform (WHT) + random sign flips**:

```python
# WHT rotation: H @ (signs * x)  where signs ∈ {-1, +1}^d
# WHT is self-inverse: H @ H = d·I, so H^{-1} = H/d
# With normalization: (H/√d) is orthonormal and self-inverse
```

Why WHT instead of QR:
- WHT is an O(d log d) butterfly operation — much faster than O(d²) matrix multiply
- Self-inverse: forward and inverse rotation are the same operation (`H @ x`, then `H @ y_hat`)
- No transpose needed: `H = H^T = H^{-1}` (up to scaling)
- Enables future fused butterfly kernel inside the attention decode loop

The random sign flips (random `±1` multiplied element-wise before WHT) provide the randomization that makes the rotation data-oblivious. Without sign flips, WHT has a fixed structure and does not uniformly randomize the distribution.

#### Fused Triton Store Kernel

SGLang's FP8 path does 4 separate operations: `div_()` + `.to(fp8)` + `.view(uint8)` + `index_put_`. vLLM's TurboQuant fuses the entire store path into one Triton kernel:

```python
# _tq_fused_store_mse (Triton kernel — simplified pseudocode)
@triton.jit
def _tq_fused_store_mse(K_ptr, Cache_ptr, Pi_ptr, Centroids_ptr, ...):
    # Load K vector for this token×head
    k = tl.load(K_ptr + offsets)
    # WHT rotation + sign flip
    k = k * sign_flip_vec          # element-wise sign randomization
    k = wht_butterfly(k)           # Walsh-Hadamard butterfly (log2(d) stages)
    k = k / tl.sqrt(HEAD_DIM)      # normalize to unit sphere
    # Lloyd-Max quantize: find nearest centroid
    centroids = tl.load(Centroids_ptr + ...)  # 2^b precomputed centroids
    idx = nearest_centroid_triton(k, centroids)  # (d,) int indices
    # Bit-pack: 2 x 4-bit indices → 1 uint8 byte (for 4-bit mode)
    packed = pack_nibbles(idx)      # (d//2,) uint8
    # Store norm for decode (needed to undo rotation scale)
    norm = tl.sqrt(tl.sum(k * k))
    tl.store(Cache_ptr + key_offset, packed)
    tl.store(Cache_ptr + norm_offset, norm)
```

This eliminates 3 extra kernel launches per attention layer per token — measured as **+18–21% decode throughput** improvement vs sequential operations.

#### Value Quantization: Uniform (Not Lloyd-Max)

Keys use Lloyd-Max (optimal for Beta distribution post-rotation). Values use **uniform scalar quantization** — simpler, faster, and empirically sufficient because the value path through softmax is less sensitive to quantization noise than the key path through QK^T. The asymmetry is deliberate:

```
Keys:   WHT rotation → Lloyd-Max → 3–4 bit indices (precision matters for attention routing)
Values: uniform scalar quantize → 3–4 bit indices (weighted sum tolerates more noise)
```

#### Norm Correction (NC)

Lloyd-Max quantization introduces a small but systematic norm distortion — the reconstructed vector `x_hat` has a slightly different L2 norm than the original `x`. In the attention softmax, this manifests as scale errors in attention logits.

Fix: store the ratio `||x|| / ||x_hat||` alongside the quantized indices. At decode time, multiply the dequantized vector by this ratio before computing attention scores. This adds one FP32 scalar per token×head to storage — small overhead (~2% of slot size), ~0.8% PPL improvement at 4-bit.

NC is applied to the MSE presets (`turboquant_4bit_nc`, `turboquant_k3v4nc`, `turboquant_3bit_nc`) but not to `turboquant_k8v4` (FP8 keys don't need it — FP8 preserves norm accurately).

#### Boundary Layer Protection

The first and last few attention layers of a transformer have different activation statistics — they are adjacent to the embedding table and the output projection. Quantizing these layers aggressively degrades output quality more than middle layers.

```bash
# Auto-protect first/last 2 layers (default)
vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_k8v4

# Explicit layer skip
vllm serve Qwen/Qwen3-4B --kv-cache-dtype turboquant_k8v4 \
  --kv-cache-dtype-skip-layers 0,1,34,35
```

Skipped layers keep FP16 KV cache. This is composable with `turboquant_4bit_nc` and other presets.

#### Stream Overlap

The TQ store kernel runs on a secondary CUDA stream (`aux_stream()` from vLLM's utility layer), overlapping KV write with the next layer's forward pass computation. This hides most of the store overhead behind compute.

---

## The 4 Production Presets

All presets use WHT rotation + asymmetric K/V quantization + boundary layer protection.

| Preset | Keys | Values | Slot (bytes) | Compression | GSM8K | NIAH |
|---|---|---|---|---|---|---|
| `turboquant_k8v4` | FP8 E4M3 | 4-bit uniform | 196 | **2.6×** | 0.860 | 100% |
| `turboquant_4bit_nc` | 4-bit MSE + NC | 4-bit uniform + NC | 136 | **3.8×** | 0.840 | 100% |
| `turboquant_k3v4_nc` | 3-bit MSE + NC | 4-bit uniform + NC | 120 | **4.3×** | 0.780 | 100% |
| `turboquant_3bit_nc` | 3-bit MSE + NC | 3-bit uniform + NC | 104 | **4.9×** | 0.720 | 100% |

Baseline GSM8K: 0.900. NIAH: 100% at all presets (perfect long-context retrieval).

Note: `turboquant_k8v4` pairs FP8 keys (good precision for attention routing) with 4-bit values (sufficient for weighted sum) — this is the recommended default when quality is the priority.

### Memory Savings (Llama 3.1 8B, 128K context, per GPU)

| KV dtype | Bits/element | KV Memory | vs FP16 |
|---|---|---|---|
| FP16 | 16 | ~8 GB | 1× |
| FP8 E4M3 | 8 | ~4 GB | 2× |
| `turboquant_k8v4` | ~6 effective | ~3.1 GB | 2.6× |
| `turboquant_4bit_nc` | ~4 effective | ~2 GB | 4× |
| `turboquant_3bit_nc` | ~3.3 effective | ~1.5 GB | 5× |

---

## Performance Benchmarks

### Throughput (Qwen3-4B, 4× RTX PRO 6000 Blackwell, CUDA graphs + compile)

| Scenario | Baseline | k8v4 | % base | t4nc | % base | t3nc | % base |
|---|---|---|---|---|---|---|---|
| short-decode (128→512) | 8,977 tok/s | 7,113 | **79%** | 6,397 | 71% | 6,114 | 68% |
| long-prefill (4096→128) | 850 tok/s | 811 | **95%** | 766 | 90% | 730 | 86% |
| mixed (512→512) | 6,618 tok/s | 5,279 | **80%** | 4,829 | 73% | 4,491 | 68% |
| very-long-prefill (8192→64) | 233 tok/s | 234 | **100%** | 224 | 96% | 216 | 93% |
| decode-heavy (64→1024) | 8,304 tok/s | 6,521 | **79%** | 5,887 | 71% | 5,430 | 65% |

**Key observation:** At very long prefill (8K→64), `k8v4` matches baseline exactly — compressed KV reduces memory bandwidth pressure enough to offset the quantization overhead. TPOT (per-token decode latency) is **faster** than baseline on long sequences for k8v4 (135.2ms vs 138.1ms).

The 79–80% throughput on short-decode scenarios reflects TQ overhead being proportionally larger on small models (Qwen3-4B). Production benefit is strongest on larger models where TQ enables serving that simply doesn't fit without compression.

### A100 (SM80) Throughput Reality

On A100 with Qwen3-8B (`turboquant_3bit_nc`, vLLM 0.19):

| Concurrency | Baseline tok/s | TQ3 tok/s | Ratio | KV Memory |
|---|---|---|---|---|
| c=1 | 84.1 | 6.3 | 0.07× | 15.3 → 4.97 GiB (3×) |
| c=16 | 1,210.7 | 98.8 | 0.08× | 15.3 → 4.97 GiB (3×) |

Memory savings are confirmed (3×). Throughput collapses because the fused attention decode kernel is not yet in mainline for A100 — the current A100 path decompresses the full KV cache from packed uint8 to BF16 on every forward call before the attention matmul. **Do not use TurboQuant on A100 for throughput-sensitive workloads until the fused decode kernel ships.**

---

## Why QJL Was Dropped in Production

The paper proves QJL makes inner product estimation unbiased. In practice:

1. The QJL correction adds a stochastic component to every K vector — `S^T · sign(S·r)` has high variance per coordinate.
2. Attention softmax amplifies this variance: small errors in `Q·K^T` logits get exponentially magnified by softmax.
3. 5+ independent community groups tested QJL and found it **hurt GSM8K and reasoning benchmarks** compared to pure Stage 1 (no QJL).
4. Norm Correction (NC) addresses the remaining quality gap more reliably: it fixes the deterministic norm distortion without adding stochastic variance.

The theoretical guarantee of QJL (unbiased inner products) turns out to be less important than the practical guarantee of NC (correct attention score magnitudes). This is a recurring theme in quantization — minimizing expected error is not the same as minimizing worst-case attention disruption.

---

## Key/Value Asymmetry: Which Needs More Bits?

Community testing revealed model-architecture-dependent behavior:

**Standard transformer models (Llama, Qwen, Mistral, Phi, Gemma):**
- Keys are the precision bottleneck — they determine attention routing via softmax
- Values tolerate more quantization noise — they are weighted by softmax scores, which averages out quantization errors
- Optimal: K at higher bit-width, V at lower (K4/V3 or FP8-K/4bit-V)

**Hybrid Mamba+Attention models (Nemotron-Cascade, etc.):**
- Values are the precision bottleneck for multi-step reasoning tasks
- K at 3-bit causes no quality loss; V at 2-bit or 4-bit determines reasoning benchmark pass/fail
- Optimal: V at FP8 or 4-bit, K can be lower

The asymmetry is model-dependent. The `turboquant-vllm` plugin provides a diagnostic:
```bash
python -m turboquant_vllm.verify \
  --model meta-llama/Llama-3.1-8B \
  --k-bits 4 --v-bits 3 \
  --threshold 0.97   # minimum acceptable cosine similarity per layer
```

---

## Usage

```bash
# Best quality/throughput trade-off (FP8 keys, 4-bit values)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype turboquant_k8v4

# Balanced compression (3.8×, recommended for production)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype turboquant_4bit_nc

# Maximum compression (4.9×, research/constrained VRAM)
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --kv-cache-dtype turboquant_3bit_nc

# Skip boundary layers explicitly (first 2, last 2)
vllm serve Qwen/Qwen3-4B \
  --kv-cache-dtype turboquant_4bit_nc \
  --kv-cache-dtype-skip-layers 0,1,34,35
```

**Availability:** vLLM nightly builds post-April 15 2026. Not yet in a stable release. SGLang does not have TurboQuant as of this writing.

---

## What TurboQuant Does Not Support (Yet)

- **Hybrid Mamba+Attention models** (Qwen3.5 dense MoE, etc.) — TurboQuant requires uniform full-attention layers; boundary layer protection for hybrid architectures is a planned follow-up PR
- **SGLang integration** — only vLLM mainline
- **Fused decode on A100** — current A100 path decompresses before attention; fused tile-local decode requires further kernel work
- **QJL residual** — intentionally dropped; may be revisited for specific use-cases
- **Pre-RoPE quantization** — like FP8, TurboQuant stores post-RoPE K (the general limitation of all current KV quantization methods)

---

## Relationship to Other Methods in Layer 18

| Method | Compression | Hardware | Calibration | Status |
|---|---|---|---|---|
| FP8 per-tensor (SGLang) | 2× | H100+ | Yes (or scale=1.0) | Production |
| FP8 per-token-head (vLLM) | 2× | H100+ | No (dynamic) | Production |
| KIVI (research) | 4× | Any | No | Research only |
| KVQuant (research) | 4–5× | Any | Yes | Research only |
| **TurboQuant (vLLM)** | **3–5×** | **Any** | **No** | **vLLM nightly** |

TurboQuant fills the gap that previously existed: **calibration-free, hardware-agnostic, near-optimal compression at 3–5×** — usable on A100, A10, RTX 4090, H100, all in the same code path. The throughput benefit on non-H100 hardware depends on the fused decode kernel landing, but the memory savings are available now.

---

## Summary

- **Algorithm:** Random rotation (WHT) → Beta-distributed coordinates → Lloyd-Max scalar quantization → pack indices. Provably within 2.7× of the Shannon lower bound.
- **Two-stage:** Stage 1 (MSE optimal) + Stage 2 (QJL residual for unbiased inner products). Production drops QJL; uses Norm Correction instead.
- **vLLM (PR #38479, April 2026):** Fused Triton store kernel, split-KV Triton decode, 4 named presets (2.6–4.9× compression), boundary layer protection, stream overlap.
- **No calibration, no fine-tuning, no model changes.** One flag: `--kv-cache-dtype turboquant_k8v4`.
- **A100 works for memory savings; throughput improvement waits on fused decode kernel.**
- **Quality at 3.8× compression:** NIAH 100%, PPL +2.7% vs FP16 on Qwen2.5-1.5B.
