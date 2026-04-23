# Layer 18 — Summary: KV Cache Quantization

## The Problem in One Sentence

KV cache is stored in BF16 (2 bytes per value); at large batch sizes or long contexts it consumes more VRAM than the model weights themselves, capping throughput and context length — **quantizing KV to FP8 (1 byte) halves that cost with minimal accuracy impact**.

---

## Part 1: Why the KV Cache Dominates VRAM

Layer 12 introduced `MHATokenToKVPool`: the flat tensor buffer that holds the Key and Value tensors for every active token in the current batch. Its size is:

```
KV cache bytes = num_layers × 2 × num_kv_heads × head_dim × num_tokens × bytes_per_element
```

For a representative production model (Llama-3.1-70B, GQA-8, 80 layers, 8 KV heads, head_dim=128):

| Scenario | KV bytes @ BF16 | KV bytes @ FP8 |
|---|---|---|
| Batch=32, context=2048 | ~170 GB | ~85 GB |
| Batch=8, context=32768 | ~170 GB | ~85 GB |
| Batch=1, context=131072 | ~170 GB | ~85 GB |

In each case, 8×H100 (640 GB total) minus model weights (~140 GB FP8) leaves ≈500 GB for KV. At BF16 that fits about 3 of the above scenarios; at FP8 it fits 6.

The **decode step** compounds this: every decode step must load all KV layers for all active tokens to compute attention. This is memory-bandwidth-bound (not compute-bound). Halving the KV size halves the bytes loaded per step → decode throughput doubles for free, no more compute needed.

KV quantization and HiCache (Layer 17) are **orthogonal and composable**: HiCache moves BF16 tensors across tiers; quantization makes each tensor 2× smaller before it is placed in any tier. Combined: FP8 KV + HiCache = 2× more tokens in VRAM + 2× more tokens in CPU RAM + 2× faster PCIe transfers between tiers.

---

## Part 2: Float8 and the Scale Problem

### The two FP8 formats

Modern GPUs (H100 Hopper, H200, Blackwell) support two FP8 encodings:

```
fp8_e4m3:  sign(1) | exponent(4) | mantissa(3)   range: ±448    ← for activations
fp8_e5m2:  sign(1) | exponent(5) | mantissa(2)   range: ±57344  ← for gradients
```

`fp8_e4m3` is preferred for KV cache quantization because it has more mantissa bits (higher precision per value) at the cost of a smaller maximum value. This matters because KV activations are generally smaller than the FP8 range when properly scaled.

### Why quantization needs a scale

BF16 values cover ~1.18×10^38 range with 7 mantissa bits. FP8 e4m3 covers ±448 with 3 mantissa bits. To map BF16 KV values into the FP8 range without clipping:

```
FP8_stored = BF16_value / k_scale    (divide by scale → shrinks into ±448)
BF16_recovered = FP8_stored × k_scale  (multiply by scale at attention time → expands back)
```

The scale must accommodate the **maximum absolute value** in the tensor. A good scale = `max(|K|) / 448` (the FP8 max).

### The scale granularity problem

**Per-tensor scale** (one float per attention layer): the scale must fit the single largest value in the entire layer × all tokens × all heads. Large outliers inflate the scale → small values lose precision. This is what SGLang's default FP8 path uses.

**Per-channel scale** (one float per feature dimension): Key tensors have **channel-wise outliers** — specific feature dimensions (channels in `head_dim`) consistently produce large values across all tokens. Per-channel scales accommodate each outlier channel independently, leaving normal channels with maximum precision. KIVI (NeurIPS 2024) demonstrated this is critical for sub-4-bit quantization.

**Per-token-head scale** (one float per token per head): adapts to each token's actual activation magnitude at write time. No calibration needed. vLLM's `fp8_per_token_head` mode implements this.

```
Scale granularity (coarser → finer):
  per-tensor (one per layer) → per-channel (one per feature dim) → per-token-head (one per token×head)
Accuracy:
  lower ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← ← → higher
Memory overhead for scales:
  ~0 bytes ← ← ← ← ← ← ← ← ← ← ← ← → significant (4B per token×head)
```

---

## Part 3: SGLang's FP8 KV Implementation

### Configuration

```bash
# With calibrated per-layer scales (recommended)
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --kv-cache-dtype fp8_e4m3 \
  --quantization-param-path /path/to/kv_cache_scales.json \
  --tp 8

# Auto-detect from FP8 checkpoint (reads kv_cache_quant_algo from config.json)
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 \
  --kv-cache-dtype auto \
  --tp 8
```

### The pipeline: model load → configure → write → read

**Step 1: Determine KV dtype** — `configure_kv_cache_dtype()` (`model_runner.py:2007`):

```python
if kv_cache_dtype == "auto":
    # Check config.json for kv_cache_quant_algo == "FP8"
    self.kv_cache_dtype = torch.float8_e4m3fn
elif kv_cache_dtype == "fp8_e4m3":
    self.kv_cache_dtype = torch.float8_e4m3fn
elif kv_cache_dtype == "fp8_e5m2":
    self.kv_cache_dtype = torch.float8_e5m2
# ... etc
```

**Step 2: Load per-layer scales** — after model weights are loaded (`model_runner.py:1288`):

```python
if kv_cache_dtype == "fp8_e4m3":
    if quantization_param_path:
        model.load_kv_cache_scales(quantization_param_path)
    else:
        logger.warning("Defaulting to scaling factors of 1.0 — may reduce accuracy!")
```

The JSON file maps attention layer names to `{"k_scale": float, "v_scale": float}`. These are loaded into `layer.k_scale_float` on each `RadixAttention` module via `BaseKVCacheMethod.process_weights_after_loading()`.

**Step 3: Allocate pool as uint8** — `MHATokenToKVPool.__init__()` (`memory_pool.py:662`):

```python
if dtype in (torch.float8_e5m2, torch.float8_e4m3fn):
    # PyTorch's index_put_ is not implemented for float8 types
    self.store_dtype = torch.uint8  # store as uint8, view as fp8 when needed
else:
    self.store_dtype = dtype
```

**Step 4: Write path** — `set_kv_buffer()` (`memory_pool.py:995`):

```python
# BF16 K/V arrive from the attention projection
if cache_k.dtype != self.dtype:     # BF16 != fp8_e4m3fn → quantize
    cache_k.div_(k_scale)            # scale down to FP8 range
    cache_v.div_(v_scale)
    cache_k = cache_k.to(self.dtype) # cast BF16 → fp8_e4m3fn
    cache_v = cache_v.to(self.dtype)
if self.store_dtype != self.dtype:  # fp8 → uint8 reinterpret
    cache_k = cache_k.view(self.store_dtype)
    cache_v = cache_v.view(self.store_dtype)
# Scatter into paged KV buffer via index_put_
```

**Step 5: Read path** — the attention kernel side:

The `get_key_buffer()` call returns `uint8` tensor. The FlashInfer/TRT-LLM backends call `.view(fp8_dtype)` to recover the FP8 interpretation, then pass the scale as `bmm1_scale` — the dequantization is **fused inside the attention kernel**, not a separate step:

```python
# FlashInfer FP8 attention call (conceptual)
flash_attn(Q, key_cache.view(fp8_e4m3fn), value_cache.view(fp8_e4m3fn),
           k_scale=layer.k_scale_float, v_scale=layer.v_scale_float)
```

---

## Part 4: vLLM's Extended Quantization Model

vLLM goes further than SGLang in three key dimensions:

### 1. Per-token-head dynamic quantization

`--kv-cache-dtype fp8_per_token_head` computes one FP8 scale per (token, head) at cache-write time, **dynamically at inference**, using the actual activation values:

```
Per-tensor:       one scale per attention layer   (calibration needed for accuracy)
Per-token-head:   one scale per token × head      (no calibration, adapts to each input)
```

This adds `2 × block_size × num_kv_heads × 4 bytes` of FP32 scale storage per page — about 8% overhead for typical configs — but eliminates the calibration workflow entirely and handles diverse input distributions gracefully.

### 2. End-to-end FP8 attention (q_scale and prob_scale)

Beyond KV storage, vLLM's `BaseKVCacheMethod` adds scales for the **compute path**:

```python
layer.q_scale    # scale for Q-projection FP8 → Q×K matmul in FP8
layer.k_scale    # scale for K cache storage
layer.v_scale    # scale for V cache storage
layer.prob_scale # scale for softmax output FP8 → P×V matmul in FP8
```

This enables the attention computation itself to run in FP8 arithmetic:
```
Q_fp8 = quantize(Q, q_scale)
attn_logits = Q_fp8 @ K_fp8.T / sqrt(d)   # FP8 × FP8 matmul
P_fp8 = quantize(softmax(attn_logits), prob_scale)
output = P_fp8 @ V_fp8                     # FP8 × FP8
```

SGLang's `BaseKVCacheMethod` only has `k_scale` and `v_scale` — no Q or P compute quantization yet.

### 3. Fused CUDA write kernel

SGLang's write path is Python-level (`div_()` + `.to()` + `.view()`). vLLM fuses all of this into a single CUDA kernel `reshape_and_cache_flash` that handles scale application, FP8 cast, and scatter into the paged layout in one kernel launch — lower overhead, supports per-head scale shape `[num_heads]`.

---

## Part 5: What the Research Says

### The K/V distribution insight (KIVI, NeurIPS 2024)

KIVI ran the first systematic study of KV cache element distributions. Key findings:

- **K channels** have persistent outliers at the same feature dimensions across all tokens. A per-channel scale accommodates each channel's range without inflating the global scale.
- **V tokens** have token-wise variation. Per-token quantization is appropriate.

At 2-bit with this asymmetric strategy: **2.6× memory reduction** (including model weights), **2.35–3.47× throughput improvement**, near-identical benchmark scores vs BF16. This is the research direction that will eventually replace FP8 KV in production.

### Pre-RoPE quantization (KVQuant, NeurIPS 2024)

The rotary positional embedding (RoPE) applied to K scrambles the channel-wise outlier structure. Quantizing K **before** RoPE (pre-RoPE) gives much more stable per-channel statistics. Both SGLang and vLLM currently cache **post-RoPE K** — a known limitation that sub-FP8 quantization will need to address.

KVQuant achieves < 0.1 perplexity degradation at 3-bit on Wikitext-2 and C4, enabling **1 million token contexts on a single A100-80GB**.

### Attention compute quantization (SageAttention2, ICML 2025)

SageAttention2 quantizes the attention matmul itself: Q and K to INT4 (per-thread granularity), P̃ and V to FP8. Result: ~3× faster than FlashAttention2 on RTX 4090. This is **orthogonal** to KV storage quantization — you can combine FP8 KV storage (memory efficiency) with SageAttention2 compute (FLOP throughput) for fully quantized attention.

---

## Part 6: Choosing the Right Configuration

### Decision tree

```
Is your workload diverse (code + text, multiple languages)?
  → Yes: Use vLLM fp8_per_token_head (dynamic scales, no calibration)
  → No: Use SGLang fp8_e4m3 with calibrated scales

Do you have a calibrated FP8 checkpoint?
  → Yes: --kv-cache-dtype auto (reads kv_cache_quant_algo from config.json)
  → No, but tolerate slight accuracy risk: --kv-cache-dtype fp8_e4m3 (scale=1.0 warning)
  → No, need best accuracy: --kv-cache-dtype auto (stays BF16)

Is VRAM still the bottleneck after FP8 KV?
  → Add HiCache (Layer 17): --enable-hierarchical-cache --hicache-ratio 2
  → FP8 KV + HiCache = 2× KV compression + multi-tier capacity

Need maximum compression for research/long-context?
  → KIVI or KVQuant (custom inference stacks, not yet in production engines)
```

### What to monitor

```promql
# If using FP8 KV, watch for accuracy regression signals:
# (monitor output quality metrics via your eval pipeline, not Prometheus)

# VRAM utilization — if still high after FP8, add HiCache
# (via existing sglang:kv_used_tokens / sglang:kv_available_tokens metrics from Layer 12)

# Decode throughput — should improve with FP8 (memory-bandwidth-bound decode)
rate(sglang:num_tokens_generated_total[1m])   # tokens generated per second

# KV cache hit rate — quantization doesn't affect this, but HiCache does
sglang:cache_hit_rate
```

---

## Relationship to Adjacent Layers

| Layer | Topic | Relationship to Layer 18 |
|---|---|---|
| **Layer 12** | `RadixCache`, `MHATokenToKVPool` | Layer 18 quantizes the values written into the same pool and pool structure |
| **Layer 17** | HiCache tiered storage | Orthogonal and composable: FP8 KV × HiCache = 2× compression per tier |
| **Layer 19** | PD disaggregation (Mooncake) | KV transfer during disaggregation is easier with FP8 (smaller payloads) |

---

## What Layer 18 Defers

- **Pre-RoPE KV quantization in production** (KVQuant technique): requires splitting `apply_rotary_pos_emb()` from the cache write — a future SGLang/vLLM architectural change
- **Sub-FP8 production kernels** (KIVI, KVQuant 2-bit/3-bit): custom CUDA dequant kernels not yet merged into SGLang or vLLM mainline
- **Training-aware KV quantization (QAT)**: all methods here are post-training quantization (PTQ)
- **Blackwell NVFP4** (vLLM): hardware not yet widely deployed; WIP in vLLM FlashInfer backend
- **SageAttention2 integration in SGLang/vLLM**: currently a standalone library; production integration is an open development item
