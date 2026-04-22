# INT8 / FP8 KV Cache on A100 — Analysis and Implementation Path

A deep-dive into KV cache quantization for NVIDIA A100 GPUs (SM 8.0), which lack native FP8 hardware support.

---

## Current SGLang KV Cache Dtype Support

The valid `--kv-cache-dtype` values in `srt/model_executor/model_runner.py::configure_kv_cache_dtype`:

```
auto        → BF16 (or FP8 if checkpoint has kv_cache_quant_algo=FP8)
fp8_e5m2    → torch.float8_e5m2
fp8_e4m3    → torch.float8_e4m3fn
bf16/bfloat16
fp4_e2m1    → torch.float4_e2m1fn_x2  (Blackwell only)
```

**INT8 is not supported.** Passing `--kv-cache-dtype int8` raises `ValueError: Unsupported kv_cache_dtype`.

FP8 KV cache with the Triton attention backend also does not work on A100 because Triton emits `cvt.f32.e4m3` PTX instructions that only exist on SM 8.9+ (Ada) and SM 9.0+ (Hopper).

---

## Why A100 Still Matters

A100 (SM 8.0) is still the dominant production GPU across AWS, Azure, GCP, and on-prem clusters. H100 costs 3–4x more and has limited availability. Llama 3, Qwen 3, and DeepSeek V3 all run on A100 clusters at scale.

---

## Why KV Cache Quantization Helps on A100

Decode is **memory-bandwidth bound** — the bottleneck is loading K and V from HBM, not the attention math itself.

```
BF16 KV cache:  2 bytes/element  → load 2 bytes, compute in BF16
INT8 KV cache:  1 byte/element   → load 1 byte, dequant inline, compute in float32
```

If dequantization is **fused inside the attention kernel**, HBM traffic halves with near-zero arithmetic overhead. Expected decode throughput improvement: **1.5–1.8x** at large batch sizes.

If dequantization is **unfused** (a separate kernel pass), HBM traffic doubles and performance is worse than BF16 — this was why a prior SGLang INT8 PR was abandoned.

---

## Where the Key Code Lives

### Write path (quantize on write)
`srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py` — `_process_kv_tensor`

```python
# Current FP8 write:
block_fp8 = (block * inv_scale).to(tl.float8e4nv)

# INT8 equivalent:
block_int8 = (block * inv_scale).round().clamp(-128, 127).to(tl.int8)
```

The write kernel is a ~1-line change per dtype.

### Read path (dequantize on read)
`srt/layers/attention/triton_ops/decode_attention.py` — `_fwd_kernel_stage1`

```python
k = tl.load(K_Buffer + offs_buf_k, ...)
qk = tl.sum(q[None, :] * k, 1)
qk *= sm_scale_withk   # sm_scale_withk = sm_scale * k_scale (scale pre-folded in)
```

`sm_scale_withk` already multiplies in `k_scale` (line ~763 in `triton_backend.py`). Triton auto-upcasts INT8 to float32 in mixed arithmetic. So the read kernel requires **no change** for INT8 — it just works if the buffer dtype is `torch.int8`.

For V, `v_scale` is applied at stage2's output:
```python
tl.store(O + ..., acc / e_sum * v_scale, ...)
```
Same pattern — works for INT8 as-is.

### Dtype dispatch
`srt/model_executor/model_runner.py` — `configure_kv_cache_dtype`

Add one `elif` branch:
```python
elif self.server_args.kv_cache_dtype == "int8":
    self.kv_cache_dtype = torch.int8
```

### Memory pool
`srt/mem_cache/memory_pool.py` — allocate cache as `torch.int8` instead of `torch.uint8`.

---

## Calibration: INT8 vs FP8

### FP8 — calibration is optional

FP8 e4m3fn has range ±448. Typical KV values after normalization are ±50 at most, so a direct cast works fine without a scale:

```python
k.to(torch.float8_e4m3fn)   # no scale needed — values fit in FP8 range
```

SGLang supports `use_provided_scale=False` (no scale) for FP8 for this reason.

### INT8 — calibration is mandatory

INT8 has uniform spacing of 1.0 over range [-128, 127]. It has **no fractional representation** — values like `0.07` map to `0` (100% error) without a scale.

You must compute a scale:
```python
scale = k.abs().max() / 127.0
k_int8 = (k / scale).round().clamp(-128, 127).to(torch.int8)
# dequant: k_int8.float() * scale
```

### Dynamic vs. static calibration

| Strategy | When | Overhead | Accuracy |
|---|---|---|---|
| Static (offline) | Pre-calibrated on sample data | Zero at inference | Degrades if KV distribution shifts |
| Dynamic (per-token) | Inline: `max(abs(K_token)) / 127` | ~1% (fusable into write kernel) | Good |

INT8 KV cache implementations (KIVI, etc.) use dynamic per-token or per-head scales. The scale computation is one `tl.max(tl.abs(k))` reduction fusable into the write kernel — not a bottleneck.

---

## The "Store FP8 Bits as uint8/int8" Option

FP8 and INT8 are both 8-bit. You can store FP8 bit patterns in uint8/int8 storage (a bitcast). **SGLang already does this for FP8** — the cache is `torch.uint8` with FP8 bits inside:

```python
# memory_pool.py:
self.store_dtype = torch.uint8  # because index_put doesn't support float8_e5m2

# naive fallback:
if store_dtype == torch.uint8:
    dtype = torch.float8_e4m3fn   # logical dtype
k = k.view(torch.uint8)           # bitcast, not numeric cast
```

This approach gives FP8-quality precision (better than INT8 — non-uniform spacing handles small values) at the same 8-bit storage. No per-token scale computation needed.

### Why this still doesn't work on A100

The blocker is in the Triton **read kernel**. When it loads `uint8`, it gets integer values (0–255), not FP8 floats. The required pointer-type bitcast in Triton:

```python
# needs tl.float8e4nv or similar FP8 pointer type
k_fp8_ptr = K_Buffer.to(tl.pointer_type(tl.float8e4nv))
k = tl.load(k_fp8_ptr + offs)
```

...causes Triton to emit `cvt.f32.e4m3` PTX instructions, which **only exist on SM 8.9+ (Ada) and SM 9.0+ (Hopper)**. On A100 (SM 8.0), this crashes. This was the exact bug in TurboQuant's early Triton kernel.

### Correct approach for A100: manual FP8 bit decoding

To decode FP8 e4m3fn bits stored in uint8 on A100, you'd manually extract the bit fields in Triton:

```python
k_u8 = tl.load(K_Buffer + offs_buf_k, ...)   # uint8 (raw FP8 bits)
sign = (k_u8 >> 7) & 1                         # bit 7
exp  = (k_u8 >> 3) & 0xF                       # bits 6–3 (biased exponent)
mant = k_u8 & 0x7                              # bits 2–0 (mantissa)
# reconstruct float32 from sign/exp/mant fields
# ... ~15–25 lines of integer arithmetic
```

This works on any GPU (pure integer instructions), adds negligible overhead in a memory-bandwidth-bound kernel, and gives correct FP8 semantics.

---

## Accuracy

INT8 with dynamic per-token scales is fine for most models (KIVI paper):

- **INT8 per-head dynamic**: negligible accuracy loss across Llama, Mistral, Falcon
- **INT4 per-head**: small degradation, acceptable for many tasks
- **INT4 per-tensor**: meaningful quality loss

FP8 bitcast (no scale, direct cast) also works well because the non-uniform encoding naturally handles fractional values near zero.

---

## Implementation Summary

### True INT8 quantization (simpler, A100-ready)

| Change | File | Effort |
|---|---|---|
| Add `int8` branch in dtype dispatch | `model_runner.py` | 3 lines |
| Allocate cache as `torch.int8` | `memory_pool.py` | 5 lines |
| Change `tl.float8e4nv` → `tl.int8` in write kernel | `trtllm_fp8_kv_kernel.py` | 1 line |
| Add dynamic scale computation (fused) | write kernel | ~30 lines |
| Read kernel | `decode_attention.py` | **0 changes** |
| `server_args.py` validation | add `"int8"` | 5 lines |
| **Total** | | **~150–200 lines** |

### FP8 bits in uint8 (better precision, more Triton complexity)

| Change | File | Effort |
|---|---|---|
| Write path | Already works | 0 lines |
| Memory pool dtype | Already uint8 | 0 lines |
| FP8 bit decoder in read kernel (A100-safe) | `decode_attention.py` | ~25 lines |
| Dtype dispatch + validation | `model_runner.py`, `server_args.py` | 10 lines |
| **Total** | | **~35 lines** |

The FP8-bits-in-uint8 approach is less code, gives better precision, and needs no per-token scale — but requires the manual bit decoder in the Triton attention kernel to work on A100.

---

## Key Files

```
REPOS/sglang/python/sglang/srt/
├── model_executor/
│   └── model_runner.py              # configure_kv_cache_dtype()
├── mem_cache/
│   └── memory_pool.py              # cache tensor allocation
├── layers/
│   ├── quantization/
│   │   └── kv_cache.py             # BaseKVCacheMethod (k_scale/v_scale)
│   └── attention/
│       ├── triton_backend.py       # dispatch: sm_scale * k_scale folding
│       └── triton_ops/
│           ├── decode_attention.py # _fwd_kernel_stage1 (read path)
│           └── trtllm_fp8_kv_kernel.py  # _process_kv_tensor (write path)
```
