# vLLM KV Cache Quantization: Formats, Modes, and Implementation

**Source:** https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py
         https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py
         https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
**Level:** L2 — Deployment reference
**Why here:** vLLM's KV cache quantization implementation is more complete than SGLang's in several key dimensions (per-token-head dynamic quantization, TurboQuant, NVFP4, q_scale/prob_scale). Reading vLLM's approach alongside SGLang's reveals the current state of the art in production KV quantization and what SGLang is likely to adopt next.

---

## Supported KV Cache Formats

```python
CacheDType = Literal[
    "auto",                  # Use model dtype (BF16 by default)
    "float16",               # FP16 storage
    "bfloat16",              # BF16 storage (same as auto for most models)
    "fp8",                   # FP8 generic (maps to e4m3 on CUDA)
    "fp8_e4m3",              # FP8 E4M3: 1 byte, range ±448 — best accuracy
    "fp8_e5m2",              # FP8 E5M2: 1 byte, range ±57344 — more range, less mantissa
    "fp8_inc",               # FP8 via llm-compressor integration
    "fp8_ds_mla",            # DeepSeek-style FP8 MLA packed layout
    "turboquant_k8v4",       # TurboQuant: K=INT8, V=INT4
    "turboquant_4bit_nc",    # TurboQuant: 4-bit non-contiguous
    "turboquant_k3v4_nc",    # TurboQuant: K=3-bit, V=4-bit NC
    "turboquant_3bit_nc",    # TurboQuant: 3-bit non-contiguous
    "int8_per_token_head",   # INT8 with dynamic per-(token,head) scales
    "fp8_per_token_head",    # FP8 with dynamic per-(token,head) scales
    "nvfp4",                 # NVFP4: packed FP4+FP8 block scales (Blackwell only)
]
```

---

## The `KVQuantMode` Enum: A Better Design

vLLM replaced ad-hoc string matching with a typed enum, making backend dispatch cleaner:

```python
class KVQuantMode(IntEnum):
    NONE = 0               # no quantization
    FP8_PER_TENSOR = 1     # per-tensor scales; fp8_e4m3, fp8_e5m2, fp8
    INT8_PER_TOKEN_HEAD = 2  # dynamic per-(token,head) for int8
    FP8_PER_TOKEN_HEAD = 3   # dynamic per-(token,head) for fp8
    NVFP4 = 4              # packed fp4 + fp8 block scales
```

SGLang uses string comparisons throughout; vLLM's enum is cleaner and easier to extend.

---

## Three Scale Modes Explained

### 1. Per-tensor scales (SGLang default, vLLM fp8_e4m3)

- **One scale** per attention layer for K, one for V.
- Source: calibrated from checkpoint JSON (via llm-compressor) or default 1.0.
- **Pros:** Simple, fast, low overhead.
- **Cons:** A single scale can't accommodate per-channel or per-token outliers. Scale=1.0 without calibration can hurt accuracy significantly.

```python
# SGLang: set_kv_buffer()
cache_k.div_(k_scale)       # scale down: BF16 → FP8 range
cache_k = cache_k.to(fp8)  # cast
cache_k = cache_k.view(uint8)  # store as uint8 (index_put workaround)
```

### 2. Per-token-head dynamic scales (vLLM fp8_per_token_head, int8_per_token_head)

- **One scale per (token, head)** computed dynamically at cache-write time.
- **No calibration needed** — scales adapt to each token's actual activation range.
- Fused in the CUDA `reshape_and_cache` kernel.
- **Extra memory cost:** `2 × block_size × num_kv_heads × 4 bytes` per page for FP32 scales.
- **Pros:** Best accuracy for FP8; eliminates the need for `--quantization-param-path`.
- **Cons:** ~8% memory overhead for scale tensors; slightly more complex kernel.

### 3. NVFP4 (vLLM, Blackwell only)

- Packed 4-bit values + FP8 block scales.
- 0.5 bytes/value (2× savings over FP8).
- Requires SM100 (NVIDIA Blackwell B100/B200) — hardware-compiled dispatch.
- API is partially WIP (FlashInfer builder raises NotImplementedError in one path).

---

## Extended Scales: q_scale and prob_scale

vLLM's `BaseKVCacheMethod` adds two extra scales beyond K and V:

```python
layer.q_scale = torch.nn.Parameter(...)    # for Q-projection FP8 quantization
layer.k_scale = torch.nn.Parameter(...)    # for K cache storage
layer.v_scale = torch.nn.Parameter(...)    # for V cache storage
layer.prob_scale = torch.nn.Parameter(...) # for softmax(QK^T) output
```

This enables **end-to-end FP8 attention**:
1. Q quantized to FP8 before matmul with K → `QK^T` computed in FP8
2. Softmax output (P) quantized to FP8 before matmul with V → `PV` computed in FP8
3. K and V stored in FP8 in the cache

SGLang only has `k_scale` and `v_scale` — no compute-path quantization yet.

---

## The CUDA Kernel: Fused Scale + Cast + Scatter

vLLM's write path is a fused CUDA kernel, not Python-level operations:

```c
void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,    // [1] for per-tensor, [num_heads] for per-head
    torch::Tensor& v_scale)
```

Key detail: `k_scale` can be shape `[1]` (per-tensor) **or** `[num_heads]` (per-head) — the `kv_scale_stride` variable dispatches appropriately. This is cleaner than SGLang's Python `div_()`.

---

## TurboQuant: K8V4 and Sub-4-bit (vLLM only)

TurboQuant (NVIDIA Research) stores K and V at different precisions:
- `turboquant_k8v4`: K in INT8, V in INT4
- `turboquant_3bit_nc`: both K and V in 3-bit non-contiguous layout

**Automatic boundary layer skipping**: the first 2 and last 2 attention layers are automatically forced to `auto` (BF16) because attention statistics at these boundary layers have different distributions that don't quantize well.

This is not yet in SGLang.

---

## The Sleep Mode Bug (Known Issue)

When a vLLM GPU goes idle and then wakes up, calibrated scales are **reset to 1.0**:

```python
def init_fp8_kv_scales(self) -> None:
    # TODO: restore calibrated scales here in the future
    k_scale_val, v_scale_val = 1.0, 1.0
    # ... fills all attention layers with 1.0
```

If you deployed with `--quantization-param-path` and the GPU sleeps, the scales are lost until restart. This is a production-correctness risk.

---

## Key Takeaways for Layer 18

- vLLM supports **far more KV quantization formats** than SGLang (13 vs 5 options).
- **Per-token-head dynamic quantization** (vLLM) is the most practical improvement: no calibration needed, better accuracy than per-tensor FP8.
- **q_scale/prob_scale** (vLLM) is the path toward fully quantized attention compute (not just KV storage).
- The fused CUDA kernel (vLLM) is more efficient than Python-level div+cast+view (SGLang).
- SGLang's `--kv-cache-dtype fp4_e2m1` and vLLM's `nvfp4` are competing FP4 approaches — neither is production-ready yet.
- The sleep-mode scale reset bug in vLLM is a reminder that quantization requires careful lifecycle management.
