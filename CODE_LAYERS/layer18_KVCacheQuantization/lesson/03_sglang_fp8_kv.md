# 03 — SGLang FP8 KV: The Write and Read Path

## What This Section Covers

Section 02 built the conceptual foundation. This section traces the **exact code path** SGLang follows to configure, write, and read FP8 KV tensors — from CLI flag parsing through model load, pool allocation, cache write at prefill time, and FP8-aware attention at decode time. All code anchors reference the SGLang repository.

---

## The Five-Step Pipeline

```
CLI flags
    │
    ▼  Step 1: configure_kv_cache_dtype()     → picks the FP8 dtype
    │
    ▼  Step 2: load_kv_cache_scales()         → loads k_scale / v_scale per layer
    │
    ▼  Step 3: MHATokenToKVPool allocation    → creates pool with store_dtype=uint8
    │
    ▼  Step 4: set_kv_buffer()                → quantize BF16 → FP8 → uint8, scatter
    │
    ▼  Step 5: attention backend read         → uint8 → FP8 view + descale in kernel
```

---

## Step 1: Configure KV Cache Dtype

**File:** `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:2007`

`configure_kv_cache_dtype()` maps the `--kv-cache-dtype` CLI string to a `torch.dtype`:

```python
def configure_kv_cache_dtype(self, server_args):
    kv_cache_dtype = server_args.kv_cache_dtype

    if kv_cache_dtype == "auto":
        # Check the model's config.json for kv_cache_quant_algo field
        # Set by quantization tools like llm-compressor when producing FP8 checkpoints
        quant_algo = getattr(model_config, "kv_cache_quant_algo", None)
        if quant_algo == "FP8":
            self.kv_cache_dtype = torch.float8_e4m3fn
        else:
            self.kv_cache_dtype = torch.bfloat16   # default, no quantization
    elif kv_cache_dtype == "fp8_e4m3":
        self.kv_cache_dtype = torch.float8_e4m3fn
    elif kv_cache_dtype == "fp8_e5m2":
        self.kv_cache_dtype = torch.float8_e5m2fnuz
    elif kv_cache_dtype == "fp4_e2m1":
        self.kv_cache_dtype = torch.float4_e2m1fn_x2   # experimental
    # bf16 stays as torch.bfloat16
```

The `"auto"` path is the recommended usage for FP8 checkpoints published on Hugging Face (e.g., `meta-llama/Meta-Llama-3.1-70B-Instruct-FP8`, `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`). These checkpoints set `kv_cache_quant_algo = "FP8"` in their `config.json`, so `--kv-cache-dtype auto` automatically enables FP8 KV without needing an explicit flag.

---

## Step 2: Load Per-Layer Scales

**File:** `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:1288`

After model weights are loaded (but before serving begins):

```python
if self.kv_cache_dtype in (torch.float8_e4m3fn, torch.float8_e5m2fnuz):
    if server_args.quantization_param_path:
        self.model.load_kv_cache_scales(server_args.quantization_param_path)
    else:
        logger.warning(
            "Using default scale factor 1.0 for FP8 KV cache. "
            "Accuracy may be reduced. Use --quantization-param-path for best results."
        )
```

The scale JSON file has the structure:
```json
{
  "model.layers.0.self_attn": {"k_scale": 0.0038, "v_scale": 0.0041},
  "model.layers.1.self_attn": {"k_scale": 0.0042, "v_scale": 0.0039},
  ...
}
```

`load_kv_cache_scales()` walks the named modules, finds each `RadixAttention` layer, and calls its `BaseKVCacheMethod.set_kv_scales(k_scale, v_scale)`, which writes into:

```python
layer.kv_cache_method.k_scale_float  # Python float, used in set_kv_buffer
layer.kv_cache_method.v_scale_float
```

**AMD ROCm path:** FP8 on ROCm uses the FNUZ encoding where the exponent bias differs from the IEEE encoding used by NVIDIA. SGLang doubles the loaded scales (`k_scale *= 2.0`) to compensate.

### What `BaseKVCacheMethod` is

**File:** `REPOS/sglang/python/sglang/srt/layers/quantization/kv_cache.py:16`

`BaseKVCacheMethod` is a `QuantizeMethodBase` mixin attached to each `RadixAttention` module when FP8 KV is enabled. It:
- Adds `k_scale` and `v_scale` as `nn.Parameter` tensors (shape `[1]`, initialized to -1.0 as a sentinel)
- `process_weights_after_loading()` populates `k_scale_float` / `v_scale_float` from the loaded parameter values
- Raises an error if scales are still -1.0 at runtime (never populated) to catch misconfiguration

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        layer.k_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)
        layer.v_scale = nn.Parameter(torch.tensor(-1.0), requires_grad=False)

    def process_weights_after_loading(self, layer):
        if layer.k_scale.item() == -1.0:
            layer.k_scale_float = 1.0   # default fallback (with warning already issued)
        else:
            layer.k_scale_float = layer.k_scale.item()
        # same for v_scale
```

---

## Step 3: Pool Allocation — Storing FP8 as uint8

**File:** `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:640`

`MHATokenToKVPool.__init__()` allocates the flat KV buffer:

```python
def __init__(self, size, dtype, ...):
    self.dtype = dtype   # torch.float8_e4m3fn

    # PyTorch's index_put_ is not implemented for float8 dtypes
    # Workaround: store as uint8 (same byte width), reinterpret as float8 on read
    if dtype in (torch.float8_e5m2, torch.float8_e4m3fn, torch.float8_e5m2fnuz, ...):
        self.store_dtype = torch.uint8
    else:
        self.store_dtype = dtype   # bfloat16 stores as bfloat16

    # Allocate: [num_layers, 2, pool_size, num_heads, head_dim]
    # (2 = K and V; stored contiguously for efficient DMA in HiCache)
    self.kv_buffer = torch.zeros(
        (num_layers, 2, size, num_heads, head_dim),
        dtype=self.store_dtype,
        device=device
    )
```

The pool uses `store_dtype=torch.uint8` but internally everything treats the buffer as FP8 through `.view(self.dtype)` before the attention kernel reads it. The bytes are identical — FP8 e4m3 is 8 bits, uint8 is 8 bits.

---

## Step 4: The Write Path — `set_kv_buffer()`

**File:** `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:995`

This is the core quantization operation. It runs once per layer per prefill chunk (and once per new token during decode for the new token's KV).

```python
def set_kv_buffer(
    self,
    layer: RadixAttention,
    loc: torch.Tensor,      # page indices for the new tokens
    cache_k: torch.Tensor,  # [num_new_tokens, num_heads, head_dim] in BF16
    cache_v: torch.Tensor,  # same shape, BF16
):
    if cache_k.dtype != self.dtype:
        # Input is BF16, pool expects FP8 — quantize

        k_scale = layer.kv_cache_method.k_scale_float  # Python float, e.g. 0.0038
        v_scale = layer.kv_cache_method.v_scale_float

        # 1. Scale down: BF16 value / scale → fits within ±448 (FP8 e4m3 max)
        cache_k = cache_k.to(torch.float32).div_(k_scale)
        cache_v = cache_v.to(torch.float32).div_(v_scale)

        # 2. Clamp to FP8 range (avoid NaN from overflow)
        cache_k = cache_k.clamp(-448, 448)
        cache_v = cache_v.clamp(-448, 448)

        # 3. Cast to FP8 dtype (rounds to nearest representable FP8 value)
        cache_k = cache_k.to(self.dtype)  # self.dtype = torch.float8_e4m3fn
        cache_v = cache_v.to(self.dtype)

    if self.store_dtype != self.dtype:
        # Reinterpret FP8 bits as uint8 for index_put_
        cache_k = cache_k.view(self.store_dtype)
        cache_v = cache_v.view(self.store_dtype)

    # 4. Scatter into paged pool (index_put_ works on uint8)
    self.kv_buffer[layer.layer_id, 0].index_put_([loc], cache_k)  # K
    self.kv_buffer[layer.layer_id, 1].index_put_([loc], cache_v)  # V
```

**What `div_()` does:** divides every element of the BF16 tensor by `k_scale`. For a tensor with max absolute value 1.7 and `k_scale = 1.7/448 = 0.0038`, this maps the range [−1.7, +1.7] → [−448, +448] — fully using the FP8 range without saturation.

**What `.to(float8_e4m3fn)` does:** rounds each float32 value to the nearest value representable in FP8 e4m3. Values outside ±448 saturate; values inside are rounded (3 mantissa bits of precision ≈ 12.5% relative error in the worst case, typically < 1% for values well within range).

---

## Step 5: The Read Path — FP8 Attention

**File:** `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py` — `get_key_buffer()` / `get_value_buffer()`

Reading back is simple: return the uint8 buffer view-cast back to FP8:

```python
def get_key_buffer(self, layer_id):
    buf = self.kv_buffer[layer_id, 0]   # uint8
    if self.store_dtype != self.dtype:
        buf = buf.view(self.dtype)       # reinterpret as float8_e4m3fn
    return buf

def get_value_buffer(self, layer_id):
    # same pattern for V
```

The FP8 tensor is then passed to the attention backend. The backend fuses the dequantization inside the kernel.

### FlashInfer FP8 attention path

**File:** `REPOS/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py:1680`

```python
# Paged decode attention with FP8 KV
flashinfer.BatchDecodeWithPagedKVCacheWrapper.run(
    q=q_tensor,          # BF16
    paged_kv_cache=...,  # FP8 e4m3 (from get_key_buffer / get_value_buffer)
    k_scale=layer.k_scale_float,
    v_scale=layer.v_scale_float,
    # FlashInfer internally: K_fp8 * k_scale → BF16 K, then compute attention
)
```

The dequantization `K_fp8 * k_scale` is fused into the FlashInfer CUDA kernel alongside the attention computation — no separate `torch.mul` call, no intermediate BF16 tensor allocated. This is why FP8 KV provides a **decode latency improvement**, not just a memory saving: the kernel loads 1 byte per KV value instead of 2, saturating the HBM bus less, and the scale multiply is free in the tensor core pipeline.

### TRT-LLM attention path

**File:** `REPOS/sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py:552`

The TRT-LLM backend goes further: it fuses the **write** quantization into its own kernel (`trtllm_fp8_kv_kernel.py`), bypassing the Python-level `set_kv_buffer()` entirely. The kernel receives BF16 K/V and the scales, and directly writes FP8 pages to the pool in a single CUDA kernel — more efficient than Python-level `div_()` + `.to()` + `.view()`.

---

## MLA + FP8 Path (DeepSeek Models)

Multi-head Latent Attention (MLA) models like DeepSeek store compressed KV representations (lower-rank projections) rather than full K/V tensors. The FP8 path is different:

**File:** `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:1467`

`MLATokenToKVPool` with FP8 expands each page to include per-tile FP32 scale values alongside the FP8 data:

```
Page layout:
  [fp8_kv_data | fp32_scales]
  ← quantized values → ← dequant factors →
```

The write function `nsa_kv_cache_store_fp8` runs a fused CUDA kernel that quantizes the MLA KV representation and writes the result + scales in one pass.

The read function `dequantize_k_cache_paged()` in `nsa/dequant_k_cache.py` reconstructs BF16 K from the packed FP8 + scale layout before the sparse attention kernel runs.

This MLA FP8 path has per-tile scales embedded in the cache, giving better accuracy than the global per-layer scales used in the MHA FP8 path.

---

## Code Anchor Reference

| Operation | File | Lines |
|---|---|---|
| `--kv-cache-dtype` choices | `server_args.py` | 4169 |
| `--quantization-param-path` | `server_args.py` | 4159 |
| `configure_kv_cache_dtype()` | `model_runner.py` | 2007 |
| `auto` detects `kv_cache_quant_algo` | `model_runner.py` | 2008–2035 |
| `load_kv_cache_scales()` call | `model_runner.py` | 1288 |
| Scale=1.0 warning | `model_runner.py` | 1305 |
| `BaseKVCacheMethod.create_weights()` | `kv_cache.py` | 30 |
| `process_weights_after_loading()` | `kv_cache.py` | 47 |
| `MHATokenToKVPool` store_dtype=uint8 | `memory_pool.py` | 662 |
| `set_kv_buffer()` write path | `memory_pool.py` | 995 |
| `get_key_buffer()` → fp8 view | `memory_pool.py` | 940 |
| FlashInfer FP8 decode attention | `flashinfer_backend.py` | 1680 |
| TRT-LLM fused FP8 write kernel | `trtllm_mha_backend.py` | 552 |
| Triton FP8 paged write kernel | `trtllm_fp8_kv_kernel.py` | 1 |
| MLA FP8 quant write (`nsa_kv_cache_store_fp8`) | `memory_pool.py` | 1467 |
| MLA FP8 dequant read | `nsa/dequant_k_cache.py` | 76 |

---

## Summary

- `configure_kv_cache_dtype()` maps `--kv-cache-dtype` to a torch dtype; `"auto"` reads the model's `config.json`
- Scales are loaded from a JSON file via `--quantization-param-path` and stored as float attributes on each attention layer's `BaseKVCacheMethod`
- The pool is allocated as `uint8` (not float8) due to missing PyTorch `index_put_` support for FP8 — bits are identical, only the type descriptor differs
- The write path: BF16 input → `div_(scale)` → `.to(fp8)` → `.view(uint8)` → `index_put_` into pool
- The read path: `uint8` buffer → `.view(fp8)` → attention kernel with fused dequantization via `k_scale`/`v_scale` arguments
- TRT-LLM fuses the write into a CUDA kernel (no Python-level quantization); FlashInfer fuses the dequantization into the attention kernel
- MLA (DeepSeek) uses a separate path with per-tile scales embedded in each cache page

**Next section:** how vLLM extends this model with per-token-head dynamic quantization, additional compute-path scales, and a fused CUDA write kernel.
