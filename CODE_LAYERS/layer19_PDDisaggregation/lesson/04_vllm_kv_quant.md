# 04 — vLLM's Extended Quantization Model

## What This Section Covers

SGLang's FP8 KV implementation (Section 03) covers the essential write/read path with per-tensor static scales. vLLM's implementation goes further in several meaningful directions: a richer type system for quantization modes, per-token-head dynamic scaling (no calibration needed), end-to-end FP8 attention compute paths, a fused CUDA write kernel, and additional hardware-specific formats like NVFP4 and TurboQuant. Understanding vLLM's design choices helps clarify the tradeoffs in SGLang's simpler model and reveals where the field is heading.

All code anchors in this section reference the vLLM repository.

---

## The Type System: `CacheDType` and `KVQuantMode`

### `CacheDType` — 13 supported formats

**File:** `REPOS/vllm/vllm/config/cache.py:18`

vLLM's public API exposes 13 KV cache dtype strings vs SGLang's 5:

```python
CacheDType = Literal[
    "auto",
    "fp8",                   # alias for fp8_e4m3
    "fp8_e4m3",              # FP8, per-tensor static (same as SGLang default)
    "fp8_e5m2",
    "fp8_per_token_head",    # FP8, dynamic scale per (token, head) ← NEW
    "int8_per_token_head",   # INT8, dynamic scale per (token, head) ← NEW
    "nvfp4",                 # NVIDIA FP4, Blackwell (SM100) hardware only ← NEW
    "turboquant_int8",       # TurboQuant INT8 KV ← NEW
    "turboquant_k8v4",       # TurboQuant K=INT8, V=INT4 ← NEW
    "turboquant_3bit_nc",    # TurboQuant 3-bit ← NEW
    "fp8_ds_mla",            # DeepSeek MLA-specific FP8 ← NEW
    "bf16",
    "fp16",
]
```

The seven formats marked NEW do not exist in SGLang. This is partly because vLLM has more NVIDIA hardware partners contributing upstream (TurboQuant is from NVIDIA Research; NVFP4 requires Blackwell B100/B200), and partly because vLLM's modular `KVQuantMode` dispatch makes adding new formats easier.

### `KVQuantMode` — centralized dispatch enum

**File:** `REPOS/vllm/vllm/v1/kv_cache_interface.py:30`

Rather than string-comparing `cache_dtype` throughout the codebase (SGLang's approach), vLLM introduced `KVQuantMode` as an internal enum:

```python
class KVQuantMode(Enum):
    NONE = "none"                    # BF16/FP16
    FP8_PER_TENSOR = "fp8"          # SGLang-equivalent
    INT8_PER_TOKEN_HEAD = "int8_pth" # dynamic per-token-head INT8
    FP8_PER_TOKEN_HEAD = "fp8_pth"  # dynamic per-token-head FP8
    NVFP4 = "nvfp4"                 # Blackwell packed 4-bit
```

All dispatch logic in the cache interface, worker, and attention backend checks `mode == KVQuantMode.X` rather than parsing strings. This makes adding new modes a two-step change: add the enum value, wire it to the backend.

---

## Per-Token-Head Dynamic Quantization

### The concept

In SGLang's per-tensor approach, one `k_scale` is shared across all tokens, all heads, and all elements in `head_dim`. This scale must be large enough to accommodate the worst-case token × head combination, leaving typical values quantized coarsely.

vLLM's `fp8_per_token_head` and `int8_per_token_head` modes compute one scale per **(token, head)** at cache-write time:

```
scale[token_i, head_j] = max(|K[token_i, head_j, :]|) / FP8_MAX
```

This scale is exact for every (token, head) pair — no calibration dataset needed, no risk of OOD inputs violating the scale. The tradeoff is storing these scales alongside the KV data.

### Scale storage overhead

**File:** `REPOS/vllm/vllm/v1/kv_cache_interface.py:132`

`page_size_bytes` for per-token-head modes includes scale storage:

```python
if mode in (KVQuantMode.FP8_PER_TOKEN_HEAD, KVQuantMode.INT8_PER_TOKEN_HEAD):
    # Each page has block_size tokens, num_kv_heads heads
    # Each scale is one float32 (4 bytes), stored for K and V separately
    scale_bytes = 2 * block_size * num_kv_heads * 4   # float32 per (token, head)
    kv_data_bytes = 2 * block_size * num_kv_heads * head_dim * 1  # 1 byte/element
    page_size = kv_data_bytes + scale_bytes
```

For typical values (block_size=16, num_kv_heads=8, head_dim=128):
- KV data: 2 × 16 × 8 × 128 × 1 byte = 32,768 bytes
- Scale storage: 2 × 16 × 8 × 4 bytes = 1,024 bytes
- Overhead: 1024/32768 ≈ **3.1%** of KV memory

This 3.1% scale overhead buys you: no calibration workflow, accurate quantization for any input distribution, and better accuracy on diverse workloads (multilingual, code+text, multi-domain).

### When to choose per-token-head over per-tensor

| Scenario | Recommended |
|---|---|
| Production with calibrated FP8 checkpoint | per-tensor (SGLang or vLLM) — simpler, no overhead |
| Diverse inputs (code, multiple languages, domain shift) | per-token-head — adapts dynamically |
| No calibration dataset available | per-token-head — no offline step needed |
| Maximum throughput, VRAM-constrained (long ctx) | per-tensor — zero scale overhead |

---

## Extended Scales: `q_scale` and `prob_scale`

### Beyond KV storage — end-to-end FP8 attention

**File:** `REPOS/vllm/vllm/model_executor/layers/quantization/kv_cache.py:18`

SGLang's `BaseKVCacheMethod` stores `k_scale` and `v_scale`. vLLM's version adds two more:

```python
class BaseKVCacheMethod(QuantizeMethodBase):
    def create_weights(self, layer):
        layer.k_scale = nn.Parameter(...)   # scale for K cache storage
        layer.v_scale = nn.Parameter(...)   # scale for V cache storage
        layer.q_scale = nn.Parameter(...)   # scale for Q-projection output → FP8 matmul
        layer.prob_scale = nn.Parameter(...) # scale for softmax output P → FP8 matmul
```

The `q_scale` and `prob_scale` enable the **attention compute path** to run in FP8:

```
Standard BF16 attention path:
  Q(BF16) @ K(BF16)^T → logits(BF16) → softmax(BF16) → P(BF16) @ V(BF16) → out(BF16)

FP8 KV storage only (SGLang / vLLM default):
  Q(BF16) @ K(FP8 → BF16)^T → logits(BF16) → softmax(BF16) → P(BF16) @ V(FP8 → BF16) → out(BF16)
                ↑ dequantize in kernel            ↑ dequantize in kernel

Full FP8 attention (vLLM with q_scale + prob_scale):
  Q(BF16 → FP8) @ K(FP8)^T → logits(FP8 → BF16) → softmax(BF16 → FP8) → P(FP8) @ V(FP8) → out(BF16)
       ↑ q_scale                                            ↑ prob_scale
```

The full FP8 path runs the two major matmuls (`Q@K^T` and `P@V`) on FP8 tensor cores, which have 2× the throughput of BF16 tensor cores on H100. This is the vLLM analog of SageAttention2 (Section 05).

**Deletion of scales for per-token-head modes:**

```python
# File: REPOS/vllm/vllm/model_executor/layers/quantization/kv_cache.py:57
if mode in (KVQuantMode.FP8_PER_TOKEN_HEAD, KVQuantMode.INT8_PER_TOKEN_HEAD):
    # Per-token-head scales are computed dynamically in the kernel — not stored as parameters
    del layer.k_scale
    del layer.v_scale
    # q_scale and prob_scale may still be kept for compute path
```

For dynamic modes, having a parameter tensor for `k_scale` is meaningless — the scale is computed on the fly from the actual token values. vLLM explicitly removes the parameter to prevent accidental use.

---

## The Fused CUDA Write Kernel

**File:** `REPOS/vllm/csrc/cache_kernels.cu:704`

SGLang's `set_kv_buffer()` quantizes in Python: `div_()` + `.to(fp8)` + `.view(uint8)` — three Python-level operations on each layer's KV before `index_put_`. vLLM fuses all of this into `reshape_and_cache_flash`:

```cuda
// Kernel: reshape_and_cache_flash
// Inputs: key (BF16), value (BF16), key_cache (uint8), value_cache (uint8)
//         k_scale (float tensor, shape [1] or [num_heads])
//         v_scale (float tensor, shape [1] or [num_heads])
// kv_scale_stride: 0 for per-tensor, 1 for per-head
__global__ void reshape_and_cache_flash_kernel(...) {
    int token_idx = blockIdx.x;
    int head_idx = blockIdx.y;
    int dim_idx = threadIdx.x;

    float scale_k = k_scale[head_idx * kv_scale_stride];  // stride=0 → same scale all heads
    float scale_v = v_scale[head_idx * kv_scale_stride];

    float k_val = key[token_idx, head_idx, dim_idx];
    float8_e4m3_t k_fp8 = float_to_fp8(k_val / scale_k);  // scale + cast in one step
    key_cache[slot_idx, head_idx, dim_idx] = reinterpret_as_uint8(k_fp8);
    // same for V
}
```

The `kv_scale_stride` dispatch supports both per-tensor (`stride=0` → all heads share `k_scale[0]`) and per-head (`stride=1` → each head uses its own scale element). This single kernel covers both the per-tensor and per-token-head modes.

### NVFP4 dispatch (Blackwell only)

**File:** `REPOS/vllm/csrc/cache_kernels.cu:731`

```cuda
if (kv_cache_dtype == "nvfp4") {
    // Dispatch to SM100-compiled FP4 kernel
    // Requires Blackwell B100/B200 (compute capability 10.0)
    reshape_and_cache_flash_fp4(...)
}
```

NVFP4 packs two FP4 values per byte (0.5 bytes per KV element) with block-scale FP8 factors per group of 16. The kernel is compiled separately for SM100 and not included in default builds — a `NotImplementedError` is raised on pre-Blackwell hardware.

---

## The Read Path: FP8 KV in Flash Attention

**File:** `REPOS/vllm/vllm/v1/attention/backends/flash_attn.py:743`

vLLM's FlashAttention backend reads FP8 K/V and passes the descale tensors to the underlying CUDA kernel:

```python
def forward(self, query, key_cache, value_cache, ...):
    if self.kv_quant_mode == KVQuantMode.FP8_PER_TENSOR:
        k_descale = self.layer.k_scale   # shape [1], one per layer
        v_descale = self.layer.v_scale

    elif self.kv_quant_mode == KVQuantMode.FP8_PER_TOKEN_HEAD:
        k_descale = page_k_scales_tensor  # shape [num_tokens, num_heads], from cache page
        v_descale = page_v_scales_tensor  # loaded from the scale region of each page

    flash_attn_with_kvcache(
        q=query,                     # BF16
        k_cache=key_cache,           # uint8 (FP8 bits)
        v_cache=value_cache,         # uint8 (FP8 bits)
        k_descale=k_descale,
        v_descale=v_descale,
        # Flash Attention internally: k_fp8 * k_descale → bf16 K → compute attention
    )
```

For per-token-head mode, the scales were written into the page alongside the KV data during the write step — they are loaded together from HBM in a single cache-line-friendly access pattern.

---

## The Sleep-Mode Scale Reset Bug

**File:** `REPOS/vllm/vllm/v1/worker/gpu_model_runner.py:885`

A known production correctness risk in vLLM: when the engine enters "sleep mode" (GPU VRAM is freed to allow other processes to use the GPU during idle periods), `init_fp8_kv_scales()` is called on wakeup to reinitialize the KV cache. This function resets all per-layer `k_scale`/`v_scale` to **1.0**:

```python
def init_fp8_kv_scales(self):
    for layer in self.model.attention_layers:
        layer.k_scale_float = 1.0   # BUG: overwrites the calibrated scale from checkpoint
        layer.v_scale_float = 1.0
```

The calibrated scales loaded from `--quantization-param-path` at startup are **lost**. After wakeup, the engine serves with scale=1.0 — potentially causing accuracy degradation until the next restart.

This is marked as a TODO in the vLLM source. Workaround: avoid sleep mode when using calibrated FP8 KV scales, or monitor accuracy after wakeup events.

---

## TurboQuant (vLLM-Only, NVIDIA Research)

TurboQuant is a family of NVIDIA Research custom quantization formats for KV cache. It does not follow the standard FP8/INT8 encoding:

| Format | K dtype | V dtype | Notes |
|---|---|---|---|
| `turboquant_int8` | INT8 | INT8 | Standard per-channel INT8 |
| `turboquant_k8v4` | INT8 | INT4 | Asymmetric: K needs more precision |
| `turboquant_3bit_nc` | 3-bit | 3-bit | Non-power-of-2, requires packed kernels |

All TurboQuant formats automatically **skip quantization for the first 2 and last 2 attention layers** — these boundary layers have atypical KV activation distributions that quantization handles poorly. The middle layers are more uniform.

TurboQuant is currently available in vLLM through NVIDIA-supplied Python packages (`vllm-turboquant`). It is not in SGLang.

---

## SGLang vs vLLM Comparison

| Feature | SGLang | vLLM |
|---|---|---|
| Per-tensor static FP8 | ✓ (`fp8_e4m3`) | ✓ (`fp8_e4m3`) |
| Per-tensor static FP8 e5m2 | ✓ | ✓ |
| Per-token-head dynamic FP8 | — | ✓ (`fp8_per_token_head`) |
| Per-token-head dynamic INT8 | — | ✓ (`int8_per_token_head`) |
| NVFP4 (Blackwell) | — | ✓ (WIP) |
| TurboQuant | — | ✓ (NVIDIA partner) |
| `q_scale` (Q compute) | — | ✓ |
| `prob_scale` (P compute) | — | ✓ |
| MLA FP8 (DeepSeek) | ✓ (per-tile scales) | ✓ (`fp8_ds_mla`) |
| Fused write CUDA kernel | Partial (TRT-LLM path) | ✓ (`reshape_and_cache_flash`) |
| `auto` from config.json | ✓ | ✓ |
| AMD ROCm FNUZ path | ✓ (2× scale factor) | ✓ |

---

## Code Anchor Reference

| Concept | File | Lines |
|---|---|---|
| `CacheDType` (13 formats) | `vllm/config/cache.py` | 18 |
| `KVQuantMode` enum | `vllm/v1/kv_cache_interface.py` | 30 |
| `page_size_bytes` with scale storage | `vllm/v1/kv_cache_interface.py` | 132 |
| `BaseKVCacheMethod` (q/k/v/prob scales) | `vllm/model_executor/layers/quantization/kv_cache.py` | 18 |
| Per-token-head scale deletion | `vllm/model_executor/layers/quantization/kv_cache.py` | 57 |
| `reshape_and_cache_flash` kernel | `csrc/cache_kernels.cu` | 704 |
| `kv_scale_stride` per-head dispatch | `csrc/cache_kernels.cu` | 718 |
| NVFP4 kernel dispatch | `csrc/cache_kernels.cu` | 731 |
| FP8 read path + descale tensors | `vllm/v1/attention/backends/flash_attn.py` | 743 |
| Per-token-head scale load from page | `vllm/v1/attention/backends/flash_attn.py` | 756 |
| Sleep-mode scale reset bug | `vllm/v1/worker/gpu_model_runner.py` | 885 |
| TurboQuant layer skip logic | `vllm/model_executor/layers/quantization/turboquant.py` | — |

---

## Summary

- vLLM's `KVQuantMode` enum provides clean dispatch across 13 dtype formats vs SGLang's string comparison over 5
- `fp8_per_token_head` and `int8_per_token_head` compute scales dynamically at inference — no calibration, ~3% scale storage overhead, better for diverse inputs
- `q_scale` and `prob_scale` extend quantization beyond KV storage to the attention compute path (Q@K and P@V matmuls in FP8)
- `reshape_and_cache_flash` CUDA kernel fuses scale + cast + scatter in one kernel; supports both per-tensor and per-head scale shapes
- NVFP4 (0.5 bytes/value) is in progress for Blackwell B100/B200; TurboQuant (NVIDIA Research) provides additional asymmetric formats
- Sleep-mode scale reset is a known production correctness risk: calibrated scales are overwritten with 1.0 after GPU wakeup

**Next section:** what the research literature reveals about quantizing below FP8 — KIVI's 2-bit asymmetric scheme, KVQuant's pre-RoPE technique, and SageAttention2's compute-path quantization.
