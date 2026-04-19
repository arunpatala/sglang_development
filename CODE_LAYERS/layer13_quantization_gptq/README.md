# Layer 12 — GPTQ Quantization

Builds on Layer 11 (prefix caching) by adding **4-bit GPTQ quantization** support. The model weights
are stored as packed int4 values and dequantized on-the-fly during the matrix multiply using the
`sgl_kernel.gptq_gemm` fused CUDA kernel (ExLlama v2).

- **Quantized model**: `JunHowie/Qwen3-0.6B-GPTQ-Int4`
- **Baseline model**: `Qwen/Qwen3-0.6B` (fp16/bf16 reference)
- **Kernel**: `sgl_kernel.gptq_gemm` — fused int4 dequant + GEMM in one CUDA kernel

---

## How GPTQ quantization works

### Weight layout

Each `nn.Linear` is replaced by a `GPTQLinear` that stores 4 quantized buffers instead of one float weight matrix:

```
qweight : [K // 8,         N]   int32   — 8 int4 values packed per int32, column-major
scales  : [K // group_size, N]  float16 — one fp16 scale per group of `group_size` input rows
qzeros  : [K // group_size, N // 8]  int32  — zero-points packed same as qweight
g_idx   : [K]               int32   — maps each input row to its scale/zero group (all-zeros for desc_act=False)
```

For Qwen3-0.6B with `group_size=128`:
- MLP `gate_proj` qweight: `[1024 // 8, 3072] = [128, 3072]` int32
- MLP `down_proj` qweight: `[3072 // 8, 1024] = [384, 1024]` int32

### Forward pass

```python
# gptq_gemm dequantizes on-the-fly and performs the matmul in one kernel:
#   w_fp = (qweight_unpacked - qzeros) * scales
# Activations must be float16 (not bfloat16) for correct kernel behavior.

x_fp16 = x.to(torch.float16)
y = gptq_gemm(x_fp16, qweight, qzeros, scales, g_idx_seq,
              use_shuffle=False, bits=4)
return y.to(orig_dtype)   # cast back to bf16
```

### Memory savings

| Model | Format | GPU memory |
|-------|--------|------------|
| Qwen3-0.6B | fp16/bf16 | ~1.2 GB |
| Qwen3-0.6B-GPTQ-Int4 | 4-bit | ~600 MB |

4× weight compression (int4 vs fp16), activations remain bf16.

---

## What changed from Layer 11

| Component | Layer 11 | Layer 12 |
|-----------|----------|----------|
| Linear layers | `nn.Linear` (fp16/bf16) | `GPTQLinear` (int4 packed weights) |
| Model package | `model/` | `model_gptq/` (new parallel package) |
| Weight loading | `named_parameters()` only | `named_parameters()` + `named_buffers()` |
| Dtype handling | `model.to(bf16)` on all tensors | Parameters → bf16, int32/fp16 buffers preserved |
| Config loading | `config.json` only | `config.json` + `quantize_config.json` |
| Kernel | `F.linear` / FlashInfer | `sgl_kernel.gptq_gemm` for linear layers |
| `verify_gptq.py` | — | **New**: 4-test correctness check |

---

## Architecture

### `model_gptq/` — parallel quantized model package

A complete copy of the `model/` package where `nn.Linear` is replaced by `GPTQLinear` in attention and MLP:

```
model_gptq/
├── gptq_linear.py     — GPTQLinear: 4-bit weight buffers + gptq_gemm forward
├── qwen3.py           — Qwen3ForCausalLM (GPTQ): load weights, quantize_config, prepare()
├── attention.py       — Qwen3Attention: q/k/v/o projections → GPTQLinear
├── mlp.py             — Qwen3MLP: gate/up/down projections → GPTQLinear
├── decoder_layer.py   — unchanged structure, uses quantized attention + MLP
├── config.py          — unchanged
├── norm.py            — unchanged (RMSNorm stays fp)
└── rope.py            — unchanged (RoPE stays fp)
```

`lm_head` and `embed_tokens` are **not quantized** (matches `quantize_config.json: "lm_head": false`).

### `GPTQLinear` — key design decisions

**Buffers, not parameters**: The four GPTQ tensors (`qweight`, `scales`, `qzeros`, `g_idx`) are registered as `nn.Module` buffers. This means:
- `model.to(dtype)` skips them — int32 stays int32, fp16 stays fp16
- `optimizer` ignores them — no gradients
- `named_parameters()` does NOT enumerate them — `load_weights()` must also call `named_buffers()`

**fp16 scales**: `gptq_gemm` reads scale bits as `float16` internally. If scales are cast to `bfloat16` (e.g. by a naïve `model.to(bf16)` call), the bit patterns are reinterpreted and produce completely wrong outputs. Scales are kept as `torch.float16` permanently.

**Activation dtype**: `gptq_gemm` only produces correct results with `float16` activations. The layer casts `x → fp16` before the kernel call and casts the result back to the original dtype (usually `bfloat16`).

**`use_shuffle=False`**: Uses the raw packed `qweight` directly without pre-processing via `gptq_shuffle`. The `prepare()` method marks the layer ready (no-op currently) for forward compatibility.

### `from_pretrained()` loading sequence

```
1. Resolve model path (HF Hub or local dir)
2. Read config.json → Qwen3Config
3. Read quantize_config.json → bits=4, group_size=128
4. Build model on CPU with GPTQLinear layers (empty buffers)
5. Stream weights from model.safetensors:
     - fp tensors (scales, norms, embed) → kept as native dtype
     - int32 tensors (qweight, qzeros, g_idx) → kept as int32
     (DO NOT call .to(dtype) on int32 tensors — corrupts packed bits)
6. Cast floating PARAMETERS to bf16
     (deliberately skips buffers so scales stay fp16)
7. Move to CUDA — buffers keep their dtypes
8. Call GPTQLinear.prepare() on all layers
9. Set eval mode
```

---

## Files

| File | Role |
|------|------|
| `model_gptq/gptq_linear.py` | **New** — `GPTQLinear`: packed int4 buffers, `gptq_gemm` forward, `prepare()` |
| `model_gptq/qwen3.py` | **New** — GPTQ `Qwen3ForCausalLM`: `quantize_config.json`, buffer-aware `load_weights` |
| `model_gptq/attention.py` | **New** — `Qwen3Attention` with `GPTQLinear` projections |
| `model_gptq/mlp.py` | **New** — `Qwen3MLP` with `GPTQLinear` gate/up/down |
| `model_gptq/` (rest) | **New** — config, norm, rope, decoder_layer (unchanged logic) |
| `model_runner.py` | **Modified** — `gptq=True` flag to load `model_gptq` instead of `model` |
| `verify_gptq.py` | **New** — 4 correctness tests: load, forward, greedy decode, fp16 comparison |

---

## Verify

```bash
python verify_gptq.py
```

**Test 1** — GPTQ model loads and uses < 700 MB GPU memory (vs ~1.2 GB for fp16)  
**Test 2** — Forward pass: no NaN or Inf in logits  
**Test 3** — Greedy decode produces valid token IDs  
**Test 4** — (Optional) Top-1 token agreement with fp16 model; top-5 overlap ≥ 3/5

---

## Benchmark

**Config**: 20 requests · concurrency=4 · max_tokens=128 · page_size=16

| Metric | Value |
|--------|-------|
| Total wall time | 8.75s |
| Output throughput | 203.0 tok/s |
| Total throughput | 319.9 tok/s |
| TTFT avg / p95 | 174ms / 721ms |
| Latency avg / p95 | 1464ms / 2542ms |

The GPTQ model runs at ~half the memory of the fp16 baseline with comparable throughput, since the
bottleneck at this batch size is KV cache bandwidth and FlashInfer kernel overhead, not weight loading.
The memory reduction matters most when fitting larger models or higher batch sizes on the same GPU.

---

## Key bugs fixed during implementation

1. **`named_buffers()` missing from `load_weights`** — `GPTQLinear` stores its tensors as buffers, not parameters. Without explicitly iterating `named_buffers()`, all GPTQ tensors were silently skipped, leaving the model with uninitialized (garbage) weights.

2. **int32 tensors cast to bfloat16** — The safetensors loader originally called `.to(dtype)` on every tensor unconditionally. Casting `qweight`/`qzeros`/`g_idx` (int32) to bfloat16 reinterprets the packed 4-bit values as bf16 bit patterns, producing completely wrong outputs. Fixed by preserving native dtype from checkpoint.

3. **fp16 scales corrupted by `model.to(bf16)`** — `gptq_gemm` reads scale memory as raw fp16 bits. A global `model.to(bfloat16)` before loading corrupted scale buffers. Fixed by casting only floating-point `parameters` (not buffers) to bf16.

4. **bfloat16 activations in `gptq_gemm`** — The kernel requires fp16 activations. Passing bf16 (the model's default dtype) yields drastically wrong outputs. Fixed by casting input to fp16 before the kernel call and casting the result back.
