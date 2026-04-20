# 02 — The qweight Layout

`qweight` is the compressed weight tensor that replaces `nn.Linear.weight`. Understanding its layout — how 4-bit values are packed into 32-bit integers, and how that packing relates to the shape expected by `gptq_gemm` — is the prerequisite for understanding both `forward()` and the checkpoint loading step.

---

## From [N, K] to [K//8, N]

A standard `nn.Linear(K, N, bias=False)` stores its weight as a `[N, K]` float16 tensor in PyTorch convention (output features first, input features second). For `K = N = 2048`: shape `[2048, 2048]`, 8.4 MB.

`GPTQLinear` stores `qweight` as `[K // pack_factor, N]` int32 where `pack_factor = 32 // bits = 8` for 4-bit. For `K = N = 2048`: shape `[256, 2048]`, 2.1 MB.

```python
# model_gptq/gptq_linear.py — qweight buffer
self.pack_factor = 32 // bits      # int4: 8 values per int32
self.register_buffer(
    "qweight",
    torch.empty(in_features // self.pack_factor, out_features, dtype=torch.int32),
)
```

The packing order is: the int32 at `qweight[r, c]` holds 8 consecutive input-feature rows for output column `c`. Row 0 of `qweight` holds the 4-bit weights for input rows 0–7 of output column `c`. Row 1 holds input rows 8–15, and so on. The packing is low-nibble first within each int32.

Note that the shape is `[K // 8, N]` — the input dimension is compressed, the output dimension is uncompressed. This is the transposed-and-packed form of the original `[N, K]` weight. `gptq_gemm` expects this exact layout and handles the transposition and unpacking internally.

---

## Why int32 Instead of uint8 or int8

GPTQ packs 8 × int4 values into each int32 rather than 4 × int4 into each int16 or 2 × int4 into each int8. This choice matches the GPU's native 32-bit register width. The ExLlama v2 kernel (on which `gptq_gemm` is based) is written to load 32-bit words and unpack 8 int4 values per thread, which maps efficiently onto CUDA's uint32 intrinsics.

An alternative packing into uint8 (2 × int4 per byte) would require the kernel to handle byte-level reads, which are slower on current GPU architectures. The 32-bit packing also simplifies the zero-point arithmetic: `qzeros` uses the same 32-bit packing as `qweight`, so both can be unpacked with identical code paths.

---

## register_buffer vs nn.Parameter

```python
self.register_buffer("qweight", torch.empty(..., dtype=torch.int32))
self.register_buffer("scales",  torch.empty(..., dtype=torch.float16))
self.register_buffer("qzeros",  torch.empty(..., dtype=torch.int32))
self.register_buffer("g_idx",   torch.zeros(..., dtype=torch.int32))
self.register_buffer("_g_idx_seq", seq)
```

`register_buffer` makes the tensor part of the module's `state_dict` (so it is saved and loaded with the model) and moves it to the correct device when the module is moved (e.g., `.to("cuda")`). Crucially, registered buffers are not `nn.Parameter` — they have `requires_grad=False` and are excluded from `model.parameters()`.

This exclusion is what enables the selective `model.to(dtype)` step during loading:

```python
# model_gptq/qwen3.py — from_pretrained, step 7
for name, param in model.named_parameters():
    param.data = param.data.to(dtype)
```

`named_parameters()` yields only `nn.Parameter` objects — `embed_tokens.weight`, `lm_head.weight`, and the `RMSNorm` weights. The buffers (`qweight`, `qzeros`, `_g_idx_seq` as int32, `scales` as float16) are not touched. If instead `model.to(bfloat16)` were called, PyTorch would cast every floating-point tensor including `scales`, converting the float16 bit pattern to bfloat16 and producing garbage values when `gptq_gemm` reads them as float16.

---

## _g_idx_seq

```python
# model_gptq/gptq_linear.py — sequential group index
seq = torch.arange(in_features, dtype=torch.int32) // group_size
self.register_buffer("_g_idx_seq", seq)
```

`_g_idx_seq[i] = i // group_size`. For `in_features = 2048` and `group_size = 128`: `_g_idx_seq = [0,0,...,0, 1,1,...,1, ..., 15,15,...,15]` with 128 repetitions per group index. This tensor maps each input row to its group, which `gptq_gemm` uses to look up the correct scale and zero-point row.

GPTQ checkpoints also contain a `g_idx` tensor, which for `desc_act=False` (Qwen3's case) is all zeros — the weights are in natural order without activation-magnitude reordering. The checkpoint's `g_idx` is registered but ignored during `forward()`; `_g_idx_seq` (the sequential assignment) is used instead. This simplifies the kernel lookup from an arbitrary permutation table to a simple integer division.

Section 03 explains `scales` and `qzeros` — the per-group quantization parameters that `gptq_gemm` uses alongside `qweight` and `_g_idx_seq`.
