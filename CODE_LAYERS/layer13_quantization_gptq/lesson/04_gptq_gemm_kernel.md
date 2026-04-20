# 04 — gptq_gemm and the use_shuffle=False Path

`gptq_gemm` is the fused CUDA kernel that performs dequantize-and-matrix-multiply in a single pass, without materializing the full float16 weight matrix in GPU memory. It is the computational core of `GPTQLinear.forward()`.

---

## The forward() Implementation

```python
# model_gptq/gptq_linear.py — GPTQLinear.forward
def forward(self, x: torch.Tensor) -> torch.Tensor:
    from sgl_kernel import gptq_gemm

    orig_dtype = x.dtype
    out_shape  = x.shape[:-1] + (self.out_features,)

    # gptq_gemm only produces correct results with fp16 activations.
    x_2d = x.reshape(-1, self.in_features).to(torch.float16)

    y = gptq_gemm(
        x_2d,
        self.qweight,
        self.qzeros,
        self.scales,
        self._g_idx_seq,   # sequential [0,0,...,1,1,...] — required for use_shuffle=False
        False,             # use_shuffle=False: raw qweight, no gptq_shuffle pre-processing
        self.bits,
    )
    return y.to(orig_dtype).reshape(out_shape)
```

Three implementation decisions deserve explanation.

---

## x.reshape(-1, in_features)

`gptq_gemm` accepts 2D input: shape `[M, K]` where `M` is the number of activation vectors and `K = in_features`. The activation tensor `x` from the model may have shape `[B, seq_len, hidden_size]` (for a prefill), `[1, n_heads, seq_len, head_dim]` (inside the attention forward, after projection), or `[B, hidden_size]` (for decode). `reshape(-1, in_features)` flattens all leading dimensions into one axis, producing `[M, K]` where `M = B × seq_len`. After `gptq_gemm`, `reshape(out_shape)` restores the original leading shape with `out_features` as the last dimension:

```python
out_shape = x.shape[:-1] + (self.out_features,)
# If x.shape = [1, 576, 2048], out_shape = (1, 576, 2048)  [or different N]
```

This is the same reshape pattern used by `F.linear` internally — `GPTQLinear` is a transparent replacement.

---

## The float16 Cast

```python
x_2d = x.reshape(-1, self.in_features).to(torch.float16)
```

The `gptq_gemm` kernel reads activation values as IEEE 754 float16. If the model runs in bfloat16 (which Layer 13 does — `DTYPE = torch.bfloat16`), passing a bfloat16 activation to `gptq_gemm` produces incorrect outputs: the kernel interprets the 16-bit bfloat16 pattern as if it were float16, yielding a very different floating-point value. The cast to float16 before the kernel call and the cast back to `orig_dtype` after are mandatory and correct.

The performance cost of these casts is small. A `[M, K]` float16 tensor created from bfloat16 requires a vectorized reformat operation (approximately one clock per element), whereas the `gptq_gemm` operation itself is a full fused GEMM — orders of magnitude more compute-intensive. The cast overhead is negligible in practice.

---

## use_shuffle=False

The `gptq_gemm` kernel supports two weight layouts:

`use_shuffle=True` expects `qweight` to be pre-permuted via `gptq_shuffle` into a tile-friendly layout that improves cache hit rate for the specific thread-block configuration of the kernel. This requires calling `gptq_shuffle(qweight, qzeros, bits)` once after loading the checkpoint and before the first forward pass.

`use_shuffle=False` expects `qweight` in the raw checkpoint layout — the packed format described in section 02. The kernel handles the lookup itself using `_g_idx_seq` to find the correct scale and zero row for each input row during the matrix multiply.

Layer 13 uses `use_shuffle=False` exclusively:

```python
# model_gptq/gptq_linear.py — prepare()
def prepare(self) -> None:
    # We use the use_shuffle=False path of gptq_gemm, which takes the raw
    # packed qweight directly (no gptq_shuffle pre-processing required).
    # The SGLang test suite only validates use_shuffle=False; the shuffled
    # path produces incorrect results with this kernel version (0.4.1).
    self._prepared = True
```

`prepare()` is effectively a no-op — it sets `_prepared = True` as a correctness marker (to detect if prepare was accidentally skipped) but performs no weight transformation. The `use_shuffle=True` path requires `gptq_shuffle` to be called, which for the specific `sgl_kernel` version in this environment produces incorrect outputs due to a kernel version mismatch with the Qwen3 GPTQ checkpoint layout. The `use_shuffle=False` path is correct and is what SGLang's test suite validates for this kernel version.

---

## What gptq_gemm Does Internally

At the conceptual level, `gptq_gemm` computes:

```
y[m, n] = Σ_k  x_fp16[m, k] × dequant(qweight, qzeros, scales, _g_idx_seq)[k, n]
```

where `dequant(...)` unpacks the 4-bit values from `qweight`, subtracts the zero-points, and multiplies by the scales — all within the CUDA thread before the multiply-accumulate. No intermediate `dequant` tensor is allocated; the float16 weight value is computed in a register and immediately used in the dot product.

This fused approach saves the GPU memory bandwidth that would be required to write and then read the full float16 weight matrix. A `[K, N] = [2048, 2048]` float16 matrix is 8.4 MB; loading it from HBM for a single GEMM takes time proportional to 8.4 MB of bandwidth. Loading only the 2.1 MB `qweight` and the 64 KB `scales`/`qzeros` requires reading approximately 2.2 MB — a 3.8× bandwidth reduction for the weight data.

For compute-bound operations (large `M`), the savings are smaller; for memory-bandwidth-bound operations (small `M`, typical in decode), the weight bandwidth reduction directly translates to faster kernel execution.

Section 05 explains how `GPTQLinear` is threaded through the entire model hierarchy via `bits` and `group_size` parameters.
