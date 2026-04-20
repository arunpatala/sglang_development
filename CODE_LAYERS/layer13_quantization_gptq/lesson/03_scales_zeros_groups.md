# 03 — Scales, Zeros, and Group Quantization

Group quantization divides each weight matrix column into fixed-size groups, assigns a separate scale and zero-point to each group, and packs the groups' weights into 4-bit integers. The group structure is what determines how much accuracy GPTQ retains relative to the full-precision model — finer groups (smaller `group_size`) give better accuracy at the cost of more scale/zero-point storage.

---

## Why Per-Group Quantization

A single global scale per column would force all `K = 2048` input rows to share one scale value. The range of weight values across a full 2048-row column can span two orders of magnitude; clamping the entire range into 4-bit integers (values 0–15) with one global scale leaves most of the 4-bit range unused for the majority of rows. Per-group quantization gives each group of 128 rows its own scale, allowing groups with small weight values to use a fine-grained scale and groups with large values a coarser one.

For `group_size = 128` and `K = 2048`: 16 groups per column. Each group has 128 weights quantized independently with its own scale and zero-point. The 4-bit values represent integers 0–15 within the group's scale; the dequantize formula maps them back:

```
w_float[i] = (qweight_unpacked[i] - qzeros[group(i)]) * scales[group(i)]
```

where `group(i) = i // group_size`.

---

## scales

```python
# model_gptq/gptq_linear.py — scales buffer
self.register_buffer(
    "scales",
    torch.empty(in_features // group_size, out_features, dtype=torch.float16),
)
# Shape: [K // group_size, N] = [16, 2048] for K=N=2048, group_size=128
```

`scales[g, n]` is the scale for group `g` of output column `n`. Its dtype is explicitly `torch.float16` — not bfloat16, not float32. This is a hard constraint imposed by the `gptq_gemm` kernel: it reads scale values as IEEE 754 float16 bit patterns using GPU hardware float16 loads. If `scales` were stored as bfloat16, the 16-bit patterns would be reinterpreted as float16, producing completely wrong scale values and catastrophically incorrect outputs.

This is why the loading code iterates `model.named_parameters()` rather than calling `model.to(bfloat16)`: the latter would cast `scales` to bfloat16. The float16 constraint on `scales` is the most important dtype correctness requirement in the entire Layer 13 codebase.

---

## qzeros

```python
# model_gptq/gptq_linear.py — qzeros buffer
self.register_buffer(
    "qzeros",
    torch.empty(in_features // group_size, out_features // self.pack_factor, dtype=torch.int32),
)
# Shape: [K // group_size, N // pack_factor] = [16, 256] for K=N=2048, group_size=128
```

Zero-points are packed the same way as weights: `pack_factor = 8` int4 values per int32, so `N // pack_factor = N // 8` columns. `qzeros[g, c]` holds the zero-points for 8 consecutive output columns `c*8 .. c*8+7` within group `g`.

The zero-point centers each group's weight distribution. For symmetric quantization (zero-point = 8 for 4-bit unsigned), `qzeros` would be all 8s packed: each int32 would contain `0x88888888`. For asymmetric quantization — common in practice because model weights are not centered at zero — `qzeros` contains the actual offset for each group.

The dequantize formula `(qweight_unpacked - qzeros) * scales` subtracts the zero-point before multiplying by the scale. This ensures that the integer value `qzeros` corresponds to `0.0` in the dequantized domain, which is important for numerical stability in attention computations where exact zeros carry meaning.

---

## Memory Cost of the Scale/Zero Infrastructure

For `K = N = 2048`, `group_size = 128`:

```
scales:  [16, 2048] float16 = 32768 × 2 bytes = 65536 bytes = 64 KB
qzeros:  [16, 256]  int32   = 4096  × 4 bytes = 16384 bytes = 16 KB
_g_idx_seq: [2048]  int32   = 2048  × 4 bytes = 8192  bytes = 8 KB
```

Total metadata per projection: approximately 88 KB. Compared to the 8.4 MB bfloat16 weight, this is about 1% overhead — negligible. The dominant cost reduction is in `qweight`: 2.1 MB vs 8.4 MB. The metadata adds a small fraction back, leaving the net reduction at approximately 3.9× for this projection size.

For the FFN projections where `K = 2048, N = 11008` (or vice versa), the scales and zeros are proportionally larger but still small relative to `qweight`. The group structure's memory overhead scales with `K × N / group_size`, while the weight savings scale with `K × N × (1 - 1/pack_factor)`. For any `pack_factor > 1` and `group_size` in the practical range (64–256), the savings always dominate.

Section 04 explains how `gptq_gemm` uses all four buffers to compute the matrix multiplication without materializing the full float16 weight matrix.
