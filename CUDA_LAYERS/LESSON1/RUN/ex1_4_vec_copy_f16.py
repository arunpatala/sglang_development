"""
Exercise 1.4 — Vectorized Copy and Scale: fp16 with uint4
===========================================================
LLM activations are fp16. Load 8 fp16 values at once using uint4 (128 bits).
uint4 = 4 × uint32 = 128 bits = 8 × fp16 packed.

Two kernels:
  copy_vec8_f16  — pure copy,  should hit peak bandwidth
  scale_vec8_f16 — scale by float scalar (load fp16, compute fp32, store fp16)

GPU: RTX 4060 Ti | Peak: 288 GB/s | Target: >230 GB/s (>80%)
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// --- Pure copy ---
// uint4 = { uint32 x, y, z, w } = 16 bytes = 128 bits = 8 × fp16.
// No arithmetic: purely tests memory bandwidth.
__global__ void copy_vec8_f16(
    const uint4* __restrict__ src,
    uint4*       __restrict__ dst,
    int n_vec)   // n_vec = n_fp16_elements / 8
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];   // LD.E.128 + ST.E.128
    }
}

// --- Scale: load fp16, compute in fp32, store fp16 ---
// This is the exact pattern used in RMSNorm, RoPE, and other elementwise ops.
// Each fp16 element is:
//   1. Unpacked from the uint4 register via reinterpret
//   2. Upcast to float32 for the multiply (avoids fp16 precision loss)
//   3. Downcast back to fp16 and packed into the output uint4
__global__ void scale_vec8_f16(
    const uint4* __restrict__ src,
    uint4*       __restrict__ dst,
    float        scale,
    int          n_vec)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // 128-bit load — one instruction fetches 8 fp16 values
    uint4 raw_in = src[idx];

    // Reinterpret the 128-bit register as an array of 8 fp16 values
    const __half* in_vals = reinterpret_cast<const __half*>(&raw_in);

    uint4 raw_out;
    __half* out_vals = reinterpret_cast<__half*>(&raw_out);

    // Process 8 elements: compiler fully unrolls this (no loop overhead)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // __half2float: fp16 → fp32 (upcasting for accuracy)
        // __float2half: fp32 → fp16 (downcasting for storage)
        out_vals[i] = __float2half(__half2float(in_vals[i]) * scale);
    }

    // 128-bit store — one instruction
    dst[idx] = raw_out;
}

torch::Tensor vec_copy_f16(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;
    copy_vec8_f16<<<grid, block>>>(
        (const uint4*)src.data_ptr(),
        (uint4*)dst.data_ptr(),
        n_vec);
    return dst;
}

torch::Tensor scale_f16(torch::Tensor src, float scale) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;
    scale_vec8_f16<<<grid, block>>>(
        (const uint4*)src.data_ptr(),
        (uint4*)dst.data_ptr(),
        scale,
        n_vec);
    return dst;
}
"""

cpp_src = """
torch::Tensor vec_copy_f16(torch::Tensor src);
torch::Tensor scale_f16(torch::Tensor src, float scale);
"""

mod = load_inline(
    name="ex1_4_vec_copy_f16",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vec_copy_f16", "scale_f16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)


def bench(fn, warmup=10, iters=200):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


N = 1024 * 1024 * 64   # 64M fp16 = 128 MB

# ---- correctness ----
x_s = torch.randn(1024, device="cuda", dtype=torch.float16)

out_copy = mod.vec_copy_f16(x_s)
torch.testing.assert_close(out_copy, x_s)
print("Copy correctness:  PASSED")

out_scale = mod.scale_f16(x_s, 0.5)
torch.testing.assert_close(out_scale, x_s * 0.5, rtol=1e-3, atol=1e-3)
print("Scale correctness: PASSED")

# ---- benchmark ----
x = torch.randn(N, device="cuda", dtype=torch.float16)
bytes_rw = N * 2 * 2   # 2 bytes per fp16 × 2 (read + write)

ms_copy = bench(lambda: mod.vec_copy_f16(x))
bw_copy = bytes_rw / ms_copy / 1e6

ms_scale = bench(lambda: mod.scale_f16(x, 0.5))
bw_scale = bytes_rw / ms_scale / 1e6

print(f"\n{'='*55}")
print(f"  Kernel       : vec copy fp16×8 (uint4, LD.E.128)")
print(f"  Elements     : {N/1e6:.0f}M fp16  ({bytes_rw/1e6:.0f} MB read+write)")
print(f"  Time         : {ms_copy:.3f} ms")
print(f"  Bandwidth    : {bw_copy:.1f} GB/s")
print(f"  Peak         : 288 GB/s  (RTX 4060 Ti)")
print(f"  Util         : {bw_copy/288*100:.1f}%")
print(f"  Target       : >230 GB/s (>80%)")
print(f"{'='*55}")
print(f"  Kernel       : scale fp16×8 (uint4 + fp32 compute)")
print(f"  Time         : {ms_scale:.3f} ms")
print(f"  Bandwidth    : {bw_scale:.1f} GB/s")
print(f"  Util         : {bw_scale/288*100:.1f}%")
print(f"{'='*55}")
