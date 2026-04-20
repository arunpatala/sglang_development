"""
Exercise 1.1 — Scalar Copy Baseline
=====================================
One thread loads one fp16 element. This is the naive pattern.
Measures bandwidth so you have a baseline to compare against ex1_3 and ex1_4.

GPU: RTX 4060 Ti | Peak: 288 GB/s | Expected here: 30–80 GB/s
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>

// Scalar copy: one thread loads one fp16, stores one fp16.
// Generates a 16-bit load (LD.E.16) — the worst-case memory access pattern.
__global__ void copy_scalar(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

torch::Tensor scalar_copy(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16 tensor");
    TORCH_CHECK(src.is_cuda(), "Expected CUDA tensor");
    auto dst = torch::empty_like(src);
    int n     = src.numel();
    int block = 256;
    int grid  = (n + block - 1) / block;
    copy_scalar<<<grid, block>>>(
        (const __half*)src.data_ptr(),
        (__half*)dst.data_ptr(),
        n);
    return dst;
}
"""

cpp_src = "torch::Tensor scalar_copy(torch::Tensor src);"

mod = load_inline(
    name="ex1_1_scalar_copy",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["scalar_copy"],
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
    return s.elapsed_time(e) / iters   # ms


N = 1024 * 1024 * 64   # 64M fp16 values = 128 MB
x = torch.randn(N, device="cuda", dtype=torch.float16)

# Correctness
out = mod.scalar_copy(x)
torch.testing.assert_close(out, x)
print("Correctness: PASSED")

# Benchmark
ms = bench(lambda: mod.scalar_copy(x))
bytes_rw = x.numel() * x.element_size() * 2   # read + write
bw = bytes_rw / ms / 1e6   # GB/s

print(f"\n{'='*50}")
print(f"  Kernel   : scalar copy (fp16, LD.E.16)")
print(f"  Elements : {N/1e6:.0f}M fp16  ({bytes_rw/1e6:.0f} MB read+write)")
print(f"  Time     : {ms:.3f} ms")
print(f"  Bandwidth: {bw:.1f} GB/s")
print(f"  Peak     : 288 GB/s  (RTX 4060 Ti)")
print(f"  Util     : {bw/288*100:.1f}%")
print(f"{'='*50}")
print(f"  Note: this number will ~3-5x in ex1_3 and ex1_4")
