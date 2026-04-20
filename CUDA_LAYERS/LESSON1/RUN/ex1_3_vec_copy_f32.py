"""
Exercise 1.3 — Vectorized Copy: float32 with float4
=====================================================
Each thread loads 4 floats (128 bits) in a single LD.E.128 instruction.
For float32, float4 is the natural 128-bit vector type.

GPU: RTX 4060 Ti | Peak: 288 GB/s | Expected here: 220–270 GB/s
Compare against ex1_1: should be 3–5x higher bandwidth.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Vectorized copy: each thread loads 4 floats (128 bits) at once via float4.
// float4 = { float x, y, z, w } = 16 bytes = 128 bits.
// Generates: ld.global.ca.v4.f32  {f1,f2,f3,f4}, [ptr];
__global__ void copy_vec4_f32(
    const float4* __restrict__ src,
    float4*       __restrict__ dst,
    int n_vec)   // n_vec = n_fp32_elements / 4
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];   // single LD.E.128 + ST.E.128
    }
}

torch::Tensor vec_copy_f32(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Expected float32 tensor");
    TORCH_CHECK(src.numel() % 4 == 0, "numel must be divisible by 4");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 4;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;
    copy_vec4_f32<<<grid, block>>>(
        (const float4*)src.data_ptr<float>(),
        (float4*)dst.data_ptr<float>(),
        n_vec);
    return dst;
}
"""

cpp_src = "torch::Tensor vec_copy_f32(torch::Tensor src);"

mod = load_inline(
    name="ex1_3_vec_copy_f32",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vec_copy_f32"],
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


N = 1024 * 1024 * 64   # 64M floats = 256 MB

# ---- correctness ----
x_small = torch.randn(4096, device="cuda", dtype=torch.float32)
out_small = mod.vec_copy_f32(x_small)
torch.testing.assert_close(out_small, x_small)
print("Correctness: PASSED")

# ---- benchmark ----
x = torch.randn(N, device="cuda", dtype=torch.float32)
ms = bench(lambda: mod.vec_copy_f32(x))
bytes_rw = x.numel() * x.element_size() * 2   # read + write
bw = bytes_rw / ms / 1e6

print(f"\n{'='*50}")
print(f"  Kernel   : vec copy (float32, float4, LD.E.128)")
print(f"  Elements : {N/1e6:.0f}M float32  ({bytes_rw/1e6:.0f} MB read+write)")
print(f"  Time     : {ms:.3f} ms")
print(f"  Bandwidth: {bw:.1f} GB/s")
print(f"  Peak     : 288 GB/s  (RTX 4060 Ti)")
print(f"  Util     : {bw/288*100:.1f}%")
print(f"{'='*50}")
