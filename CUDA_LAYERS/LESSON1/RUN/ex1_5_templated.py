"""
Exercise 1.5 — Templated Vectorized Elementwise Kernel
=======================================================
The production pattern: one templated kernel handles any elementwise op.
The operation is passed as a functor (struct with __device__ operator()).
The compiler inlines the functor — zero overhead vs hand-written kernel.

Kernels implemented here:
  relu_f16   — ReLU activation
  silu_f16   — SiLU (Swish) activation  x * sigmoid(x)
  gelu_f16   — GeLU activation           x * Phi(x)  (approx)
  add_f16    — elementwise add of two tensors

GPU: RTX 4060 Ti | Peak: 288 GB/s
"""

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Operator functors — each becomes an inline PTX sequence, zero call overhead
// ---------------------------------------------------------------------------

struct ReluOp {
    __device__ __forceinline__ __half operator()(__half x) const {
        // fmaxf: fused max with 0, one PTX instruction
        return __float2half(fmaxf(__half2float(x), 0.f));
    }
};

struct SiluOp {
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    __device__ __forceinline__ __half operator()(__half x) const {
        float v = __half2float(x);
        return __float2half(v / (1.f + expf(-v)));
    }
};

struct GeluOp {
    // GeLU approx: 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3)))
    __device__ __forceinline__ __half operator()(__half x) const {
        float v = __half2float(x);
        float c = 0.7978845608f;   // sqrt(2/pi)
        float t = tanhf(c * (v + 0.044715f * v * v * v));
        return __float2half(0.5f * v * (1.f + t));
    }
};

// ---------------------------------------------------------------------------
// Generic vectorized elementwise kernel
// VEC_SIZE = 8: loads 8 fp16 values as one uint4 (128 bits)
// OpFunc: any functor with __device__ __half operator()(__half)
// ---------------------------------------------------------------------------
template<int VEC_SIZE, typename OpFunc>
__global__ void elementwise_vec(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int           n_vec,    // = n_elements / VEC_SIZE
    OpFunc        op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // 128-bit load
    uint4 raw_in = ((const uint4*)src)[idx];
    const __half* in_vals = reinterpret_cast<const __half*>(&raw_in);

    uint4 raw_out;
    __half* out_vals = reinterpret_cast<__half*>(&raw_out);

    // Apply op to each of the 8 fp16 elements (fully unrolled)
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        out_vals[i] = op(in_vals[i]);
    }

    // 128-bit store
    ((uint4*)dst)[idx] = raw_out;
}

// ---------------------------------------------------------------------------
// Two-input vectorized elementwise (for ops like add, mul)
// ---------------------------------------------------------------------------
template<int VEC_SIZE, typename OpFunc>
__global__ void elementwise_vec2(
    const __half* __restrict__ a,
    const __half* __restrict__ b,
    __half*       __restrict__ dst,
    int           n_vec,
    OpFunc        op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    uint4 raw_a = ((const uint4*)a)[idx];
    uint4 raw_b = ((const uint4*)b)[idx];
    const __half* va = reinterpret_cast<const __half*>(&raw_a);
    const __half* vb = reinterpret_cast<const __half*>(&raw_b);

    uint4 raw_out;
    __half* out_vals = reinterpret_cast<__half*>(&raw_out);

    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        out_vals[i] = op(va[i], vb[i]);
    }

    ((uint4*)dst)[idx] = raw_out;
}

struct AddOp {
    __device__ __forceinline__ __half operator()(__half a, __half b) const {
        return __hadd(a, b);   // native fp16 add — one PTX instruction
    }
};

// ---------------------------------------------------------------------------
// PyTorch-visible launch wrappers
// ---------------------------------------------------------------------------

torch::Tensor relu_f16(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    elementwise_vec<8><<<(n_vec + 255) / 256, 256>>>(
        (const __half*)src.data_ptr(), (__half*)dst.data_ptr(), n_vec, ReluOp{});
    return dst;
}

torch::Tensor silu_f16(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    elementwise_vec<8><<<(n_vec + 255) / 256, 256>>>(
        (const __half*)src.data_ptr(), (__half*)dst.data_ptr(), n_vec, SiluOp{});
    return dst;
}

torch::Tensor gelu_f16(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    elementwise_vec<8><<<(n_vec + 255) / 256, 256>>>(
        (const __half*)src.data_ptr(), (__half*)dst.data_ptr(), n_vec, GeluOp{});
    return dst;
}

torch::Tensor add_f16(torch::Tensor a, torch::Tensor b) {
    TORCH_CHECK(a.dtype() == torch::kFloat16 && b.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(a.numel() == b.numel(), "Size mismatch");
    TORCH_CHECK(a.numel() % 8 == 0, "numel must be divisible by 8");
    auto dst  = torch::empty_like(a);
    int n_vec = a.numel() / 8;
    elementwise_vec2<8><<<(n_vec + 255) / 256, 256>>>(
        (const __half*)a.data_ptr(), (const __half*)b.data_ptr(),
        (__half*)dst.data_ptr(), n_vec, AddOp{});
    return dst;
}
"""

cpp_src = """
torch::Tensor relu_f16(torch::Tensor src);
torch::Tensor silu_f16(torch::Tensor src);
torch::Tensor gelu_f16(torch::Tensor src);
torch::Tensor add_f16(torch::Tensor a, torch::Tensor b);
"""

mod = load_inline(
    name="ex1_5_templated",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_f16", "silu_f16", "gelu_f16", "add_f16"],
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


# ---- correctness ----
x = torch.randn(4096, device="cuda", dtype=torch.float16)
y = torch.randn(4096, device="cuda", dtype=torch.float16)

torch.testing.assert_close(mod.relu_f16(x), torch.relu(x), rtol=1e-3, atol=1e-3)
print("ReLU correctness: PASSED")

torch.testing.assert_close(mod.silu_f16(x), F.silu(x), rtol=1e-2, atol=1e-2)
print("SiLU correctness: PASSED")

torch.testing.assert_close(mod.gelu_f16(x), F.gelu(x), rtol=1e-2, atol=5e-3)
print("GeLU correctness: PASSED")

torch.testing.assert_close(mod.add_f16(x, y), x + y, rtol=1e-3, atol=1e-3)
print("Add  correctness: PASSED")

# ---- benchmark ----
N = 1024 * 1024 * 64   # 64M fp16 = 128 MB
xb = torch.randn(N, device="cuda", dtype=torch.float16)
yb = torch.randn(N, device="cuda", dtype=torch.float16)
bytes_rw = N * 2 * 2   # 2 bytes/elem × read+write

print(f"\n{'='*55}")
print(f"  Elements: {N/1e6:.0f}M fp16  ({bytes_rw/1e6:.0f} MB read+write)")
print(f"  Peak    : 288 GB/s  (RTX 4060 Ti)")
print(f"{'='*55}")

for name, fn in [
    ("ReLU    (fmaxf)", lambda: mod.relu_f16(xb)),
    ("SiLU    (x*sig)", lambda: mod.silu_f16(xb)),
    ("GeLU    (tanh) ", lambda: mod.gelu_f16(xb)),
    ("Add     (hadd) ", lambda: mod.add_f16(xb, yb)),
]:
    ms = bench(fn)
    bw = bytes_rw / ms / 1e6
    # add has 2 reads + 1 write
    if "Add" in name:
        bw = N * 2 * 3 / ms / 1e6
    print(f"  {name}: {ms:.3f} ms | {bw:.1f} GB/s | {bw/288*100:.0f}% util")

print(f"{'='*55}")
