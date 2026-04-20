"""
Exercise 2.4 — Vectorized fp16 Load + Warp Reduce (RMSNorm pass-1)
===================================================================
Combine Lesson 1 (128-bit vectorized loads) with Lesson 2 (warp shuffle reduce).

Each thread:
  1. Loads 8 fp16 values as a uint4 (one LD.E.128 instruction — Lesson 1)
  2. Upcasts each to float32 and accumulates partial sum-of-squares
  3. Contributes its partial to warp_reduce_sum — Lesson 2

After the warp reduce:
  4. Every lane holds total sum_sq for the entire row
  5. Each thread independently computes rms_rcp = rsqrt(sum_sq / D + eps)
  6. Second pass: load input + weight, output = input * rms_rcp * weight

This file implements BOTH passes — the complete RMSNorm for hidden_dim=256.

GPU: RTX 4060 Ti | Reference: norm.cuh apply_norm_impl lines 60–109
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ─── Warp reduce sum (from Lesson 2) ─────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// ─── Pass 1 only: compute sum-of-squares per row ─────────────────────────────
// hidden_dim = 256: 32 threads × 8 fp16/thread = 256 elements per row.
// Each thread loads one uint4 (128-bit), processes 8 fp16, contributes
// to a partial sum_sq, then warp_reduce gives the total.
__global__ void sum_of_squares_f16(
    const uint4* __restrict__ src,   // packed fp16: each uint4 = 8 fp16
    float*       __restrict__ out,   // one float per row (per block)
    int          n_vec)              // = hidden_dim / 8 = 32 for dim=256
{
    int lane = threadIdx.x;          // 0–31
    int row  = blockIdx.x;

    // 128-bit load: one instruction fetches 8 fp16 values (Lesson 1)
    uint4  raw  = src[row * n_vec + lane];
    __half* vals = reinterpret_cast<__half*>(&raw);

    // Accumulate partial sum-of-squares in float32
    float local_sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v  = __half2float(vals[i]);
        local_sq += v * v;
    }

    // Warp reduce: every thread gets total sum_sq for this row (Lesson 2)
    float total_sq = warp_reduce_sum(local_sq);

    if (lane == 0) out[row] = total_sq;
}

// ─── Full RMSNorm (pass 1 + pass 2, hidden_dim=256) ──────────────────────────
// Pass 1: load input, compute sum_sq, warp reduce → rms_rcp
// Pass 2: load input + weight, scale by rms_rcp, store
// Both passes combined in one kernel — no second HBM round-trip for input.
__global__ void rmsnorm_f16_dim256(
    const uint4* __restrict__ x,      // input   (batch, 256) fp16
    const uint4* __restrict__ w,      // weight  (256,) fp16
    uint4*       __restrict__ out,    // output  (batch, 256) fp16
    float        eps,
    int          n_vec)               // = 256 / 8 = 32
{
    int lane = threadIdx.x;   // 0–31
    int row  = blockIdx.x;

    // Pass 1: load 8 fp16 → accumulate sum_sq → warp reduce
    uint4  raw_x = x[row * n_vec + lane];
    __half* xv   = reinterpret_cast<__half*>(&raw_x);

    float local_sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v  = __half2float(xv[i]);
        local_sq += v * v;
    }

    float total_sq = warp_reduce_sum(local_sq);               // Lesson 2
    float rms_rcp  = rsqrtf(total_sq / (n_vec * 8) + eps);   // 1 / rms

    // Pass 2: load weight, normalize, store
    uint4  raw_w = w[lane];     // weight is the same for every row
    __half* wv   = reinterpret_cast<__half*>(&raw_w);

    uint4  raw_out;
    __half* ov = reinterpret_cast<__half*>(&raw_out);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float xi = __half2float(xv[i]);
        float wi = __half2float(wv[i]);
        ov[i]    = __float2half(xi * rms_rcp * wi);
    }

    out[row * n_vec + lane] = raw_out;   // 128-bit store
}

torch::Tensor row_sum_of_squares(torch::Tensor x) {
    TORCH_CHECK(x.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 256, "Shape must be (B, 256)");
    int batch  = x.size(0);
    int n_vec  = 256 / 8;   // = 32
    auto out   = torch::zeros({batch}, torch::dtype(torch::kFloat32).device(x.device()));
    sum_of_squares_f16<<<batch, 32>>>(
        (const uint4*)x.data_ptr(), out.data_ptr<float>(), n_vec);
    return out;
}

torch::Tensor rmsnorm_f16(torch::Tensor x, torch::Tensor w, float eps) {
    TORCH_CHECK(x.dtype() == torch::kFloat16 && w.dtype() == torch::kFloat16);
    TORCH_CHECK(x.dim() == 2 && x.size(1) == 256, "Shape must be (B, 256)");
    TORCH_CHECK(w.numel() == 256, "Weight must be length 256");
    int batch = x.size(0);
    int n_vec = 256 / 8;
    auto out  = torch::empty_like(x);
    rmsnorm_f16_dim256<<<batch, 32>>>(
        (const uint4*)x.data_ptr(),
        (const uint4*)w.data_ptr(),
        (uint4*)out.data_ptr(),
        eps,
        n_vec);
    return out;
}
"""

cpp_src = """
torch::Tensor row_sum_of_squares(torch::Tensor x);
torch::Tensor rmsnorm_f16(torch::Tensor x, torch::Tensor w, float eps);
"""

mod = load_inline(
    name="ex2_4_vec_load_warp_reduce",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["row_sum_of_squares", "rmsnorm_f16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)


def bench(fn, warmup=20, iters=500):
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


# ---- correctness: sum_of_squares ----
batch, hidden = 512, 256
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
out_sq = mod.row_sum_of_squares(x)
ref_sq = (x.float() ** 2).sum(dim=1)
torch.testing.assert_close(out_sq, ref_sq, rtol=1e-2, atol=1e-2)
print("row_sum_of_squares: PASSED")

# ---- correctness: full RMSNorm ----
w = torch.ones(hidden, device="cuda", dtype=torch.float16)
eps = 1e-5
out_rms = mod.rmsnorm_f16(x, w, eps)
ref_rms = x.float() / torch.sqrt((x.float() ** 2).mean(dim=1, keepdim=True) + eps)
ref_rms = (ref_rms * w.float()).to(torch.float16)
torch.testing.assert_close(out_rms, ref_rms, rtol=1e-2, atol=1e-2)
print("rmsnorm_f16       : PASSED")

# ---- benchmark ----
batch_big = 4096
x_big = torch.randn(batch_big, hidden, device="cuda", dtype=torch.float16)
w_big = torch.ones(hidden, device="cuda", dtype=torch.float16)

ms = bench(lambda: mod.rmsnorm_f16(x_big, w_big, eps))
bytes_rw = batch_big * hidden * 2 * 3   # read x, read w, write out
bw = bytes_rw / ms / 1e6

print(f"\n{'='*60}")
print(f"  Kernel     : RMSNorm fp16, hidden=256 (2-pass, warp reduce)")
print(f"  Shape      : ({batch_big}, {hidden})  fp16")
print(f"  Time       : {ms:.3f} ms")
print(f"  Bandwidth  : {bw:.1f} GB/s  (read x + read w + write out)")
print(f"  Peak       : 288 GB/s (RTX 4060 Ti)")
print(f"  Util       : {bw/288*100:.1f}%")
print(f"  Technique  : uint4 LD.E.128 (Lesson 1) + warp_reduce_sum (Lesson 2)")
print(f"{'='*60}")
print(f"  This is exactly what SGLang's norm.cuh apply_norm_impl does.")
print(f"  Next: ex2_5 — extend to multi-warp blocks (hidden_dim > 256)")
