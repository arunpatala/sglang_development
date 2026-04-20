"""
Exercise 2.5 — Two-Level Block Reduce (Multi-Warp, Preview of Lesson 3)
=======================================================================
A single warp covers 32 threads. Real inference blocks use 128–256 threads
(4–8 warps). This requires a second level: smem to communicate across warps.

Pattern (used in SGLang's cta.cuh):
  Phase 1: Each warp does warp_reduce_sum independently  (5 shuffles, no smem)
  Phase 2: Lane 0 of each warp writes partial to smem[warp_id]
           __syncthreads()
  Phase 3: First warp loads smem values, runs warp_reduce again

Total cost: 10 shuffles + 1 barrier vs naive smem: O(N) serial adds + 1 barrier.

Also demonstrates a full RMSNorm that works at any hidden_dim (not just 256).

GPU: RTX 4060 Ti | Threads: 32/64/128/256 | Zero extra smem beyond 1 slot/warp
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ─── Warp reduce (Lesson 2) ───────────────────────────────────────────────────
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// ─── Two-level block reduce ───────────────────────────────────────────────────
// Works for any blockDim.x that is a multiple of 32 and ≤ 1024.
// smem must have at least (blockDim.x / 32) floats.
__device__ float block_reduce_sum(float val, float* smem) {
    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = blockDim.x  / 32;

    // Phase 1: each warp reduces its 32 lanes → one partial sum per warp
    float warp_sum = warp_reduce_sum(val);

    // Phase 2: lane 0 of each warp stores its partial sum to smem
    if (lane == 0) smem[warp_id] = warp_sum;
    __syncthreads();   // only sync in this entire reduce

    // Phase 3: first warp reduces the partial sums stored in smem
    if (warp_id == 0) {
        // lane index doubles as warp index for this second reduce
        float v = (lane < n_warps) ? smem[lane] : 0.f;
        warp_sum = warp_reduce_sum(v);
        if (lane == 0) smem[0] = warp_sum;   // broadcast result via smem
    }
    __syncthreads();   // ensure smem[0] is visible to all warps

    return smem[0];    // every thread reads the final result
}

// ─── Block reduce kernel: any multiple-of-32 thread count ────────────────────
__global__ void block_reduce_kernel(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int          n_per_block)
{
    extern __shared__ float smem[];   // 1 float per warp
    int gid = blockIdx.x * n_per_block + threadIdx.x;
    float val = (threadIdx.x < n_per_block) ? src[gid] : 0.f;
    float total = block_reduce_sum(val, smem);
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}

// ─── Full RMSNorm with variable hidden_dim (multi-warp) ──────────────────────
// Each block handles one row. Block size = hidden_dim / 8 threads.
// Works for any hidden_dim divisible by 256 (up to 8192 in practice).
__global__ void rmsnorm_f16_multiblock(
    const uint4* __restrict__ x,
    const uint4* __restrict__ w,
    uint4*       __restrict__ out,
    float        eps,
    int          hidden_dim,
    int          n_vec)        // = hidden_dim / 8
{
    extern __shared__ float smem[];   // [n_warps] floats for block reduce
    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int row     = blockIdx.x;

    // Pass 1: each thread loads 8 fp16, accumulates partial sum_sq
    uint4  raw_x = x[row * n_vec + threadIdx.x];
    __half* xv   = reinterpret_cast<__half*>(&raw_x);

    float local_sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v  = __half2float(xv[i]);
        local_sq += v * v;
    }

    // Two-level block reduce: sum partial sum_sq across all warps
    float total_sq = block_reduce_sum(local_sq, smem);
    float rms_rcp  = rsqrtf(total_sq / hidden_dim + eps);

    // Pass 2: load weight, normalize, store
    uint4  raw_w = w[threadIdx.x];
    __half* wv   = reinterpret_cast<__half*>(&raw_w);
    uint4  raw_out;
    __half* ov   = reinterpret_cast<__half*>(&raw_out);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        ov[i] = __float2half(__half2float(xv[i]) * rms_rcp * __half2float(wv[i]));
    }
    out[row * n_vec + threadIdx.x] = raw_out;
}

torch::Tensor block_reduce(torch::Tensor src, int threads) {
    TORCH_CHECK(src.dtype() == torch::kFloat32);
    TORCH_CHECK(src.numel() % threads == 0);
    int blocks    = src.numel() / threads;
    int n_warps   = threads / 32;
    int smem_size = n_warps * sizeof(float);
    auto out      = torch::zeros({blocks}, src.options());
    block_reduce_kernel<<<blocks, threads, smem_size>>>(
        src.data_ptr<float>(), out.data_ptr<float>(), threads);
    return out;
}

torch::Tensor rmsnorm_f16_var(torch::Tensor x, torch::Tensor w, float eps) {
    TORCH_CHECK(x.dtype() == torch::kFloat16 && w.dtype() == torch::kFloat16);
    TORCH_CHECK(x.dim() == 2);
    int batch      = x.size(0);
    int hidden_dim = x.size(1);
    TORCH_CHECK(hidden_dim % 256 == 0, "hidden_dim must be multiple of 256");
    int n_vec      = hidden_dim / 8;
    int threads    = n_vec;              // one thread per uint4 load
    TORCH_CHECK(threads <= 1024, "hidden_dim too large for single block");
    int n_warps    = threads / 32;
    int smem_size  = n_warps * sizeof(float);
    auto out       = torch::empty_like(x);
    rmsnorm_f16_multiblock<<<batch, threads, smem_size>>>(
        (const uint4*)x.data_ptr(),
        (const uint4*)w.data_ptr(),
        (uint4*)out.data_ptr(),
        eps,
        hidden_dim,
        n_vec);
    return out;
}
"""

cpp_src = """
torch::Tensor block_reduce(torch::Tensor src, int threads);
torch::Tensor rmsnorm_f16_var(torch::Tensor x, torch::Tensor w, float eps);
"""

mod = load_inline(
    name="ex2_5_block_reduce",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["block_reduce", "rmsnorm_f16_var"],
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


# ---- correctness: block reduce ----
print("Block reduce correctness:")
for threads in [32, 64, 128, 256]:
    N = 4096 * threads
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    out = mod.block_reduce(x, threads)
    ref = x.view(-1, threads).sum(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    n_warps = threads // 32
    print(f"  {threads:4d} threads ({n_warps} warps): PASSED")

# ---- correctness: RMSNorm at multiple hidden dims ----
print("\nRMSNorm correctness (multi-warp):")
eps = 1e-5
for hidden in [256, 512, 1024, 2048, 4096]:
    batch = 256
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    w = torch.ones(hidden, device="cuda", dtype=torch.float16)
    out = mod.rmsnorm_f16_var(x, w, eps)
    ref = x.float() / torch.sqrt((x.float() ** 2).mean(dim=1, keepdim=True) + eps)
    ref = ref.to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    n_warps = hidden // 256
    print(f"  hidden={hidden:5d} ({n_warps:2d} warps per block): PASSED")

# ---- benchmark ----
print("\nRMSNorm bandwidth (multi-warp, hidden=4096):")
batch, hidden = 2048, 4096
x_big = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w_big = torch.ones(hidden, device="cuda", dtype=torch.float16)

ms = bench(lambda: mod.rmsnorm_f16_var(x_big, w_big, eps))
bytes_rw = batch * hidden * 2 * 3
bw = bytes_rw / ms / 1e6

print(f"\n{'='*60}")
print(f"  Kernel     : RMSNorm fp16 (two-level block reduce)")
print(f"  Shape      : ({batch}, {hidden})  fp16")
print(f"  Threads    : {hidden // 8} per block  ({hidden // 256} warps)")
print(f"  Time       : {ms:.3f} ms")
print(f"  Bandwidth  : {bw:.1f} GB/s")
print(f"  Peak       : 288 GB/s (RTX 4060 Ti)")
print(f"  Util       : {bw/288*100:.1f}%")
print(f"  Target     : >70% (RMSNorm is memory-bound, not compute-bound)")
print(f"{'='*60}")
print(f"  This is the full SGLang norm.cuh CTA path for hidden_dim > 256.")
