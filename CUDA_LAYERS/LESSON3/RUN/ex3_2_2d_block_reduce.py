"""
ex3_2_2d_block_reduce.py — Block Reduce with 2D Thread Block
=============================================================
Implements the cross-warp reduction using the 2D block layout:
    dim3(32, num_warps)
      threadIdx.x = lane within warp (0–31)
      threadIdx.y = warp index (0–num_warps-1)

This is the exact layout used in norm.cuh (RMSNorm).

Algorithm:
  1. Each thread accumulates a partial sum over its slice of the row
  2. Warp-level butterfly reduces 32 lanes to 1 value (no smem)
  3. Lane 0 of each warp writes partial to smem[warp_id]
  4. __syncthreads() — barrier
  5. Warp 0 reads smem[lane] for all warp partial sums
  6. Warp 0 butterfly reduces → final sum in lane 0
  7. Lane 0 writes smem[0], __syncthreads()
  8. All threads read smem[0]

Usage:
    python ex3_2_2d_block_reduce.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <stdint.h>

// ─── 2D block reduce ──────────────────────────────────────────────────────────
// threadIdx.x = lane (0–31), threadIdx.y = warp_id (0–num_warps-1)
// smem must have at least blockDim.y floats pre-allocated
__device__ __forceinline__
float block_reduce_sum_2d(float val, float* smem) {
    int lane    = threadIdx.x;
    int warp_id = threadIdx.y;
    int n_warps = blockDim.y;

    // Step 1: intra-warp butterfly (5 shuffles — Lesson 2)
    // After this, every lane in the warp holds the warp's partial sum.
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);

    // Step 2: lane 0 of each warp writes its partial sum to smem.
    // smem[0]→bank 0, smem[1]→bank 1, ..., smem[n_warps-1]→bank n_warps-1.
    // n_warps ≤ 32 always (blockDim.x * blockDim.y ≤ 1024, blockDim.x=32 → blockDim.y ≤ 32)
    // So we write to at most 32 distinct banks — zero conflicts.
    if (lane == 0) smem[warp_id] = val;
    __syncthreads();    // ← barrier 1: all warp partial sums in smem before warp 0 reads

    // Step 3: warp 0 loads partial sums and reduces them.
    // lane i reads smem[i] → bank i → zero conflicts.
    // Threads beyond n_warps contribute 0 (identity for sum).
    val = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;
    if (warp_id == 0) {
        val += __shfl_xor_sync(0xffffffff, val, 16);
        val += __shfl_xor_sync(0xffffffff, val,  8);
        val += __shfl_xor_sync(0xffffffff, val,  4);
        val += __shfl_xor_sync(0xffffffff, val,  2);
        val += __shfl_xor_sync(0xffffffff, val,  1);
        // Broadcast: write total to smem[0] so ALL warps can read it
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();    // ← barrier 2: smem[0] written before all warps read it

    return smem[0];     // valid in every thread
}

// Row-wise sum kernel.
// One block per row, launched as dim3(32, n_warps).
// Each thread processes n_per_row / (32 * n_warps) floats via strided loop.
__global__ void row_sum_2d(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int n_per_row)
{
    extern __shared__ float smem[];

    int row     = blockIdx.x;
    int n_th    = blockDim.x * blockDim.y;           // total threads per block
    int tid     = threadIdx.y * blockDim.x + threadIdx.x;   // flat thread id

    // Each thread sums its slice of the row
    float partial = 0.f;
    for (int i = tid; i < n_per_row; i += n_th)
        partial += src[row * n_per_row + i];

    float total = block_reduce_sum_2d(partial, smem);

    // Only one thread writes the result
    if (threadIdx.x == 0 && threadIdx.y == 0)
        out[row] = total;
}

torch::Tensor row_sum_2d_kernel(torch::Tensor src, int n_warps) {
    TORCH_CHECK(src.dim() == 2, "src must be 2D [batch, n_per_row]");
    int batch     = src.size(0);
    int n_per_row = src.size(1);

    auto out = torch::zeros({batch}, src.options());
    dim3 block(32, n_warps);
    int smem_bytes = n_warps * sizeof(float);

    row_sum_2d<<<batch, block, smem_bytes>>>(
        src.data_ptr<float>(), out.data_ptr<float>(), n_per_row);
    return out;
}
"""

cpp_src = "torch::Tensor row_sum_2d_kernel(torch::Tensor src, int n_warps);"

mod = load_inline(
    name="ex3_2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["row_sum_2d_kernel"],
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


# ── Correctness: test all warp counts and hidden dims ────────────────────────
print("Correctness tests:")
for n_warps in [1, 2, 4, 8, 16]:
    for n_per_row in [32, 256, 512, 1024, 4096]:
        if n_per_row < 32 * n_warps:
            continue   # need at least one element per thread
        batch = 256
        x = torch.randn(batch, n_per_row, device="cuda", dtype=torch.float32)
        out = mod.row_sum_2d_kernel(x, n_warps)
        ref = x.sum(dim=1)
        torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    print(f"  {n_warps:2d} warps  (hidden sizes 32–4096): PASSED")

print()

# ── Diagram: what each thread does ────────────────────────────────────────────
print("Thread layout for dim3(32, 4) — 4 warps, 128 threads total:")
print()
print("  threadIdx.y (warp_id):  0      1      2      3")
print("  threadIdx.x (lane):    0–31   0–31   0–31   0–31")
print()
print("  For hidden=512, n_vec=64:")
print("    Each thread covers: 64/(32*4) = 0.5 chunks → strided loop")
print("    tid=0  processes vec[0],  vec[128], ...")
print("    tid=1  processes vec[1],  vec[129], ...")
print("    tid=127 processes vec[127], vec[255], ...")
print()
print("  smem layout after Step 2:")
print("    smem[0] = warp 0 partial sum  (bank 0)")
print("    smem[1] = warp 1 partial sum  (bank 1)")
print("    smem[2] = warp 2 partial sum  (bank 2)")
print("    smem[3] = warp 3 partial sum  (bank 3)")
print("    → 4 distinct banks, zero conflicts")
print()

# ── Smem usage ────────────────────────────────────────────────────────────────
print("Smem usage:")
for n_warps in [1, 2, 4, 8, 16]:
    smem = n_warps * 4
    print(f"  {n_warps:2d} warps → smem = {smem:3d} bytes  "
          f"({smem/1024:.3f} KB of 99 KB limit)")
print()
print("The smem footprint for cross-warp reduce is negligible.")
print("It frees up smem for tile caching in Flash Attention (Lesson 5+).")
print()

# ── Self-test question ────────────────────────────────────────────────────────
print("─" * 60)
print("Self-test question:")
print()
print("  The block reduce uses two __syncthreads(). Can you eliminate")
print("  the second one?")
print()
print("  Answer: NO. After warp 0 writes smem[0], other warps (warp_id > 0)")
print("  have not seen the write yet. Without the second __syncthreads(),")
print("  warps 1–N would read smem[0] before warp 0 has written it.")
print("  They would read the old value (from the first __syncthreads() era)")
print("  or garbage — a data race.")
print()
print("  The second barrier guarantees: warp 0's write to smem[0] is")
print("  visible to all threads before any thread reads smem[0].")
