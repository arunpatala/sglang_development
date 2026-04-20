"""
Exercise 2.2 — Warp Reduce Sum with __shfl_xor_sync
====================================================
Butterfly warp reduction: 5 rounds of XOR shuffle, all 32 lanes active
simultaneously every round. Zero shared memory. Every lane receives the
full sum at the end.

XOR offsets: 16 → 8 → 4 → 2 → 1  (log2(32) = 5 steps)

GPU: RTX 4060 Ti | PTX: 5 × shfl.sync.bfly.b32 instructions
Compare against ex2_1: 5 instructions vs 32 serial adds.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Butterfly warp-level sum reduction using __shfl_xor_sync.
// 5 rounds: XOR with 16, 8, 4, 2, 1.
// After round k, each lane holds the partial sum of 2^k original values.
// After round 5: every lane holds the complete sum of all 32 inputs.
// Zero shared memory. Zero __syncthreads(). 5 PTX instructions.
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);   // lane i ↔ lane (i XOR 16)
    val += __shfl_xor_sync(0xffffffff, val,  8);   // lane i ↔ lane (i XOR 8)
    val += __shfl_xor_sync(0xffffffff, val,  4);   // lane i ↔ lane (i XOR 4)
    val += __shfl_xor_sync(0xffffffff, val,  2);   // lane i ↔ lane (i XOR 2)
    val += __shfl_xor_sync(0xffffffff, val,  1);   // lane i ↔ lane (i XOR 1)
    return val;   // every lane returns the same total
}

// One warp (32 threads) per block.
// Each block reduces 32 consecutive floats → one output float.
__global__ void reduce_warp_sum(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    int lane = threadIdx.x;              // 0–31
    int gid  = blockIdx.x * 32 + lane;  // global element index

    float val   = src[gid];
    float total = warp_reduce_sum(val);  // all 32 lanes participate

    // Every lane holds the correct sum — only lane 0 needs to write
    if (lane == 0) out[blockIdx.x] = total;
}

torch::Tensor warp_reduce(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(src.numel() % 32 == 0, "numel must be divisible by 32");
    int blocks = src.numel() / 32;
    auto out   = torch::zeros({blocks}, src.options());
    // Exactly 32 threads per block — one warp, no smem needed
    reduce_warp_sum<<<blocks, 32>>>(src.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
"""

cpp_src = "torch::Tensor warp_reduce(torch::Tensor src);"

mod = load_inline(
    name="ex2_2_warp_reduce_sum",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["warp_reduce"],
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
    return s.elapsed_time(e) / iters   # ms


# ---- correctness ----
N = 1024 * 32
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.warp_reduce(x)
ref = x.view(-1, 32).sum(dim=1)
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("Correctness: PASSED")
print(f"  out[0]={out[0].item():.4f}   ref[0]={ref[0].item():.4f}")

# ---- benchmark ----
N_BIG = 1024 * 1024 * 32   # 32M floats = 128 MB
x_big = torch.randn(N_BIG, device="cuda", dtype=torch.float32)

ms = bench(lambda: mod.warp_reduce(x_big))
bytes_rd = N_BIG * 4
bw = bytes_rd / ms / 1e6

print(f"\n{'='*55}")
print(f"  Kernel     : warp_reduce_sum (butterfly, __shfl_xor_sync)")
print(f"  Elements   : {N_BIG//1e6:.0f}M float32  ({bytes_rd//1e6:.0f} MB read)")
print(f"  Time       : {ms:.3f} ms")
print(f"  Bandwidth  : {bw:.1f} GB/s  (effective)")
print(f"  Peak       : 288 GB/s (RTX 4060 Ti)")
print(f"  Util       : {bw/288*100:.1f}%")
print(f"  PTX        : 5 × shfl.sync.bfly.b32  (zero smem)")
print(f"{'='*55}")
print(f"  This is the inner loop of RMSNorm pass-1 and attention softmax.")
print(f"  Next: ex2_3 — same pattern for max instead of sum.")
