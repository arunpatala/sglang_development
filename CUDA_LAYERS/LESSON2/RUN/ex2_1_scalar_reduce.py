"""
Exercise 2.1 — Scalar Reduce Baseline (shared memory, serial)
=============================================================
Naive block reduce: one thread does all the addition sequentially.
This is the worst-case pattern — establishes a correctness baseline
and a floor to beat in ex2_2.

Each block of 32 threads reduces 32 floats into one output float.

GPU: RTX 4060 Ti | Instruction count: 32 serial adds + 32 smem ops
Compare against ex2_2: the shuffle version uses 5 instructions total.
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Naive block reduce using shared memory.
// One thread (tid=0) does all the work after loading to smem.
// Every other thread is idle during the serial add loop.
__global__ void block_reduce_naive(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int          n_per_block)   // = blockDim.x = 32 for this exercise
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * n_per_block + tid;

    // All threads load their value into shared memory
    smem[tid] = (tid < n_per_block) ? src[gid] : 0.f;
    __syncthreads();

    // Thread 0 serially sums all values — O(N) work on one thread
    if (tid == 0) {
        float total = 0.f;
        for (int i = 0; i < n_per_block; i++) {
            total += smem[i];   // 32 serial reads from smem
        }
        out[blockIdx.x] = total;
    }
}

torch::Tensor naive_reduce(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(src.numel() % 32 == 0, "numel must be divisible by 32");
    int n      = src.numel();
    int blocks = n / 32;
    auto out   = torch::zeros({blocks}, src.options());
    // 32 threads per block, 32 floats of shared memory
    block_reduce_naive<<<blocks, 32, 32 * sizeof(float)>>>(
        src.data_ptr<float>(), out.data_ptr<float>(), 32);
    return out;
}
"""

cpp_src = "torch::Tensor naive_reduce(torch::Tensor src);"

mod = load_inline(
    name="ex2_1_scalar_reduce",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["naive_reduce"],
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
N = 1024 * 32   # 1024 blocks × 32 elements each
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.naive_reduce(x)
ref = x.view(-1, 32).sum(dim=1)
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("Correctness: PASSED")
print(f"  out[0]={out[0].item():.4f}   ref[0]={ref[0].item():.4f}")

# ---- benchmark ----
N_BIG = 1024 * 1024 * 32   # 32M floats = 128 MB
x_big = torch.randn(N_BIG, device="cuda", dtype=torch.float32)

ms = bench(lambda: mod.naive_reduce(x_big))
bytes_rd = N_BIG * 4   # read only (output is 1/32 the size, negligible)
bw = bytes_rd / ms / 1e6

print(f"\n{'='*55}")
print(f"  Kernel     : scalar reduce (smem serial, 32 threads)")
print(f"  Elements   : {N_BIG//1e6:.0f}M float32  ({bytes_rd//1e6:.0f} MB read)")
print(f"  Time       : {ms:.3f} ms")
print(f"  Bandwidth  : {bw:.1f} GB/s  (effective)")
print(f"  Peak       : 288 GB/s (RTX 4060 Ti)")
print(f"  Util       : {bw/288*100:.1f}%")
print(f"  Note       : Only 1/32 threads active during reduce (31/32 idle)")
print(f"{'='*55}")
print(f"  Next: ex2_2 — same reduce using 5 warp shuffles instead of 32 serial adds")
