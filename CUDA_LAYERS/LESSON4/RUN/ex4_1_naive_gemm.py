"""
ex4_1_naive_gemm.py — Naive GEMM: One Thread Per Output Element
================================================================
The simplest possible GEMM: thread (ty, tx) computes C[row][col] by iterating
over the K dimension. No shared memory, no tensor cores — just global memory.

This is the baseline every subsequent GEMM kernel is compared against.
It deliberately performs poorly to show WHY tiling and tensor cores are needed.

Algorithm:
  C[row][col] = sum over k of A[row][k] * B[k][col]
  Each thread reads K elements from A[row][0..K] and K elements from B[0..K][col]
  → O(K) global memory reads per output element, no reuse

Expected performance: VERY SLOW — 0.1–1% of peak TFLOPS.
The bottleneck is L2 cache thrash: adjacent threads race over the same rows of B.

Usage:
    python ex4_1_naive_gemm.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <stdint.h>

// ─── Naive GEMM (fp32) ────────────────────────────────────────────────────────
//
// Grid:  (ceil(N/BLOCK), ceil(M/BLOCK))  — one block per BLOCK×BLOCK output tile
// Block: (BLOCK, BLOCK) = BLOCK² threads
// Thread (threadIdx.x, threadIdx.y) computes one output element C[row][col].
//
// Memory access pattern:
//   Each thread reads A[row][0..K-1] sequentially (coalesced across the loop body)
//   Each thread reads B[0..K-1][col] with stride N floats (column read — not great)
//   Within a warp: different threads read different cols of the same B row → coalesced ✓
//   But different k iterations of the same thread read different rows of B → stride N

__global__ void naive_gemm_fp32(
    const float* __restrict__ A,  // [M, K] row-major
    const float* __restrict__ B,  // [K, N] row-major
    float*       __restrict__ C,  // [M, N] output
    int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];

    C[row * N + col] = sum;
}

torch::Tensor run_naive_gemm(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat32 && B.dtype() == torch::kFloat32,
                "A and B must be fp32");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "inputs must be 2D");
    TORCH_CHECK(A.size(1) == B.size(0), "A.K must equal B.K");
    TORCH_CHECK(A.is_contiguous() && B.is_contiguous(), "inputs must be contiguous");

    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());

    const int BLOCK = 16;
    dim3 block(BLOCK, BLOCK);
    dim3 grid((N + BLOCK - 1) / BLOCK, (M + BLOCK - 1) / BLOCK);

    naive_gemm_fp32<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_src = "torch::Tensor run_naive_gemm(torch::Tensor A, torch::Tensor B);"

mod = load_inline(
    name="ex4_1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_naive_gemm"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)


def bench(fn, warmup=5, iters=20):
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


PEAK_TFLOPS = 44.0  # fp16 tensor core peak, RTX 4060 Ti
PEAK_BW     = 288.0  # GB/s

# ── Correctness ────────────────────────────────────────────────────────────────
print("Correctness tests:")
for M, N, K in [(64, 64, 64), (128, 256, 128), (512, 512, 512), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    out  = mod.run_naive_gemm(A, B)
    ref  = torch.mm(A, B)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    print(f"  M={M:4d} N={N:4d} K={K:4d}: PASSED")

print()

# ── Thread access pattern explanation ─────────────────────────────────────────
print("Thread access pattern analysis (M=N=K=1024):")
M = N = K = 1024
print(f"  Output elements:       {M*N:,} (each computed by one thread)")
print(f"  Threads in block:      16×16 = 256")
print(f"  Total blocks:          {(N//16) * (M//16)} = {N//16}×{M//16}")
print()
print(f"  Per thread: reads A[row][0..{K-1}] = {K*4} bytes (stride-1, good)")
print(f"              reads B[0..{K-1}][col] = {K*4} bytes (stride={N*4}B, bad for L2)")
print()
print(f"  Total A reads:   {M*K*4/1e6:.0f} MB (each row read N={N} times)")
print(f"  Total B reads:   {K*N*4/1e6:.0f} MB (each col read M={M} times)")
print(f"  Ideal total:     {(M*K + K*N)*4/1e6:.0f} MB if perfectly cached")
print(f"  Without caching: {2*M*N*K*4/1e9:.1f} GB (A and B each read M or N times)")
print()

# ── Bandwidth bottleneck analysis ─────────────────────────────────────────────
print("Roofline analysis (M=N=K=1024):")
flops = 2 * M * N * K
bytes_ideal  = (M * K + K * N + M * N) * 4  # if perfectly cached
bytes_naive  = 2 * M * N * K * 4           # worst-case (no reuse at all)
arith_ideal  = flops / bytes_ideal
arith_naive  = flops / bytes_naive
ridge = PEAK_TFLOPS * 1e12 / (PEAK_BW * 1e9)

print(f"  FLOPs:                    {flops/1e9:.2f} GFLOP")
print(f"  Bytes (ideal, cached):    {bytes_ideal/1e6:.0f} MB")
print(f"  Bytes (no reuse, worst):  {bytes_naive/1e9:.1f} GB")
print(f"  Arithmetic intensity:     {arith_ideal:.0f} FLOP/byte (ideal) | {arith_naive:.2f} FLOP/byte (worst)")
print(f"  Ridge point:              {ridge:.0f} FLOP/byte")
print(f"  GEMM is:                  compute-bound if arith intensity > {ridge:.0f}")
print(f"  Naive is:                 memory-bound (cache thrash degrades intensity)")
print()

# ── Benchmark ─────────────────────────────────────────────────────────────────
print("Benchmark (fp32 naive GEMM):")
print(f"  {'M=N=K':>8}  {'threads':>7}  {'ms':>8}  {'GFLOPS':>8}  {'% peak':>8}  {'Notes'}")
print(f"  {'-'*70}")

for size in [256, 512, 1024]:
    M = N = K = size
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)

    n_iters = 100 if size <= 512 else 10
    ms = bench(lambda: mod.run_naive_gemm(A, B), warmup=3, iters=n_iters)
    gflops = 2 * M * N * K / ms / 1e6
    pct    = gflops / (PEAK_TFLOPS * 1e3) * 100  # vs fp16 tensor core peak
    blocks = ((N + 15) // 16) * ((M + 15) // 16)
    note   = "memory-bound, L2 thrash" if pct < 5 else "approaching compute-bound"
    print(f"  {size:>8}  {blocks:>7}  {ms:>8.2f}  {gflops:>8.1f}  {pct:>7.3f}%  {note}")

print()

# ── Why it's slow: measurement ─────────────────────────────────────────────────
print("─" * 60)
M = N = K = 512
A = torch.randn(M, K, device="cuda", dtype=torch.float32)
B = torch.randn(K, N, device="cuda", dtype=torch.float32)

ms_naive = bench(lambda: mod.run_naive_gemm(A, B), warmup=3, iters=50)
ms_torch = bench(lambda: torch.mm(A, B), warmup=5, iters=200)
gflops_naive = 2 * M * N * K / ms_naive / 1e6
gflops_torch = 2 * M * N * K / ms_torch / 1e6

print(f"Comparison at M=N=K=512:")
print(f"  Naive kernel:  {ms_naive:.3f} ms  | {gflops_naive:.1f} GFLOPS  (fp32, no reuse)")
print(f"  torch.mm:      {ms_torch:.3f} ms  | {gflops_torch:.1f} GFLOPS  (cuBLAS, fp32)")
print(f"  Speedup from cuBLAS: {ms_naive / ms_torch:.0f}×")
print()
print("The naive kernel is slow because:")
print("  1. Each A element is read N times by different threads — L2 evicted before reuse")
print("  2. Each B column access strides N floats = 4 KB apart — L2 cache-line wasted")
print("  3. No instruction-level pipelining: each FMA waits for the previous load")
print("  4. Low arithmetic intensity: L2 bandwidth saturated, tensor cores idle")
print()
print("Exercise 4.2 (tiled GEMM) fixes problem 1 and 2 using shared memory.")
print("Exercise 4.3 (WMMA) additionally uses tensor cores for 8× compute throughput.")
print()
print("ncu profiling command for this kernel:")
print("  ncu --kernel-name naive_gemm_fp32 \\")
print("      --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\\")
print("  sm__throughput.avg.pct_of_peak_sustained_elapsed,\\")
print("  sm__warps_active.avg.pct_of_peak_sustained_active \\")
print("      python ex4_1_naive_gemm.py")
print()
print("Expected: sm__throughput very low (<10%), dram_throughput moderate")
print("The kernel is limited by L2 bandwidth and memory access latency, not compute.")
