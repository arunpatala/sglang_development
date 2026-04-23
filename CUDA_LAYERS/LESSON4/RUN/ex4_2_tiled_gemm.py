"""
ex4_2_tiled_gemm.py — Tiled GEMM with Shared Memory Blocking
=============================================================
Divides the K dimension into TILE×TILE blocks. All threads in a block
cooperatively load one A tile and one B tile into shared memory, then
each thread accumulates its partial sum from smem — reusing every loaded
element TILE times.

Key idea: smem as a software-managed L1 cache.
  Without tiling: each element of A loaded N times from HBM
  With tiling (TILE=32): each element of A loaded N/32 times from HBM
  → TILE× reduction in HBM traffic

Two-pass bank conflict analysis:
  Write (load) phase:
    A tile: thread (ty, tx) writes As[ty][tx]  → bank tx % 32 → sequential ✓
    B tile: thread (ty, tx) writes Bs[ty][tx]  → bank tx % 32 → sequential ✓
  Compute phase:
    As[ty][k]: ty fixed in warp, k varies in loop → BROADCAST (same addr) ✓
    Bs[k][tx]: k fixed in loop step, tx varies → sequential banks ✓
  Zero bank conflicts — tiled GEMM with 2D blocks needs no swizzle.

Two tile sizes: TILE=16 (smem=2×1KB) and TILE=32 (smem=2×4KB).
TILE=32 is better: each element loaded 2× fewer times, more register-level reuse.

Reference: GEMM_BASICS.md → "Tiled GEMM" section
Target: measurably faster than naive, building intuition for WMMA

Usage:
    python ex4_2_tiled_gemm.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <stdint.h>

// ─── Tiled GEMM TILE=16 (fp32) ────────────────────────────────────────────────
// Block: dim3(16, 16) = 256 threads per block
// Each block computes a 16×16 output tile.
// smem: 2 × float[16][16] = 2 KB
__global__ void tiled_gemm_t16(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[16][16];
    __shared__ float Bs[16][16];

    int row = blockIdx.y * 16 + threadIdx.y;
    int col = blockIdx.x * 16 + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < (K + 15) / 16; ++t) {
        // Cooperatively load A tile: As[ty][tx] = A[row][t*16 + tx]
        int a_col = t * 16 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        // Cooperatively load B tile: Bs[ty][tx] = B[t*16 + ty][col]
        int b_row = t * 16 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();   // barrier 1: smem load complete

        // Compute: thread (ty, tx) accumulates C[row][col] from the current tile
        // As[ty][k]: same ty for all threads in a warp → broadcast (zero conflict)
        // Bs[k][tx]: consecutive tx → sequential banks (zero conflict)
        #pragma unroll
        for (int k = 0; k < 16; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();   // barrier 2: prevent smem overwrite before compute done
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

// ─── Tiled GEMM TILE=32 (fp32) ────────────────────────────────────────────────
// Block: dim3(32, 32) = 1024 threads per block (maximum on sm_89)
// Each block computes a 32×32 output tile.
// smem: 2 × float[32][32] = 8 KB
// Each element loaded 2× less often than TILE=16 → better HBM utilization
__global__ void tiled_gemm_t32(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K)
{
    __shared__ float As[32][32];
    __shared__ float Bs[32][32];

    int row = blockIdx.y * 32 + threadIdx.y;
    int col = blockIdx.x * 32 + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < (K + 31) / 32; ++t) {
        int a_col = t * 32 + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;

        int b_row = t * 32 + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < 32; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = sum;
}

torch::Tensor run_tiled_t16(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (M + 15) / 16);
    tiled_gemm_t16<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor run_tiled_t32(torch::Tensor A, torch::Tensor B) {
    int M = A.size(0), K = A.size(1), N = B.size(1);
    auto C = torch::zeros({M, N}, A.options());
    dim3 block(32, 32);
    dim3 grid((N + 31) / 32, (M + 31) / 32);
    tiled_gemm_t32<<<grid, block>>>(
        A.data_ptr<float>(), B.data_ptr<float>(), C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_src = """
torch::Tensor run_tiled_t16(torch::Tensor A, torch::Tensor B);
torch::Tensor run_tiled_t32(torch::Tensor A, torch::Tensor B);
"""

mod = load_inline(
    name="ex4_2",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_tiled_t16", "run_tiled_t32"],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
    verbose=False,
)


def bench(fn, warmup=10, iters=100):
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


PEAK_TFLOPS = 44.0   # fp16 tensor core peak, RTX 4060 Ti (reference ceiling)
FP32_PEAK   = 15.0   # fp32 CUDA core theoretical peak (approximate)

# ── Correctness ────────────────────────────────────────────────────────────────
print("Correctness tests:")
for M, N, K in [(64, 64, 64), (128, 256, 128), (512, 512, 512), (1024, 1024, 1024),
                (1000, 900, 800)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    ref = torch.mm(A, B)
    t16 = mod.run_tiled_t16(A, B)
    t32 = mod.run_tiled_t32(A, B)
    torch.testing.assert_close(t16, ref, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(t32, ref, rtol=1e-4, atol=1e-4)
    print(f"  M={M:4d} N={N:4d} K={K:4d}: TILE=16 PASSED  TILE=32 PASSED")

print()

# ── Memory reuse analysis ──────────────────────────────────────────────────────
print("Memory reuse analysis (M=N=K=1024):")
M = N = K = 1024
tiles_per_row_K = (K + 31) // 32    # number of K-tiles for TILE=32

print(f"  TILE=32 kernel:")
print(f"    Total K-tiles per block: {tiles_per_row_K} tiles × (32*4 + 32*4) bytes = "
      f"{tiles_per_row_K * 256} bytes loaded per thread")
print(f"    Each smem element reused TILE=32 times in the inner loop")
print(f"    vs naive: same element accessed {N} times (once per output column)")
print(f"  Ideal HBM reads: {(M*K + K*N)*4/1e6:.0f} MB  (each element once)")
print(f"  Naive HBM reads: {2*M*N*K*4/1e9:.1f} GB  (no reuse)")
print(f"  Tiling reduction: ~{N//32}× vs naive for A rows, ~{M//32}× for B columns")
print()

# ── smem usage ────────────────────────────────────────────────────────────────
print("Shared memory usage:")
print(f"  TILE=16: 2 × 16×16×4 = {2*16*16*4} bytes = {2*16*16*4/1024:.2f} KB")
print(f"  TILE=32: 2 × 32×32×4 = {2*32*32*4} bytes = {2*32*32*4/1024:.2f} KB")
print(f"  (Limit: 99 KB per block on sm_89)")
print()

# ── Benchmark ──────────────────────────────────────────────────────────────────
print("Benchmark (fp32 tiled GEMM):")
print(f"  {'M=N=K':>8}  {'kernel':>12}  {'ms':>7}  {'GFLOPS':>8}  {'% fp32 peak':>12}  {'smem':>8}")
print(f"  {'-'*70}")

for size in [512, 1024, 2048]:
    M = N = K = size
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    flops = 2 * M * N * K

    n_iters = 200 if size <= 512 else (50 if size <= 1024 else 20)

    for kernel_name, fn, smem in [
        ("TILE=16", lambda: mod.run_tiled_t16(A, B), "2 KB"),
        ("TILE=32", lambda: mod.run_tiled_t32(A, B), "8 KB"),
        ("torch.mm", lambda: torch.mm(A, B), "cuBLAS"),
    ]:
        ms = bench(fn, warmup=5, iters=n_iters)
        gflops = flops / ms / 1e6
        pct    = gflops / (FP32_PEAK * 1e3) * 100
        mark   = "✓" if pct > 30 else ("~" if pct > 10 else "✗")
        print(f"  {size:>8}  {kernel_name:>12}  {ms:>7.3f}  {gflops:>8.1f}  "
              f"{pct:>11.1f}%  {smem:>8}  {mark}")

print()
print(f"  Reference peak: fp32 CUDA cores ~{FP32_PEAK} TFLOPS | fp16 tensor cores ~{PEAK_TFLOPS} TFLOPS")
print()

# ── Bank conflict demonstration ────────────────────────────────────────────────
print("─" * 60)
print("Bank conflict analysis — TILE=32 inner loop:")
print()
print("  Compute phase reads (thread warp = same ty, different tx):")
print()
print("    As[threadIdx.y][k]: ty is fixed within a warp, k is the loop variable")
print("    → ALL 32 threads in the warp read the SAME address (ty, k)")
print("    → hardware BROADCAST: 1 read serves all 32 threads → zero conflict ✓")
print()
print("    Bs[k][threadIdx.x]: k is fixed within one inner-loop step, tx varies 0..31")
print("    → smem addresses: k*32+0, k*32+1, ..., k*32+31")
print("    → banks: 0, 1, 2, ..., 31 → all distinct → zero conflict ✓")
print()
print("  Load phase:")
print("    As[threadIdx.y][threadIdx.x]: ty varies across warps, tx varies within warp")
print("    → within warp: all write to different banks (tx=0..31) → zero conflict ✓")
print("    Bs: same analysis ✓")
print()
print("  Result: TILE=32 tiled GEMM with 2D block has ZERO bank conflicts.")
print("  Verify with ncu: see profile.sh and ex4_5_inspect_ptx.sh")
print()
print("  Compare: the NAIVE kernel also has zero bank conflicts — but it's slow")
print("  because of L2 cache thrash (global memory, not smem, is the bottleneck).")
print()
print("Next: Exercise 4.3 replaces the inner loop FMAs with WMMA tensor core instructions.")
print("      Same tiling strategy, 8× more compute throughput per smem element loaded.")
