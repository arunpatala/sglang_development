"""
ex4_3_wmma_gemm.py — WMMA GEMM: fp16 Tensor Cores
===================================================
Uses nvcuda::wmma (WMMA API) to execute 16×16×16 matrix-multiply-accumulate
instructions on the RTX 4060 Ti's 4th-generation tensor cores.

Two versions:
  v1 (no smem): One warp per 16×16 output tile. Each warp loads from global
     memory directly. Simple to understand, but bandwidth-limited when many
     warps reload overlapping A/B regions.

  v2 (smem tiled): 16 warps per block compute a 64×64 output tile. A and B
     tiles are loaded cooperatively into smem, then each warp issues WMMA
     from smem. Same tiling benefit as tiled GEMM, but with tensor cores.

WMMA API summary (from TENSOR_CORES.md):
  1. Declare: wmma::fragment<matrix_a / matrix_b / accumulator, 16,16,16, T, layout>
  2. Load:    wmma::load_matrix_sync(frag, ptr, ldm)
  3. MMA:     wmma::mma_sync(d_frag, a_frag, b_frag, c_frag)
  4. Store:   wmma::store_matrix_sync(ptr, frag, ldm, layout)

Requirement: M, N, K must be multiples of 16 (enforced in Python launcher).
             v2 additionally requires M, N multiples of 64 for simplicity.

Reference: TENSOR_CORES.md, REPOS/flashinfer/include/flashinfer/mma.cuh
Target: >22 TFLOPS = >50% of 44 TFLOPS peak fp16 on RTX 4060 Ti

Usage:
    python ex4_3_wmma_gemm.py
"""

import torch
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

// ─── v1: One warp per 16×16 output tile, global memory loads ──────────────────
//
// Grid:  (N/16, M/16) — one block per 16×16 output tile
// Block: 32 threads = one warp
//
// This warp:
//   - owns C tile: rows [tile_row*16 .. +16], cols [tile_col*16 .. +16]
//   - loops over K in steps of 16
//   - loads a_frag from A[tile_row*16..+16][k..k+16]  (ldm = K)
//   - loads b_frag from B[k..k+16][tile_col*16..+16]  (ldm = N)
//   - accumulates: c_frag += a_frag @ b_frag
//   - stores c_frag to C[tile_row*16..+16][tile_col*16..+16]
//
// Requires: M, N, K all divisible by 16
__global__ void wmma_gemm_v1(
    const __half* __restrict__ A,   // [M, K] fp16, row-major
    const __half* __restrict__ B,   // [K, N] fp16, row-major
    float*        __restrict__ C,   // [M, N] fp32 output
    int M, int N, int K)
{
    int tile_row = blockIdx.y;   // index into the M/16 grid
    int tile_col = blockIdx.x;   // index into the N/16 grid

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float>                c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    int row_offset = tile_row * 16;
    int col_offset = tile_col * 16;

    for (int k = 0; k < K; k += 16) {
        // Load A sub-tile: A[row_offset .. +16][k .. +16]
        wmma::load_matrix_sync(a_frag, A + row_offset * K + k, K);
        // Load B sub-tile: B[k .. +16][col_offset .. +16]
        wmma::load_matrix_sync(b_frag, B + k * N + col_offset, N);
        // Tensor core MMA: c_frag += a_frag @ b_frag
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    // Store fp32 accumulator to output
    wmma::store_matrix_sync(C + row_offset * N + col_offset, c_frag, N,
                             wmma::mem_row_major);
}

// ─── v2: 16 warps per block, smem-tiled, 64×64 output per block ───────────────
//
// Block: 512 threads = 16 warps, arranged as 4×4 in the output tile
//   warp_row = warp_id / 4  (0..3) — which 16-row sub-tile
//   warp_col = warp_id % 4  (0..3) — which 16-col sub-tile
//
// Grid:  (N/64, M/64)
//
// smem:
//   As[64][16] __half = 64*16*2 = 2048 bytes
//   Bs[16][64] __half = 16*64*2 = 2048 bytes
//   Total: 4 KB (well within 99 KB limit)
//
// For each K tile of size BK=16:
//   1. All 512 threads cooperatively load As and Bs from global
//   2. __syncthreads()
//   3. Each warp loads its 16×16 WMMA fragments from smem
//   4. wmma::mma_sync accumulates
//   5. __syncthreads() before next tile
//
// Requires: M, N divisible by 64, K divisible by 16

const int BM = 64;   // output block height
const int BN = 64;   // output block width
const int BK = 16;   // K-tile depth (= WMMA_K)

__global__ void wmma_gemm_v2(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    float*        __restrict__ C,
    int M, int N, int K)
{
    __shared__ __half As[BM][BK];   // A tile: 64 rows × 16 K-cols  (2 KB)
    __shared__ __half Bs[BK][BN];   // B tile: 16 K-rows × 64 cols  (2 KB)

    int warp_id  = threadIdx.x / 32;   // 0..15 (16 warps per block)
    int warp_row = warp_id / 4;        // 0..3 (row in the 4×4 warp grid)
    int warp_col = warp_id % 4;        // 0..3 (col in the 4×4 warp grid)

    int block_row = blockIdx.y * BM;   // first global row this block covers
    int block_col = blockIdx.x * BN;   // first global col this block covers

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int k_start = 0; k_start < K; k_start += BK) {
        // ── Cooperative load of A tile [BM, BK] from global memory ──────────
        // 512 threads load BM*BK = 64*16 = 1024 fp16 elements
        // Each thread loads 2 elements (one per loop pass)
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            int r = idx / BK, c = idx % BK;
            int gr = block_row + r;
            int gk = k_start + c;
            As[r][c] = (gr < M && gk < K) ? A[gr * K + gk] : __float2half(0.f);
        }

        // ── Cooperative load of B tile [BK, BN] from global memory ──────────
        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            int r = idx / BN, c = idx % BN;
            int gk = k_start + r;
            int gc = block_col + c;
            Bs[r][c] = (gk < K && gc < N) ? B[gk * N + gc] : __float2half(0.f);
        }
        __syncthreads();   // barrier: all smem tiles fully loaded

        // ── Each warp computes its 16×16 sub-tile via WMMA ──────────────────
        wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

        // A sub-tile for this warp: As[warp_row*16 .. +16][0 .. 16] (ldm = BK)
        wmma::load_matrix_sync(a_frag, &As[warp_row * 16][0], BK);
        // B sub-tile for this warp: Bs[0 .. 16][warp_col*16 .. +16] (ldm = BN)
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_col * 16], BN);

        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();   // barrier: protect smem before next tile load
    }

    // ── Store this warp's 16×16 output tile ─────────────────────────────────
    int out_row = block_row + warp_row * 16;
    int out_col = block_col + warp_col * 16;
    if (out_row < M && out_col < N)
        wmma::store_matrix_sync(C + out_row * N + out_col, c_frag, N,
                                 wmma::mem_row_major);
}

// ─── Launchers ────────────────────────────────────────────────────────────────

torch::Tensor run_wmma_v1(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "A and B must be fp16");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(M % 16 == 0 && N % 16 == 0 && K % 16 == 0,
                "M, N, K must be multiples of 16");

    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32)
                                                         .device(A.device()));
    dim3 grid(N / 16, M / 16);
    dim3 block(32);    // one warp per block

    wmma_gemm_v1<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr()),
        reinterpret_cast<const __half*>(B.data_ptr()),
        C.data_ptr<float>(), M, N, K);
    return C;
}

torch::Tensor run_wmma_v2(torch::Tensor A, torch::Tensor B) {
    TORCH_CHECK(A.dtype() == torch::kFloat16 && B.dtype() == torch::kFloat16,
                "A and B must be fp16");
    int M = A.size(0), K = A.size(1), N = B.size(1);
    TORCH_CHECK(M % 64 == 0 && N % 64 == 0 && K % 16 == 0,
                "M, N must be multiples of 64; K must be a multiple of 16");

    auto C = torch::zeros({M, N}, torch::TensorOptions().dtype(torch::kFloat32)
                                                         .device(A.device()));
    dim3 grid(N / BN, M / BM);
    dim3 block(512);   // 16 warps = 512 threads

    wmma_gemm_v2<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr()),
        reinterpret_cast<const __half*>(B.data_ptr()),
        C.data_ptr<float>(), M, N, K);
    return C;
}
"""

cpp_src = """
torch::Tensor run_wmma_v1(torch::Tensor A, torch::Tensor B);
torch::Tensor run_wmma_v2(torch::Tensor A, torch::Tensor B);
"""

mod = load_inline(
    name="ex4_3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_wmma_v1", "run_wmma_v2"],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
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


def pad_to_multiple(x, multiple):
    """Pad tensor rows and cols to the next multiple."""
    m, n = x.shape
    pm = (m + multiple - 1) // multiple * multiple
    pn = (n + multiple - 1) // multiple * multiple
    if pm == m and pn == n:
        return x
    return F.pad(x, (0, pn - n, 0, pm - m))


PEAK_TFLOPS = 44.0   # RTX 4060 Ti fp16 tensor core peak

# ── Correctness ────────────────────────────────────────────────────────────────
print("Correctness tests (fp16 inputs, fp32 accumulation):")

for M, N, K in [(16, 16, 16), (64, 64, 64), (256, 256, 256), (1024, 1024, 1024)]:
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)

    # Reference: float32 matmul
    ref = torch.mm(A.float(), B.float())

    # v1: needs M, N, K % 16 == 0
    out_v1 = mod.run_wmma_v1(A, B)
    torch.testing.assert_close(out_v1, ref, rtol=1e-2, atol=1e-2)
    print(f"  M={M:4d} N={N:4d} K={K:4d}: v1 PASSED", end="")

    # v2: needs M, N % 64, K % 16
    if M % 64 == 0 and N % 64 == 0:
        out_v2 = mod.run_wmma_v2(A, B)
        torch.testing.assert_close(out_v2, ref, rtol=1e-2, atol=1e-2)
        print("  v2 PASSED")
    else:
        print("  v2 skipped (need M,N%64==0)")

print()

# ── Fragment layout visualization ─────────────────────────────────────────────
print("Fragment info (from TENSOR_CORES.md):")
print("  wmma::fragment<accumulator, 16,16,16, float>:")
print("    num_elements = 8 per thread")
print("    32 threads × 8 floats = 256 = full 16×16 output tile")
print("    Layout: hardware-defined, opaque to programmer")
print()
print("  wmma::load_matrix_sync(a_frag, ptr, ldm):")
print("    Loads 16×16 = 256 fp16 values distributed across 32 threads")
print("    ptr:  top-left of the 16×16 sub-matrix")
print("    ldm:  leading dimension (stride in elements between rows)")
print("    Example: A[4096][4096], tile at row=0, col=16 → ldm=4096, ptr=A+16")
print()

# ── WMMA performance vs WMMA smem ──────────────────────────────────────────────
print("Benchmark (fp16 WMMA GEMM):")
print(f"  {'M=N=K':>8}  {'version':>10}  {'ms':>7}  {'TFLOPS':>8}  {'% peak':>8}  {'notes'}")
print(f"  {'-'*65}")

for size in [1024, 2048]:
    M = N = K = size
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    flops = 2 * M * N * K

    for version, fn in [
        ("v1 global", lambda: mod.run_wmma_v1(A, B)),
        ("v2 smem  ", lambda: mod.run_wmma_v2(A, B)),
        ("torch.mm ", lambda: torch.mm(A, B).float()),
    ]:
        ms = bench(fn, warmup=10, iters=200)
        tflops = flops / ms / 1e9
        pct    = tflops / PEAK_TFLOPS * 100
        mark   = "✓" if pct > 50 else ("~" if pct > 20 else "✗")
        print(f"  {size:>8}  {version:>10}  {ms:>7.3f}  {tflops:>8.2f}  {pct:>7.1f}%  {mark}")

    print()

# ── v1 vs v2 analysis ──────────────────────────────────────────────────────────
print("─" * 60)
M = N = K = 2048
A = torch.randn(M, K, device="cuda", dtype=torch.float16)
B = torch.randn(K, N, device="cuda", dtype=torch.float16)

ms_v1 = bench(lambda: mod.run_wmma_v1(A, B))
ms_v2 = bench(lambda: mod.run_wmma_v2(A, B))

print(f"v1 vs v2 at M=N=K={M}:")
print(f"  v1 (global loads): {ms_v1:.3f} ms")
print(f"  v2 (smem tiled):   {ms_v2:.3f} ms  ({ms_v1/ms_v2:.1f}× faster)")
print()
print("Why v2 is faster than v1:")
print("  v1: Each warp issues K/16 WMMA fragment loads from global memory.")
print("      Adjacent warps (different tile rows) reload the SAME B columns.")
print("      For N/16 × M/16 warps total, each B element is loaded M/16 times from HBM.")
print()
print("  v2: 16 warps share one smem tile. The A and B tiles are loaded once into smem")
print("      by all 512 threads together, then each warp issues WMMA from smem.")
print(f"      Each B element loaded only M/64 = {M//64} times from HBM vs M/16 = {M//16} times.")
print(f"      → {(M//16)//(M//64)}× reduction in redundant B reads from HBM.")
print()
print("v2 is still not at cuBLAS/peak because:")
print("  - smem bank conflicts possible on WMMA fragment loads (BK=16, fp16 layout)")
print("    → production CUTLASS applies XOR swizzle (see SMEM_SWIZZLE.md)")
print("  - No double-buffering (cp.async): smem load and compute not overlapped")
print("    → production Flash Attention uses cp.async pipeline (Lesson 6)")
print("  - Only 4 WMMA tiles per warp per block (4×4 warp grid, 1 WMMA per tile)")
print("    → CUTLASS uses register-level tiling (each thread accumulates a micro-tile)")
print()
print("But for learning: v2 IS the structure that all production attention uses.")
print("The Flash Attention prefill kernel = v2 + causal mask + cp.async pipeline.")
print()
print("PTX to inspect (see ex4_5_inspect_ptx.sh):")
print("  wmma_gemm_v1 → look for: wmma.load, wmma.mma.sync, wmma.store")
print("  wmma_gemm_v2 → same, but loads come from smem addresses (As[], Bs[])")
print("  SASS:          HMMA.16816.F32 — the actual 4th-gen tensor core instruction")
