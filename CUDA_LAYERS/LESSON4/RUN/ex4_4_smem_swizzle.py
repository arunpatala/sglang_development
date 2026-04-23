"""
ex4_4_smem_swizzle.py — Shared Memory Bank Conflicts and the XOR Swizzle
=========================================================================
Demonstrates bank conflicts using matrix transpose as a clean example.
The same principle applies to WMMA fragment loads in Flash Attention.

Matrix transpose: out = in^T  (N×N square matrix)

Three versions:
  A (naive):    write tile[ty][tx], read tile[tx][ty] — 32-way conflict on reads
  B (padded):   tile[TILE][TILE+1] — +1 column shifts banks, zero conflicts
  C (swizzled): write tile[ty][tx^ty], read tile[tx][ty^tx] — zero conflicts,
                no extra memory

Why the conflict happens (naive read):
  All 32 threads in a warp have ty fixed (warp = same y, different x).
  Reading tile[tx][ty]: address = (tx * 32 + ty) * 4 bytes.
  bank = (tx * 32 + ty) % 32 = ty  ← CONSTANT for the warp.
  All 32 threads hit bank ty → 32-way serialization → 32× slower.

Why XOR swizzle fixes it:
  Reading tile[tx][ty^tx]: address = (tx * 32 + ty^tx) * 4 bytes.
  bank = ty ^ tx  ← for tx=0..31 with fixed ty: ty^0, ty^1, ..., ty^31 = all distinct.
  32 different banks → zero conflict.

Bandwidth target: a well-written transpose should hit >80% of 288 GB/s.
  For N×N float: 2 × N² × 4 bytes per transpose (one read, one write).

Reference: SMEM_SWIZZLE.md, REPOS/flashinfer/include/flashinfer/permuted_smem.cuh

Usage:
    python ex4_4_smem_swizzle.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <stdint.h>

// Matrix transpose configuration
// TILE_DIM  = smem tile side length (must be a power of 2 ≤ 32 for XOR to work cleanly)
// BLOCK_ROWS = number of rows each thread block covers per load iteration
//              = blockDim.y  (must divide TILE_DIM evenly)
#define TILE_DIM   32
#define BLOCK_ROWS  8

// ─── Version A: Naive (32-way bank conflict on reads) ─────────────────────────
//
// Write: thread (ty, tx) → tile[ty][tx]    bank = tx  (distinct per warp) ✓
// Read:  thread (ty, tx) → tile[tx][ty]    bank = ty  (SAME for all in warp) ✗
//        → 32-way serialization on every read transaction
__global__ void transpose_naive(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

    int x = blockIdx.x * TILE_DIM + threadIdx.x;   // global col
    int y = blockIdx.y * TILE_DIM + threadIdx.y;   // global row

    // Write to smem — no conflict: thread i reads in[...] and writes tile[ty][tx]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];

    __syncthreads();

    // Transposed output coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Read from smem — 32-way conflict: tile[tx][ty] has bank = ty (all same in warp)
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
            //                            ↑ column read: address = tx*32+ty+j → bank = ty+j
}

// ─── Version B: Padded (+1 column per row) ─────────────────────────────────
//
// Padding: tile[TILE_DIM][TILE_DIM + 1] instead of tile[TILE_DIM][TILE_DIM]
// Row stride = 33 floats instead of 32 floats
//
// Read:  thread (ty, tx) → tile[tx][ty]
//        address = (tx * 33 + ty) * 4 bytes
//        bank = (tx * 33 + ty) % 32 = (tx * (32+1) + ty) % 32 = (tx + ty) % 32
//        For warp (ty fixed, tx=0..31): banks = ty, ty+1, ..., ty+31 (mod 32) = all distinct ✓
//
// Cost: 32 extra floats per tile = 128 bytes wasted smem
__global__ void transpose_padded(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // +1 padding is the only change

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            tile[threadIdx.y + j][threadIdx.x] = in[(y + j) * N + x];

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            out[(y + j) * N + x] = tile[threadIdx.x][threadIdx.y + j];
            //                     ↑ now bank = (tx + ty+j)%32: all distinct ✓
}

// ─── Version C: XOR swizzle (zero extra memory) ────────────────────────────
//
// Write: tile[ty+j][tx ^ (ty+j)] = in[...]
//   Bank at write: (tx^(ty+j)) % 32 = tx^(ty+j) — for warp (ty fixed, tx=0..31)
//   tx^c where c is constant: bijection on {0..31} → all distinct ✓ (no new conflicts)
//
// Read:  tile[tx][(ty+j) ^ tx]
//   Bank at read: ((ty+j)^tx) % 32 = (ty+j)^tx — for warp (ty fixed, tx=0..31)
//   (ty+j)^0, (ty+j)^1, ..., (ty+j)^31: bijection on {0..31} → all distinct ✓
//
// Correctness: write by thread(ty+j=r, tx=c) goes to tile[r][c^r].
//              tile[r][c] = in[r][c^r]  (the value for logical position (r, c^r) of in)
//              read tile[tx][(ty+j)^tx] retrieves tile[a][b] where a=tx, b=(ty+j)^tx
//              = in[a][b^a] = in[tx][(ty+j)^tx^tx] = in[tx][ty+j] ✓ (correct transpose)
__global__ void transpose_swizzled(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];   // SAME size as naive — no waste

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Write with XOR swizzle: scatter the column index to avoid read-phase conflicts
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            tile[threadIdx.y + j][threadIdx.x ^ (threadIdx.y + j)] = in[(y + j) * N + x];
            //                                  ↑ XOR swizzle on column index

    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Read with matching unswizzle: bank = (ty+j)^tx — all distinct per warp ✓
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            out[(y + j) * N + x] = tile[threadIdx.x][(threadIdx.y + j) ^ threadIdx.x];
            //                                        ↑ same XOR on the column index
}

torch::Tensor run_naive(torch::Tensor in) {
    int N = in.size(0);
    auto out = torch::empty_like(in);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    transpose_naive<<<grid, block>>>(out.data_ptr<float>(), in.data_ptr<float>(), N);
    return out;
}

torch::Tensor run_padded(torch::Tensor in) {
    int N = in.size(0);
    auto out = torch::empty_like(in);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    transpose_padded<<<grid, block>>>(out.data_ptr<float>(), in.data_ptr<float>(), N);
    return out;
}

torch::Tensor run_swizzled(torch::Tensor in) {
    int N = in.size(0);
    auto out = torch::empty_like(in);
    dim3 grid(N / TILE_DIM, N / TILE_DIM);
    dim3 block(TILE_DIM, BLOCK_ROWS);
    transpose_swizzled<<<grid, block>>>(out.data_ptr<float>(), in.data_ptr<float>(), N);
    return out;
}
"""

cpp_src = """
torch::Tensor run_naive(torch::Tensor in);
torch::Tensor run_padded(torch::Tensor in);
torch::Tensor run_swizzled(torch::Tensor in);
"""

mod = load_inline(
    name="ex4_4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_naive", "run_padded", "run_swizzled"],
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


PEAK_BW = 288.0

# ── Correctness ────────────────────────────────────────────────────────────────
print("Correctness tests (output == in.T):")
for N in [32, 64, 128, 256, 512, 1024, 2048]:
    x   = torch.randn(N, N, device="cuda", dtype=torch.float32)
    ref = x.T.contiguous()
    out_a = mod.run_naive(x)
    out_b = mod.run_padded(x)
    out_c = mod.run_swizzled(x)
    torch.testing.assert_close(out_a, ref, rtol=0, atol=0)
    torch.testing.assert_close(out_b, ref, rtol=0, atol=0)
    torch.testing.assert_close(out_c, ref, rtol=0, atol=0)
    print(f"  N={N:5d}: naive PASSED  padded PASSED  swizzled PASSED")

print()
print("All three variants produce exactly the same output (transpose is exact).")
print("Bank conflicts are a performance issue, NOT a correctness issue.")
print()

# ── Bank analysis ──────────────────────────────────────────────────────────────
print("Bank access analysis (TILE_DIM=32, float32):")
print()
print("Write phase — thread (ty, tx) writes tile[ty][tx]:")
print("  address = (ty * 32 + tx) * 4 bytes")
print("  bank = tx % 32")
print("  Warp (ty fixed, tx=0..31): banks = 0, 1, ..., 31 → ZERO conflict ✓")
print()
print("Naive read — thread (ty, tx) reads tile[tx][ty]:")
print("  address = (tx * 32 + ty) * 4 bytes")
print("  bank = (tx * 32 + ty) % 32 = ty  (tx*32 always ≡ 0 mod 32)")
print("  Warp (ty fixed, tx=0..31): bank = ty for ALL threads → 32-WAY conflict ✗")
print("  Serialization: 32 transactions instead of 1 → 32× slower")
print()
print("Padded read — thread (ty, tx) reads tile[tx][ty] from tile[32][33]:")
print("  address = (tx * 33 + ty) * 4 bytes")
print("  bank = (tx * 33 + ty) % 32 = (tx + ty) % 32  (33 ≡ 1 mod 32)")
print("  Warp (ty fixed, tx=0..31): banks = ty, ty+1, ..., ty+31 (mod 32) → all distinct ✓")
print()
print("XOR swizzle write — tile[ty][tx ^ ty]:")
print("  address = (ty * 32 + tx^ty) * 4 bytes")
print("  bank = (tx ^ ty) % 32")
print("  Warp (ty fixed, tx=0..31): tx^ty = ty^0, ty^1, ..., ty^31 = all distinct ✓")
print()
print("XOR swizzle read — tile[tx][(ty) ^ tx]:")
print("  address = (tx * 32 + ty^tx) * 4 bytes")
print("  bank = (ty ^ tx) % 32")
print("  Warp (ty fixed, tx=0..31): ty^0, ty^1, ..., ty^31 = all distinct ✓")
print()

# ── Benchmark ──────────────────────────────────────────────────────────────────
print("Bandwidth benchmark (N×N matrix transpose):")
print(f"  {'N':>6}  {'naive ms':>9}  {'pad ms':>9}  {'swiz ms':>9}  "
      f"{'speedup':>8}  {'GB/s (swiz)':>12}  {'util':>7}")
print(f"  {'-'*70}")

for N in [512, 1024, 2048, 4096]:
    x = torch.randn(N, N, device="cuda", dtype=torch.float32)
    bytes_rw = 2 * N * N * 4  # read N² floats + write N² floats

    ms_a = bench(lambda: mod.run_naive(x))
    ms_b = bench(lambda: mod.run_padded(x))
    ms_c = bench(lambda: mod.run_swizzled(x))

    bw_c = bytes_rw / ms_c / 1e6
    util = bw_c / PEAK_BW * 100
    speedup = ms_a / ms_c
    print(f"  {N:>6}  {ms_a:>9.3f}  {ms_b:>9.3f}  {ms_c:>9.3f}  "
          f"{speedup:>7.1f}×  {bw_c:>11.1f}  {util:>6.1f}%")

print()
print(f"  Peak: {PEAK_BW} GB/s | Target: >70% (>201 GB/s)")
print()

# ── smem size comparison ──────────────────────────────────────────────────────
print("─" * 60)
print("Shared memory usage comparison:")
print(f"  naive:    float[32][32]   = {32*32*4} bytes  = {32*32*4/1024:.2f} KB")
print(f"  padded:   float[32][33]   = {32*33*4} bytes  = {32*33*4/1024:.2f} KB  (+{32*4}B)")
print(f"  swizzled: float[32][32]   = {32*32*4} bytes  = {32*32*4/1024:.2f} KB  (same as naive)")
print()
print("XOR swizzle = same performance as padding, zero extra memory.")
print()

# ── Connection to Flash Attention ─────────────────────────────────────────────
print("─" * 60)
print("How this connects to Flash Attention (Lesson 5+):")
print()
print("  In ex4_3 WMMA v2, we load smem tiles As[64][16] and Bs[16][64].")
print("  The WMMA fragment load 'wmma::load_matrix_sync' accesses these tiles")
print("  with a specific hardware-defined access pattern.")
print()
print("  For fp16 tiles, the WMMA access can cause 2-way bank conflicts because:")
print("    - fp16 element = 2 bytes")
print("    - bank = (byte_addr / 4) % 32  → two adjacent fp16 share a bank")
print("    - row stride of As[64][16] = 16 halves = 32 bytes = 8 banks")
print("    - warp accesses across multiple rows → bank collision possible")
print()
print("  permuted_smem.cuh applies the SAME XOR principle to the uint4/fp16 smem layout:")
print("    smem[row][col ^ (row % swizzle_factor)]")
print("  This eliminates the WMMA-specific bank conflicts in Flash Attention's")
print("  Q, K, V tile loads from smem.")
print()
print("  After Lesson 5, you can read permuted_smem.cuh and understand exactly")
print("  why every access is conflict-free.")
print()
print("ncu command to verify bank conflicts:")
print("  ncu --kernel-name 'transpose_naive' \\")
print("      --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\")
print("      python ex4_4_smem_swizzle.py")
print()
print("Expected output:")
print("  transpose_naive:    bank_conflicts_ld = 31 per warp (32-way)")
print("  transpose_padded:   bank_conflicts_ld = 0")
print("  transpose_swizzled: bank_conflicts_ld = 0")
