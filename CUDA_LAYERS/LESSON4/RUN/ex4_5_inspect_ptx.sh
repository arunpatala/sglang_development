#!/usr/bin/env bash
# ex4_5_inspect_ptx.sh — PTX and SASS Inspection for Lesson 4 Kernels
# =====================================================================
# Compiles the Lesson 4 GEMM and WMMA kernels to PTX, then inspects
# the output for key instruction signatures:
#
#   - ffma.rn    → fp32 scalar fused-multiply-add (CUDA cores)
#   - hmma.sync  → fp16×fp16 WMMA instruction (PTX level)
#   - HMMA.16816 → 4th-gen tensor core instruction (SASS level)
#   - ldu.global / ld.global → global memory loads
#   - ld.shared  / st.shared → smem loads/stores
#   - bar.sync   → __syncthreads()
#
# Usage:
#   bash ex4_5_inspect_ptx.sh              # all analyses
#   bash ex4_5_inspect_ptx.sh naive        # naive GEMM PTX only
#   bash ex4_5_inspect_ptx.sh wmma         # WMMA PTX + SASS
#   bash ex4_5_inspect_ptx.sh tiled        # tiled GEMM PTX
#   bash ex4_5_inspect_ptx.sh swizzle      # smem swizzle PTX
#
# Requires: nvcc (from CUDA Toolkit, same version as torch)
#           cuobjdump (for SASS, comes with CUDA Toolkit)

set -e
MODE=${1:-all}

ARCH="sm_89"   # Ada Lovelace, RTX 4060 Ti
NVCC_FLAGS="-O3 -arch=${ARCH} --use_fast_math -std=c++17"
TMP="/tmp/lesson4_ptx"
mkdir -p "$TMP"

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
    echo ""
}

count_or_zero() {
    grep -c "$1" "$2" 2>/dev/null || echo "0"
}

# ─── Naive GEMM PTX ───────────────────────────────────────────────────────────
if [ "$MODE" = "naive" ] || [ "$MODE" = "all" ]; then
    header "NAIVE GEMM (fp32) — PTX instruction analysis"

    cat > "$TMP/naive.cu" << 'EOF'
__global__ void naive_gemm_fp32(
    const float* A, const float* B, float* C, int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;
    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
EOF

    echo "Compiling naive GEMM to PTX..."
    nvcc --ptx $NVCC_FLAGS "$TMP/naive.cu" -o "$TMP/naive.ptx" 2>&1

    echo "PTX instruction counts for naive_gemm_fp32:"
    echo ""
    echo "  Global loads  (ld.global.*):    $(count_or_zero 'ld\.global' $TMP/naive.ptx)"
    echo "  Global stores (st.global.*):    $(count_or_zero 'st\.global' $TMP/naive.ptx)"
    echo "  FP32 FMA      (fma.rn.f32):     $(count_or_zero 'fma\.rn\.f32' $TMP/naive.ptx)"
    echo "  Shuffles      (shfl.sync):      $(count_or_zero 'shfl\.sync' $TMP/naive.ptx)"
    echo "  Smem loads    (ld.shared):      $(count_or_zero 'ld\.shared' $TMP/naive.ptx)"
    echo "  Smem stores   (st.shared):      $(count_or_zero 'st\.shared' $TMP/naive.ptx)"
    echo "  Sync barriers (bar.sync):       $(count_or_zero 'bar\.sync' $TMP/naive.ptx)"
    echo "  WMMA ops      (wmma.*):         $(count_or_zero 'wmma\.' $TMP/naive.ptx)"
    echo ""
    echo "Expected:"
    echo "  Global loads  ≥ 2 per loop iter (A[row][k] + B[k][col])"
    echo "  FP32 FMA      1 per loop iter (sum += A * B)"
    echo "  Smem:         0 (no smem in naive GEMM)"
    echo "  WMMA:         0 (no tensor cores in naive GEMM)"
    echo ""
    echo "Key FMA instruction from PTX:"
    grep -n "fma\.rn\.f32" "$TMP/naive.ptx" | head -3
fi

# ─── Tiled GEMM PTX ───────────────────────────────────────────────────────────
if [ "$MODE" = "tiled" ] || [ "$MODE" = "all" ]; then
    header "TILED GEMM (fp32, TILE=32) — PTX smem instruction analysis"

    cat > "$TMP/tiled.cu" << 'EOF'
__global__ void tiled_gemm_t32(
    const float* A, const float* B, float* C, int M, int N, int K)
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
    if (row < M && col < N) C[row * N + col] = sum;
}
EOF

    echo "Compiling tiled GEMM to PTX..."
    nvcc --ptx $NVCC_FLAGS "$TMP/tiled.cu" -o "$TMP/tiled.ptx" 2>&1

    echo "PTX instruction counts for tiled_gemm_t32:"
    echo ""
    echo "  Global loads  (ld.global.*):    $(count_or_zero 'ld\.global' $TMP/tiled.ptx)"
    echo "  Global stores (st.global.*):    $(count_or_zero 'st\.global' $TMP/tiled.ptx)"
    echo "  FP32 FMA      (fma.rn.f32):     $(count_or_zero 'fma\.rn\.f32' $TMP/tiled.ptx)"
    echo "  Smem loads    (ld.shared.*):    $(count_or_zero 'ld\.shared' $TMP/tiled.ptx)"
    echo "  Smem stores   (st.shared.*):    $(count_or_zero 'st\.shared' $TMP/tiled.ptx)"
    echo "  Sync barriers (bar.sync):       $(count_or_zero 'bar\.sync' $TMP/tiled.ptx)"
    echo ""
    echo "Expected:"
    echo "  Global loads: 2 per tile (one for A, one for B)"
    echo "  Smem stores:  2 per tile (write As and Bs)"
    echo "  Smem loads:   32×2 = 64 per tile (inner loop reads As[ty][k] and Bs[k][tx])"
    echo "                (compiler may cache As[ty][k] in a register → broadcast → 32 ld.shared)"
    echo "  bar.sync:     2 per tile (one after load, one after compute)"
    echo "  FMA:          32 per tile (#pragma unroll expands inner loop)"
    echo ""
    echo "Key smem instructions from PTX:"
    grep -n "ld\.shared\|st\.shared\|bar\.sync" "$TMP/tiled.ptx" | head -10
fi

# ─── WMMA GEMM PTX and SASS ───────────────────────────────────────────────────
if [ "$MODE" = "wmma" ] || [ "$MODE" = "all" ]; then
    header "WMMA GEMM (fp16, v1 and v2) — Tensor Core instruction analysis"

    cat > "$TMP/wmma.cu" << 'EOF'
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm_v1(
    const __half* A, const __half* B, float* C, int M, int N, int K)
{
    int tile_row = blockIdx.y, tile_col = blockIdx.x;
    wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);
    int row_off = tile_row * 16, col_off = tile_col * 16;
    for (int k = 0; k < K; k += 16) {
        wmma::load_matrix_sync(a_frag, A + row_off * K + k, K);
        wmma::load_matrix_sync(b_frag, B + k * N + col_off, N);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }
    wmma::store_matrix_sync(C + row_off * N + col_off, c_frag, N, wmma::mem_row_major);
}

const int BM=64, BN=64, BK=16;
__global__ void wmma_gemm_v2(
    const __half* A, const __half* B, float* C, int M, int N, int K)
{
    __shared__ __half As[BM][BK];
    __shared__ __half Bs[BK][BN];
    int warp_id = threadIdx.x/32, warp_row = warp_id/4, warp_col = warp_id%4;
    int block_row = blockIdx.y * BM, block_col = blockIdx.x * BN;
    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);
    for (int k_start = 0; k_start < K; k_start += BK) {
        for (int idx = threadIdx.x; idx < BM*BK; idx += blockDim.x) {
            int r=idx/BK, c=idx%BK;
            As[r][c] = (block_row+r<M && k_start+c<K) ? A[(block_row+r)*K+k_start+c] : __float2half(0.f);
        }
        for (int idx = threadIdx.x; idx < BK*BN; idx += blockDim.x) {
            int r=idx/BN, c=idx%BN;
            Bs[r][c] = (k_start+r<K && block_col+c<N) ? B[(k_start+r)*N+block_col+c] : __float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b_frag;
        wmma::load_matrix_sync(a_frag, &As[warp_row*16][0], BK);
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_col*16], BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
        __syncthreads();
    }
    int out_row = block_row + warp_row*16, out_col = block_col + warp_col*16;
    if (out_row < M && out_col < N)
        wmma::store_matrix_sync(C + out_row*N + out_col, c_frag, N, wmma::mem_row_major);
}
EOF

    echo "Compiling WMMA kernel to PTX..."
    nvcc --ptx $NVCC_FLAGS "$TMP/wmma.cu" -o "$TMP/wmma.ptx" 2>&1

    echo "PTX instruction counts for wmma_gemm_v1:"
    echo ""
    echo "  WMMA load A   (wmma.load.a.*):  $(count_or_zero 'wmma\.load\.a\.' $TMP/wmma.ptx)"
    echo "  WMMA load B   (wmma.load.b.*):  $(count_or_zero 'wmma\.load\.b\.' $TMP/wmma.ptx)"
    echo "  WMMA MMA      (wmma.mma.sync):  $(count_or_zero 'wmma\.mma\.sync' $TMP/wmma.ptx)"
    echo "  WMMA store    (wmma.store.d.*): $(count_or_zero 'wmma\.store\.d\.' $TMP/wmma.ptx)"
    echo "  Global loads  (ld.global.*):    $(count_or_zero 'ld\.global' $TMP/wmma.ptx)"
    echo "  fp32 FMA      (fma.rn.f32):     $(count_or_zero 'fma\.rn\.f32' $TMP/wmma.ptx)"
    echo ""
    echo "Expected for v1 (loop body unrolled or one iteration):"
    echo "  wmma.load.a:  1 (loads A fragment)"
    echo "  wmma.load.b:  1 (loads B fragment)"
    echo "  wmma.mma.sync: 1 (tensor core instruction)"
    echo "  wmma.store.d:  1 (stores C fragment)"
    echo "  fp32 FMA:     0 (tensor cores replace scalar FMA)"
    echo ""
    echo "Key WMMA instructions:"
    grep -n "wmma\." "$TMP/wmma.ptx" | head -15
    echo ""

    # SASS analysis: look for HMMA instruction
    echo "── SASS Analysis (hardware-level instructions) ─────────────────────"
    echo ""
    echo "Compiling to cubin for SASS inspection..."
    nvcc -cubin $NVCC_FLAGS "$TMP/wmma.cu" -o "$TMP/wmma.cubin" 2>&1
    echo ""
    echo "SASS instructions (cuobjdump --dump-sass):"
    cuobjdump --dump-sass "$TMP/wmma.cubin" 2>/dev/null | grep -E "HMMA|LDG|STG|BAR" | head -20 || \
        echo "  (cuobjdump not available or no matching instructions)"
    echo ""
    echo "What to look for in SASS:"
    echo "  HMMA.16816.F32  → 4th-gen tensor core MMA (sm_89 Ada specific)"
    echo "                    This is one warp instruction that computes 16×16×16 FMAs"
    echo "  LDG.E.128       → 128-bit global load (4 floats or 8 fp16 per instruction)"
    echo "  LDS.U.128       → 128-bit smem load"
    echo "  BAR.SYNC        → __syncthreads()"
    echo ""
    echo "The HMMA instruction is the hardware-level tensor core operation."
    echo "One HMMA executes in 1 warp-instruction and performs 16×16×16 = 4096 ops."
fi

# ─── Swizzle PTX ──────────────────────────────────────────────────────────────
if [ "$MODE" = "swizzle" ] || [ "$MODE" = "all" ]; then
    header "SMEM SWIZZLE — comparing naive vs XOR address patterns in PTX"

    cat > "$TMP/swizzle.cu" << 'EOF'
#define TILE_DIM 32
#define BLOCK_ROWS 8

__global__ void transpose_naive(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x*TILE_DIM + threadIdx.x, y = blockIdx.y*TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j)<N && x<N) tile[threadIdx.y+j][threadIdx.x] = in[(y+j)*N+x];
    __syncthreads();
    x = blockIdx.y*TILE_DIM + threadIdx.x; y = blockIdx.x*TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j)<N && x<N) out[(y+j)*N+x] = tile[threadIdx.x][threadIdx.y+j];
}

__global__ void transpose_swizzled(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];
    int x = blockIdx.x*TILE_DIM + threadIdx.x, y = blockIdx.y*TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j)<N && x<N) tile[threadIdx.y+j][threadIdx.x ^ (threadIdx.y+j)] = in[(y+j)*N+x];
    __syncthreads();
    x = blockIdx.y*TILE_DIM + threadIdx.x; y = blockIdx.x*TILE_DIM + threadIdx.y;
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y+j)<N && x<N) out[(y+j)*N+x] = tile[threadIdx.x][(threadIdx.y+j) ^ threadIdx.x];
}
EOF

    echo "Compiling swizzle kernels to PTX..."
    nvcc --ptx $NVCC_FLAGS "$TMP/swizzle.cu" -o "$TMP/swizzle.ptx" 2>&1

    echo "PTX instruction counts:"
    echo ""
    echo "  [transpose_naive]:"
    echo "    ld.shared:  $(count_or_zero 'ld\.shared' $TMP/swizzle.ptx)"
    echo "    st.shared:  $(count_or_zero 'st\.shared' $TMP/swizzle.ptx)"
    echo ""
    echo "  [both kernels have same PTX counts — conflicts only appear at hardware level]"
    echo "  Both compile to ld.shared and st.shared — bank conflicts are runtime behavior"
    echo ""
    echo "PTX smem address generation — naive:"
    grep -A2 "st\.shared\|ld\.shared" "$TMP/swizzle.ptx" | head -20
    echo ""
    echo "  The XOR shows up as an 'xor.b32' instruction BEFORE the ld.shared/st.shared"
    echo "  The PTX address is pre-computed with XOR, then used in the smem access"
    echo ""
    echo "XOR instructions in swizzle kernel:"
    grep -c "xor\.b32" "$TMP/swizzle.ptx" && echo "  xor.b32 instructions found (address swizzling)" || echo "  (possibly inlined by optimizer)"
fi

echo ""
echo "========================================================"
echo "  Summary of what to look for in Lesson 4 PTX/SASS:"
echo ""
echo "  1. Naive GEMM:"
echo "       ld.global: K iterations × 2 loads (A + B)"
echo "       fma.rn.f32: K iterations × 1 FMA"
echo "       No smem, no WMMA"
echo ""
echo "  2. Tiled GEMM:"
echo "       ld.global: ceil(K/TILE) tiles × 2 global loads"
echo "       st.shared: 2 per tile (write As, Bs)"
echo "       ld.shared: TILE×2 per tile (inner loop reads)"
echo "       bar.sync: 2 per tile (after load, after compute)"
echo "       fma.rn.f32: TILE per tile (unrolled inner loop)"
echo ""
echo "  3. WMMA GEMM:"
echo "       wmma.load.a + wmma.load.b: one per K tile"
echo "       wmma.mma.sync: one per K tile (= HMMA.16816.F32 in SASS)"
echo "       wmma.store.d: once at end"
echo "       NO fma.rn.f32 — tensor cores replaced ALL scalar FMAs!"
echo ""
echo "  4. Swizzle:"
echo "       xor.b32 instruction added before ld.shared in swizzled kernel"
echo "       Bank conflict count verifiable with ncu (see profile.sh)"
echo ""
echo "  Key insight: WMMA replaces the K-iteration inner loop of FMAs"
echo "  with a single hardware tensor core instruction. The same data"
echo "  movement (load, compute, store) remains — just the compute"
echo "  throughput is 8× higher."
echo "========================================================"
