# Tensor Cores and the WMMA API

Reference companion for Lesson 4, exercises 4.3 and 4.4. Keep this open alongside `mma.cuh` while reading.

---

## What Tensor Cores Do

A tensor core is a dedicated hardware unit that computes a small matrix multiply-accumulate (MMA) in a single warp instruction. On Ada Lovelace (sm_89, RTX 4060 Ti):

```
One HMMA.16816 instruction (4th-gen tensor core):
  Inputs:  A[16][16] fp16,  B[16][16] fp16
  Accumulate: C[16][16] fp32  (running accumulator)
  Output: D[16][16] = A @ B + C  (fp32)
  Cost: 1 warp-instruction (all 32 threads participate)
  Throughput: 1 instruction per 1 cycle (on one SM)
```

**Comparison to scalar fp16 FMA (CUDA core):**

| Operation | Outputs per Instruction | Per SM per Cycle |
|---|---|---|
| Scalar fp16 FMA | 1 float (1 element of C) | 64 (2 warps × 32 threads) |
| WMMA 16×16×16 | 256 floats (one 16×16 tile) | ~512 (with pipelining) |
| Effective speedup | — | ~8× over scalar fp16 |

This 8× speedup is why GEMM-heavy workloads (transformers) use tensor cores exclusively.

---

## The WMMA API

`nvcuda::wmma` provides a C++ wrapper over PTX tensor core instructions. Three steps:

### Step 1: Declare Fragments

```cuda
#include <mma.h>
using namespace nvcuda;

// A fragment: 16 rows × 16 K-columns (row-major layout)
wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;

// B fragment: 16 K-rows × 16 columns (row-major layout)
wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;

// Accumulator fragment: 16×16 output tile (fp32)
wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
```

**Fragment template parameters:**
- Kind: `matrix_a`, `matrix_b`, or `accumulator`
- M, N, K: must be 16, 16, 16 for fp16 → fp32 on sm_89
- Element type: `__half` for A/B, `float` for fp32 accumulator
- Layout: `row_major` or `col_major` (only for A and B, not accumulator)

**What is a fragment?**
It is an opaque array of registers distributed across the 32 threads in a warp. Each thread holds 8 values of the 256-element 16×16 tile. The hardware determines which 8 elements each thread holds — you do not access them individually.

```cuda
// Fragment internal layout (DO NOT depend on this):
// fragment<accumulator, ...> has .num_elements = 8  (per thread, 8 × 32 = 256 total)
// fragment<matrix_a, ...>    has .num_elements = 8  (per thread, 8 × 32 = 256 total)
// You CAN access elements directly for initialization:
for (int i = 0; i < c_frag.num_elements; ++i)
    c_frag.x[i] = 0.f;   // manual init (prefer fill_fragment)
```

### Step 2: Load from Memory

```cuda
// Load A tile from global or shared memory
// ptr: pointer to top-left of the 16×16 sub-matrix
// ldm: leading dimension (stride in elements between rows)
wmma::load_matrix_sync(a_frag, ptr, ldm);

// Examples:
// From global memory, row-major A[M][K]:
wmma::load_matrix_sync(a_frag, A + tile_row * 16 * K + k_offset, K);
//                                  ↑ top-left element       ↑ leading dim = K

// From shared memory As[64][16] (BM=64, BK=16):
wmma::load_matrix_sync(a_frag, &As[warp_row * 16][0], 16);
//                                ↑ warp's row offset   ↑ leading dim = BK = 16
```

**The `ldm` (leading dimension):**
For a row-major matrix stored as a 1D array, `ldm` is the number of elements between the start of one row and the start of the next. For global A[M][K], `ldm = K`. For shared `As[64][16]`, `ldm = 16`.

### Step 3: Multiply-Accumulate

```cuda
// D = A × B + C  (in-place: d and c are the same fragment)
wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
// c_frag is both input (accumulator) and output (result)
// After this call, c_frag holds the partial sum so far
```

Calling `mma_sync` multiple times accumulates across K tiles:
```cuda
wmma::fill_fragment(c_frag, 0.f);         // zero accumulator

for (int k = 0; k < K; k += 16) {
    wmma::load_matrix_sync(a_frag, ...);   // 16 columns of A
    wmma::load_matrix_sync(b_frag, ...);   // 16 rows of B
    wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);  // accumulate
}
// c_frag now holds the complete C tile = sum over all K tiles
```

### Step 4: Store Result

```cuda
// Store accumulator to memory
wmma::store_matrix_sync(ptr, c_frag, ldm, wmma::mem_row_major);
// ptr: pointer to top-left of output 16×16 tile
// ldm: leading dimension of the output matrix
// layout: mem_row_major or mem_col_major
```

---

## Complete Single-Warp WMMA Kernel

One warp per 16×16 output tile. Loads directly from global memory (no smem):

```cuda
#include <mma.h>
using namespace nvcuda;

__global__ void wmma_gemm_v1(
    const __half* A,   // [M, K] row-major
    const __half* B,   // [K, N] row-major
    float*        C,   // [M, N] output, fp32
    int M, int N, int K)
{
    // This block (= one warp) computes C[tile_row*16..+16][tile_col*16..+16]
    int tile_row = blockIdx.y;  // which 16-row block
    int tile_col = blockIdx.x;  // which 16-col block

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    int row_offset = tile_row * 16;  // global row start
    int col_offset = tile_col * 16;  // global col start

    for (int k = 0; k < K; k += 16) {
        const __half* a_ptr = A + row_offset * K + k;   // A[tile_row*16][k]
        const __half* b_ptr = B + k * N + col_offset;   // B[k][tile_col*16]
        wmma::load_matrix_sync(a_frag, a_ptr, K);   // ldm = K (full row stride)
        wmma::load_matrix_sync(b_frag, b_ptr, N);   // ldm = N
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);
    }

    float* c_ptr = C + row_offset * N + col_offset;
    wmma::store_matrix_sync(c_ptr, c_frag, N, wmma::mem_row_major);
}

// Launch: requires M, N, K all multiples of 16
// grid  = dim3(N/16, M/16), block = 32 (one warp)
```

**Limitation of v1:** Each warp loads its own 16-column slice of A and 16-row slice of B from HBM on every iteration. With many warps (one per output tile), adjacent tiles reload overlapping A and B regions → redundant HBM reads.

---

## Tiled WMMA with smem (v2)

Load blocks of A and B into smem cooperatively, then each warp issues WMMA from smem:

```
Block: BM=64, BN=64, BK=16  →  16 warps (4 rows × 4 cols of 16×16 tiles)
Threads per block: 16 warps × 32 = 512 threads
Grid: (N/64, M/64)

smem:
  As[64][16] __half  =  64*16*2  = 2048 bytes = 2 KB
  Bs[16][64] __half  =  16*64*2  = 2048 bytes = 2 KB
  Total: 4 KB (well within 99 KB limit on sm_89)
```

```cuda
const int BM=64, BN=64, BK=16;

__global__ void wmma_gemm_tiled(
    const __half* A, const __half* B, float* C, int M, int N, int K)
{
    __shared__ __half As[BM][BK];   // A tile
    __shared__ __half Bs[BK][BN];   // B tile

    int warp_id  = threadIdx.x / 32;
    int warp_row = warp_id / 4;           // 0..3
    int warp_col = warp_id % 4;           // 0..3
    int block_row = blockIdx.y * BM;      // global row start
    int block_col = blockIdx.x * BN;      // global col start

    wmma::fragment<wmma::accumulator, 16,16,16, float> c_frag;
    wmma::fill_fragment(c_frag, 0.f);

    for (int k_start = 0; k_start < K; k_start += BK) {
        // ── Cooperative load A tile [BM, BK] ───────────────────────────
        for (int idx = threadIdx.x; idx < BM * BK; idx += blockDim.x) {
            int r = idx / BK, c_idx = idx % BK;
            int gr = block_row + r, gk = k_start + c_idx;
            As[r][c_idx] = (gr < M && gk < K) ? A[gr * K + gk] : __float2half(0.f);
        }
        // ── Cooperative load B tile [BK, BN] ───────────────────────────
        for (int idx = threadIdx.x; idx < BK * BN; idx += blockDim.x) {
            int r = idx / BN, c_idx = idx % BN;
            int gk = k_start + r, gc = block_col + c_idx;
            Bs[r][c_idx] = (gk < K && gc < N) ? B[gk * N + gc] : __float2half(0.f);
        }
        __syncthreads();

        // ── WMMA from smem ──────────────────────────────────────────────
        wmma::fragment<wmma::matrix_a, 16,16,16, __half, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16,16,16, __half, wmma::row_major> b_frag;
        // A sub-tile: As[warp_row*16 .. +16][0..BK]
        wmma::load_matrix_sync(a_frag, &As[warp_row * 16][0], BK);
        // B sub-tile: Bs[0..BK][warp_col*16 .. +16]
        wmma::load_matrix_sync(b_frag, &Bs[0][warp_col * 16], BN);
        wmma::mma_sync(c_frag, a_frag, b_frag, c_frag);

        __syncthreads();    // protect smem before next tile load
    }

    // ── Store output ────────────────────────────────────────────────────
    int out_row = block_row + warp_row * 16;
    int out_col = block_col + warp_col * 16;
    if (out_row < M && out_col < N)
        wmma::store_matrix_sync(C + out_row * N + out_col, c_frag, N,
                                 wmma::mem_row_major);
}
```

---

## What the Compiler Generates

After compiling with `-arch=sm_89`, inspect the PTX:

```bash
nvcc --ptx -arch=sm_89 my_wmma.cu -o my_wmma.ptx
grep -n "wmma\|hmma\|ldmatrix" my_wmma.ptx | head -20
```

**PTX (WMMA-level):**
```ptx
wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {%f0,...,%f7}, [%rd0], 4096;
wmma.load.b.sync.aligned.row.m16n16k16.global.f16 {%f8,...,%f15}, [%rd1], 4096;
wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 {...}, {...}, {...}, {...};
wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [%rd2], {...}, 4096;
```

**SASS (hardware instructions after `cuobjdump --dump-sass`):**
```sass
HMMA.16816.F32 R0, R8, R16, R0       ← 4th-gen tensor core instruction (sm_89)
```

The PTX `wmma.mma.sync` compiles to a single `HMMA` instruction per 16×16×16 tile.

---

## Fragment Layout Details

Although the fragment layout is opaque, it follows a fixed pattern on each architecture:

For `wmma::fragment<accumulator, 16, 16, 16, float>`:
- 32 threads × 8 fp32 values per thread = 256 total = full 16×16 tile
- Rows 0,1 → threads 0..7 (each thread holds one element of rows 0 and 1)
- Actual row/col assignment per thread is hardware-defined, not documented
- **You do not need to know the layout** — just use the API

For `wmma::fragment<matrix_a, 16, 16, 16, __half, row_major>`:
- 32 threads × 8 fp16 values = 256 half = full 16×16 tile
- Row-major means rows are contiguous in the source matrix → `load_matrix_sync` with ldm

---

## WMMA Constraints (sm_89)

| Parameter | Requirement |
|---|---|
| M, N, K | Must be 16, 16, 16 |
| A element type | `__half` or `__nv_bfloat16` |
| B element type | `__half` or `__nv_bfloat16` |
| C element type | `float` (fp32) or `__half` |
| A layout | `row_major` or `col_major` |
| B layout | `row_major` or `col_major` |
| Alignment | Pointer must be 16-byte aligned for `aligned` load variant |
| Thread count | All 32 threads of the warp must call the WMMA functions |
| No branching | Cannot conditionally skip wmma calls for some threads in a warp |

**On sm_89 vs other architectures:**
- sm_70 (Volta): WMMA fp16 introduced, 16×16×16
- sm_75 (Turing): adds INT8 and INT4 WMMA
- sm_80 (Ampere): TF32 WMMA, BF16 WMMA
- sm_89 (Ada): 4th-gen tensor cores, all above plus FP8 (via lower-level PTX)
- sm_90 (Hopper): adds `wgmma` (warpgroup MMA) — 64×64×16 instead of 16×16×16

---

## Reading `mma.cuh` (FlashInfer)

After this lesson, you can read `REPOS/flashinfer/include/flashinfer/mma.cuh`. Key patterns:

```cpp
// FlashInfer's MMA wrapper (lower-level than WMMA, uses inline PTX):
__device__ void mma_sync_m16n16k16_row_col_f16f16f32(
    uint32_t* C, const uint32_t* A, const uint32_t* B) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(C[0]), ...
        : "r"(A[0]), ...
    );
}
```

FlashInfer uses raw PTX instead of `nvcuda::wmma` for more control over the fragment layout. The principle is identical — it's just a lower-level interface.

**Key difference: `mma.sync` vs `wmma.mma.sync`:**
- `wmma.mma.sync m16n16k16`: WMMA API, one 16×16×16 per warp
- `mma.sync m16n8k16`: PTX MMA, one 16×8×16 per warp (FlashInfer splits the 16×16 into two 16×8)

---

## Why fp32 Accumulation Matters

For large K (e.g., K=4096 in transformer projections):
- fp16 sum of 4096 terms: relative error ≈ √4096 × fp16_epsilon ≈ 64 × 9.8e-4 ≈ 6%
- fp32 accumulator: relative error ≈ √4096 × fp32_epsilon ≈ 64 × 1.2e-7 ≈ 7.7e-6

Using fp32 accumulation (`wmma::fragment<accumulator, ..., float>`) is critical for numerical accuracy in transformer projections. The `--use_fast_math` flag does NOT affect this — the accumulation is always in fp32 when the C fragment is declared as `float`.

---

## Quick Reference

| API Call | What It Does |
|---|---|
| `wmma::fill_fragment(c, 0.f)` | Zero-initialize accumulator |
| `wmma::load_matrix_sync(frag, ptr, ldm)` | Load 16×16 tile from memory |
| `wmma::mma_sync(d, a, b, c)` | d = a×b + c (in-place when d=c) |
| `wmma::store_matrix_sync(ptr, frag, ldm, layout)` | Store 16×16 tile to memory |
| `frag.num_elements` | Number of elements per thread (8 for 16×16 fragments) |
| `frag.x[i]` | Direct element access (use sparingly) |

| Fragment Kind | M | N | K | A/B type | Acc type |
|---|---|---|---|---|---|
| `matrix_a` row_major | 16 | — | 16 | `__half` | — |
| `matrix_b` row_major | — | 16 | 16 | `__half` | — |
| `accumulator` | 16 | 16 | — | — | `float` |
