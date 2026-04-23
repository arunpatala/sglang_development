# Shared Memory Swizzling for GEMM

Reference companion for Lesson 4, exercise 4.4. Read alongside `permuted_smem.cuh`.

---

## Where Bank Conflicts Appear in GEMM

Standard tiled GEMM with a 2D block (threadIdx.y = row, threadIdx.x = col) has zero bank conflicts because all reads are either sequential or broadcast. But two scenarios break this:

### Scenario 1: Matrix Transpose (Classic Teaching Example)

When you transpose a 32×32 tile stored in smem:
- **Write (no conflict):** thread `(ty, tx)` writes `smem[ty][tx]`. Within a warp (ty fixed, tx=0..31), all 32 accesses go to consecutive banks. ✓
- **Read (32-way conflict):** thread `(ty, tx)` reads `smem[tx][ty]` (row and col swapped). Within a warp (ty fixed, tx=0..31), accesses go to `smem[0..31][ty]`. Each access `smem[tx][ty]` hits bank `(tx*32 + ty) % 32 = ty % 32` — **same bank for all 32 threads**. → 32-way conflict! 32× serialization.

### Scenario 2: WMMA Fragment Load from smem

When loading a WMMA fragment (`wmma::load_matrix_sync`) from shared memory:
- The tensor core hardware accesses the 16×16 fp16 tile via specific thread-to-element assignments
- If the tile is stored in a naive column-major layout, the warp's 32 threads can hit overlapping banks
- The `permuted_smem.cuh` swizzle is designed to eliminate these conflicts for the exact WMMA access pattern

This lesson focuses on **Scenario 1** (matrix transpose) as a clean, verifiable demonstration of the swizzle principle.

---

## Bank Conflict Analysis for Matrix Transpose

Tile: `float smem[TILE_DIM][TILE_DIM]` with `TILE_DIM = 32`.

**Write phase** — thread `(ty, tx)` writes `smem[ty][tx]`:
```
Byte address = (ty * 32 + tx) * 4
Bank = ((ty * 32 + tx) * 4 / 4) % 32 = (ty * 32 + tx) % 32 = tx % 32

For a warp (ty fixed, tx = 0..31): banks = 0, 1, 2, ..., 31  → all distinct → ZERO conflict ✓
```

**Read phase** — thread `(ty, tx)` reads `smem[tx][ty]` (row and col SWAPPED for transpose):
```
Byte address = (tx * 32 + ty) * 4
Bank = (tx * 32 + ty) % 32 = ty % 32  (because tx*32 mod 32 = 0 for any tx)

For a warp (ty fixed, tx = 0..31): bank = ty for ALL threads → 32-WAY CONFLICT ✗
```

The read phase is the problem: **all 32 threads in a warp hit the same bank** because the column stride (32 floats × 4 bytes = 128 bytes) is exactly 32 banks × 4 bytes, so each row wraps back to bank 0.

---

## Fix 1: Padding

Add one extra column to each row: `float smem[TILE_DIM][TILE_DIM + 1]`.

**Read phase** — thread `(ty, tx)` reads `smem[tx][ty]` from padded array:
```
Byte address = (tx * 33 + ty) * 4
Bank = (tx * 33 + ty) % 32

= (tx * (32 + 1) + ty) % 32
= (tx + ty) % 32

For a warp (ty fixed, tx = 0..31): banks = ty%32, (ty+1)%32, ..., (ty+31)%32
= all 32 distinct values → ZERO conflict ✓
```

**Cost:** 1 extra float per row = TILE_DIM extra floats = 32 × 4 = 128 bytes wasted smem per tile.
- For a 64×64 tile: 64 × 4 = 256 extra bytes wasted (negligible)
- For a 128×128 tile: 128 × 4 = 512 extra bytes wasted

**Padding is the standard fix for matrix transpose.** It is simple, correct, and the overhead is tiny.

---

## Fix 2: XOR Swizzle

Rearrange elements in smem using XOR without using extra memory:

```
Write with swizzle: thread (ty, tx) writes to smem[ty][tx ^ ty] instead of smem[ty][tx]
Read  with swizzle: thread (ty, tx) reads from smem[tx][ty ^ tx] instead of smem[tx][ty]
```

**Why this works for the write (no new conflict introduced):**
```
For write, thread (ty, tx), warp has ty fixed, tx = 0..31:
  smem address = (ty * 32 + (tx ^ ty)) * 4
  bank = (tx ^ ty) % 32

For tx = 0..31 with fixed ty: tx ^ ty = 0^ty, 1^ty, ..., 31^ty
XOR with a constant is a bijection on {0..31} → all 32 distinct values → ZERO conflict ✓
```

**Why this works for the read:**
```
For read, thread (ty, tx), warp has ty fixed, tx = 0..31:
  smem address = (tx * 32 + (ty ^ tx)) * 4
  bank = (ty ^ tx) % 32

For tx = 0..31 with fixed ty: ty ^ 0, ty ^ 1, ..., ty ^ 31
= all 32 distinct values → ZERO conflict ✓
```

**Correctness verification:**
The value written at `smem[ty][tx ^ ty]` was originally `in[ty][tx]`.
When reading `smem[tx][ty ^ tx]`, we retrieve the element written by the thread where `write_ty = tx` and `write_tx = ty` (because `write_ty = tx` and `write_tx ^ write_ty = ty ^ tx`, so `write_tx = ty`).
That was `in[write_ty][write_tx] = in[tx][ty]`, which is the correct transposed value. ✓

**Cost:** Zero extra memory. Same performance as padding. The XOR is folded into the address calculation by the compiler at zero extra cost.

---

## Code: Three Versions

```cuda
const int TILE_DIM  = 32;
const int BLOCK_ROWS = 8;   // threads per tile row (blockDim.y)

// ─── Version A: Naive (32-way conflict on reads) ───────────────────────────
__global__ void transpose_naive(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];

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
            //                            ↑ column read → 32-way conflict!
}

// ─── Version B: Padded (+1 column per row) ────────────────────────────────
__global__ void transpose_padded(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM + 1];  // ← +1 shifts column banks

    // (same write and read code, just tile has one extra column)
    // The +1 changes the effective stride to 33 floats:
    //   bank = (tx * 33 + ty) % 32 = (tx + ty) % 32 → all distinct ✓
}

// ─── Version C: XOR swizzle (no extra memory) ─────────────────────────────
__global__ void transpose_swizzled(float* out, const float* in, int N) {
    __shared__ float tile[TILE_DIM][TILE_DIM];  // same size as naive

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;

    // Write with swizzle: tile[ty][tx ^ ty] instead of tile[ty][tx]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            tile[threadIdx.y + j][threadIdx.x ^ (threadIdx.y + j)] = in[(y + j) * N + x];
    __syncthreads();

    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Read with swizzle: tile[tx][ty ^ tx] instead of tile[tx][ty]
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS)
        if ((y + j) < N && x < N)
            out[(y + j) * N + x] = tile[threadIdx.x][(threadIdx.y + j) ^ threadIdx.x];
            //                            ↑ (ty+j) ^ tx → all distinct banks ✓
}
```

---

## Expected Performance

| Version | Bank Conflicts (ld) | Time | Notes |
|---|---|---|---|
| Naive | 31 per warp access | Slowest | 32-way serialization on read |
| Padded | 0 | Same as swizzled | +128 bytes smem wasted |
| XOR swizzle | 0 | Same as padded | Zero extra memory |

Typical speedup from fixing bank conflicts: **2–10×** depending on tile size and occupancy.

On RTX 4060 Ti, a well-written transpose should hit >80% of the theoretical 288 GB/s bandwidth limit (2 * N² * sizeof(float) bytes accessed for an N×N transpose).

---

## The Swizzle in permuted_smem.cuh

FlashInfer's `permuted_smem.cuh` applies the same XOR principle to WMMA fragment loads:

```cpp
// From permuted_smem.cuh (simplified):
template <typename T, int vec_size>
__device__ T* smem_get_permuted_ptr(T* smem, int row, int col, int stride) {
    // XOR the column with the row index (divided by vec_size due to vectorized loads)
    int permuted_col = col ^ (row / num_bits_per_swizzle_pattern);
    return smem + row * stride + permuted_col * vec_size;
}
```

The WMMA fragment load accesses 16×16 elements of a tile. With a naive layout `As[16][16]`:
- `wmma::load_matrix_sync` assigns specific elements to each of the 32 threads
- The hardware access pattern can cause 2-way or 4-way conflicts for fp16 tiles
- The swizzle in `permuted_smem.cuh` ensures that the WMMA-specific access pattern hits distinct banks

**When to apply swizzle for WMMA:**
- BK=16, fp16: the 16-wide tile has 16×2=32 bytes per row = 8 banks. The WMMA pattern can still cause 2-way conflicts.
- BK=32, fp16: 32×2=64 bytes per row = 16 banks. More potential conflicts.
- For BK=16 with carefully chosen tile dimensions, some layouts are conflict-free without swizzle.

**In this lesson:** we demonstrate the swizzle on matrix transpose (clean example). The full WMMA swizzle for production Flash Attention is in Lesson 5 with `permuted_smem.cuh`.

---

## Detecting Bank Conflicts with ncu

```bash
# Count bank conflicts for a specific kernel:
ncu --kernel-name "transpose_naive" \
    --metrics \
      l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
      l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
      l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum \
    python ex4_4_smem_swizzle.py

# For the three variants in one run:
ncu --kernel-name "(transpose_naive|transpose_padded|transpose_swizzled)" \
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \
    python ex4_4_smem_swizzle.py
```

**Interpreting ncu output:**
- `bank_conflicts_ld = 0` → no conflicts on reads
- `bank_conflicts_ld = 31` per warp per access → 32-way conflict (N-way conflict shows N-1)
- `sectors_shared_ld` → number of 32-byte transactions; with N-way conflict, N× more transactions than ideal

---

## Swizzle Generalization

The XOR pattern generalizes to any power-of-2 tile size:

| TILE_DIM | Stride (bytes) | Conflict without swizzle | XOR bits |
|---|---|---|---|
| 16 floats | 64 bytes | 2-way (stride = 16 banks, 32 threads → 2 share each) | col ^= row |
| 32 floats | 128 bytes | 32-way | col ^= row |
| 64 floats | 256 bytes | 32-way | col ^= (row % 4) or XOR higher bits |

For `float4` (128-bit) smem arrays — e.g., fp16 stored as `uint4` (8 halves):
- Each row is `TILE_DIM / 8` uint4 elements
- bank = (row * TILE_DIM/8 + col) % 32 where col is 0..TILE_DIM/8-1
- The XOR swizzle is applied to the uint4 column index: `col ^= row`

**Rule of thumb:** any time the smem stride (bytes per row) is a multiple of 128 bytes (32 banks × 4 bytes), column reads will conflict. Apply XOR swizzle.

---

## Bank Conflict Avoidance Checklist for GEMM

Before adding smem to a GEMM kernel:

1. **Are you loading tiles into smem row by row?** ✓ (consecutive banks per row = zero conflict)
2. **Are you reading from smem row by row?** ✓ (Bs[k][threadIdx.x] = sequential = zero conflict)
3. **Are you reading smem column by column?** ✗ (need padding or swizzle)
4. **Are WMMA fragment loads from smem?** → Check `permuted_smem.cuh` for the production pattern
5. **After fixing, verify with ncu** `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum = 0`

---

## Quick Reference

| Pattern | Bank Conflict | Fix |
|---|---|---|
| `smem[ty][tx]` (row read, unit stride) | None | — |
| `smem[tx][ty]` (column read, stride-32) | 32-way | Padding or XOR swizzle |
| `smem[ty][tx ^ ty]` (XOR write) | None | (this IS the swizzle) |
| `smem[tx][(ty) ^ tx]` (XOR read) | None | (this IS the swizzle) |
| WMMA `load_matrix_sync` from smem | Possible | See `permuted_smem.cuh` |

| Swizzle Formula | bank(tx, ty fixed) | Result |
|---|---|---|
| Naive: `smem[tx][ty]` | `ty % 32` = constant | 32-way conflict |
| Padded: `smem[tx][ty]` (TILE+1) | `(tx + ty) % 32` | all distinct |
| XOR: `smem[tx][ty ^ tx]` | `(ty ^ tx) % 32` | all distinct |
