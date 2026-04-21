# Bank Conflict Patterns

Catalog of common shared memory access patterns: which ones cause conflicts and why.

---

## The Bank Conflict Table

| Access Pattern | Bank | Conflict? | Cost |
|---|---|---|---|
| `smem[tx]` | `tx % 32` | None (32 distinct banks) | 1 cycle |
| `smem[tx + offset]` | `(tx+offset) % 32` | None if all offsets differ | 1 cycle |
| `smem[tx * 2]` | `(tx*2) % 32` | 2-way (tx=0,16 → bank 0) | 2 cycles |
| `smem[tx * 16]` | `(tx*16) % 32` | 2-way | 2 cycles |
| `smem[tx * 32]` | `0` always | 32-way | 32 cycles |
| `smem[0]` (all threads) | `0` (broadcast) | None (broadcast) | 1 cycle |
| `smem[warp_id]` (1 per warp) | `warp_id` | None | 1 cycle |

---

## Pattern 1: Sequential Access (No Conflict)

```cuda
// Thread tx reads smem[tx]
// Banks: 0,1,2,...,31 → all distinct → zero conflicts
float v = smem[threadIdx.x];
```

**Why:** Each of the 32 threads in a warp accesses a different bank.

**Appears in:** Cross-warp reduce read — `smem[lane]` where `lane = threadIdx.x`

---

## Pattern 2: Stride-2 Access (2-Way Conflict)

```cuda
// Thread tx reads smem[tx * 2]
// tx=0  → smem[0]  → bank 0
// tx=1  → smem[2]  → bank 2
// tx=16 → smem[32] → bank 0  ← conflict with tx=0
// tx=17 → smem[34] → bank 2  ← conflict with tx=1
// Result: 2-way conflict — 2 cycles per warp access
float v = smem[threadIdx.x * 2];
```

**Appears in:** Matrix transpose with naive layout (reads columns instead of rows).

**Fix:** Use padding — `smem[ROW][COL + 1]` shifts column addresses so each row maps to different banks.

---

## Pattern 3: Stride-32 Access (32-Way Conflict — Worst Case)

```cuda
// Thread tx reads smem[tx * 32]
// All 32 threads → bank 0 → 32-way conflict → 32× serialization
float v = smem[threadIdx.x * 32];
```

**When it appears:** When you have a 2D smem array of shape `[32][something]` and access by column:
```cuda
__shared__ float smem[num_warps][32];   // row-major
// Column access (stride = 32 = width):
float v = smem[tx][ty];   // ty constant, tx varies → stride-32 → conflict
```

**Fix:** Transpose the access: access `smem[ty][tx]` (row access, unit stride → no conflict).

---

## Pattern 4: The Block Reduce Pattern (No Conflict)

```cuda
// Write: each warp writes one value to smem[warp_id]
// With 16 warps: writes to smem[0] through smem[15]
// Banks: 0,1,...,15 → all distinct → zero conflicts
if (threadIdx.x == 0)
    smem[threadIdx.y] = warp_partial;

// Read: warp 0 reads smem[0..15]
// thread 0 reads smem[0]  → bank 0
// thread 1 reads smem[1]  → bank 1
// ...
// thread 15 reads smem[15] → bank 15
// threads 16-31: load 0.f (no memory access)
// Result: 16 distinct banks → zero conflicts
float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.f;
```

**This is the exact pattern in `block_reduce_sum_2d`.**

---

## Pattern 5: The Broadcast Pattern (Hardware-Optimized)

```cuda
// All threads in a warp read the SAME address → broadcast
// Hardware detects this and issues ONE read, broadcasts result
// Zero conflicts regardless of bank
float result = smem[0];   // all 32 threads read smem[0]
```

**Appears in:** Final read of `smem[0]` in block reduce — every thread gets the total.

**Note:** The broadcast optimization only applies when ALL threads read the EXACT same address. If even one thread reads a different address, it is a conflict.

---

## Pattern 6: GEMM Tile Access (Requires Swizzle)

For tiled GEMM (Lesson 5), each thread accesses a column of the A tile stored in shared memory:

```cuda
// Without swizzle: thread tx loads column tx of A tile
// A tile shape: [BLOCK_K][BLOCK_M]
// Column load: smem[k][threadIdx.x] for k=0,1,...
// All k values → same bank (tx % 32 = tx, but stride is BLOCK_M words)
// If BLOCK_M = 128 (fp16): 128 halves = 64 banks → 2-way conflict

// With swizzle: XOR column with row to stagger bank access
int swizzled_col = threadIdx.x ^ (k % 8);
float v = smem[k][swizzled_col];   // different bank for each k → zero conflict
```

**Reference:** `REPOS/flashinfer/include/flashinfer/permuted_smem.cuh` — the full swizzle implementation for MMA fragment layouts.

---

## Detecting Bank Conflicts with ncu

```bash
# Count bank conflicts in your kernel:
ncu --kernel-name my_kernel \
    --metrics \
      l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
      l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \
    python my_script.py
```

**Interpreting the output:**
- `bank_conflicts_ld = 0` → no conflicts on reads (correct pattern)
- `bank_conflicts_st = 0` → no conflicts on writes
- `bank_conflicts_ld = 31` → 32-way conflict (every warp access serialized 32 times)
- `bank_conflicts_ld = 1` → 2-way conflict (serialized 2 times)

**General formula:** `N-way conflict → bank_conflicts count = N - 1` per warp per instruction.

---

## Bank Conflict Avoidance Checklist

Before launching a kernel with shared memory, check these:

1. **Warp-level writes to smem:** Do all 32 threads write to different banks?
   - `smem[threadIdx.x]` → OK (banks 0–31)
   - `smem[threadIdx.x * stride]` → only OK if stride is odd or 0

2. **Warp-level reads from smem:** Same question.

3. **Cross-warp patterns (like block reduce):** Does each warp write to a unique slot?
   - `smem[warp_id]` → OK (warp_id maps to distinct banks as long as n_warps ≤ 32)

4. **2D array accesses:** Are you accessing by row (unit stride = no conflict) or by column (stride = row_width = possible conflict)?

5. **Final broadcast:** After the reduce, reading `smem[0]` from all threads → hardware broadcast → zero conflict. ✓

---

## The Rule of Thumb

```
If two threads in the same warp access the same 4-byte bank
with different addresses → conflict.

smem[tx]         → banks 0–31, all different → SAFE
smem[tx * k]     → conflict if gcd(k, 32) > 1
                   k=1 (safe), k=2 (2-way), k=4 (4-way), k=32 (32-way)
smem[warp_id]    → banks 0–(N-1) for N warps, all different → SAFE (N ≤ 32)
same address     → hardware broadcast → SAFE
```
