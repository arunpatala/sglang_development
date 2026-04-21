# Shared Memory Basics

Reference companion for Lesson 3. Keep this open alongside `norm.cuh` while reading.

---

## Hardware Layout

Shared memory (`__shared__`) is a fast SRAM scratchpad on each SM. Unlike global memory (GDDR6X, ~300 cycle latency), shared memory is ~32 cycles from the compute units.

```
SM (Streaming Multiprocessor) — RTX 4060 Ti, sm_89
├── Register File        256 KB / SM   (~1 cycle)
├── Shared Memory / L1   100 KB / SM   (~32 cycles)  ← this lesson
└── L2 Cache             32 MB total   (~80–120 cycles)
     └── GDDR6X          16 GB total   (~300 cycles, 288 GB/s)
```

The 100 KB is shared between L1 cache and explicit shared memory. The split is configurable via `cudaFuncSetAttribute`:
- Default on sm_89: 32 KB L1 / 64 KB smem (or 64/32, or 16/84, etc.)
- Max explicit smem per block: 99 KB (1 KB reserved for overhead)
- Most kernels use far less — RMSNorm uses 4–64 bytes for the reduce buffer

---

## Bank Structure

Shared memory is organized as 32 banks, each 4 bytes wide.

```
Bank:    0    1    2    3    4    5   ...  31    0    1   ...
Bytes:  0-3  4-7  8-11 12-15 16-19  20-23     124-127 128-131 ...
```

**Bank formula:** `bank_number = (byte_address / 4) % 32`

**For different types:**

| Type | Bytes | Elements per bank slot | Elements between same-bank accesses |
|---|---|---|---|
| `float` (32b) | 4 | 1 | 32 floats apart |
| `__half` (16b) | 2 | 2 | 16 halves apart |
| `double` (64b) | 8 | 0.5 | 16 doubles apart |
| `uint4` (128b) | 16 | 0.25 | 8 uint4s apart |

**For float arrays:**
- `smem[0]` → bank 0
- `smem[1]` → bank 1
- `smem[31]` → bank 31
- `smem[32]` → bank 0 (wraps around)

---

## Bank Conflict Rules

**Zero conflicts (1-cycle access):**
Two accesses in the same warp hit different banks, OR all threads access the same address (broadcast — hardware optimizes this).

**N-way conflict (N-cycle serialization):**
N threads in the same warp access N different addresses that all map to the same bank.

```
// These 32 threads hit 32 different banks → ZERO conflict (1 cycle):
smem[threadIdx.x]       // banks 0, 1, 2, ..., 31

// These 32 threads all hit bank 0 → 32-WAY conflict (32 cycles):
smem[threadIdx.x * 32]  // bank (tx*32*4/4)%32 = (tx*32)%32 = 0 for all tx

// These 32 threads: 16 pairs hit shared banks → 2-WAY conflict (2 cycles):
smem[threadIdx.x * 16]  // bank (tx*16)%32: tx=0,16→bank0; tx=1,17→bank16; etc.
```

**Broadcast exception:** If ALL 32 threads in a warp read the SAME address, the hardware issues a single read and broadcasts it to all threads. Zero conflicts. This is how `smem[0]` is efficiently broadcast at the end of `block_reduce_sum_2d`.

---

## Synchronization

### `__syncthreads()`

Block-wide barrier. Every thread must reach this point before any thread continues.

```cuda
// Pattern: write smem, sync, read smem
smem[warp_id] = warp_partial;  // write
__syncthreads();                // barrier: writes visible to all threads
float v = smem[lane];          // safe: all writes complete
```

**Cost:** When all threads are already at the barrier, ~few cycles. When some threads are stalled waiting for memory (long-latency loads), the barrier stalls the entire block until they complete — potentially hundreds of cycles.

**When REQUIRED:**
- After writing smem values that other warps need to read
- After reading smem values before another write pass overwrites them

**When NOT needed:**
- Between warp shuffles (intra-warp operations are self-synchronized)
- After reading from smem when only the writer's warp will read those values
- Between global memory accesses that don't go through smem

### `__syncwarp(mask)`

Warp-level barrier (cheaper). Synchronizes only 32 threads within one warp.

```cuda
// Only needed when threads within one warp diverged and you need to
// ensure they re-converge before the next operation.
// On Volta+ (all modern GPUs): required after warp-level operations
// if there was prior thread divergence within the warp.
__syncwarp(0xffffffff);  // all 32 lanes
```

In practice, `__syncwarp` is rarely needed in modern CUDA code unless you're using warp-divergent control flow before a shuffle. The `__shfl_xor_sync` already takes a mask parameter.

---

## Declaring Shared Memory

### Static allocation (size known at compile time)

```cuda
__global__ void my_kernel() {
    __shared__ float smem[32];          // 32 floats = 128 bytes
    __shared__ float smem_2d[8][32];    // 2D array: 8 rows × 32 cols
    __shared__ uint4 tile[64];          // 64 uint4 = 1024 bytes
}
```

### Dynamic allocation (size specified at launch)

```cuda
__global__ void my_kernel() {
    extern __shared__ float smem[];     // size set at launch
}

// Launch:
int smem_bytes = n_warps * sizeof(float);
my_kernel<<<grid, block, smem_bytes>>>(args...);
```

**Using dynamic smem for multiple purposes (pointer arithmetic):**

```cuda
extern __shared__ float smem[];
float* reduce_buf = smem;              // n_warps floats
uint4* tile       = (uint4*)(smem + n_warps);  // rest for tile cache
```

⚠️ **Alignment:** `uint4` requires 16-byte alignment. Ensure the first section is padded to a 16-byte boundary:
```cuda
int reduce_bytes = ((n_warps * sizeof(float) + 15) / 16) * 16;
float* reduce_buf = smem;
uint4* tile       = (uint4*)((char*)smem + reduce_bytes);
int total_smem    = reduce_bytes + n_tiles * sizeof(uint4);
```

---

## Smem as Tile Cache (Preview of Flash Attention)

In Flash Attention (Lessons 5–6), large KV tiles are loaded from GDDR6X into smem and reused multiple times. The same bank-conflict rules apply:

```
// Q tile in smem: [BLOCK_M, head_dim] = 64 × 128 floats = 32 KB
// K tile in smem: [BLOCK_N, head_dim] = 64 × 128 floats = 32 KB
// Total: 64 KB out of 99 KB available on sm_89
```

The smem layout must be swizzled (column XOR row) to avoid bank conflicts when reading in the MMA (tensor core) access pattern — this is what `permuted_smem.cuh` does. You'll implement this in Lesson 5.

---

## Quick Reference

| Question | Answer |
|---|---|
| How many banks? | 32 |
| Bank width | 4 bytes |
| Bank formula | `(byte_addr / 4) % 32` |
| Latency | ~32 cycles (vs 300 for GDDR6X) |
| Max smem per block (sm_89) | 99 KB |
| `smem[i]` bank for float | `i % 32` |
| `smem[i*32]` bank for float | `0` always → 32-way conflict |
| Broadcast: all threads read same address | 0 conflicts (hardware optimizes) |
| `__syncthreads()` | Block-wide barrier, all threads must reach it |
| `__syncwarp()` | Warp-wide barrier, only 32 threads |
| Static smem | `__shared__ T name[N];` |
| Dynamic smem | `extern __shared__ T name[];` + 3rd launch arg |
