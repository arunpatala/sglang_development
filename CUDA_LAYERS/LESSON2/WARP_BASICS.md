# WARP BASICS
## What a warp is, SIMT execution, lane IDs, divergence

---

## The One-Sentence Version

A warp is the fundamental unit of GPU execution: 32 threads that always run the same instruction at the same time, with private registers but shared program counter — and whose values can be exchanged in a single instruction with no memory involved.

---

## Part 1 — SIMT: Single Instruction, Multiple Threads

The GPU executes threads in groups of 32. This group is called a **warp** (NVIDIA term) or **wavefront** (AMD term). Every thread in a warp executes the same instruction in the same clock cycle. The CPU equivalent would be AVX-512 SIMD — except a warp has 32 "lanes" instead of 16, and the lanes are full programmable threads, not fixed-width registers.

```
CPU (scalar):         GPU (warp = 32 threads in lockstep):
──────────────        ──────────────────────────────────────
clock 1: inst A       clock 1: ALL 32 threads execute inst A simultaneously
clock 2: inst B       clock 2: ALL 32 threads execute inst B simultaneously
clock 3: inst C       clock 3: ALL 32 threads execute inst C simultaneously
```

The hardware has one instruction decoder per warp. It broadcasts one decoded instruction to all 32 execution units in that warp's slice of the SM.

---

## Part 2 — Lane ID, Warp ID, and Thread Indexing

Every thread knows its position within its warp (lane ID) and which warp it belongs to (warp ID). These are computed from `threadIdx.x`:

```cuda
int lane_id  = threadIdx.x % 32;   // 0–31, position within warp
int warp_id  = threadIdx.x / 32;   // 0, 1, 2, ..., (blockDim.x/32 - 1)
int n_warps  = blockDim.x  / 32;   // number of warps in this block
```

For a block of 256 threads (8 warps):
```
Thread   0– 31: warp 0, lanes 0–31
Thread  32– 63: warp 1, lanes 0–31
Thread  64– 95: warp 2, lanes 0–31
Thread  96–127: warp 3, lanes 0–31
Thread 128–159: warp 4, lanes 0–31
Thread 160–191: warp 5, lanes 0–31
Thread 192–223: warp 6, lanes 0–31
Thread 224–255: warp 7, lanes 0–31
```

Lane IDs always restart from 0 for each warp. `threadIdx.x % 32` is the single most useful expression in any warp-level kernel.

---

## Part 3 — Registers Are Private, But Can Be Shared

Each thread has its own private registers — `float val` in thread 5 is a physically different register from `float val` in thread 6. The compiler assigns registers from the SM's register file (256 KB on RTX 4060 Ti, per SM).

This privacy is why you need special instructions to see another thread's values.

**Within a warp**, the GPU has a dedicated cross-bar interconnect that lets any lane read any other lane's register in one clock cycle. This is the warp shuffle.

**Across warps**, there is no such hardware. You must write to shared memory and synchronize.

```
Within a warp (lanes 0–31):
  Lane 5 reads lane 13's register → 1 clock (warp shuffle)

Across warps (e.g., warp 0 reads warp 3's value):
  Warp 3 writes to smem → __syncthreads() → Warp 0 reads smem
  Cost: ~30–80 cycles for smem + 1 full barrier
```

---

## Part 4 — Warp Divergence (What to Avoid)

If threads within a warp take different branches (`if`/`else`), the warp cannot execute both paths simultaneously. Instead:

1. Hardware executes the `if` path with threads not taking it **masked off** (their writes are suppressed)
2. Then executes the `else` path with the other threads masked off

Both paths are executed sequentially. Throughput halves (or worse).

```cuda
// BAD — diverges if lane_id is sometimes < 16 and sometimes >= 16
if (lane_id < 16) {
    val = val * 2.f;       // lanes 0–15 execute, 16–31 masked
} else {
    val = val + 1.f;       // lanes 16–31 execute, 0–15 masked
}
// Both paths run. Total cost = cost(path A) + cost(path B)

// GOOD — all lanes take the same branch (uniform condition)
if (blockIdx.x < some_threshold) {
    // All 32 lanes in the warp take the same path — no divergence
}
```

**Key rule:** Divergence is harmless if it's between warps (different warps take different branches — they execute independently). It is costly only within a single warp.

---

## Part 5 — Active Masks and `__syncwarp`

Modern CUDA (Volta+) requires you to specify which lanes are participating in a warp-level operation via an **active mask** (`unsigned int`, one bit per lane).

```cuda
// All 32 lanes active:
__shfl_xor_sync(0xffffffff, val, 8);

// Only lanes 0–15 active (lower half):
__shfl_xor_sync(0x0000ffff, val, 8);

// Compute the mask dynamically (for use after divergence):
unsigned int mask = __ballot_sync(0xffffffff, condition);
```

`__syncwarp(mask)` — synchronizes only the lanes listed in `mask`. Much cheaper than `__syncthreads()` (which synchronizes the entire block). Use when you need ordering within a warp after a shuffle.

In practice: **always pass `0xffffffff`** for any warp-level reduction where all threads are active. This is the safe default.

---

## Part 6 — Warp Occupancy on RTX 4060 Ti

The SM can hold multiple warps simultaneously. While one warp waits for a memory load (~300 cycles), the scheduler runs another warp. This is **latency hiding**.

```
RTX 4060 Ti (sm_89, Ada Lovelace):
  34 SMs
  Max 48 warps per SM (= 48 × 32 = 1,536 concurrent threads per SM)
  Max 32 blocks per SM
  Registers per SM: 65,536 (32-bit)

If a kernel uses 32 registers per thread:
  Each thread needs 32 registers
  Each warp (32 threads) needs 32 × 32 = 1,024 registers
  65,536 / 1,024 = 64 warps would fit — but capped at 48 by hardware
  → 48 warps × 32 threads = 1,536 concurrent threads per SM

If a kernel uses 64 registers per thread:
  Each warp needs 64 × 32 = 2,048 registers
  65,536 / 2,048 = 32 warps → only 32 warps → lower occupancy
```

**Register pressure** is why loading 8 fp16 into one `uint4` (Lesson 1) matters: one register group vs 8 separate `__half` registers frees up registers for more warps.

---

## Part 7 — The 2D Block Layout (Used in RMSNorm)

Production norm kernels use a 2D block layout instead of the usual 1D:

```cuda
dim3 block(32, num_warps);   // tx = lane, ty = warp_id
// threadIdx.x = lane (0–31)
// threadIdx.y = warp (0–num_warps-1)
```

This makes `threadIdx.x` always equal to the lane ID and `threadIdx.y` always equal to the warp ID — no modulo needed.

```cuda
// 1D block (common):
int lane    = threadIdx.x % 32;
int warp_id = threadIdx.x / 32;

// 2D block (RMSNorm pattern):
int lane    = threadIdx.x;    // cleaner
int warp_id = threadIdx.y;    // cleaner
```

Both layouts produce the same hardware behavior — threads are linearized into warps by `threadIdx.x + threadIdx.y * blockDim.x`. The 2D layout is just a readability convention.

---

## Quick Reference

```
Warp size:             32 threads (always)
Lane ID:               threadIdx.x % 32   (or threadIdx.x if 2D block)
Warp ID:               threadIdx.x / 32   (or threadIdx.y if 2D block)
Register exchange:     __shfl_xor_sync    (within warp, 1 cycle, 0 smem)
Cross-warp exchange:   shared memory + __syncthreads()
Active mask:           0xffffffff for all lanes active
Sub-warp mask:         e.g. 0x0000ffff for lanes 0–15 only
Divergence cost:       serializes divergent paths within a warp
Occupancy limiter:     registers per thread × 32 threads / 65536 regs per SM
```
