# SHUFFLE BASICS
## All four warp shuffle variants, their PTX, and when to use each

---

## The One-Sentence Version

Warp shuffles are a family of four CUDA intrinsics that move a value from one thread's register directly into another thread's register — within a warp, in one clock cycle, with zero shared memory.

---

## Part 1 — The Four Shuffle Variants

```
Function                          Partner lane (who does thread X read from?)
────────────────────────────────────────────────────────────────────────────
__shfl_sync(m, val, src)          Fixed source: always reads from lane `src`
__shfl_up_sync(m, val, delta)     Reads from lane (my_lane - delta)
__shfl_down_sync(m, val, delta)   Reads from lane (my_lane + delta)
__shfl_xor_sync(m, val, mask)     Reads from lane (my_lane XOR mask)
```

All four have the same signature pattern:
```cuda
T __shfl_*_sync(unsigned mask, T val, int arg, int width=32);
```

- `mask`: bitmask of participating lanes (use `0xffffffff` for all 32)
- `val`: this thread's value to contribute (and the return value gets overwritten)
- `arg`: source lane, delta, or XOR mask depending on variant
- `width`: sub-warp width (always 32 for full-warp operations)
- **Returns:** the value from the partner lane

---

## Part 2 — `__shfl_sync` (Broadcast from Fixed Lane)

**Use case:** Broadcast a single value from one lane to all others.

```cuda
// Every thread gets lane 0's value:
float v = __shfl_sync(0xffffffff, val, 0);

// Every thread gets the value from lane equal to (blockIdx.x % 32):
float v = __shfl_sync(0xffffffff, val, blockIdx.x % 32);
```

**Out-of-range behavior:** If `src >= width`, the hardware wraps: `src = src % width`. Reading from lane 33 reads from lane 1.

**Real use case — RoPE position broadcast:** In rotary position embedding, the position scalar (a float) is computed once by lane 0 and then broadcast to all lanes:
```cuda
float pos_val = (lane == 0) ? compute_position() : 0.f;
pos_val = __shfl_sync(0xffffffff, pos_val, 0);  // broadcast from lane 0
```

**PTX:**
```ptx
shfl.sync.idx.b32  %r_dst, %r_src, src_lane, 0x1f, 0xffffffff;
; .idx = indexed (fixed source lane)
```

---

## Part 3 — `__shfl_down_sync` and `__shfl_up_sync` (Prefix Scan)

**`__shfl_down_sync(mask, val, delta)`** — lane `i` reads from lane `i + delta`.

```
delta=1:  lane 0 reads lane 1,  lane 1 reads lane 2,  ..., lane 30 reads lane 31
          lane 31 reads "out of range" → returns its own val (clamped)
```

**`__shfl_up_sync(mask, val, delta)`** — lane `i` reads from lane `i - delta`.

```
delta=1:  lane 1 reads lane 0,  lane 2 reads lane 1,  ..., lane 31 reads lane 30
          lane 0 reads "out of range" → returns its own val (clamped)
```

**Real use case — inclusive prefix scan (cumulative sum):**
```cuda
// After this code, lane i holds sum of lanes 0..i
float scan_val = val;
for (int delta = 1; delta < 32; delta *= 2) {
    float neighbor = __shfl_up_sync(0xffffffff, scan_val, delta);
    if (lane_id >= delta) scan_val += neighbor;
}
// lane 0: v0
// lane 1: v0 + v1
// lane 2: v0 + v1 + v2
// ...
```

This appears in top-k selection (sorting partial results within a warp) but NOT in the RMSNorm/softmax reduction path. For reductions where every thread needs the total, use `__shfl_xor_sync` instead.

**PTX:**
```ptx
shfl.sync.down.b32  %r_dst, %r_src, delta, 0x1f, 0xffffffff;
shfl.sync.up.b32    %r_dst, %r_src, delta, 0x00, 0xffffffff;
; Note: up/down differ in the clamp field (0x1f vs 0x00) to prevent wraparound
```

---

## Part 4 — `__shfl_xor_sync` (Butterfly Reduce — the one you use most)

**`__shfl_xor_sync(mask, val, laneMask)`** — lane `i` reads from lane `i XOR laneMask`.

```
laneMask=16:  lane  0 ↔ lane 16,  lane  1 ↔ lane 17,  ..., lane 15 ↔ lane 31
laneMask=8:   lane  0 ↔ lane  8,  lane  1 ↔ lane  9,  ..., lane 23 ↔ lane 31
laneMask=4:   lane  0 ↔ lane  4,  lane  1 ↔ lane  5,  ..., lane 27 ↔ lane 31
laneMask=2:   lane  0 ↔ lane  2,  lane  1 ↔ lane  3,  ..., lane 29 ↔ lane 31
laneMask=1:   lane  0 ↔ lane  1,  lane  2 ↔ lane  3,  ..., lane 30 ↔ lane 31
```

**Key symmetry property:** XOR is symmetric — `i XOR m == j` iff `j XOR m == i`. So the shuffle is always a two-way exchange: when lane 5 reads from lane 13 (5 XOR 8 = 13), lane 13 simultaneously reads from lane 5.

This symmetric exchange is what makes the butterfly work: every round, each lane gets the value from its symmetric partner, adds it to its own, and now both hold a partial sum of two independent values. After 5 rounds, each lane holds the sum of all 32.

**PTX:**
```ptx
shfl.sync.bfly.b32  %r_dst, %r_src, 16, 0x1f, 0xffffffff;
; .bfly = butterfly (XOR)
; 16    = the XOR mask
; 0x1f  = 31 = warp_size - 1 (prevents out-of-warp reads)
; 0xffffffff = participation mask
```

---

## Part 5 — Shuffling 64-bit Values and Structs

`__shfl_xor_sync` only handles 32-bit values natively. For 64-bit (double, int64_t), split into two 32-bit halves:

```cuda
// Shuffle a double across lanes:
__device__ double shfl_xor_double(double val, int mask) {
    // Split the 64-bit double into two 32-bit ints
    int lo = __double2loint(val);
    int hi = __double2hiint(val);
    // Shuffle each half separately
    lo = __shfl_xor_sync(0xffffffff, lo, mask);
    hi = __shfl_xor_sync(0xffffffff, hi, mask);
    // Recombine
    return __hiloint2double(hi, lo);
}

// Usage in double-precision warp reduce:
__device__ double warp_reduce_sum_double(double val) {
    val += shfl_xor_double(val, 16);
    val += shfl_xor_double(val, 8);
    val += shfl_xor_double(val, 4);
    val += shfl_xor_double(val, 2);
    val += shfl_xor_double(val, 1);
    return val;
}
```

For shuffling a struct, shuffle each field independently:
```cuda
struct float2_val { float x, y; };
__device__ float2_val shfl_xor_float2(float2_val v, int mask) {
    v.x = __shfl_xor_sync(0xffffffff, v.x, mask);
    v.y = __shfl_xor_sync(0xffffffff, v.y, mask);
    return v;
}
```

---

## Part 6 — Sub-Warp Reductions (kNumThreads < 32)

Sometimes you want to reduce within a subset of a warp — for example, 8 threads sharing a head in multi-query attention, or 4 threads covering one fp16 vector.

```cuda
// Reduce among 8 threads (lanes 0–7, or 8–15, or 16–23, or 24–31):
// Only XOR with masks smaller than 8:
float warp_reduce_sum_8(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 4);
    val += __shfl_xor_sync(0xffffffff, val, 2);
    val += __shfl_xor_sync(0xffffffff, val, 1);
    return val;
}
// After this: lanes 0–7 hold sum of lanes 0–7
//             lanes 8–15 hold sum of lanes 8–15
//             etc. (each group of 8 is independent)
```

This is exactly what the SGLang `warp::reduce_sum<kNumThreads>` template does:
```cpp
// kNumThreads=8 → loop starts at mask=4 (kNumThreads/2), not 16
for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
    value = value + __shfl_xor_sync(active_mask, value, mask, 32);
```

**Why starting at the right offset matters:** If you start XOR at 16 for an 8-thread reduction, you'd pull values from outside your group of 8, corrupting the result.

---

## Part 7 — The `_sync` Requirement (Volta and Later)

Pre-Volta CUDA had `__shfl_xor` (no `_sync` suffix). On Volta (sm_70) and later, including Ada (sm_89), the `_sync` variants are required. The old non-sync variants are deprecated and generate warnings.

**Why:** Pre-Volta, all threads in a warp were guaranteed to be at the same PC. Volta introduced independent thread scheduling — threads within a warp can be at different instructions. The `_sync` mask tells the hardware which threads must have all reached this point before any can read results.

```cuda
// WRONG on Volta+ (deprecated, removed in CUDA 11):
val = __shfl_xor(val, 8);

// CORRECT on all modern CUDA:
val = __shfl_xor_sync(0xffffffff, val, 8);
```

On Ada (sm_89), using the non-sync variants compiles but may produce incorrect results in code with divergence. Always use `_sync`.

---

## Decision Guide: Which Shuffle to Use?

```
Goal                               Shuffle to use
──────────────────────────────────────────────────────────────────────
Warp reduce sum / max / min        __shfl_xor_sync  (butterfly XOR)
Broadcast lane 0 to all lanes      __shfl_sync with src=0
Broadcast any specific lane        __shfl_sync with src=lane_id
Prefix scan (cumsum, prefix max)   __shfl_up_sync or __shfl_down_sync
Rotate values (shift all by 1)     __shfl_down_sync with delta=1
Sub-warp reduce (N threads, N<32)  __shfl_xor_sync starting at N/2
```

For everything in this curriculum (RMSNorm, softmax, attention reduce), you will use `__shfl_xor_sync` almost exclusively.

---

## Quick Reference: PTX Instructions

```
shfl.sync.idx.b32   dst, src, lane, clamp, mask   — __shfl_sync
shfl.sync.up.b32    dst, src, delta, 0x00, mask   — __shfl_up_sync
shfl.sync.down.b32  dst, src, delta, 0x1f, mask   — __shfl_down_sync
shfl.sync.bfly.b32  dst, src, xor_mask, 0x1f, mask — __shfl_xor_sync

clamp field:
  0x1f (31) for idx/down/bfly: out-of-range lanes clamp to [0, width-1]
  0x00 for up: out-of-range lanes clamp to [0, lane_id] (returns own value)

mask field (always 0xffffffff for all lanes):
  bitmask, bit i = 1 means lane i participates
```
