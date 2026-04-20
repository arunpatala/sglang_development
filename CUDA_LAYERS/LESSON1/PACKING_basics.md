# PACKING BASICS
## How and why we pack multiple values into one register

---

## The One-Sentence Version

Packing means cramming multiple small values (like 8 fp16 numbers) into one big register (128 bits), so you move all of them with a single load instruction instead of 8 separate ones.

---

## Part 1 — Why Packing Exists: The Memory Bus Problem

The GPU memory bus is **128 bits wide** at the thread level. Every time a thread sends a memory request, the smallest useful payload is 128 bits.

If you request 16 bits (one fp16), you get 128 bits back — but you only use 16 of them. The other 112 bits were fetched for nothing.

```
Memory bus: ─────────────────128 bits─────────────────

Scalar fp16 load:
  Thread wants:  [__half][              wasted              ]
                   16 b                 112 b

Packed uint4 load:
  Thread wants:  [__half][__half][__half][__half][__half][__half][__half][__half]
                  all 128 bits used → 8x more useful work per request
```

Packing is the technique of filling that 128-bit slot completely, every time.

---

## Part 2 — The 128-Bit Types: What They Are

CUDA has three built-in 128-bit types. They are all the same width — only the type label changes:

```cpp
float4  v;   // 4 × float32 = 128 bits
int4    v;   // 4 × int32   = 128 bits  
uint4   v;   // 4 × uint32  = 128 bits  ← most commonly used for fp16 packing
```

These are structs with four named fields:

```cpp
// uint4 definition (from CUDA headers):
struct uint4 {
    unsigned int x;   // 32 bits
    unsigned int y;   // 32 bits
    unsigned int z;   // 32 bits
    unsigned int w;   // 32 bits
    //               = 128 bits total
};
```

The GPU treats the four 32-bit fields as one atomic 128-bit unit for loads and stores.
One load instruction, one memory request, four 32-bit registers filled simultaneously.

---

## Part 3 — How We Pack fp16 Into uint4

One `uint32` can hold **two fp16** values (each fp16 = 16 bits, two fit in 32 bits).
One `uint4` has four `uint32` values → **eight fp16** values total.

```
uint4:
  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │   x (32-bit) │   y (32-bit) │   z (32-bit) │   w (32-bit) │
  ├───────┬──────┼───────┬──────┼───────┬──────┼───────┬──────┤
  │ fp16_0│fp16_1│ fp16_2│fp16_3│ fp16_4│fp16_5│ fp16_6│fp16_7│
  └───────┴──────┴───────┴──────┴───────┴──────┴───────┴──────┘
     16b   16b    16b   16b    16b   16b    16b   16b
  = 128 bits total = 8 fp16 values
```

We access those 8 fp16 values using `reinterpret_cast` — we tell the compiler "treat this 128-bit blob as an array of 8 halves":

```cpp
uint4 raw = /* loaded from memory */;

// reinterpret: no conversion, no copy — just a different "view" of the same bits
__half* vals = reinterpret_cast<__half*>(&raw);

// Now vals[0] through vals[7] are the 8 fp16 values
float v0 = __half2float(vals[0]);   // access first fp16
float v7 = __half2float(vals[7]);   // access eighth fp16
```

**`reinterpret_cast` does NOT touch the bits.** It's a compile-time instruction that tells the compiler how to interpret a pointer. Zero cost at runtime.

---

## Part 4 — The Actual Load and Store Instructions

### Loading 8 fp16 values (128-bit load):
```cpp
// src points to fp16 data in global memory
// idx is the thread's position in units of "8-element groups"

uint4 raw = ((const uint4*)src)[idx];
//          ─────┬────────
//               └── cast src (a __half*) to uint4* so the compiler
//                   knows to fetch 128 bits at once
```

What this looks like in PTX (you can see this with `nvcc --ptx`):
```ptx
ld.global.ca.v4.u32  {%r1, %r2, %r3, %r4}, [%rd4];
```
- `v4.u32` = four 32-bit unsigned integers = 128 bits
- `{%r1, %r2, %r3, %r4}` = four registers filled simultaneously
- one instruction

### Storing 8 fp16 results (128-bit store):
```cpp
uint4 out_raw;
__half* out_vals = reinterpret_cast<__half*>(&out_raw);

// fill out_vals[0..7] with computed results...

((uint4*)dst)[idx] = out_raw;
//              ─┬─
//               └── same trick: cast dst to uint4* to emit a 128-bit store
```

PTX:
```ptx
st.global.wb.v4.u32  [%rd5], {%r1, %r2, %r3, %r4};
```

---

## Part 5 — The Full Pattern (Memorize This)

This exact code appears in every memory-bound LLM kernel: RMSNorm, RoPE, SiLU, embedding lookup, attention output scaling.

```cpp
__global__ void elementwise_fp16(
    const __half* __restrict__ src,   // input: fp16 array
    __half*       __restrict__ dst,   // output: fp16 array
    int           n_vec)              // n_elements / 8
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // STEP 1: 128-bit load — one instruction fetches 8 fp16 values
    uint4 raw_in = ((const uint4*)src)[idx];

    // STEP 2: reinterpret as array of 8 fp16 (free, compile-time only)
    const __half* in_vals = reinterpret_cast<const __half*>(&raw_in);

    // STEP 3: allocate output register group (128-bit, all in registers)
    uint4 raw_out;
    __half* out_vals = reinterpret_cast<__half*>(&raw_out);

    // STEP 4: process each of the 8 elements
    //         #pragma unroll tells compiler to eliminate the loop entirely
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = __half2float(in_vals[i]);  // upcast to fp32
        v = /* your operation here */;
        out_vals[i] = __float2half(v);        // downcast back to fp16
    }

    // STEP 5: 128-bit store — one instruction writes all 8 results
    ((uint4*)dst)[idx] = raw_out;
}
```

After `#pragma unroll`, the compiler eliminates the loop and produces 8 sequential operations with no branching overhead. The result is identical to hand-writing the operation 8 times.

---

## Part 6 — Packing for float32: float4

For float32 tensors, one `float4` holds 4 values (4 × 32 bits = 128 bits):

```cpp
__global__ void elementwise_f32(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    float        scale,
    int          n_vec)   // n_elements / 4
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    float4 v = ((const float4*)src)[idx];   // load 4 floats

    // access individual elements via .x .y .z .w
    v.x *= scale;
    v.y *= scale;
    v.z *= scale;
    v.w *= scale;

    ((float4*)dst)[idx] = v;   // store 4 floats
}
```

`float4` uses the `.x`, `.y`, `.z`, `.w` fields — no `reinterpret_cast` needed because the type matches directly.

---

## Part 7 — half2: Packing at the Arithmetic Level

`half2` is different from `uint4`. Instead of being a load container, `half2` exists because the GPU has **hardware instructions that process two fp16 values simultaneously**.

```
half2 in one 32-bit register:
  ┌──────────────┬──────────────┐
  │   hi (__half)│   lo (__half)│
  └──────────────┴──────────────┘
         16 bits        16 bits
```

```cpp
__half a = __float2half(1.0f);
__half b = __float2half(2.0f);

// Pack two halves into one half2 (lives in one 32-bit register)
__half2 pair = __halves2half2(a, b);

// Now compute on BOTH simultaneously — one instruction:
__half2 doubled = __hmul2(pair, __float2half2_rn(2.0f));
// result: {2.0, 4.0} — both lanes computed in parallel

// Unpack:
__half hi = __high2half(doubled);   // 4.0
__half lo = __low2half(doubled);    // 2.0
```

**When to use `half2` vs `uint4`:**
- `uint4` = container for loading/storing 8 fp16 values at once (memory efficiency)
- `half2` = container for doing arithmetic on 2 fp16 values at once (compute efficiency)

In high-performance kernels, you load with `uint4`, then process with `half2` pairs.

---

## Part 8 — Alignment: The Critical Rule

**128-bit loads require 16-byte aligned pointers.**

```
Aligned pointer:     address is divisible by 16

Address 0x0000:  OK  (0 / 16 = 0, remainder 0)
Address 0x0010:  OK  (16 / 16 = 1, remainder 0)
Address 0x0001:  BAD (1 / 16 = 0, remainder 1) → crash or slow
Address 0x0008:  BAD (8 / 16 = 0, remainder 8) → crash or slow
```

**Why this matters:** If you try to load `uint4` from an address that isn't 16-byte aligned, one of two things happens:
1. On newer GPUs: the hardware splits it into two 64-bit loads (half the bandwidth)
2. On others: it's an illegal instruction → your kernel crashes silently (wrong results or CUDA error)

**PyTorch always allocates aligned memory:**
```python
x = torch.randn(N, device="cuda", dtype=torch.float16)
# x.data_ptr() is guaranteed to be aligned to at least 512 bytes
```

**When you slice a tensor, alignment can break:**
```python
x = torch.randn(1024, device="cuda", dtype=torch.float16)
x_slice = x[1:]   # now starts at offset 2 bytes — NOT 16-byte aligned!
# Passing x_slice.data_ptr() to a uint4 kernel → misaligned access
```

**The safe pattern: only use uint4 on whole tensors, not on byte-offset slices.**

---

## Part 9 — Stride and Index Arithmetic

When you cast `__half*` to `uint4*`, the index unit changes:

```cpp
// If src is a __half pointer:
// src[i] accesses fp16 element i  (offset: i × 2 bytes)

// When cast to uint4*:
// ((uint4*)src)[i] accesses 16 bytes starting at i × 16 bytes
//                 = 8 fp16 elements (i × 8 ... i × 8 + 7)

// Your thread idx must account for this:
int thread_idx = blockIdx.x * blockDim.x + threadIdx.x;
// thread_idx accesses elements [thread_idx*8 ... thread_idx*8 + 7]
// total threads needed = n_fp16_elements / 8
```

Launching the kernel:
```cpp
int n = tensor.numel();          // total fp16 elements
int n_vec = n / 8;               // one thread per group of 8
int grid = (n_vec + 255) / 256;
kernel<<<grid, 256>>>(src, dst, n_vec);
```

---

## Part 10 — Visual Summary of All Packing Patterns

```
Pattern          Code            Bits  Holds        Use for
─────────────────────────────────────────────────────────────────────
scalar fp16      __half*          16   1 fp16        naive (avoid)
scalar fp32      float*           32   1 fp32        small tensors only
half pair        __half2          32   2 fp16        arithmetic on pairs
float2           float2           64   2 fp32        medium loads
uint2            uint2            64   4 fp16        medium loads
float4           float4          128   4 fp32        128-bit load fp32  ✓
uint4            uint4           128   8 fp16        128-bit load fp16  ✓
int4             int4            128   4 int32       128-bit integer    ✓
─────────────────────────────────────────────────────────────────────
Target: always use 128-bit loads (float4 or uint4) for fp32 or fp16
```

---

## Part 11 — Debugging Packing Errors

**Symptom:** kernel produces wrong results but no crash.  
**Cause:** misaligned load silently reads the wrong bytes.  
**Fix:** print `(uintptr_t)ptr % 16` — should be 0.

**Symptom:** CUDA error "an illegal memory access was encountered".  
**Cause:** pointer not aligned to 16 bytes.  
**Fix:** only cast whole PyTorch tensors to `uint4*`, never slices.

**Symptom:** first few results correct, rest wrong.  
**Cause:** `n_vec` computed wrong (forgot to divide by 8).  
**Fix:** `n_vec = src.numel() / 8`, not `src.numel()`.

**Symptom:** last few elements wrong or missing.  
**Cause:** `numel()` not divisible by 8 — the last partial group is never processed.  
**Fix:** pad your tensor size to be divisible by 8, or handle the tail separately.

```cpp
// Tail handling (handle the last <8 elements after the vectorized loop):
int n_vec = n / 8;
int tail  = n % 8;
// vectorized kernel handles [0 ... n_vec*8 - 1]
// tail loop handles [n_vec*8 ... n-1]
for (int i = n_vec * 8 + threadIdx.x; i < n; i += blockDim.x) {
    dst[i] = __float2half(__half2float(src[i]) * scale);
}
```

---

## The One Pattern to Remember

```
LOAD:    uint4 raw = ((const uint4*)src)[idx];
UNPACK:  const __half* v = reinterpret_cast<const __half*>(&raw);
COMPUTE: for (int i = 0; i < 8; ++i) { out[i] = op(v[i]); }
PACK:    uint4 out_raw; __half* ov = reinterpret_cast<__half*>(&out_raw);
STORE:   ((uint4*)dst)[idx] = out_raw;
```

This pattern, with different `op()` bodies, is 90% of all elementwise LLM kernels.
