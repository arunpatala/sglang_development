# DATA TYPES BASICS
## Every floating-point type used in LLM inference kernels

---

## The One-Sentence Version

Every data type is just a bit pattern. The number of bits decides how many values you can represent (range) and how accurately (precision). LLM inference uses small types to save memory bandwidth.

---

## Part 1 — What a Floating-Point Number Actually Is

A floating-point number stores three things packed into its bits:

```
Sign  ×  2^Exponent  ×  Mantissa

Sign:     1 bit   — positive or negative
Exponent: N bits  — how big (the "scale")
Mantissa: M bits  — how precise (the "detail")
```

Concrete example: the number `1.5` in float32 (32-bit float):

```
bit layout of float32:
 31   30    23  22                    0
 ┌─┬──────────┬────────────────────────┐
 │S│ Exponent │       Mantissa         │
 │0│ 01111111 │ 10000000000000000000000│
 └─┴──────────┴────────────────────────┘
  ↑      ↑                ↑
  +    127 (bias)       .5 (1.10 in binary = 1 + 0.5)

Value = (-1)^0 × 2^(127-127) × 1.1_binary = 1.0 × 1.5 = 1.5
```

The key insight: **more exponent bits → bigger range. More mantissa bits → more precision.**

---

## Part 2 — All the Types, Side by Side

```
Type          Bits  Sign  Exp  Mantissa  Max value       Notes
────────────────────────────────────────────────────────────────────
float (fp32)   32    1     8     23      3.4 × 10^38     default C float
half  (fp16)   16    1     5     10      6.5 × 10^4      standard ML type
bfloat16       16    1     8      7      3.4 × 10^38     Google Brain float
fp8 e4m3        8    1     4      3      448             NVidia Ada+
fp8 e5m2        8    1     5      2      57344           NVidia Ada+
────────────────────────────────────────────────────────────────────
```

### Why this matters for LLM inference:

- **float32**: used for accumulation (inside GEMM), weight updates, loss computation
- **float16**: model weights and activations in most inference today (half the memory of fp32)
- **bfloat16**: same range as fp32 (avoids overflow), used in training and Llama-family models
- **fp8 e4m3**: weights and activations in quantized inference (Ada/Hopper native), 2× smaller than fp16
- **fp8 e5m2**: used for gradients (needs wider range), less common in inference

---

## Part 3 — How They Look in CUDA C++

### float32 — plain `float`
```cpp
float x = 1.5f;
float y = x * 2.0f;   // normal C arithmetic
```

### float16 — `__half` (from `<cuda_fp16.h>`)
```cpp
#include <cuda_fp16.h>

__half h = __float2half(1.5f);   // float → half
float  f = __half2float(h);       // half  → float

// You CANNOT do arithmetic directly in old CUDA:
// __half z = h * h;  // might not work — use float2half(__half2float(h) * __half2float(h))

// But you CAN use the intrinsics:
__half z = __hmul(h, h);          // half * half → half (hardware instruction)
__half w = __hadd(h, h);          // half + half → half
__half2 pair = __halves2half2(h, h);  // pack two halves into one 32-bit register
```

### bfloat16 — `nv_bfloat16` (from `<cuda_bf16.h>`)
```cpp
#include <cuda_bf16.h>

nv_bfloat16 b = __float2bfloat16(1.5f);
float       f = __bfloat162float(b);

// Same intrinsic pattern as half:
nv_bfloat16 z = __hmul(b, b);
```

### fp8 — `__nv_fp8_e4m3` / `__nv_fp8_e5m2` (from `<cuda_fp8.h>`)
```cpp
#include <cuda_fp8.h>

__nv_fp8_e4m3 q = __nv_fp8_e4m3(1.5f);   // float → fp8 (Ada+ only)
float f = (float)q;                         // fp8 → float

// fp8 has NO hardware arithmetic — you always:
// 1. Load fp8
// 2. Convert to fp16/fp32
// 3. Compute in fp16/fp32
// 4. Convert back to fp8
// 5. Store fp8
```

---

## Part 4 — The Vector Types (This Is Where It Gets Important)

CUDA has vector types that pack multiple scalar values into a single wider type.
These are the types you use for 128-bit loads.

### Built-in CUDA vector types:

```
Scalar type    × 2           × 4           × 8 (not built-in, use uint4)
───────────────────────────────────────────────────────────────────────
float          float2         float4        → 4 floats = 128 bits ✓
int            int2           int4          → 4 ints   = 128 bits ✓
uint           uint2          uint4         → 4 uints  = 128 bits ✓
short          short2         short4        
half           half2          (no half4)    → use uint4 reinterpret for 8 halves
```

### How they are defined (what they look like internally):

```cpp
// These are literally structs with fields x, y, z, w:
struct float2 { float x, y; };
struct float4 { float x, y, z, w; };
struct int4   { int x, y, z, w;   };
struct uint4  { unsigned int x, y, z, w; };

// Access individual fields:
float4 v;
v.x = 1.0f;
v.y = 2.0f;
float sum = v.x + v.y;   // direct field access
```

### The `half2` type — two fp16 in one 32-bit register:

```cpp
// half2 is special: it's a 32-bit type holding two fp16 values packed together.
// CUDA has hardware instructions that operate on both halves simultaneously.
__half2 pair;
pair.x = 1.0_h;   // first fp16
pair.y = 2.0_h;   // second fp16

// Make a half2 from two halves:
__half a = __float2half(1.5f);
__half b = __float2half(2.5f);
__half2 ab = __halves2half2(a, b);

// Math on both elements simultaneously (one instruction!):
__half2 doubled = __hmul2(ab, __float2half2_rn(2.0f));   // [3.0, 5.0]
```

---

## Part 5 — Precision vs Range Trade-offs

```
              fp8 e4m3         fp16          bfloat16         float32
Bits:            8              16               16              32
Range:        [-448, 448]   [-65504, 65504]  [-3.4e38, 3.4e38]  [-3.4e38, 3.4e38]
Spacing at 1:   0.125          0.001           0.008             1.2e-7
Smallest >0:    0.015625       6e-8            1.2e-38           1.2e-38
```

**What "spacing at 1" means:** if you store the value `1.0`, the next representable value is `1 + spacing`. Anything between them gets rounded.

- fp8: spacing of 0.125 → values must differ by at least 12.5% to be distinguished
- fp16: spacing of 0.001 → 0.1% precision, enough for most inference
- bfloat16: wider range than fp16 (same as fp32 exponent) but coarser mantissa

**Why bfloat16 over fp16 for training?**  
fp16 overflows at 65504. Gradient magnitudes often exceed this. bfloat16 can represent up to 3.4e38, same as fp32, so no overflow.

**Why fp8 for quantized inference?**  
fp8 is 2× smaller than fp16 → 2× faster memory loads → 2× faster memory-bound kernels (RMSNorm, RoPE, etc.) at the cost of slight accuracy loss.

---

## Part 6 — Type Conversion Functions (the full list)

These appear constantly in kernels. Memorize the pattern: `__type_to_type()`.

```cpp
// float ↔ half
__float2half(x)         // float → __half (rounds to nearest)
__float2half_rn(x)      // float → __half (explicit round-to-nearest)
__half2float(x)         // __half → float

// float ↔ bfloat16  
__float2bfloat16(x)     // float → nv_bfloat16
__bfloat162float(x)     // nv_bfloat16 → float

// float2 ↔ half2
__float22half2_rn(x)    // float2 → __half2
__half22float2(x)       // __half2 → float2

// float2 ↔ bfloat162
__float22bfloat162_rn(x)  // float2 → __nv_bfloat162
__bfloat1622float2(x)     // __nv_bfloat162 → float2

// half ↔ half2
__half2half2(x)         // broadcast one half to both lanes of half2
__halves2half2(a, b)    // pack two halves into half2

// Reading raw bits (useful for packing — see PACKING_basics.md)
__half_as_ushort(x)     // __half → uint16 (same bits, no conversion)
__ushort_as_half(x)     // uint16 → __half (same bits, no conversion)
```

---

## Part 7 — How the GPU Stores These in Registers

A GPU register is **32 bits**. This is the fundamental unit.

```
1 float32   = 1 register       (32 bits)
1 float16   = ½ register       (16 bits) — but you use a 32-bit register, waste 16 bits
1 half2     = 1 register       (32 bits) — two fp16 packed, no waste
1 float4    = 4 registers      (128 bits) — treated as a "vector register group"
1 uint4     = 4 registers      (128 bits) — same, different type label
1 fp8       = ¼ register       (8 bits)  — stored in a 32-bit register, waste 24 bits
```

**Practical implication:** if your kernel uses many `__half` variables, each wastes half a register. Use `__half2` or `uint4` packing to use registers efficiently (more elements per register = more parallelism per SM).

---

## Part 8 — The `__restrict__` Keyword

You see `__restrict__` on every pointer in production kernels:

```cpp
__global__ void kernel(const __half* __restrict__ src, __half* __restrict__ dst, int n)
```

`__restrict__` tells the compiler: **these pointers do not alias** (they don't point to overlapping memory). Without it, every store might invalidate the cached value of every load — the compiler generates conservative code. With it, the compiler can keep loaded values in registers across store instructions, generating faster PTX.

**Always use `__restrict__` on kernel pointer arguments.**

---

## Part 9 — Common Mistakes with Types

```
Mistake                          Why it's wrong                   Fix
───────────────────────────────────────────────────────────────────────────────
__half a = 1.5;                  1.5 is double, implicit cast     __half a = __float2half(1.5f);
a * b where a,b are __half       No * operator in old CUDA        __hmul(a, b)
(float)bfloat16_val              Won't compile                    __bfloat162float(val)
storing float into __half*       Writes 4 bytes into 2-byte slot  use __float2half first
float4 from unaligned ptr        misaligned memory access crash    ensure ptr % 16 == 0
half arithmetic in fp16 mode     precision loss on accumulation    accumulate in float32
```

---

## Part 10 — What You Will See in Every Kernel

This pattern appears in RMSNorm, RoPE, activation, attention — everything:

```cpp
// The universal LLM kernel pattern for fp16 input/output:

// 1. LOAD: 8 fp16 values in one 128-bit instruction
uint4 raw = ((const uint4*)src)[thread_idx];
const __half* vals = reinterpret_cast<const __half*>(&raw);

// 2. COMPUTE: upcast to float32 for accuracy, do math
float acc = 0.0f;
#pragma unroll
for (int i = 0; i < 8; ++i) {
    float v = __half2float(vals[i]);
    acc += v * v;   // example: compute sum of squares for RMSNorm
}

// 3. STORE: pack results back into uint4 and write 128 bits at once
uint4 out_raw;
__half* out_vals = reinterpret_cast<__half*>(&out_raw);
float rms_inv = rsqrtf(acc / 8.0f + eps);
#pragma unroll
for (int i = 0; i < 8; ++i) {
    out_vals[i] = __float2half(__half2float(vals[i]) * rms_inv);
}
((uint4*)dst)[thread_idx] = out_raw;
```

This exact pattern is in `flashinfer/include/flashinfer/norm.cuh`. Once you understand the data types and packing, that file becomes completely readable.

---

## Quick Reference Card

```
Need to...                              Use this
──────────────────────────────────────────────────────────────────
Load 4 float32 at once (128-bit)        float4
Load 8 float16 at once (128-bit)        uint4  (reinterpret as __half*)
Load 2 float16 as a pair (32-bit)       __half2
Convert float32 → float16              __float2half(x)
Convert float16 → float32              __half2float(x)
Multiply two float16                    __hmul(a, b)
Add two float16                         __hadd(a, b)
Multiply two half2 (2 elements each)    __hmul2(a, b)
Check if two pointers alias             add __restrict__ to both
Store result as float16                 __float2half(result)
```
