# PTX BASICS
## What PTX is, how to read it, and how to write it inline

---

## The One-Sentence Version

PTX is the assembly language of NVIDIA GPUs. CUDA C++ compiles to PTX first, then PTX compiles to actual machine code (SASS). You read PTX to understand what the hardware actually does, and write inline PTX to access GPU instructions that C++ can't express.

---

## Part 1 — The Compilation Pipeline

```
Your CUDA C++ code
        │
        │  nvcc (NVIDIA compiler)
        ▼
     PTX code                   ← human-readable intermediate assembly
        │
        │  ptxas (PTX assembler)
        ▼
     SASS (Streaming ASSembler)  ← actual GPU machine code (binary)
        │
        │  GPU executes SASS
        ▼
     Results on GPU
```

**PTX** (Parallel Thread Execution) is a virtual ISA — an intermediate assembly language that NVIDIA defines and keeps stable across GPU generations. The same PTX can run on Pascal, Volta, Ampere, Ada, and Hopper (with the compiler filling in architecture-specific optimizations).

**SASS** is the actual binary that runs on the chip. You rarely read SASS directly — PTX is the right level to inspect.

### How to get PTX from your kernel:
```bash
# Compile a .cu file to PTX:
nvcc --ptx -arch=sm_89 -O3 my_kernel.cu -o my_kernel.ptx
cat my_kernel.ptx

# Or compile and save PTX alongside the binary:
nvcc -arch=sm_89 -O3 --keep my_kernel.cu
# (generates my_kernel.ptx, my_kernel.cubin, etc.)

# In load_inline, add this to extra_cuda_cflags:
extra_cuda_cflags=["-O3", "-arch=sm_89", "--keep"]
# PTX saved to /tmp/torch_extensions/.../
```

---

## Part 2 — How to Read PTX

PTX looks like assembly: instructions operate on virtual registers. There are unlimited virtual registers (the compiler assigns real registers later).

### Register naming:
```ptx
%rd1    — 64-bit register (d = doubleword), used for addresses/pointers
%r1     — 32-bit register (r = register), used for int32 / uint32
%f1     — 32-bit float register
%rs1    — 16-bit register (s = short), used for fp16 / uint16
%p1     — predicate register (1-bit, true/false), used for branches
```

### A complete annotated PTX example

This is the PTX for a scalar fp16 copy kernel:

```ptx
.visible .entry copy_scalar(
    .param .u64 copy_scalar_param_0,    // src pointer (64-bit address)
    .param .u64 copy_scalar_param_1,    // dst pointer
    .param .u32 copy_scalar_param_2)    // n (integer)
{
    .reg .u16  %rs<2>;    // declare 16-bit register variables
    .reg .u32  %r<5>;     // declare 32-bit registers
    .reg .u64  %rd<5>;    // declare 64-bit registers
    .reg .pred %p<2>;     // declare predicate registers

    // Load function parameters from .param space into registers:
    ld.param.u64 %rd1, [copy_scalar_param_0];   // src
    ld.param.u64 %rd2, [copy_scalar_param_1];   // dst
    ld.param.u32 %r2,  [copy_scalar_param_2];   // n

    // Compute thread index: blockIdx.x * blockDim.x + threadIdx.x
    mov.u32 %r1, %ctaid.x;       // blockIdx.x
    mov.u32 %r3, %ntid.x;        // blockDim.x
    mov.u32 %r4, %tid.x;         // threadIdx.x
    mad.lo.s32 %r5, %r1, %r3, %r4;   // r5 = blockIdx.x * blockDim.x + threadIdx.x

    // Bounds check: if (idx >= n) return
    setp.ge.s32 %p1, %r5, %r2;   // p1 = (r5 >= n)
    @%p1 bra $L__BB0_2;           // if p1 is true, jump to return label

    // Compute address: src + idx * 2 (fp16 = 2 bytes)
    mul.wide.s32 %rd3, %r5, 2;
    add.s64 %rd4, %rd1, %rd3;    // rd4 = src + idx*2

    // LOAD: 16-bit load from global memory
    ld.global.cs.u16 %rs1, [%rd4];   // ← THIS is the 16-bit load you want to avoid

    // Compute dst address:
    add.s64 %rd5, %rd2, %rd3;
    
    // STORE: 16-bit store
    st.global.wb.u16 [%rd5], %rs1;

$L__BB0_2:
    ret;
}
```

Now compare this to the **vectorized** version:

```ptx
// The key difference — the load instruction:

// Scalar (16-bit):
ld.global.cs.u16 %rs1, [%rd4];

// Vectorized (128-bit):
ld.global.ca.v4.u32 {%r1, %r2, %r3, %r4}, [%rd4];
//                   ─────────────────────
//                   four 32-bit registers filled by ONE instruction
```

---

## Part 3 — The Key PTX Instructions (the ones that actually matter)

### Memory Instructions

```ptx
ld.global.ca.TYPE  dst, [addr];     // load from global memory (cache all — L1+L2)
ld.global.cs.TYPE  dst, [addr];     // load from global memory (cache streaming — L2 only)
ld.global.cg.TYPE  dst, [addr];     // load from global memory (cache in L2 only)
st.global.wb.TYPE  [addr], src;     // store to global memory (write-back)
st.global.cs.TYPE  [addr], src;     // store, cache streaming
```

**Types for load/store:**
```ptx
u16    — 16-bit unsigned (fp16 scalar)
u32    — 32-bit unsigned
u64    — 64-bit unsigned
f32    — 32-bit float
f64    — 64-bit float
v2.u32 — vector of 2 × u32 = 64-bit load (float2 / uint2)
v4.u32 — vector of 4 × u32 = 128-bit load (float4 / uint4) ← most important
v4.f32 — vector of 4 × f32 = 128-bit load (float4 with float arithmetic)
```

**Cache hints:**
```
.ca  = cache all        — put in L1 and L2  (good for data reused soon)
.cs  = cache streaming  — put in L2 only    (good for one-time reads, like streaming copy)
.cg  = cache global     — put in L2 only, don't evict others from L1
.wb  = write-back       — update L2 on store
.wt  = write-through    — bypass L2, go straight to GDDR6X
```

For the **scalar copy**, the compiler chose `.cs` (cache streaming) because it sees the data is accessed only once. For **compute kernels** where you reuse data (attention tiles), use `.ca`.

---

### Math Instructions

These are the ones that appear in every LLM kernel:

```ptx
// ex2.approx — computes 2^x (single instruction)
// Used instead of expf() because 2^x = e^(x*ln2), one PTX instruction vs ~20
ex2.approx.ftz.f32  %f_out, %f_in;
// .ftz = flush-to-zero: subnormal inputs become 0 (faster, acceptable for ML)

// lg2.approx — computes log2(x) (single instruction)  
lg2.approx.ftz.f32  %f_out, %f_in;

// rsqrt.approx — computes 1/sqrt(x) (single instruction)
// Used in RMSNorm, LayerNorm
rsqrt.approx.ftz.f32  %f_out, %f_in;

// rcp.approx — computes 1/x (single instruction)
rcp.approx.ftz.f32  %f_out, %f_in;

// tanh.approx — computes tanh(x) (used in GeLU activation)
tanh.approx.f32  %f_out, %f_in;
// Also: tanh.approx.f16x2 for two fp16 simultaneously

// mad — multiply-add: dst = a * b + c (fused, one instruction)
mad.rn.f32  %f_out, %f_a, %f_b, %f_c;   // dst = a*b + c

// fma — same as mad but different syntax
fma.rn.f32  %f_out, %f_a, %f_b, %f_c;

// mul, add, sub — scalar arithmetic
mul.f32  %f_out, %f_a, %f_b;
add.f32  %f_out, %f_a, %f_b;
```

**Why use `ex2.approx` instead of `expf()`?**

`expf(x)` compiles to multiple PTX instructions (Taylor series approximation in software). The hardware instruction `ex2.approx` does it in ONE clock cycle. Since softmax computation calls `exp()` for every element of the attention matrix, this is a massive speedup.

To use `exp` via PTX: `exp(x) = 2^(x * log2(e)) = ex2(x * 1.44269504f)`

---

### Warp Shuffle Instructions

These let threads within a warp exchange data WITHOUT going through shared memory.
Used in warp-level reduce (Lesson 2), online softmax (Lesson 4).

```ptx
// shfl.sync.bfly — butterfly (XOR) shuffle
// Each thread gets the value from thread (lane_id XOR mask)
shfl.sync.bfly.b32  %r_out, %r_in, mask, 0x1f, 0xffffffff;
//                                  ────  ────  ──────────
//                                  XOR   max   all threads active
//                  mask for XOR: 1,2,4,8,16 → warp-level reduce

// Example: mask=1 means thread 0 gets from thread 1, thread 1 gets from thread 0, etc.
// After XOR shuffle with mask=16,8,4,2,1: each thread has the warp sum
```

The C++ wrapper for this (from `flashinfer/math.cuh`):
```cpp
__forceinline__ __device__ float shfl_xor_sync(float x, int mask) {
    float y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(y) : "f"(x), "r"(mask));
    return y;
}

// Warp reduce sum using shuffles:
float warp_reduce_sum(float val) {
    val += shfl_xor_sync(val, 16);   // exchange with threads ±16
    val += shfl_xor_sync(val, 8);    // exchange with threads ±8
    val += shfl_xor_sync(val, 4);    // exchange with threads ±4
    val += shfl_xor_sync(val, 2);    // exchange with threads ±2
    val += shfl_xor_sync(val, 1);    // exchange with threads ±1
    return val;   // all 32 threads now have the sum of all 32 values
}
```

---

### Control Flow Instructions

```ptx
setp.ge.s32  %p1, %r1, %r2;   // p1 = (r1 >= r2)  [s32 = signed 32-bit compare]
setp.lt.u64  %p1, %rd1, %rd2; // p1 = (rd1 < rd2) [u64 = unsigned 64-bit compare]
@%p1 bra $LABEL;               // if p1 is true, jump to LABEL
@!%p1 bra $LABEL;              // if p1 is false, jump to LABEL

ret;                            // return from kernel
bar.sync 0;                    // __syncthreads() — block-level barrier
```

---

## Part 4 — Writing Inline PTX

You can embed PTX directly in CUDA C++ using `asm volatile`. This is how FlashInfer implements its fast math functions.

### The syntax:
```cpp
asm volatile(
    "ptx_instruction_here %output, %input;"  // PTX string
    : "=TYPE"(output_var)                    // output operands (= means written to)
    : "TYPE"(input_var)                      // input operands
);
```

**Type codes:**
```
"=f"  — float32 output
"=r"  — 32-bit int/uint output
"=h"  — 16-bit output (for fp16)
"=l"  — 64-bit output
"f"   — float32 input
"r"   — 32-bit int input
"h"   — 16-bit input
```

### Real examples from `flashinfer/math.cuh`:

```cpp
// 2^x: ex2.approx in one PTX instruction
__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    //                                 ──  ──     ──────   ───────
    //                            output  input  output    input
    //                            reg     reg    binding   binding
    return y;
}

// log2(x)
__forceinline__ __device__ float ptx_log2(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 1/sqrt(x)
__forceinline__ __device__ float rsqrt(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 2^x for fp16 (processes one fp16 value):
__forceinline__ __device__ half ptx_exp2(half x) {
    ushort y_u16;
    asm volatile("ex2.approx.f16 %0, %1;"
                 : "=h"(y_u16)
                 : "h"(__half_as_ushort(x)));
    //                  ─────────────────
    //                  __half → ushort (same bits) because PTX wants "h" type
    return __ushort_as_half(y_u16);
}

// 2^x for two fp16 simultaneously (half2):
__forceinline__ __device__ half2 ptx_exp2(half2 x) {
    uint32_t y_u32;
    uint32_t x_u32 = *(uint32_t*)&x;   // reinterpret half2 as uint32
    asm volatile("ex2.approx.f16x2 %0, %1;" : "=r"(y_u32) : "r"(x_u32));
    return *(half2*)&y_u32;
}
```

### Why `volatile`?

`asm volatile` tells the compiler: **never optimize this away or reorder it**. Without `volatile`, the compiler might decide the inline assembly has no visible side effects and delete it entirely.

---

## Part 5 — The `ftz` Modifier Explained

You see `.ftz` on almost every PTX math instruction:
```ptx
ex2.approx.ftz.f32
rsqrt.approx.ftz.f32
```

`ftz` = **flush to zero**. 

IEEE-754 defines "subnormal" numbers — very small values close to zero with reduced precision. Handling subnormals in hardware takes extra cycles. `.ftz` says: "if the input or output would be a subnormal, just make it zero instead."

For ML workloads this is almost always acceptable — subnormal attention weights or activations are so close to zero they don't affect output quality. The speedup is significant on some architectures.

**The exception:** sampling kernels that compute log-probabilities. Values near zero matter there. That's why `flashinfer/sampling.cuh` uses inline PTX to *bypass* `--use_fast_math` and get IEEE-754 compliant arithmetic.

---

## Part 6 — `--use_fast_math` and What It Does

Adding `--use_fast_math` to `nvcc` flags automatically:
1. Enables `.ftz` on all float operations
2. Replaces `sinf`, `cosf`, `expf`, `logf` with approximate hardware versions
3. Enables `__sincosf` (compute sin and cos simultaneously)
4. Enables `fma` everywhere (fused multiply-add, no intermediate rounding)

```bash
nvcc -O3 -arch=sm_89 --use_fast_math kernel.cu
```

In `load_inline`:
```python
extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"]
```

**Use for:** all elementwise kernels, RMSNorm, RoPE, activation, attention compute  
**Don't use for:** sampling/probability kernels where numerical correctness matters

---

## Part 7 — Reading PTX Practically

When you want to verify your kernel uses 128-bit loads:

```bash
# 1. Compile your kernel to PTX:
nvcc --ptx -arch=sm_89 -O3 my_kernel.cu -o my_kernel.ptx

# 2. Look for load instructions:
grep "ld.global" my_kernel.ptx

# Good (128-bit): 
#   ld.global.ca.v4.u32  {%r1, %r2, %r3, %r4}, [%rd1];
#   ld.global.ca.v4.f32  {%f1, %f2, %f3, %f4}, [%rd1];

# Bad (16-bit):
#   ld.global.cs.u16  %rs1, [%rd1];
```

When you want to check your kernel uses fast math:
```bash
grep "ex2.approx\|rsqrt.approx\|lg2.approx" my_kernel.ptx
# If these appear → fast math PTX is being used
# If you see libdevice calls instead → --use_fast_math is missing
```

When you want to count instructions per thread:
```bash
wc -l my_kernel.ptx
# Fewer lines generally = fewer instructions = more efficient
```

---

## Part 8 — Special PTX Variables (threadIdx, blockIdx, etc.)

```ptx
%tid.x,  %tid.y,  %tid.z   — threadIdx.x, threadIdx.y, threadIdx.z
%ntid.x, %ntid.y, %ntid.z  — blockDim.x,  blockDim.y,  blockDim.z
%ctaid.x, %ctaid.y         — blockIdx.x,  blockIdx.y
%nctaid.x                  — gridDim.x

%laneid                    — lane within warp (0-31), no C++ equivalent
%warpid                    — warp within SM (0-63 on Ada)
%smid                      — SM index (which SM is running this block)
```

`%laneid` is especially useful — you don't need to compute `threadIdx.x % 32` in PTX:
```ptx
mov.u32 %r_lane, %laneid;   // directly gives lane 0-31
```

In C++:
```cpp
// Get lane ID in C++ (compiles to mov.u32 %r, %laneid):
unsigned lane_id = threadIdx.x & 31;   // equivalent to threadIdx.x % 32
```

---

## Part 9 — The Most Important PTX Instructions for This Curriculum

Ranked by how often they appear in LLM kernels:

```
Priority  Instruction          C++ equivalent      Where used
────────────────────────────────────────────────────────────────────────
  1     ld.global.ca.v4.u32   uint4 load          Every kernel
  2     st.global.wb.v4.u32   uint4 store         Every kernel
  3     ex2.approx.ftz.f32    ptx_exp2()          Softmax, online softmax
  4     rsqrt.approx.ftz.f32  math::rsqrt()       RMSNorm, LayerNorm
  5     shfl.sync.bfly.b32    __shfl_xor_sync()   Warp reduce (Lesson 2)
  6     mad.rn.f32            fmaf()              GEMM, dot products
  7     tanh.approx.f32       tanhf()             GeLU activation
  8     lg2.approx.ftz.f32    ptx_log2()          Online softmax (LSE)
  9     bar.sync              __syncthreads()     Shared memory sync
 10     setp + @bra           if() bounds check   Every kernel
────────────────────────────────────────────────────────────────────────
```

---

## Part 10 — Common Inline PTX Patterns to Copy-Paste

```cpp
// Fast exp2 (use instead of expf)
__forceinline__ __device__ float exp2_approx(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Usage: exp(x) == exp2_approx(x * 1.44269504f)

// Fast log2 (use instead of logf)
__forceinline__ __device__ float log2_approx(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Usage: log(x) == log2_approx(x) * 0.693147180f

// Fast rsqrt (use instead of 1.0f/sqrtf(x))
__forceinline__ __device__ float rsqrt_approx(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Usage: 1.0f/sqrt(x) == rsqrt_approx(x)

// Warp shuffle (for reduce — Lesson 2)
__forceinline__ __device__ float shfl_xor(float x, int mask) {
    float y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(y) : "f"(x), "r"(mask));
    return y;
}

// Warp reduce sum (combine all 5 patterns above)
__forceinline__ __device__ float warp_sum(float x) {
    x += shfl_xor(x, 16);
    x += shfl_xor(x, 8);
    x += shfl_xor(x, 4);
    x += shfl_xor(x, 2);
    x += shfl_xor(x, 1);
    return x;
}

// Warp reduce max
__forceinline__ __device__ float warp_max(float x) {
    float y;
    // shfl_xor then fmaxf:
    #define SHFL_XOR_F(x, m) ({ float _t; asm volatile("shfl.sync.bfly.b32 %0,%1,%2,0x1f,0xffffffff;":"=f"(_t):"f"(x),"r"(m)); _t; })
    x = fmaxf(x, SHFL_XOR_F(x, 16));
    x = fmaxf(x, SHFL_XOR_F(x, 8));
    x = fmaxf(x, SHFL_XOR_F(x, 4));
    x = fmaxf(x, SHFL_XOR_F(x, 2));
    x = fmaxf(x, SHFL_XOR_F(x, 1));
    return x;
}
```

---

## Quick Reference

```
To see PTX:           nvcc --ptx -arch=sm_89 -O3 file.cu -o file.ptx
128-bit load in PTX:  ld.global.ca.v4.u32 {%r1,%r2,%r3,%r4}, [%rd1]
16-bit load in PTX:   ld.global.cs.u16 %rs1, [%rd1]        ← bad, fix with uint4 cast
Fast exp:             ex2.approx.ftz.f32  (NOT libdevice__expf)
Fast sqrt recip:      rsqrt.approx.ftz.f32
Warp exchange:        shfl.sync.bfly.b32
Inline PTX syntax:    asm volatile("instr %0,%1;" : "=f"(out) : "f"(in));
ftz:                  flush subnormals to zero (faster, fine for ML)
```
