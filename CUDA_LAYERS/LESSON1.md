# Lesson 1 — 128-bit Vectorized Loads and the GPU Memory Hierarchy

**Phase:** 0.3  
**Time:** 1 day  
**Prerequisite:** Basic CUDA (grid/block/thread, `__global__`, global memory read/write)  
**Next lesson:** Lesson 2 — Warp Shuffle Reduce  

---

## Why This Is Lesson 1

Every kernel you will write in this curriculum — RMSNorm, RoPE, attention decode, MLP activation — is bottlenecked by how fast it reads and writes HBM (High Bandwidth Memory). If your memory access pattern is inefficient, no algorithmic optimization will save you.

The single most impactful change you can make to any elementwise kernel is switching from scalar loads to **128-bit vectorized loads**. This turns 8 separate 16-bit memory instructions into 1 single 128-bit instruction, saturating the memory pipeline.

After this lesson, you will understand exactly why `vec_dtypes.cuh` exists in FlashInfer, and you'll be able to use the same pattern in every kernel you write.

---

## Part 1 — The GPU Memory Hierarchy

Before writing any code, understand the hardware you're programming against.

Your GPU: **RTX 4060 Ti, Ada Lovelace, sm_89, 16 GB GDDR6X, 288 GB/s, CUDA 13.0**

```
CPU                    GPU  (RTX 4060 Ti — your machine)
 │                      │
 │                   ┌──┴─────────────────────────────────────┐
 │                   │  GDDR6X (High Bandwidth Memory)        │
 │                   │  RTX 4060 Ti: 288 GB/s (128-bit bus)  │
 │                   │  16 GB capacity, ~300 cycles latency   │
 │                   │  (cf. A100: 2,039 GB/s | H100: 3.35TB/s) │
 │                   └───────────────┬────────────────────────┘
 │                                   │  L2 Cache (32 MB on RTX 4060 Ti)
 │                                   │  ~80–120 cycles latency
 │                                   │
 │                   ┌───────────────┴────────────────────────┐
 │                   │  Streaming Multiprocessors (SMs)       │
 │                   │  34 SMs on RTX 4060 Ti (Ada)          │
 │                   │                                        │
 │                   │  ┌──────────────────────────────────┐  │
 │                   │  │  L1 / Shared Memory (SRAM)       │  │
 │                   │  │  up to 100 KB per SM (Ada)       │  │
 │                   │  │  ─────────────────────────────── │  │
 │                   │  │  Registers                       │  │
 │                   │  │  ~1 cycle, 256 KB per SM         │  │
 │                   │  │  32 warps × 32 threads × 255 reg │  │
 │                   │  └──────────────────────────────────┘  │
 │                   └────────────────────────────────────────┘
```

**The core problem:** GDDR6X is fast (288 GB/s) but far (~300 cycles). Every kernel that reads/writes model weights or activations is going to GDDR6X. Your job is to use every byte of that bandwidth efficiently.

**Ada Lovelace (sm_89) capabilities relevant to this curriculum:**

| Feature | Available? | Notes |
|---|---|---|
| `cp.async` (async gmem→smem copy) | YES | Ampere+, use in Phase 0.6 |
| Tensor cores (4th gen) | YES | FP8/FP16/BF16/FP32/TF32 |
| Hardware FP8 conversion | YES | Ada native, use in Phase 7 |
| `__shfl_xor_sync` warp shuffle | YES | All modern GPUs |
| TMA (Tensor Memory Accelerator) | **NO** | Hopper (sm_90) only |
| PDL `griddepcontrol` | **NO** | Hopper (sm_90) only — skip norm.cuh PDL sections |
| CUTLASS CuTe TMA kernels | **NO** | Hopper (sm_90) only — skip MLA kernel |

**The key metric:** Memory bandwidth utilization = `(bytes actually transferred) / (peak bandwidth × elapsed time)`.  
A naive kernel might hit 20% utilization. A well-written vectorized kernel hits 80–90%.

---

## Part 2 — Why Scalar Loads Are Wasteful

Consider this naive elementwise multiply kernel:

```cuda
__global__ void scale_naive(const __half* x, __half* out, float scale, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = __float2half(__half2float(x[idx]) * scale);
    }
}
```

On Ada (sm_89), this generates a `LD.E.16` PTX instruction — a 16-bit (2-byte) load.

The L2 cache transaction size is **128 bytes** (one cache line). When you load 2 bytes, you still pull the entire 128-byte cache line, but you only use 2 of those bytes. The other 126 bytes are loaded from GDDR6X but discarded.

**Utilization: 2 / 128 = 1.6%**. You're wasting 98.4% of your memory bandwidth.

The GPU hardware is designed to load **128 bits (16 bytes)** at once in a single instruction. If every thread loads 16 bytes, the entire cache line is consumed:
- 32 threads × 16 bytes = 512 bytes per warp → 4 full cache lines → 100% utilization.

---

## Part 3 — CUDA Vector Types for 128-bit Loads

CUDA provides built-in vector types that map to 128-bit loads:

```
Type       Bytes   Elements          PTX instruction
float4     16      4 × float32       LD.E.128
int4       16      4 × int32         LD.E.128  
uint4      16      4 × uint32        LD.E.128   ← used in merge_attn_states.cu
double2    16      2 × float64       LD.E.128
```

For fp16/bf16, you pack more elements into the same 128 bits:
```
8 × fp16 = 128 bits → load as uint4, reinterpret as __half*
8 × bf16 = 128 bits → same pattern
4 × float = 128 bits → load as float4
```

### The Raw Approach: `float4` and `reinterpret_cast`

```cuda
// Load 4 floats (128 bits) in one instruction:
float4 val = *reinterpret_cast<const float4*>(ptr + idx * 4);

// Load 8 fp16 values (128 bits) as a uint4, then access as half*:
uint4 raw = *reinterpret_cast<const uint4*>(ptr + idx * 8);
__half* vals = reinterpret_cast<__half*>(&raw);
// vals[0] through vals[7] are now accessible
```

### The FlashInfer Approach: `vec_t<T, N>`

FlashInfer wraps this in a template so you don't have to think about the reinterpret:

```cpp
// From vec_dtypes.cuh — the float x 4 specialization:
template <>
struct vec_t<float, 4> {
    float4 data;                                // 128 bits, one register group

    void load(const float* ptr) {
        data = *((float4*)ptr);                 // single LD.E.128 instruction
    }
    void store(float* ptr) const {
        *((float4*)ptr) = data;                 // single ST.E.128 instruction
    }
    float& operator[](size_t i) {
        return ((float*)(&data))[i];            // element access within register
    }
};
```

And for fp16 (8 values = 128 bits):
```cpp
// vec_t<half, 8> stores as uint4 internally:
template <>
struct vec_t<half, 8> {
    uint4 data;                                 // 128 bits = 8 × fp16

    void load(const half* ptr) {
        data = *((uint4*)ptr);                  // single LD.E.128
    }
    void store(half* ptr) const {
        *((uint4*)ptr) = data;
    }
    half& operator[](size_t i) {
        return ((half*)(&data))[i];
    }
};
```

The `cast_load` / `cast_store` methods additionally handle type conversion between fp16/bf16/fp8/float in the same operation — used heavily in RMSNorm (load fp16 input, compute in float32, store fp16 output).

---

## Part 4 — Alignment Requirement

**Critical rule:** the pointer passed to a 128-bit load MUST be 16-byte aligned.

If `ptr` is not 16-byte aligned, the `float4` load will either:
- Generate two 64-bit loads instead (slower), or
- Cause a misaligned memory access error (crash)

`torch.empty(n, dtype=torch.float16)` on CUDA always returns 256-byte aligned memory.  
`vec_size = gcd(16 / sizeof(T), hidden_dim)` in FlashInfer is how they automatically choose the right vector size given the dtype and dimension — for fp16 with `hidden_dim` divisible by 8, you get `vec_size = 8` (128-bit). For `hidden_dim % 8 != 0`, it falls back to a smaller vector size.

---

## Part 5 — The PTX Instructions

When you write `float4 val = *reinterpret_cast<const float4*>(ptr)`, nvcc generates:

```ptx
ld.global.ca.v4.f32  {%f1, %f2, %f3, %f4}, [%rd1];
```

- `ld.global` — load from global (HBM) memory
- `.ca` — cache at L1 and L2 (cache all)
- `.v4.f32` — 4 × float32 = 128 bits
- One instruction, one memory transaction

For the store:
```ptx
st.global.wb.v4.f32  [%rd2], {%f1, %f2, %f3, %f4};
```

You can verify this in Nsight Compute under "Source" view, or by compiling with `--ptx` flag:
```bash
nvcc --ptx -o kernel.ptx kernel.cu
```

---

## Part 6 — The Key PTX Math Operations

These appear throughout FlashInfer and are worth knowing now since they'll appear in every lesson.

From `REPOS/flashinfer/include/flashinfer/math.cuh`:

```cpp
// 2^x in ONE PTX instruction (vs expf which is multi-step)
// Used everywhere in online softmax instead of expf
__forceinline__ __device__ float ptx_exp2(float x) {
    float y;
    asm volatile("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}
// Usage: instead of expf(x), use ptx_exp2(x * log2e)  where log2e = 1.44269504f

// log2(x) in ONE PTX instruction
__forceinline__ __device__ float ptx_log2(float x) {
    float y;
    asm volatile("lg2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// 1/sqrt(x) in ONE PTX instruction  (used in RMSNorm)
__forceinline__ __device__ float rsqrt(float x) {
    float y;
    asm volatile("rsqrt.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

// Warp butterfly shuffle (used in warp reduce — Lesson 2)
__forceinline__ __device__ float shfl_xor_sync(float x, int lane_mask) {
    float y;
    asm volatile("shfl.sync.bfly.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(y) : "f"(x), "r"(lane_mask));
    return y;
}
```

These are the core building blocks. `ptx_exp2` + `ptx_log2` + `rsqrt` + `shfl_xor_sync` appear in almost every kernel from Lesson 2 onwards.

---

## Exercise 1.1 — Scalar Copy Baseline

Implement a scalar copy kernel and measure its bandwidth. This is your baseline.

```python
# file: ex1_1_scalar_copy.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>

// Scalar copy: one thread loads one fp16, stores one fp16
__global__ void copy_scalar(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];
    }
}

torch::Tensor scalar_copy(torch::Tensor src) {
    auto dst = torch::empty_like(src);
    int n = src.numel();
    int block = 256;
    int grid  = (n + block - 1) / block;
    copy_scalar<<<grid, block>>>(
        (__half*)src.data_ptr(),
        (__half*)dst.data_ptr(),
        n);
    return dst;
}
"""

cpp_src = "torch::Tensor scalar_copy(torch::Tensor src);"

mod = load_inline(
    name="ex1_1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["scalar_copy"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],   # Ada Lovelace (RTX 4060 Ti)
    verbose=False,
)

def bench(fn, warmup=10, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

N = 1024 * 1024 * 64  # 64M fp16 values = 128 MB
x = torch.randn(N, device="cuda", dtype=torch.float16)

ms = bench(lambda: mod.scalar_copy(x))
bytes_transferred = x.numel() * x.element_size() * 2   # read + write
bw = bytes_transferred / ms / 1e6
print(f"Scalar copy:  {ms:.3f} ms | {bw:.1f} GB/s")
# RTX 4060 Ti peak: ~288 GB/s. Scalar expect: 30–80 GB/s (10–28% utilization)
```

**Expected output:** You will see the bandwidth is well below peak.  
Note the number. You'll compare against it in Exercise 1.3.

---

## Exercise 1.2 — Understand the Assembly

Before writing the vectorized version, look at what PTX the scalar kernel generates.

```bash
# Add to extra_cuda_cflags: "--ptx"
# Or compile manually:
nvcc --ptx -arch=sm_89 -O3 ex1_kernel.cu -o ex1_kernel.ptx
cat ex1_kernel.ptx | grep "ld.global"
```

You will see something like:
```ptx
ld.global.cs.u16 %rs1, [%rd4];    // 16-bit load — cs = cache streaming (L2 only)
```

This is a **16-bit load** (`u16`). One element per instruction.  
After writing Exercise 1.3, you'll see it become `ld.global.ca.v4.u32` or `v4.f32` — a **128-bit load**.

---

## Exercise 1.3 — Vectorized Copy (float4)

Now implement the vectorized version for **float32** using `float4`.

```python
# file: ex1_3_vec_copy_f32.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Vectorized copy: each thread loads 4 floats (128 bits) at once
__global__ void copy_vec4_f32(
    const float4* __restrict__ src,
    float4*       __restrict__ dst,
    int n_vec)           // n_vec = n_elements / 4
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];   // single LD.E.128 + ST.E.128
    }
}

torch::Tensor vec_copy_f32(torch::Tensor src) {
    // src must be float32 and size divisible by 4
    auto dst = torch::empty_like(src);
    int n     = src.numel();
    int n_vec = n / 4;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;

    copy_vec4_f32<<<grid, block>>>(
        (float4*)src.data_ptr<float>(),
        (float4*)dst.data_ptr<float>(),
        n_vec);
    return dst;
}
"""

cpp_src = "torch::Tensor vec_copy_f32(torch::Tensor src);"

mod = load_inline(
    name="ex1_3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vec_copy_f32"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],   # Ada Lovelace (RTX 4060 Ti)
    verbose=False,
)

N = 1024 * 1024 * 64  # 64M floats = 256 MB
x = torch.randn(N, device="cuda", dtype=torch.float32)

def bench(fn, warmup=10, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

ms = bench(lambda: mod.vec_copy_f32(x))
bw = x.numel() * x.element_size() * 2 / ms / 1e6
print(f"Vectorized f32 copy: {ms:.3f} ms | {bw:.1f} GB/s")
# RTX 4060 Ti peak: 288 GB/s. Expect: 220–270 GB/s (75–95% of peak)

# Verify correctness
out = mod.vec_copy_f32(x)
torch.testing.assert_close(out, x)
print("Correctness: PASSED")
```

---

## Exercise 1.4 — Vectorized Copy for fp16 (the actual production case)

LLM activations are fp16/bf16. Load 8 fp16 values at once using `uint4` (128 bits).

```python
# file: ex1_4_vec_copy_f16.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// Load 8 fp16 values (128 bits) as uint4, store as uint4
// uint4 = { uint32 x, uint32 y, uint32 z, uint32 w } = 16 bytes = 128 bits
// Each uint32 holds 2 × fp16 values packed together
__global__ void copy_vec8_f16(
    const uint4* __restrict__ src,
    uint4*       __restrict__ dst,
    int n_vec)           // n_vec = n_fp16_elements / 8
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];   // LD.E.128 + ST.E.128
    }
}

// Scale each fp16 element by a float scalar (compute in fp32, store fp16)
__global__ void scale_vec8_f16(
    const uint4* __restrict__ src,
    uint4*       __restrict__ dst,
    float        scale,
    int n_vec)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // 128-bit load — one instruction
    uint4 raw = src[idx];

    // Reinterpret the 128-bit register as 8 × fp16
    __half* vals = reinterpret_cast<__half*>(&raw);

    // Process all 8 fp16 values (fully unrolled by compiler)
    uint4 out_raw;
    __half* out_vals = reinterpret_cast<__half*>(&out_raw);

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        // Upcast to float32 for compute, downcast back to fp16
        out_vals[i] = __float2half(__half2float(vals[i]) * scale);
    }

    // 128-bit store — one instruction
    dst[idx] = out_raw;
}

torch::Tensor vec_copy_f16(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;
    copy_vec8_f16<<<grid, block>>>(
        (uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), n_vec);
    return dst;
}

torch::Tensor scale_f16(torch::Tensor src, float scale) {
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    int block = 256;
    int grid  = (n_vec + block - 1) / block;
    scale_vec8_f16<<<grid, block>>>(
        (uint4*)src.data_ptr(), (uint4*)dst.data_ptr(), scale, n_vec);
    return dst;
}
"""

cpp_src = """
torch::Tensor vec_copy_f16(torch::Tensor src);
torch::Tensor scale_f16(torch::Tensor src, float scale);
"""

mod = load_inline(
    name="ex1_4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["vec_copy_f16", "scale_f16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],   # Ada Lovelace (RTX 4060 Ti)
    verbose=False,
)

def bench(fn, warmup=10, iters=200):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

N = 1024 * 1024 * 64   # 64M fp16 = 128 MB
x = torch.randn(N, device="cuda", dtype=torch.float16)

ms_copy = bench(lambda: mod.vec_copy_f16(x))
bw_copy = N * 2 * 2 / ms_copy / 1e6   # 2 bytes/elem × 2 (read+write)
print(f"Vec copy  (fp16×8): {ms_copy:.3f} ms | {bw_copy:.1f} GB/s")
# RTX 4060 Ti: target >230 GB/s (>80% of 288 GB/s peak)

ms_scale = bench(lambda: mod.scale_f16(x, 0.5))
bw_scale = N * 2 * 2 / ms_scale / 1e6
print(f"Vec scale (fp16×8): {ms_scale:.3f} ms | {bw_scale:.1f} GB/s")

# Correctness checks
out_copy = mod.vec_copy_f16(x)
torch.testing.assert_close(out_copy, x)
print("Copy correctness: PASSED")

out_scale = mod.scale_f16(x, 0.5)
torch.testing.assert_close(out_scale, x * 0.5, rtol=1e-3, atol=1e-3)
print("Scale correctness: PASSED")
```

---

## Exercise 1.5 — Generalized Vectorized Elementwise (Template)

This is the production pattern. One templated kernel handles any element count and any elementwise operation.

```python
# file: ex1_5_templated.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// Generic vectorized elementwise kernel
// VEC_SIZE: number of fp16 elements per thread (must be power of 2, max 8 for 128-bit)
// OpFunc:   applied to each element (template, zero overhead)
template<int VEC_SIZE, typename OpFunc>
__global__ void elementwise_vec(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int           n_vec,         // total_elements / VEC_SIZE
    OpFunc        op)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n_vec) return;

    // ---- 128-bit load ----
    // For VEC_SIZE=8: uint4. For VEC_SIZE=4: uint2. For VEC_SIZE=2: uint32.
    static_assert(VEC_SIZE == 8, "This example only shows VEC_SIZE=8");
    uint4 raw_in = ((const uint4*)(src))[idx];
    __half* in_vals = reinterpret_cast<__half*>(&raw_in);

    uint4 raw_out;
    __half* out_vals = reinterpret_cast<__half*>(&raw_out);

    // ---- Process VEC_SIZE elements: fully unrolled, no branch ----
    #pragma unroll
    for (int i = 0; i < VEC_SIZE; ++i) {
        out_vals[i] = op(in_vals[i]);
    }

    // ---- 128-bit store ----
    ((uint4*)(dst))[idx] = raw_out;
}

// Example ops (compiled as inline device functions, zero overhead):
struct ScaleOp {
    float scale;
    __device__ __half operator()(__half x) const {
        return __float2half(__half2float(x) * scale);
    }
};

struct ReluOp {
    __device__ __half operator()(__half x) const {
        return __float2half(fmaxf(__half2float(x), 0.f));
    }
};

struct SiluOp {
    __device__ __half operator()(__half x) const {
        float v = __half2float(x);
        return __float2half(v / (1.f + expf(-v)));   // SiLU(x) = x * sigmoid(x)
    }
};

torch::Tensor relu_f16(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    elementwise_vec<8><<<(n_vec+255)/256, 256>>>(
        (__half*)src.data_ptr(), (__half*)dst.data_ptr(), n_vec, ReluOp{});
    return dst;
}

torch::Tensor silu_f16(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n_vec = src.numel() / 8;
    elementwise_vec<8><<<(n_vec+255)/256, 256>>>(
        (__half*)src.data_ptr(), (__half*)dst.data_ptr(), n_vec, SiluOp{});
    return dst;
}
"""

cpp_src = """
torch::Tensor relu_f16(torch::Tensor src);
torch::Tensor silu_f16(torch::Tensor src);
"""

mod = load_inline(
    name="ex1_5",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["relu_f16", "silu_f16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],   # Ada Lovelace (RTX 4060 Ti)
    verbose=False,
)

x = torch.randn(1024*1024*64, device="cuda", dtype=torch.float16)

out_relu = mod.relu_f16(x)
ref_relu = torch.relu(x)
torch.testing.assert_close(out_relu, ref_relu, rtol=1e-3, atol=1e-3)
print("ReLU correctness: PASSED")

out_silu = mod.silu_f16(x)
ref_silu = torch.nn.functional.silu(x)
torch.testing.assert_close(out_silu, ref_silu, rtol=1e-2, atol=1e-2)
print("SiLU correctness: PASSED")
```

---

## Exercise 1.6 — Read the FlashInfer Source

Now that you understand what `uint4` loads are doing, read the actual production implementation.

```
File: REPOS/flashinfer/include/flashinfer/vec_dtypes.cuh

Read in this order:
1. Lines 447–466  — vec_t<float_t, vec_size> generic interface (the API)
2. Find "vec_t<float, 4>" — how float4 load/store works
3. Find "vec_t<half, 8>"  — how uint4 is used for 8×fp16
4. Lines 475–497  — cast_load_impl and cast_store_impl
                    (load fp16, process in float32, store fp16 — used in RMSNorm)
```

**Questions to answer from the source:**
1. What does `fill(float_t val)` do, and when would you need it?
2. What is `cast_load` vs `load`? When does `cast_load` do type conversion?
3. Why does `vec_t<half, 8>` store `uint4 data` internally instead of `half data[8]`?

---

## Exercise 1.7 — Profile and Compare with Nsight Compute

Run Nsight Compute on your scalar vs vectorized kernels. Compare the key metrics.

```bash
# Profile the scalar copy kernel:
ncu --kernel-name "copy_scalar" \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    python ex1_1_scalar_copy.py

# Profile the vectorized copy kernel:
ncu --kernel-name "copy_vec8_f16" \
    --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio \
    python ex1_4_vec_copy_f16.py
```

**What to look for:**

| Metric | Scalar | Vectorized | Explanation |
|---|---|---|---|
| `sm__throughput` | ~15% | ~80%+ | Overall SM utilization |
| `sectors_per_request` (L1→L2) | ~1 | ~8 | How many 32B sectors per load instruction |
| `l1tex__t_bytes` (global loads) | = tensor bytes | = tensor bytes | Should be the same |

The `sectors_per_request` metric tells you the most. Scalar loads = 1 sector per request. Vectorized loads = 8 sectors per request (one full 128-byte cache line per warp per instruction). This is the hardware evidence that you're using the memory bus efficiently.

---

## Observed Results (RTX 4060 Ti)

Running `run_all.py` produces this on the RTX 4060 Ti:

```
Kernel                                   ms     GB/s   Util
-----------------------------------------------------------------
Scalar copy (fp16, LD.E.16)           1.09   245 GB/s  85%  ✓
Vec copy    (f32,  float4 LD.128)     2.29   235 GB/s  82%  ✓
Vec copy    (fp16, uint4  LD.128)     1.14   236 GB/s  82%  ✓
Vec scale   (fp16, uint4 + fp32)      1.14   235 GB/s  82%  ✓
PyTorch baseline (cudaMemcpyAsync)    1.14   235 GB/s  82%  ✓
```

**Key insight from the results:** The scalar copy also achieves ~85% bandwidth. This is not a contradiction of the theory — it is the hardware's **memory coalescing** unit at work.

When thread `i` accesses element `i` (consecutive), the GPU memory controller sees 32 consecutive threads accessing 32 consecutive 2-byte addresses = 64 bytes contiguous, and automatically combines them into a single 128-byte cache-line transaction. So the *effective* load is already 128 bytes wide, even though each instruction only moves 16 bits.

**Why vectorized loads still matter (and why you should always use them):**

1. **Instruction throughput**: Scalar emits 8× more load instructions per byte. Each instruction occupies an issue slot in the warp scheduler. For compute-bound kernels the freed-up slots matter.
2. **Non-unit stride access**: If your kernel reads every 2nd element (e.g., GQA key head interleaving), coalescing breaks down. Vectorized loads + explicit stride computation are the correct solution.
3. **Register pressure**: Loading 8 fp16 into one `uint4` register is cleaner than 8 separate `__half` registers that the compiler must juggle.
4. **Clarity and intent**: Using `uint4` makes it explicit to the compiler that this is a 128-bit access. Scalar loads leave it to the hardware's best-effort coalescing.
5. **Compute kernels**: In RMSNorm and attention, each thread must process multiple elements anyway. Vectorizing makes the "fetch N elements, process them, store" structure explicit and correct.

The takeaway: **the numbers are almost the same here because this is a perfect-coalescing case** (unit-stride copy). In real inference kernels with irregular access patterns, vectorized loads are the difference between 30% and 80% bandwidth.

---

## Summary: What You Learned

```
1. GPU memory hierarchy (RTX 4060 Ti): GDDR6X (288 GB/s, ~300cy) → L2 (32MB) → L1/SRAM (100KB/SM) → Registers

2. Why scalar loads waste bandwidth:
   - 16-bit load pulls 128-byte cache line: 1.6% utilization
   - 128-bit load pulls 128-byte cache line: 100% utilization

3. The 128-bit load types:
   float4  = 4 × float32 = 128 bits
   uint4   = 4 × uint32  = 128 bits = 8 × fp16 (via reinterpret)
   int4    = 4 × int32   = 128 bits

4. Alignment rule: pointer must be 16-byte aligned for 128-bit loads

5. PTX math intrinsics: ptx_exp2, ptx_log2, rsqrt, shfl_xor_sync
   (from math.cuh — used in every kernel from Lesson 2 onwards)

6. The vec_t<T, N> abstraction in FlashInfer:
   - Hides the reinterpret_cast behind a clean interface
   - cast_load / cast_store handles fp16↔float32 conversion
   - Every kernel in the curriculum uses this abstraction
```

---

## Self-Test Before Moving to Lesson 2

You are ready for Lesson 2 (Warp Shuffle Reduce) when you can:

- [ ] Write a 128-bit vectorized copy for fp16 from memory without looking at your notes
- [ ] Explain why `uint4` is used for fp16 instead of `half8`
- [ ] Explain the 16-byte alignment requirement and what breaks if violated
- [ ] Look at your ncu output and identify the `sectors_per_request` metric
- [ ] Write an elementwise kernel that loads fp16, computes in fp32, stores fp16 using 128-bit loads

---

## Reference Files for This Lesson

| File | What to Read | Lines |
|---|---|---|
| `REPOS/flashinfer/include/flashinfer/vec_dtypes.cuh` | `vec_t` interface, float×4 and half×8 specializations | 447–466, then search `vec_t<float, 4>` and `vec_t<half, 8>` |
| `REPOS/flashinfer/include/flashinfer/math.cuh` | All of it — 157 lines, every function matters | 1–157 |
| `REPOS/sglang/sgl-kernel/csrc/attention/merge_attn_states.cu` | Lines 42–99 — `uint4` pack_128b_t pattern in real production code | 42–99 |
| `REPOS/flashinfer/include/flashinfer/norm.cuh` | Lines 56–66 — vectorized load in RMSNorm (preview of Lesson 4) | 56–66 |
