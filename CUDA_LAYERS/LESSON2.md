# Lesson 2 — Warp Shuffle Reduce

**Phase:** 0.4  
**Time:** 1 day  
**Prerequisite:** Lesson 1 — 128-bit Vectorized Loads  
**Next lesson:** Lesson 3 — Shared Memory Block Reduce  

---

## Why This Is Lesson 2

Every reduction kernel you will ever write — softmax, RMSNorm, attention QK dot product, layer norm, online max computation — has the same inner loop: **take values held by multiple threads, sum or max them into a single value visible to all threads**.

The fastest way to do this for values held within a single warp (32 threads) is `__shfl_xor_sync`. No shared memory. No synchronization barrier. Pure register-to-register exchange at the speed of the register file.

After this lesson, you will understand exactly why `warp.cuh` in SGLang's JIT kernel library exists, and you'll be able to write `warp_reduce_sum` and `warp_reduce_max` from memory in under 5 minutes.

---

## Part 1 — The Warp: Your Fundamental Execution Unit

Your GPU executes 32 threads together in lockstep. This group of 32 threads is called a **warp**.

```
Block (e.g., 256 threads)
├── Warp 0:  threads  0– 31  ← execute together, one instruction at a time
├── Warp 1:  threads 32– 63
├── Warp 2:  threads 64– 95
├── ...
└── Warp 7:  threads 224–255
```

Within a warp:
- All 32 threads execute the same instruction simultaneously (SIMT — Single Instruction, Multiple Threads)
- Each thread has a **lane ID** (0–31): `int lane = threadIdx.x % 32;`
- Each thread has its own private registers — `val` in thread 5 is a different register from `val` in thread 6

**The key property:** Because all 32 threads in a warp are synchronized by the hardware with no overhead, you can exchange register values between lanes with a single instruction. No shared memory. No barrier. This is the warp shuffle.

### Why warps are always 32 threads

The warp size of 32 is fundamental to NVIDIA GPU hardware — the register file, the instruction scheduler, and the memory coalescing unit are all designed around 32. On all current CUDA GPUs (Pascal through Blackwell), `warpSize == 32`. On the RTX 4060 Ti (Ada, sm_89):

```
RTX 4060 Ti — Warp Execution Facts
─────────────────────────────────────────────────────────
34 SMs × up to 48 active warps/SM = 1,632 concurrent warps
Each warp = 32 threads, each with up to 255 registers
Register file: 256 KB per SM (65,536 × 32-bit registers)
Warp scheduler: 4 schedulers per SM, 1 instruction/warp/clock
Warp switches: free (zero overhead, hardware register renaming)
```

---

## Part 2 — The Problem: Reducing Across 32 Lanes

Suppose each of the 32 threads in a warp holds one float. You want the **sum** of all 32 values, and you want **every thread to know the result**.

### The naive approach: serial (wrong for GPU)

```cuda
// THIS IS WRONG — don't do this
__shared__ float smem[32];
smem[lane] = val;
__syncwarp();    // even this is overhead
float total = 0;
for (int i = 0; i < 32; i++) total += smem[i];  // one thread does all work
```

Problems: 32× serial work. Shared memory round-trip (~30 cycles each way). Not parallelized.

### The right approach: parallel butterfly reduction

Reduce in `log2(32) = 5` steps. At each step, each thread simultaneously exchanges its value with a partner at offset `mask` (using XOR on lane IDs), and adds:

```
Step 1: XOR mask=16  →  lane 0 ↔ lane 16, lane 1 ↔ lane 17, ..., lane 15 ↔ lane 31
Step 2: XOR mask=8   →  lane 0 ↔ lane 8,  lane 1 ↔ lane 9,  ..., lane 23 ↔ lane 31
Step 3: XOR mask=4   →  lane 0 ↔ lane 4,  lane 1 ↔ lane 5,  ..., lane 27 ↔ lane 31
Step 4: XOR mask=2   →  lane 0 ↔ lane 2,  lane 1 ↔ lane 3,  ..., lane 29 ↔ lane 31
Step 5: XOR mask=1   →  lane 0 ↔ lane 1,  lane 2 ↔ lane 3,  ..., lane 30 ↔ lane 31
```

After all 5 steps, every lane holds the sum of all 32 original values.

**Visualization (8-lane example for clarity):**

```
Initial:   lane: [v0]  [v1]  [v2]  [v3]  [v4]  [v5]  [v6]  [v7]

Step 1 (XOR 4):  each lane adds value from partner lane XOR 4
   lane 0 gets v0+v4,  lane 1 gets v1+v5,  lane 2 gets v2+v6,  lane 3 gets v3+v7
   lane 4 gets v0+v4,  lane 5 gets v1+v5,  lane 6 gets v2+v6,  lane 7 gets v3+v7

Step 2 (XOR 2):  each lane adds from partner lane XOR 2
   lane 0 gets (v0+v4)+(v2+v6),  lane 1 gets (v1+v5)+(v3+v7)
   lane 2 gets (v0+v4)+(v2+v6),  lane 3 gets (v1+v5)+(v3+v7)
   lane 4 gets (v0+v4)+(v2+v6),  ...  (symmetric)

Step 3 (XOR 1):  each lane adds from partner lane XOR 1
   lane 0 gets v0+v1+v2+v3+v4+v5+v6+v7   ← full sum!
   lane 1 gets v0+v1+v2+v3+v4+v5+v6+v7   ← same!
   ... every lane holds the total sum
```

**Key property:** The XOR butterfly is self-symmetric — every lane computes the full reduction simultaneously. All 32 threads participate equally, each doing 5 additions and 5 register exchanges. Total: 5 steps instead of 32 serial steps.

---

## Part 3 — `__shfl_xor_sync`: The Warp Shuffle Instruction

```cuda
T __shfl_xor_sync(unsigned mask, T val, int laneMask, int width=32);
```

- `mask`: active lane bitmask — always use `0xffffffff` (all 32 lanes active)
- `val`: the value **this thread** contributes (read from this thread's register)
- `laneMask`: XOR with lane ID to find the partner lane
- `width`: warp size (always 32)
- **Returns:** the value from the **partner lane's** register

**The `_sync` suffix matters:** On Volta and later (all modern GPUs including Ada), you must use the `_sync` variants. They take a mask of participating threads and guarantee all listed threads have executed the shuffle before any reads the result — even if threads had diverged earlier.

### All four shuffle variants (for reference)

```cuda
// Exchange with XOR partner  (butterfly reduce — what we use)
__shfl_xor_sync(0xffffffff, val, laneMask);

// Exchange with fixed lane   (broadcast: lane 0 → everyone)
__shfl_sync(0xffffffff, val, srcLane);

// Exchange with lane+offset  (shift down: prefix scan)
__shfl_down_sync(0xffffffff, val, offset);

// Exchange with lane-offset  (shift up: prefix scan)
__shfl_up_sync(0xffffffff, val, offset);
```

For reductions, `__shfl_xor_sync` is what you always use. The others appear in scans (prefix sum), broadcast patterns, and RoPE rotary embedding.

---

## Part 4 — Writing `warp_reduce_sum` and `warp_reduce_max`

### warp_reduce_sum

```cuda
__forceinline__ __device__ float warp_reduce_sum(float val) {
    // 5 rounds of butterfly XOR: 16, 8, 4, 2, 1
    // Each round: fetch value from XOR partner, add to own value
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;  // all 32 lanes return the same sum
}
```

This is 5 instructions. No shared memory. No sync. It is hard to write anything faster.

### warp_reduce_max

```cuda
__forceinline__ __device__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;  // all 32 lanes return the same max
}
```

**Why start at 16, not 1?** Starting at 16 ensures that after step 1, each lane already holds a partial sum of 2 values (itself + partner 16 away). After step 2, it holds a partial of 4, then 8, then 16, then 32. Starting at 1 would also work (the butterfly is commutative) but XOR from highest to lowest is the canonical form and the one compilers expect.

---

## Part 5 — The PTX Behind the Shuffle

When nvcc compiles `__shfl_xor_sync(0xffffffff, val, 16)`, it generates:

```ptx
shfl.sync.bfly.b32  %r_dst, %r_src, 16, 0x1f, 0xffffffff;
```

- `shfl.sync` — synchronized shuffle (Volta+ syntax)
- `.bfly` — butterfly mode (XOR with laneMask)
- `.b32` — 32-bit value being exchanged
- `%r_src` — source register (this thread's value going out)
- `16` — the XOR mask applied to the lane ID
- `0x1f` — clamp mask (31 = warp size − 1, ensures wrap within warp)
- `0xffffffff` — participation mask (all 32 lanes active)

One instruction. One clock cycle (register file latency). The hardware interconnect between lanes within a warp is a dedicated cross-bar — no memory bus involved.

**Compile and inspect:**
```bash
nvcc --ptx -arch=sm_89 -O3 ex2_2_warp_reduce.cu -o ex2_2.ptx
grep "shfl.sync" ex2_2.ptx
# You should see 5 lines: one per XOR step (16, 8, 4, 2, 1)
```

---

## Part 6 — The SGLang Production Implementation

The actual production code in SGLang (`warp.cuh`) uses the same butterfly pattern, templated for any numeric type and any participating lane count:

```cpp
// From: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/warp.cuh

template <uint32_t kNumThreads = kWarpThreads, typename T>
SGL_DEVICE T reduce_sum(T value, uint32_t active_mask = kFullMask) {
    static_assert(kNumThreads >= 1 && kNumThreads <= kWarpThreads);
    static_assert(std::has_single_bit(kNumThreads), "must be pow of 2");
    #pragma unroll
    for (int mask = kNumThreads / 2; mask > 0; mask >>= 1)
        value = value + __shfl_xor_sync(active_mask, value, mask, 32);
    return value;
}

template <typename T>
SGL_DEVICE T reduce_max(T value, uint32_t active_mask = kFullMask) {
    #pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        value = math::max(value, __shfl_xor_sync(active_mask, value, mask, 32));
    return value;
}
```

Key differences from the raw form:
- `kNumThreads` template param: works for sub-warp reductions (e.g., 16 threads only) by starting the loop at `kNumThreads/2` instead of hardcoded 16
- `active_mask` param: allows partial warp participation (important when some threads have exited via `if` branches)
- `math::max`: type-generic max that dispatches to `fmaxf` for float, `__hmax` for fp16, etc.
- `#pragma unroll`: compiler fully unrolls the 5-iteration loop to 5 independent instructions

### Where it is used in production

**In `norm.cuh` (RMSNorm):**
```cpp
// From: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh
float sum_of_squares = 0.0f;
#pragma unroll
for (auto i = 0u; i < N; ++i) {
    const auto fp32_input = cast<fp32x2_t>(input[i]);   // unpack fp16×2 → float2
    sum_of_squares += fp32_input.x * fp32_input.x;
    sum_of_squares += fp32_input.y * fp32_input.y;
}
sum_of_squares = warp::reduce_sum(sum_of_squares);       // ← your Lesson 2 primitive
// ... then rsqrt, then normalize
```

Each thread holds the sum-of-squares for its own subset of the hidden dimension. The warp shuffle collects all partial sums into a single total. Then every thread independently computes `rsqrt(total / d + eps)` — the single norm factor.

**In `cta.cuh` (cross-warp, preview of Lesson 3):**
```cpp
// From: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/cta.cuh
template <typename T>
SGL_DEVICE void reduce_max(T value, float* smem, float min_value = 0.0f) {
    const uint32_t warp_id = threadIdx.x / kWarpThreads;
    smem[warp_id] = warp::reduce_max(value);   // ← step 1: intra-warp (Lesson 2)
    __syncthreads();
    if (warp_id == 0) {
        const auto tx = threadIdx.x;
        const auto local_value = tx * kWarpThreads < blockDim.x ? smem[tx] : min_value;
        const auto max_value = warp::reduce_max(local_value);  // ← step 2: across warps
        smem[0] = max_value;
    }
    // caller must __syncthreads() before reading smem[0]
}
```

This two-level pattern (warp shuffle → smem write → second warp shuffle) is the complete block reduce. You'll implement this in Lesson 3.

---

## Exercise 2.1 — Scalar Reduce Baseline

Before shuffles, write a naive per-block reduce using shared memory so you have a correctness baseline and a performance floor.

```python
# file: ex2_1_scalar_reduce.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Naive block reduce: each thread contributes one float, result in out[blockIdx.x]
__global__ void block_reduce_naive(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int n_per_block)
{
    extern __shared__ float smem[];
    int tid = threadIdx.x;
    int gid = blockIdx.x * n_per_block + tid;

    smem[tid] = (tid < n_per_block) ? src[gid] : 0.f;
    __syncthreads();

    // Serial reduce in smem — every thread reads all values (bad)
    if (tid == 0) {
        float total = 0.f;
        for (int i = 0; i < n_per_block; i++) total += smem[i];
        out[blockIdx.x] = total;
    }
}

torch::Tensor naive_reduce(torch::Tensor src) {
    int n = src.numel();           // must be divisible by 32
    int blocks = n / 32;
    auto out = torch::zeros({blocks}, src.options());
    block_reduce_naive<<<blocks, 32, 32*sizeof(float)>>>(
        src.data_ptr<float>(), out.data_ptr<float>(), 32);
    return out;
}
"""

cpp_src = "torch::Tensor naive_reduce(torch::Tensor src);"
mod = load_inline(name="ex2_1", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["naive_reduce"],
                  extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

N = 1024 * 32
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.naive_reduce(x)

# Reference: reshape to (1024, 32) and sum each row
ref = x.view(-1, 32).sum(dim=1)
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("Naive reduce: PASSED")
print(f"Sample: out[0]={out[0].item():.4f}  ref[0]={ref[0].item():.4f}")
```

---

## Exercise 2.2 — `warp_reduce_sum` with `__shfl_xor_sync`

Implement the butterfly warp reduce. Each block has exactly 32 threads (one warp). Each thread computes the total sum and writes it to `out[blockIdx.x]`.

```python
# file: ex2_2_warp_reduce_sum.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Butterfly warp-level sum reduction — 5 rounds, no shared memory
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// One block = one warp (32 threads). Each block reduces 32 floats.
__global__ void reduce_warp(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int n_per_warp)
{
    int lane = threadIdx.x;                         // 0–31
    int gid  = blockIdx.x * n_per_warp + lane;

    float val = src[gid];
    float total = warp_reduce_sum(val);             // every lane holds the sum

    if (lane == 0) out[blockIdx.x] = total;         // only need one write
}

torch::Tensor warp_reduce(torch::Tensor src) {
    int n = src.numel();       // must be divisible by 32
    int blocks = n / 32;
    auto out = torch::zeros({blocks}, src.options());
    reduce_warp<<<blocks, 32>>>(src.data_ptr<float>(), out.data_ptr<float>(), 32);
    return out;
}
"""

cpp_src = "torch::Tensor warp_reduce(torch::Tensor src);"
mod = load_inline(name="ex2_2", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["warp_reduce"],
                  extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

N = 1024 * 32
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.warp_reduce(x)
ref = x.view(-1, 32).sum(dim=1)
torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
print("warp_reduce_sum: PASSED")

# Timing
def bench(fn, warmup=20, iters=500):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters): fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters

ms = bench(lambda: mod.warp_reduce(x))
print(f"warp_reduce: {ms*1000:.1f} us  ({N*4*2/ms/1e6:.1f} GB/s)")
# This is compute-bound not memory-bound — the timing will be tiny
```

**Question to answer:** Every lane calls `warp_reduce_sum` and every lane gets the full sum back. But we only write `out[blockIdx.x]` from lane 0. Why is it correct for lane 0 to write the full sum even though lane 0's initial value was just one of the 32 inputs?

---

## Exercise 2.3 — `warp_reduce_max`

```python
# file: ex2_3_warp_reduce_max.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;
}

__global__ void reduce_max_warp(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    int lane = threadIdx.x;
    int gid  = blockIdx.x * 32 + lane;
    float val = src[gid];
    float result = warp_reduce_max(val);
    if (lane == 0) out[blockIdx.x] = result;
}

torch::Tensor warp_reduce_max(torch::Tensor src) {
    int blocks = src.numel() / 32;
    auto out = torch::zeros({blocks}, src.options());
    reduce_max_warp<<<blocks, 32>>>(src.data_ptr<float>(), out.data_ptr<float>());
    return out;
}
"""

cpp_src = "torch::Tensor warp_reduce_max(torch::Tensor src);"
mod = load_inline(name="ex2_3", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["warp_reduce_max"],
                  extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

N = 1024 * 32
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.warp_reduce_max(x)
ref = x.view(-1, 32).max(dim=1).values
torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
print("warp_reduce_max: PASSED")
```

---

## Exercise 2.4 — Warp Reduce with Vectorized fp16 Loads

Combine Lesson 1 (128-bit loads) with Lesson 2 (warp reduce). Each thread loads 8 fp16 values as a `uint4`, accumulates their sum-of-squares in float32, then warp-reduces across all 32 threads.

This is the exact inner loop of RMSNorm.

```python
# file: ex2_4_vec_load_warp_reduce.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// Each warp processes 32*8 = 256 fp16 values.
// Each thread loads 8 fp16 (128-bit), computes partial sum-of-squares,
// then warp reduces. One row of a hidden_dim=256 tensor.
__global__ void sum_of_squares_f16(
    const uint4*  __restrict__ src,   // packed fp16: each uint4 = 8 fp16
    float*        __restrict__ out,   // one float per block (one row per block)
    int n_vec)                        // n_vec = hidden_dim / 8 = 32 for dim=256
{
    int lane  = threadIdx.x;          // 0–31 (one warp per block)
    int row   = blockIdx.x;

    // 128-bit load: 8 fp16 values in one instruction (Lesson 1)
    uint4 raw = src[row * n_vec + lane];
    __half* vals = reinterpret_cast<__half*>(&raw);

    // Accumulate partial sum-of-squares in float32
    float local_sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        float v = __half2float(vals[i]);
        local_sq += v * v;
    }

    // Warp reduce across all 32 lanes → total sum-of-squares for this row
    float total_sq = warp_reduce_sum(local_sq);

    if (lane == 0) out[row] = total_sq;
}

torch::Tensor row_sum_of_squares(torch::Tensor x) {
    // x shape: (batch, hidden_dim), hidden_dim must be divisible by 256
    int batch  = x.size(0);
    int hidden = x.size(1);
    int n_vec  = hidden / 8;   // uint4 per thread (hidden / 8 elements per thread)
    // Constraint: n_vec must == 32 (one warp) for this simple kernel
    TORCH_CHECK(n_vec == 32, "hidden_dim must be 256 for this exercise");

    auto out = torch::zeros({batch}, torch::dtype(torch::kFloat32).device(x.device()));
    sum_of_squares_f16<<<batch, 32>>>(
        (const uint4*)x.data_ptr(), out.data_ptr<float>(), n_vec);
    return out;
}
"""

cpp_src = "torch::Tensor row_sum_of_squares(torch::Tensor x);"
mod = load_inline(name="ex2_4", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["row_sum_of_squares"],
                  extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

batch, hidden = 1024, 256
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)

out = mod.row_sum_of_squares(x)
ref = (x.float() ** 2).sum(dim=1)
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
print("sum_of_squares_f16: PASSED")
print(f"out[0]={out[0].item():.2f}  ref[0]={ref[0].item():.2f}")
```

**What you just built:** The pass-1 inner loop of RMSNorm. The only remaining steps are `rsqrt(total_sq / hidden + eps)` and a second pass to scale every element.

---

## Exercise 2.5 — Multi-Warp Block Reduce (Preview of Lesson 3)

A single warp handles 32 threads. But production kernels use 256 threads (8 warps). Here is the two-level pattern: warp shuffle first, then shared memory across warps.

```python
# file: ex2_5_block_reduce.py
import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// Block reduce: up to 1024 threads, arbitrary number of warps
// Pattern: each warp reduces internally via shuffle, writes result to smem,
//          then first warp reduces smem.
__device__ float block_reduce_sum(float val) {
    __shared__ float smem[32];  // one slot per warp (max 32 warps per block)

    int lane    = threadIdx.x % 32;
    int warp_id = threadIdx.x / 32;
    int n_warps = blockDim.x  / 32;

    // Step 1: each warp reduces its 32 lanes internally
    float warp_sum = warp_reduce_sum(val);

    // Step 2: lane 0 of each warp writes its partial sum to shared memory
    if (lane == 0) smem[warp_id] = warp_sum;
    __syncthreads();

    // Step 3: first warp loads all partial sums from smem and reduces them
    float block_sum = 0.f;
    if (warp_id == 0) {
        // Load: use lane index as warp index (n_warps ≤ 32 always)
        float v = (lane < n_warps) ? smem[lane] : 0.f;
        block_sum = warp_reduce_sum(v);   // second warp shuffle on partial sums
    }
    return block_sum;  // valid in every thread of warp 0; broadcast not needed
}

__global__ void block_reduce_kernel(
    const float* __restrict__ src,
    float*       __restrict__ out,
    int n_per_block)
{
    int gid = blockIdx.x * n_per_block + threadIdx.x;
    float val = (threadIdx.x < n_per_block) ? src[gid] : 0.f;
    float total = block_reduce_sum(val);
    if (threadIdx.x == 0) out[blockIdx.x] = total;
}

torch::Tensor block_reduce(torch::Tensor src, int threads) {
    int n = src.numel();
    int blocks = n / threads;
    auto out = torch::zeros({blocks}, src.options());
    block_reduce_kernel<<<blocks, threads>>>(
        src.data_ptr<float>(), out.data_ptr<float>(), threads);
    return out;
}
"""

cpp_src = "torch::Tensor block_reduce(torch::Tensor src, int threads);"
mod = load_inline(name="ex2_5", cpp_sources=cpp_src, cuda_sources=cuda_src,
                  functions=["block_reduce"],
                  extra_cuda_cflags=["-O3", "-arch=sm_89"], verbose=False)

for threads in [32, 64, 128, 256]:
    N = 4096 * threads
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    out = mod.block_reduce(x, threads)
    ref = x.view(-1, threads).sum(dim=1)
    torch.testing.assert_close(out, ref, rtol=1e-4, atol=1e-4)
    print(f"block_reduce({threads} threads): PASSED")
```

**The two sync boundaries:**
1. `__syncthreads()` after smem writes — ensures all warps' partial sums are visible before warp 0 reads them
2. No second `__syncthreads()` needed because only warp 0 reads smem, and all its threads already completed the `__syncthreads()`

**Why smem[32] is always enough:** CUDA guarantees at most `blockDim.x / 32` warps per block, and `blockDim.x ≤ 1024`, so there are at most `1024 / 32 = 32` warps. One slot per warp — exactly 32 slots.

---

## Exercise 2.6 — Read the SGLang Source

Now that you've written these primitives, read the production implementations.

```
File: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/warp.cuh

Read in this order:
1. Lines 1–32   — reduce_sum: note kNumThreads template param vs hardcoded 32
                  Why does `for (int mask = kNumThreads/2; mask > 0; mask >>= 1)`
                  generate the same 5-step butterfly?
2. Lines 34–54  — reduce_max: note math::max dispatch
3. Think: what does kNumThreads=16 do? Which 16 lanes participate?

File: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh

Read:
4. Lines 60–76  — apply_norm_impl: find the warp::reduce_sum call
                  What is sum_of_squares after the warp reduce?
                  What does each thread contribute (how many fp16 elements)?
5. Lines 78–94  — CTA path: when kUseCTA is true, smem is used for cross-warp
                  reduce. This is Exercise 2.5's pattern in production C++.

File: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/cta.cuh

Read:
6. Lines 1–40   — cta::reduce_max: the two-level pattern with warp:: + smem
                  Note: no trailing __syncthreads() — why? (see the comment)
```

**Questions to answer from the source:**
1. In `warp::reduce_sum`, the loop starts at `kNumThreads/2`. For `kNumThreads=32`, this is 16. For `kNumThreads=8`, this is 4. Explain why starting at 4 instead of 16 is correct for an 8-thread reduction.
2. In `norm.cuh` `apply_norm_impl`, `N` is the number of `PackedFloat` elements per thread. If each `PackedFloat` is `fp16x2_t` (two fp16 values), and N=4, how many fp16 values does each thread process? How many total fp16 per warp?
3. In `cta.cuh`, why does `reduce_max` NOT have a trailing `__syncthreads()`, and what must the caller do before reading `smem[0]`?

---

## Exercise 2.7 — Profile with Nsight Compute

Compare naive shared-memory reduce vs warp shuffle reduce.

```bash
# Profile naive block reduce (Exercise 2.1):
ncu --kernel-name "block_reduce_naive" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum,\
smsp__inst_executed.sum \
    python ex2_1_scalar_reduce.py

# Profile warp shuffle reduce (Exercise 2.2):
ncu --kernel-name "reduce_warp" \
    --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum,\
smsp__inst_executed.sum \
    python ex2_2_warp_reduce_sum.py
```

**What to look for:**

| Metric | Naive (smem) | Warp Shuffle | Explanation |
|---|---|---|---|
| `smem sectors ld` | > 0 | **0** | Shuffle uses zero shared memory |
| `smem sectors st` | > 0 | **0** | No writes to smem either |
| `inst_executed` | High (serial loop) | Low (5 shuffles) | 5 instructions vs 32 serial adds |
| `sm__throughput` | Low | Higher | More warps fit → better occupancy |

The shared memory sectors metric should be **zero** for the warp shuffle kernel — that is the hardware proof that `__shfl_xor_sync` does not touch the shared memory bank.

---

## Observed Results (RTX 4060 Ti)

These reductions are too fast to measure reliably with CPU timing. Use kernel-level timing or Nsight Compute. The key observations are:

```
Kernel                           Instruction count (for 32 elements)
─────────────────────────────────────────────────────────────────────
naive smem reduce (serial)       ~32 adds + 32 smem loads + 1 smem write
warp_reduce_sum  (butterfly)     5 shuffle-add instructions
warp_reduce_max  (butterfly)     5 shuffle-fmax instructions
block_reduce_sum (2-level)       5 shuffle + 1 smem write + 5 shuffle = 11 total

Shared memory transactions:
  naive:         64 (32 writes + 32 reads of smem)
  warp shuffle:  0  (pure register-to-register)
  block reduce:  2  (1 write per warp + 1 read per warp = 2 smem transactions)
```

The warp shuffle reduce has **zero** shared memory pressure. For 256-thread blocks running RMSNorm on 4096 tokens, this translates to ~32× fewer shared memory transactions vs a naive reduce — freeing up the shared memory bandwidth for the actual tensor data.

---

## Summary: What You Learned

```
1. Warp = 32 threads, always synchronized, zero-cost register exchange between lanes

2. __shfl_xor_sync(mask, val, laneMask):
   - Exchange values between lanes with XOR-partner addresses
   - One PTX instruction: shfl.sync.bfly.b32
   - Zero shared memory, zero synchronization overhead

3. Butterfly reduction (5 steps for 32 lanes):
   XOR 16 → XOR 8 → XOR 4 → XOR 2 → XOR 1
   After 5 steps: every lane holds the complete reduction result

4. warp_reduce_sum(val):   5 × (val += shfl_xor(val, mask))
   warp_reduce_max(val):   5 × (val = fmaxf(val, shfl_xor(val, mask)))

5. Two-level block reduce (for multi-warp blocks):
   Step 1: warp shuffle → partial result per warp      (no smem)
   Step 2: lane 0 writes to smem, __syncthreads()
   Step 3: first warp shuffles smem values             (1 smem round-trip)

6. SGLang production pattern (warp.cuh):
   - Templated on kNumThreads for sub-warp reductions
   - Same butterfly loop: for (mask = kN/2; mask > 0; mask >>= 1)
   - Used inside norm.cuh RMSNorm pass-1 (sum_of_squares)
   - Used inside cta.cuh for cross-warp max (softmax, attention)

7. Where warp_reduce appears in the inference stack:
   - RMSNorm:  reduce sum_of_squares → rsqrt → normalize
   - Softmax:  reduce max → shift → reduce exp_sum → divide
   - Attention: reduce max and exp_sum per query row (online softmax)
```

---

## Self-Test Before Moving to Lesson 3

You are ready for Lesson 3 (Shared Memory Block Reduce) when you can:

- [ ] Write `warp_reduce_sum` from memory without notes (5 lines)
- [ ] Write `warp_reduce_max` from memory without notes (5 lines)
- [ ] Explain why the XOR offsets are 16, 8, 4, 2, 1 and not 1, 2, 4, 8, 16
- [ ] Explain what `__shfl_xor_sync(0xffffffff, val, 8)` returns to lane 5 (answer: lane 5 XOR 8 = lane 13's value)
- [ ] Explain why `warp_reduce_sum` with `kNumThreads=16` starts at mask=8
- [ ] Write the two-level block reduce without looking at notes
- [ ] Look at `norm.cuh apply_norm_impl` and point to exactly where `warp_reduce_sum` is called and what it's summing

---

## Reference Files for This Lesson

| File | What to Read | Lines |
|---|---|---|
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/warp.cuh` | `reduce_sum` and `reduce_max` — the production butterfly implementation | 1–54 |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/cta.cuh` | `cta::reduce_max` — the two-level warp+smem block reduce | 1–40 |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh` | `apply_norm_impl` lines 60–94 — warp reduce inside RMSNorm | 60–94 |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/math.cuh` | `device::math::max`, `rsqrt` — type-generic dispatch | 1–71 |
| `REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/utils.cuh` | `kWarpThreads`, `kFullMask` constants | 114–117 |
