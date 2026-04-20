# TORCH CUDA BASICS
## How Python talks to CUDA kernels through PyTorch

---

## The One-Sentence Version

PyTorch tensors on CUDA are just pointers to GPU memory. You pass that pointer to your CUDA kernel, do work on the GPU, and return a new tensor — Python never touches the raw data.

---

## Part 1 — The Big Picture: What Happens When You Call a Kernel

```
Python                         C++/CUDA
──────                         ────────
tensor = torch.randn(...)
                                   ↓  (PyTorch manages the allocation)
                               [GPU memory block]

mod.my_kernel(tensor)
    │
    │  1. Python calls C++ binding (via pybind11)
    │  2. C++ receives at::Tensor object
    │  3. C++ calls .data_ptr() → raw void* pointer to GPU memory
    │  4. C++ launches CUDA kernel with that pointer
    │  5. Kernel runs on GPU (Python is blocked, waiting)
    │  6. C++ wraps result pointer into new at::Tensor
    │  7. Returns new at::Tensor back to Python
    ▼
result = at::Tensor   →   Python sees it as a regular torch.Tensor
```

The GPU memory is never copied to CPU. Python just holds a handle (the tensor object) that knows where the data lives on the GPU.

---

## Part 2 — The Three Ways to Write CUDA for PyTorch

### Method 1: `load_inline` — fastest iteration (used in all exercises here)

Write CUDA code as a Python string, compile on first run, cache forever.

```python
import torch
from torch.utils.cpp_extension import load_inline

# The CUDA kernel + a C++ wrapper that PyTorch can call
cuda_src = r"""
#include <cuda_fp16.h>

__global__ void my_kernel(const float* src, float* dst, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = src[i] * 2.0f;
}

// This is the function Python will call.
// It takes a torch::Tensor and returns a torch::Tensor.
torch::Tensor double_it(torch::Tensor src) {
    auto dst = torch::empty_like(src);          // allocate output on GPU
    int n    = src.numel();
    my_kernel<<<(n+255)/256, 256>>>(
        src.data_ptr<float>(),                  // raw GPU pointer (float*)
        dst.data_ptr<float>(),
        n);
    return dst;
}
"""

# The C++ declaration PyTorch needs to expose the function to Python
cpp_src = "torch::Tensor double_it(torch::Tensor src);"

mod = load_inline(
    name="my_module",           # cache key — use a unique name per kernel
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["double_it"],    # which C++ functions to expose to Python
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,              # True prints compilation output
)

x = torch.tensor([1.0, 2.0, 3.0], device="cuda")
y = mod.double_it(x)
print(y)   # tensor([2., 4., 6.], device='cuda:0')
```

### Method 2: `CUDAExtension` — proper installable package

Use when you want to `pip install -e .` the kernel as a real Python package.
This is how `sgl-kernel` is built.

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_kernels",
    ext_modules=[
        CUDAExtension(
            name="my_kernels",
            sources=["my_kernel.cu", "binding.cpp"],
            extra_compile_args={
                "cxx":  ["-O3"],
                "nvcc": ["-O3", "-arch=sm_89", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```bash
pip install -e .        # builds and installs
python -c "import my_kernels; print(my_kernels.double_it(x))"
```

### Method 3: Triton — Python-native GPU kernels (no C++ at all)

Triton lets you write GPU kernels in Python-like syntax. SGLang uses this for
many ops (decode attention, prefill attention, RoPE). Covered later in the curriculum.

```python
import triton
import triton.language as tl

@triton.jit
def double_kernel(src_ptr, dst_ptr, n, BLOCK: tl.constexpr):
    pid  = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offs < n
    x    = tl.load(src_ptr + offs, mask=mask)
    tl.store(dst_ptr + offs, x * 2.0, mask=mask)
```

For this curriculum we use `load_inline` (Method 1) — fastest to iterate, no build step.

---

## Part 3 — `torch::Tensor` in C++: What You Can Do With It

When your C++ wrapper receives a `torch::Tensor`, here are all the methods you'll use:

```cpp
torch::Tensor t = /* received from Python */;

// ── Shape and type ──────────────────────────────────────────────────
t.numel()              // total number of elements (int64)
t.size(0)              // size of dimension 0
t.size(1)              // size of dimension 1
t.dim()                // number of dimensions
t.dtype()              // torch::kFloat32, torch::kFloat16, etc.
t.device()             // cuda or cpu
t.is_cuda()            // true if on GPU
t.is_contiguous()      // true if memory layout is row-major contiguous

// ── Raw pointer ─────────────────────────────────────────────────────
t.data_ptr<float>()    // float*  — use when dtype is float32
t.data_ptr<at::Half>() // __half* — use when dtype is float16
t.data_ptr()           // void*   — untyped, needs cast
t.data_ptr<int>()      // int*    — use when dtype is int32

// ── Allocating output tensors ───────────────────────────────────────
torch::empty_like(t)               // same shape/dtype/device, uninitialized
torch::zeros_like(t)               // same shape/dtype/device, zeroed
torch::ones_like(t)                // same shape/dtype/device, filled with 1
torch::empty({4, 128}, t.options()) // specific shape, same dtype/device as t

torch::empty({n}, torch::TensorOptions()
    .dtype(torch::kFloat16)
    .device(torch::kCUDA, 0))     // explicit dtype and device

// ── Checks (throw exceptions if failed) ─────────────────────────────
TORCH_CHECK(t.is_cuda(),    "Expected CUDA tensor");
TORCH_CHECK(t.is_contiguous(), "Expected contiguous tensor");
TORCH_CHECK(t.dtype() == torch::kFloat16, "Expected fp16");
TORCH_CHECK(t.numel() % 8 == 0, "numel must be divisible by 8");
TORCH_CHECK(t.dim() == 2,  "Expected 2D tensor");
```

---

## Part 4 — `data_ptr<T>()`: Getting the Raw GPU Pointer

`data_ptr<T>()` returns a raw C++ pointer to the first byte of the tensor's data on GPU.

```cpp
// float32 tensor:
float* ptr = tensor.data_ptr<float>();

// float16 tensor:
//   at::Half is PyTorch's name for fp16 (same bits as __half)
at::Half* ptr = tensor.data_ptr<at::Half>();
//   cast to __half* for CUDA intrinsics:
__half* ptr = (__half*)tensor.data_ptr<at::Half>();
//   or just use void*:
__half* ptr = (__half*)tensor.data_ptr();

// int32 tensor:
int* ptr = tensor.data_ptr<int>();
```

**The pointer points directly into GPU memory.** You can pass it straight to `<<<>>>` kernel launches. No copy, no transfer — the kernel runs on the data in place.

```cpp
torch::Tensor my_op(torch::Tensor src) {
    auto dst = torch::empty_like(src);

    // get raw pointers
    const __half* src_ptr = (__half*)src.data_ptr();
    __half*       dst_ptr = (__half*)dst.data_ptr();
    int n = src.numel();

    // launch kernel
    int block = 256;
    int grid  = (n + block - 1) / block;
    my_kernel<<<grid, block>>>(src_ptr, dst_ptr, n);

    // return the output tensor (Python gets this as a torch.Tensor)
    return dst;
}
```

---

## Part 5 — Allocating Output Tensors

Never allocate inside a CUDA kernel. Always pre-allocate in C++ and pass the pointer in.

```cpp
// Pattern 1: same shape and dtype as input (most common)
auto out = torch::empty_like(src);

// Pattern 2: same device/dtype, different shape
auto out = torch::empty({batch, seq_len, heads, head_dim}, src.options());
// .options() returns dtype + device + layout of src

// Pattern 3: explicit dtype (e.g., output is float32 even if input is fp16)
auto out = torch::empty(src.sizes(),
    src.options().dtype(torch::kFloat32));

// Pattern 4: specific device, specific dtype
auto out = torch::empty({n},
    torch::TensorOptions()
        .dtype(torch::kFloat16)
        .device(torch::kCUDA, 0));    // GPU 0

// Pattern 5: return multiple tensors
// Python receives: out, lse = mod.my_kernel(q, k, v)
std::tuple<torch::Tensor, torch::Tensor> my_kernel(torch::Tensor q, ...) {
    auto out = torch::empty_like(q);
    auto lse = torch::empty({batch, heads, seq_q}, q.options().dtype(torch::kFloat32));
    // ... launch kernel ...
    return {out, lse};
}
```

---

## Part 6 — CUDA Streams and Synchronization

### The default stream

By default, every CUDA operation goes into the **default stream**. Operations in the same stream execute in order, serialized. This means:

```python
y = mod.my_kernel(x)    # runs on GPU
z = y + 1               # PyTorch op — also on GPU, same default stream, runs after
print(z)                # Python reads z — this forces synchronization
```

You don't need to manually synchronize unless you're **timing** or doing **host-side reads**.

### When you MUST synchronize

```python
# WRONG — timing without sync:
start = time.time()
y = mod.my_kernel(x)
end = time.time()
print(end - start)   # measures kernel LAUNCH time, not kernel EXECUTION time

# CORRECT — use CUDA events:
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
y = mod.my_kernel(x)
e.record()
torch.cuda.synchronize()   # wait for GPU to finish
print(s.elapsed_time(e), "ms")   # now measures actual GPU time
```

### Reading results back to CPU

```python
y = mod.my_kernel(x)      # on GPU
y_cpu = y.cpu()           # .cpu() triggers synchronization + D2H transfer
y_np  = y.numpy()         # only works after .cpu()
val   = y[0].item()       # .item() forces sync + reads one element
```

---

## Part 7 — The Complete Template for a CUDA Kernel in load_inline

This is the exact pattern used in every exercise in this folder:

```python
import torch
from torch.utils.cpp_extension import load_inline

# ── CUDA source (kernel + C++ wrapper) ──────────────────────────────
cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ----- GPU kernel -----
__global__ void my_kernel_impl(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    float         param,
    int           n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = __half2float(src[i]);
    v = v * param;
    dst[i] = __float2half(v);
}

// ----- C++ wrapper (Python calls this) -----
torch::Tensor my_kernel(torch::Tensor src, float param) {
    // 1. validate inputs
    TORCH_CHECK(src.is_cuda(),              "Expected CUDA tensor");
    TORCH_CHECK(src.dtype() == torch::kFloat16, "Expected fp16");
    TORCH_CHECK(src.is_contiguous(),        "Expected contiguous tensor");

    // 2. allocate output
    auto dst = torch::empty_like(src);

    // 3. get raw pointers
    const __half* src_ptr = (__half*)src.data_ptr();
    __half*       dst_ptr = (__half*)dst.data_ptr();
    int n = src.numel();

    // 4. compute grid size and launch
    int block = 256;
    int grid  = (n + block - 1) / block;
    my_kernel_impl<<<grid, block>>>(src_ptr, dst_ptr, param, n);

    // 5. return output tensor
    return dst;
}
"""

# ── C++ declaration (tells pybind11 what to expose) ─────────────────
cpp_src = "torch::Tensor my_kernel(torch::Tensor src, float param);"

# ── Compile and load ─────────────────────────────────────────────────
mod = load_inline(
    name="my_kernel",           # unique cache key
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["my_kernel"],
    extra_cuda_cflags=[
        "-O3",
        "-arch=sm_89",          # Ada Lovelace (RTX 4060 Ti)
        # "--use_fast_math",    # add for kernels that don't need exact math
    ],
    verbose=False,              # set True to see compilation output
)

# ── Use it from Python ───────────────────────────────────────────────
x = torch.randn(1024 * 1024, device="cuda", dtype=torch.float16)
y = mod.my_kernel(x, 0.5)

# correctness check
torch.testing.assert_close(y, x * 0.5, rtol=1e-3, atol=1e-3)
print("Correctness: PASSED")
```

---

## Part 8 — Passing Different Argument Types

Python → C++ type mapping through pybind11:

```
Python type      C++ / pybind11 type      Notes
────────────────────────────────────────────────────────────────────────
torch.Tensor     torch::Tensor            always pass by value
int              int64_t  or  int         Python ints are 64-bit
float            double  or  float        Python floats are 64-bit
bool             bool
list[int]        std::vector<int64_t>
None             std::optional<T>         use c10::optional<torch::Tensor>
```

```cpp
// Example: kernel taking multiple tensors + scalar params
torch::Tensor attention_decode(
    torch::Tensor q,                // [batch, heads, head_dim]
    torch::Tensor k,                // [batch, kv_len, kv_heads, head_dim]
    torch::Tensor v,                // [batch, kv_len, kv_heads, head_dim]
    float sm_scale,                 // 1/sqrt(head_dim)
    int   num_kv_heads)             // for GQA
{
    // ... body ...
}

// C++ declaration:
"torch::Tensor attention_decode(torch::Tensor, torch::Tensor, torch::Tensor, float, int64_t);"
```

```python
# Python call:
out = mod.attention_decode(q, k, v, 1.0 / math.sqrt(128), 8)
```

---

## Part 9 — Checking Tensor Properties Before Launching

Always validate before launching. A kernel with wrong dtype produces silent garbage results — the checks below catch mistakes immediately.

```cpp
torch::Tensor my_op(torch::Tensor src, torch::Tensor weight) {
    // shape checks
    TORCH_CHECK(src.dim() == 2,
        "src must be 2D, got ", src.dim(), "D");
    TORCH_CHECK(src.size(1) == weight.size(0),
        "shape mismatch: src (", src.size(1), ") vs weight (", weight.size(0), ")");

    // dtype checks
    TORCH_CHECK(src.dtype()    == torch::kFloat16, "src must be fp16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be fp16");

    // device checks
    TORCH_CHECK(src.is_cuda(),    "src must be on CUDA");
    TORCH_CHECK(weight.is_cuda(), "weight must be on CUDA");

    // contiguity check (vectorized kernels need contiguous memory)
    TORCH_CHECK(src.is_contiguous(),    "src must be contiguous");
    TORCH_CHECK(weight.is_contiguous(), "weight must be contiguous");

    // divisibility (for vectorized loads)
    TORCH_CHECK(src.size(1) % 8 == 0,
        "hidden_dim must be divisible by 8, got ", src.size(1));

    // ... rest of function
}
```

In Python, these `TORCH_CHECK` failures throw a `RuntimeError` with the message:

```python
mod.my_op(x_cpu, w)   # x is on CPU
# RuntimeError: src must be on CUDA
```

---

## Part 10 — Timing Correctly: The Benchmark Template

```python
def bench(fn, warmup=10, iters=200):
    """
    Warmup: runs the kernel several times so GPU clocks stabilize
            and JIT caches are warm.
    Measure: uses CUDA events (not Python time.time) for accurate GPU timing.
    """
    # warmup — also ensures the kernel is compiled (load_inline compiles lazily)
    for _ in range(warmup):
        fn()

    # synchronize before starting timer
    torch.cuda.synchronize()

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    s.record()                  # place timestamp marker in CUDA stream
    for _ in range(iters):
        fn()                    # kernel launches (async — Python returns immediately)
    e.record()                  # place end marker

    torch.cuda.synchronize()    # wait for all GPU work to complete

    return s.elapsed_time(e) / iters   # milliseconds per iteration


# Usage:
x = torch.randn(1024 * 1024 * 64, device="cuda", dtype=torch.float16)
ms = bench(lambda: mod.my_kernel(x, 0.5))

# Convert to bandwidth:
bytes_rw = x.numel() * x.element_size() * 2   # read + write
gb_per_s = bytes_rw / ms / 1e6
print(f"{ms:.3f} ms  |  {gb_per_s:.1f} GB/s  |  {gb_per_s/288*100:.1f}% util")
```

---

## Part 11 — What `load_inline` Does Under the Hood

```
First call to mod = load_inline(...):

  1. Writes cuda_src to a temp .cu file
  2. Writes cpp_src to a temp .cpp file  
  3. Runs nvcc to compile .cu → .o
  4. Runs g++ to compile .cpp → .o
  5. Links both .o files into a shared library (.so)
  6. Loads the .so with dlopen()
  7. Exposes the declared functions as mod.function_name

  Compiled .so is cached in: /tmp/torch_extensions/<name>/
  Subsequent Python runs: detects cache, skips compilation (~0.1s)
  Only recompiles if: source string changes or cache is cleared

Second call (same Python session or new session):
  1. Checks cache → found
  2. Loads .so from cache
  3. Done in <0.1 seconds
```

To force recompile (if you change the kernel):
```python
mod = load_inline(
    name="my_kernel_v2",    # change the name → new cache key → recompiles
    ...
)
```

Or clear the cache manually:
```bash
rm -rf /tmp/torch_extensions/my_kernel/
```

---

## Part 12 — Tensor Memory Layout: Contiguous vs Non-Contiguous

CUDA kernels assume that `ptr[i]` and `ptr[i+1]` are adjacent in memory (stride = 1 element). This is the "contiguous" layout.

PyTorch operations like `.transpose()`, `.permute()`, `.narrow()` return **non-contiguous** tensors — the data is still the original allocation, but the strides say "skip N elements between logical positions."

```python
x = torch.randn(4, 8, device="cuda")    # shape [4,8], contiguous
y = x.T                                  # shape [8,4], NOT contiguous — same data

print(x.is_contiguous())   # True
print(y.is_contiguous())   # False
print(x.stride())          # (8, 1)  — row-major
print(y.stride())          # (1, 8)  — column-major
```

**Your kernel assumes contiguous. If you pass non-contiguous, data_ptr() still returns
the original allocation start, but the strides are wrong → garbage results.**

The fix:
```python
# In Python before calling your kernel:
y_contig = y.contiguous()   # allocates new contiguous copy if needed
out = mod.my_kernel(y_contig)

# Or check in C++ and tell the user:
TORCH_CHECK(src.is_contiguous(), "src must be contiguous — call .contiguous() first");
```

---

## Quick Reference

```
Task                                    Code
──────────────────────────────────────────────────────────────────
get float32 pointer                     src.data_ptr<float>()
get fp16 pointer                        (__half*)src.data_ptr()
allocate output (same as input)         torch::empty_like(src)
allocate output (explicit dtype)        torch::empty(src.sizes(), src.options().dtype(torch::kFloat32))
total element count                     src.numel()
shape of dimension i                    src.size(i)
check fp16                              src.dtype() == torch::kFloat16
check on GPU                            src.is_cuda()
check contiguous                        src.is_contiguous()
runtime assertion                       TORCH_CHECK(cond, "message")
time a kernel correctly                 torch.cuda.Event + synchronize
force sync (for .item(), .cpu())        torch.cuda.synchronize()
clear compile cache                     rm -rf /tmp/torch_extensions/<name>/
make contiguous in Python               tensor.contiguous()
```
