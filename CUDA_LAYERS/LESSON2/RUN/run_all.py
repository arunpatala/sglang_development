"""
run_all.py — Run all Lesson 2 exercises in sequence
=====================================================
Runs all five exercises, checks correctness, then prints a final
comparison table showing instruction count and smem usage.

Usage:
    # From LESSON2/RUN/ directory:
    python run_all.py

    # Or from the project root:
    python CUDA_LAYERS/LESSON2/RUN/run_all.py
"""

import sys
import os
import subprocess
import torch

# ── helpers ──────────────────────────────────────────────────────────────────

PEAK_BW = 288.0   # GB/s, RTX 4060 Ti


def banner(title):
    width = 60
    print(f"\n{'#'*width}")
    print(f"#  {title}")
    print(f"{'#'*width}\n")


def run_script(path):
    """Run a Python script in this directory, print output."""
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    if result.returncode != 0:
        print(f"  [FAILED] {path}")
        print(result.stderr[-3000:])
        return False
    print(result.stdout)
    return True


def bench(fn, warmup=20, iters=500):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters   # ms


# ── GPU info ─────────────────────────────────────────────────────────────────

banner("GPU Info")
prop = torch.cuda.get_device_properties(0)
cap  = torch.cuda.get_device_capability(0)
print(f"  GPU         : {prop.name}")
print(f"  Capability  : sm_{cap[0]}{cap[1]}")
print(f"  Memory      : {prop.total_memory / 1e9:.1f} GB")
print(f"  SMs         : {prop.multi_processor_count}")
print(f"  Max smem/blk: {prop.shared_memory_per_block // 1024} KB")
print(f"  CUDA        : {torch.version.cuda}")
print(f"  Peak BW     : {PEAK_BW} GB/s (theoretical)")


# ── run each exercise ─────────────────────────────────────────────────────────

exercises = [
    ("Exercise 2.1 — Scalar Reduce (naive smem, baseline)",
     "ex2_1_scalar_reduce.py"),
    ("Exercise 2.2 — Warp Reduce Sum (__shfl_xor_sync)",
     "ex2_2_warp_reduce_sum.py"),
    ("Exercise 2.3 — Warp Reduce Max + Softmax",
     "ex2_3_warp_reduce_max.py"),
    ("Exercise 2.4 — Vec fp16 Load + Warp Reduce (RMSNorm dim=256)",
     "ex2_4_vec_load_warp_reduce.py"),
    ("Exercise 2.5 — Two-Level Block Reduce (multi-warp RMSNorm)",
     "ex2_5_block_reduce.py"),
]

for title, script in exercises:
    banner(title)
    run_script(script)


# ── final comparison table ────────────────────────────────────────────────────

banner("FINAL COMPARISON — RMSNorm Bandwidth Summary")

from torch.utils.cpp_extension import load_inline

# Inline kernels for head-to-head benchmark
cuda_all = r"""
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

// 1. RMSNorm dim=256, single warp per row
__global__ void rmsnorm_warp(
    const uint4* __restrict__ x, const uint4* __restrict__ w,
    uint4* __restrict__ out, float eps, int n_vec)
{
    int lane = threadIdx.x;
    int row  = blockIdx.x;
    uint4 rx = x[row * n_vec + lane];
    __half* xv = reinterpret_cast<__half*>(&rx);
    float sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { float v = __half2float(xv[i]); sq += v*v; }
    float rms_rcp = rsqrtf(warp_reduce_sum(sq) / (n_vec*8) + eps);
    uint4 rw = w[lane];
    __half* wv = reinterpret_cast<__half*>(&rw);
    uint4 ro; __half* ov = reinterpret_cast<__half*>(&ro);
    #pragma unroll
    for (int i = 0; i < 8; ++i) ov[i] = __float2half(__half2float(xv[i]) * rms_rcp * __half2float(wv[i]));
    out[row * n_vec + lane] = ro;
}

// 2. RMSNorm any dim, multi-warp block reduce
__device__ float block_reduce_sum(float val, float* smem) {
    int lane = threadIdx.x % 32, wid = threadIdx.x / 32, nw = blockDim.x / 32;
    float ws = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = ws;
    __syncthreads();
    if (wid == 0) {
        float v = (lane < nw) ? smem[lane] : 0.f;
        ws = warp_reduce_sum(v);
        if (lane == 0) smem[0] = ws;
    }
    __syncthreads();
    return smem[0];
}
__global__ void rmsnorm_block(
    const uint4* __restrict__ x, const uint4* __restrict__ w,
    uint4* __restrict__ out, float eps, int hidden_dim, int n_vec)
{
    extern __shared__ float smem[];
    int row = blockIdx.x;
    uint4 rx = x[row * n_vec + threadIdx.x];
    __half* xv = reinterpret_cast<__half*>(&rx);
    float sq = 0.f;
    #pragma unroll
    for (int i = 0; i < 8; ++i) { float v = __half2float(xv[i]); sq += v*v; }
    float rms_rcp = rsqrtf(block_reduce_sum(sq, smem) / hidden_dim + eps);
    uint4 rw = w[threadIdx.x];
    __half* wv = reinterpret_cast<__half*>(&rw);
    uint4 ro; __half* ov = reinterpret_cast<__half*>(&ro);
    #pragma unroll
    for (int i = 0; i < 8; ++i) ov[i] = __float2half(__half2float(xv[i]) * rms_rcp * __half2float(wv[i]));
    out[row * n_vec + threadIdx.x] = ro;
}

// 3. PyTorch reference RMSNorm (for comparison)

torch::Tensor run_rmsnorm_warp(torch::Tensor x, torch::Tensor w, float eps) {
    auto out = torch::empty_like(x);
    int batch = x.size(0), n_vec = x.size(1) / 8;
    rmsnorm_warp<<<batch, 32>>>((const uint4*)x.data_ptr(),(const uint4*)w.data_ptr(),(uint4*)out.data_ptr(),eps,n_vec);
    return out;
}
torch::Tensor run_rmsnorm_block(torch::Tensor x, torch::Tensor w, float eps) {
    auto out = torch::empty_like(x);
    int batch = x.size(0), hidden = x.size(1), n_vec = hidden / 8;
    int threads = n_vec, nw = threads / 32;
    rmsnorm_block<<<batch, threads, nw*sizeof(float)>>>((const uint4*)x.data_ptr(),(const uint4*)w.data_ptr(),(uint4*)out.data_ptr(),eps,hidden,n_vec);
    return out;
}
"""

cpp_all = """
torch::Tensor run_rmsnorm_warp(torch::Tensor x, torch::Tensor w, float eps);
torch::Tensor run_rmsnorm_block(torch::Tensor x, torch::Tensor w, float eps);
"""

mod = load_inline(
    name="run_all_l2",
    cpp_sources=cpp_all,
    cuda_sources=cuda_all,
    functions=["run_rmsnorm_warp", "run_rmsnorm_block"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)

eps   = 1e-5
batch = 4096

results = []

# RMSNorm warp: dim=256
hidden = 256
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w = torch.ones(hidden, device="cuda", dtype=torch.float16)
bw_bytes = batch * hidden * 2 * 3   # read x + read w + write out

ms = bench(lambda: mod.run_rmsnorm_warp(x, w, eps))
bw = bw_bytes / ms / 1e6
results.append(("RMSNorm warp  (dim=256,  32 threads, 1 warp )", ms, bw, bw / PEAK_BW * 100))

# RMSNorm block: dim=512
hidden = 512
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w = torch.ones(hidden, device="cuda", dtype=torch.float16)
bw_bytes = batch * hidden * 2 * 3
ms = bench(lambda: mod.run_rmsnorm_block(x, w, eps))
bw = bw_bytes / ms / 1e6
results.append(("RMSNorm block (dim=512,  64 threads, 2 warps)", ms, bw, bw / PEAK_BW * 100))

# RMSNorm block: dim=4096
hidden = 4096
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w = torch.ones(hidden, device="cuda", dtype=torch.float16)
bw_bytes = batch * hidden * 2 * 3
ms = bench(lambda: mod.run_rmsnorm_block(x, w, eps))
bw = bw_bytes / ms / 1e6
results.append(("RMSNorm block (dim=4096, 512 threads, 16 wrp)", ms, bw, bw / PEAK_BW * 100))

# PyTorch reference: dim=4096
hidden = 4096
x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w = torch.ones(hidden, device="cuda", dtype=torch.float16)
bw_bytes = batch * hidden * 2 * 3
ms = bench(lambda: torch.nn.functional.rms_norm(x.float(), (hidden,), w.float(), eps).half())
bw = bw_bytes / ms / 1e6
results.append(("PyTorch rms_norm (dim=4096, reference)       ", ms, bw, bw / PEAK_BW * 100))

hdr = f"  {'Kernel':<46}  {'ms':>6}  {'GB/s':>6}  {'Util':>6}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, ms, bw, util in results:
    mark = "✓" if util > 65 else ("~" if util > 40 else "✗")
    print(f"  {name}  {ms:>6.3f}  {bw:>6.1f}  {util:>5.1f}%  {mark}")

print(f"\n  Peak: {PEAK_BW} GB/s | Target: >65% for RMSNorm (memory-bound)")
print(f"  RTX 4060 Ti — Ada Lovelace — sm_89")
print(f"\n  Key insight: warp shuffle used in all kernels above.")
print(f"  The only difference is whether smem is needed for cross-warp reduce.")
print(f"    dim=256:  1 warp  → warp_reduce_sum only (zero smem)")
print(f"    dim>256:  N warps → warp_reduce_sum + smem (1 slot per warp)")
