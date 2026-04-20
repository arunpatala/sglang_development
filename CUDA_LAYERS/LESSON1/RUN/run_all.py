"""
run_all.py — Run all Lesson 1 exercises in sequence
=====================================================
Produces a final bandwidth comparison table.

Usage:
    # From project root:
    PATH="/home/arun/PROJECTS/sglang_development/.conda/bin:$PATH" \
        /home/arun/PROJECTS/sglang_development/.conda/bin/python \
        CUDA_LAYERS/LESSON1/RUN/run_all.py

    # Or activate the conda env first:
    conda activate /home/arun/PROJECTS/sglang_development/.conda
    python run_all.py
"""

import sys
import subprocess
import torch

PEAK_BW = 288.0   # GB/s, RTX 4060 Ti

def banner(title):
    width = 60
    print(f"\n{'#'*width}")
    print(f"#  {title}")
    print(f"{'#'*width}\n")


def run_script(path):
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        print(f"  [FAILED] {path}")
        print(result.stderr[-2000:])
        return False
    print(result.stdout)
    return True


# ── shared bench utility (same as in each file) ──────────────────────────────
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


# ── print GPU info ────────────────────────────────────────────────────────────
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


# ── run each exercise ────────────────────────────────────────────────────────
exercises = [
    ("Exercise 1.1 — Scalar Copy (fp16)",      "ex1_1_scalar_copy.py"),
    ("Exercise 1.3 — Vec Copy f32 (float4)",   "ex1_3_vec_copy_f32.py"),
    ("Exercise 1.4 — Vec Copy/Scale fp16 (uint4)", "ex1_4_vec_copy_f16.py"),
    ("Exercise 1.5 — Templated Elementwise",   "ex1_5_templated.py"),
]

for title, script in exercises:
    banner(title)
    run_script(script)


# ── final comparison table (inline, not via subprocess) ─────────────────────
banner("FINAL COMPARISON — Bandwidth Summary")

from torch.utils.cpp_extension import load_inline

# Ex 1.1 — scalar fp16
cuda_s = r"""
#include <cuda_fp16.h>
__global__ void copy_scalar(const __half* __restrict__ s, __half* __restrict__ d, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = s[i];
}
torch::Tensor scalar_copy(torch::Tensor src) {
    auto dst = torch::empty_like(src);
    int n = src.numel();
    copy_scalar<<<(n+255)/256,256>>>((const __half*)src.data_ptr(),(__half*)dst.data_ptr(),n);
    return dst;
}
"""
m1 = load_inline(name="run_all_s1",cpp_sources="torch::Tensor scalar_copy(torch::Tensor);",
                 cuda_sources=cuda_s, functions=["scalar_copy"],
                 extra_cuda_cflags=["-O3","-arch=sm_89"], verbose=False)

# Ex 1.3 — float4
cuda_f4 = r"""
__global__ void copy_f4(const float4* __restrict__ s, float4* __restrict__ d, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) d[i]=s[i];
}
torch::Tensor vec_copy_f32(torch::Tensor src) {
    auto dst = torch::empty_like(src);
    int n = src.numel()/4;
    copy_f4<<<(n+255)/256,256>>>((const float4*)src.data_ptr<float>(),(float4*)dst.data_ptr<float>(),n);
    return dst;
}
"""
m3 = load_inline(name="run_all_s3",cpp_sources="torch::Tensor vec_copy_f32(torch::Tensor);",
                 cuda_sources=cuda_f4, functions=["vec_copy_f32"],
                 extra_cuda_cflags=["-O3","-arch=sm_89"], verbose=False)

# Ex 1.4 — uint4 fp16
cuda_u4 = r"""
#include <cuda_fp16.h>
#include <stdint.h>
__global__ void copy_u4(const uint4* __restrict__ s, uint4* __restrict__ d, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i<n) d[i]=s[i];
}
__global__ void scale_u4(const uint4* __restrict__ s, uint4* __restrict__ d, float sc, int n) {
    int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i>=n) return;
    uint4 r = s[i];
    const __half* iv = reinterpret_cast<const __half*>(&r);
    uint4 o; __half* ov = reinterpret_cast<__half*>(&o);
    #pragma unroll
    for(int j=0;j<8;++j) ov[j]=__float2half(__half2float(iv[j])*sc);
    d[i]=o;
}
torch::Tensor vec_copy_f16(torch::Tensor src) {
    auto dst=torch::empty_like(src); int n=src.numel()/8;
    copy_u4<<<(n+255)/256,256>>>((const uint4*)src.data_ptr(),(uint4*)dst.data_ptr(),n);
    return dst;
}
torch::Tensor scale_f16(torch::Tensor src, float sc) {
    auto dst=torch::empty_like(src); int n=src.numel()/8;
    scale_u4<<<(n+255)/256,256>>>((const uint4*)src.data_ptr(),(uint4*)dst.data_ptr(),sc,n);
    return dst;
}
"""
m4 = load_inline(name="run_all_s4",
                 cpp_sources="torch::Tensor vec_copy_f16(torch::Tensor);\ntorch::Tensor scale_f16(torch::Tensor, float);",
                 cuda_sources=cuda_u4, functions=["vec_copy_f16","scale_f16"],
                 extra_cuda_cflags=["-O3","-arch=sm_89"], verbose=False)

N = 1024 * 1024 * 64
f16 = torch.randn(N, device="cuda", dtype=torch.float16)
f32 = torch.randn(N, device="cuda", dtype=torch.float32)

results = []

ms = bench(lambda: m1.scalar_copy(f16))
bw = N * 2 * 2 / ms / 1e6
results.append(("Scalar copy (fp16, LD.E.16)      ", ms, bw, bw / PEAK_BW * 100))

ms = bench(lambda: m3.vec_copy_f32(f32))
bw = N * 4 * 2 / ms / 1e6
results.append(("Vec copy   (f32,  float4 LD.128) ", ms, bw, bw / PEAK_BW * 100))

ms = bench(lambda: m4.vec_copy_f16(f16))
bw = N * 2 * 2 / ms / 1e6
results.append(("Vec copy   (fp16, uint4  LD.128) ", ms, bw, bw / PEAK_BW * 100))

ms = bench(lambda: m4.scale_f16(f16, 0.5))
bw = N * 2 * 2 / ms / 1e6
results.append(("Vec scale  (fp16, uint4 + fp32)  ", ms, bw, bw / PEAK_BW * 100))

ms = bench(lambda: torch.empty_like(f16).copy_(f16))
bw = N * 2 * 2 / ms / 1e6
results.append(("PyTorch baseline (cudaMemcpyAsync)", ms, bw, bw / PEAK_BW * 100))

hdr = f"  {'Kernel':<40}  {'ms':>6}  {'GB/s':>6}  {'Util':>6}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, ms, bw, util in results:
    mark = "✓" if util > 75 else ("~" if util > 40 else "✗")
    print(f"  {name}  {ms:>6.3f}  {bw:>6.1f}  {util:>5.1f}%  {mark}")

print(f"\n  Peak: {PEAK_BW} GB/s | Target: >230 GB/s (>80%) for vectorized kernels")
print(f"  RTX 4060 Ti — Ada Lovelace — sm_89")
