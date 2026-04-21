"""
ex3_1_bank_conflicts.py — Shared Memory Bank Conflict Demonstration
====================================================================
Demonstrates the hardware cost of bank conflicts using two contrasting
shared memory access patterns.

Pattern A (no conflict):  thread i reads smem[i]      → bank i % 32   → 32 distinct banks
Pattern B (32-way conflict): thread i reads smem[i*32] → bank 0 for all → serialized 32×

Bank rule: bank = (byte_address / 4) % 32
           For float: smem[i] is in bank i % 32

Usage:
    python ex3_1_bank_conflicts.py

    # See hardware conflict count with ncu:
    ncu --kernel-name smem_with_conflict \\
        --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \\
        python ex3_1_bank_conflicts.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <stdint.h>

// ─── Pattern A: No bank conflicts ─────────────────────────────────────────────
// Thread i writes smem[i] → bank i % 32 → 32 distinct banks → zero conflicts
// Thread i reads  smem[i] → bank i % 32 → zero conflicts
__global__ void smem_no_conflict(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int n)
{
    extern __shared__ float smem[];
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Write: each thread i writes to smem[i] — distinct banks
    smem[threadIdx.x] = src[gid];
    __syncthreads();

    // Read: each thread i reads from smem[i] — distinct banks
    dst[gid] = smem[threadIdx.x] * 2.f;
}

// ─── Pattern B: 32-way bank conflicts ─────────────────────────────────────────
// Thread i writes smem[i * 32]:
//   thread 0 → smem[0]   → byte addr 0   → bank 0
//   thread 1 → smem[32]  → byte addr 128 → bank 0  (128/4 = 32, 32 % 32 = 0)
//   thread 2 → smem[64]  → byte addr 256 → bank 0
//   ...
//   thread 31 → smem[992] → byte addr 3968 → bank 0
// All 32 threads hit bank 0 → hardware serializes 32 accesses → 32× slowdown
__global__ void smem_with_conflict(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int n)
{
    // Need smem large enough: max index = threadIdx.x * 32 = 31 * 32 = 992
    __shared__ float smem[1024];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    // Write: stride-32 — all threads map to bank 0
    smem[threadIdx.x * 32] = src[gid];
    __syncthreads();

    // Read: stride-32 — 32-way bank conflict on every read
    dst[gid] = smem[threadIdx.x * 32] * 2.f;
}

// ─── Pattern C: 2-way bank conflict (stride-16) ───────────────────────────────
// Thread i writes smem[i * 16]:
//   thread 0 → smem[0]   → bank 0
//   thread 16 → smem[256] → bank 0  ← only one conflict partner
//   thread 1 → smem[16]  → bank 16
//   thread 17 → smem[272] → bank 16 ← only one conflict partner
// 2-way conflict (half as bad as 32-way)
__global__ void smem_stride16_conflict(
    const float* __restrict__ src,
    float*       __restrict__ dst,
    int n)
{
    // Max index = 31 * 16 = 496 → 512 slots enough
    __shared__ float smem[512];

    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n) return;

    smem[threadIdx.x * 16] = src[gid];
    __syncthreads();
    dst[gid] = smem[threadIdx.x * 16] * 2.f;
}

torch::Tensor run_no_conflict(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n     = src.numel();
    int block = 32;
    int grid  = (n + block - 1) / block;
    smem_no_conflict<<<grid, block, block * sizeof(float)>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    return dst;
}

torch::Tensor run_with_conflict(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n     = src.numel();
    int block = 32;
    int grid  = (n + block - 1) / block;
    // smem: 1024 floats = 4096 bytes (each thread needs smem[tx*32], max tx=31 → index 992)
    smem_with_conflict<<<grid, block, 1024 * sizeof(float)>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    return dst;
}

torch::Tensor run_stride16(torch::Tensor src) {
    auto dst  = torch::empty_like(src);
    int n     = src.numel();
    int block = 32;
    int grid  = (n + block - 1) / block;
    smem_stride16_conflict<<<grid, block, 512 * sizeof(float)>>>(
        src.data_ptr<float>(), dst.data_ptr<float>(), n);
    return dst;
}
"""

cpp_src = """
torch::Tensor run_no_conflict(torch::Tensor src);
torch::Tensor run_with_conflict(torch::Tensor src);
torch::Tensor run_stride16(torch::Tensor src);
"""

mod = load_inline(
    name="ex3_1",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_no_conflict", "run_with_conflict", "run_stride16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)


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
    return s.elapsed_time(e) / iters


N = 32 * 10000
x = torch.randn(N, device="cuda", dtype=torch.float32)

# ── Correctness ────────────────────────────────────────────────────────────────
out_a = mod.run_no_conflict(x)
out_b = mod.run_with_conflict(x)
out_c = mod.run_stride16(x)
torch.testing.assert_close(out_a, x * 2.0, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(out_b, x * 2.0, rtol=1e-5, atol=1e-5)
torch.testing.assert_close(out_c, x * 2.0, rtol=1e-5, atol=1e-5)
print("All three patterns produce correct results.")
print("Bank conflicts don't break correctness — they only slow you down.\n")

# ── Benchmark ─────────────────────────────────────────────────────────────────
ms_a = bench(lambda: mod.run_no_conflict(x))
ms_b = bench(lambda: mod.run_with_conflict(x))
ms_c = bench(lambda: mod.run_stride16(x))

print(f"Pattern A — stride-1  (no conflict):    {ms_a*1000:6.2f} µs  (baseline)")
print(f"Pattern C — stride-16 (2-way conflict): {ms_c*1000:6.2f} µs  ({ms_c/ms_a:.1f}× slower)")
print(f"Pattern B — stride-32 (32-way conflict):{ms_b*1000:6.2f} µs  ({ms_b/ms_a:.1f}× slower)")
print()
print("Expected pattern: no conflict fastest, stride-32 slowest.")
print("The 32-way conflict serializes all 32 warp threads' smem accesses.")
print()
print("─" * 60)
print("To measure exact bank conflict counts:")
print()
print("  ncu --kernel-name smem_no_conflict \\")
print("      --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\\")
print("  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum \\")
print("      python ex3_1_bank_conflicts.py")
print()
print("Expected ncu output:")
print("  smem_no_conflict:    bank_conflicts_ld = 0,  bank_conflicts_st = 0")
print("  smem_with_conflict:  bank_conflicts_ld = 31, bank_conflicts_st = 31  (per warp)")
print()
print("─" * 60)
print("Bank assignment formula: bank = (byte_address / 4) % 32")
print()
print("smem[0]   → byte 0   → bank  0")
print("smem[1]   → byte 4   → bank  1")
print("smem[31]  → byte 124 → bank 31")
print("smem[32]  → byte 128 → bank  0  ← wraps around")
print()
print("stride-32 access: smem[tx * 32]")
print("  tx=0  → smem[0]   → byte 0    → bank 0")
print("  tx=1  → smem[32]  → byte 128  → bank 0  ← conflict!")
print("  tx=31 → smem[992] → byte 3968 → bank 0  ← conflict!")
print("All 32 threads hit bank 0 → 32-way conflict → 32 serial transactions")
