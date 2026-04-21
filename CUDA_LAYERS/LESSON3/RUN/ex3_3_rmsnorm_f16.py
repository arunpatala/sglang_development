"""
ex3_3_rmsnorm_f16.py — RMSNorm for fp16 tensors, any hidden dimension
======================================================================
This is the first real transformer model layer.

RMSNorm is called before every attention block and every MLP block in
Llama, Mistral, Qwen, and DeepSeek. It normalizes each token's hidden
state by its root-mean-square.

Formula: output[i] = input[i] / RMS(input) * weight[i]
         where RMS(x) = sqrt(mean(x²) + eps)

This kernel combines all three lessons:
  Lesson 1: 128-bit vectorized loads (uint4 = 8 × fp16)
  Lesson 2: Warp shuffle reduce (intra-warp sum-of-squares)
  Lesson 3: Block reduce with smem (cross-warp sum-of-squares)
  + rsqrtf → MUFU.RSQ PTX instruction (1 clock cycle)

Two-pass structure:
  Pass 1: load x → compute x² → block_reduce → rms_rcp = rsqrt(mean_sq + eps)
  Pass 2: load x + weight → output = x * rms_rcp * w → store

Target: >70% of peak bandwidth = >201 GB/s on RTX 4060 Ti

Usage:
    python ex3_3_rmsnorm_f16.py
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
#include <cuda_fp16.h>
#include <stdint.h>

// ─── Primitives ───────────────────────────────────────────────────────────────

__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

__device__ __forceinline__
float block_reduce_sum_2d(float val, float* smem) {
    int lane    = threadIdx.x;
    int warp_id = threadIdx.y;
    int n_warps = blockDim.y;

    val = warp_reduce_sum(val);

    if (lane == 0) smem[warp_id] = val;
    __syncthreads();

    val = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;
    }
    __syncthreads();

    return smem[0];
}

// ─── RMSNorm kernel ──────────────────────────────────────────────────────────
// 2D block: threadIdx.x = lane (0–31), threadIdx.y = warp_id
// One block per row (one token's hidden state).
// Each thread processes n_vec / (32 * n_warps) uint4 chunks per pass.
// n_vec = hidden_dim / 8  (8 fp16 values per uint4)
//
// Constraints:
//   hidden_dim must be divisible by 8
//   n_warps = min(ceil(n_vec / 32), 16)  → max 512 threads per block
__global__ void rmsnorm_f16(
    const uint4* __restrict__ x,       // [batch, n_vec]  (n_vec = hidden/8)
    const uint4* __restrict__ weight,  // [n_vec]
    uint4*       __restrict__ out,     // [batch, n_vec]
    float        eps,
    int          n_vec,
    int          hidden_dim)
{
    extern __shared__ float smem[];

    int row     = blockIdx.x;
    int n_th    = blockDim.x * blockDim.y;                      // 32 * n_warps
    int tid     = threadIdx.y * blockDim.x + threadIdx.x;       // flat thread id

    // ── Pass 1: load x, compute partial sum-of-squares ───────────────────────
    float sum_sq = 0.f;
    for (int i = tid; i < n_vec; i += n_th) {
        // 128-bit load (Lesson 1) — single LD.E.128 instruction
        uint4 raw = x[row * n_vec + i];
        // Reinterpret 128 bits as 8 fp16 values
        __half* vals = reinterpret_cast<__half*>(&raw);
        // Accumulate in float32 for numerical stability
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float v = __half2float(vals[j]);
            sum_sq += v * v;
        }
    }

    // Cross-warp block reduce: all threads contribute their partial sum_sq
    float total_sq = block_reduce_sum_2d(sum_sq, smem);

    // rsqrtf → MUFU.RSQ PTX instruction — one GPU clock cycle
    float rms_rcp = rsqrtf(total_sq / (float)hidden_dim + eps);

    // ── Pass 2: load x + weight, normalize, store ────────────────────────────
    for (int i = tid; i < n_vec; i += n_th) {
        uint4 rx = x[row * n_vec + i];       // 128-bit load of input
        uint4 rw = weight[i];                // 128-bit load of weight

        __half* xv = reinterpret_cast<__half*>(&rx);
        __half* wv = reinterpret_cast<__half*>(&rw);

        uint4 ro;
        __half* ov = reinterpret_cast<__half*>(&ro);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float xf = __half2float(xv[j]);
            float wf = __half2float(wv[j]);
            ov[j] = __float2half(xf * rms_rcp * wf);
        }

        out[row * n_vec + i] = ro;   // 128-bit store
    }
}

torch::Tensor run_rmsnorm_f16(
    torch::Tensor x,
    torch::Tensor weight,
    float eps)
{
    TORCH_CHECK(x.dtype() == torch::kFloat16, "x must be fp16");
    TORCH_CHECK(weight.dtype() == torch::kFloat16, "weight must be fp16");
    TORCH_CHECK(x.dim() == 2, "x must be [batch, hidden_dim]");
    TORCH_CHECK(x.is_contiguous() && weight.is_contiguous(),
                "tensors must be contiguous");

    int batch      = x.size(0);
    int hidden_dim = x.size(1);
    TORCH_CHECK(hidden_dim % 8 == 0,
                "hidden_dim must be divisible by 8 for uint4 loads");

    auto out  = torch::empty_like(x);
    int n_vec = hidden_dim / 8;

    // Choose num_warps: enough threads to cover n_vec, max 16 warps (512 threads)
    int n_warps = (n_vec + 31) / 32;
    if (n_warps > 16) n_warps = 16;

    dim3 block(32, n_warps);
    int smem_bytes = n_warps * sizeof(float);

    rmsnorm_f16<<<batch, block, smem_bytes>>>(
        reinterpret_cast<const uint4*>(x.data_ptr()),
        reinterpret_cast<const uint4*>(weight.data_ptr()),
        reinterpret_cast<uint4*>(out.data_ptr()),
        eps,
        n_vec,
        hidden_dim);

    return out;
}
"""

cpp_src = """
torch::Tensor run_rmsnorm_f16(torch::Tensor x, torch::Tensor weight, float eps);
"""

mod = load_inline(
    name="ex3_3",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_rmsnorm_f16"],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
    verbose=False,
)


def rms_norm_ref(x: torch.Tensor, w: torch.Tensor, eps: float = 1e-5):
    """Reference: compute in float32 for correctness."""
    x_f = x.float()
    rms = torch.sqrt(x_f.pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x_f / rms * w.float()).half()


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


PEAK_BW = 288.0   # GB/s, RTX 4060 Ti
eps     = 1e-5

# ── Correctness across all production hidden dims ────────────────────────────
print("Correctness tests:")
for hidden in [256, 512, 1024, 2048, 4096, 8192, 11008]:
    batch = 32
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    w = torch.randn(hidden, device="cuda", dtype=torch.float16)
    out = mod.run_rmsnorm_f16(x, w, eps)
    ref = rms_norm_ref(x, w, eps)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
    print(f"  hidden={hidden:6d}: PASSED")

print()

# ── Edge cases ────────────────────────────────────────────────────────────────
print("Edge cases:")

# Single token
x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
w = torch.ones(128, device="cuda", dtype=torch.float16)
out = mod.run_rmsnorm_f16(x, w, eps)
ref = rms_norm_ref(x, w, eps)
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
print("  batch=1, hidden=128: PASSED")

# Large batch
x = torch.randn(8192, 4096, device="cuda", dtype=torch.float16)
w = torch.ones(4096, device="cuda", dtype=torch.float16)
out = mod.run_rmsnorm_f16(x, w, eps)
ref = rms_norm_ref(x, w, eps)
torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
print("  batch=8192, hidden=4096: PASSED")

print()

# ── Bandwidth benchmarks ──────────────────────────────────────────────────────
print("Bandwidth benchmarks (batch=4096):")
print(f"  {'hidden':>8}  {'n_warps':>7}  {'threads':>7}  {'ms':>7}  {'GB/s':>7}  {'Util':>7}")
print(f"  {'-'*60}")

batch = 4096
for hidden in [256, 512, 1024, 2048, 4096, 8192]:
    x = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    w = torch.ones(hidden, device="cuda", dtype=torch.float16)

    n_vec   = hidden // 8
    n_warps = min(max((n_vec + 31) // 32, 1), 16)
    threads = 32 * n_warps

    ms       = bench(lambda: mod.run_rmsnorm_f16(x, w, eps))
    bw_bytes = batch * hidden * 2 * 3   # read x + read weight + write out
    bw       = bw_bytes / ms / 1e6
    util     = bw / PEAK_BW * 100
    mark     = "✓" if util > 65 else ("~" if util > 40 else "✗")

    print(f"  {hidden:>8}  {n_warps:>7}  {threads:>7}  {ms:>7.3f}  "
          f"{bw:>7.1f}  {util:>6.1f}%  {mark}")

print()
print(f"Peak: {PEAK_BW} GB/s | Target: >70% (>201 GB/s)")
print()

# ── Two-pass cost analysis ────────────────────────────────────────────────────
print("─" * 60)
print("Two-pass analysis for hidden=4096, batch=4096:")
hidden = 4096
batch  = 4096
bw_bytes_pass1 = batch * hidden * 2        # read x once
bw_bytes_pass2 = batch * hidden * 2 * 2   # read x again + read weight
bw_bytes_write = batch * hidden * 2        # write out
total_bytes    = bw_bytes_pass1 + bw_bytes_pass2 + bw_bytes_write
print(f"  Pass 1 reads x:       {bw_bytes_pass1/1e6:.0f} MB")
print(f"  Pass 2 reads x again: {bw_bytes_pass1/1e6:.0f} MB  ← this is the wasted read")
print(f"  Pass 2 reads weight:  {bw_bytes_pass1/1e6:.0f} MB")
print(f"  Write output:         {bw_bytes_write/1e6:.0f} MB")
print(f"  Total:                {total_bytes/1e6:.0f} MB = 3 × tensor_size")
print()
print("Question: How would you eliminate the second x read?")
print("Answer: Cache x in shared memory between pass 1 and pass 2.")
print("        → Exercise 3.4: Fused Add + RMSNorm")
print()
print("ncu profiling command:")
print("  ncu --kernel-name rmsnorm_f16 \\")
print("      --metrics gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\\")
print("  sm__warps_active.avg.pct_of_peak_sustained_active,\\")
print("  l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum \\")
print("      python ex3_3_rmsnorm_f16.py")
print()
print("Expected: dram_throughput >70%, bank_conflicts = 0")
