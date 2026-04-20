"""
Exercise 2.3 — Warp Reduce Max with __shfl_xor_sync
====================================================
Same 5-step butterfly pattern as warp_reduce_sum, but uses fmaxf
instead of addition. Every lane receives the global max of all 32 inputs.

Used in:
  - Softmax: reduce max across the query row before exp()
  - Online softmax (Flash Attention): track running max across KV tiles
  - Top-k: find the k-th largest value within a warp

GPU: RTX 4060 Ti | PTX: 5 × shfl.sync.bfly.b32 + 5 × fmax.ftz.f32
"""

import torch
from torch.utils.cpp_extension import load_inline

cuda_src = r"""
// Warp-level max reduction using __shfl_xor_sync + fmaxf.
// Identical structure to warp_reduce_sum — same XOR offsets, same 5 steps.
// fmaxf handles NaN correctly (returns the non-NaN operand).
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;   // all 32 lanes hold the global max
}

// One warp per block. Each block finds max of 32 floats.
__global__ void reduce_warp_max(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    int lane = threadIdx.x;
    int gid  = blockIdx.x * 32 + lane;

    float val    = src[gid];
    float result = warp_reduce_max(val);

    if (lane == 0) out[blockIdx.x] = result;
}

// Demonstrate the full softmax numerics pattern:
// max reduce → shift → exp → sum reduce → normalize.
// Each block processes one "row" of 32 logits.
__global__ void softmax_warp(
    const float* __restrict__ logits,
    float*       __restrict__ probs,
    int          row_len)   // = 32 for this exercise
{
    int lane = threadIdx.x;
    int row  = blockIdx.x;
    float x  = logits[row * row_len + lane];

    // Step 1: reduce max (for numerical stability)
    float row_max = warp_reduce_max(x);

    // Step 2: shift and exponentiate
    float ex = expf(x - row_max);

    // Step 3: reduce sum of exp values
    float ex_sum = ex;
    ex_sum += __shfl_xor_sync(0xffffffff, ex_sum, 16);
    ex_sum += __shfl_xor_sync(0xffffffff, ex_sum,  8);
    ex_sum += __shfl_xor_sync(0xffffffff, ex_sum,  4);
    ex_sum += __shfl_xor_sync(0xffffffff, ex_sum,  2);
    ex_sum += __shfl_xor_sync(0xffffffff, ex_sum,  1);

    // Step 4: normalize
    probs[row * row_len + lane] = ex / ex_sum;
}

torch::Tensor warp_reduce_max_fn(torch::Tensor src) {
    TORCH_CHECK(src.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(src.numel() % 32 == 0, "numel must be divisible by 32");
    int blocks = src.numel() / 32;
    auto out   = torch::zeros({blocks}, src.options());
    reduce_warp_max<<<blocks, 32>>>(src.data_ptr<float>(), out.data_ptr<float>());
    return out;
}

torch::Tensor warp_softmax(torch::Tensor logits) {
    TORCH_CHECK(logits.dtype() == torch::kFloat32, "Expected float32");
    TORCH_CHECK(logits.dim() == 2 && logits.size(1) == 32, "Shape must be (B, 32)");
    auto probs = torch::empty_like(logits);
    int B = logits.size(0);
    softmax_warp<<<B, 32>>>(
        logits.data_ptr<float>(), probs.data_ptr<float>(), 32);
    return probs;
}
"""

cpp_src = """
torch::Tensor warp_reduce_max_fn(torch::Tensor src);
torch::Tensor warp_softmax(torch::Tensor logits);
"""

mod = load_inline(
    name="ex2_3_warp_reduce_max",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["warp_reduce_max_fn", "warp_softmax"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)

# ---- correctness: max reduce ----
N = 1024 * 32
x = torch.randn(N, device="cuda", dtype=torch.float32)
out = mod.warp_reduce_max_fn(x)
ref = x.view(-1, 32).max(dim=1).values
torch.testing.assert_close(out, ref, rtol=1e-5, atol=1e-5)
print("warp_reduce_max: PASSED")

# ---- correctness: softmax ----
B = 512
logits = torch.randn(B, 32, device="cuda", dtype=torch.float32)
probs  = mod.warp_softmax(logits)
ref_probs = torch.softmax(logits, dim=1)
torch.testing.assert_close(probs, ref_probs, rtol=1e-5, atol=1e-5)
print("warp_softmax   : PASSED")
print(f"  Row 0 sum  : {probs[0].sum().item():.6f}  (should be 1.0)")
print(f"  Row 0 max  : {probs[0].max().item():.4f}")

print(f"\n{'='*55}")
print(f"  Kernel : warp_reduce_max (5 × shfl_xor + fmaxf)")
print(f"  Kernel : warp_softmax    (max reduce → exp → sum reduce → div)")
print(f"  Both kernels: zero shared memory, zero __syncthreads()")
print(f"  Max reduce uses same butterfly XOR as sum reduce — just fmaxf not +")
print(f"{'='*55}")
print(f"  Next: ex2_4 — combine Lesson 1 (vec loads) + Lesson 2 (reduce)")
print(f"        → RMSNorm pass-1: load fp16, accumulate sum-of-squares, warp reduce")
