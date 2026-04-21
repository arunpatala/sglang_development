"""
ex3_4_fused_add_rmsnorm.py — Fused Residual Add + RMSNorm (Phase 2.7)
======================================================================
Every transformer forward pass does this before every attention and MLP block:

    hidden = RMSNorm(hidden + residual)

The unfused version requires 5 HBM accesses:
  read hidden, read residual → write x=hidden+residual → read x → write output

The fused version caches x in shared memory between pass 1 and pass 2:
  read hidden, read residual → [smem cache x] → write output + update residual

HBM access count: 5 → 3 (a 40% reduction in HBM traffic for this operation).

This is torch.ops.sgl_kernel.fused_add_rmsnorm in production SGLang.
Reference: REPOS/flashinfer/include/flashinfer/norm.cuh FusedAddRMSNormKernel

Smem layout:
  [0..n_warps*4)     : reduce buffer (n_warps floats)
  [n_warps*4..)      : x cache      (n_vec uint4 = n_vec * 16 bytes)

Constraint: n_warps*4 + n_vec*16 ≤ 99 KB (RTX 4060 Ti limit)
  For hidden=4096: 16*4 + 512*16 = 8256 bytes → fine
  For hidden=8192: 16*4 + 1024*16 = 16448 bytes → fine
  For hidden=49152 (some large models): would need a fallback

Usage:
    python ex3_4_fused_add_rmsnorm.py
"""

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

__device__ __forceinline__
float block_reduce_sum_2d(float val, float* smem) {
    int lane = threadIdx.x, wid = threadIdx.y, nw = blockDim.y;
    val = warp_reduce_sum(val);
    if (lane == 0) smem[wid] = val;
    __syncthreads();
    val = (wid == 0 && lane < nw) ? smem[lane] : 0.f;
    if (wid == 0) { val = warp_reduce_sum(val); if (lane == 0) smem[0] = val; }
    __syncthreads();
    return smem[0];
}

// Fused Add + RMSNorm
//
// smem layout:
//   float  reduce_smem[n_warps]    ← cross-warp reduction buffer
//   uint4  x_smem[n_vec]           ← x = input+residual cache (avoids re-read from GDDR6X)
//
// After the kernel:
//   residual is updated in-place to x (the transformer needs the updated residual)
//   out contains RMSNorm(x) * weight
__global__ void fused_add_rmsnorm_f16(
    const uint4* __restrict__ input,    // [batch, n_vec]  — the new hidden state
    uint4*       __restrict__ residual, // [batch, n_vec]  — updated in-place to x
    const uint4* __restrict__ weight,   // [n_vec]
    uint4*       __restrict__ out,      // [batch, n_vec]
    float        eps,
    int          n_vec,
    int          hidden_dim,
    int          n_warps)
{
    extern __shared__ float smem[];
    float* reduce_smem = smem;                           // n_warps floats
    uint4* x_smem      = (uint4*)(smem + n_warps);       // n_vec uint4

    int row   = blockIdx.x;
    int n_th  = blockDim.x * blockDim.y;
    int tid   = threadIdx.y * blockDim.x + threadIdx.x;

    // ── Pass 1: compute x = input + residual, cache in smem, sum squares ─────
    float sum_sq = 0.f;
    for (int i = tid; i < n_vec; i += n_th) {
        uint4 ri = input[row * n_vec + i];
        uint4 rr = residual[row * n_vec + i];

        __half* iv = reinterpret_cast<__half*>(&ri);
        __half* rv = reinterpret_cast<__half*>(&rr);

        uint4 rx;
        __half* xv = reinterpret_cast<__half*>(&rx);

        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            float xf = __half2float(iv[j]) + __half2float(rv[j]);
            xv[j]    = __float2half(xf);
            sum_sq  += xf * xf;
        }

        // Cache x in smem — pass 2 reads from here instead of GDDR6X
        x_smem[i] = rx;

        // Update residual in-place: the next layer needs the post-add value
        residual[row * n_vec + i] = rx;
    }
    // Note: the __syncthreads inside block_reduce_sum_2d covers the smem writes above.
    // All x_smem[i] writes complete before any thread reads x_smem[j] in pass 2,
    // because the second __syncthreads() in block_reduce is a full block barrier.

    float total_sq = block_reduce_sum_2d(sum_sq, reduce_smem);
    float rms_rcp  = rsqrtf(total_sq / (float)hidden_dim + eps);

    // ── Pass 2: read x from smem (NOT GDDR6X), normalize ─────────────────────
    for (int i = tid; i < n_vec; i += n_th) {
        uint4 rx = x_smem[i];     // ~32 cycle latency vs ~300 for GDDR6X
        uint4 rw = weight[i];     // still must come from GDDR6X (weight not cached)

        __half* xv = reinterpret_cast<__half*>(&rx);
        __half* wv = reinterpret_cast<__half*>(&rw);

        uint4 ro;
        __half* ov = reinterpret_cast<__half*>(&ro);

        #pragma unroll
        for (int j = 0; j < 8; ++j)
            ov[j] = __float2half(__half2float(xv[j]) * rms_rcp * __half2float(wv[j]));

        out[row * n_vec + i] = ro;
    }
}

void run_fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,   // modified in-place
    torch::Tensor weight,
    torch::Tensor out,
    float eps)
{
    TORCH_CHECK(input.dtype() == torch::kFloat16, "input must be fp16");
    int batch      = input.size(0);
    int hidden_dim = input.size(1);
    TORCH_CHECK(hidden_dim % 8 == 0, "hidden_dim must be divisible by 8");

    int n_vec   = hidden_dim / 8;
    int n_warps = min(max((n_vec + 31) / 32, 1), 16);

    dim3 block(32, n_warps);

    // smem: n_warps floats (reduce) + n_vec uint4 (x cache)
    // Align the uint4 section to 16 bytes: n_warps floats = n_warps*4 bytes
    // Pad to next 16-byte boundary: ceil(n_warps*4/16)*16
    int reduce_bytes = ((n_warps * sizeof(float) + 15) / 16) * 16;
    int x_bytes      = n_vec * sizeof(uint4);   // n_vec * 16
    int smem_bytes   = reduce_bytes + x_bytes;

    TORCH_CHECK(smem_bytes <= 99 * 1024,
                "smem requirement exceeds 99 KB — hidden_dim too large for fused kernel");

    fused_add_rmsnorm_f16<<<batch, block, smem_bytes>>>(
        reinterpret_cast<const uint4*>(input.data_ptr()),
        reinterpret_cast<uint4*>(residual.data_ptr()),
        reinterpret_cast<const uint4*>(weight.data_ptr()),
        reinterpret_cast<uint4*>(out.data_ptr()),
        eps,
        n_vec,
        hidden_dim,
        n_warps);
}
"""

cpp_src = """
void run_fused_add_rmsnorm(
    torch::Tensor input,
    torch::Tensor residual,
    torch::Tensor weight,
    torch::Tensor out,
    float eps);
"""

mod = load_inline(
    name="ex3_4",
    cpp_sources=cpp_src,
    cuda_sources=cuda_src,
    functions=["run_fused_add_rmsnorm"],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
    verbose=False,
)


def fused_ref(inp, res, w, eps=1e-5):
    """Reference: float32 computation."""
    x = inp.float() + res.float()
    rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)
    out = (x / rms * w.float()).half()
    return out, x.half()   # output, updated residual


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


PEAK_BW = 288.0
eps     = 1e-5

# ── Correctness ────────────────────────────────────────────────────────────────
print("Correctness tests (output + residual update):")
for hidden in [256, 512, 1024, 2048, 4096, 8192]:
    batch = 16
    inp = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    res = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
    w   = torch.randn(hidden, device="cuda", dtype=torch.float16)
    out = torch.empty_like(inp)

    # Clone because kernel modifies residual in-place
    res_fused = res.clone()
    mod.run_fused_add_rmsnorm(inp, res_fused, w, out, eps)

    ref_out, ref_res = fused_ref(inp, res, w, eps)
    torch.testing.assert_close(out, ref_out, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(res_fused, ref_res, rtol=1e-3, atol=1e-3)
    print(f"  hidden={hidden:6d}: output PASSED, residual PASSED")

print()

# ── Smem analysis ─────────────────────────────────────────────────────────────
print("Shared memory usage analysis:")
print(f"  {'hidden':>8}  {'n_warps':>7}  {'reduce':>9}  {'x_cache':>9}  {'total':>9}  {'limit':>7}")
print(f"  {'-'*65}")
for hidden in [256, 512, 1024, 2048, 4096, 8192, 16384]:
    n_vec     = hidden // 8
    n_warps   = min(max((n_vec + 31) // 32, 1), 16)
    reduce_b  = ((n_warps * 4 + 15) // 16) * 16
    x_b       = n_vec * 16
    total_b   = reduce_b + x_b
    ok        = "✓" if total_b <= 99*1024 else "✗ EXCEEDS"
    print(f"  {hidden:>8}  {n_warps:>7}  {reduce_b:>8}B  {x_b/1024:>8.1f}KB"
          f"  {total_b/1024:>8.1f}KB  {ok}")

print()

# ── Bandwidth comparison: fused vs unfused ────────────────────────────────────
print("Bandwidth comparison (batch=4096, hidden=4096):")
hidden = 4096
batch  = 4096
inp  = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
res  = torch.randn(batch, hidden, device="cuda", dtype=torch.float16)
w    = torch.ones(hidden, device="cuda", dtype=torch.float16)
out  = torch.empty_like(inp)

# Unfused: separate add + rmsnorm (approximate — does extra work)
x_add = torch.empty_like(inp)

from torch.utils.cpp_extension import load_inline as li
cuda_unfused = r"""
#include <cuda_fp16.h>
#include <stdint.h>
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xffffffff, v, 16); v += __shfl_xor_sync(0xffffffff, v, 8);
    v += __shfl_xor_sync(0xffffffff, v, 4);  v += __shfl_xor_sync(0xffffffff, v, 2);
    v += __shfl_xor_sync(0xffffffff, v, 1);  return v;
}
__device__ float block_rs(float v, float* sm) {
    int l=threadIdx.x, w=threadIdx.y, n=blockDim.y;
    v=warp_reduce_sum(v); if(l==0)sm[w]=v; __syncthreads();
    v=(w==0&&l<n)?sm[l]:0.f; if(w==0){v=warp_reduce_sum(v);if(l==0)sm[0]=v;}
    __syncthreads(); return sm[0];
}
__global__ void rmsnorm_plain(const uint4* x, const uint4* wt, uint4* out,
                               float eps, int nv, int d) {
    extern __shared__ float sm[];
    int row=blockIdx.x, nth=blockDim.x*blockDim.y, tid=threadIdx.y*32+threadIdx.x;
    float sq=0.f;
    for(int i=tid;i<nv;i+=nth){uint4 r=x[row*nv+i];__half* v=(__half*)&r;
        for(int j=0;j<8;++j){float f=__half2float(v[j]);sq+=f*f;}}
    float rcp=rsqrtf(block_rs(sq,sm)/(float)d+eps);
    for(int i=tid;i<nv;i+=nth){uint4 rx=x[row*nv+i],rw=wt[i],ro;
        __half *xv=(__half*)&rx,*wv=(__half*)&rw,*ov=(__half*)&ro;
        for(int j=0;j<8;++j)ov[j]=__float2half(__half2float(xv[j])*rcp*__half2float(wv[j]));
        out[row*nv+i]=ro;}
}
torch::Tensor run_plain(torch::Tensor x, torch::Tensor wt, float eps) {
    auto out=torch::empty_like(x); int b=x.size(0),h=x.size(1),nv=h/8;
    int nw=min(max((nv+31)/32,1),16);
    rmsnorm_plain<<<b,dim3(32,nw),nw*4>>>((uint4*)x.data_ptr(),(uint4*)wt.data_ptr(),(uint4*)out.data_ptr(),eps,nv,h);
    return out;
}
"""
mplain = li(name="ex3_4_plain", cpp_sources="torch::Tensor run_plain(torch::Tensor,torch::Tensor,float);",
            cuda_sources=cuda_unfused, functions=["run_plain"],
            extra_cuda_cflags=["-O3","-arch=sm_89","--use_fast_math"], verbose=False)

# x = inp + res (separate op)
x_add = inp + res

ms_unfused = bench(lambda: (mplain.run_plain(x_add, w, eps),))
ms_fused   = bench(lambda: mod.run_fused_add_rmsnorm(inp, res.clone(), w, out, eps))

# Unfused bytes: read x (once, after separate add) + read w + write out
unfused_bytes = batch * hidden * 2 * 3
# Fused bytes: read inp + read res + write residual + read w + write out = 5 tensors
# But pass 2 reads x from smem → effectively: 2 reads (inp+res) + 1 write (residual) + 1 read (w) + 1 write (out)
fused_bytes   = batch * hidden * 2 * 5  # conservative (includes residual write)

bw_unfused = unfused_bytes / ms_unfused / 1e6
bw_fused   = fused_bytes   / ms_fused   / 1e6

print(f"  Unfused RMSNorm (x cached by prior add): "
      f"{ms_unfused:.3f} ms | {bw_unfused:.1f} GB/s")
print(f"  Fused   Add+RMSNorm (x in smem):         "
      f"{ms_fused:.3f} ms | {bw_fused:.1f} GB/s")
print()
print("Key insight:")
print("  The fused kernel reads x (= input + residual) exactly ONCE from GDDR6X.")
print("  Pass 2 reads x from smem (32 cycle latency vs 300 cycles from GDDR6X).")
print("  This saves one full tensor read = batch * hidden * 2 bytes per layer.")
print(f"  For batch=4096, hidden=4096: saves {batch*hidden*2/1e6:.0f} MB per norm call.")
print()
print("SGLang production call:")
print("  from sgl_kernel import fused_add_rmsnorm")
print("  fused_add_rmsnorm(hidden_state, residual, norm_weight, eps)")
print("  # hidden_state is updated in-place with the norm output")
print("  # residual    is updated in-place with hidden_state + residual")
