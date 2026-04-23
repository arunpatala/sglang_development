"""
run_all.py — Run all Lesson 4 exercises in sequence
====================================================
Runs all four exercises, checks correctness, then prints a final
comparison table showing all GEMM variants and their TFLOPS.

The comparison uses M=N=K=1024 (small enough for naive to finish in seconds,
large enough to be representative).

Usage:
    # From LESSON4/RUN/ directory:
    python run_all.py

    # Or from the project root:
    python CUDA_LAYERS/LESSON4/RUN/run_all.py
"""

import sys
import os
import subprocess
import torch
from torch.utils.cpp_extension import load_inline

# ── helpers ───────────────────────────────────────────────────────────────────

PEAK_TFLOPS = 44.0   # fp16 tensor core peak, RTX 4060 Ti
PEAK_BW     = 288.0  # GB/s GDDR6X


def banner(title):
    width = 62
    print(f"\n{'#'*width}")
    print(f"#  {title}")
    print(f"{'#'*width}\n")


def run_script(path):
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


def bench(fn, warmup=10, iters=200):
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


# ── GPU info ──────────────────────────────────────────────────────────────────

banner("GPU Info")
prop = torch.cuda.get_device_properties(0)
cap  = torch.cuda.get_device_capability(0)
print(f"  GPU           : {prop.name}")
print(f"  Capability    : sm_{cap[0]}{cap[1]}")
print(f"  Memory        : {prop.total_memory / 1e9:.1f} GB")
print(f"  SMs           : {prop.multi_processor_count}")
print(f"  Max smem/blk  : {prop.shared_memory_per_block // 1024} KB")
print(f"  CUDA          : {torch.version.cuda}")
print(f"  Peak BW       : {PEAK_BW} GB/s (theoretical)")
print(f"  Peak FP16     : ~{PEAK_TFLOPS} TFLOPS (tensor cores, theoretical)")

# ── run each exercise ─────────────────────────────────────────────────────────

exercises = [
    ("Exercise 4.1 — Naive GEMM (global memory only)",
     "ex4_1_naive_gemm.py"),
    ("Exercise 4.2 — Tiled GEMM (shared memory blocking)",
     "ex4_2_tiled_gemm.py"),
    ("Exercise 4.3 — WMMA GEMM (fp16 tensor cores)",
     "ex4_3_wmma_gemm.py"),
    ("Exercise 4.4 — smem Swizzle (bank conflict elimination)",
     "ex4_4_smem_swizzle.py"),
]

for title, script in exercises:
    banner(title)
    run_script(script)


# ── FINAL COMPARISON — all kernels inline ────────────────────────────────────

banner("FINAL COMPARISON — All GEMM Variants")

# Compile all kernels in one load_inline call for speed
cuda_all = r"""
#include <cuda_fp16.h>
#include <stdint.h>
#include <mma.h>
using namespace nvcuda;

// ── A: Naive GEMM (fp32) ──────────────────────────────────────────────────────
__global__ void cmp_naive(const float* A, const float* B, float* C,
                           int M, int N, int K) {
    int row = blockIdx.y*16 + threadIdx.y, col = blockIdx.x*16 + threadIdx.x;
    if (row>=M || col>=N) return;
    float s=0.f;
    for (int k=0; k<K; ++k) s += A[row*K+k] * B[k*N+col];
    C[row*N+col] = s;
}

// ── B: Tiled GEMM TILE=16 (fp32) ──────────────────────────────────────────────
__global__ void cmp_tiled16(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[16][16], Bs[16][16];
    int row=blockIdx.y*16+threadIdx.y, col=blockIdx.x*16+threadIdx.x;
    float s=0.f;
    for (int t=0; t<(K+15)/16; ++t) {
        int ac=t*16+threadIdx.x, br=t*16+threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row<M&&ac<K) ? A[row*K+ac] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (br<K&&col<N) ? B[br*N+col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k=0; k<16; ++k) s += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row<M&&col<N) C[row*N+col]=s;
}

// ── C: Tiled GEMM TILE=32 (fp32) ──────────────────────────────────────────────
__global__ void cmp_tiled32(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float As[32][32], Bs[32][32];
    int row=blockIdx.y*32+threadIdx.y, col=blockIdx.x*32+threadIdx.x;
    float s=0.f;
    for (int t=0; t<(K+31)/32; ++t) {
        int ac=t*32+threadIdx.x, br=t*32+threadIdx.y;
        As[threadIdx.y][threadIdx.x] = (row<M&&ac<K) ? A[row*K+ac] : 0.f;
        Bs[threadIdx.y][threadIdx.x] = (br<K&&col<N) ? B[br*N+col] : 0.f;
        __syncthreads();
        #pragma unroll
        for (int k=0; k<32; ++k) s += As[threadIdx.y][k]*Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row<M&&col<N) C[row*N+col]=s;
}

// ── D: WMMA v1 (fp16, global loads, one warp per 16×16 tile) ─────────────────
__global__ void cmp_wmma_v1(const __half* A, const __half* B, float* C,
                             int M, int N, int K) {
    int tr=blockIdx.y, tc=blockIdx.x;
    wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
    wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf, 0.f);
    int ro=tr*16, co=tc*16;
    for (int k=0; k<K; k+=16) {
        wmma::load_matrix_sync(af, A+ro*K+k, K);
        wmma::load_matrix_sync(bf, B+k*N+co, N);
        wmma::mma_sync(cf, af, bf, cf);
    }
    wmma::store_matrix_sync(C+ro*N+co, cf, N, wmma::mem_row_major);
}

// ── E: WMMA v2 (fp16, smem tiled, 64×64 per block) ───────────────────────────
const int BM=64, BN=64, BK=16;
__global__ void cmp_wmma_v2(const __half* A, const __half* B, float* C,
                             int M, int N, int K) {
    __shared__ __half As[BM][BK], Bs[BK][BN];
    int wid=threadIdx.x/32, wr=wid/4, wc=wid%4;
    int br=blockIdx.y*BM, bc=blockIdx.x*BN;
    wmma::fragment<wmma::accumulator,16,16,16,float> cf;
    wmma::fill_fragment(cf, 0.f);
    for (int ks=0; ks<K; ks+=BK) {
        for (int i=threadIdx.x; i<BM*BK; i+=blockDim.x) {
            int r=i/BK, c=i%BK;
            As[r][c] = (br+r<M&&ks+c<K) ? A[(br+r)*K+ks+c] : __float2half(0.f);
        }
        for (int i=threadIdx.x; i<BK*BN; i+=blockDim.x) {
            int r=i/BN, c=i%BN;
            Bs[r][c] = (ks+r<K&&bc+c<N) ? B[(ks+r)*N+bc+c] : __float2half(0.f);
        }
        __syncthreads();
        wmma::fragment<wmma::matrix_a,16,16,16,__half,wmma::row_major> af;
        wmma::fragment<wmma::matrix_b,16,16,16,__half,wmma::row_major> bf;
        wmma::load_matrix_sync(af, &As[wr*16][0], BK);
        wmma::load_matrix_sync(bf, &Bs[0][wc*16], BN);
        wmma::mma_sync(cf, af, bf, cf);
        __syncthreads();
    }
    int or_=br+wr*16, oc=bc+wc*16;
    if (or_<M&&oc<N)
        wmma::store_matrix_sync(C+or_*N+oc, cf, N, wmma::mem_row_major);
}

// ── Launchers ─────────────────────────────────────────────────────────────────

torch::Tensor la(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1);
    auto C=torch::zeros({M,N},A.options());
    cmp_naive<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(
        A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
    return C;
}
torch::Tensor lb(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1);
    auto C=torch::zeros({M,N},A.options());
    cmp_tiled16<<<dim3((N+15)/16,(M+15)/16),dim3(16,16)>>>(
        A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
    return C;
}
torch::Tensor lc(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1);
    auto C=torch::zeros({M,N},A.options());
    cmp_tiled32<<<dim3((N+31)/32,(M+31)/32),dim3(32,32)>>>(
        A.data_ptr<float>(),B.data_ptr<float>(),C.data_ptr<float>(),M,N,K);
    return C;
}
torch::Tensor ld_(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1);
    auto C=torch::zeros({M,N},torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    cmp_wmma_v1<<<dim3(N/16,M/16),32>>>(
        (const __half*)A.data_ptr(),(const __half*)B.data_ptr(),C.data_ptr<float>(),M,N,K);
    return C;
}
torch::Tensor le(torch::Tensor A, torch::Tensor B) {
    int M=A.size(0),K=A.size(1),N=B.size(1);
    auto C=torch::zeros({M,N},torch::TensorOptions().dtype(torch::kFloat32).device(A.device()));
    cmp_wmma_v2<<<dim3(N/BN,M/BM),512>>>(
        (const __half*)A.data_ptr(),(const __half*)B.data_ptr(),C.data_ptr<float>(),M,N,K);
    return C;
}
"""

cpp_all = """
torch::Tensor la(torch::Tensor A, torch::Tensor B);
torch::Tensor lb(torch::Tensor A, torch::Tensor B);
torch::Tensor lc(torch::Tensor A, torch::Tensor B);
torch::Tensor ld_(torch::Tensor A, torch::Tensor B);
torch::Tensor le(torch::Tensor A, torch::Tensor B);
"""

mod = load_inline(
    name="run_all_l4",
    cpp_sources=cpp_all,
    cuda_sources=cuda_all,
    functions=["la", "lb", "lc", "ld_", "le"],
    extra_cuda_cflags=["-O3", "-arch=sm_89", "--use_fast_math"],
    verbose=False,
)

# ── Correctness quick-check ───────────────────────────────────────────────────

print("Quick correctness check (M=N=K=1024):")
M = N = K = 1024
Af32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
Bf32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
Af16 = Af32.half()
Bf16 = Bf32.half()
ref32 = torch.mm(Af32, Bf32)
ref16 = torch.mm(Af16.float(), Bf16.float())

torch.testing.assert_close(mod.la(Af32, Bf32),  ref32, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(mod.lb(Af32, Bf32),  ref32, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(mod.lc(Af32, Bf32),  ref32, rtol=1e-4, atol=1e-4)
torch.testing.assert_close(mod.ld_(Af16, Bf16), ref16, rtol=1e-2, atol=1e-2)
torch.testing.assert_close(mod.le(Af16, Bf16),  ref16, rtol=1e-2, atol=1e-2)
print("  All 5 kernels: PASSED")
print()

# ── Final benchmark table ─────────────────────────────────────────────────────
#
# GEMM kernels: M=N=K=1024
#   flops = 2 * 1024^3 = 2.1 GFLOP
#   fp32 CUDA core peak ≈ 15 TFLOPS
#   fp16 tensor core peak ≈ 44 TFLOPS
#
# Matrix transpose (bandwidth test): N=2048
#   bytes = 2 * 2048^2 * 4 = 32 MB

print("FINAL COMPARISON — GEMM Variants (M=N=K=1024)")
print()

M = N = K = 1024
Af32 = torch.randn(M, K, device="cuda", dtype=torch.float32)
Bf32 = torch.randn(K, N, device="cuda", dtype=torch.float32)
Af16 = Af32.half()
Bf16 = Bf32.half()
flops = 2 * M * N * K

results = []

# A: Naive GEMM (fp32) — use fewer iters, it's slow
ms = bench(lambda: mod.la(Af32, Bf32), warmup=3, iters=10)
tflops = flops / ms / 1e9
results.append(("Naive GEMM fp32   (16×16 block)", "fp32", ms, tflops,
                "global mem, no reuse"))

# B: Tiled TILE=16 (fp32)
ms = bench(lambda: mod.lb(Af32, Bf32), warmup=5, iters=100)
tflops = flops / ms / 1e9
results.append(("Tiled fp32 T=16   (256 threads)", "fp32", ms, tflops,
                "2KB smem, 16× reuse"))

# C: Tiled TILE=32 (fp32)
ms = bench(lambda: mod.lc(Af32, Bf32), warmup=5, iters=100)
tflops = flops / ms / 1e9
results.append(("Tiled fp32 T=32   (1024 threads)", "fp32", ms, tflops,
                "8KB smem, 32× reuse"))

# D: WMMA v1 (fp16, global)
ms = bench(lambda: mod.ld_(Af16, Bf16), warmup=10, iters=200)
tflops = flops / ms / 1e9
results.append(("WMMA v1 fp16  (global loads)", "fp16", ms, tflops,
                "tensor cores, no smem"))

# E: WMMA v2 (fp16, smem)
ms = bench(lambda: mod.le(Af16, Bf16), warmup=10, iters=200)
tflops = flops / ms / 1e9
results.append(("WMMA v2 fp16  (smem 64×64)", "fp16", ms, tflops,
                "tensor cores + smem"))

# F: torch.mm fp32 reference
ms = bench(lambda: torch.mm(Af32, Bf32), warmup=10, iters=300)
tflops = flops / ms / 1e9
results.append(("torch.mm fp32     (cuBLAS)", "fp32", ms, tflops,
                "cuBLAS reference"))

# G: torch.mm fp16 reference
ms = bench(lambda: torch.mm(Af16, Bf16), warmup=10, iters=300)
tflops = flops / ms / 1e9
results.append(("torch.mm fp16     (cuBLAS)", "fp16", ms, tflops,
                "cuBLAS reference"))

# Print table
hdr = f"  {'Kernel':<37}  {'dtype':>5}  {'ms':>6}  {'TFLOPS':>7}  {'% peak':>7}  {'Notes'}"
print(hdr)
print("  " + "-" * (len(hdr) - 2))
for name, dtype, ms, tflops, notes in results:
    pct = tflops / PEAK_TFLOPS * 100
    mark = "✓" if pct > 50 else ("~" if pct > 20 else ("·" if pct > 5 else "✗"))
    print(f"  {name:<37}  {dtype:>5}  {ms:>6.3f}  {tflops:>7.2f}  {pct:>6.1f}%  {mark}  {notes}")

print()
print(f"  Peak fp16 tensor cores: {PEAK_TFLOPS} TFLOPS | Peak GDDR6X: {PEAK_BW} GB/s")
print(f"  RTX 4060 Ti — Ada Lovelace — sm_89")
print()

# ── Matrix transpose comparison ───────────────────────────────────────────────

print("─" * 70)
print()

# Compile transpose variants inline
cuda_tp = r"""
#define TILE_DIM 32
#define BLOCK_ROWS 8
__global__ void tp_naive(float* o, const float* i, int N) {
    __shared__ float t[TILE_DIM][TILE_DIM];
    int x=blockIdx.x*TILE_DIM+threadIdx.x, y=blockIdx.y*TILE_DIM+threadIdx.y;
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) if ((y+j)<N&&x<N) t[threadIdx.y+j][threadIdx.x]=i[(y+j)*N+x];
    __syncthreads();
    x=blockIdx.y*TILE_DIM+threadIdx.x; y=blockIdx.x*TILE_DIM+threadIdx.y;
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) if ((y+j)<N&&x<N) o[(y+j)*N+x]=t[threadIdx.x][threadIdx.y+j];
}
__global__ void tp_swiz(float* o, const float* i, int N) {
    __shared__ float t[TILE_DIM][TILE_DIM];
    int x=blockIdx.x*TILE_DIM+threadIdx.x, y=blockIdx.y*TILE_DIM+threadIdx.y;
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) if ((y+j)<N&&x<N) t[threadIdx.y+j][threadIdx.x^(threadIdx.y+j)]=i[(y+j)*N+x];
    __syncthreads();
    x=blockIdx.y*TILE_DIM+threadIdx.x; y=blockIdx.x*TILE_DIM+threadIdx.y;
    for (int j=0; j<TILE_DIM; j+=BLOCK_ROWS) if ((y+j)<N&&x<N) o[(y+j)*N+x]=t[threadIdx.x][(threadIdx.y+j)^threadIdx.x];
}
torch::Tensor run_tp_naive(torch::Tensor x) {
    int N=x.size(0); auto o=torch::empty_like(x);
    dim3 g(N/TILE_DIM,N/TILE_DIM), b(TILE_DIM,BLOCK_ROWS);
    tp_naive<<<g,b>>>(o.data_ptr<float>(),x.data_ptr<float>(),N); return o;
}
torch::Tensor run_tp_swiz(torch::Tensor x) {
    int N=x.size(0); auto o=torch::empty_like(x);
    dim3 g(N/TILE_DIM,N/TILE_DIM), b(TILE_DIM,BLOCK_ROWS);
    tp_swiz<<<g,b>>>(o.data_ptr<float>(),x.data_ptr<float>(),N); return o;
}
"""
mod_tp = load_inline(
    name="run_all_l4_tp",
    cpp_sources="torch::Tensor run_tp_naive(torch::Tensor); torch::Tensor run_tp_swiz(torch::Tensor);",
    cuda_sources=cuda_tp,
    functions=["run_tp_naive", "run_tp_swiz"],
    extra_cuda_cflags=["-O3", "-arch=sm_89"],
    verbose=False,
)

print("BONUS: Matrix Transpose Bandwidth (N=2048, float32)")
print()
N_tp = 2048
x_tp = torch.randn(N_tp, N_tp, device="cuda", dtype=torch.float32)
bw_bytes = 2 * N_tp * N_tp * 4  # read + write

tp_results = []
for name, fn in [
    ("Naive (32-way conflict)", lambda: mod_tp.run_tp_naive(x_tp)),
    ("XOR swizzle (no conflict)", lambda: mod_tp.run_tp_swiz(x_tp)),
    ("torch.T (PyTorch ref)", lambda: x_tp.T.contiguous()),
]:
    ms = bench(fn, warmup=20, iters=500)
    bw = bw_bytes / ms / 1e6
    util = bw / PEAK_BW * 100
    tp_results.append((name, ms, bw, util))

hdr_tp = f"  {'Kernel':<30}  {'ms':>7}  {'GB/s':>7}  {'Util':>7}"
print(hdr_tp)
print("  " + "-" * (len(hdr_tp) - 2))
for name, ms, bw, util in tp_results:
    mark = "✓" if util > 65 else ("~" if util > 40 else "✗")
    print(f"  {name:<30}  {ms:>7.3f}  {bw:>7.1f}  {util:>6.1f}%  {mark}")

print()
print(f"  Peak: {PEAK_BW} GB/s | Target: >70% (>201 GB/s)")
print()

# ── Key insights ──────────────────────────────────────────────────────────────
print("─" * 70)
print()
print("Key insights from Lesson 4:")
print()
print("  1. Naive GEMM:   <1% of fp16 peak — memory bandwidth is the ceiling")
print("     Not because the GPU is slow, but because L2 cache thrashes with")
print("     O(M*N*K) bytes accessed. The arithmetic intensity is too low.")
print()
print("  2. Tiled GEMM:   5–20% of fp32 CUDA core peak (via register reuse)")
print("     TILE× reduction in HBM traffic. All 8 KB smem reused in inner loop.")
print("     Zero bank conflicts with 2D block + row-read pattern.")
print()
print("  3. WMMA v1:      15–30% — tensor cores active but bandwidth-limited")
print("     Without smem, N/16 warps each reload the same B columns → HBM bound.")
print()
print("  4. WMMA v2:      25–50% — tensor cores + smem tiling")
print("     64×64 tiles loaded once into smem, 16 warps share each tile.")
print("     This IS the structure of FlashAttention's QK and PV matrix products.")
print()
print("  5. Swizzle:      eliminates 32-way bank conflict → 2–10× speedup")
print("     Padded and XOR-swizzled are equally fast, XOR uses no extra memory.")
print("     Same principle applies to WMMA smem loads (see permuted_smem.cuh).")
print()
print("  6. torch.mm:     ~50–90% of peak via CUTLASS (highly optimized)")
print("     Uses: double-buffered smem, register blocking, LDMATRIX,")
print("     persistent thread blocks, and architecture-specific tuning.")
print("     Your WMMA v2 is 2–4× from cuBLAS — the remaining gap is:")
print("       a) No double-buffering (cp.async pipeline, Lesson 6)")
print("       b) No register blocking (each thread accumulates a micro-tile)")
print("       c) No smem swizzle for WMMA loads (permuted_smem.cuh, Lesson 5)")
print()
print("  Next lessons build on WMMA v2:")
print("    Lesson 5 (Flash Prefill): QK^T and P*V are both WMMA v2 + causal mask")
print("    Lesson 6 (Flash Decode):  online softmax wraps around the WMMA compute")
