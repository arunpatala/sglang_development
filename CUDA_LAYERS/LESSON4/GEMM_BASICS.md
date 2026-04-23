# GEMM Basics

Reference companion for Lesson 4. Keep this open alongside `mma.cuh` and the SGLang FFN code while reading.

---

## Why GEMM Dominates LLM Inference

Every linear projection in a transformer is a GEMM:

```
Q = hidden @ W_Q^T     [batch*seq, hidden] × [hidden, head_dim*n_heads]
K = hidden @ W_K^T     same shape
V = hidden @ W_V^T     same shape
O = attn_out @ W_O^T   [batch*seq, hidden] × [hidden, hidden]

gate = hidden @ W_gate  [batch*seq, hidden] × [hidden, ffn_dim]
up   = hidden @ W_up    [batch*seq, hidden] × [hidden, ffn_dim]
down = mlp_out @ W_down [batch*seq, ffn_dim] × [ffn_dim, hidden]

logits = hidden @ lm_head  [batch*seq, hidden] × [hidden, vocab_size]
```

For Llama 3 8B (hidden=4096, ffn=14336, heads=32, head_dim=128, vocab=128k):

| Operation | M | N | K | FLOPs (batch=1, seq=1) |
|---|---|---|---|---|
| Q/K/V projection | 1 | 4096 | 4096 | 25 MFLOP each |
| O projection | 1 | 4096 | 4096 | 25 MFLOP |
| gate/up proj | 1 | 14336 | 4096 | 117 MFLOP each |
| down proj | 1 | 4096 | 14336 | 117 MFLOP |
| lm_head | 1 | 128k | 4096 | 1 GFLOP |

**During prefill (seq_len=1024):** M=1024, GEMM dominates — tensor-core bound.
**During decode (seq_len=1):** M=1, tiny GEMM — KV cache read dominates (memory-bound).

---

## Problem Setup

**GEMM:** C[M, N] = A[M, K] × B[K, N]

```
A: [M, K]  — left matrix (e.g., token hidden states)
B: [K, N]  — right matrix (e.g., weight matrix, stored transposed)
C: [M, N]  — output (e.g., projected hidden states)

C[i][j] = sum over k of A[i][k] * B[k][j]   (2*K FLOPs per output element)
Total FLOPs = 2 * M * N * K
```

For M=N=K=4096: `2 * 4096³ ≈ 137 GFLOP` per matrix multiply.

---

## Naive Implementation: One Thread Per Output Element

```
Grid:  (ceil(N/16), ceil(M/16))   — each block covers 16×16 output elements
Block: (16, 16) = 256 threads
Thread (tx, ty): computes C[blockIdx.y*16+ty][blockIdx.x*16+tx]
```

```cuda
__global__ void naive_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M || col >= N) return;

    float sum = 0.f;
    for (int k = 0; k < K; ++k)
        sum += A[row * K + k] * B[k * N + col];
    C[row * N + col] = sum;
}
```

### Why Naive GEMM Is Slow

For a single output element C[i][j], the kernel reads:
- Row i of A:    K elements × sizeof(float) = 4K bytes
- Column j of B: K elements × sizeof(float) = 4K bytes (non-contiguous — cache-unfriendly!)

For the full output C[M][N]:
- Reads of A: M × N × K × 4 bytes (each A element read N times by different threads)
- Reads of B: M × N × K × 4 bytes (each B element read M times)
- Total: 2 × M × N × K × 4 bytes (ignoring cache)

For M=N=K=1024: `2 × 1024³ × 4 = 8 GB` of reads, but L2 is only 32 MB!

At 288 GB/s peak: 8 GB / 288 GB/s = 28ms just for memory, assuming perfect bandwidth utilization.
In practice: the GPU doesn't have time to hide all that latency → **stall-dominated**.

### Access Pattern Problems

```
Thread 0 reading column 0 of B: B[0][0], B[1][0], B[2][0], B[3][0]...
  Stride = N floats = N*4 bytes between accesses
  For N=1024: stride = 4096 bytes = 64 cache lines apart
  → Every B access is a cache miss → L2 thrash

Thread 1 reading column 1 of B: B[0][1], B[1][1], ...
  Same problem, different column

All threads in a warp reading different columns of the SAME row of B:
  B[k][col], B[k][col+1], ..., B[k][col+31]  → coalesced! ✓
  But B[k+1][col], B[k+1][col+1]... → coalesced ✓ on each k iteration
  → B reads are coalesced; it's A that may cause issues for certain thread mappings
```

**Key insight:** The naive kernel works, but each element of A is read by N threads and each element of B is read by M threads — with no reuse from caches for large matrices.

---

## Tiled GEMM: Shared Memory Blocking

Divide the computation into square tiles of size TILE×TILE:

```
For each tile index t in range(ceil(K / TILE)):
    1. All threads in the block cooperatively load A[block_row..+TILE, t*TILE..+TILE] into smem
    2. All threads cooperatively load B[t*TILE..+TILE, block_col..+TILE] into smem
    3. __syncthreads() — all smem loads complete
    4. Each thread computes its partial sum from smem: sum += As[ty][k] * Bs[k][tx]
    5. __syncthreads() — prevent overwriting smem before all threads finish reading
```

```cuda
const int TILE = 32;

__global__ void tiled_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float sum = 0.f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int a_col = t * TILE + threadIdx.x;
        As[threadIdx.y][threadIdx.x] = (row < M && a_col < K) ? A[row * K + a_col] : 0.f;
        int b_row = t * TILE + threadIdx.y;
        Bs[threadIdx.y][threadIdx.x] = (b_row < K && col < N) ? B[b_row * N + col] : 0.f;
        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        __syncthreads();
    }
    if (row < M && col < N) C[row * N + col] = sum;
}
```

### Memory Reduction from Tiling

Without tiling:
- Each element of A is read by N threads → N reads total per element
- Each element of B is read by M threads → M reads total per element

With tiling (TILE×TILE blocks):
- Each element of A is loaded once per tile iteration that uses its row segment → ceil(N/TILE) times
- Each element of B is loaded once per tile → ceil(M/TILE) times
- Total HBM reads = M×K × (N/TILE)/N + K×N × (M/TILE)/M... 

Actually, more precisely:
- Each A element is read exactly `ceil(N/TILE)` times across all blocks in its row
- But from the HBM perspective: each tile of A is loaded once from HBM, then reused TILE times in registers
- **Reuse factor = TILE**: each element loaded into smem is used TILE times for the dot product

For TILE=32: 32× reduction in HBM bandwidth requirement vs naive (in theory).

### smem Bank Conflicts in Tiled GEMM

During the compute phase, threads read from As and Bs:

```
As[threadIdx.y][k]: all threads in a warp (same ty, k fixed) read the SAME address
  → hardware broadcast → ZERO conflict

Bs[k][threadIdx.x]: all threads in a warp (different tx, k fixed) read consecutive addresses
  → sequential access → ZERO conflict
```

**Tiled GEMM with square 2D blocks has zero bank conflicts.** ✓

The bank conflict problem appears in:
1. GEMM with 1D thread blocks (non-standard layout)
2. WMMA fragment loads from smem with certain tile sizes
3. Matrix transpose (a clean demonstration — see SMEM_SWIZZLE.md)

---

## FLOP Counting and Roofline

### Computing TFLOPS

```python
elapsed_ms = ...   # measured kernel time
flops = 2 * M * N * K
tflops = flops / elapsed_ms / 1e9   # TFLOPS (note: 1e9 because ms, not s)
# Check: 2 * 1024^3 / 1ms / 1e9 = 2.1 TFLOPS
```

### Hardware Ceilings (RTX 4060 Ti, sm_89)

| Computation Mode | Peak TFLOPS |
|---|---|
| FP32 (CUDA cores, scalar FMA) | ~15 TFLOPS |
| FP16 (CUDA cores, scalar FMA) | ~30 TFLOPS |
| FP16 + BF16 (4th-gen tensor cores) | ~44 TFLOPS |
| INT8 (tensor cores) | ~88 TOPS |
| FP8 (tensor cores) | ~176 TOPS |

Memory bandwidth: 288 GB/s (GDDR6X)
Ridge point: 44 TFLOPS / 288 GB/s = 153 FLOP/byte

### Arithmetic Intensity of GEMM

For C = A @ B with A[M,K], B[K,N], C[M,N]:
- FLOPs: 2 * M * N * K
- Bytes (ideal, no reuse): (M*K + K*N + M*N) * sizeof(T)

For M=N=K=4096, fp16 (2 bytes):
- FLOPs: 2 * 4096³ ≈ 137 GFLOP
- Bytes: 3 * 4096² * 2 ≈ 96 MB
- Arithmetic intensity: 137,000 MFLOP / 96 MB ≈ **1,427 FLOP/byte**

1,427 >> 153 (ridge point) → **GEMM is compute-bound**, not bandwidth-bound.

This is the fundamental reason GEMM needs tensor cores: the bottleneck is compute throughput, not memory bandwidth.

---

## Kernel Performance Summary

| Kernel | TFLOPS at M=N=K=1024 | % of fp16 Peak | Bottleneck |
|---|---|---|---|
| Naive (fp32) | 0.01–0.5 | <1% | L2 cache thrash, memory latency |
| Tiled TILE=16 (fp32) | 1–5 | — | Register usage, warp divergence |
| Tiled TILE=32 (fp32) | 3–8 | — | Reaching fp32 CUDA core compute |
| WMMA v1 (fp16, global loads) | 5–15 | 11–34% | Memory bandwidth (no smem reuse) |
| WMMA v2 (fp16, smem tiled) | 15–30 | 34–68% | Tensor core utilization |
| cuBLAS / torch.mm (fp16) | 25–40 | 57–91% | Near peak, highly optimized |

**Key progression:** naive → tiled shows the value of smem reuse. tiled → WMMA shows the value of tensor cores.

---

## How This Connects to Production Code

### In SGLang / FlashInfer

All GEMM operations in SGLang go through one of:
1. `torch.nn.Linear` → dispatches to cuBLAS GEMM internally
2. Custom CUTLASS kernels for quantized GEMM (FP8, INT8, AWQ, GPTQ)
3. Triton GEMM kernels (for AMD compatibility)

The tiled+WMMA pattern you build in this lesson IS what CUTLASS does — just with more engineering:
- Larger tiles (128×128×32 or 256×128×32)
- Double-buffered smem (prefetch next tile while computing current)
- Register-level blocking (each thread accumulates a small register tile)
- Software pipelining with `cp.async`

### Reading CUTLASS After This Lesson

After implementing tiled WMMA, you can read:
- `REPOS/flashinfer/include/flashinfer/mma.cuh` — the MMA instruction wrappers
- `REPOS/flashinfer/include/flashinfer/permuted_smem.cuh` — the swizzled smem layout
- `REPOS/sglang/sgl-kernel/csrc/attention/` — attention uses WMMA for QK and PV products

### The WMMA → MMA Transition

`nvcuda::wmma` (this lesson) is a high-level API that maps to PTX `wmma` instructions.
Production kernels use the lower-level PTX `mma.sync` directly via inline PTX or CUTLASS `cute::MMA`.

```ptx
# WMMA API generates this in PTX (visible with nvcc --ptx):
wmma.load.a.sync.aligned.row.m16n16k16.global.f16 {...}, [ptr], lda
wmma.mma.sync.aligned.row.col.m16n16k16.f32.f16.f16.f32 {...}, {...}, {...}, {...}
wmma.store.d.sync.aligned.row.m16n16k16.global.f32 [ptr], {...}, ldc

# The actual SM89 tensor core instruction in SASS:
HMMA.16816.F32 Rd, Ra, Rb, Rc   ← executes in 1 warp-instruction
```

---

## Quick Reference

| Concept | Value |
|---|---|
| GEMM FLOPs | 2 * M * N * K |
| Tiling benefit | TILE× reduction in HBM reads |
| WMMA tile size (fp16) | 16×16×16 (A: 16×16, B: 16×16, C: 16×16) |
| WMMA fragment | opaque 32-thread register array, 8 values/thread |
| Tensor core speedup vs scalar | ~8× for fp16 |
| Target TFLOPS (this lesson) | >22 TFLOPS = >50% of 44 TFLOPS peak |
| smem for 64×64 fp16 tile | 64×64×2 bytes = 8 KB |
| Bank conflicts in tiled GEMM | Zero (2D block layout: row reads = broadcast or sequential) |
