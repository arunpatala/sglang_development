/*
 * Exercise 2.6 — PTX Inspection: Scalar Shared-Memory Reduce (for comparison)
 * =============================================================================
 * Compile with --ptx and compare against ex2_6_warp_reduce_kernel.cu:
 *
 *   nvcc --ptx -arch=sm_89 -O3 ex2_6_scalar_reduce_kernel.cu -o ex2_6_scalar.ptx
 *   grep "ld.shared\|st.shared" ex2_6_scalar.ptx
 *
 * You will see st.shared and ld.shared instructions — shared memory reads
 * and writes. These do NOT appear in the shuffle version.
 *
 * Then compare instruction count:
 *   grep "shfl.sync"  ex2_6_warp_reduce.ptx  | wc -l   → 5
 *   grep "ld.shared"  ex2_6_scalar.ptx        | wc -l   → 32  (serial loop)
 *   grep "st.shared"  ex2_6_scalar.ptx        | wc -l   → 1   (smem write)
 *
 * The warp shuffle version has:
 *   - 5 shfl.sync instructions (vs 33 smem ops in the naive version)
 *   - Zero shared memory allocation
 *   - No __syncwarp() or __syncthreads() required
 */

// Naive serial reduce using shared memory (baseline for comparison)
__global__ void reduce_scalar_smem(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    __shared__ float smem[32];
    int tid   = threadIdx.x;
    smem[tid] = src[blockIdx.x * 32 + tid];
    __syncthreads();

    // Thread 0 does all the work — 31 other threads sit idle
    if (tid == 0) {
        float total = 0.f;
        for (int i = 0; i < 32; i++) {
            total += smem[i];   // 32 ld.shared instructions in the PTX
        }
        out[blockIdx.x] = total;
    }
}
