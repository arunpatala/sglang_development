/*
 * Exercise 1.2 — PTX Inspection: Vectorized Kernel
 * ==================================================
 * Compile with --ptx to see the 128-bit load instruction:
 *
 *   nvcc --ptx -arch=sm_89 -O3 ex1_2_vec_kernel.cu -o ex1_2_vec.ptx
 *   grep "ld.global" ex1_2_vec.ptx
 *
 * You will see something like:
 *   ld.global.ca.v4.u32  {%r1,%r2,%r3,%r4}, [%rd4];    <-- 128-bit load
 *     - .ca   = cache at L1 and L2
 *     - .v4   = 4 × 32-bit words = 128 bits total
 *     - u32   = unsigned 32-bit (holds 2 fp16 values each = 8 fp16 total)
 *
 * Compare to the scalar .ptx: one instruction moves 8× more data.
 */

#include <cuda_fp16.h>
#include <stdint.h>

// VECTORIZED version: generates LD.E.128 (128-bit load)
__global__ void copy_vec8_ptx(
    const uint4* __restrict__ src,
    uint4*       __restrict__ dst,
    int n_vec)   // n_vec = n_fp16_elements / 8
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_vec) {
        dst[idx] = src[idx];   // <--- this is the 128-bit load
    }
}
