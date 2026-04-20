/*
 * Exercise 1.2 — PTX Inspection: Scalar Kernel
 * ==============================================
 * Compile this with --ptx to see the 16-bit load instruction:
 *
 *   nvcc --ptx -arch=sm_89 -O3 ex1_2_scalar_kernel.cu -o ex1_2_scalar.ptx
 *   grep "ld.global" ex1_2_scalar.ptx
 *
 * You will see:
 *   ld.global.cs.u16 %rs1, [%rd4];    <-- 16-bit load
 *
 * Then compile ex1_2_vec_kernel.cu and compare:
 *   nvcc --ptx -arch=sm_89 -O3 ex1_2_vec_kernel.cu -o ex1_2_vec.ptx
 *   grep "ld.global" ex1_2_vec.ptx
 *
 * You will see:
 *   ld.global.ca.v4.u32  {%r1,%r2,%r3,%r4}, [%rd4];    <-- 128-bit load
 */

#include <cuda_fp16.h>
#include <stdint.h>

// SCALAR version: generates LD.E.16 (16-bit load)
__global__ void copy_scalar_ptx(
    const __half* __restrict__ src,
    __half*       __restrict__ dst,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        dst[idx] = src[idx];   // <--- this is the 16-bit load
    }
}
