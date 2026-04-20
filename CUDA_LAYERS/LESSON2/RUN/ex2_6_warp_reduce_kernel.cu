/*
 * Exercise 2.6 — PTX Inspection: Warp Shuffle Reduce
 * ====================================================
 * Compile this with --ptx to see shfl.sync.bfly.b32 instructions:
 *
 *   nvcc --ptx -arch=sm_89 -O3 ex2_6_warp_reduce_kernel.cu -o ex2_6_warp_reduce.ptx
 *   grep "shfl.sync" ex2_6_warp_reduce.ptx
 *
 * You will see exactly 5 lines — one per butterfly round:
 *
 *   shfl.sync.bfly.b32  %f2, %f1, 16, 31, -1;   // round 1: XOR 16
 *   shfl.sync.bfly.b32  %f3, %f2, 8,  31, -1;   // round 2: XOR 8
 *   shfl.sync.bfly.b32  %f4, %f3, 4,  31, -1;   // round 3: XOR 4
 *   shfl.sync.bfly.b32  %f5, %f4, 2,  31, -1;   // round 4: XOR 2
 *   shfl.sync.bfly.b32  %f6, %f5, 1,  31, -1;   // round 5: XOR 1
 *
 * PTX field breakdown:
 *   shfl.sync  — synchronized shuffle (Volta+, sm_70+)
 *   .bfly      — butterfly mode (XOR partner)
 *   .b32       — 32-bit value exchanged (one float register)
 *   16         — the XOR mask applied to the lane ID
 *   31         — clamp = warp_size - 1 (prevents out-of-warp reads)
 *   -1         — participation mask = 0xffffffff (all 32 lanes active)
 *
 * Compare to the scalar smem reduce (ex2_6_scalar_reduce_kernel.cu):
 *   The smem version generates ld.shared and st.shared instructions.
 *   The shuffle version generates shfl.sync — zero memory, one clock.
 */

// Warp reduce sum — generates exactly 5 shfl.sync.bfly.b32 instructions
__device__ __forceinline__ float warp_reduce_sum(float val) {
    val += __shfl_xor_sync(0xffffffff, val, 16);
    val += __shfl_xor_sync(0xffffffff, val,  8);
    val += __shfl_xor_sync(0xffffffff, val,  4);
    val += __shfl_xor_sync(0xffffffff, val,  2);
    val += __shfl_xor_sync(0xffffffff, val,  1);
    return val;
}

// Warp reduce max — same butterfly, fmax instruction instead of fadd
__device__ __forceinline__ float warp_reduce_max(float val) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, 16));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  8));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  4));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  2));
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val,  1));
    return val;
}

// Entry point: one warp per block, reduce 32 floats
__global__ void reduce_warp_sum_ptx(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    int lane  = threadIdx.x;
    float val = src[blockIdx.x * 32 + lane];
    float tot = warp_reduce_sum(val);
    if (lane == 0) out[blockIdx.x] = tot;
}

__global__ void reduce_warp_max_ptx(
    const float* __restrict__ src,
    float*       __restrict__ out)
{
    int lane  = threadIdx.x;
    float val = src[blockIdx.x * 32 + lane];
    float mx  = warp_reduce_max(val);
    if (lane == 0) out[blockIdx.x] = mx;
}
