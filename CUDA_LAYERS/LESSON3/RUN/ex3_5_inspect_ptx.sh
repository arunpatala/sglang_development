#!/usr/bin/env bash
# ex3_5_inspect_ptx.sh — PTX Inspection and Bank Conflict Profiling
# ==================================================================
# Inspects the PTX output of the block reduce and RMSNorm kernels.
# Also profiles bank conflicts and memory throughput with Nsight Compute.
#
# Usage:
#   bash ex3_5_inspect_ptx.sh           # run everything
#   bash ex3_5_inspect_ptx.sh ptx       # PTX analysis only
#   bash ex3_5_inspect_ptx.sh conflicts # bank conflict ncu metrics only
#   bash ex3_5_inspect_ptx.sh rmsnorm   # RMSNorm bandwidth only
#   bash ex3_5_inspect_ptx.sh fused     # fused add+rmsnorm only
#
# Requires: ncu (Nsight Compute CLI) from CUDA Toolkit

set -e
MODE=${1:-all}

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
    echo ""
}

# ─── PTX Analysis ─────────────────────────────────────────────────────────────
if [ "$MODE" = "ptx" ] || [ "$MODE" = "all" ]; then
    header "PTX: shared memory instructions in block_reduce_sum_2d"

    # Write a minimal kernel for PTX extraction
    cat > /tmp/ex3_probe_kernel.cu << 'EOF'
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

__device__ float block_reduce_sum_2d(float val, float* smem) {
    int lane    = threadIdx.x;
    int warp_id = threadIdx.y;
    int n_warps = blockDim.y;

    val = warp_reduce_sum(val);

    if (lane == 0) smem[warp_id] = val;   // st.shared
    __syncthreads();                       // bar.sync

    val = (warp_id == 0 && lane < n_warps) ? smem[lane] : 0.f;  // ld.shared
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
        if (lane == 0) smem[0] = val;     // st.shared
    }
    __syncthreads();                       // bar.sync
    return smem[0];                        // ld.shared
}

__global__ void probe_block_reduce(float* x, float* out) {
    extern __shared__ float smem[];
    float val = x[threadIdx.y * 32 + threadIdx.x];
    float total = block_reduce_sum_2d(val, smem);
    if (threadIdx.x == 0 && threadIdx.y == 0) out[blockIdx.x] = total;
}
EOF

    echo "Compiling probe kernel to PTX..."
    nvcc --ptx -arch=sm_89 -O3 /tmp/ex3_probe_kernel.cu -o /tmp/ex3_probe.ptx 2>&1
    echo ""

    echo "PTX instructions (smem operations):"
    echo ""

    echo "--- shfl.sync (warp shuffles, Lesson 2 butterfly) ---"
    grep -c "shfl.sync" /tmp/ex3_probe.ptx && \
        echo "  Expected: 10 (5 per warp_reduce_sum call × 2 calls)" || true

    echo ""
    echo "--- ld.shared (shared memory loads) ---"
    grep -n "ld.shared" /tmp/ex3_probe.ptx | head -10
    grep -c "ld.shared" /tmp/ex3_probe.ptx && \
        echo "  Expected: ~2 (smem[lane] read + smem[0] read)" || true

    echo ""
    echo "--- st.shared (shared memory stores) ---"
    grep -n "st.shared" /tmp/ex3_probe.ptx | head -10
    grep -c "st.shared" /tmp/ex3_probe.ptx && \
        echo "  Expected: ~2 (smem[warp_id] write + smem[0] write)" || true

    echo ""
    echo "--- bar.sync (__syncthreads) ---"
    grep -c "bar.sync" /tmp/ex3_probe.ptx && \
        echo "  Expected: 2 (one per __syncthreads() call)" || true

    echo ""
    echo "--- rsqrt (MUFU.RSQ, appears in rmsnorm kernel) ---"
    echo "  To see rsqrt: compile rmsnorm kernel and grep for 'rsqrt.approx'"

    echo ""
    echo "Full PTX saved to: /tmp/ex3_probe.ptx"
    echo "Open it to see the complete instruction sequence."
fi

# ─── Bank Conflict Profiling ───────────────────────────────────────────────────
if [ "$MODE" = "conflicts" ] || [ "$MODE" = "all" ]; then
    header "Bank Conflicts: no-conflict vs stride-32 (32-way conflict)"

    METRICS_CONFLICT="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

    for KERNEL in smem_no_conflict smem_with_conflict smem_stride16_conflict; do
        echo "--- $KERNEL ---"
        ncu \
            --kernel-name "$KERNEL" \
            --metrics "$METRICS_CONFLICT" \
            --target-processes all \
            python ex3_1_bank_conflicts.py 2>&1 \
            | grep -E "($KERNEL|bank_conflict|sectors_shared|throughput)" \
            | head -8
        echo ""
    done

    echo "What to look for:"
    echo "  smem_no_conflict:         bank_conflicts_ld = 0"
    echo "  smem_stride16_conflict:   bank_conflicts_ld = 1 per warp (2-way)"
    echo "  smem_with_conflict:       bank_conflicts_ld = 31 per warp (32-way)"
fi

# ─── RMSNorm Bandwidth ────────────────────────────────────────────────────────
if [ "$MODE" = "rmsnorm" ] || [ "$MODE" = "all" ]; then
    header "RMSNorm fp16 — bandwidth, occupancy, bank conflicts"

    METRICS_RMSNORM="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

    ncu \
        --kernel-name "rmsnorm_f16" \
        --metrics "$METRICS_RMSNORM" \
        --target-processes all \
        python ex3_3_rmsnorm_f16.py 2>&1 \
        | grep -E "(rmsnorm_f16|dram|bytes|warps_active|bank_conflict|throughput)" \
        | head -20

    echo ""
    echo "Expected:"
    echo "  dram_throughput:  >70% (>201 GB/s on RTX 4060 Ti)"
    echo "  bank_conflicts:   0    (smem[warp_id] pattern)"
    echo "  global_load_bytes ≈ 2 × batch × hidden × 2  (read x twice + read weight)"
    echo "  global_store_bytes ≈ batch × hidden × 2      (write output)"
fi

# ─── Fused vs Unfused ─────────────────────────────────────────────────────────
if [ "$MODE" = "fused" ] || [ "$MODE" = "all" ]; then
    header "Fused Add+RMSNorm — compare DRAM reads vs unfused"

    METRICS_FUSED="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

    echo "--- fused_add_rmsnorm_f16 ---"
    ncu \
        --kernel-name "fused_add_rmsnorm_f16" \
        --metrics "$METRICS_FUSED" \
        --target-processes all \
        python ex3_4_fused_add_rmsnorm.py 2>&1 \
        | grep -E "(fused_add|dram|bytes|throughput)" \
        | head -12

    echo ""
    echo "Expected for fused kernel (batch=4096, hidden=4096):"
    echo "  global_load_bytes  ≈ 4×4096×4096×2 = 128 MB  (inp + res + weight = 3 reads,"
    echo "                                                  NOT 4 — x is read once only)"
    echo "  shared_load_bytes  ≈ 4×4096×4096×2 = 32 MB   (x read from smem in pass 2)"
    echo "  global_store_bytes ≈ 4×4096×4096×2 = 64 MB   (residual update + output)"
    echo ""
    echo "Compare to unfused rmsnorm: global_load_bytes would be ~4 reads (x twice + weight)"
fi

echo ""
echo "========================================================"
echo "  Summary of what to look for:"
echo ""
echo "  1. PTX instruction counts for block_reduce_sum_2d:"
echo "       shfl.sync: 10 (5 per warp_reduce × 2 calls)"
echo "       ld.shared:  2 (smem[lane] + smem[0])"
echo "       st.shared:  2 (smem[wid]  + smem[0])"
echo "       bar.sync:   2 (__syncthreads × 2)"
echo ""
echo "  2. Bank conflicts:"
echo "       smem_no_conflict → 0 conflicts"
echo "       smem_with_conflict → 31 conflicts per warp (32-way serialization)"
echo "       rmsnorm_f16 → 0 conflicts (smem[warp_id] is the correct pattern)"
echo ""
echo "  3. RMSNorm DRAM throughput target: >70% of 288 GB/s = >201 GB/s"
echo ""
echo "  4. Fused kernel global loads: 3 tensor reads (not 4)"
echo "       Pass 2 reads x from smem → saves one GDDR6X read"
echo ""
echo "Full report (open in ncu-ui):"
echo "  ncu --set full -o lesson3_rmsnorm python ex3_3_rmsnorm_f16.py"
echo "  ncu-ui lesson3_rmsnorm.ncu-rep"
echo "========================================================"
