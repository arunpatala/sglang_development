#!/usr/bin/env bash
# profile.sh — Nsight Compute Profiling for Lesson 3 Kernels
# ===========================================================
# Profiles all Lesson 3 kernels and prints a comparison of:
#   - Bank conflict count (should be 0 for correct smem patterns)
#   - HBM bandwidth utilization (target: >70% for RMSNorm)
#   - Warp occupancy
#   - Smem transaction counts
#
# Usage:
#   bash profile.sh [conflicts|rmsnorm|fused|all]
#
# Requires: ncu (Nsight Compute CLI) — comes with CUDA Toolkit 12+

set -e
MODE=${1:-all}

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
}

# Metrics for bank conflict analysis
METRICS_SMEM="l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum"

# Metrics for bandwidth analysis
METRICS_BW="gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

# Combined metrics
METRICS_ALL="$METRICS_SMEM,$METRICS_BW"

# ─── Bank conflict comparison ─────────────────────────────────────────────────
if [ "$MODE" = "conflicts" ] || [ "$MODE" = "all" ]; then
    header "BANK CONFLICTS — comparing access patterns (ex3_1)"

    for KERNEL in smem_no_conflict smem_stride16_conflict smem_with_conflict; do
        echo ""
        echo "  Kernel: $KERNEL"
        ncu \
            --kernel-name "$KERNEL" \
            --metrics "$METRICS_SMEM" \
            --target-processes all \
            python ex3_1_bank_conflicts.py 2>&1 \
            | grep -E "($KERNEL|bank_conflict|sectors_pipe_lsu_mem_shared)" \
            | head -8
    done

    echo ""
    echo "  Expected:"
    echo "    smem_no_conflict      : bank_conflicts_ld=0, bank_conflicts_st=0"
    echo "    smem_stride16_conflict: bank_conflicts_ld=1 per warp  (2-way)"
    echo "    smem_with_conflict    : bank_conflicts_ld=31 per warp (32-way)"
fi

# ─── 2D block reduce ──────────────────────────────────────────────────────────
if [ "$MODE" = "block_reduce" ] || [ "$MODE" = "all" ]; then
    header "2D BLOCK REDUCE — smem usage (ex3_2)"

    ncu \
        --kernel-name "row_sum_2d" \
        --metrics "$METRICS_SMEM,smsp__inst_executed.sum" \
        --target-processes all \
        python ex3_2_2d_block_reduce.py 2>&1 \
        | grep -E "(row_sum_2d|bank_conflict|sectors_shared|inst_executed)" \
        | head -12

    echo ""
    echo "  Expected:"
    echo "    bank_conflicts_ld = 0  (smem[warp_id] → distinct banks)"
    echo "    bank_conflicts_st = 0"
    echo "    sectors_shared_ld = 1 per warp (just smem[0] final read)"
    echo "    sectors_shared_st = 2 per warp (smem[wid] + smem[0])"
fi

# ─── RMSNorm bandwidth ────────────────────────────────────────────────────────
if [ "$MODE" = "rmsnorm" ] || [ "$MODE" = "all" ]; then
    header "RMSNORM fp16 — bandwidth and bank conflicts (ex3_3)"

    ncu \
        --kernel-name "rmsnorm_f16" \
        --metrics "$METRICS_ALL" \
        --target-processes all \
        python ex3_3_rmsnorm_f16.py 2>&1 \
        | grep -E "(rmsnorm_f16|dram|global|warps_active|bank_conflict|throughput|sectors_shared)" \
        | head -20

    echo ""
    echo "  Expected:"
    echo "    gpu__dram_throughput      : >70%  (>201 GB/s on RTX 4060 Ti)"
    echo "    sm__warps_active          : >50%  (good occupancy)"
    echo "    bank_conflicts_ld         : 0     (correct smem pattern)"
    echo "    global_load_bytes         : ≈ 2 × batch × hidden × 2 B  (read x twice + weight)"
    echo "    global_store_bytes        : ≈ batch × hidden × 2 B"
fi

# ─── Fused add + rmsnorm ──────────────────────────────────────────────────────
if [ "$MODE" = "fused" ] || [ "$MODE" = "all" ]; then
    header "FUSED ADD+RMSNORM — smem cache reduces DRAM reads (ex3_4)"

    METRICS_FUSED="$METRICS_BW,\
l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum"

    ncu \
        --kernel-name "fused_add_rmsnorm_f16" \
        --metrics "$METRICS_FUSED" \
        --target-processes all \
        python ex3_4_fused_add_rmsnorm.py 2>&1 \
        | grep -E "(fused_add|dram|global|shared|warps_active|bank_conflict|throughput)" \
        | head -20

    echo ""
    echo "  Expected for fused kernel (batch=4096, hidden=4096, fp16):"
    echo "    global_load_bytes  : ≈ 3 × 4096 × 4096 × 2 B = 96 MB"
    echo "                         (inp + res + weight — NOT x in pass 2)"
    echo "    shared_load_bytes  : ≈ 1 × 4096 × 4096 × 2 B = 32 MB"
    echo "                         (x read from smem in pass 2)"
    echo "    global_store_bytes : ≈ 2 × 4096 × 4096 × 2 B = 64 MB"
    echo "                         (residual update + output)"
    echo "    bank_conflicts_ld  : 0 (smem[wid] pattern + sequential x_smem access)"
    echo ""
    echo "  Unfused comparison: global_load_bytes would be ≈ 4 × 32 MB = 128 MB"
    echo "    (x read twice from GDDR6X instead of once from smem)"
fi

echo ""
echo "========================================================"
echo "  SUMMARY — What Lesson 3 Teaches:"
echo ""
echo "  1. Bank conflicts:"
echo "       smem[tx * 32] → 32-way conflict → 32× slower"
echo "       smem[warp_id] → distinct banks → zero conflict"
echo ""
echo "  2. Two __syncthreads() per block reduce (minimum):"
echo "       barrier 1: after smem[warp_id] write"
echo "       barrier 2: after smem[0] write (warp 0)"
echo ""
echo "  3. RMSNorm bandwidth target: >70% of 288 GB/s = >201 GB/s"
echo "       memory-bound: optimize for bandwidth, not compute"
echo ""
echo "  4. Fused add+norm saves 1 GDDR6X round-trip per layer:"
echo "       8 KB smem x-cache → 32 cycle access vs 300 cycle GDDR6X"
echo ""
echo "  To save full report for GUI analysis:"
echo "    ncu --set full -o lesson3_report python ex3_3_rmsnorm_f16.py"
echo "    ncu-ui lesson3_report.ncu-rep"
echo "========================================================"
