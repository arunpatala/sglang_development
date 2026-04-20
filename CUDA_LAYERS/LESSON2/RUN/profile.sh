#!/usr/bin/env bash
# Exercise 2.7 — Profile with Nsight Compute
# ===========================================
# Compares smem usage, instruction count, and shared memory transactions
# between naive reduce (ex2_1) and warp shuffle reduce (ex2_2).
#
# Usage:
#   bash profile.sh [naive|shuffle|rmsnorm|all]
#
# Requires: ncu (Nsight Compute CLI) — comes with CUDA Toolkit

set -e

# Metrics that expose the smem vs shuffle difference:
# - smsp__inst_executed:         total instructions executed per warp
# - l1tex__t_sectors_pipe_lsu_mem_shared_op_ld:  smem read transactions
# - l1tex__t_sectors_pipe_lsu_mem_shared_op_st:  smem write transactions
# - sm__warps_active.avg.pct_of_peak_sustained_active: warp occupancy
METRICS="smsp__inst_executed.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_sectors_pipe_lsu_mem_shared_op_st.sum,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"

MODE=${1:-all}

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
}

if [ "$MODE" = "naive" ] || [ "$MODE" = "all" ]; then
    header "NAIVE REDUCE — block_reduce_naive (ex2_1)"
    ncu \
        --kernel-name "block_reduce_naive" \
        --metrics "$METRICS" \
        --target-processes all \
        python ex2_1_scalar_reduce.py 2>&1 \
        | grep -E "(block_reduce_naive|Metric|inst_executed|shared|throughput|warps)" \
        | head -30
fi

if [ "$MODE" = "shuffle" ] || [ "$MODE" = "all" ]; then
    header "WARP SHUFFLE REDUCE — reduce_warp_sum (ex2_2)"
    ncu \
        --kernel-name "reduce_warp_sum" \
        --metrics "$METRICS" \
        --target-processes all \
        python ex2_2_warp_reduce_sum.py 2>&1 \
        | grep -E "(reduce_warp_sum|Metric|inst_executed|shared|throughput|warps)" \
        | head -30

    header "WARP SHUFFLE MAX — reduce_warp_max (ex2_3)"
    ncu \
        --kernel-name "reduce_warp_max" \
        --metrics "$METRICS" \
        --target-processes all \
        python ex2_3_warp_reduce_max.py 2>&1 \
        | grep -E "(reduce_warp_max|Metric|inst_executed|shared|throughput|warps)" \
        | head -30
fi

if [ "$MODE" = "rmsnorm" ] || [ "$MODE" = "all" ]; then
    header "RMSNORM fp16 dim=256 — rmsnorm_f16_dim256 (ex2_4)"
    METRICS_RMSNORM="smsp__inst_executed.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
sm__throughput.avg.pct_of_peak_sustained_elapsed"
    ncu \
        --kernel-name "rmsnorm_f16_dim256" \
        --metrics "$METRICS_RMSNORM" \
        --target-processes all \
        python ex2_4_vec_load_warp_reduce.py 2>&1 \
        | grep -E "(rmsnorm_f16|Metric|bytes|throughput|dram)" \
        | head -30

    header "RMSNORM fp16 multi-warp — rmsnorm_f16_multiblock (ex2_5)"
    ncu \
        --kernel-name "rmsnorm_f16_multiblock" \
        --metrics "$METRICS_RMSNORM" \
        --target-processes all \
        python ex2_5_block_reduce.py 2>&1 \
        | grep -E "(rmsnorm_f16_multi|Metric|bytes|throughput|dram)" \
        | head -30
fi

echo ""
echo "========================================================"
echo "  What to look for:"
echo ""
echo "  l1tex__t_sectors_pipe_lsu_mem_shared_op_ld (smem reads):"
echo "    Naive reduce  → many  (32 serial ld.shared per block)"
echo "    Warp shuffle  → 0     (zero shared memory used)"
echo "    Block reduce  → small (1 slot per warp written to smem)"
echo ""
echo "  smsp__inst_executed:"
echo "    Naive reduce  → high  (32 serial adds, only 1/32 threads active)"
echo "    Warp shuffle  → low   (5 shfl.sync instructions, all 32 threads)"
echo ""
echo "  sm__warps_active:"
echo "    Naive reduce  → lower (31/32 threads idle during serial loop)"
echo "    Warp shuffle  → higher (all threads active every instruction)"
echo ""
echo "  gpu__dram_throughput (for RMSNorm kernels):"
echo "    Target: >70% of peak (288 GB/s → >200 GB/s)"
echo "    If lower: likely register pressure reducing occupancy"
echo "========================================================"
echo ""
echo "For full analysis, dump a report and open in ncu-ui:"
echo "  ncu --set full -o lesson2_report python ex2_4_vec_load_warp_reduce.py"
echo "  ncu-ui lesson2_report.ncu-rep   (if GUI available)"
