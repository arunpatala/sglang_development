#!/usr/bin/env bash
# profile.sh — Nsight Compute Profiling for Lesson 4 Kernels
# ===========================================================
# Profiles all Lesson 4 kernels and reports:
#   - SM compute throughput (tensor core utilization)
#   - HBM/GDDR6X memory throughput
#   - Shared memory bank conflicts
#   - Warp occupancy
#   - Global memory load/store bytes
#
# Usage:
#   bash profile.sh [naive|tiled|wmma|swizzle|all]
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

# ── Metric sets ───────────────────────────────────────────────────────────────

# Core throughput metrics
METRICS_COMPUTE="\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
sm__warps_active.avg.pct_of_peak_sustained_active,\
smsp__sass_thread_inst_executed_op_ffma_pred_on.sum,\
smsp__sass_thread_inst_executed_op_hmma_pred_on.sum"

# Memory throughput metrics
METRICS_MEM="\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
l2__throughput.avg.pct_of_peak_sustained_elapsed"

# Shared memory metrics
METRICS_SMEM="\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum,\
l1tex__t_bytes_pipe_lsu_mem_shared_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_shared_op_st.sum"

# All combined
METRICS_ALL="${METRICS_COMPUTE},${METRICS_MEM},${METRICS_SMEM}"

# ─── Naive GEMM ───────────────────────────────────────────────────────────────
if [ "$MODE" = "naive" ] || [ "$MODE" = "all" ]; then
    header "NAIVE GEMM (ex4_1) — memory-bound baseline"

    ncu \
        --kernel-name "naive_gemm_fp32" \
        --metrics "${METRICS_COMPUTE},${METRICS_MEM}" \
        --target-processes all \
        python ex4_1_naive_gemm.py 2>&1 \
        | grep -E "(naive_gemm|throughput|warps_active|dram|bytes|ffma|hmma)" \
        | head -15

    echo ""
    echo "  Expected for M=N=K=512 fp32 naive GEMM:"
    echo "    sm__throughput:       very low (<10%) — memory stalls dominate"
    echo "    dram_throughput:      moderate (~30-60%) — L2 thrashing hurts bandwidth"
    echo "    warps_active:         low (<40%) — threads stall waiting for global loads"
    echo "    smsp__ffma_count:     > 0 (scalar fp32 FMA — no tensor cores)"
    echo "    smsp__hmma_count:     0   (no WMMA instructions)"
    echo "    global_load_bytes:    very large (no smem caching → many redundant loads)"
fi

# ─── Tiled GEMM ───────────────────────────────────────────────────────────────
if [ "$MODE" = "tiled" ] || [ "$MODE" = "all" ]; then
    header "TILED GEMM (ex4_2) — shared memory reduces HBM reads"

    for KERNEL in tiled_gemm_t16 tiled_gemm_t32; do
        echo ""
        echo "  Kernel: $KERNEL"
        ncu \
            --kernel-name "$KERNEL" \
            --metrics "${METRICS_ALL}" \
            --target-processes all \
            python ex4_2_tiled_gemm.py 2>&1 \
            | grep -E "($KERNEL|throughput|warps_active|dram|bytes|bank_conflict|ffma|hmma)" \
            | head -12
    done

    echo ""
    echo "  Expected for M=N=K=1024 fp32 tiled GEMM:"
    echo "    sm__throughput:         higher than naive (>20%) — smem reuse helps"
    echo "    dram_throughput:        lower than naive (smem absorbs reloads)"
    echo "    bank_conflicts_ld:      0  (2D block + row reads = zero conflict)"
    echo "    bank_conflicts_st:      0  (row writes = zero conflict)"
    echo "    global_load_bytes:      ~(M*K + K*N)*4 = ~8MB for 1024 (one pass each)"
    echo "    shared_load_bytes:      >> global (smem elements each read TILE times)"
    echo "    smsp__ffma_count:       > 0 (scalar fp32 FMA)"
    echo "    smsp__hmma_count:       0   (no tensor cores)"
fi

# ─── WMMA GEMM ────────────────────────────────────────────────────────────────
if [ "$MODE" = "wmma" ] || [ "$MODE" = "all" ]; then
    header "WMMA GEMM (ex4_3) — tensor core utilization"

    for KERNEL in wmma_gemm_v1 wmma_gemm_v2; do
        echo ""
        echo "  Kernel: $KERNEL"
        ncu \
            --kernel-name "$KERNEL" \
            --metrics "${METRICS_ALL}" \
            --target-processes all \
            python ex4_3_wmma_gemm.py 2>&1 \
            | grep -E "($KERNEL|throughput|warps_active|dram|bytes|bank_conflict|ffma|hmma)" \
            | head -15
    done

    echo ""
    echo "  Expected for M=N=K=2048 fp16 WMMA GEMM:"
    echo ""
    echo "  wmma_gemm_v1 (global loads, no smem):"
    echo "    sm__throughput:       moderate (30-50%) — tensor cores active"
    echo "    dram_throughput:      high (bandwidth-limited without smem tiling)"
    echo "    bank_conflicts_ld:    0 (no smem used)"
    echo "    smsp__ffma_count:     0 (scalar FMAs replaced by tensor cores)"
    echo "    smsp__hmma_count:     > 0 (HMMA.16816 tensor core instructions!)"
    echo "    global_load_bytes:    M*K*2 + K*N*2 bytes (each A/B element loaded M/16 times)"
    echo ""
    echo "  wmma_gemm_v2 (smem tiled, 64×64 blocks):"
    echo "    sm__throughput:       higher than v1 (smem reduces bandwidth pressure)"
    echo "    dram_throughput:      lower than v1  (smem absorbs reloads)"
    echo "    bank_conflicts_ld:    possibly 1 (2-way) for fp16 smem WMMA loads"
    echo "                          → this is what permuted_smem.cuh (Lesson 5) fixes"
    echo "    smsp__hmma_count:     > 0 (tensor cores used)"
    echo ""
    echo "  Key insight: smsp__hmma_count > 0 confirms tensor cores are active."
    echo "  smsp__ffma_count = 0 confirms scalar FMAs have been fully replaced."
fi

# ─── smem Swizzle ─────────────────────────────────────────────────────────────
if [ "$MODE" = "swizzle" ] || [ "$MODE" = "all" ]; then
    header "SMEM SWIZZLE (ex4_4) — bank conflict elimination"

    for KERNEL in transpose_naive transpose_padded transpose_swizzled; do
        echo ""
        echo "  Kernel: $KERNEL"
        ncu \
            --kernel-name "$KERNEL" \
            --metrics "${METRICS_SMEM},gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed" \
            --target-processes all \
            python ex4_4_smem_swizzle.py 2>&1 \
            | grep -E "($KERNEL|bank_conflict|dram|bytes_pipe_lsu_mem_shared|throughput)" \
            | head -8
    done

    echo ""
    echo "  Expected for N=2048 matrix transpose:"
    echo "    transpose_naive:"
    echo "      bank_conflicts_ld = 31 per warp (32-way serialization)"
    echo "      bank_conflicts_st = 0  (write phase: sequential banks)"
    echo "      shared_ld_bytes:   N*N*4 but requires 32× more transactions"
    echo "      dram_throughput:   lower (smem bottleneck stalls the pipeline)"
    echo ""
    echo "    transpose_padded:"
    echo "      bank_conflicts_ld = 0"
    echo "      bank_conflicts_st = 0"
    echo "      dram_throughput:   higher (smem no longer the bottleneck)"
    echo ""
    echo "    transpose_swizzled:"
    echo "      bank_conflicts_ld = 0"
    echo "      bank_conflicts_st = 0 (swizzled write also conflict-free)"
    echo "      dram_throughput:   same as padded (identical performance)"
    echo "      smem:              same size as naive (no padding overhead)"
fi

echo ""
echo "========================================================"
echo "  SUMMARY — What Lesson 4 Teaches:"
echo ""
echo "  1. Naive GEMM:"
echo "       bottleneck = L2 cache thrash + global memory latency"
echo "       FMA throughput << 1% of peak → compute idle"
echo ""
echo "  2. Tiled GEMM:"
echo "       smem reduces HBM reads by TILE= factor"
echo "       zero bank conflicts with 2D block layout"
echo "       FMA throughput 5–20% of fp32 CUDA core peak"
echo ""
echo "  3. WMMA GEMM:"
echo "       HMMA instructions replace scalar FMAs (8× throughput)"
echo "       smem tiling (v2) reduces bandwidth pressure"
echo "       TFLOPS 20–50% of fp16 tensor core peak"
echo "       bank conflicts possible in WMMA smem loads → Lesson 5 fix"
echo ""
echo "  4. smem Swizzle:"
echo "       naive: 32-way bank conflict on column reads → 32× serialization"
echo "       padded: +1 col eliminates conflict (128B wasted per tile)"
echo "       XOR swizzle: same perf, zero extra memory"
echo "       production fix: permuted_smem.cuh (Flash Attention Lesson 5)"
echo ""
echo "  Metric targets:"
echo "    Tiled GEMM fp32:   >20% of fp32 CUDA core peak (~15 TFLOPS)"
echo "    WMMA v2 fp16:      >50% of fp16 tensor core peak (~44 TFLOPS)"
echo "    Matrix transpose:  >70% of 288 GB/s GDDR6X bandwidth"
echo ""
echo "  To save a full report:"
echo "    ncu --set full -o lesson4_wmma python ex4_3_wmma_gemm.py"
echo "    ncu-ui lesson4_wmma.ncu-rep"
echo "========================================================"
