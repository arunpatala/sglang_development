#!/usr/bin/env bash
# Exercise 1.7 — Profile with Nsight Compute
# ===========================================
# Compares the key memory metrics between scalar and vectorized kernels.
#
# Usage:
#   bash profile.sh [scalar|vec|all]
#
# Requires: ncu (Nsight Compute CLI) — comes with CUDA Toolkit

set -e

# ncu spawns fresh subprocesses that don't inherit the conda environment.
# Prepend the conda bin dir so that the correct Python, ninja, and nvcc are found.
CONDA_BIN=/home/arun/PROJECTS/sglang_development/.conda/bin
PYTHON=$CONDA_BIN/python
export PATH="$CONDA_BIN:$PATH"

METRICS="l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum,\
sm__throughput.avg.pct_of_peak_sustained_elapsed,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"

MODE=${1:-all}

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
}

if [ "$MODE" = "scalar" ] || [ "$MODE" = "all" ]; then
    header "SCALAR COPY — copy_scalar kernel (ex1_1)"
    sudo env PATH="$PATH" ncu \
        --kernel-name "copy_scalar" \
        --metrics "$METRICS" \
        --target-processes all \
        $PYTHON ex1_1_scalar_copy.py 2>&1
fi

if [ "$MODE" = "vec" ] || [ "$MODE" = "all" ]; then
    header "VECTORIZED COPY — copy_vec8_f16 kernel (ex1_4)"
    sudo env PATH="$PATH" ncu \
        --kernel-name "copy_vec8_f16" \
        --metrics "$METRICS" \
        --target-processes all \
        $PYTHON ex1_4_vec_copy_f16.py 2>&1

    header "VECTORIZED SCALE — scale_vec8_f16 kernel (ex1_4)"
    sudo env PATH="$PATH" ncu \
        --kernel-name "scale_vec8_f16" \
        --metrics "$METRICS" \
        --target-processes all \
        $PYTHON ex1_4_vec_copy_f16.py 2>&1
fi

echo ""
echo "========================================================"
echo "  What to look for:"
echo ""
echo "  sectors_per_request:"
echo "    Scalar     → ~1   (16-bit load → 1 sector)"
echo "    Vectorized → ~8   (128-bit load → 8 sectors = full cache line)"
echo ""
echo "  gpu__dram_throughput:"
echo "    Scalar     → 10–30%"
echo "    Vectorized → 75–95%"
echo ""
echo "  sm__throughput:"
echo "    Scalar     → very low (SM stalls waiting for memory)"
echo "    Vectorized → higher (fewer stalls)"
echo "========================================================"
echo ""
echo "For full analysis, use the interactive Nsight Compute GUI:"
echo "  ncu --set full -o profile_report $PYTHON ex1_4_vec_copy_f16.py"
echo "  ncu-ui profile_report.ncu-rep   (if GUI available)"
