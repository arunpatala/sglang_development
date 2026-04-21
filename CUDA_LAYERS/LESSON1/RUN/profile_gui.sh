#!/usr/bin/env bash
# profile_gui.sh — Generate Nsight Compute report files for GUI analysis
# ======================================================================
# Collects the FULL metric set (hundreds of counters) for each kernel
# and saves a .ncu-rep file that can be opened in the Nsight Compute GUI.
#
# Usage:
#   bash profile_gui.sh [scalar|vec|all]
#
# Output files (created in the current directory):
#   profile_scalar.ncu-rep     — copy_scalar kernel
#   profile_vec_copy.ncu-rep   — copy_vec8_f16 kernel
#   profile_vec_scale.ncu-rep  — scale_vec8_f16 kernel
#
# Open any report with:
#   ncu-ui profile_scalar.ncu-rep
#
# NOTE: --set full replays each kernel many times (one pass per metric group).
#       Expect 30–120 seconds per kernel. This is normal.

set -e

# sudo strips PATH, so ninja and the conda Python are invisible to the subprocess.
# sudo env PATH="$PATH" forwards our PATH into the root process.
CONDA_BIN=/home/arun/PROJECTS/sglang_development/.conda/bin
PYTHON=$CONDA_BIN/python
export PATH="$CONDA_BIN:$PATH"

MODE=${1:-all}

header() {
    echo ""
    echo "========================================================"
    echo "  $1"
    echo "========================================================"
}

if [ "$MODE" = "scalar" ] || [ "$MODE" = "all" ]; then
    header "Profiling copy_scalar → profile_scalar.ncu-rep"
    sudo env PATH="$PATH" ncu \
        --kernel-name "copy_scalar" \
        --set full \
        --target-processes all \
        -o profile_scalar \
        $PYTHON ex1_1_scalar_copy.py 2>&1
    echo "  Saved: profile_scalar.ncu-rep"
fi

if [ "$MODE" = "vec" ] || [ "$MODE" = "all" ]; then
    header "Profiling copy_vec8_f16 → profile_vec_copy.ncu-rep"
    sudo env PATH="$PATH" ncu \
        --kernel-name "copy_vec8_f16" \
        --set full \
        --target-processes all \
        -o profile_vec_copy \
        $PYTHON ex1_4_vec_copy_f16.py 2>&1
    echo "  Saved: profile_vec_copy.ncu-rep"

    header "Profiling scale_vec8_f16 → profile_vec_scale.ncu-rep"
    sudo env PATH="$PATH" ncu \
        --kernel-name "scale_vec8_f16" \
        --set full \
        --target-processes all \
        -o profile_vec_scale \
        $PYTHON ex1_4_vec_copy_f16.py 2>&1
    echo "  Saved: profile_vec_scale.ncu-rep"
fi

echo ""
echo "========================================================"
echo "  Reports saved. Open with:"
echo ""
echo "  ncu-ui profile_scalar.ncu-rep"
echo "  ncu-ui profile_vec_copy.ncu-rep"
echo "  ncu-ui profile_vec_scale.ncu-rep"
echo ""
echo "  Pages to look at in the GUI:"
echo ""
echo "  1. Speed of Light"
echo "     Two bars: Memory % vs Compute %."
echo "     Whichever is taller is your bottleneck."
echo "     copy/scale kernels → Memory bar should dominate (~80%)."
echo ""
echo "  2. Memory Workload Analysis"
echo "     L1 hit rate, L2 hit rate, DRAM GB/s."
echo "     Sectors/request → 1 = scalar (wasteful), 8 = vectorized (good)."
echo ""
echo "  3. Warp State Statistics"
echo "     'Stall LG throttle' = warps waiting for global memory."
echo "     scalar kernel → most warps stalled here."
echo "     vectorized    → fewer stalls, more time in 'No stall'."
echo ""
echo "  4. Source Counters (if source is available)"
echo "     Shows per-line memory access counts — pinpoints hot lines."
echo "========================================================"
