#!/usr/bin/env bash
# Exercise 2.6 — Inspect PTX: scalar smem reduce vs warp shuffle reduce
#
# Run: bash ex2_6_inspect_ptx.sh
#
# What you will see:
#   Scalar:  ld.shared / st.shared instructions (smem round-trips)
#   Shuffle: shfl.sync.bfly.b32 instructions (register-to-register, zero smem)

set -e

echo "================================================================"
echo "  Ex 2.6 — PTX Inspection: smem reduce vs warp shuffle reduce"
echo "================================================================"

echo ""
echo "Compiling scalar smem reduce to PTX..."
nvcc --ptx -arch=sm_89 -O3 ex2_6_scalar_reduce_kernel.cu -o ex2_6_scalar.ptx
echo "Done: ex2_6_scalar.ptx"

echo ""
echo "Compiling warp shuffle reduce to PTX..."
nvcc --ptx -arch=sm_89 -O3 ex2_6_warp_reduce_kernel.cu -o ex2_6_warp_reduce.ptx
echo "Done: ex2_6_warp_reduce.ptx"

echo ""
echo "================================================================"
echo "  SCALAR — shared memory stores (st.shared):"
echo "================================================================"
grep "st.shared" ex2_6_scalar.ptx || echo "  (none)"

echo ""
echo "================================================================"
echo "  SCALAR — shared memory loads (ld.shared):"
echo "================================================================"
grep "ld.shared" ex2_6_scalar.ptx | head -5
echo "  ..."
SMEM_LD=$(grep -c "ld.shared" ex2_6_scalar.ptx 2>/dev/null || echo 0)
echo "  Total ld.shared count: $SMEM_LD"

echo ""
echo "================================================================"
echo "  SHUFFLE — shfl.sync.bfly (register-to-register exchange):"
echo "================================================================"
grep "shfl.sync" ex2_6_warp_reduce.ptx || echo "  (none)"
SHFL=$(grep -c "shfl.sync" ex2_6_warp_reduce.ptx 2>/dev/null || echo 0)
echo ""
echo "  Total shfl.sync count: $SHFL  (should be 10: 5 for sum + 5 for max)"

echo ""
echo "================================================================"
echo "  SHUFFLE — any shared memory loads? (should be zero):"
echo "================================================================"
SHFL_SMEM=$(grep -c "ld.shared\|st.shared" ex2_6_warp_reduce.ptx 2>/dev/null || echo 0)
echo "  ld.shared / st.shared count in shuffle kernel: $SHFL_SMEM"
if [ "$SHFL_SMEM" -eq 0 ]; then
    echo "  ✓ Confirmed: warp shuffle uses ZERO shared memory"
else
    echo "  ✗ Unexpected: shuffle kernel accessed smem — check your kernel"
fi

echo ""
echo "================================================================"
echo "  What to look for:"
echo ""
echo "  Scalar reduce PTX:"
echo "    st.shared.f32   [smem], val     — thread writes to smem"
echo "    ld.shared.f32   val, [smem+N]   — thread 0 reads each element"
echo "    bar.sync 0                      — __syncthreads barrier"
echo ""
echo "    → 1 st.shared + 32 ld.shared + 1 bar.sync = 34 smem ops"
echo ""
echo "  Shuffle reduce PTX:"
echo "    shfl.sync.bfly.b32  dst, src, 16, 31, -1"
echo "    shfl.sync.bfly.b32  dst, src, 8,  31, -1"
echo "    shfl.sync.bfly.b32  dst, src, 4,  31, -1"
echo "    shfl.sync.bfly.b32  dst, src, 2,  31, -1"
echo "    shfl.sync.bfly.b32  dst, src, 1,  31, -1"
echo ""
echo "    → 5 shfl.sync instructions, zero smem, zero barrier"
echo "    → Each shfl.sync: register read-write in ~1 clock cycle"
echo "================================================================"
