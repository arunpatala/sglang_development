#!/usr/bin/env bash
# Exercise 1.2 — Inspect PTX assembly to see scalar vs vectorized loads
#
# Run: bash ex1_2_inspect_ptx.sh

set -e

echo "================================================================"
echo "  Ex 1.2 — PTX Inspection: scalar vs 128-bit loads"
echo "================================================================"

# Compile both kernels to PTX
echo ""
echo "Compiling scalar kernel to PTX..."
nvcc --ptx -arch=sm_89 -O3 ex1_2_scalar_kernel.cu -o ex1_2_scalar.ptx
echo "Done: ex1_2_scalar.ptx"

echo ""
echo "Compiling vectorized kernel to PTX..."
nvcc --ptx -arch=sm_89 -O3 ex1_2_vec_kernel.cu -o ex1_2_vec.ptx
echo "Done: ex1_2_vec.ptx"

echo ""
echo "================================================================"
echo "  SCALAR ld.global instructions:"
echo "================================================================"
grep "ld.global" ex1_2_scalar.ptx || echo "  (no ld.global found — check kernel name)"

echo ""
echo "================================================================"
echo "  VECTORIZED ld.global instructions:"
echo "================================================================"
grep "ld.global" ex1_2_vec.ptx || echo "  (no ld.global found — check kernel name)"

echo ""
echo "================================================================"
echo "  SCALAR st.global instructions:"
echo "================================================================"
grep "st.global" ex1_2_scalar.ptx

echo ""
echo "================================================================"
echo "  VECTORIZED st.global instructions:"
echo "================================================================"
grep "st.global" ex1_2_vec.ptx

echo ""
echo "================================================================"
echo "  What to look for:"
echo "    Scalar:      ld.global.cs.u16  (16 bits = 1 fp16)"
echo "    Vectorized:  ld.global.ca.v4.u32  (128 bits = 8 fp16)"
echo ""
echo "  The vectorized kernel moves 8x more data per instruction."
echo "  This is the root cause of the bandwidth difference."
echo "================================================================"
