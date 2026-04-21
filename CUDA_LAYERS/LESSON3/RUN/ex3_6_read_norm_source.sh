#!/usr/bin/env bash
# ex3_6_read_norm_source.sh — Guided Reading of Production norm.cuh
# ==================================================================
# Prints a guided reading guide with specific questions to answer
# before looking at the production norm.cuh implementation.
#
# Usage:
#   bash ex3_6_read_norm_source.sh

cat <<'GUIDE'
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUIDED READING: Production RMSNorm Implementation
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Read the files in this exact order. Answer each question before
reading the answer — this is the most important part of the process.

After struggling with your own implementation (Exercises 3.3–3.4),
the production code will now make complete sense. Every line exists
for a reason you now understand.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 1: REPOS/flashinfer/include/flashinfer/norm.cuh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

─── Section A: RMSNormKernel<VEC_SIZE, T> (lines 37–111) ─────────────

Q1: Look at the template parameters <uint32_t VEC_SIZE, typename T>.
    What is VEC_SIZE, and what does it control?
    Where is it determined? (Hint: look at the launcher, lines 113–146)

    Answer: VEC_SIZE = number of T values loaded per thread per iteration.
            = gcd(16 / sizeof(T), d):
              • fp16 (2B): 16/2=8 → VEC_SIZE=8 when d%8==0
              • fp32 (4B): 16/4=4 → VEC_SIZE=4 when d%4==0
            Chosen to maximize load width (128-bit per thread) while
            respecting alignment (d must be divisible by VEC_SIZE).

Q2: Look at the smem declaration. How many bytes does it use?
    How does this compare to the 100 KB limit on sm_89?

    Answer: smem = num_warps × sizeof(float) = num_warps × 4 bytes.
            For 16 warps (maximum): 64 bytes. Negligible.

Q3: Lines 56–66: the vectorized load loop. The loop increments by
    blockDim.y * blockDim.x. Explain why.

    Answer: blockDim.y * blockDim.x = num_warps × 32 = total threads
            per block. Strided iteration: tid=0 handles indices
            0, n_threads, 2*n_threads, ... covering all n_vec positions.
            This ensures every position is processed exactly once.

Q4: Lines 68–76: the intra-warp shuffle reduce. This uses:
      for (int offset = 16; offset > 0; offset >>= 1)
          sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);
    What is this doing? (You wrote this in Lesson 2.)

    Answer: Butterfly warp reduce — 5 rounds of XOR shuffle.
            After 5 rounds, every lane in the warp holds the warp's
            sum of sum_sq values. Identical to warp_reduce_sum().

Q5: Lines 68–84: the cross-warp smem reduce. Find:
    (a) where lane 0 writes to smem
    (b) the first __syncthreads()
    (c) where warp 0 reads from smem
    (d) the second __syncthreads()

    Verify: threadIdx.x is lane, threadIdx.y is warp_id.
    Verify: smem[threadIdx.y] writes to bank threadIdx.y → zero conflicts.

Q6: Line ~87: rsqrt call. What PTX instruction does this compile to?

    Answer: rsqrt.approx.ftz.f32 → MUFU.RSQ instruction.
            MUFU = Multifunction Unit (dedicated math unit per SM).
            RSQ = Reciprocal Square Root.
            Latency: 1 clock cycle. This is why rsqrtf beats
            1.f / sqrtf(x) (which is multiple instructions).

Q7: Lines 87–111: Pass 2. Notice there is NO __syncthreads() before
    this pass. Why is it safe to proceed without a sync?

    Answer: The second __syncthreads() inside the cross-warp reduce
            already synchronizes all threads. After that barrier:
            • smem[0] contains the final total (written by warp 0)
            • All threads have read smem[0] into local rms_rcp
            • rms_rcp is now a private register — no smem needed
            Pass 2 only reads from global memory (x and weight) and
            writes to global memory (y). No smem coordination needed.

─── Section B: SM90 PDL path (skip on sm_89) ─────────────────────────

Q8: Find the #if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900) block.
    What does griddepcontrol.wait do?

    Answer: Programmatic Dependent Launch (PDL) on Hopper (sm_90).
            griddepcontrol.wait — the kernel waits until the SM
            scheduler signals that dependent kernels have been launched.
            griddepcontrol.launch_dependents — triggers launch of
            kernels that depend on this one's output.
            Enables overlapping kernel execution without host sync.
            NOT available on sm_89 — this entire path is dead code
            on your RTX 4060 Ti.

─── Section C: RMSNormQuantKernel (lines 148–227) ────────────────────

Q9: This kernel outputs FP8 instead of fp16/bf16.
    Find where the FP8 clamping and casting happens.
    What is the clamp range and why?

    Answer: FP8 e4m3 range: [-448, 448].
            The kernel computes: val = clamp(val * scale_inv, -448, 448)
            then casts to __nv_fp8_e4m3.
            scale_inv = 1.0 / out_scale.
            The per-tensor scale ensures the normalized values fit
            within the FP8 dynamic range.
            This is called before every FP8 GEMM in DeepSeek-V3.

─── Section D: FusedAddRMSNormKernel (lines 387–477) ─────────────────

Q10: Find where x = input + residual is computed.
     Where is x stored after computation?

     Answer: x is computed in pass 1 and stored to smem_x (shared memory).
             It is ALSO written back to the residual tensor in global
             memory (the transformer needs the updated residual for the
             next layer's residual connection).

Q11: In pass 2, where does the kernel read x from?

     Answer: From smem_x — the smem cache.
             NOT from global memory. This saves one full global read
             of the hidden state tensor.

Q12: Find the smem allocation. What are the two regions?

     Answer:
       Region 1: smem for cross-warp reduce (num_warps floats = 4-64 bytes)
       Region 2: smem_x for x cache (n_vec × sizeof(T) × VEC_SIZE bytes)
       Total: typically 4–16 KB for production hidden dims.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 2: REPOS/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q13: Find the TORCH_LIBRARY_FRAGMENT macro. What does it do?

     Answer: Registers the C++ function as a PyTorch operator:
             torch.ops.sgl_kernel.fused_add_rmsnorm
             This makes it callable from Python without explicitly
             loading the .so — PyTorch dispatches through its operator
             registry.

Q14: Find enable_pdl. When is the SM90 path used vs the default path?

     Answer: enable_pdl is true only when the runtime detects SM90
             (H100/H200). On SM89 (RTX 4060 Ti), enable_pdl=false and
             the kernel launches without griddepcontrol.
             The SM90 path uses Programmatic Dependent Launch to
             overlap kernel execution — not available on Ada.

Q15: What is the Python API that calls this kernel?
     Trace the call from model code → Python → C++ → GPU.

     Answer:
       Python model code:
         from sgl_kernel import fused_add_rmsnorm
         fused_add_rmsnorm(hidden, residual, weight, eps)
       ↓
       sgl_kernel/python/sgl_kernel/elementwise.py:
         torch.ops.sgl_kernel.fused_add_rmsnorm.default(...)
       ↓
       TORCH_LIBRARY_FRAGMENT dispatches to C++ function
       ↓
       norm::FusedAddRMSNorm<scalar_t><<<grid, block, smem>>>(...)
       ↓
       GPU executes fused_add_rmsnorm_kernel PTX

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 3: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/impl/norm.cuh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q16: Lines 60–76: apply_norm_impl with kUseCTA=false.
     Why is no shared memory needed in this path?

     Answer: kUseCTA=false means the hidden dim fits within one warp:
             32 threads × VEC_SIZE elements ≤ hidden_dim.
             The warp shuffle alone (Lesson 2) completes the reduction.
             No cross-warp communication needed → zero smem.

Q17: Lines 78–94: kUseCTA=true path. This uses cta::reduce_sum.
     Is this the same 2-level (warp+smem) pattern you implemented?

     Answer: Yes — exactly the same pattern.
             cta::reduce_sum in cta.cuh:
               1. warp::reduce_sum (5 shuffles)
               2. smem[warp_id] write + __syncthreads
               3. warp 0 reads smem, second shuffle
             Identical to block_reduce_sum_2d in Exercise 3.2.

Q18: What is PackedFloat? How many scalar fp16 values does each thread
     hold per PackedFloat element when T=half?

     Answer: PackedFloat = fp16x2_t = two fp16 values packed into one
             32-bit register (using CUDA's __half2 type internally).
             Each thread holds N PackedFloat elements = N × 2 scalar fp16.
             For N=4: 8 scalar fp16 values per thread per load iteration.
             This matches VEC_SIZE=8 in norm.cuh.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
FILE 4: REPOS/sglang/python/sglang/jit_kernel/include/sgl_kernel/cta.cuh
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q19: Lines 1–40: cta::reduce_max. Notice there is NO trailing
     __syncthreads() at the end. Why? What must the caller do
     before reading smem[0]?

     Answer: cta::reduce_max intentionally omits the trailing sync
             because the caller may need to do other work before
             reading smem[0]. The function contract (documented in
             a comment) says: caller must __syncthreads() before
             reading smem[0].
             This is a deliberate API design: avoid unnecessary syncs
             in case the caller batches multiple smem operations.

Q20: In cta::reduce_sum (if present), compare the loop start value
     to warp::reduce_sum. Why does cta iterate over warp indices
     while warp iterates over lane masks?

     Answer: They reduce different things:
               warp::reduce_sum: reduces 32 lanes within one warp
                 → XOR offsets 16,8,4,2,1 (lane-level butterfly)
               cta::reduce (step 3): reduces N warp partial sums
                 → warp 0 reads smem[0..N-1] then applies warp shuffle
                 → the same butterfly, but now each "lane value" is a
                   warp partial sum rather than a scalar element

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHECKLIST: You understand norm.cuh when you can:

  [ ] Explain why gcd(16/sizeof(T), d) computes the vector size
  [ ] Trace the complete execution flow of RMSNormKernel:
        template instantiation → launcher → kernel → two passes
  [ ] Explain why there are exactly 2 __syncthreads() in the reduce
  [ ] Point to the line where MUFU.RSQ is generated
  [ ] Explain what FusedAddRMSNorm saves vs plain RMSNorm
  [ ] Explain what TORCH_LIBRARY_FRAGMENT does
  [ ] Explain when kUseCTA is false (single warp) vs true (multi-warp)
  [ ] Explain what cta::reduce_max's missing trailing __syncthreads means

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GUIDE
