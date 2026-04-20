# CUDA Kernel Learning Plan
## "Learn CUDA by Building Deep Learning Inference Kernels"

**Prerequisite:** Basic CUDA knowledge (grid/block/thread, global memory, kernel syntax) + NVIDIA GPU architecture basics (SMs, warps, HBM vs SRAM).

**Reference:** See `PLAN.md` for the full kernel curriculum with all 55 kernels.
**Goal:** Go from CUDA basics → writing production-grade LLM inference kernels.

**Your hardware:**

| Property | Value |
|---|---|
| GPU | NVIDIA GeForce RTX 4060 Ti |
| Architecture | Ada Lovelace |
| Compute Capability | sm_89 |
| VRAM | 16 GB GDDR6X |
| Memory Bandwidth | 288 GB/s |
| Peak FP16 TFLOPS | ~44 (4th-gen tensor cores) |
| SM Count | 34 |
| Shared Memory / SM | 100 KB |
| L2 Cache | 32 MB |
| CUDA Version | 13.0 |
| Driver | 580.126.09 |
| Compile flag | `-arch=sm_89` |

---

## Core Learning Strategy

### The Loop (apply to every single kernel)

```
1. READ the reference implementation       (5–10 min)
2. IMPLEMENT it yourself without looking   (30–120 min)
3. TEST against PyTorch reference          (10 min)
4. COMPARE your version to reference       (20 min)  ← most people skip this
5. PROFILE with ncu                        (20 min)
6. OPTIMIZE one thing, re-measure, repeat  until >70% peak bandwidth
```

Step 4 is the most important step most people skip. You learn the most by reading the production code **after** you've already struggled with your own version, because now you understand *why* every line exists.

### Iteration Environment

```bash
# Fast feedback loop — recompiles in ~5 seconds:
# Terminal 1: auto-rerun tests on save
watch -n 1 python test_my_kernel.py

# Terminal 2: profile when ready
ncu --kernel-name "my_kernel_name" --set full python test_my_kernel.py

# IDE layout:
# Left pane:  your kernel file (my_kernel.cu)
# Right pane: reference kernel (e.g., norm.cuh or decode_attention.py)
```

Use `torch.utils.cpp_extension.load_inline` for all Phase 0–3 kernels.
No CMake, no build system — compile a `.cu` string from Python directly.
See `PLAN.md → Appendix B` for the exact template.

---

## What to Skip (you already know this)

Given you know CUDA basics and GPU architecture:

- **Skip** Phase 0.1 (vector add fp32) — you know this
- **Skip** Phase 0.2 (fp16/bf16 types) — review quickly, don't implement
- **Do NOT skip** Phase 0.3–0.6 — most "CUDA basics" courses never cover these, but every kernel in this repo uses them

**Quick self-test:** Can you write a warp-level reduce-sum using `__shfl_xor_sync` from memory in under 5 minutes? If yes, skip 0.4. If not, do it first.

---

## Recommended Order (not PLAN.md order)

The PLAN.md lists kernels in logical dependency order. This section gives the **optimal learning order** for someone who knows the basics — front-loading the patterns that appear most frequently.

---

### Week 1 — Foundations You're Probably Missing

**Day 1: 128-bit Vectorized Loads (Phase 0.3)**

Why first: every bandwidth-bound kernel uses this. RMSNorm, RoPE, activation, embedding — all of them. Without this, everything you write will be 2–4× slower than it needs to be.

Goal: write a kernel that copies a fp16 tensor using `float4` (128-bit) loads, verify it hits >90% of peak HBM bandwidth on your GPU.

```
Reference: REPOS/flashinfer/include/flashinfer/vec_dtypes.cuh
           Look at vec_t<T, N>::load() and vec_t<T, N>::store()
Key PTX:   LDG.E.128 (global load, 128-bit, evict-normal)
```

---

**Day 2: Warp Shuffle Reduce (Phase 0.4)**

Why second: the inner loop of RMSNorm, softmax, and attention QK dot product all use this.

Goal: write `warp_reduce_sum(float val)` using `__shfl_xor_sync`. Then write `warp_reduce_max`. No shared memory allowed.

```
Reference: REPOS/flashinfer/include/flashinfer/math.cuh
           Look at math::shfl_xor_sync
Key insight: butterfly reduction tree — XOR offsets 16, 8, 4, 2, 1
```

---

**Day 3: Shared Memory Block Reduce (Phase 0.5)**

Why third: cross-warp reduction pattern used in every norm and softmax kernel.

Goal: write a block-level reduce-sum that handles any number of warps. 2D block `dim3(32, num_warps)`: lane=threadIdx.x, warp=threadIdx.y. Each warp reduces internally via shuffle, writes to `smem[ty]`, first warp reduces `smem`.

```
Reference: REPOS/flashinfer/include/flashinfer/norm.cuh lines 68–84
           This exact 2D-block warp+smem pattern is the entire RMSNorm reduce
Key pitfall: bank conflicts — use smem[ty] not smem[tx]
```

---

**Day 4–5: `cp.async` Async Pipeline (Phase 0.6)**

Why this week: Flash Attention decode hides memory latency using this. It's architecturally different from everything else.

Goal: write a double-buffered memcpy: while computing on buffer A, asynchronously load buffer B. Measure the speedup vs synchronous load.

```
Reference: REPOS/flashinfer/include/flashinfer/cp_async.cuh
           load_128b<PrefetchMode::kPrefetch>()
           commit_group() / wait_group<N>()
Key insight: cp.async bypasses the register file — DMA directly HBM→smem
Architecture: Ampere (SM80) and later only
```

---

### Week 2 — The Normalization Kernels (Start Here for Model Layers)

**Why RMSNorm before elementwise:** RMSNorm combines vectorized loads + warp shuffle + shared memory + 2-pass algorithm + quantized output — all the patterns at once. After this, elementwise kernels (Phase 1) will feel trivial.

---

**Day 6–7: RMSNorm (Phase 2.5)**

This is the single most instructive kernel in the whole curriculum. 200 lines in `norm.cuh` that teach you everything.

Goal: implement `RMSNormKernel<VEC_SIZE, T>` from scratch:
- Pass 1: vectorized load, compute `sum_sq += x*x`, warp shuffle reduce, smem cross-warp reduce
- Compute `rms_rcp = rsqrt(sum_sq / d + eps)`
- Pass 2: load input + weight, compute `output = input * rms_rcp * weight`, vectorized store

```
Reference: REPOS/flashinfer/include/flashinfer/norm.cuh
           RMSNormKernel lines 37–111
           RMSNorm launcher lines 113–146 (how vec_size is chosen: gcd(16/sizeof(T), d))

Test reference:
    rms_ref = x / torch.sqrt(torch.mean(x.float()**2, dim=-1, keepdim=True) + eps) * w
    torch.testing.assert_close(out, rms_ref.to(dtype), rtol=1e-2, atol=1e-2)

Profile target: >70% of peak HBM bandwidth
Measure: bytes = (batch * hidden * sizeof(T)) * 3  (read input, read weight, write output)
```

Key things to notice in the reference:
- `vec_size = gcd(16 / sizeof(T), d)` — auto-selects 128b alignment
- `dim3(32, num_warps)` — tx=lane, ty=warp — not the usual 1D block
- SM90 `griddepcontrol.wait` / `griddepcontrol.launch_dependents` — Programmatic Dependent Launch

---

**Day 8: Fused Add + RMSNorm (Phase 2.7)**

The most impactful single-kernel optimization in a transformer: fuses the residual add into the norm pass.

Goal: extend your RMSNorm to also accept a residual tensor. In pass 1, compute `x = input + residual` and store both to smem (the intermediate x) and residual (updated in-place). In pass 2, read x from smem instead of HBM.

```
Reference: REPOS/flashinfer/include/flashinfer/norm.cuh
           FusedAddRMSNormKernel lines 387–477
Key insight: smem_x caches the (input+residual) sum between pass 1 and pass 2
             Without fusion: 4 HBM accesses (read input, read residual, read x again, write output)
             With fusion: 3 HBM accesses (read input+residual together, write output+residual)

SGLang binding: REPOS/sglang/sgl-kernel/csrc/elementwise/fused_add_rms_norm_kernel.cu
```

---

**Day 9: Quantized RMSNorm — FP8 output (Phase 2.5 variant)**

Goal: extend RMSNormKernel to output FP8 (e4m3). Add: `scale_inv = 1.0 / scale`, clamp output to `[-448, 448]`, cast to `__nv_fp8_e4m3`.

```
Reference: REPOS/flashinfer/include/flashinfer/norm.cuh
           RMSNormQuantKernel lines 148–227
Why: every FP8 inference pipeline calls this before every GEMM
```

---

### Week 2 (continued) — Elementwise Kernels

Now that you understand vectorized loads + the model pipeline, elementwise kernels are fast to implement.

---

**Day 10: `act_and_mul` — SiLU Gate (Phase 1.3)**

The MLP bottleneck in every modern LLM.

Goal: one block per token. Input layout: `[gate_half | up_half]` concatenated. Load 8 fp16 from gate, 8 fp16 from up, compute `SiLU(gate[i]) * up[i]`, store.

```
Reference: REPOS/flashinfer/include/flashinfer/activation.cuh
           act_and_mul_kernel lines 29–64
Key: template<typename T, float (*Activation)(const float&)> — zero-cost abstraction
     The same kernel handles SiLU, GeLU, ReLU via template instantiation

SGLang Python call: sgl_kernel.silu_and_mul(input)  →  torch.ops.sgl_kernel.silu_and_mul
```

---

**Day 11: RoPE — Non-interleaved (Llama style) (Phase 1.5)**

Goal: inline device function (no separate launch). Each thread handles `vec_size` elements of one head. Load `x[half:]` (the paired half), compute `sin/cos`, rotate.

```
Reference: REPOS/flashinfer/include/flashinfer/pos_enc.cuh
           vec_apply_llama_rope lines 104–126
Key: __sincosf(embed, &sin, &cos)  — fused sin+cos in one PTX instruction
     Runs INSIDE the attention decode kernel — not a separate launch
     Pairing: x[i] pairs with x[i + rotary_dim/2]
```

---

### Week 3 — GEMM and Tensor Cores

**Do not rush this week.** The tiled GEMM concepts are prerequisite for Flash Attention prefill (week 5). Take the full week.

---

**Day 12–13: Naive GEMM → Tiled GEMM (Phase 3.1–3.2)**

Goal: implement `C = A @ B` twice:
1. Naive: one thread per output element, O(K) reads of A and B each — measure how bad it is
2. Tiled: load `TILE×TILE` blocks of A and B into smem, compute the tile in registers, `__syncthreads()` between tiles

```
Profile the speedup. Tiled should be 5–20× faster than naive.
Key metric: L2 cache hit rate — naive has 0%, tiled has high hit rate
```

---

**Day 14–15: WMMA GEMM — Tensor Cores (Phase 3.3)**

Goal: rewrite the tiled GEMM using `nvcuda::wmma`:
- `wmma::fragment<matrix_a, 16, 16, 16, __half, row_major> a_frag`
- `wmma::load_matrix_sync(a_frag, smem_a, 16)`
- `wmma::mma_sync(c_frag, a_frag, b_frag, c_frag)`
- `wmma::store_matrix_sync(output, c_frag, N, row_major)`

```
Reference: REPOS/flashinfer/include/flashinfer/mma.cuh
Measure: TFLOPS achieved vs peak ~44 TFLOPS (RTX 4060 Ti fp16 with 4th-gen tensor cores)
Key insight: tensor cores execute 16×16×16 MMA in one warp instruction
             This is 8× more throughput than scalar fp16 FMA
```

---

**Day 16: Shared Memory Swizzling (Phase 3.4)**

Goal: instrument your tiled GEMM with Nsight Compute. Look for `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld`. Apply swizzle pattern to eliminate bank conflicts.

```
Reference: REPOS/flashinfer/include/flashinfer/permuted_smem.cuh
           REPOS/flashinfer/include/flashinfer/frag_layout_swizzle.cuh
Key insight: 32-bank smem, bank = (byte_address / 4) % 32
             Swizzle XORs the column index with the row index to stagger access
```

---

### Week 4 — Flash Attention: Decode Path

**Read this first, before any code:**

Open `REPOS/flashinfer/include/flashinfer/attention/state.cuh` (83 lines).
Read `state_t::merge()` until you can explain it to someone without notes.
This is the mathematical primitive behind everything in weeks 4 and 5.

The 3-tuple `(o, m, d)` where:
- `m` = running max of pre-softmax logits
- `d` = running sum of `exp(logit - m)`
- `o` = running weighted sum of V: `exp(logit - m) * v / d`

Two partial states can be merged exactly:
```
m_new = max(m_a, m_b)
d_new = d_a * exp(m_a - m_new) + d_b * exp(m_b - m_new)
o_new = o_a * (d_a/d_new) * exp(m_a - m_new) + o_b * (d_b/d_new) * exp(m_b - m_new)
```

This is why Flash Attention doesn't need to recompute softmax when processing KV in tiles.

---

**Day 17: Online Softmax 1-pass (Phase 2.4)**

Before writing attention, implement online softmax over a 1D vector.

Goal: scan left-to-right, maintaining `(m, d)` state. At each element: update `m = max(m, x)`, rescale `d *= exp(m_old - m)`, add `d += exp(x - m)`. Final: `output[i] = exp(input[i] - m) / d`.

```
Verify: matches torch.softmax exactly (except numerics)
This IS flash attention — just without the V accumulation
```

---

**Day 18: Single-Sequence Flash Decode (Phase 4.3)**

Goal: implement single-request attention where one query token attends over a dense K/V sequence using online softmax.

```
Grid: (num_heads,)
Block: (head_dim / vec_size, 1)  — threads cover the head dimension
Algorithm:
  load q (stays in registers)
  for kv_tile in range(0, seq_len, BLOCK_N):
      load K tile → compute qk dot products → update (m, d, acc) online
  normalize: o = acc / d

Reference: REPOS/flashinfer/include/flashinfer/attention/decode.cuh
           compute_qk (lines 62–116) + update_local_state (lines 131–144)

Test: compare to torch.nn.functional.scaled_dot_product_attention (float32 ref)
```

---

**Day 19: MergeStateKernel (Phase 4.4)**

Goal: given two partial attention outputs `(v_a, lse_a)` and `(v_b, lse_b)` where `lse = m + log(d)`, compute the combined output.

```
Reference: REPOS/flashinfer/include/flashinfer/attention/cascade.cuh
           MergeStateKernel lines 44–71
           MergeStateInPlaceKernel lines 73–end

Also: REPOS/sglang/sgl-kernel/csrc/attention/merge_attn_states.cu
      This is the same kernel implemented independently in sgl-kernel
      Study both — they have different optimizations worth comparing

Grid: (num_tokens,)
Block: (head_dim / vec_size, num_heads)
Key: uint4 128-bit vectorized loads — 8 fp16 values per thread per instruction
```

---

**Day 20–21: Flash Decode with Split-K (Phase 4.5)**

Goal: extend to split the KV sequence across multiple thread blocks (one per split). Each block produces a partial `(v_partial, lse_partial)`. A second kernel (Phase 4.4) merges them.

```
Grid: (batch_size, num_heads, num_kv_splits)
Each block handles KV range [split_start, split_end)

Reference: REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/decode_attention.py
           _fwd_kernel_stage1 (Stage 1: compute partial attention per split)
           _fwd_kernel_stage2 (Stage 2: reduce partial outputs)
           _decode_att_m_fwd (launcher: grid = (batch, head, MAX_KV_SPLITS))

Read the Triton version first — it's the same algorithm but more readable.
Then implement in CUDA.
```

---

**Day 22–23: GQA Decode (Phase 4.6)**

Goal: extend split-K decode to handle Grouped Query Attention (multiple Q heads sharing one K/V head).

```
Change: kv_head_idx = q_head_idx / kv_group_num
        Each Q head still has its own output, but reads from shared K/V

In Triton: cur_kv_head = cur_head // kv_group_num
In CUDA: same logic, one line change in the kernel body

Models: Llama 3 (kv_group_num=8), Mistral (kv_group_num=8), Qwen (varies)
```

---

**Day 24–25: Paged Decode (Phase 4.7–4.8)**

Goal: replace the dense K/V buffer with a paged KV cache accessed via `kv_indices[]` page table.

```
Reference: REPOS/flashinfer/include/flashinfer/page.cuh
           paged_kv_t struct — understand every field

Data structures:
  k_data: [max_pages, num_kv_heads, page_size, head_dim]
  indices: [total_pages]          — flat page index table
  indptr:  [batch_size + 1]       — CSR: seq i uses indices[indptr[i]..indptr[i+1]]
  last_page_len: [batch_size]     — how full is the last page

Key change in kernel:
  old: k_ptr = k_buffer + token_idx * stride
  new: page_idx = kv_indices[kv_indptr[seq] + token // page_size]
       k_ptr = k_data + page_idx * stride_page + (token % page_size) * stride_n

Phase 4.8 addition: uint_fastdiv for token // page_size and token % page_size
  GPU integer division = 20+ cycles
  uint_fastdiv precomputes Newton-Raphson inverse → multiply + shift instead
  Reference: REPOS/flashinfer/include/flashinfer/fastdiv.cuh
```

By end of week 4 you have a production-grade paged flash decode kernel. This is the most important compute primitive in any LLM serving system.

---

### Week 5 — Flash Attention: Prefill Path

Prefill is harder than decode. It requires tensor core MMA, 2D tiling over both Q and KV sequences, and causal masking. Prerequisites: all of weeks 1–4.

---

**Day 26–27: Tiled Prefill with Online Softmax (Phase 5.3)**

Goal: implement FlashAttention-2 prefill. Tile Q into blocks of `BLOCK_M` rows, tile K/V into blocks of `BLOCK_N` rows. For each Q tile:
- Load Q block into smem
- For each KV tile: load K/V, compute `S = Q @ K^T / sqrt(d)`, apply causal mask, update `(m, d, acc)` online
- After all KV tiles: normalize `acc /= d`, write output

```
Reference: REPOS/flashinfer/include/flashinfer/attention/prefill.cuh
           SharedStorageQKVO struct (the smem union — Q/K/V smem reused for sync buffers)
           KernelTraits struct (tile shapes as compile-time constants)

Triton reference (more readable):
REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/prefill_attention.py

Block sizes (from extend_attention.py):
  RTX 4060 Ti (SM89): BLOCK_M=64, BLOCK_N=64 (100KB smem limit vs 192KB on A100)
  A100 (SM80): BLOCK_M=64, BLOCK_N=128
  H100 (SM90): BLOCK_M=64, BLOCK_N=64 (Hopper constraints for large head_dim)
  AMD:         BLOCK_M=64, BLOCK_N=64
```

---

**Day 28: Varlen (Ragged) Prefill (Phase 5.4)**

Goal: batch multiple sequences of different lengths into one kernel call without padding.

```
Input: packed tensor [total_tokens, num_heads, head_dim]
cu_seqlens_q: [batch_size + 1]  — cumulative token counts
cu_seqlens_k: [batch_size + 1]

Each block uses its blockIdx to look up which sequence it belongs to:
  seq_idx = find_batch_idx(blockIdx.x, cu_seqlens_q)  (binary search)
  q_start = cu_seqlens_q[seq_idx]
  q_len   = cu_seqlens_q[seq_idx + 1] - q_start

Reference: REPOS/flashinfer/csrc/batch_prefill.cu (binding)
           REPOS/flashinfer/include/flashinfer/attention/prefill.cuh BatchPrefillWithRaggedKVCacheKernel
```

---

**Day 29–30: Extend Attention (Phase 5.5)**

Goal: new Q tokens attend to both (a) existing KV cache tokens and (b) newly added KV tokens in one kernel.

```
Reference: REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/extend_attention.py
           extend_attention_fwd function — most complex kernel in the triton_ops folder

This is the core of continuous batching — a follow-up message attends to the
full conversation history that's already in the KV cache.
```

---

### Week 6 and Beyond

After week 5 you can read and understand the full FlashInfer and SGLang attention codebase. The remaining phases are important but each one is a self-contained new domain:

| Week | Phase | Focus |
|---|---|---|
| 6 | Phase 6 (Spec Decoding) | `merge_attn_states.cu`, `speculative_sampling.cu`, `sampling.cuh` |
| 7 | Phase 7 (MoE) | `moe_align_kernel.cu`, `moe_fused_gate.cu`, grouped GEMM |
| 8 | Phase 8 (Quantization) | `per_token_group_quant_8bit.cu`, `fp8_gemm_kernel.cu`, `awq_kernel.cu` |

---

## The Single Most Important File to Read First

Before writing any attention code, read this in full — it is only 83 lines:

```
REPOS/flashinfer/include/flashinfer/attention/state.cuh
```

The `state_t::merge()` function is the mathematical core of everything in weeks 4–6. If you understand it, Flash Attention is just indexing. If you don't, it will feel like magic forever.

---

## Key Reference Files by Week

| Week | What You're Building | Primary Reference | Secondary Reference |
|---|---|---|---|
| 1 | Vectorized load, warp shuffle, cp.async | `vec_dtypes.cuh`, `cp_async.cuh`, `math.cuh` | — |
| 2 | RMSNorm, FusedAddRMSNorm, act_and_mul, RoPE | `norm.cuh`, `activation.cuh`, `pos_enc.cuh` | `fused_add_rms_norm_kernel.cu` |
| 3 | Tiled GEMM, WMMA tensor cores, smem swizzle | `mma.cuh`, `permuted_smem.cuh` | CUTLASS tile docs |
| 4 | Flash decode, split-K, GQA, paged KV | `state.cuh`, `cascade.cuh`, `decode.cuh`, `page.cuh` | `triton_ops/decode_attention.py` |
| 5 | Flash prefill, varlen, extend | `prefill.cuh`, `triton_ops/extend_attention.py` | `hopper/mainloop.cuh` |
| 6 | Spec decoding kernels | `merge_attn_states.cu`, `sampling.cuh`, `speculative_sampling.cu` | `cascade.cuh` |
| 7 | MoE routing + grouped GEMM | `moe_align_kernel.cu`, flashinfer `fused_moe/` | `moe_fused_gate.cu` |
| 8 | Quantization | `per_token_group_quant_8bit.cu`, `fp8_gemm_kernel.cu` | `awq_kernel.cu`, `gptq_kernel.cu` |

---

## Profiling Checkpoints

**Your GPU: RTX 4060 Ti — 288 GB/s peak bandwidth, ~44 TFLOPS peak FP16**

At the end of each phase, hit these numbers before moving on:

| Kernel | Target Metric | Good Value (RTX 4060 Ti) | How to Measure |
|---|---|---|---|
| Vectorized copy | GDDR6X bandwidth | >230 GB/s (>80% of 288 GB/s) | `bytes / elapsed_ms / 1e6` |
| RMSNorm | GDDR6X bandwidth | >200 GB/s (>70% of peak) | `3 * batch * hidden * sizeof(T) / ms / 1e6` |
| Tiled GEMM (WMMA) | Tensor core TFLOPS | >22 TFLOPS (>50% of 44 TFLOPS) | `2*M*N*K / ms / 1e12` |
| Flash decode | GDDR6X bandwidth | >170 GB/s (>60% of peak) | `KV_bytes_read / ms / 1e6` |
| Flash prefill | Tensor core TFLOPS | >26 TFLOPS (>60% of peak) | `2*seq²*heads*head_dim / ms / 1e12` |

---

## Common Mistakes and How to Avoid Them

| Mistake | Symptom | Fix |
|---|---|---|
| Non-vectorized loads | Low bandwidth (<30% peak) | Use `float4` / `uint4` — always load 128 bits |
| Bank conflicts in smem | Low smem throughput in ncu | Apply swizzle: `col ^= row` in smem index |
| Missing `__syncthreads()` | Random wrong answers, works sometimes | Add sync between smem write and read phases |
| Unaligned smem writes | PTX compiler inserts extra moves | `alignas(16)` on smem arrays |
| Using `expf` instead of `ptx_exp2` | Slower softmax | `exp2(x * log2e)` = one PTX instruction vs `expf` multi-step |
| `-use_fast_math` in sampling | Wrong probability distributions | Only bypass for sampling (see `ieee_mul` in `sampling.cuh`) |
| Too many registers per thread | Low occupancy (<25%) | Check with `--ptxas-options=-v`, reduce with `-maxrregcount=64` |
| Smem too large | Kernel launch fails | RTX 4060 Ti max: 100KB per SM; A100: 164KB; H100: 228KB |
| Forgetting `cuda_synchronize` before timing | Wrong timing measurements | Always sync before `start.record()` and after `end.record()` |
| Testing only one shape | Kernel fails in production | Parametrize tests over 5+ shapes including edge cases (batch=1, hidden=128) |

---

## Tools Reference

```bash
# Timing
python -c "
import torch
start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
start.record(); fn(); end.record()
torch.cuda.synchronize()
print(f'{start.elapsed_time(end):.3f} ms')
"

# Hardware profiling
ncu --kernel-name "my_kernel" --set full python script.py
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed python script.py

# System timeline
nsys profile --trace=cuda,nvtx python script.py
nsys-ui report1.nsys-rep

# Register/smem usage (add to nvcc flags)
--ptxas-options=-v

# Check GPU architecture
python -c "import torch; print(torch.cuda.get_device_capability())"
# (8,9) = RTX 4060 Ti / RTX 4090 (Ada, your GPU)
# (8,0) = A100 (Ampere)
# (9,0) = H100 (Hopper)
# (10,0) = B200 (Blackwell)
```

---

## Anti-pattern: Do Not Get Stuck

Set a timer. If you spend more than 2 hours on any single kernel without making progress, look at the reference. The goal is understanding, not suffering.

The concepts compound — things about RMSNorm that confuse you now will click after you've written Flash Attention. Write the kernel, test it, note what you'd fix, move on. Come back later.

**Total time estimate:** 5–6 weeks at 3–4 hours/day. Faster if you already know the GPU arch well, slower if you're also learning ncu/profiling for the first time.
