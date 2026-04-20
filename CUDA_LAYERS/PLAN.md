# Learn CUDA by Building Deep Learning Inference Kernels

A bottom-up curriculum for implementing production LLM inference kernels from scratch.
Each phase introduces exactly one new hardware concept. Later kernels build on earlier ones.

**Reference repos:**
- SGLang kernels: `REPOS/sglang/sgl-kernel/csrc/` and `REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/`
- FlashInfer headers: `REPOS/flashinfer/include/flashinfer/`
- Key files: `vec_dtypes.cuh`, `cp_async.cuh`, `math.cuh`, `norm.cuh`, `pos_enc.cuh`, `activation.cuh`, `decode.cuh`, `prefill.cuh`, `state.cuh`, `cascade.cuh`, `page.cuh`, `sampling.cuh`

---

## Phase 0 — CUDA Fundamentals (Warm-up)

These are not model layers. They are the mechanical building blocks that every kernel in phases 1–8 relies on.
Read `vec_dtypes.cuh` and `cp_async.cuh` alongside these exercises.

| # | Exercise | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 0.1 | Vector Add (fp32) | Grid/block/thread indexing, `blockIdx`, `threadIdx`, global memory load/store | Everywhere — basis of all elementwise ops | The "hello world" of GPU parallelism; every element gets one thread |
| 0.2 | Vector Add (fp16 + bf16) | `__half`, `__nv_bfloat16`, `__float2half`, `__half2float` type casting | All LLM weights and activations are fp16/bf16 in production | LLMs run in fp16/bf16 to halve memory bandwidth; you must be comfortable with these types |
| 0.3 | 128-bit vectorized load/store | `float4`, `uint4`, `int4` — compiles to single `LDG.128` PTX instruction | Every bandwidth-bound kernel: norm, activation, RoPE, embedding | Loads 4 floats (or 8 fp16) in one instruction; saturates HBM bandwidth without this |
| 0.4 | Warp shuffle reduce (`__shfl_xor_sync`) | Warp-level communication with no shared memory, butterfly reduction tree | Inner loop of RMSNorm, LayerNorm, softmax, attention QK dot product | 32 threads share values in registers without touching shared memory — ~5× faster than smem reduce for small reductions |
| 0.5 | Shared memory block reduce | `extern __shared__`, `__syncthreads()`, bank conflict patterns | Cross-warp reduction in RMSNorm, LayerNorm, flash decode Stage 2 | When a reduction spans multiple warps, intermediate results must pass through shared memory |
| 0.6 | `cp.async` async pipeline | `cp.async.cg.shared.global`, `commit_group`, `wait_group`, double-buffer pattern | Flash Attention prefill and decode KV tile loading | Copies HBM→smem without stalling the compute pipeline; overlaps memory latency with MMA compute |

---

## Phase 1 — Elementwise Kernels (Memory-Bandwidth Bound)

One thread per element. The GPU is bottlenecked by HBM bandwidth, not compute.
Goal: write kernels that fully saturate memory bandwidth using vectorized 128-bit loads.
Read `activation.cuh` and `pos_enc.cuh`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 1.1 | ReLU (fp16, vectorized) | `#pragma unroll`, `vec_t<T, N>` templated loads, one block per token row | Legacy activation (BERT, early GPT) — rarely used in modern LLMs | Simplest nonlinearity; good vectorization practice before moving to complex activations |
| 1.2 | SiLU (`x * sigmoid(x)`) | Template function pointer `float (*Activation)(const float&)`, `__expf` | MLP gating in Llama, Mistral, Qwen, DeepSeek — all SwiGLU-based models | Modern LLMs use SiLU as the gate activation in their FFN; must be fused to avoid extra HBM traffic |
| 1.3 | GeLU | Approximation via `tanh(0.797 * (x + 0.044 * x^3))` vs exact `erf` | MLP in GPT-2/3, Falcon — older models before SwiGLU | Slightly more expensive than SiLU; approximation trades accuracy for speed |
| 1.4 | `act_and_mul` (SiLU gate fusion) | Strided 2D indexing: input layout `[gate_half \| up_half]`, single-pass fused read | **Every forward pass MLP block** in Llama 2/3, Mistral, Qwen, DeepSeek | After gate and up projections are computed, this fuses `SiLU(gate) * up` into one kernel — avoids materializing intermediate tensors and saves one full HBM round-trip |
| 1.5 | RoPE non-interleaved (Llama style) | `__sincosf` (fused sin+cos), half-dimension rotation: pairs `x[i]` with `x[i + d/2]` | Q and K projections in **every modern LLM**: Llama, Mistral, Qwen, Falcon | Rotary Positional Embeddings encode token position into the Q/K vectors before attention; runs fused inside the attention kernel — no separate launch |
| 1.6 | RoPE interleaved (GPT-NeoX style) | `i ^ 1` index trick pairs adjacent elements: `x[0]↔x[1]`, `x[2]↔x[3]` | GPT-NeoX, StableLM, some Mistral variants | Different rotation convention from Llama; the `i^1` trick is elegant bit manipulation worth knowing |
| 1.7 | Token embedding lookup (gather) | Irregular memory access, coalescing vs non-coalescing patterns, vocab sharding | **Very first operation** in every transformer: maps token IDs to dense vectors | `output[i] = weight[token_id[i]]` — random access into a large embedding table; coalescing depends on whether token IDs are adjacent |

---

## Phase 2 — Reduction Kernels (Compute-Bound with Cross-Thread Communication)

One thread block per row of the hidden state. Every thread in the block cooperates to compute a single scalar (sum, max, norm) over the full hidden dimension.
Goal: master the 2-pass warp-then-block reduction pattern used in every normalization layer.
Read `norm.cuh` and `math.cuh`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 2.1 | Row-wise sum | 2D block `(32, num_warps)`, `smem[ty]` inter-warp accumulate, `__syncthreads()` barrier | Building block for LayerNorm mean, softmax denominator | Foundational pattern: warp shuffle collapses 32 lanes to 1, shared memory collapses warps to 1 scalar |
| 2.2 | Row-wise max | Same 2D block pattern with `max` operator | Numerically stable softmax, flash attention online max tracking | Computing the maximum before exponentiation prevents overflow: `exp(x - max)` is safe, `exp(x)` is not |
| 2.3 | Softmax (3-pass naive) | Pass 1: find max. Pass 2: `exp(x - max)`. Pass 3: divide by sum | **Output logits** over vocabulary before sampling (32k–128k classes) | The probability distribution over next tokens; 3 passes × 128k vocab = expensive; motivates online softmax |
| 2.4 | Online softmax (1-pass fused) | Running `(m, d)` state: `m = max(m, x)`, `d = d * exp(m_old - m) + exp(x - m)` in one scan | Prerequisite for Flash Attention — same algorithm, applied to Q×K logits | Reduces 3 HBM passes to 1; this exact `(m, d)` state is what `state_t` in FlashInfer encodes |
| 2.5 | RMSNorm | `math::rsqrt` → GPU `MUFU.RSQ` instruction, `griddepcontrol` PDL for SM90+ | **Before every attention block and MLP block** in Llama, Mistral, Qwen | RMS normalization stabilizes training and inference; simpler than LayerNorm (no mean subtraction) and slightly faster |
| 2.6 | LayerNorm | Mean + variance (diff-of-squares trick `E[x²]-E[x]²`), optional shared-mem cache of hidden | BERT, GPT-2, Falcon — pre-2022 models | Full mean + variance normalization; USE_DIFF_OF_SQUARES flag in FlashInfer avoids second scan when smem is available |
| 2.7 | Fused Add + RMSNorm | Cache `x = input + residual` in shared memory between pass 1 and pass 2 — eliminates one HBM read | **Every residual connection** in every modern LLM, called after attention and after MLP | The residual add and normalization are always adjacent in the compute graph; fusing them saves one full hidden-state tensor read from HBM — significant at large batch sizes |

---

## Phase 3 — GEMM Kernels (Tensor-Core Bound)

Most of the FLOPs in a transformer live in linear projections. Learn how matrix multiplication maps to GPU hardware.
Goal: understand tiling, tensor cores, and register blocking before moving to Flash Attention.
Read `mma.cuh` and `permuted_smem.cuh`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 3.1 | Naive GEMM (global memory) | 2D thread grid, `C[row][col] += A[row][k] * B[k][col]` loop, global memory bottleneck | Conceptual baseline — never used in production | Forces you to see why naive GEMM is 100× slower than cuBLAS; each element of A and B is read O(N) times |
| 3.2 | Tiled GEMM (shared memory blocking) | Tile A and B into smem blocks, `__syncthreads()` between tiles, compute tile in registers | Still not production-grade but demonstrates the key idea | Each element now read only once into smem, then reused many times; this is the core idea behind all fast GEMMs |
| 3.3 | WMMA GEMM (FP16 tensor cores) | `nvcuda::wmma::fragment`, `load_matrix_sync`, `mma_sync`, `store_matrix_sync` — 16×16×16 tiles | **Q, K, V projections** and **lm_head** in every transformer during prefill | Tensor cores execute a 16×16 MMA in a single warp instruction — 8× throughput of scalar FP16 FMA; all production GEMMs use these |
| 3.4 | Register-blocked GEMM with swizzled smem layout | Swizzle pattern to eliminate shared memory bank conflicts, `#pragma unroll` register tiling | Hidden inside CUTLASS-based kernels (fp8_gemm, group_gemm) | Bank conflicts halve smem bandwidth; the swizzle pattern in `permuted_smem.cuh` is the production fix |
| 3.5 | Batched strided GEMM | `gridDim.z` iterates over batch, pointer arithmetic `A + batch * stride_A` | **Q, K, V projections** when sequence batch > 1, **O projection**, **FFN gate/up/down** | A transformer layer is one batched GEMM per projection; `batch_size × seq_len` rows, `hidden_dim` columns |
| 3.6 | FP8 GEMM with quantization scale | `__nv_fp8_e4m3` / `__nv_fp8_e5m2` types, per-tensor scale `C_fp16 = (A_fp8 × B_fp8) * scale`, clamp range `±448` | Quantized inference for DeepSeek, Llama 3.1 405B FP8 checkpoints | FP8 halves weight memory again vs BF16; the scale factor must be fused into the GEMM epilogue to avoid materializing an intermediate |

---

## Phase 4 — Flash Attention: Decode Path (Memory-Bandwidth Bound)

During token generation (decode), batch_size queries attend over the full KV cache sequence. This is memory-bandwidth-bound — the bottleneck is reading K and V from HBM.
Goal: build the paged flash decode kernel from scratch, understanding every data structure.
Read `state.cuh`, `cascade.cuh`, `decode.cuh`, `page.cuh`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 4.1 | `state_t` struct: merge two softmax states | Pure math (no kernel): `m = max(m_a, m_b)`, `d = d_a*exp(m_a-m) + d_b*exp(m_b-m)`, `o = (o_a * scale_a + o_b * scale_b)` | Mathematical primitive used in every flash attention and cascade merge | This 3-tuple `(o, m, d)` is the entire mathematical justification for split-K flash decoding — two partial attention outputs can be combined exactly without recomputation |
| 4.2 | Naive scaled dot-product attention (dense KV, single sequence) | 3-pass: materialize full `S = Q×K^T / sqrt(d)`, apply softmax row-wise, compute `O = S×V` | Attention in every transformer layer — conceptual baseline | Requires O(seq_len²) memory for the score matrix; for seq_len=128k this is 128GB — motivates Flash Attention |
| 4.3 | Single-sequence Flash Decode (dense KV, 1 split) | Online softmax scan over KV sequence, running `(m, d, acc)` state, normalize at end | Every decode step in every autoregressive LLM | Processes KV in tiles that fit in SRAM; never materializes the full score matrix; O(1) memory instead of O(n²) |
| 4.4 | `MergeStateKernel` (cascade / split-K reduction) | `vec_t::cast_load` / `cast_store`, one thread per `(token, head, vec)` element, `log-sum-exp` combine | Used when decode is split across multiple KV partitions (long sequences, chunked prefill) | Takes two partial attention outputs `(v_a, lse_a)` and `(v_b, lse_b)` and combines them into one correct output — enables parallelism over the KV sequence dimension |
| 4.5 | Flash Decode with split-K (multi-split, Stage 1 + Stage 2) | Grid `(batch, heads, num_kv_splits)` for Stage 1, separate reduction kernel for Stage 2, `num_kv_splits` tuning | Decode for **very long context** (>32k tokens): Llama 3.1, Qwen-Long, Claude | Without splitting, decode of a 128k-token sequence is sequential across 128k KV tiles; split-K parallelizes over those tiles |
| 4.6 | GQA (Grouped Query Attention) decode | `kv_head_idx = q_head_idx / kv_group_num`, multiple Q heads share one K/V head, amortize KV load | **Llama 2 (GQA), Llama 3 (GQA), Mistral, Qwen** — almost every modern model uses GQA | GQA reduces KV cache size by `kv_group_num` (e.g., 8×); without it a 70B model's KV cache is 8× larger — GQA is why large models can run at all |
| 4.7 | Paged Decode (`page_size = 1`, CSR index) | `paged_kv_t` struct: `k_data[max_pages, heads, page_size, d]`, `indices[]` page table, `indptr[]` CSR offsets | **All production LLM serving**: SGLang, vLLM, TGI — token-level paging | Without paging, you must pre-allocate max_seq_len memory per request; paging allocates pages on demand, enabling 5–10× higher throughput through better memory utilization |
| 4.8 | Paged Decode (configurable `page_size = N`) | `uint_fastdiv` for `t / page_size` and `t % page_size` without integer divide, `last_page_len` partial page | Production SGLang / FlashInfer default (page_size=16 or 64) | GPU integer division is 20+ cycles; `uint_fastdiv` precomputes Newton-Raphson inverse so division becomes multiply+shift — critical when called millions of times per second |

---

## Phase 5 — Flash Attention: Prefill Path (Compute-Bound)

During prompt processing (prefill), all input tokens are processed together. Compute-bound — the bottleneck is tensor core utilization.
Goal: build tiled MMA-based flash prefill with causal masking, varlen batching, and KV cache extend.
Read `prefill.cuh`, `mma.cuh`, `SharedStorageQKVO`, `KernelTraits`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 5.1 | Causal mask application | Triangular comparison `kv_idx > q_idx → -inf`, applied inside the softmax loop | **Every decoder-only LLM** during prefill (Llama, GPT, Qwen, DeepSeek) | Autoregressive generation requires each token to attend only to past tokens; the causal mask enforces this by setting future positions to `-inf` before softmax |
| 5.2 | Naive prefill attention (dense, no tiling) | 2D grid `(Q_tile, head)`, Q tile in registers, iterate K/V tiles from global memory | Conceptual baseline | Requires full `seq_len × seq_len` score matrix in registers/smem — impossible for seq_len > 512 |
| 5.3 | FlashAttention-2 prefill (tiled online softmax) | `SharedStorageQKVO` union (Q/K/V tiles share smem with sync buffers), tiled load + online `(m,d)` update, `mma_sync` for QK and PV | **Prefill for all context lengths** — this is what runs when you submit a prompt | Tiles Q into SRAM blocks of `BLOCK_M`, tiles K/V into blocks of `BLOCK_N`; the union of Q/K/V smem and the sync smem means zero SRAM waste |
| 5.4 | Varlen (ragged) prefill batching | `cu_seqlens_q[batch+1]` CSR offsets into packed token tensor, heterogeneous sequence lengths | **Production serving** — requests have different prompt lengths | A batch of 8 requests with lengths [512, 1024, 256, 2048, ...] are packed into one tensor without padding; `cu_seqlens` tells each block which slice to process |
| 5.5 | Extend attention (prefill + KV cache) | Q tokens attend to both: (a) newly added KV tokens, (b) existing cached KV tokens from previous turns | **Continuous batching / multi-turn chat** in SGLang, vLLM | When a user sends a follow-up message, the new tokens must attend to the full conversation history that's already in the KV cache — this is not a pure prefill nor a pure decode |
| 5.6 | FlashAttention-3 Hopper prefill (TMA mainloop) | `cp.async.bulk` TMA (Tensor Memory Accelerator), warp specialization (producer warps load, consumer warps compute), `cudaBarrierArrive/Wait` | H100 / H200 — used when `SM90` arch is detected in SGLang | TMA removes the address generation overhead from the compute warps; warp specialization overlaps load and compute with zero stalls; gives ~25% speedup over FA-2 on H100 |

---

## Phase 6 — Speculative Decoding Kernels

These kernels connect everything from phases 1–5 and add sampling and verification logic for speculative execution.
Read `speculative_sampling.cu`, `merge_attn_states.cu`, `sampling.cuh`, `topk.cuh`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 6.1 | Top-K sampling (CUB radix sort) | `cub::DeviceRadixSort::SortPairsDescending`, temporary storage allocation, device-wide algorithms | **Token sampling** after lm_head at every decode step | After the model produces logits over 128k vocabulary tokens, top-K selects the K highest-probability candidates before sampling — reduces compute and improves output quality |
| 6.2 | Top-P nucleus sampling | `cub::DeviceScan::InclusiveSum` for prefix sum, binary search for cumulative threshold | **Token sampling** — most production deployments use top-P | Top-P (nucleus) sampling accumulates probabilities in descending order and keeps only tokens whose cumulative probability ≤ p; more adaptive than top-K for varying entropy |
| 6.3 | Speculative rejection sampling | IEEE-754 PTX `mul.rn.f32` / `div.rn.f32` (bypass `-use_fast_math` for correctness), `curand_philox4x32` PRNG | **Speculative decoding verification** in SGLang EAGLE, n-gram, draft models | Given draft tokens from a fast model, compare draft vs target probabilities; accept draft token if `rand() < p_target / p_draft`; must be numerically exact to maintain distribution |
| 6.4 | `merge_attn_states` (prefix + suffix merge) | 128-bit `uint4` packed loads (8 fp16 in one instruction), write-once guard `if (pack_idx == 0)` for LSE output | **Chunked prefill** and **speculative decoding** with two separate attention computations | When the KV sequence is split into prefix (cached) and suffix (new), their attention outputs must be combined; this is the CUDA implementation of `state_t::merge()` from phase 4.1 |
| 6.5 | N-gram draft matching | Bitwise pack/unpack, token comparison with early exit, `packbit.cu` style | **N-gram speculative decoding** in SGLang (the fastest spec decoding method) | Searches the existing context for the longest suffix that matches the most recent tokens; if found, predicts the next N tokens without any model forward pass — effectively free throughput |

---

## Phase 7 — Mixture of Experts (MoE) Kernels

MoE replaces a single FFN with many expert FFNs, routing each token to 2–8 experts.
New challenge: irregular computation — each expert gets a different number of tokens, making standard batched GEMM inefficient.
Read `moe_align_kernel.cu`, `moe_fused_gate.cu`, flashinfer `fused_moe/`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 7.1 | Top-K expert routing (per-token) | Warp-level arg-max / arg-top-K, scatter token indices to per-expert lists | **Every MoE FFN**: DeepSeek-V3 (256 experts, top-8), Mixtral (8 experts, top-2), Qwen-MoE | The router is a small linear layer whose output is converted to expert assignments; this kernel does the argmax/top-K and builds the routing tables |
| 7.2 | Token alignment / histogram | `atomicAdd` per-expert counter, `cudaMemset`, expert padding to GEMM-friendly tile sizes | Inside MoE dispatch before grouped GEMM | GEMM requires aligned dimensions; if expert 3 gets 73 tokens but GEMM tile = 128, we must pad to 128 — this kernel computes padding offsets |
| 7.3 | Fused gate + sigmoid/softmax routing | Combine top-K selection, softmax over expert logits, and gating weight computation in one kernel | DeepSeek-V3 fine-grained expert selection, Mixtral expert gating | Avoids multiple passes over the router logit tensor; gating weights (how much of each expert's output to keep) are computed here |
| 7.4 | Grouped GEMM (batched expert FFN) | CUTLASS `GroupedGemm` with pointer arrays `[A_expert0, A_expert1, ...]`, variable-M grouped problem | The **dominant cost** in MoE inference — replaces two dense FFN GEMMs | Each expert is one GEMM; 256 experts = 256 GEMMs, but grouped GEMM fuses them; critical for DeepSeek-V3 where 256 experts run every layer |
| 7.5 | MoE output reduction (weighted sum) | Scatter-add with expert gating weights: `out[token] += weight[expert] * expert_out[token, expert]` | After all expert GEMMs, recombine into the main residual stream | Each token's final MoE output is a weighted average of its assigned experts' outputs — this scatter-add merges them back into the token dimension |

---

## Phase 8 — Quantization Kernels

Quantization reduces weight/activation precision from BF16 → INT8 / FP8 / INT4 to save memory and increase throughput.
Read `fp8_gemm_kernel.cu`, `gptq/gptq_kernel.cu`, `awq_kernel.cu`, `per_token_group_quant_8bit.cu`.

| # | Kernel | New CUDA Concept | Where It Appears in a Model | Function in Inference |
|---|---|---|---|---|
| 8.1 | Per-token dynamic FP8 quantization | `fmaxf` abs-max reduction over hidden dim, `clamp(±448)`, `cast_store` to `__nv_fp8_e4m3` | Applied to activations **before every FP8 GEMM** — Q/K/V/O projections, FFN gate/up/down | FP8 has max value 448 (E4M3); must compute per-token scale `scale = max(|x|) / 448` and multiply before casting — done in one fused pass |
| 8.2 | Per-channel weight dequantization | Broadcast scale multiply: `x_fp16 = x_int8 * scale[channel]`, vectorized over hidden dim | **Preprocess** before INT8 matmul — converts stored INT8 weights back to compute dtype | Weights are stored in INT8 for memory but dequantized to FP16/BF16 just before matmul; done in small fast kernel, not part of the GEMM itself |
| 8.3 | INT8 GEMM with per-tensor scale | INT8 accumulate to INT32, then `C_fp16 = INT32_acc * (scale_A * scale_B)` in epilogue | W8A8 quantized inference (static quantization for batch throughput) | INT8 gives 2× memory bandwidth and 4× tensor core throughput vs FP16; the scale fusion in the GEMM epilogue is what makes this exact |
| 8.4 | Blockwise FP8 GEMM (DeepSeek style) | Per-128-element block scales for weights, per-token scales for activations, scale applied in GEMM epilogue | **DeepSeek-V3 / DeepSeek-R1** FP8 linear layers | Blockwise scaling allows more accurate quantization than per-tensor (one scale per 128 elements vs one scale per whole matrix); the block scales are fused into the CUTLASS epilogue |
| 8.5 | AWQ 4-bit weight unpacking | Bit manipulation: `(packed >> shift) & 0xF` to unpack 4-bit weights, `group_size` per-group scales and zeros | **AWQ quantized** Llama, Mistral, Qwen models (4-bit for ~4× memory reduction) | 4-bit weights are packed 2-per-byte; the unpack kernel converts them to FP16 just before GEMM; group_size=128 means one scale+zero per 128 weights |
| 8.6 | GPTQ 4-bit GEMM | Column-packing of 4-bit weights, marlin-style register layout for tensor core consumption | **GPTQ quantized** models — community standard for <70B models on consumer GPUs | GPTQ uses a different packing layout than AWQ; Marlin reformats the packed weights once at load time into a register-friendly layout, amortizing the unpacking cost |

---

## Difficulty Curve Summary

```
Phase 0  ──  "Hello GPU"          1 thread = 1 element, global memory, types
Phase 1  ──  Elementwise          Vectorized 128b loads, template dispatch, HBM saturation
Phase 2  ──  Reductions           Warp shuffle + shared memory, 2-pass across hidden dim
Phase 3  ──  GEMM                 Tiling, tensor core WMMA/MMA, smem swizzle, epilogues
Phase 4  ──  Decode Attention     Online softmax state, paged KV, split-K, GQA
Phase 5  ──  Prefill Attention    MMA tiling, varlen batching, TMA/warp specialization (H100)
Phase 6  ──  Spec Decoding        Multi-kernel pipelines, PRNG, exact IEEE arithmetic
Phase 7  ──  MoE Routing          Irregular compute, grouped GEMM, atomic scatter
Phase 8  ──  Quantization         Type packing/unpacking, scale fusion, mixed precision
```

---

## Reading Checklist (what each phase unlocks in the real codebase)

| After Phase | You Can Now Fully Read |
|---|---|
| Phase 0 | `vec_dtypes.cuh`, `cp_async.cuh`, `math.cuh`, `utils.cuh` |
| Phase 1 | `activation.cuh` (act_and_mul), `pos_enc.cuh` (inline RoPE device functions) |
| Phase 2 | `norm.cuh` (RMSNorm, FusedAddRMSNorm, LayerNorm, all Quant variants) |
| Phase 3 | `mma.cuh`, `permuted_smem.cuh`, `frag_layout_swizzle.cuh`, `prefill.cuh` KernelTraits struct |
| Phase 4 | `state.cuh`, `cascade.cuh`, `decode.cuh`, `page.cuh`, SGLang `triton_ops/decode_attention.py` |
| Phase 5 | `prefill.cuh` SharedStorageQKVO, `hopper/mainloop.cuh`, `hopper/prefill_sm90.cuh`, `flash_extension.cc` |
| Phase 6 | `sampling.cuh`, `speculative_sampling.cu`, `merge_attn_states.cu`, `eagle_utils.cu`, `ngram_utils.cu` |
| Phase 7 | `moe_align_kernel.cu`, `moe_fused_gate.cu`, `moe_topk_softmax_kernels.cu`, flashinfer `fused_moe/` |
| Phase 8 | `fp8_gemm_kernel.cu`, `per_token_group_quant_8bit.cu`, `gptq/gptq_kernel.cu`, `awq_kernel.cu` |

---

## Notes on Architecture Variants

> **Your machine: RTX 4060 Ti — Ada Lovelace, sm_89, 16 GB GDDR6X, 288 GB/s, CUDA 13.0**
>
> Use `-arch=sm_89` for all exercises. See the "What Works on sm_89" table below.

| Feature | sm_89 (RTX 4060 Ti) | sm_80 (A100) | sm_90 (H100) |
|---|---|---|---|
| `cp.async` async pipeline | YES | YES | YES |
| Warp shuffle `__shfl_xor_sync` | YES | YES | YES |
| 4th-gen Tensor Cores (FP8/FP16/BF16) | YES | NO (3rd gen) | YES |
| Hardware FP8 conversion | YES | NO | YES |
| `griddepcontrol` / PDL | **NO** | NO | YES |
| TMA (Tensor Memory Accelerator) | **NO** | NO | YES |
| CUTLASS CuTe TMA (MLA kernel) | **NO** | NO | YES |
| HBM bandwidth | 288 GB/s | 2,039 GB/s | 3,350 GB/s |
| Shared memory per SM | 100 KB | 192 KB | 228 KB |
| L2 cache | 32 MB | 40 MB | 50 MB |
| SM count | 34 | 108 | 132 |
| Peak FP16 TFLOPS (tensor cores) | ~44 | 312 | 989 |

**What to skip on sm_89:**
- Phase 5.6 (FlashAttention-3 TMA mainloop) — requires sm_90
- Phase 6.2 (CUTLASS MLA decode) — requires sm_90
- Any `griddepcontrol` / PDL in norm kernels — skip those code paths in `norm.cuh`

**Phases 0–5.5, 6.1 (all non-TMA attention), 7 (FP8 — native on Ada), 8 (sampling) all work on sm_89.**

- **SM80 (A100)**: All phases 0–6 apply directly. `cp.async` available. No TMA. 2,039 GB/s.
- **SM90 (H100/H200)**: Phase 5.6 adds TMA + warp specialization. `griddepcontrol` PDL in norm kernels. CUTLASS 3.x CuTe for MLA.
- **SM100 (B200/GB200)**: `cutlass_sm100_mla/` for MLA decode. MXFP8 / NVFP4 quantization (Phase 8). Blackwell FMHA in flashinfer.
- **HIP (AMD MI300X)**: Phase 4–5 kernels have HIP variants. Block sizes differ (BLOCK_N=8 for decode). `is_hip()` guards in SGLang triton kernels.

Total kernel count: ~55 kernels across 9 phases.
Estimated learning time: 4–8 weeks for someone comfortable with C++, starting from zero GPU programming experience.

---

## Appendix A — How SGLang Calls Triton and CUDA Kernels

Understanding the call stack from Python model code down to GPU silicon is essential before writing your own kernels.

### The Three Integration Patterns

#### Pattern 1: `torch.ops.sgl_kernel.*` (CUDA kernels compiled via CMake)

This is the primary path for all custom CUDA kernels in `sgl-kernel/csrc/`.

```
Python model code
    │
    │  sgl_kernel.fused_add_rmsnorm(input, residual, weight, eps)
    ▼
sgl-kernel/python/sgl_kernel/elementwise.py
    │
    │  torch.ops.sgl_kernel.fused_add_rmsnorm.default(input, residual, weight, eps, enable_pdl)
    ▼
TORCH_LIBRARY_FRAGMENT(sgl_kernel, m) { m.def("fused_add_rmsnorm(...)"); }
    │   [registered in csrc/elementwise/fused_add_rms_norm_kernel.cu]
    │   [loaded from sgl_kernel/sm90/common_ops.cpython-3xx.so at import time]
    ▼
norm::FusedAddRMSNorm<scalar_t><<<grid, block, smem, stream>>>(...)
    │   [from flashinfer/include/flashinfer/norm.cuh, included by the .cu file]
    ▼
GPU executes FusedAddRMSNormKernel PTX
```

**Key files in this chain:**
- `sgl-kernel/python/sgl_kernel/elementwise.py` — Python API (`fused_add_rmsnorm`, `silu_and_mul`, `rotary_embedding`, etc.)
- `sgl-kernel/python/sgl_kernel/load_utils.py` — detects GPU arch (SM90 vs SM100) and `dlopen`s the right `.so` at import time
- `sgl-kernel/CMakeLists.txt` — builds two shared libraries: `common_ops_sm90_build` (with `-use_fast_math`) and `common_ops_sm100_build` (without, for numerical precision)
- `sgl-kernel/csrc/*/` — the actual `.cu` files; each calls `TORCH_LIBRARY_FRAGMENT` to register ops

**How the `.so` gets loaded:**
```python
# load_utils.py does this at sgl_kernel import time:
spec = importlib.util.spec_from_file_location("common_ops", "sgl_kernel/sm90/common_ops.cpython-311.so")
common_ops = importlib.util.module_from_spec(spec)
spec.loader.exec_module(common_ops)
# This triggers TORCH_LIBRARY_FRAGMENT registration, making torch.ops.sgl_kernel.* available
```

#### Pattern 2: Triton `@triton.jit` kernels (JIT compiled on first call)

Used for the attention backends (`triton_backend.py`) and fallback paths.

```
Python model code
    │
    │  from sglang.srt.layers.attention.triton_backend import TritonAttnBackend
    ▼
triton_backend.py → calls decode_attention_fwd(q, k_buffer, v_buffer, ...)
    │
    ▼
triton_ops/decode_attention.py
    │
    │  _fwd_kernel_stage1[grid](Q, K_Buffer, V_Buffer, ...,
    │       BLOCK_DMODEL=128, BLOCK_N=64, kv_group_num=8)
    ▼
Triton JIT compiles _fwd_kernel_stage1 to PTX on first call,
caches compiled kernel for subsequent calls with same shapes
    │
    ▼
GPU executes generated PTX
```

**Key files:**
- `python/sglang/srt/layers/attention/triton_ops/decode_attention.py` — `_fwd_kernel_stage1`, `_fwd_kernel_stage2`, `decode_attention_fwd`
- `python/sglang/srt/layers/attention/triton_ops/extend_attention.py` — `extend_attention_fwd`
- `python/sglang/srt/layers/attention/triton_ops/prefill_attention.py` — `context_attention_fwd`
- `python/sglang/srt/layers/attention/triton_backend.py` — `TritonAttnBackend` wires these together

**Triton compilation trigger:**
```python
# First call with new shapes triggers JIT:
_fwd_kernel_stage1[grid](Q, K, V, ..., BLOCK_DMODEL=128)
# Triton caches: ~/.triton/cache/<hash>/kernel.ptx
# Subsequent calls with same (shapes, dtypes, constexprs) reuse cached PTX
```

**Triton vs CUDA tradeoffs:**
- Triton: faster to write, auto-tunes, but ~10–20% slower than hand-written CUDA for the same kernel
- CUDA: maximum control, maximum performance, harder to write correctly
- SGLang uses Triton as a readable reference / AMD fallback, CUDA for production on NVIDIA

#### Pattern 3: FlashInfer wrapper (prebuilt `.whl` CUDA kernels)

Used for the default `flashinfer_backend.py`.

```
Python model code
    │
    ▼
flashinfer_backend.py
    │  wrapper = BatchDecodeWithPagedKVCacheWrapper(workspace_buffer)
    │  wrapper.plan(kv_indptr, kv_indices, ...)     # host-side planning (no GPU)
    │  output = wrapper.run(q, kv_data)             # launches CUDA kernel
    ▼
flashinfer Python package (installed wheel)
    │  calls into flashinfer/csrc/batch_decode.cu via pybind11
    ▼
BatchDecodeWithPagedKVCacheKernel<<<grid, block>>>(...)
    │  [precompiled into the flashinfer wheel, arch-specific .so]
    ▼
GPU executes kernel
```

**The `plan()` + `run()` split:** FlashInfer separates planning (computing grid dims, workspace layout, tile counts) from execution. This amortizes the Python-side scheduling overhead across multiple decode steps when using CUDA graphs.

### Backend Selection in SGLang

```python
# python/sglang/srt/layers/attention/attention_registry.py
BACKENDS = {
    "flashinfer": FlashInferAttnBackend,   # default on NVIDIA SM80+
    "triton":     TritonAttnBackend,        # fallback / AMD
    "flashattention": FlashAttentionBackend, # uses sgl_kernel.flash_attn
}
# Selected via --attention-backend CLI flag or auto-detected
```

---

## Appendix B — Compiling Your Own CUDA Kernels

### Option 1: `torch.utils.cpp_extension.load_inline` (fastest for learning)

No CMake, no build system. Compile a `.cu` file from Python at runtime.

```python
import torch
from torch.utils.cpp_extension import load_inline

cuda_source = """
#include <cuda_fp16.h>

__global__ void rms_norm_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weight,
    __half* __restrict__ output,
    int hidden_dim, float eps)
{
    int row = blockIdx.x;
    float sum_sq = 0.f;

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[row * hidden_dim + i]);
        sum_sq += x * x;
    }
    // warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        sum_sq += __shfl_xor_sync(0xffffffff, sum_sq, offset);

    float rms_rcp = rsqrtf(sum_sq / float(hidden_dim) + eps);

    for (int i = threadIdx.x; i < hidden_dim; i += blockDim.x) {
        float x = __half2float(input[row * hidden_dim + i]);
        float w = __half2float(weight[i]);
        output[row * hidden_dim + i] = __float2half(x * rms_rcp * w);
    }
}

torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, float eps) {
    auto output = torch::empty_like(input);
    int batch = input.size(0), hidden = input.size(1);
    rms_norm_kernel<<<batch, 128>>>(
        (__half*)input.data_ptr(), (__half*)weight.data_ptr(),
        (__half*)output.data_ptr(), hidden, eps);
    return output;
}
"""

cpp_source = 'torch::Tensor rms_norm(torch::Tensor input, torch::Tensor weight, float eps);'

module = load_inline(
    name="my_rms_norm",
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=["rms_norm"],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-arch=sm_89"],  # Ada (RTX 4060 Ti)
    verbose=True,
)

# Usage:
x = torch.randn(4, 4096, device="cuda", dtype=torch.float16)
w = torch.ones(4096, device="cuda", dtype=torch.float16)
out = module.rms_norm(x, w, 1e-6)
```

**Compilation happens once**, result cached in `/tmp/torch_extensions/`. Subsequent Python runs reuse the cached `.so`.

### Option 2: `torch.utils.cpp_extension.CUDAExtension` (proper package)

For kernels you want to install and reuse like `sgl-kernel`.

```
my_kernel/
├── setup.py
├── my_kernels.cu        ← kernel implementation
└── my_kernels_binding.cpp  ← TORCH_LIBRARY registration
```

```python
# setup.py
from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="my_kernels",
    ext_modules=[
        CUDAExtension(
            name="my_kernels",
            sources=["my_kernels.cu", "my_kernels_binding.cpp"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": [
                    "-O3",
                    "--use_fast_math",
                    "-arch=sm_89",           # Ada Lovelace (RTX 4060 Ti)
                    # "-arch=sm_80",         # A100 (Ampere)
                    # "-arch=sm_90",         # H100 (Hopper)
                    "-std=c++17",
                ],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
```

```bash
pip install -e .          # editable install, recompiles on change
# OR
python setup.py build_ext --inplace   # builds .so in current directory
```

### Option 3: CMake (sgl-kernel style, for production)

Follow the pattern in `REPOS/sglang/sgl-kernel/CMakeLists.txt`:

```cmake
cmake_minimum_required(VERSION 3.26)
project(my_kernel LANGUAGES CXX CUDA)
find_package(Torch REQUIRED)
find_package(Python COMPONENTS Interpreter Development.Module REQUIRED)

Python_add_library(my_kernel MODULE my_kernel.cu)
target_compile_options(my_kernel PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math -arch=sm_89>)  # Ada (RTX 4060 Ti)
target_link_libraries(my_kernel PRIVATE ${TORCH_LIBRARIES})
```

```bash
mkdir build && cd build
cmake .. -DCMAKE_PREFIX_PATH=$(python -c "import torch; print(torch.utils.cmake_prefix_path)")
make -j$(nproc)
```

### Key `nvcc` Flags Reference

| Flag | Effect | When to Use |
|---|---|---|
| `-arch=sm_89` | Compile for Ada Lovelace | **RTX 4060 Ti (your GPU)**, RTX 4090 |
| `-arch=sm_80` | Compile for A100 (Ampere) | A100 / RTX 3090 |
| `-arch=sm_90` | Compile for H100 (Hopper) | H100 / H200 |
| `-arch=sm_90a` | H100 with Hopper-specific PTX (TMA, wgmma) | H100 only, faster |
| `--use_fast_math` | Enables `ftz`, approximate trig, fused mul-add | Most kernels except sampling |
| `-O3` | Maximum compiler optimization | Always |
| `-std=c++17` | Required for CUTLASS, FlashInfer | Always |
| `--ptxas-options=-v` | Print register/smem usage per kernel | When debugging occupancy |
| `-maxrregcount=64` | Cap registers per thread (increases occupancy) | When register-limited |
| `-lineinfo` | Embed source line info in binary (for Nsight) | Profiling builds |
| `-G` | Full debug info, disable optimizations | Debugging only — 10–100× slower |

---

## Appendix C — Testing Kernels

### The Standard Testing Pattern

Every kernel in `sgl-kernel/tests/` follows this exact pattern — copy it:

```python
# tests/test_my_rmsnorm.py
import pytest
import torch
import my_kernels  # your compiled extension

# --- Reference implementation (pure PyTorch, slow but provably correct) ---
def rms_norm_reference(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6):
    rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + eps)
    return (x.float() / rms * weight.float()).to(x.dtype)

# --- Parametrized test: cover all realistic shapes ---
@pytest.mark.parametrize("hidden_dim", [128, 512, 1024, 4096, 8192, 11008])
@pytest.mark.parametrize("batch_size", [1, 4, 16, 32])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rms_norm(hidden_dim, batch_size, dtype):
    torch.manual_seed(42)
    x = torch.randn(batch_size, hidden_dim, device="cuda", dtype=dtype)
    w = torch.randn(hidden_dim, device="cuda", dtype=dtype)

    ref = rms_norm_reference(x, w)
    out = my_kernels.rms_norm(x, w, 1e-6)

    # fp16 accumulates error — use loose tolerances
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

# --- Edge cases: always test these explicitly ---
def test_rms_norm_single_element():
    x = torch.randn(1, 128, device="cuda", dtype=torch.float16)
    w = torch.ones(128, device="cuda", dtype=torch.float16)
    out = my_kernels.rms_norm(x, w, 1e-6)
    ref = rms_norm_reference(x, w)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)

def test_rms_norm_large_batch():
    x = torch.randn(512, 4096, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(4096, device="cuda", dtype=torch.bfloat16)
    out = my_kernels.rms_norm(x, w, 1e-5)
    ref = rms_norm_reference(x, w, 1e-5)
    torch.testing.assert_close(out, ref, rtol=1e-1, atol=1e-1)

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
```

```bash
# Run tests:
pytest tests/test_my_rmsnorm.py -v
pytest tests/test_my_rmsnorm.py -v -k "hidden_dim-4096"   # filter
pytest tests/test_my_rmsnorm.py -v --tb=short              # compact traceback
```

### Tolerance Guide for fp16/bf16

| Precision | Typical `rtol` | Typical `atol` | Notes |
|---|---|---|---|
| fp32 vs fp32 | 1e-5 | 1e-5 | Should be nearly exact |
| fp16 kernel vs fp32 ref | 1e-2 | 1e-2 | fp16 has ~3 decimal places |
| bf16 kernel vs fp32 ref | 1e-1 | 1e-1 | bf16 has only ~2 decimal places |
| fp8 kernel vs fp16 ref | 0.5 | 0.5 | fp8 is coarse — check statistical properties instead |

### Testing Attention Kernels Specifically

Attention kernels need special care because of the large input space:

```python
def attention_reference(q, k, v, causal=True, scale=None):
    """Exact reference using PyTorch scaled_dot_product_attention."""
    # q: [batch, heads, seq_q, head_dim]
    # k: [batch, heads, seq_k, head_dim]
    # v: [batch, heads, seq_k, head_dim]
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # Use float32 for reference to avoid fp16 numerical issues
    q, k, v = q.float(), k.float(), v.float()
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, attn_mask=None, is_causal=causal, scale=scale
    ).to(q.dtype)

@pytest.mark.parametrize("seq_len", [64, 256, 1024, 4096])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("num_heads", [8, 32])
def test_flash_decode(seq_len, head_dim, num_heads):
    q = torch.randn(1, num_heads, 1, head_dim, device="cuda", dtype=torch.float16)
    k = torch.randn(1, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    v = torch.randn(1, num_heads, seq_len, head_dim, device="cuda", dtype=torch.float16)
    ref = attention_reference(q, k, v, causal=False)
    out = my_flash_decode(q, k, v)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
```

---

## Appendix D — Profiling Kernels

### Tool 1: `torch.cuda.Event` — Quick Wall-Clock Timing

```python
import torch

def bench(fn, warmup=10, iters=100):
    """GPU timing with CUDA events — the correct way to time GPU kernels."""
    # Warmup: fill caches, JIT compile Triton kernels, prime cuBLAS
    for _ in range(warmup):
        fn()

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()

    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters  # milliseconds per call

# Example:
x = torch.randn(32, 4096, device="cuda", dtype=torch.float16)
w = torch.ones(4096, device="cuda", dtype=torch.float16)

ms = bench(lambda: my_kernels.rms_norm(x, w, 1e-6))
bytes_accessed = x.numel() * x.element_size() * 2 + w.numel() * w.element_size()  # read x + write out, read w
bandwidth_gb = bytes_accessed / ms / 1e6  # GB/s
print(f"Time: {ms:.3f} ms | Bandwidth: {bandwidth_gb:.1f} GB/s")
# RTX 4060 Ti peak: 288 GB/s. Target >230 GB/s (>80%). If <100 GB/s, check access patterns.
```

### Tool 2: `torch.profiler` — Python-Level Timeline

```python
import torch
from torch.profiler import profile, record_function, ProfilerActivity

with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    record_shapes=True,
    with_stack=False,
) as prof:
    for _ in range(20):
        with record_function("rms_norm"):
            out = my_kernels.rms_norm(x, w, 1e-6)
        torch.cuda.synchronize()

# Print table:
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

# Export Chrome trace (open at chrome://tracing):
prof.export_chrome_trace("trace.json")
```

### Tool 3: Nsight Compute (`ncu`) — Hardware-Level Profiling

This is the most important profiling tool. It shows register usage, shared memory, memory throughput, compute utilization, and warp stalls.

```bash
# Profile a Python script — collect all metrics (slow, use for one kernel at a time):
ncu --set full python test_my_rmsnorm.py

# Profile with specific metrics only (faster):
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,\
    l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
    sm__warps_active.avg.pct_of_peak_sustained_active \
    python test_my_rmsnorm.py

# Save report and open in Nsight Compute GUI:
ncu --export report.ncu-rep python test_my_rmsnorm.py
ncu-ui report.ncu-rep

# Profile only your kernel (not PyTorch internals):
ncu --kernel-name "rms_norm_kernel" --set full python test_my_rmsnorm.py
```

**Key metrics to look at:**

| Metric | What It Tells You | Good Value |
|---|---|---|
| `sm__throughput` | Overall SM utilization | >80% (compute-bound kernels) |
| `l2_global_load_bytes` | Bytes read from HBM via L2 | Should match your input tensor size |
| `l1tex__t_bytes` | L1 cache hit rate | High = good locality |
| `sm__warps_active` | Occupancy (active warps / max warps) | >50% for most kernels |
| `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld` | Memory transaction efficiency | Should be 128 bytes (full cache line) |
| `stall_long_sb` | Warp stall from long scoreboard (memory latency) | Low = memory latency hidden |
| `stall_mio_throttle` | Warp stall from memory instruction throttle | Low = not memory-instruction limited |

### Tool 4: Nsight Systems (`nsys`) — System-Level Timeline

Use this to see the big picture: which kernels run, how long each takes, are there CPU↔GPU sync stalls, are CUDA graphs working.

```bash
# Record full trace:
nsys profile --trace=cuda,nvtx,osrt python my_inference_script.py

# Open in Nsight Systems GUI:
nsys-ui report1.nsys-rep
```

Add NVTX markers in your code to annotate the timeline:

```python
import torch.cuda.nvtx as nvtx

nvtx.range_push("prefill")
output = model.forward(tokens)
nvtx.range_pop()
```

### Tool 5: Kernel Occupancy Calculator

Before profiling, check theoretical occupancy to understand hardware limits:

```python
# Check register/smem usage after compiling:
# nvcc --ptxas-options=-v my_kernel.cu
# Output: ptxas info: Used 32 registers, 4096 bytes smem, 0 bytes cmem[0]

# RTX 4060 Ti (sm_89) limits:
# - 65536 registers per SM, 32 threads per warp, max 48 warps per SM
# - If your kernel uses 64 regs: 65536 / (64 regs * 32 threads) = 32 warps = 67% occupancy
# - Max smem per SM: 100KB. If kernel uses 48KB smem: max 2 blocks per SM

# In Python: check with
from torch.cuda import get_device_properties
prop = get_device_properties(0)
print(f"Max shared memory per block: {prop.max_shared_memory_per_block // 1024} KB")
print(f"Max registers per block: {prop.max_registers_per_block}")
print(f"Multiprocessors: {prop.multi_processor_count}")
print(f"Max threads per SM: {prop.max_threads_per_multi_processor}")
```

### Profiling Workflow for Each New Kernel

```
1. Write kernel → 2. Test correctness → 3. Measure bandwidth/TFLOPS
     │                      │                        │
     │              torch.testing.assert_close    torch.cuda.Event timing
     │
4. Profile with ncu → 5. Identify bottleneck → 6. Fix and repeat
     │
     ├── Memory-bound?  → Check L2 hit rate, vectorize loads (use float4/uint4)
     ├── Compute-bound? → Check SM throughput, use tensor cores (WMMA/MMA)
     ├── Occupancy-low? → Reduce registers (--maxrregcount) or smem
     └── Warp stalls?   → Add cp.async pipelining, unroll loops
```

### Roofline Analysis — Know When You're Done Optimizing

For every kernel, compute its **arithmetic intensity** (FLOPs / bytes) and compare to the hardware roofline:

```
Arithmetic Intensity = Total FLOPs / Total Bytes Read+Written

RTX 4060 Ti specs (your machine):
  Peak FP16 TFLOPS:     ~44 TFLOPS (with 4th-gen tensor cores)
  Peak GDDR6X Bandwidth: 288 GB/s
  Ridge point:          44e12 / 288e9 = ~153 FLOP/byte

A100 specs (reference):
  Peak FP16 TFLOPS:     312 TFLOPS (with tensor cores)
  Peak HBM Bandwidth:   2000 GB/s
  Ridge point:          312e12 / 2000e9 = 156 FLOP/byte

RMSNorm (batch=32, hidden=4096):
  FLOPs:  32 * 4096 * (2 + 1) ≈ 400k ops  (square, sum, sqrt, multiply per element)
  Bytes:  32 * 4096 * 2bytes * 2 (read+write) = 524KB
  Intensity: ~0.75 FLOP/byte → DEEPLY memory-bound
  → Optimize for bandwidth (vectorize, coalesce) NOT compute

Flash Decode (batch=32, heads=32, kv_len=4096, head_dim=128):
  FLOPs:  32 * 32 * 4096 * 128 * 2 ≈ 1 TFLOP  (Q×K dot products + softmax + P×V)
  Bytes:  KV cache read = 32 * 4096 * 128 * 2bytes * 2 = 67MB
  Intensity: ~15 FLOP/byte → Still memory-bound during decode
  → Focus on saturating HBM bandwidth, minimize redundant reads
```

**Rule of thumb:**
- `intensity < ridge_point` → memory-bound → optimize memory access (vectorize, coalesce, reduce reads)
- `intensity > ridge_point` → compute-bound → optimize compute (tensor cores, unroll, occupancy)
