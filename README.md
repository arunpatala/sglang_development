# SGLang Development — Layer-by-Layer LLM Inference

A ground-up implementation of LLM inference, built one architectural layer at a time. Each layer adds a single focused improvement over the previous one. The code targets Qwen3-0.6B on a single GPU, but the techniques and structures map directly to production runtimes like [SGLang](https://github.com/sgl-project/sglang).

---

## Table of Contents

| Layer | Topic | Sections |
|------:|-------|---------|
| [0](#layer-0--naive-inference) | Naive Inference | 7 |
| [1](#layer-1--manual-decode) | Manual Decode | 8 |
| [2](#layer-2--kv-cache) | KV Cache | 9 |
| [3](#layer-3--static-batching) | Static Batching | 8 |
| [4](#layer-4--model-loading) | Model Loading | 4 |
| [5](#layer-5--model-layers) | Model Layers from Scratch | 7 |
| [6](#layer-6--continuous-batching) | Continuous Batching | 8 |
| [7](#layer-7--packed-batching) | Packed Batching (FlashInfer) | 6 |
| [8](#layer-8--paged-attention) | Paged Attention — KVPool | 7 |
| [9](#layer-9--paged-attention-regtotokenpool--triton) | Paged Attention — ReqToTokenPool + Triton | 7 |
| [10](#layer-10--chunked-prefill) | Chunked Prefill | — |
| [11](#layer-11--prefix-caching) | Prefix Caching | — |
| [12](#layer-12--gptq-quantization) | GPTQ Quantization | — |
| [13](#layer-13--speculative-decoding) | Speculative Decoding | — |

---

## Layer 0 — Naive Inference

**Directory:** [`CODE_LAYERS/layer0_naive_inference/`](CODE_LAYERS/layer0_naive_inference/)

Calls `model.generate()` and returns text. HuggingFace handles everything — config parsing, weight loading, the decode loop, sampling, and KV caching. The server exposes a single `/generate` endpoint. This layer is the baseline: it establishes what the output looks like and gives a number to beat.

**Sections:**

- [`01_llm_intro.md`](CODE_LAYERS/layer0_naive_inference/lesson/01_llm_intro.md) — What Large Language Models Are and What They Do
- [`02_tokens.md`](CODE_LAYERS/layer0_naive_inference/lesson/02_tokens.md) — Tokens: The Unit of Language
- [`03_messages_and_chat_template.md`](CODE_LAYERS/layer0_naive_inference/lesson/03_messages_and_chat_template.md) — Messages, Roles, and the Chat Template
- [`04_model_loading_and_generate_loop.md`](CODE_LAYERS/layer0_naive_inference/lesson/04_model_loading_and_generate_loop.md) — Model Loading and the Generate Loop
- [`05_inference_server.md`](CODE_LAYERS/layer0_naive_inference/lesson/05_inference_server.md) — The Inference Server
- [`06_metrics_and_benchmark.md`](CODE_LAYERS/layer0_naive_inference/lesson/06_metrics_and_benchmark.md) — Metrics and Benchmarking
- [`07_whats_next.md`](CODE_LAYERS/layer0_naive_inference/lesson/07_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer0_naive_inference/lesson/summary.md) — Layer 0 Summary

---

## Layer 1 — Manual Decode

**Directory:** [`CODE_LAYERS/layer1_manual_decode/`](CODE_LAYERS/layer1_manual_decode/)

Opens the `model.generate()` black box: calls `model()` directly, receives raw logits, and implements the token-selection and sampling loop manually. Computation is identical to Layer 0 but every step is now visible in our code. This is the foundation all future optimisations build on.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer1_manual_decode/lesson/00_outline.md) — Lesson Outline
- [`01_intro.md`](CODE_LAYERS/layer1_manual_decode/lesson/01_intro.md) — Token-Level Generation
- [`02_the_forward_pass.md`](CODE_LAYERS/layer1_manual_decode/lesson/02_the_forward_pass.md) — The Forward Pass
- [`03_causal_masking.md`](CODE_LAYERS/layer1_manual_decode/lesson/03_causal_masking.md) — Causal Masking
- [`04_prefill_and_decode.md`](CODE_LAYERS/layer1_manual_decode/lesson/04_prefill_and_decode.md) — Prefill and Decode
- [`05_logits_and_sampling.md`](CODE_LAYERS/layer1_manual_decode/lesson/05_logits_and_sampling.md) — Logits, Softmax, and Sampling
- [`06_the_decode_loop.md`](CODE_LAYERS/layer1_manual_decode/lesson/06_the_decode_loop.md) — The Decode Loop
- [`07_ttft_and_tpot.md`](CODE_LAYERS/layer1_manual_decode/lesson/07_ttft_and_tpot.md) — TTFT and TPOT
- [`08_whats_next.md`](CODE_LAYERS/layer1_manual_decode/lesson/08_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer1_manual_decode/lesson/summary.md) — Layer 1 Summary

---

## Layer 2 — KV Cache

**Directory:** [`CODE_LAYERS/layer2_kv_cache/`](CODE_LAYERS/layer2_kv_cache/)

Eliminates the redundant recomputation of key and value tensors on every decode step. After prefill, K/V tensors for all previous tokens are saved in a `KVCache` object and handed back to the model on the next step, so each decode step processes only the single new token. `server.py` and `benchmark.py` are untouched; the benchmark numbers change dramatically.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer2_kv_cache/lesson/00_outline.md) — Lesson Outline
- [`01_the_decode_loop.md`](CODE_LAYERS/layer2_kv_cache/lesson/01_the_decode_loop.md) — The Decode Loop
- [`02_keys_values_queries.md`](CODE_LAYERS/layer2_kv_cache/lesson/02_keys_values_queries.md) — Keys, Values, and Queries in Attention
- [`03_the_cache.md`](CODE_LAYERS/layer2_kv_cache/lesson/03_the_cache.md) — What the KV Cache Is
- [`04_past_key_values.md`](CODE_LAYERS/layer2_kv_cache/lesson/04_past_key_values.md) — `past_key_values` in HuggingFace
- [`05_kv_cache_class.md`](CODE_LAYERS/layer2_kv_cache/lesson/05_kv_cache_class.md) — The `KVCache` Class
- [`06_prefill_and_decode.md`](CODE_LAYERS/layer2_kv_cache/lesson/06_prefill_and_decode.md) — Prefill and Decode in Layer 2
- [`07_the_full_loop.md`](CODE_LAYERS/layer2_kv_cache/lesson/07_the_full_loop.md) — The Full Loop
- [`08_benchmark_results.md`](CODE_LAYERS/layer2_kv_cache/lesson/08_benchmark_results.md) — What the Numbers Show
- [`09_whats_next.md`](CODE_LAYERS/layer2_kv_cache/lesson/09_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer2_kv_cache/lesson/summary.md) — Layer 2 Summary

---

## Layer 3 — Static Batching

**Directory:** [`CODE_LAYERS/layer3_static_batching/`](CODE_LAYERS/layer3_static_batching/)

Processes B requests simultaneously instead of one at a time. Prefill becomes a single `[B, max_prompt_len]` forward pass; every decode step sends `[B, 1]` to the GPU. GPU utilisation climbs from ~5% toward 80%+ as batch size increases. Requests are left-padded to the same length; position IDs are assigned per-token to maintain correct RoPE positions.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer3_static_batching/lesson/00_outline.md) — Lesson Outline
- [`01_the_decode_loop.md`](CODE_LAYERS/layer3_static_batching/lesson/01_the_decode_loop.md) — The Decode Loop
- [`02_the_tokenizer.md`](CODE_LAYERS/layer3_static_batching/lesson/02_the_tokenizer.md) — The Tokenizer
- [`03_left_padding_and_position_ids.md`](CODE_LAYERS/layer3_static_batching/lesson/03_left_padding_and_position_ids.md) — Left Padding and Position IDs
- [`04_batched_sampling.md`](CODE_LAYERS/layer3_static_batching/lesson/04_batched_sampling.md) — Batched Sampling
- [`05_the_finished_mask.md`](CODE_LAYERS/layer3_static_batching/lesson/05_the_finished_mask.md) — The Finished Mask
- [`06_padding_waste.md`](CODE_LAYERS/layer3_static_batching/lesson/06_padding_waste.md) — Padding Waste and GPU Utilisation
- [`07_the_full_loop.md`](CODE_LAYERS/layer3_static_batching/lesson/07_the_full_loop.md) — The Full Loop
- [`08_whats_next.md`](CODE_LAYERS/layer3_static_batching/lesson/08_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer3_static_batching/lesson/summary.md) — Layer 3 Summary

---

## Layer 4 — Model Loading

**Directory:** [`CODE_LAYERS/layer4_model_loading/`](CODE_LAYERS/layer4_model_loading/)

Takes ownership of config parsing and weight loading, replacing `AutoModelForCausalLM.from_pretrained`. Reads `config.json` directly, resolves local or Hub model paths (calling `snapshot_download` when needed), and loads `.safetensors` shards manually into model parameters with a key-remapping step. The HuggingFace model class is still used for the forward pass; weight loading is now ours.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer4_model_loading/lesson/00_outline.md) — Lesson Outline
- [`01_the_decode_loop.md`](CODE_LAYERS/layer4_model_loading/lesson/01_the_decode_loop.md) — The Decode Loop
- [`02_config_loading.md`](CODE_LAYERS/layer4_model_loading/lesson/02_config_loading.md) — Config Loading
- [`03_weight_loading.md`](CODE_LAYERS/layer4_model_loading/lesson/03_weight_loading.md) — Weight Loading
- [`04_whats_next.md`](CODE_LAYERS/layer4_model_loading/lesson/04_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer4_model_loading/lesson/summary.md) — Layer 4 Summary

---

## Layer 5 — Model Layers from Scratch

**Directory:** [`CODE_LAYERS/layer5_model_layers/`](CODE_LAYERS/layer5_model_layers/)

Replaces the HuggingFace forward computation with our own `Qwen3Model`. Five files implement the architecture: `norm.py` (RMSNorm), `mlp.py` (SwiGLU MLP), `rope.py` (RoPE and `apply_rotary_pos_emb`), `attention.py` (GQA with per-head QK norm), and `decoder_layer.py`. The stack is composed into `Qwen3ForCausalLM` in `model/qwen3.py`. No HuggingFace classes remain in the forward path.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer5_model_layers/lesson/00_outline.md) — Lesson Outline
- [`01_the_decode_loop.md`](CODE_LAYERS/layer5_model_layers/lesson/01_the_decode_loop.md) — The Decode Loop
- [`02_rmsnorm_and_decoder_layer.md`](CODE_LAYERS/layer5_model_layers/lesson/02_rmsnorm_and_decoder_layer.md) — RMSNorm and the Decoder Layer
- [`03_rope.md`](CODE_LAYERS/layer5_model_layers/lesson/03_rope.md) — Rotary Position Embedding
- [`04_additive_mask.md`](CODE_LAYERS/layer5_model_layers/lesson/04_additive_mask.md) — The Additive Attention Mask
- [`05_attention.md`](CODE_LAYERS/layer5_model_layers/lesson/05_attention.md) — Attention: GQA and Per-Head QK Norm
- [`06_the_full_loop.md`](CODE_LAYERS/layer5_model_layers/lesson/06_the_full_loop.md) — The Full Loop
- [`07_whats_next.md`](CODE_LAYERS/layer5_model_layers/lesson/07_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer5_model_layers/lesson/summary.md) — Layer 5 Summary

---

## Layer 6 — Continuous Batching

**Directory:** [`CODE_LAYERS/layer6_continuous_batching/`](CODE_LAYERS/layer6_continuous_batching/)

Replaces the blocking `generate_batch` call with a continuous-batching scheduler. A background thread maintains a `waiting_queue` and a `running` list. On every iteration it prefills waiting requests one at a time (B=1), runs one decode step across all running requests (B=N), then evicts any that emitted EOS. A finished request is removed immediately — no head-of-line blocking. Each request stores its own growing `PerReqKVCache`.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer6_continuous_batching/lesson/00_outline.md) — Lesson Outline
- [`01_the_scheduler_loop.md`](CODE_LAYERS/layer6_continuous_batching/lesson/01_the_scheduler_loop.md) — The Scheduler Loop
- [`02_the_request.md`](CODE_LAYERS/layer6_continuous_batching/lesson/02_the_request.md) — The Request
- [`03_per_request_kv_cache.md`](CODE_LAYERS/layer6_continuous_batching/lesson/03_per_request_kv_cache.md) — Per-Request KV Cache
- [`04_prefill.md`](CODE_LAYERS/layer6_continuous_batching/lesson/04_prefill.md) — Prefill
- [`05_the_decode_step.md`](CODE_LAYERS/layer6_continuous_batching/lesson/05_the_decode_step.md) — The Decode Step
- [`06_thread_safety.md`](CODE_LAYERS/layer6_continuous_batching/lesson/06_thread_safety.md) — Thread Safety
- [`07_the_full_loop.md`](CODE_LAYERS/layer6_continuous_batching/lesson/07_the_full_loop.md) — The Full Loop
- [`08_whats_next.md`](CODE_LAYERS/layer6_continuous_batching/lesson/08_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer6_continuous_batching/lesson/summary.md) — Layer 6 Summary

---

## Layer 7 — Packed Batching (FlashInfer)

**Directory:** [`CODE_LAYERS/layer7_packed_batching/`](CODE_LAYERS/layer7_packed_batching/)

Removes padding waste in the attention kernel. Layer 6's `BatchedKVCache` left-padded every request's KV history to `max_kv_len`; with one 500-token request and fifteen 3-token requests, 97% of attention compute was masked zeros. Layer 7 replaces this with `PackedKVCache`, which concatenates all KV histories back-to-back and uses FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` to attend only over real tokens. Introduces `ForwardBatch`/`ForwardMode` and the `PagedBackend` dispatch pattern.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer7_packed_batching/lesson/00_outline.md) — Lesson Outline
- [`01_from_padded_to_packed.md`](CODE_LAYERS/layer7_packed_batching/lesson/01_from_padded_to_packed.md) — From Padded to Packed
- [`02_packed_kv_cache.md`](CODE_LAYERS/layer7_packed_batching/lesson/02_packed_kv_cache.md) — The Packed KV Cache
- [`03_packing_and_attending.md`](CODE_LAYERS/layer7_packed_batching/lesson/03_packing_and_attending.md) — Packing and Attending
- [`04_the_attention_dispatch.md`](CODE_LAYERS/layer7_packed_batching/lesson/04_the_attention_dispatch.md) — The Attention Dispatch
- [`05_the_full_loop.md`](CODE_LAYERS/layer7_packed_batching/lesson/05_the_full_loop.md) — The Full Loop
- [`06_whats_next.md`](CODE_LAYERS/layer7_packed_batching/lesson/06_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer7_packed_batching/lesson/summary.md) — Layer 7 Summary

---

## Layer 8 — Paged Attention (KVPool)

**Directory:** [`CODE_LAYERS/layer8_paged_attention/`](CODE_LAYERS/layer8_paged_attention/)

Eliminates the per-step float KV gather. Layer 7's `PackedKVCache.update()` still gathered all historical K/V from separate `PerReqKVCache` allocations and concatenated them into a new buffer every step — O(total_kv_tokens) float bandwidth, growing without bound. Layer 8 replaces all per-request caches with a single global `KVPool`: two pre-allocated flat tensors per layer sized to 85% of free GPU memory. Every token is written once into a pool slot and never moved. The decode step builds a `kv_indices` integer array (slot addresses) and passes it to FlashInfer's `BatchDecodeWithPagedKVCacheWrapper`, which reads the pool directly.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer8_paged_attention/lesson/00_outline.md) — Lesson Outline
- [`01_from_packed_to_paged.md`](CODE_LAYERS/layer8_paged_attention/lesson/01_from_packed_to_paged.md) — From Packed to Paged
- [`02_the_kv_pool.md`](CODE_LAYERS/layer8_paged_attention/lesson/02_the_kv_pool.md) — The KV Pool
- [`03_prefill_writing_to_pool.md`](CODE_LAYERS/layer8_paged_attention/lesson/03_prefill_writing_to_pool.md) — Prefill: Writing to the Pool
- [`04_decode_indexing_the_pool.md`](CODE_LAYERS/layer8_paged_attention/lesson/04_decode_indexing_the_pool.md) — Decode: Indexing the Pool
- [`05_the_attention_dispatch.md`](CODE_LAYERS/layer8_paged_attention/lesson/05_the_attention_dispatch.md) — The Attention Dispatch
- [`06_the_full_loop.md`](CODE_LAYERS/layer8_paged_attention/lesson/06_the_full_loop.md) — The Full Loop
- [`07_whats_next.md`](CODE_LAYERS/layer8_paged_attention/lesson/07_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer8_paged_attention/lesson/summary.md) — Layer 8 Summary

---

## Layer 9 — Paged Attention: ReqToTokenPool + Triton

**Directory:** [`CODE_LAYERS/layer9_paged_attention2/`](CODE_LAYERS/layer9_paged_attention2/)

Eliminates the remaining O(Σ kv_tokens) costs in Layer 8's decode step. Introduces `ReqToTokenPool`, a GPU-resident `[max_batch, max_pages_per_req]` int32 table that stores page indices on-device from the moment of prefill. A Triton kernel (`create_flashinfer_kv_indices_triton`) reads that table and builds `kv_indices` entirely on-GPU — no Python loop, no CPU-to-GPU copy of historical index data. Simultaneously raises `page_size` to 16, reducing `kv_indices` length, page alloc frequency, and pool lookup count by 16×. The KV pool shape changes from `[total_slots, n_kv, dim]` to `[total_pages, page_size, n_kv, dim]`. The decode wrapper is created once at startup and reused; `kv_indptr` is computed via `torch.cumsum` on-GPU.

**Sections:**

- [`00_outline.md`](CODE_LAYERS/layer9_paged_attention2/lesson/00_outline.md) — Lesson Outline
- [`01_from_page_size_1_to_paged.md`](CODE_LAYERS/layer9_paged_attention2/lesson/01_from_page_size_1_to_paged.md) — From Page-size-1 to Paged
- [`02_req_to_token_pool.md`](CODE_LAYERS/layer9_paged_attention2/lesson/02_req_to_token_pool.md) — ReqToTokenPool
- [`03_paged_kvpool_and_prefill.md`](CODE_LAYERS/layer9_paged_attention2/lesson/03_paged_kvpool_and_prefill.md) — Paged KVPool and Prefill
- [`04_decode_conditional_alloc_and_gpu_indexing.md`](CODE_LAYERS/layer9_paged_attention2/lesson/04_decode_conditional_alloc_and_gpu_indexing.md) — Decode: Conditional Allocation and GPU Indexing
- [`05_the_triton_kernel.md`](CODE_LAYERS/layer9_paged_attention2/lesson/05_the_triton_kernel.md) — The Triton Kernel
- [`06_the_full_loop.md`](CODE_LAYERS/layer9_paged_attention2/lesson/06_the_full_loop.md) — The Full Loop
- [`07_whats_next.md`](CODE_LAYERS/layer9_paged_attention2/lesson/07_whats_next.md) — What Comes Next
- [`summary.md`](CODE_LAYERS/layer9_paged_attention2/lesson/summary.md) — Layer 9 Summary

---

## Layer 10 — Chunked Prefill

**Directory:** [`CODE_LAYERS/layer11_chunked_prefill/`](CODE_LAYERS/layer11_chunked_prefill/)

Replaces the B=1 sequential prefill loop with batched and chunked prefill. Multiple waiting requests are prefilled together in one EXTEND forward pass. Long prompts are split into chunks of up to `chunk_size` tokens; each chunk is co-scheduled with the ongoing decode requests in the same forward pass using a mixed-mode attention kernel. Eliminates the stall where a long prompt blocks the decode loop for the duration of its prefill.

**README:** [`README.md`](CODE_LAYERS/layer11_chunked_prefill/README.md)

---

## Layer 11 — Prefix Caching

**Directory:** [`CODE_LAYERS/layer12_prefix_caching/`](CODE_LAYERS/layer12_prefix_caching/)

Adds prefix caching via a compressed radix tree. When two requests share a common prefix — a system prompt, a few-shot template — the second request reuses the KV pages computed by the first, skipping prefill for those tokens entirely. Page indices for cached prefixes are looked up in the radix tree and written directly into `ReqToTokenPool`, with no transformer forward pass for the shared portion.

**README:** [`README.md`](CODE_LAYERS/layer12_prefix_caching/README.md)

---

## Layer 12 — GPTQ Quantization

**Directory:** [`CODE_LAYERS/layer13_quantization_gptq/`](CODE_LAYERS/layer13_quantization_gptq/)

Adds 4-bit GPTQ quantization support. Model weights are stored as packed int4 values and dequantized on-the-fly during matrix multiplies using the `sgl_kernel.gptq_gemm` fused CUDA kernel (ExLlama v2). Reduces the model's GPU memory footprint by roughly 4×, leaving more space for the KV pool and enabling larger batch sizes on the same hardware.

**README:** [`README.md`](CODE_LAYERS/layer13_quantization_gptq/README.md)

---

## Layer 13 — Speculative Decoding

**Directory:** [`CODE_LAYERS/layer14_speculative_decoding/`](CODE_LAYERS/layer14_speculative_decoding/)

Adds standalone speculative decoding. A small draft model proposes N tokens in serial; the large target model verifies all N+1 positions in a single parallel EXTEND pass. Accepted tokens are committed to the KV pool; the first rejected token triggers a rollback. When acceptance rates are high, effective throughput approaches N× the target model's single-token rate.

**README:** [`README.md`](CODE_LAYERS/layer14_speculative_decoding/README.md)
