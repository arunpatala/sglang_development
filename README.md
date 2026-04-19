# SGLang Development — Layer-by-Layer LLM Inference

A ground-up implementation of LLM inference, built one architectural layer at a time. Each layer adds a single focused improvement over the previous one. The code targets Qwen3-0.6B on a single GPU, but the techniques and structures map directly to production runtimes like [SGLang](https://github.com/sgl-project/sglang).

---

## Table of Contents

| Layer | Topic | First Lesson |
|------:|-------|-------------|
| [0](#layer-0--naive-inference) | Naive Inference | [`01_llm_intro.md`](CODE_LAYERS/layer0_naive_inference/lesson/01_llm_intro.md) |
| [1](#layer-1--manual-decode) | Manual Decode | [`00_outline.md`](CODE_LAYERS/layer1_manual_decode/lesson/00_outline.md) |
| [2](#layer-2--kv-cache) | KV Cache | [`00_outline.md`](CODE_LAYERS/layer2_kv_cache/lesson/00_outline.md) |
| [3](#layer-3--static-batching) | Static Batching | [`00_outline.md`](CODE_LAYERS/layer3_static_batching/lesson/00_outline.md) |
| [4](#layer-4--model-loading) | Model Loading | [`00_outline.md`](CODE_LAYERS/layer4_model_loading/lesson/00_outline.md) |
| [5](#layer-5--model-layers) | Model Layers from Scratch | [`00_outline.md`](CODE_LAYERS/layer5_model_layers/lesson/00_outline.md) |
| [6](#layer-6--continuous-batching) | Continuous Batching | [`00_outline.md`](CODE_LAYERS/layer6_continuous_batching/lesson/00_outline.md) |
| [7](#layer-7--packed-batching) | Packed Batching (FlashInfer) | [`00_outline.md`](CODE_LAYERS/layer7_packed_batching/lesson/00_outline.md) |
| [8](#layer-8--paged-attention) | Paged Attention — KVPool | [`00_outline.md`](CODE_LAYERS/layer8_paged_attention/lesson/00_outline.md) |
| [9](#layer-9--paged-attention-regtotokenpool--triton) | Paged Attention — ReqToTokenPool + Triton | [`00_outline.md`](CODE_LAYERS/layer9_paged_attention2/lesson/00_outline.md) |
| [10](#layer-10--chunked-prefill) | Chunked Prefill | [`README.md`](CODE_LAYERS/layer11_chunked_prefill/README.md) |
| [11](#layer-11--prefix-caching) | Prefix Caching | [`README.md`](CODE_LAYERS/layer12_prefix_caching/README.md) |
| [12](#layer-12--gptq-quantization) | GPTQ Quantization | [`README.md`](CODE_LAYERS/layer13_quantization_gptq/README.md) |
| [13](#layer-13--speculative-decoding) | Speculative Decoding | [`README.md`](CODE_LAYERS/layer14_speculative_decoding/README.md) |

---

## Layer 0 — Naive Inference

**Directory:** [`CODE_LAYERS/layer0_naive_inference/`](CODE_LAYERS/layer0_naive_inference/)

Calls `model.generate()` and returns text. HuggingFace handles everything — config parsing, weight loading, the decode loop, sampling, and KV caching. The server exposes a single `/generate` endpoint. This layer is the baseline: it establishes what the output looks like and gives a number to beat.

**Lesson:** [`lesson/01_llm_intro.md`](CODE_LAYERS/layer0_naive_inference/lesson/01_llm_intro.md)

---

## Layer 1 — Manual Decode

**Directory:** [`CODE_LAYERS/layer1_manual_decode/`](CODE_LAYERS/layer1_manual_decode/)

Opens the `model.generate()` black box: calls `model()` directly, receives raw logits, and implements the token-selection and sampling loop manually. Computation is identical to Layer 0 but every step is now visible in our code. This is the foundation all future optimisations build on.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer1_manual_decode/lesson/00_outline.md)

---

## Layer 2 — KV Cache

**Directory:** [`CODE_LAYERS/layer2_kv_cache/`](CODE_LAYERS/layer2_kv_cache/)

Eliminates the redundant recomputation of key and value tensors on every decode step. After prefill, K/V tensors for all previous tokens are saved in a `KVCache` object and handed back to the model on the next step, so each decode step processes only the single new token. `server.py` and `benchmark.py` are untouched; the benchmark numbers change dramatically.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer2_kv_cache/lesson/00_outline.md)

---

## Layer 3 — Static Batching

**Directory:** [`CODE_LAYERS/layer3_static_batching/`](CODE_LAYERS/layer3_static_batching/)

Processes B requests simultaneously instead of one at a time. Prefill becomes a single `[B, max_prompt_len]` forward pass; every decode step sends `[B, 1]` to the GPU. GPU utilisation climbs from ~5% toward 80%+ as batch size increases. Requests are left-padded to the same length; position IDs are assigned per-token to maintain correct RoPE positions.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer3_static_batching/lesson/00_outline.md)

---

## Layer 4 — Model Loading

**Directory:** [`CODE_LAYERS/layer4_model_loading/`](CODE_LAYERS/layer4_model_loading/)

Takes ownership of config parsing and weight loading, replacing `AutoModelForCausalLM.from_pretrained`. Reads `config.json` directly, resolves local or Hub model paths (calling `snapshot_download` when needed), and loads `.safetensors` shards manually into model parameters with a key-remapping step. The HuggingFace model class is still used for the forward pass; weight loading is now ours.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer4_model_loading/lesson/00_outline.md)

---

## Layer 5 — Model Layers from Scratch

**Directory:** [`CODE_LAYERS/layer5_model_layers/`](CODE_LAYERS/layer5_model_layers/)

Replaces the HuggingFace forward computation with our own `Qwen3Model`. Five files implement the architecture: `norm.py` (RMSNorm), `mlp.py` (SwiGLU MLP), `rope.py` (RoPE and `apply_rotary_pos_emb`), `attention.py` (GQA with per-head QK norm), and `decoder_layer.py`. The stack is composed into `Qwen3ForCausalLM` in `model/qwen3.py`. No HuggingFace classes remain in the forward path.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer5_model_layers/lesson/00_outline.md)

---

## Layer 6 — Continuous Batching

**Directory:** [`CODE_LAYERS/layer6_continuous_batching/`](CODE_LAYERS/layer6_continuous_batching/)

Replaces the blocking `generate_batch` call with a continuous-batching scheduler. A background thread maintains a `waiting_queue` and a `running` list. On every iteration it prefills waiting requests one at a time (B=1), runs one decode step across all running requests (B=N), then evicts any that emitted EOS. A finished request is removed immediately — no head-of-line blocking. Each request stores its own growing `PerReqKVCache`.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer6_continuous_batching/lesson/00_outline.md)

---

## Layer 7 — Packed Batching (FlashInfer)

**Directory:** [`CODE_LAYERS/layer7_packed_batching/`](CODE_LAYERS/layer7_packed_batching/)

Removes padding waste in the attention kernel. Layer 6's `BatchedKVCache` left-padded every request's KV history to `max_kv_len`; with one 500-token request and fifteen 3-token requests, 97% of attention compute was masked zeros. Layer 7 replaces this with `PackedKVCache`, which concatenates all KV histories back-to-back and uses FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` to attend only over real tokens. Introduces `ForwardBatch`/`ForwardMode` and the `PagedBackend` dispatch pattern.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer7_packed_batching/lesson/00_outline.md)

---

## Layer 8 — Paged Attention (KVPool)

**Directory:** [`CODE_LAYERS/layer8_paged_attention/`](CODE_LAYERS/layer8_paged_attention/)

Eliminates the per-step float KV gather. Layer 7's `PackedKVCache.update()` still gathered all historical K/V from separate `PerReqKVCache` allocations and concatenated them into a new buffer every step — O(total_kv_tokens) float bandwidth, growing without bound. Layer 8 replaces all per-request caches with a single global `KVPool`: two pre-allocated flat tensors per layer sized to 85% of free GPU memory. Every token is written once into a pool slot and never moved. The decode step builds a `kv_indices` integer array (slot addresses) and passes it to FlashInfer's `BatchDecodeWithPagedKVCacheWrapper`, which reads the pool directly.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer8_paged_attention/lesson/00_outline.md)

---

## Layer 9 — Paged Attention: ReqToTokenPool + Triton

**Directory:** [`CODE_LAYERS/layer9_paged_attention2/`](CODE_LAYERS/layer9_paged_attention2/)

Eliminates the remaining O(Σ kv_tokens) costs in Layer 8's decode step. Introduces `ReqToTokenPool`, a GPU-resident `[max_batch, max_pages_per_req]` int32 table that stores page indices on-device from the moment of prefill. A Triton kernel (`create_flashinfer_kv_indices_triton`) reads that table and builds `kv_indices` entirely on-GPU — no Python loop, no CPU-to-GPU copy of historical index data. Simultaneously raises `page_size` to 16, reducing `kv_indices` length, page alloc frequency, and pool lookup count by 16×. The KV pool shape changes from `[total_slots, n_kv, dim]` to `[total_pages, page_size, n_kv, dim]`. The decode wrapper is created once at startup and reused; `kv_indptr` is computed via `torch.cumsum` on-GPU.

**Lesson:** [`lesson/00_outline.md`](CODE_LAYERS/layer9_paged_attention2/lesson/00_outline.md)

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
