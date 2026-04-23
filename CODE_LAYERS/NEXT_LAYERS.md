# Next Layers — Recommended Development Order

## What Exists (Layers 0–19)

| Layer | Topic | Key Concept |
|-------|-------|-------------|
| 0 | Naive inference | HF model.generate, no cache |
| 1 | Manual decode loop | Autoregressive loop, white-box |
| 2 | KV cache | past_key_values, O(1) TPOT |
| 3 | Static batching | Padded batch, shared decode loop |
| 4 | Custom model loading | From-scratch Qwen3, safetensors |
| 5 | Model layers | Full Qwen3 architecture |
| 6 | Continuous batching | asyncio.Future, per-req KV, dynamic batch |
| 7 | Packed batching | FlashInfer ragged KV, zero padding |
| 8 | Paged attention | Global KVPool, slot_indices, page_size=1 |
| 9 | Paged attention 2 | req_to_token table, Triton kv_indices kernel |
| 11 | Chunked prefill | Batched extend, page packing, PrefillAdder |
| 12 | Prefix caching | RadixCache, LRU eviction, ref-counting |
| 13 | GPTQ quantization | INT4 packed weights, gptq_gemm kernel |
| 14 | Speculative decoding | Draft+target, KV rewind, accept/reject |
| 15 | Router/Gateway | Round-robin, least-load, prefix-cache-aware routing |
| 16 | Kubernetes | Deployment, scaling, health checks |
| 17 | Tiered KV cache | HiCache, VRAM→CPU→disk offload, PCIe bandwidth |
| 18 | KV cache quantization | FP8 KV, scale calibration, uint8 workaround (lesson only) |
| 19 | PD Disaggregation | Prefill/decode split, KV transfer |

**Note**: Layers 15–19 appear to have stale READMEs (all show Layer 13 content). The lesson/ folders contain the real content.

---

## Gap Analysis vs NEW_LAYERS.md

| Requested Topic | Status | Notes |
|----------------|--------|-------|
| Overlap scheduler | ❌ Missing | Sarathi-Serve / NanoFlow overlap loop |
| CUDA graphs + torch.compile | ❌ Missing | High priority — 20-40% decode speedup |
| Streaming SSE | ❌ Missing | Needed for real API usability |
| Structured outputs (xgrammar) | ❌ Missing | Constrained decoding |
| Sampling kernels | ❌ Missing | Triton top-k/top-p/temperature |
| LoRA | ❌ Missing | Adapter serving, punica kernel |
| Quantization pt2 (AWQ, FP8 weights, INT8) | ❌ Missing | Layer 13 only covers GPTQ |
| SpecDec pt2 (EAGLE3, n-gram, tree attn, MTP) | ❌ Missing | Layer 14 only covers vanilla spec dec |
| Diffusion models (block diffusion, weDLM) | ❌ Missing | Different paradigm entirely |
| Tensor parallelism | ❌ Missing | Required for 7B+ models |
| Data parallelism | ❌ Missing | Multi-replica serving |
| MoE | ❌ Missing | Expert routing, expert parallelism |
| Pipeline parallelism | ❌ Missing | Multi-node |
| Cluster LLMs (autoscaling, cold starts) | ❌ Missing | Ops layer |
| Semantic router | ❌ Missing | Difficulty-based routing |
| MIG | ❌ Missing | GPU partitioning |
| Embeddings / RAG | ❌ Missing | Different model type |
| VLM | ❌ Missing | Vision encoder + LLM |
| Image diffusion | ❌ Missing | Text→image |
| Multimodal (audio, video) | ❌ Missing | Complex |
| Monitoring (Grafana/Prometheus) | ❌ Missing | Ops layer |
| Agents / tool calling | ❌ Missing | Application layer |
| Native Sparse Attention | ❌ Missing | DeepSeek NSA |
| KV cache compression | ❌ Missing | Eviction, H2O, SnapKV |

---

## Recommended Next Layers (Prioritized)

### Tier 1 — Core Engine Completeness (build these next)

**Layer 20 — CUDA Graphs**
- Prerequisite: Layer 11 (chunked prefill), Layer 9 (paged KV)
- What: Capture the decode loop as a CUDA graph per batch-size bucket. Eliminate Python kernel-launch overhead.
- Why now: 20–40% throughput gain on every existing layer. Foundational for production.
- Key concepts: `torch.cuda.CUDAGraph`, graph capture, bucket sizes, keeping prefill eager
- Builds on: model_runner.decode_step

**Layer 21 — Streaming SSE**
- Prerequisite: Layer 6 (continuous batching), Layer 15 (router)
- What: Server-Sent Events, per-token flush, async generator, cancel-on-disconnect
- Why now: Every real API streams. Without this the server is not usable in practice.
- Key concepts: `StreamingResponse`, `asyncio.Queue`, TTFT vs TPOT under streaming

**Layer 22 — Overlap Scheduling (Sarathi-Serve / NanoFlow)**
- Prerequisite: Layer 11 (chunked prefill), Layer 20 (CUDA graphs)
- What: Pipeline CPU scheduling of batch N+1 while GPU runs batch N. Chunked prefill interleaved with decode.
- Why now: Fixes TTFT collapse under mixed load. Natural extension of chunked prefill.
- Key concepts: double-buffering, overlap_loop, prefill/decode interleaving

**Layer 23 — Tensor Parallelism**
- Prerequisite: Layer 4/5 (custom model), Layer 9 (paged KV)
- What: Megatron column/row parallel splits, all-reduce placement, vocab sharding
- Why now: Can't serve 7B+ without it. Unlocks larger models for all subsequent layers.
- Key concepts: `dist.all_reduce`, column-parallel Linear, row-parallel Linear, TP rank

### Tier 2 — Quantization & Sampling Depth

**Layer 24 — Quantization Part 2: AWQ + FP8 Weights**
- Prerequisite: Layer 13 (GPTQ)
- What: AWQ (activation-aware weight quantization), FP8 weight quantization, INT8 SmoothQuant
- Why now: AWQ is the other dominant W4A16 method. FP8 weights are the production standard for H100.
- Key concepts: activation outlier smoothing, per-channel scales, `torch.float8_e4m3fn` matmul

**Layer 25 — Sampling Kernels (Triton)**
- Prerequisite: Layer 7 (Triton tiled GEMM from TOC)
- What: Triton kernels for temperature scaling, top-k, top-p (nucleus), min-p, repetition penalty
- Why now: Sampling is on the critical path every decode step. Python-level sampling is a bottleneck at large batch.
- Key concepts: parallel top-k via radix sort, fused temperature+softmax, batch sampling

**Layer 26 — Structured Outputs (xgrammar)**
- Prerequisite: Layer 25 (sampling kernels), Layer 21 (streaming)
- What: Constrained decoding via grammar masks, xgrammar integration, JSON/regex schemas
- Why now: Structured outputs are a top production requirement for agents and APIs.
- Key concepts: context-free grammar, token mask computation, logit bias application

### Tier 3 — Advanced Decoding

**Layer 27 — Speculative Decoding Part 2: EAGLE3 + n-gram + Tree Attention**
- Prerequisite: Layer 14 (spec dec), Layer 23 (tensor parallel)
- What: EAGLE3 (draft head trained on target hidden states), n-gram lookup drafting, tree attention for parallel verification of branching drafts
- Why now: EAGLE3 is the production spec dec method in SGLang. 3–4× speedup vs vanilla.
- Key concepts: draft head, hidden state reuse, tree mask, topk branching

**Layer 28 — LoRA Serving**
- Prerequisite: Layer 4/5 (custom model), Layer 23 (tensor parallel)
- What: LoRA adapter loading, punica batched LoRA kernel, multi-adapter serving (different adapters per request in same batch)
- Key concepts: `lora_A`, `lora_B`, rank-r decomposition, punica `bgmv` kernel

### Tier 4 — Model Architecture Expansion

**Layer 29 — Mixture of Experts (MoE)**
- Prerequisite: Layer 23 (tensor parallel)
- What: Expert routing (top-k gating), expert parallelism, load balancing, DeepSeek-style shared experts
- Key concepts: `gate_proj` → router logits, `torch.topk`, expert dispatch, all-to-all

**Layer 30 — Vision Language Models (VLM)**
- Prerequisite: Layer 4/5 (custom model), Layer 11 (chunked prefill)
- What: Vision encoder (ViT/SigLIP), image token embedding, multimodal prefill, cross-attention vs prefix injection
- Key concepts: image patch tokenization, vision tower, `pixel_values` → KV prefix

**Layer 31 — Native Sparse Attention (NSA)**
- Prerequisite: Layer 8/9 (paged attention), Layer 7 (Triton)
- What: DeepSeek NSA — block-sparse attention with compressed KV, sliding window + global tokens
- Key concepts: block sparsity pattern, compressed KV representation, NSA Triton kernel

### Tier 5 — Ops & Infrastructure

**Layer 32 — Monitoring (Prometheus + Grafana)**
- Prerequisite: Layer 15 (router), Layer 16 (k8s)
- What: Prometheus metrics (TTFT, TPOT, queue depth, KV utilization, cache hit rate), Grafana dashboards
- Key concepts: `prometheus_client`, histogram buckets, alerting rules

**Layer 33 — KV Cache Compression (H2O / SnapKV)**
- Prerequisite: Layer 12 (prefix caching), Layer 17 (tiered KV)
- What: Attention-score-based token eviction, H2O heavy hitter oracle, SnapKV clustering
- Key concepts: accumulated attention scores, eviction policy, accuracy vs compression tradeoff

**Layer 34 — Agents & Tool Calling**
- Prerequisite: Layer 21 (streaming), Layer 26 (structured outputs)
- What: OpenAI tool_calls format, parallel tool execution, multi-turn state management, ReAct loop
- Key concepts: function schema, tool result injection, streaming partial JSON

---

## Suggested Build Order (Next 5)

```
Layer 20 — CUDA Graphs          ← biggest throughput gain, no new concepts needed
Layer 21 — Streaming SSE        ← makes the server actually usable
Layer 22 — Overlap Scheduling   ← completes the scheduler story
Layer 23 — Tensor Parallelism   ← unlocks 7B+ models, unblocks MoE/VLM
Layer 24 — Quantization Pt2     ← AWQ + FP8 weights, natural follow-on to Layer 13
```
