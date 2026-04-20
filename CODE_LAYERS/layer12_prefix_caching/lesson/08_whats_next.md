# 08 — What Comes Next

Layer 12 eliminates redundant KV computation for shared prefixes. Requests with identical system prompts pay the full prefill cost only once; all subsequent requests benefit from the cached pages. The scheduling, paged KV management, and extend/decode kernel infrastructure are unchanged. A different bottleneck now dominates.

---

## Model Weight Memory

For Qwen3-1.7B in bfloat16, the 28 decoder layers contain the following projection matrices per layer:

- `q_proj`: `2048 × 2048 × 2 bytes = 8.4 MB`
- `k_proj` and `v_proj`: `2048 × 1024 × 2 bytes = 4.2 MB` each
- `o_proj`: `2048 × 2048 × 2 bytes = 8.4 MB`
- `gate_proj` and `up_proj`: `2048 × 11008 × 2 bytes = 45.1 MB` each
- `down_proj`: `11008 × 2048 × 2 bytes = 45.1 MB`

Per layer: approximately 160 MB. Across 28 layers: approximately 4.5 GB. Add `embed_tokens` and `lm_head` and the total reaches ~5 GB. On a 16 GB GPU, after the model is loaded, only about 11 GB remains for the KV pool, activations, and the PyTorch caching allocator overhead.

After prefix caching, the pool can hold more unique pages — but the pool's maximum size is still bounded by VRAM not consumed by weights. At typical request rates where the KV pool is the binding constraint, reducing weight memory directly increases the pool capacity and thus the maximum achievable throughput.

---

## What Layer 13 Adds

Layer 13 replaces `nn.Linear` with `GPTQLinear` for all projection matrices in `Qwen3Attention` and `Qwen3MLP`. GPTQ (Generative Pre-trained Transformer Quantization) stores weights as 4-bit integers packed 8 per int32, with per-group scales in fp16 and packed zero-points in int32. The `gptq_gemm` fused CUDA kernel from `sgl_kernel` dequantizes weights on-the-fly and computes the matrix multiplication in a single pass — no separate dequantize step, no full fp16 weight materialization.

The memory reduction:
- `nn.Linear(2048, 2048)` in bfloat16: `2048 × 2048 × 2 = 8.4 MB`.
- `GPTQLinear(2048, 2048, bits=4, group_size=128)` in int4: `2048 × 2048 / 2 ≈ 2.1 MB` for `qweight` plus `~0.1 MB` for `scales` and `qzeros`. Total ≈ 2.2 MB.

The 4× reduction in weight memory translates to approximately 4× more room for the KV pool at fixed GPU capacity. A server that could hold 4096 tokens in the pool at bfloat16 precision can hold approximately 16384 tokens after GPTQ quantization — directly enabling larger decode batches and longer context windows.

---

## What Changes in Layer 13

A new file `model_gptq/gptq_linear.py` defines `GPTQLinear`. The `model_gptq/` directory mirrors `model/` but replaces `nn.Linear` in `attention.py`, `decoder_layer.py`, and `qwen3.py` with `GPTQLinear`. `model_runner.py` gains a `use_gptq: bool` flag that selects between the two model classes at startup.

`radix_cache.py`, `kv_cache.py`, `scheduler.py`, `forward_batch.py`, `server.py`, and `tokenizer.py` are unchanged. The KV management stack does not know or care whether the model computing the K/V tensors uses bfloat16 or quantized weights. The K/V values themselves are computed as bfloat16 activations regardless of the weight precision — `GPTQLinear.forward` casts back to the original activation dtype after the `gptq_gemm` call.

The pattern holds: one mechanism, one new file, the benchmark measures the specific improvement (throughput at fixed VRAM, or pool capacity).
