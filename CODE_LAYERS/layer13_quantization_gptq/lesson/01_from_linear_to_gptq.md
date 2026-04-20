# 01 — From nn.Linear to GPTQLinear

## From Layer 12 to Layer 13

Layer 12 added a `RadixCache` that caches K/V pages for shared prompt prefixes, reducing redundant computation at inference time. The model weights themselves remained in bfloat16 — `nn.Linear` projections, with each weight matrix stored as a 2D float16 tensor. For Qwen3-1.7B:

```python
# Layer 12 — model/attention.py (standard bfloat16 linear projections)
self.q_proj = nn.Linear(config.hidden_size, q_dim, bias=False)
# Weight: [q_dim, hidden_size] bfloat16 = [2048, 2048] × 2 bytes = 8.4 MB
self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
# Weight: [1024, 2048] × 2 = 4.2 MB
self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=False)
self.o_proj = nn.Linear(q_dim, config.hidden_size, bias=False)
# Weight: [2048, 2048] × 2 = 8.4 MB
```

Across 28 layers, just the attention projections consume approximately 3.4 GB. Adding the three FFN projections per layer (`gate_proj`, `up_proj`, `down_proj` each at `[11008, 2048]` or `[2048, 11008]`) pushes the total to approximately 5 GB. On a 16 GB GPU, 5 GB of weights leaves only ~11 GB for the KV pool — limiting peak token capacity to roughly 20,000 tokens (at the Layer 9 pool sizing formula for Qwen3-1.7B's 8 KV heads and 128 head dim across 28 layers).

In Layer 13, the same four lines read:

```python
# Layer 13 — model_gptq/attention.py (4-bit GPTQ quantized projections)
self.q_proj = GPTQLinear(config.hidden_size, q_dim, bits, group_size)
# Effective storage: [2048//8, 2048] int32 qweight ≈ 2.1 MB + ~0.1 MB scales/zeros
self.k_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
self.v_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
self.o_proj = GPTQLinear(q_dim, config.hidden_size, bits, group_size)
```

`GPTQLinear` is a drop-in replacement: `q_proj(hidden_states)` calls `GPTQLinear.forward` and returns the same `[B, q_len, q_dim]` shape tensor. Everything downstream — `repeat_kv`, head splitting, `PagedExtendBackend._extend_forward`, FlashInfer — operates on bfloat16 activations and cannot distinguish the two.

---

## The Memory Arithmetic

`GPTQLinear` packs 8 int4 values into each int32 element (`pack_factor = 32 // 4 = 8`). The `qweight` tensor has shape `[K // pack_factor, N]` = `[K // 8, N]`. For `K = N = 2048`:

```
qweight: [256, 2048] int32 = 256 × 2048 × 4 bytes = 2.1 MB
scales:  [16, 2048] float16 = 16 × 2048 × 2 bytes = 0.06 MB  (group_size=128)
qzeros:  [16, 256] int32 = 16 × 256 × 4 bytes = 0.02 MB
```

Total per attention projection: approximately 2.2 MB vs 8.4 MB in bfloat16. The FFN projections follow the same arithmetic with larger K and N. Across 28 layers, the total weight footprint drops from ~5 GB to approximately 1.25 GB — freeing roughly 3.75 GB for the KV pool.

This additional pool space supports approximately 3× more concurrent tokens, directly improving throughput and context window capacity without changing the KV management code at all.

---

## What Changes and What Does Not

One new file `model_gptq/gptq_linear.py` contains the entire `GPTQLinear` implementation: buffer registration, `prepare()`, and `forward()` with the `gptq_gemm` call. The four files `model_gptq/attention.py`, `model_gptq/decoder_layer.py`, and `model_gptq/qwen3.py` mirror their `model/` counterparts but substitute `GPTQLinear` for `nn.Linear` in projection layers and thread `bits`/`group_size` through the constructor hierarchy. `model_runner.py` gains a two-line dispatch: `use_gptq` selects between `model.Qwen3ForCausalLM` and `model_gptq.Qwen3ForCausalLM`.

`kv_cache.py`, `radix_cache.py`, `scheduler.py`, `forward_batch.py`, `server.py`, `tokenizer.py`, and all K/V management code are unchanged. The K/V tensors are always bfloat16 activations regardless of weight precision; the KV pool, `ExtendKVCtx`, `DecodeKVCtx`, and FlashInfer see no difference. The sections below explain the GPTQ weight format, the kernel, and the loading sequence in detail.
