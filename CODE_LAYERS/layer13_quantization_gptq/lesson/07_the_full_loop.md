# 07 — The Full Loop

The GPTQ model is transparent to every layer of the inference stack except the attention and FFN projections. This section traces one prefill and one decode step through the complete system to confirm where `gptq_gemm` fires and what the rest of the stack sees.

---

## Startup: Model Load and Pool Sizing

```python
# model_runner.py — ModelRunner.__init__ with use_gptq=True
from model_gptq import Qwen3ForCausalLM as ModelClass
self.model = ModelClass.from_pretrained(model_path, dtype=DTYPE)
# Model loaded: ~1.25 GB VRAM for weights and buffers

cfg = self.model.model.config
free_bytes, _ = torch.cuda.mem_get_info()
# free_bytes ≈ 14.75 GB (16 GB card, 1.25 GB consumed)

bytes_per_token = (
    cfg.num_hidden_layers * 2
    * cfg.num_key_value_heads
    * cfg.head_dim
    * (torch.finfo(DTYPE).bits // 8)
)
# = 28 × 2 × 8 × 128 × 2 = 114688 bytes per token

max_pages = int(free_bytes * _KV_MEMORY_FRACTION / (page_size * bytes_per_token))
# ≈ 14.75 GB × 0.85 / (16 × 114688) ≈ 6830 pages × 16 tokens/page ≈ 109,000 tokens
```

For comparison, with bfloat16 weights (~5 GB), `free_bytes ≈ 11 GB`, yielding approximately 5,100 pages ≈ 81,000 tokens. The GPTQ model supports approximately 35% more token capacity at the same GPU memory budget, from weight savings alone.

`RadixCache` is initialized if `enable_prefix_caching=True`. Both are the same as Layer 12.

---

## Prefill: The Extend Forward Pass

A 512-token request arrives. `PrefillAdder` processes it (with optional prefix matching if the system prompt is cached). `prefill_batch([req])` runs:

Steps 1–6 are identical to Layer 12. `compute_write_info` allocates 32 pages, updates `req_to_token_pool`, the Triton kernel builds `kv_indices`, `begin_forward` plans the extend kernel.

Step 7 calls `self.model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)`.

Inside the model:

**Embedding:** `embed_tokens(ids_t)` — standard bfloat16 lookup. Output: `[1, 512, 2048]` bfloat16.

**Layer 0 — input_layernorm:** `RMSNorm` in bfloat16. Output: `[1, 512, 2048]` bfloat16.

**Layer 0 — Qwen3Attention:**
- `q_proj(hidden_states)` → `GPTQLinear.forward`:
  - Cast: `[1, 512, 2048] bfloat16 → [512, 2048] float16`
  - `gptq_gemm([512, 2048] fp16, qweight [256, 2048] int32, ...)` → `[512, 2048] float16`
  - Cast back: `[512, 2048] float16 → [512, 2048] bfloat16`
  - Reshape: `[1, 512, 2048] bfloat16`
- Same for `k_proj` and `v_proj` (output: `[1, 512, 1024]`).
- Split into heads, apply RoPE.
- `PagedExtendBackend._extend_forward`: `ctx.store(0, k, v)` writes 32 pool pages; `extend_wrapper.forward(q_fi, ...)` → FlashInfer paged prefill → `[512, 16, 128]` bfloat16.
- `o_proj(attn_out)` → `GPTQLinear.forward` → `[1, 512, 2048]` bfloat16.

**Layer 0 — Qwen3MLP:**
- `gate_proj(x)` → `GPTQLinear.forward` → `[1, 512, 11008]` bfloat16
- `up_proj(x)` → `GPTQLinear.forward` → `[1, 512, 11008]` bfloat16
- `F.silu(gate) * up` → `[1, 512, 11008]` bfloat16
- `down_proj(...)` → `GPTQLinear.forward` → `[1, 512, 2048]` bfloat16

This pattern — 7 `GPTQLinear.forward` calls per decoder layer — repeats for all 28 layers. The pool writes and FlashInfer calls are the same as in Layer 12. The attention outputs and FFN outputs are bfloat16 tensors of the same shapes.

**lm_head:** Standard `nn.Linear` in bfloat16. Logits: `[1, 512, 151936]`. First output token sampled from `logits[0, -1, :]`.

---

## Decode: One Step

`decode_step([req])` runs. Steps 1–6 (metadata computation, conditional page alloc, Triton kernel, `begin_forward`) are identical to Layer 12.

The model forward call processes one token per request: `last_toks [B, 1]` → through 28 layers, each with 7 `GPTQLinear.forward` calls for the projections. The FlashInfer decode kernel `BatchDecodeWithPagedKVCacheWrapper.forward` reads the full KV history from the pool via `kv_indices`. Output token sampled.

The total per-decode-step GPU work for weight loading: 28 layers × 7 projections × 2.2 MB = approximately 432 MB of int32 data read by `gptq_gemm`. Compare with the bfloat16 model: 28 × 7 × 8.4 MB = approximately 1646 MB. The GPTQ model reads approximately 3.8× less weight data per decode step, directly improving decode throughput on memory-bandwidth-bound hardware.

---

## What the RadixCache and Scheduler See

Neither the `RadixCache` nor the scheduler is aware of weight quantization. The scheduler calls `prefill_batch` and `decode_step`; the cache calls `insert` and `match_prefix` on page indices. All these operations work on `slot_indices` — integer page addresses in the pool — and KV tensor shapes. Since the K/V tensors are always bfloat16 (regardless of weight precision), the pool layout, page sizes, and `req_to_token_pool` structure are identical to Layer 12. The `use_gptq` flag affects only the model initialization path and nothing downstream.

Section 08 explains what Layer 13's quantization leaves on the table and what Layer 14's speculative decoding addresses next.
