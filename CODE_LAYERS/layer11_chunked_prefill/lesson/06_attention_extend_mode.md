# 06 — The Attention Backend: EXTEND Mode

The attention backend is the final stop for `ForwardBatch` inside the model. In Layer 9, the backend had two paths: `PREFILL` (F.sdpa over a single request's rectangular prompt) and `DECODE` (FlashInfer paged decode over a batch). Layer 11 replaces `PREFILL` with `EXTEND` — a FlashInfer paged-prefill call that handles arbitrary batch sizes and partial-chunk continuations — and adds `NOCACHE` for the standalone testing path.

---

## ForwardMode.EXTEND

```python
# forward_batch.py — ForwardMode enum
class ForwardMode(Enum):
    EXTEND  = auto()   # paged prefill (full prompt or chunk); uses ExtendKVCtx
    DECODE  = auto()   # paged decode one token per request;   uses DecodeKVCtx
    NOCACHE = auto()   # plain F.sdpa, no KV pool (verify baseline)
```

The rename from `PREFILL` to `EXTEND` is conceptually important: the old `PREFILL` mode ran F.sdpa over a new prompt starting at position 0. `EXTEND` runs the paged-prefill kernel starting at any `kv_committed_len` offset — it extends the existing KV pool, whether that pool is empty or contains 2048 tokens from prior chunks. The operation is a generalization that subsumes both fresh prefill and chunk continuation.

`NOCACHE` is the path taken when `kv_cache is None` — no pool, no FlashInfer, just `F.sdpa` over the raw Q/K/V. This path exists for offline testing of the model without a running server and pool.

---

## PagedExtendBackend._extend_forward

```python
# model/backend.py — _extend_forward
def _extend_forward(
    self,
    q: torch.Tensor,          # [1, n_q_heads, total_tokens, head_dim]
    k: torch.Tensor,          # [1, n_kv_heads, total_tokens, head_dim]
    v: torch.Tensor,
    layer_idx: int,
    forward_batch: ForwardBatch,
) -> torch.Tensor:

    # Step 1: write K/V into the pool
    if forward_batch.kv_cache is not None:
        forward_batch.kv_cache.store(layer_idx, k, v)   # WriteInfo scatter

    # Step 2: reshape Q for FlashInfer's [T, n_heads, head_dim] convention
    B, n_q, T, D = q.shape
    q_fi = q.squeeze(0).permute(1, 0, 2)     # [T, n_q_heads, head_dim]

    # Step 3: pool tensors — [total_pages, page_size, n_kv_heads, head_dim]
    k_paged = forward_batch.kv_cache.k_pool[layer_idx]
    v_paged = forward_batch.kv_cache.v_pool[layer_idx]

    # Step 4: paged prefill call
    attn_out = forward_batch.kv_cache.extend_wrapper.forward(
        q_fi, (k_paged, v_paged), causal=True
    )                                          # [T, n_q_heads, head_dim]

    # Step 5: restore shape
    return attn_out.permute(1, 0, 2).unsqueeze(0)   # [1, n_q_heads, T, head_dim]
```

`ctx.store(layer_idx, k, v)` is called first, before the attention computation. This ensures that the newly written K/V is available for the extend kernel to read — FlashInfer's paged prefill with `causal=True` lets each query token attend to all KV positions at or before its index, including the ones being written in this same call.

The pool tensor `k_paged` has shape `[total_pages, page_size, n_kv_heads, head_dim]` — the four-dimensional layout that FlashInfer's paged kernel expects. No `unsqueeze` is needed (unlike Layer 8's `page_size=1` path that required inserting a synthetic page dimension). `extend_wrapper.forward(q_fi, (k_paged, v_paged), causal=True)` runs with the plan that `begin_forward` established — `qo_indptr`, `kv_indptr`, `kv_indices`, `kv_last_page_lens` are all already stored in the wrapper.

---

## Contrast with the Old PREFILL Path

In Layer 9, the prefill path used `F.sdpa` directly:

```python
# Layer 9 — model/backend.py — _prefill_forward (simplified)
if forward_batch.kv_cache is not None:
    forward_batch.kv_cache.store(layer_idx, k, v)  # PrefillKVCtx page write

k_rep = repeat_kv(k, self.num_kv_groups)   # GQA expand
v_rep = repeat_kv(v, self.num_kv_groups)

additive_mask = build_additive_mask(forward_batch.attention_mask, q_len, kv_len, ...)
return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale)
```

The old path was B=1 only — `F.sdpa` on the full prompt as a rectangle. It required `repeat_kv` to expand K/V from 8 KV heads to 16 Q heads for GQA, and required an explicit additive causal mask. Neither is needed in the EXTEND path: FlashInfer handles GQA natively (no `repeat_kv` needed) and enforces causality internally (no mask construction needed).

The other consequence is that the old path could not handle `kv_committed_len > 0`. A `k` tensor of shape `[1, n_kv_heads, 512, head_dim]` passed to `F.sdpa` would attend only over the 512 new tokens, not over the 512 cached tokens from the prior chunk. The extend kernel, by contrast, reads the full page history via `kv_indices` and attends over all committed + new positions.

---

## NOCACHE Path

When `forward_batch.mode == ForwardMode.NOCACHE`, the backend falls back to F.sdpa with an explicit causal mask:

```python
# model/backend.py — _nocache_forward
k_rep = repeat_kv(k, self.num_kv_groups)
v_rep = repeat_kv(v, self.num_kv_groups)
additive_mask = build_additive_mask(forward_batch.attention_mask, q_len, kv_len, ...)
return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale)
```

This is the same code as Layer 9's `_prefill_forward` without the pool write. It exists only for testing the model architecture without setting up a full `KVPool`. Production inference always uses `EXTEND` or `DECODE`.

---

## qwen3.py: ForwardBatch Construction

`Qwen3Model.forward` now constructs `ForwardBatch` internally by inspecting the `kv_cache` type:

```python
# model/qwen3.py — Qwen3Model.forward (simplified)
if kv_cache is None:
    fb = ForwardBatch(mode=ForwardMode.NOCACHE, kv_cache=None, attention_mask=attention_mask)
elif hasattr(kv_cache, 'extend_wrapper'):
    fb = ForwardBatch(mode=ForwardMode.EXTEND, kv_cache=kv_cache, attention_mask=None)
elif hasattr(kv_cache, 'wrapper'):
    fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=kv_cache, attention_mask=None)
```

`extend_wrapper` is the duck-typing marker on `ExtendKVCtx`; `wrapper` is the marker on `DecodeKVCtx`. `attention_mask=None` is passed for EXTEND and DECODE because FlashInfer handles causality internally. Only NOCACHE needs a mask.

This design keeps the `model_runner.py` API backward-compatible: `prefill_batch` and `decode_step` pass `kv_cache=ctx` without specifying a `ForwardMode`. The mode is detected from the context type. Section 07 traces a full scheduler iteration to show all these pieces working together.
