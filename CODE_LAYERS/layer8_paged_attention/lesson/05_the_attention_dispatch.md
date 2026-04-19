# 05 — The Attention Dispatch

`Qwen3Attention.forward` is the point in the model where the choice of kernel is made. In Layer 7's original form, this was expressed as two `hasattr` checks — duck-typing on the `kv_cache` argument — that routed to either `F.sdpa` or FlashInfer. In Layer 8, the dispatch is replaced with a backend object that owns all kernel logic: `Qwen3Attention` stores one `PagedBackend` instance at `__init__` time and calls exactly one method in `forward`.

---

## The Dispatch Line

```python
# model/attention.py — Qwen3Attention.forward (key lines)
attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
```

This is the only place in `attention.py` that varies between PREFILL and DECODE. `self.backend` is a `PagedBackend` instance created once in `Qwen3Attention.__init__`:

```python
self.backend = PagedBackend(config)
```

`PagedBackend.__init__` stores the GQA group count and the attention scale:

```python
class PagedBackend:
    def __init__(self, config: Qwen3Config) -> None:
        self.num_kv_groups = config.num_kv_groups   # 2  (16Q / 8KV)
        self.scale         = config.head_dim ** -0.5
```

`PagedBackend.forward` reads `forward_batch.mode` and routes to one of two private methods. Adding a new kernel means writing a new class in `model/backend.py` and changing the `self.backend =` line in `Qwen3Attention.__init__` — `attention.py` never changes again.

---

## The PREFILL Path: `_prefill_forward`

```python
def _prefill_forward(self, q, k, v, layer_idx, forward_batch) -> torch.Tensor:
    kv = forward_batch.kv_cache

    if kv is not None:
        kv.store(layer_idx, k, v)          # pool write (side-effect)

    k_rep = repeat_kv(k, self.num_kv_groups)
    v_rep = repeat_kv(v, self.num_kv_groups)

    B, _, q_len, _ = q.shape
    kv_len = k.shape[2]
    additive_mask = build_additive_mask(
        forward_batch.attention_mask, q_len, kv_len, q.dtype, q.device
    )

    return F.scaled_dot_product_attention(
        q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale
    )
```

The pool write happens first. `kv.store(layer_idx, k, v)` scatters the freshly computed K/V into the pool slots assigned to this request's prompt tokens, as section 03 described. Attention then runs over the same in-memory tensors `k` and `v` — not over the pool. The pool write and the attention computation use the same data; they are independent operations on independent memory.

`repeat_kv` expands the 8 KV heads to 16 to match the Q head count before `F.sdpa`. This is the GQA expansion that has been present since Layer 5. FlashInfer's paged kernel handles GQA natively on the decode path, so `repeat_kv` is not needed there — but `F.sdpa` has no built-in GQA support and requires equal head counts.

`build_additive_mask` is called inside `_prefill_forward`, not in `Qwen3Model.forward` as it was in earlier layers. In Layer 7 and earlier, `_build_additive_mask` ran once per forward pass in `Qwen3Model.forward` and the resulting `[B, 1, q_len, kv_len]` tensor was shared across all 28 decoder layers via the `attention_mask` parameter. Now it runs once per attention layer call inside the backend. The cost is identical — the function constructs a small CPU tensor and sends it to the GPU — but the responsibility sits with the code that knows which kernel will use it. On the decode path, `build_additive_mask` is never called: `forward_batch.attention_mask` is `None` and FlashInfer does not use it.

When `forward_batch.kv_cache` is `None` — the case used by `verify_batch.py`'s reference forward pass — the `store` call is skipped. Attention still runs with `F.sdpa` over the prompt K/V, producing the same output as the paged path. This is how the verify script confirms numerical correctness: the no-cache path provides the ground truth, and the paged path should match it within floating-point tolerance.

---

## The DECODE Path: `_decode_forward`

```python
def _decode_forward(self, q, k, v, layer_idx, forward_batch) -> torch.Tensor:
    kv = forward_batch.kv_cache

    q_fi = q.squeeze(2)    # [B, n_q_heads,  head_dim]  — remove q_len=1
    k_fi = k.squeeze(2)    # [B, n_kv_heads, head_dim]
    v_fi = v.squeeze(2)

    kv.store(layer_idx, k_fi, v_fi)    # write new token to pool first

    k_paged = kv.k_pool[layer_idx].unsqueeze(1)   # [total_slots, 1, n_kv, head_dim]
    v_paged = kv.v_pool[layer_idx].unsqueeze(1)

    attn_out = kv.wrapper.forward(q_fi, (k_paged, v_paged))
    return attn_out.unsqueeze(2)                  # [B, n_q_heads, 1, head_dim]
```

`squeeze(2)` removes the `q_len=1` dimension that the rest of the model uses for layout compatibility with `F.sdpa`. FlashInfer's NHD convention treats each row as one token, so `[B, n_heads, head_dim]` is the correct input shape when attending B single-token queries.

The pool write happens before `wrapper.forward`. Section 04 established that the new token's slot was already included in `kv_indices` when `begin_forward` was called. FlashInfer reads from the pool immediately inside `wrapper.forward` — it does not know whether the new slot has been written yet. Writing before calling `wrapper.forward` is a correctness requirement: if the write were deferred until after `wrapper.forward`, FlashInfer would read zeros from the new slot and produce incorrect logits.

`unsqueeze(1)` on the pool tensors inserts the page dimension that FlashInfer's paged API requires. The pool shape is `[total_slots, n_kv_heads, head_dim]`; FlashInfer expects `[total_slots, page_size, n_kv_heads, head_dim]`. With `page_size=1`, this is `[total_slots, 1, n_kv_heads, head_dim]`. FlashInfer accesses the relevant rows via the `kv_indices` plan established in `begin_forward` — no intermediate tensor is assembled; no rows are copied.

`wrapper.forward` outputs `[B, n_q_heads, head_dim]`. The `unsqueeze(2)` call reinserts the seq dimension so the result is layout-compatible with the `F.sdpa` output path's `[B, n_heads, q_len, head_dim]`. The merge-heads and output projection that follow in `Qwen3Attention.forward` see the same shape regardless of which path ran.

---

## No SDPA Fallback for Decode

Layer 7 offered an SDPA fallback for decode because `PackedKVCache` assembled a contiguous ragged float buffer — once assembled, that buffer could be attended over by either FlashInfer's ragged kernel or, with padding, by `F.sdpa`. Layer 8 has no such buffer. The K/V history for every token lives in the pool, addressed only by slot index. To run `F.sdpa` over a request's full KV history, the caller would have to gather all the relevant pool rows into a contiguous tensor first — the exact float copy that Layer 8 eliminates. An SDPA fallback for decode is therefore not just absent; it is structurally excluded by the pool design. The decode path runs FlashInfer's `BatchDecodeWithPagedKVCacheWrapper`, or it does not run at all.
