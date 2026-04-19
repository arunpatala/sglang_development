# Layer 7 — Summary

Layer 7 replaces `BatchedKVCache` and `F.scaled_dot_product_attention` in the decode step with `PackedKVCache` and FlashInfer's ragged attention kernel, eliminating the padding waste that made Layer 6's batched forward compute attention over zero-filled columns. The scheduler, the request dataclass, `prefill()`, `tokenizer.py`, and the entire `model/` package except `attention.py` are unchanged.

---

## From Layer 6 to Layer 7

In Layer 6, every decode step padded all running requests' KV histories to the same length before calling `F.sdpa`:

```python
# Layer 6 — pad all KV caches to max_kv_len, call F.sdpa
attn_mask = torch.zeros(B, max_kv + 1, dtype=torch.long, device=DEVICE)
for i, kv_len in enumerate(kv_lens):
    attn_mask[i, max_kv - kv_len:] = 1   # left zeros = padding

batch_kv = BatchedKVCache(reqs, max_kv)
logits = self.model(last_toks, attention_mask=attn_mask,
                    kv_cache=batch_kv, position_ids=pos_ids)
batch_kv.write_back()
```

When one request has 500 tokens of history and fifteen others have 3, the attention kernel processes 501 positions per row — 497 of which are masked to −∞ and contribute nothing. The compute scales with `max_kv_len`, not with the actual average length.

In Layer 7, the `attn_mask` construction is gone. The KV histories are packed into a ragged tensor, and FlashInfer attends over only the real tokens:

```python
# Layer 7 — pack KV caches into a ragged tensor, call FlashInfer
pack_kv = PackedKVCache(reqs, self._workspace)
pack_kv.plan(num_q_heads=cfg.num_attention_heads,
             num_kv_heads=cfg.num_key_value_heads,
             head_dim=cfg.head_dim, dtype=DTYPE)

logits = self.model(last_toks, attention_mask=None,
                    kv_cache=pack_kv, position_ids=pos_ids)
pack_kv.write_back()
pack_kv.end_forward()
```

`attention_mask=None` because FlashInfer does not use a mask tensor — it knows where each request's KV slice begins and ends from the `indptr` arrays computed in `PackedKVCache.__init__`. `plan()` is called once and the plan is reused by all 28 attention layers in the same forward pass.

---

## The Packed KV Cache

`PackedKVCache` is built from the list of active requests and a pre-allocated workspace tensor:

```python
class PackedKVCache:
    def __init__(self, reqs: list, workspace: torch.Tensor) -> None:
        kv_lens = [r.kv_cache.get_seq_length() for r in reqs]

        # Each request contributes exactly 1 query token (decode step).
        self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEVICE)

        # kv_indptr[i+1] - kv_indptr[i] = kv_len_i + 1 (history + new token).
        kv_full_lens = [l + 1 for l in kv_lens]
        self.kv_indptr = torch.tensor(
            [0] + list(accumulate(kv_full_lens)), dtype=torch.int32, device=DEVICE
        )

        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )
```

`qo_indptr` is trivial: every request contributes exactly one query token during decode, so the offsets are just `[0, 1, 2, ..., B]`. `kv_indptr` is the important one. For three requests with KV lengths 10, 6, and 4, the full lengths after appending the new token are 11, 7, and 5, giving `kv_indptr = [0, 11, 18, 23]`. FlashInfer reads request `i`'s packed KV as the slice from `kv_indptr[i]` to `kv_indptr[i+1]` in the ragged tensor — exactly 11, 7, and 5 real tokens respectively, with no zero padding between them.

```
Layer 6 (padded):               Layer 7 (packed):
┌────────────────────────┐      ┌──────────────┐
│ req0  [  pad  ] [ 10 ] │      │ req0 [10+1]  │  kv_indptr: 0 → 11
│ req1  [  pad  ] [  6 ] │      │ req1 [ 6+1]  │  kv_indptr: 11 → 18
│ req2  [  pad  ] [  4 ] │      │ req2 [ 4+1]  │  kv_indptr: 18 → 23
└────────────────────────┘      └──────────────┘
shape: [3, kv, 10, dim]         shape: [23, kv, dim]
```

The workspace tensor — a 256 MB `torch.uint8` buffer allocated once at `ModelRunner.__init__` — is passed to FlashInfer so it can place any temporary buffers inside it during `begin_forward`, avoiding mid-step allocations.

---

## Packing and Attending

`PackedKVCache.update` is called by each attention layer during the forward pass. It gathers the historical KV for each request, appends the new decode token, and concatenates everything into one contiguous ragged tensor:

```python
def update(self, layer_idx, new_k, new_v):
    segs_k, segs_v = [], []
    for i, req in enumerate(self._reqs):
        hist_k = req.kv_cache._k[layer_idx]   # [1, n_kv, L_i, dim]
        # Reshape to NHD layout: [L_i, n_kv, dim]
        hist_k_nhd = hist_k.squeeze(0).permute(1, 0, 2).contiguous()
        segs_k.append(hist_k_nhd)
        segs_k.append(new_k[i].unsqueeze(0))   # [1, n_kv, dim] — new token
    k_packed = torch.cat(segs_k, dim=0)        # [total_kv_tokens, n_kv, dim]
    ...
    self._new_k[layer_idx] = new_k   # saved for write_back
    return k_packed, v_packed
```

The `+1` in `kv_indptr` corresponds to this: the new decode token is appended inside `update()` before FlashInfer sees the ragged tensor. FlashInfer therefore attends over `L_i + 1` tokens for request `i` — the full KV history plus the current new key — without `causal=True` masking, since `q_len=1` means there are no future positions to mask out.

After the full 28-layer forward pass, `write_back()` reads `_new_k[layer_idx]` (shape `[B, n_kv_heads, head_dim]`), reshapes each request's slice back to `[1, n_kv_heads, 1, head_dim]`, and appends it to `req.kv_cache` via the same `torch.cat` on `dim=2` that `PerReqKVCache` has always used. `end_forward()` calls `wrapper.end_forward()` to release FlashInfer's internal state before the next decode step.

---

## The Attention Dispatch

`Qwen3Attention.forward` now selects between two backends based on the type of cache it receives:

```python
if kv_cache is not None and hasattr(kv_cache, "wrapper"):
    # ── FlashInfer path (PackedKVCache, decode B=N) ───────────────
    q_fi = q.squeeze(2)     # [B, n_q_heads, head_dim]  (NHD layout)
    k_fi = k.squeeze(2)     # [B, n_kv_heads, head_dim]
    v_fi = v.squeeze(2)

    k_packed, v_packed = kv_cache.update(self.layer_idx, k_fi, v_fi)
    # [total_kv_tokens, n_kv_heads, head_dim]

    attn_out = kv_cache.wrapper.forward(q_fi, k_packed, v_packed)
    # [B, n_q_heads, head_dim]
    attn_out = attn_out.unsqueeze(2)

else:
    # ── F.sdpa path (PerReqKVCache or no cache, prefill B=1) ──────
    if kv_cache is not None:
        k, v = kv_cache.update(self.layer_idx, k, v)
    k = repeat_kv(k, self.num_kv_groups)
    v = repeat_kv(v, self.num_kv_groups)
    attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=self.scale)
```

The dispatch key is `hasattr(kv_cache, "wrapper")`. `PackedKVCache` exposes a `wrapper` property (the `BatchPrefillWithRaggedKVCacheWrapper`); `PerReqKVCache` does not. This duck-type check requires no `isinstance` call and no changes to the cache interface.

Two things differ on the FlashInfer path. First, tensors are in NHD layout (`[tokens, heads, dim]`) rather than the HND layout (`[batch, heads, seq, dim]`) that `F.sdpa` expects — hence the `squeeze(2)` before `update()` and `unsqueeze(2)` after. Second, FlashInfer handles GQA natively: it reads from 8 KV heads while computing attention for 16 Q heads without any explicit `repeat_kv` expansion. The `F.sdpa` path still needs `repeat_kv` and continues to do so unchanged.

---

## The Full Loop

The scheduler and prefill are identical to Layer 6 — a new request is enqueued, the scheduler calls `prefill(req)` with a `PerReqKVCache` and `F.sdpa`, the first token is sampled, and the request enters `_running`.

On the next scheduler iteration, `decode_step(_running)` runs. `kv_lens` is collected from each request's `PerReqKVCache`. `last_toks [B, 1]` and `pos_ids [B, 1]` are built per request — position IDs are still `kv_len_i` per request, preserving per-request RoPE correctness. `PackedKVCache(reqs, self._workspace)` is constructed: `qo_indptr` and `kv_indptr` are computed, and the FlashInfer wrapper is initialised with the workspace.

`plan()` calls `wrapper.begin_forward` once with the head counts and data type for this step. Then `self.model(last_toks, attention_mask=None, kv_cache=pack_kv, position_ids=pos_ids)` runs the 28-layer forward pass. Each `Qwen3Attention.forward` fires the FlashInfer branch: `update()` gathers and concatenates ragged KV segments for that layer, `wrapper.forward()` attends only over the `total_kv_tokens` real tokens across all requests, and the output is unsqueezed back to `[B, n_q_heads, 1, dim]`.

After the forward pass, `write_back()` appends the one new K/V token per request to their respective `PerReqKVCache`s. `end_forward()` releases the FlashInfer state. Sampling is identical to Layer 6: one token per request, `output_ids` updated, finished requests returned to the scheduler, which resolves their futures via `loop.call_soon_threadsafe`.

---

## What Comes Next

Layer 7 eliminates padding waste in the attention kernel. But a hidden copy cost remains. On every decode step, `PackedKVCache.update()` gathers all per-request historical K/V tensors — which live in separate `PerReqKVCache` allocations — and concatenates them into a new contiguous ragged buffer. This gather costs `O(total_kv_tokens)` memory bandwidth per step. As the running batch accumulates history, this copy grows and becomes the dominant per-step cost.

Memory fragmentation also persists. Each `PerReqKVCache` still grows by one `torch.cat` allocation per layer per decode step. Long-running requests accumulate hundreds of small allocations per layer, none of which can be reused until the request finishes.

Layer 8 addresses both by replacing `PerReqKVCache` with a paged block table. Physical KV memory is divided into fixed-size pages allocated from a shared pool. A request's KV is stored in whatever pages are available, not in a contiguous tensor it owns. FlashInfer's paged attention kernel reads directly from the block table — no gather, no copy, no per-step allocation. The scheduler and `model/attention.py` dispatch table are unchanged; the work moves entirely into a new page-table-based `kv_cache.py` and the `BlockManager` in `model_runner.py`.
