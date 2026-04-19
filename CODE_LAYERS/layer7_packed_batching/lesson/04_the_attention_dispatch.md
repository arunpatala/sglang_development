# 04 — The Attention Dispatch

The decode step (section 01) passes `kv_cache=pack_kv` to the model's forward call with `attention_mask=None`. Both facts would have crashed the Layer 6 attention code — it expected a non-None mask and had no concept of a FlashInfer wrapper. Layer 7's `Qwen3Attention.forward` is the only place in the model that changes, and the change is a single dispatch branch that routes the computation to either `F.sdpa` or FlashInfer depending on which cache type it receives.

---

## The Dispatch Table

```python
def forward(
    self,
    hidden_states:  torch.Tensor,          # [B, q_len, hidden]
    cos:            torch.Tensor,          # [B, q_len, head_dim]
    sin:            torch.Tensor,
    attention_mask: torch.Tensor | None,   # additive mask (F.sdpa path); None for FlashInfer
    kv_cache=None,                         # PerReqKVCache | PackedKVCache | None
) -> torch.Tensor:
```

The three possible `kv_cache` values map to three behaviours:

| `kv_cache` type | `hasattr(kv_cache, "wrapper")` | Backend | Used when |
|---|---|---|---|
| `None` | — | `F.sdpa` | no-cache prefill (verify scripts) |
| `PerReqKVCache` | `False` | `F.sdpa` | B=1 prefill in scheduler |
| `PackedKVCache` | `True` | FlashInfer | B=N decode step |

The dispatch key is `hasattr(kv_cache, "wrapper")`. `PackedKVCache` exposes a `wrapper` property that returns the `BatchPrefillWithRaggedKVCacheWrapper`. `PerReqKVCache` has no `wrapper` attribute. This duck-type check requires no `isinstance` call and no import of either cache class into `attention.py`.

---

## The FlashInfer Path

```python
if kv_cache is not None and hasattr(kv_cache, "wrapper"):
    # ── FlashInfer path (PackedKVCache, decode B=N) ───────────────────

    # Reshape q/k/v from [B, heads, 1, dim] → [B, heads, dim]  (NHD, q_len removed)
    q_fi = q.squeeze(2)   # [B, n_q_heads,  head_dim]
    k_fi = k.squeeze(2)   # [B, n_kv_heads, head_dim]
    v_fi = v.squeeze(2)   # [B, n_kv_heads, head_dim]

    # update() packs historical KV + new token into a ragged tensor.
    # Returns (k_packed, v_packed): [total_kv_tokens, n_kv_heads, head_dim]
    k_packed, v_packed = kv_cache.update(self.layer_idx, k_fi, v_fi)

    # FlashInfer handles GQA natively — no repeat_kv needed.
    # wrapper.forward uses the plan from begin_forward called in plan().
    attn_out = kv_cache.wrapper.forward(q_fi, k_packed, v_packed)
    # Output: [B, n_q_heads, head_dim]

    # Re-insert the seq dim so the merge-heads step below is layout-compatible.
    attn_out = attn_out.unsqueeze(2)   # [B, n_q_heads, 1, head_dim]
```

**`squeeze(2)` before `update`.** Up to this point, `q`, `k`, and `v` are in HND layout — `[B, n_heads, q_len, head_dim]` — with `q_len=1` for a decode step. FlashInfer uses NHD layout — `[tokens, n_heads, head_dim]`. `squeeze(2)` removes the `q_len=1` dimension, giving `[B, n_heads, head_dim]`. Each row of this tensor is one query (or key) token for one request, which is exactly what NHD expects when `q_len=1`.

**`kv_cache.update(self.layer_idx, k_fi, v_fi)`.** This is where `PackedKVCache.update()` runs for this specific layer. It gathers the historical K/V from each request's `PerReqKVCache` for `self.layer_idx`, converts to NHD, appends the new token `k_fi[i]` for each request, and concatenates all segments into `[total_kv_tokens, n_kv_heads, head_dim]`. The return value is the fully packed ragged tensor that FlashInfer will read. Section 03 covers this in detail.

**`kv_cache.wrapper.forward(q_fi, k_packed, v_packed)`.** FlashInfer dispatches based on the plan established in `begin_forward`: it knows the `qo_indptr` (one query per request) and `kv_indptr` (where each request's KV slice ends) from when `plan()` was called. It attends request `i`'s query `q_fi[i]` over the slice `k_packed[kv_indptr[i]:kv_indptr[i+1]]` — which is exactly `kv_len_i + 1` tokens, the full real history plus the new key. Zero columns are never touched. The output is `[B, n_q_heads, head_dim]`.

**`unsqueeze(2)` after `wrapper.forward`.** The subsequent merge-heads and output projection code expects `attn_out` in `[B, n_heads, q_len, head_dim]` form. `unsqueeze(2)` reinserts the `q_len=1` dimension, making the FlashInfer output layout-compatible with the `F.sdpa` output without any further branching.

---

## GQA Without `repeat_kv`

Qwen3-0.6B has 16 Q heads and 8 KV heads — a 2:1 ratio known as Grouped Query Attention (GQA). The `F.sdpa` path handles this by expanding the 8 KV heads to 16 before the attention call:

```python
# F.sdpa path — explicit GQA expansion
k = repeat_kv(k, self.num_kv_groups)   # [B, 8, kv_len, dim] → [B, 16, kv_len, dim]
v = repeat_kv(v, self.num_kv_groups)
attn_out = F.scaled_dot_product_attention(q, k, v, ...)
```

`repeat_kv` uses `expand` — a zero-copy view with broadcast strides — followed by `reshape`, which may force a copy. Either way, the KV tensors grow to match the Q head count before the kernel sees them.

FlashInfer handles GQA natively. The kernel takes `q` with 16 heads and KV with 8 heads directly, computing the 2:1 group assignment internally without any explicit expansion. This eliminates the `repeat_kv` call entirely on the FlashInfer path — and, crucially, eliminates the memory cost of holding a 2× expanded KV tensor. For large batches with long histories, this is a meaningful reduction in peak memory bandwidth.

---

## The F.sdpa Path Is Unchanged

```python
else:
    # ── F.sdpa path (PerReqKVCache or no cache, prefill B=1) ───────
    if kv_cache is not None:
        k, v = kv_cache.update(self.layer_idx, k, v)

    k = repeat_kv(k, self.num_kv_groups)   # GQA expansion
    v = repeat_kv(v, self.num_kv_groups)

    attn_out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,
        scale=self.scale,
    )   # [B, n_heads, q_len, head_dim]
```

Everything from Layer 5 and Layer 6's `attention.py` is here unchanged. `PerReqKVCache.update()` appends the new K/V and returns the full accumulated rectangular cache. `repeat_kv` expands KV heads. `F.sdpa` runs with the additive mask passed from the caller. This path is used by `prefill()` (which always passes a `PerReqKVCache`) and by the no-cache forward calls in `verify.py` scripts. The FlashInfer path is strictly additive — it does not remove or alter the existing code.
