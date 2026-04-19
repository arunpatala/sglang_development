# 03 — Packing and Attending

Section 02 established how `PackedKVCache.__init__` computes `kv_indptr` and creates the FlashInfer wrapper. The indptr arrays describe the ragged layout to FlashInfer, but they contain no actual tensor data — the packed K/V tensor is constructed inside each attention layer during the forward pass, one layer at a time. This section covers that construction: `update()`, the NHD reshape, the per-request segment concatenation, `write_back()`, and `end_forward()`.

---

## `update()`: Building the Ragged Tensor Per Layer

```python
def update(
    self,
    layer_idx: int,
    new_k: torch.Tensor,   # [B, n_kv_heads, head_dim] — new decode token, RoPE'd
    new_v: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    segs_k: List[torch.Tensor] = []
    segs_v: List[torch.Tensor] = []

    for i, req in enumerate(self._reqs):
        # Historical KV for this layer: [1, n_kv, L_i, head_dim]
        hist_k = req.kv_cache._k[layer_idx]
        hist_v = req.kv_cache._v[layer_idx]

        # [1, n_kv, L_i, dim] → [L_i, n_kv, dim]  (NHD layout)
        hist_k_nhd = hist_k.squeeze(0).permute(1, 0, 2).contiguous()
        hist_v_nhd = hist_v.squeeze(0).permute(1, 0, 2).contiguous()

        segs_k.append(hist_k_nhd)
        segs_k.append(new_k[i].unsqueeze(0))   # [1, n_kv, dim]
        segs_v.append(hist_v_nhd)
        segs_v.append(new_v[i].unsqueeze(0))

    # Save new tokens for write_back.
    self._new_k[layer_idx] = new_k   # [B, n_kv, dim]
    self._new_v[layer_idx] = new_v

    return (
        torch.cat(segs_k, dim=0),   # [total_kv_tokens, n_kv, dim]
        torch.cat(segs_v, dim=0),
    )
```

`update()` is called once per attention layer, 28 times per decode step. Each call receives the new decode token's K/V for the current layer: `new_k` and `new_v` of shape `[B, n_kv_heads, head_dim]` — one token per request, already RoPE-encoded by `apply_rotary_pos_emb`.

**NHD reshape.** `PerReqKVCache` stores K/V in HND layout — `[1, n_kv_heads, seq_len, head_dim]`, matching what `F.sdpa` expects. FlashInfer uses NHD layout — `[tokens, n_kv_heads, head_dim]`. The conversion is `squeeze(0).permute(1, 0, 2).contiguous()`:

```
[1, n_kv, L_i, dim]
  → squeeze(0) → [n_kv, L_i, dim]
  → permute(1, 0, 2) → [L_i, n_kv, dim]   (NHD)
  → contiguous() → ensures memory is contiguous for torch.cat
```

`contiguous()` is required because `permute` returns a view with non-contiguous strides, and `torch.cat` on non-contiguous tensors would trigger an implicit copy with potentially worse memory access patterns. Making it contiguous explicitly here gives a predictable single copy.

**Appending the new token.** After the historical K is converted to NHD, `new_k[i].unsqueeze(0)` adds the new decode token for request `i` immediately after its history. `new_k[i]` has shape `[n_kv_heads, head_dim]`; `.unsqueeze(0)` gives `[1, n_kv_heads, head_dim]` — one token in NHD layout. The result for request `i` is `[L_i + 1, n_kv_heads, head_dim]` when the two segments are listed consecutively in `segs_k`.

**`torch.cat` across all requests.** After the loop, `segs_k` contains `2B` tensors (history and new token for each request, interleaved). A single `torch.cat(segs_k, dim=0)` concatenates them all into `[total_kv_tokens, n_kv_heads, head_dim]` where `total_kv_tokens = sum(kv_len_i + 1)`. This is the exact tensor that FlashInfer expects, laid out in the order that `kv_indptr` describes.

**Saving for `write_back`.** `self._new_k[layer_idx] = new_k` stores the `[B, n_kv_heads, head_dim]` tensor containing the new token's K for every request. This is consumed after the full 28-layer forward pass by `write_back()`.

---

## `write_back()`: Growing Each `PerReqKVCache`

```python
def write_back(self) -> None:
    for layer_idx, new_k in self._new_k.items():
        new_v = self._new_v[layer_idx]
        for i, req in enumerate(self._reqs):
            # [n_kv, dim] → [1, n_kv, 1, dim]
            k_tok = new_k[i].unsqueeze(0).unsqueeze(2)
            v_tok = new_v[i].unsqueeze(0).unsqueeze(2)
            cache = req.kv_cache
            if layer_idx in cache._k:
                cache._k[layer_idx] = torch.cat([cache._k[layer_idx], k_tok], dim=2)
                cache._v[layer_idx] = torch.cat([cache._v[layer_idx], v_tok], dim=2)
            else:
                cache._k[layer_idx] = k_tok
                cache._v[layer_idx] = v_tok
```

`write_back()` is called once after the full 28-layer forward pass completes. For each layer that was processed (all 28 will appear in `self._new_k`), it extracts the new token's K for each request `i` — `new_k[i]` of shape `[n_kv_heads, head_dim]` — and reshapes it to `[1, n_kv_heads, 1, head_dim]` to match `PerReqKVCache`'s HND storage format. It then appends this tensor to `req.kv_cache._k[layer_idx]` via `torch.cat` on `dim=2` (the sequence dimension).

The result: every active request's `PerReqKVCache` grows by exactly one token across all 28 layers. The historical tensors that were read by `update()` are not modified — `update()` gathered them into the packed ragged tensor but never wrote back to them. Only the new token is written, keeping the per-request cache consistent without any redundant copying.

---

## `end_forward()`: Releasing FlashInfer State

```python
def end_forward(self) -> None:
    self._wrapper.end_forward()
```

`end_forward()` is called after `write_back()` in `decode_step`. It releases any internal state that FlashInfer allocated during `begin_forward` and stored in the workspace buffer. Without it, FlashInfer may hold references into the workspace that prevent the next decode step from reusing the buffer correctly. Like `plan()`, it is a bookkeeping call required by FlashInfer's API contract — the actual tensor work is done inside `wrapper.forward()` during the attention layers.

The sequence in `model_runner.decode_step` is therefore:

```python
pack_kv = PackedKVCache(reqs, self._workspace)   # build indptrs
pack_kv.plan(...)                                 # begin_forward
logits = self.model(...)                          # 28× update + wrapper.forward
pack_kv.write_back()                              # grow per-request caches
pack_kv.end_forward()                             # release FlashInfer state
```

`PackedKVCache` is a temporary object: it is created at the start of each decode step and garbage-collected after `end_forward()` returns. The durable state — the growing KV history — lives entirely in each request's `PerReqKVCache`, which survives across decode steps for the lifetime of the request.
