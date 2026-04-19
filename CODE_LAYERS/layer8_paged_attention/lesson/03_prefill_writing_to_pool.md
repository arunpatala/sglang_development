# 03 — Prefill: Writing to the Pool

Prefill in Layer 8 allocates pool slots before the forward pass and writes K/V into them as a side-effect during the forward pass. The attention computation itself — `F.sdpa` over the prompt — is unchanged from Layer 7. What changes is the destination of the K/V tensors: instead of growing a per-request `PerReqKVCache`, they are written directly into the rows of the global `KVPool`.

---

## Slot Allocation in `model_runner.prefill`

```python
def prefill(self, req: Req) -> None:
    prompt_len = len(req.input_ids)

    slots = self.kv_pool.alloc(prompt_len)
    req.slot_indices = slots   # store on req for decode steps to extend

    ids  = torch.tensor([req.input_ids], device=DEVICE)          # [1, L]
    mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
    pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)  # [1, L]

    ctx = PrefillKVCtx(slots, self.kv_pool)
    fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
    with torch.no_grad():
        logits = self.model(ids, forward_batch=fb, position_ids=pos)
```

`kv_pool.alloc(prompt_len)` pops `prompt_len` slot indices from the free list. As section 02 established, this is a pure Python list slice — no GPU work. The returned list is stored on `req.slot_indices`, where it persists for the lifetime of the request and grows by one entry per decode step.

`PrefillKVCtx` is a lightweight context object that carries the slot list and a reference to the pool. `ForwardBatch` wraps it with `mode=ForwardMode.PREFILL` and `attention_mask=mask`. The binary mask (1 for every real token, none are padding in B=1 prefill) is passed through `ForwardBatch.attention_mask` and read by `PagedBackend._prefill_forward` to build the additive causal mask.

---

## `PrefillKVCtx` and the Scatter Write

```python
class PrefillKVCtx:
    def __init__(self, slot_indices: List[int], kv_pool: KVPool) -> None:
        self._kv_pool = kv_pool
        self._slot_t  = torch.tensor(slot_indices, dtype=torch.int64, device=DEVICE)

    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        # k: [1, n_kv_heads, prompt_len, head_dim]
        k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()   # [L, n_kv, D]
        v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()
        self._kv_pool.k_pool[layer_idx][self._slot_t] = k_nhd
        self._kv_pool.v_pool[layer_idx][self._slot_t] = v_nhd
```

`_slot_t` is a pre-converted `int64` tensor built once in `__init__`, not on every `store` call. PyTorch's advanced (fancy) indexing requires an integer tensor as the index; converting the Python list once and reusing it avoids a repeated CPU-to-GPU transfer inside the 28 `store` calls that happen during the forward pass.

The layout conversion matters. `k` arrives from `Qwen3Attention` as `[1, n_kv_heads, prompt_len, head_dim]` — the standard `[B, heads, seq, dim]` format that `F.sdpa` and the rest of the model use. The pool rows are `[n_kv_heads, head_dim]` per slot, and multiple slots must be written at once. The target layout for the scatter is `[prompt_len, n_kv_heads, head_dim]` — one row per token, matching the NHD convention that `k_pool[layer]` uses.

`squeeze(0)` removes the batch dimension (batch size is always 1 during prefill), giving `[n_kv_heads, prompt_len, head_dim]`. `permute(1, 0, 2)` reorders the axes to `[prompt_len, n_kv_heads, head_dim]`. `contiguous()` ensures the tensor is laid out in the new order in memory before the scatter write — without it, the advanced-index assignment would need to deal with non-contiguous strides, which can trigger implicit copies or incorrect writes.

`k_pool[layer_idx][self._slot_t] = k_nhd` is PyTorch's advanced index assignment. It is equivalent to `k_pool[layer_idx][slot_t[0]] = k_nhd[0]`, `k_pool[layer_idx][slot_t[1]] = k_nhd[1]`, etc., but issued as a single fused scatter on the GPU. The pool rows are updated in place; no new tensor is allocated. After `store` returns, `k_pool[layer_idx]` permanently holds the key vectors for all prompt tokens at `layer_idx`, addressed by their slot indices. Future decode steps will read these rows via `kv_indices` — no copy is ever needed.

---

## Why Prefill Still Uses `F.sdpa`

Attention during prefill runs `F.sdpa` over the freshly computed K and V tensors, not over the pool. The `store` call is a side-effect that happens before attention runs:

```python
# PagedBackend._prefill_forward (simplified)
if forward_batch.kv_cache is not None:
    forward_batch.kv_cache.store(layer_idx, k, v)   # pool write (side-effect)

k_rep = repeat_kv(k, self.num_kv_groups)             # GQA expansion
v_rep = repeat_kv(v, self.num_kv_groups)
additive_mask = build_additive_mask(
    forward_batch.attention_mask, q_len, kv_len, q.dtype, q.device
)
return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale)
```

The reason prefill does not use the paged decode kernel is architectural: `BatchDecodeWithPagedKVCacheWrapper` is designed for the case where every query token attends over a long KV history that already exists in the pool. During prefill, there is no prior history — every token is attending over other tokens in the same prompt. The attention pattern is a standard causal self-attention over a rectangular `[1, heads, prompt_len, head_dim]` K/V, for which `F.sdpa` with a causal mask is the correct and most efficient kernel. FlashInfer's paged decode kernel would have no pool history to read and would require the prompt's K/V to already be in the pool before attention runs — a chicken-and-egg problem that the side-effect pattern sidesteps.

After `model_runner.prefill` returns, `req.slot_indices` holds the complete list of pool row indices for this request's prompt, across all 28 layers. The pool write is permanent and requires no write-back step — unlike `PackedKVCache.write_back()` in Layer 7, which had to scatter new K/V back to each `PerReqKVCache` after the forward pass. The pool write during `store()` is the write-back.
