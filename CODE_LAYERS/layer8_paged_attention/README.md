# Layer 7 — Paged KV Cache

Extends Layer 6 (packed batching) by replacing the per-request `PerReqKVCache`
growing lists with a **pre-allocated global KV pool** indexed by integer slot
tables.  Zero-copy paged attention via FlashInfer.

---

## What changed vs Layer 6

| Aspect | Layer 6 (PackedKVCache) | Layer 7 (Paged KV Cache) |
|--------|------------------------|--------------------------|
| KV storage | One growing tensor per request (`PerReqKVCache`) | Single global pool `[total_slots, n_kv, head_dim]` |
| Decode KV gather | Copy all historical K/V into packed buffer (float copy) | Build `kv_indices` integer list; no KV data moved |
| Memory reclaim | GC'd when Python object is freed | Instant: `kv_pool.free(req.slot_indices)` returns ints |
| FlashInfer API | `BatchPrefillWithRaggedKVCacheWrapper` | `BatchPrefillWithPagedKVCacheWrapper` (page_size=1) |
| Prefill attention | F.sdpa + write to `PackedKVCache` | F.sdpa + write to `KVPool` slots |
| Request object | `req.kv_cache` (PerReqKVCache) | `req.slot_indices` (List[int]) |

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                       KVPool (global)                      │
│                                                           │
│  k_pool[layer]:  [total_slots, n_kv_heads, head_dim]     │
│  v_pool[layer]:  [total_slots, n_kv_heads, head_dim]     │
│                                                           │
│  free_slots: [1, 2, 3, 4, 5, ...]  (slot 0 = padding)   │
└───────────────────────────────────────────────────────────┘
          ↑ alloc(prompt_len)        ↑ alloc(1) per step
          │ prefill                  │ decode
          ↓ free(slot_indices)       │
         finished requests           │

req.slot_indices = [42, 18, 91, 7, ...]  ← one int per token
                                             grows each decode step
```

### Prefill  (B=1, `PrefillKVCtx`)

```
model(ids, kv_cache=PrefillKVCtx(slots))
  → each attention layer:
      kv_cache.store(layer, k, v)          # k/v → pool[slots]   ← side effect
      F.sdpa(q, k_rep, v_rep, causal_mask) # fresh tensors, no pool read
```

### Decode  (B=N, `DecodeKVCtx`)

```
build kv_indices = [hist_slots_req0..., new_slot_0,
                    hist_slots_req1..., new_slot_1, ...]   # integers only

wrapper.begin_forward(qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, ...)

model(last_toks, kv_cache=DecodeKVCtx(wrapper))
  → each attention layer:
      kv_cache.store(layer, k_fi, v_fi)    # new token → pool[new_slots]
      wrapper.forward(q, (k_pool[l].unsqueeze(1), v_pool[l].unsqueeze(1)))
                                           # FlashInfer reads pool via kv_indices
wrapper.end_forward()
```

No KV data is copied.  The only allocation each step is one `int` slot per request.

---

## Key data structures

### `KVPool`
- `k_pool[l]`, `v_pool[l]`: pre-allocated at server start, never reallocated
- `free_slots: List[int]`: O(1) pop for alloc, O(k) extend for free
- Pool size = `available_GPU_memory × 0.85 / bytes_per_token`

### `PrefillKVCtx`  (duck-typed by `hasattr(obj, 'prefill_slots')`)
- Holds `slot_indices` (prompt token slots) and a reference to `KVPool`
- `store(layer, k, v)`: scatter `[prompt_len, n_kv, head_dim]` into pool rows

### `DecodeKVCtx`  (duck-typed by `hasattr(obj, 'wrapper')`)
- Holds FlashInfer wrapper, pool references, and `new_slots` tensor
- `store(layer, k_fi, v_fi)`: write `[B, n_kv, head_dim]` into `new_slots` rows

### FlashInfer paged API
```
BatchPrefillWithPagedKVCacheWrapper (q_len=1 per request = decode behaviour)

begin_forward(qo_indptr,           # [B+1]  = [0, 1, 2, ..., B]
              kv_indptr,           # [B+1]  cumsum of kv_lens_with_new
              kv_indices,          # [Σ kv_len_i + B]  pool slot indices
              kv_last_page_lens,   # [B]    = [1, 1, ..., 1]  (page_size=1)
              num_qo_heads, num_kv_heads, head_dim,
              page_size=1,
              causal=False)

forward(q,                         # [B, n_q_heads, head_dim]
        (k_pool[layer].unsqueeze(1),   # [total_slots, 1, n_kv_heads, head_dim]
         v_pool[layer].unsqueeze(1)))
→ [B, n_q_heads, head_dim]
```

`unsqueeze(1)` adds the `page_size=1` dimension as a zero-copy view.

> **Why `BatchPrefillWithPagedKVCacheWrapper` and not `BatchDecodeWithPagedKVCacheWrapper`?**
> The decode variant requires JIT compilation (ninja) which may not be installed.
> The prefill wrapper has pre-built cubins and supports `q_len=1` per request
> (identical to decode semantics when `causal=False`).

---

## Files

| File | Role |
|------|------|
| `kv_cache.py` | `KVPool`, `PrefillKVCtx`, `DecodeKVCtx` |
| `model/attention.py` | Three-way dispatch: None / PrefillKVCtx / DecodeKVCtx |
| `model_runner.py` | KVPool init, prefill (slot alloc), decode_step (paged FlashInfer) |
| `request.py` | `slot_indices: List[int]` replaces `kv_cache: PerReqKVCache` |
| `scheduler.py` | Unchanged from Layer 6 |
| `server.py` | Unchanged from Layer 6 |

---

## Next: Layer 8 — Radix Cache / Prefix Caching

Paged layout naturally supports prefix sharing: if two requests share a common
prefix, their prompt slots can point to the **same** physical pages, cutting
prefill cost and pool consumption for repeated system prompts.
