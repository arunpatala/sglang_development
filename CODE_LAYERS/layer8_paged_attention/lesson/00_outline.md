# Layer 8 — Lesson Outline

## What This Lesson Covers

Layer 7 eliminated padding waste in the attention kernel by replacing `BatchedKVCache` and `F.sdpa` with `PackedKVCache` and FlashInfer's ragged kernel. But a hidden per-step copy remained: every decode step, `PackedKVCache.update()` gathered all active requests' historical K/V tensors — living in separate `PerReqKVCache` allocations — and concatenated them into a new contiguous buffer. That gather costs `O(total_kv_tokens)` float bandwidth per step and grows as the batch accumulates history. Memory fragmentation also persisted: each `PerReqKVCache` grew by one `torch.cat` per layer per decode step, making repeated small allocations that the GPU allocator cannot efficiently reuse.

Layer 8 eliminates both by replacing `PerReqKVCache` and `PackedKVCache` with a single global `KVPool`. At server startup, `ModelRunner` pre-allocates two flat tensors per layer — `k_pool[layer]` and `v_pool[layer]` — sized to absorb 85% of free GPU memory. Every token ever processed is written into a slot in these tensors and stays there. No token's K/V data is ever copied again. The only per-step work is allocating a new slot index for the new token and building a `kv_indices` integer array that tells FlashInfer which pool slots belong to each request. FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` reads directly from the pool via that index — no gather, no float copy, no intermediate buffer.

The change touches two files: `kv_cache.py` (which adds `KVPool`, `PrefillKVCtx`, and `DecodeKVCtx` and removes everything from Layer 7) and `model/attention.py` (which updates the dispatch to distinguish `PrefillKVCtx` from `DecodeKVCtx`). `model_runner.py` changes to size and allocate the pool at startup and to build `kv_indptr`/`kv_indices` instead of constructing a `PackedKVCache`. `scheduler.py`, `request.py`, `server.py`, `tokenizer.py`, and the model layers are unchanged.

The sections follow the decode path top to bottom: the structural shift from gathering to indexing, the pool layout and slot arithmetic, prefill writing into the pool, decode building the index and calling FlashInfer, the attention dispatch between the two contexts, the full call trace, and what page_size=1 leaves on the table for Layer 9.

---

## Sections

### 01 — From Packed to Paged (`01_from_packed_to_paged.md`)
- Layer 7's `decode_step` with `PackedKVCache.update()`: the `torch.cat` gather of all per-request float KV histories every step; the `O(total_kv_tokens)` cost that grows with batch age
- Layer 8's `decode_step` with `KVPool`: no `PackedKVCache`, no `write_back()`; instead `kv_pool.alloc(1)` per request, `kv_indices` construction, `begin_forward`, model forward, `end_forward`, `req.slot_indices.append`
- The one startup addition: `KVPool` sized from `torch.cuda.mem_get_info()` after model load — `_KV_MEMORY_FRACTION = 0.85` of free GPU memory; `kv_memory_fraction` can be overridden from `config.yml`
- What is unchanged: `prefill()` still runs B=1 per request, position IDs are still per-request `kv_len_i`, sampling and finished-request handling are identical

### 02 — The KV Pool (`02_the_kv_pool.md`)
- The flat pool layout: `k_pool[layer] = [total_slots, n_kv_heads, head_dim]` — one row per token slot, shared across all requests, all decode steps; slot 0 reserved as FlashInfer padding
- Pool sizing at `ModelRunner.__init__`: `free_bytes * kv_memory_fraction / bytes_per_token` where `bytes_per_token = n_layers * 2 * n_kv_heads * head_dim * 2` (bfloat16)
- `KVPool.alloc(n)`: pops `n` entries from `free_slots` (a Python `List[int]`) — `O(n)`, cheap; raises `RuntimeError` on exhaustion (OOM)
- `KVPool.free(slots)`: extends `free_slots` with the returned indices — immediate, no GC dependency; contrast with `PerReqKVCache` whose tensors waited for Python GC
- `req.slot_indices: List[int]` — stored on the request object; grows by 1 per decode step (an integer append, not a tensor allocation)

### 03 — Prefill: Writing to the Pool (`03_prefill_writing_to_pool.md`)
- Prefill in Layer 8 opens the same way as Layer 7: B=1, `F.sdpa`, causal mask — but instead of a `PerReqKVCache` that appends to a growing tensor, it uses a `PrefillKVCtx` that writes into pre-allocated pool slots
- `model_runner.prefill`: `kv_pool.alloc(prompt_len)` → `req.slot_indices = slots`; builds `PrefillKVCtx(slots, kv_pool)` and passes it as `kv_cache=ctx` to the model
- `PrefillKVCtx.store(layer_idx, k, v)`: converts `k` from `[1, n_kv, L, D]` to `[L, n_kv, D]` (NHD) via `squeeze(0).permute(1,0,2).contiguous()`, then scatter-writes into pool rows `k_pool[layer_idx][slot_t] = k_nhd` — an in-place indexed write, no allocation
- Why prefill still uses `F.sdpa`: the prompt's K/V is freshly computed and rectangular; the pool write is a side-effect that stores the result for future decode steps; FlashInfer's paged decode kernel is not needed until decode begins
- `PrefillKVCtx.prefill_slots` as the dispatch marker: `attention.py` checks `hasattr(kv_cache, "prefill_slots")`; `DecodeKVCtx` does not have this attribute

### 04 — Decode: Indexing the Pool (`04_decode_indexing_the_pool.md`)
- Every decode step, Layer 8 builds a `kv_indices` integer array rather than a float tensor: for each request, `[historical_slots..., new_slot]`; this is a `List[int]` → `torch.int32` conversion, not a float gather
- `kv_indptr [B+1]`: cumulative sum of `(len(req.slot_indices) + 1)` — same role as in Layer 7's `PackedKVCache` but constructed from integer slot counts, not float KV lengths
- `kv_last_page_lens [B]`: always `ones(B)` because `page_size=1` — every slot holds exactly one token, so the last page always has exactly one entry
- `BatchDecodeWithPagedKVCacheWrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens, n_q_heads, n_kv_heads, head_dim, page_size=1, data_type, q_data_type)` — one call per decode step; FlashInfer plans the kernel and builds its internal index
- `DecodeKVCtx`: carries `wrapper`, `k_pool`, `v_pool` (references, no copy), and `new_slots [B]` (the freshly allocated slot per request); attention.py calls `ctx.store()` then `ctx.wrapper.forward()`

### 05 — The Attention Dispatch (`05_the_attention_dispatch.md`)
- Layer 8 `attention.py` has two dispatch branches where Layer 7 had one: `hasattr(kv_cache, "prefill_slots")` for prefill and `hasattr(kv_cache, "wrapper")` for decode; the no-cache branch for standalone testing remains
- Prefill branch: `kv_cache.store(layer_idx, k, v)` writes pool slots (side-effect), then `repeat_kv` + `F.sdpa` runs self-attention over the prompt — same causal mask as Layer 7's prefill path
- Decode branch: `kv_cache.store(layer_idx, k_fi, v_fi)` writes the new token to `new_slots` first (FlashInfer reads it immediately below); then `k_paged = kv_cache.k_pool[layer].unsqueeze(1)` — the `unsqueeze(1)` inserts the page dimension FlashInfer expects (`[total_slots, page_size=1, n_kv, head_dim]`)
- `wrapper.forward(q_fi, (k_paged, v_paged))` — FlashInfer uses `kv_indices` from `begin_forward` to slice the relevant rows from the pool for each request; output `[B, n_q_heads, head_dim]`, unsqueeze back to `[B, n_q_heads, 1, head_dim]`
- GQA: FlashInfer handles 16Q/8KV natively on the decode path; `repeat_kv` is still needed on the prefill `F.sdpa` path (unchanged from all previous layers)

### 06 — The Full Loop (`06_the_full_loop.md`)
- End-to-end trace of two concurrently running requests, naming every component in execution order
- Step 1 — Request arrival and prefill: `kv_pool.alloc(prompt_len)` → `PrefillKVCtx(slots, pool)` → 28 attention layers each call `ctx.store()` (pool write) + `F.sdpa` (self-attention); first token sampled; `req.slot_indices` holds all prompt slot indices
- Step 2 — `decode_step` entry: `kv_pool.alloc(1)` per request for the new token slot; `kv_indices` and `kv_indptr` built from `req.slot_indices + new_slot` — integer construction only, no float data touched
- Step 3 — `begin_forward` and forward pass: one `begin_forward` call; 28 attention layers each call `ctx.store()` (writes new K/V to `new_slots` in pool), then `wrapper.forward()` (FlashInfer reads the full slot history from the pool via `kv_indices` — no copy, no intermediate buffer)
- Step 4 — `end_forward`, slot append, sampling: `end_forward()` releases FlashInfer state; `req.slot_indices.append(new_slot)` (integer list append, not tensor allocation); sampling and result handling identical to Layer 7; finished requests call `kv_pool.free(req.slot_indices)` — slots immediately returned to the free list

### 07 — What Comes Next (`07_whats_next.md`)
- The remaining inefficiency at `page_size=1`: every slot holds exactly one token, so FlashInfer constructs an index entry per token; for a 1000-token history this means 1000 index lookups per layer per decode step — still `O(total_kv_tokens)` but in integers, not floats
- Layer 9 (paged attention with larger page size): grouping tokens into fixed-size pages (e.g., 16 tokens/page) reduces the `kv_indices` array from `total_tokens` to `ceil(total_tokens / page_size)` entries; the `kv_last_page_lens` tensor correctly handles partially-filled pages at the boundary
- What page sizing changes: `KVPool` shape becomes `[total_pages, page_size, n_kv, dim]`; `alloc` and `free` operate on pages not tokens; `kv_indptr` counts pages not tokens; `kv_last_page_lens` carries the fill count of the last page per request
- What stays the same: `KVPool.free()` for finished requests, per-request `slot_indices` (now page indices), pool sizing from GPU memory, the same `BatchDecodeWithPagedKVCacheWrapper` API

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 8 concepts to SGLang source: `KVPool` → `ReqToTokenPool` / `TokenToKVPool` in `srt/mem_pool.py`; `kv_indices` → `req_to_token_pool`; `kv_indptr` → built in `ModelRunner.forward_batch`; `KVPool.free` → `token_to_kv_pool.free_slots`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Pool sizing from free GPU memory | `model_runner.py` line 74: `free_bytes, _ = torch.cuda.mem_get_info()` |
| `bytes_per_token` formula | `model_runner.py` line 75: `cfg.num_hidden_layers * 2 * cfg.num_key_value_heads * cfg.head_dim * 2` |
| `KVPool` class | `kv_cache.py` line 90: `class KVPool:` |
| `k_pool` / `v_pool` allocation | `kv_cache.py` line 113: `torch.zeros(total_slots, n_kv_heads, head_dim, dtype=dtype, device=DEVICE)` |
| Free-slot list init | `kv_cache.py` line 123: `self.free_slots: List[int] = list(range(1, total_slots))` |
| `KVPool.alloc` | `kv_cache.py` line 137: `def alloc(self, n: int) -> List[int]:` |
| `KVPool.free` | `kv_cache.py` line 147: `def free(self, slots: List[int]) -> None:` |
| `PrefillKVCtx` class | `kv_cache.py` line 156: `class PrefillKVCtx:` |
| `PrefillKVCtx.store` — scatter write | `kv_cache.py` line 177: `self._kv_pool.k_pool[layer_idx][self._slot_t] = k_nhd` |
| `DecodeKVCtx` class | `kv_cache.py` line 197: `class DecodeKVCtx:` |
| `DecodeKVCtx.store` — new token write | `kv_cache.py` line 230: `self.k_pool[layer_idx][self.new_slots] = k_fi` |
| `kv_indptr` / `kv_indices` build | `model_runner.py` line 186: `kv_lens_plus1 = [len(r.slot_indices) + 1 for r in reqs]` |
| `begin_forward` call | `model_runner.py` line 219: `decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens, ...)` |
| Prefill dispatch (`prefill_slots`) | `model/attention.py` line 116: `if kv_cache is not None and hasattr(kv_cache, "prefill_slots"):` |
| Pool write in prefill | `model/attention.py` line 120: `kv_cache.store(self.layer_idx, k, v)` |
| Decode dispatch (`wrapper`) | `model/attention.py` line 133: `elif kv_cache is not None and hasattr(kv_cache, "wrapper"):` |
| Pool tensor page-dim convention | `model/attention.py` line 147: `k_paged = kv_cache.k_pool[self.layer_idx].unsqueeze(1)` |
| FlashInfer paged forward | `model/attention.py` line 151: `attn_out = kv_cache.wrapper.forward(q_fi, (k_paged, v_paged))` |
| Slot append after decode step | `model_runner.py` line 252: `req.slot_indices.append(new_slots[i])` |
| Pool free on finish | `model_runner.py` line 265: `self.kv_pool.free(req.slot_indices)` |
| `kv_memory_fraction` in config | `config.yml` — `kv_memory_fraction: 0.85`; read by `server.py` and forwarded to `ModelRunner` |
