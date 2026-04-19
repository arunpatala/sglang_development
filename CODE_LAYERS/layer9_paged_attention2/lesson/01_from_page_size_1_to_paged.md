# 01 — From Page-size-1 to Paged

## From Layer 8 to Layer 9

Layer 8 eliminated the per-step float KV gather by replacing `PackedKVCache` with a pre-allocated `KVPool` and building a `kv_indices` integer array each decode step. Two costs remained. First, `kv_indices` was assembled in Python by iterating over each request's `slot_indices` list and appending the new slot — an `O(Σ kv_tokens)` Python loop followed by a `torch.tensor(...)` call that sent the entire array to the GPU:

```python
# Layer 8 — decode_step (key lines)
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_indices_list = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # O(kv_len) per request
    kv_indices_list.append(new_slots[i])

kv_indptr   = torch.tensor(kv_indptr_list,  dtype=torch.int32, device=DEVICE)
kv_indices  = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)   # CPU→GPU
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)             # always 1

decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD")
decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens,
                             ..., page_size=1, ...)
ctx = DecodeKVCtx(wrapper=decode_wrapper, k_pool=kv_pool.k_pool,
                  v_pool=kv_pool.v_pool, new_slots=new_slots_t)
fb  = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)
decode_wrapper.end_forward()
for i, req in enumerate(reqs):
    req.slot_indices.append(new_slots[i])
```

Second, `page_size=1` meant the pool had one row per token. For a batch with B=16 requests averaging 500 accumulated tokens each, `kv_indices` held 8000 entries — 32 KB of int32 transferred to the GPU every step, requiring 8000 individual pool-row lookups per attention layer. FlashInfer's index-lookup cost scaled linearly with total token history, and `kv_last_page_lens` was always ones because every "page" held exactly one token.

In Layer 9, the same decode step reads:

```python
# Layer 9 — decode_step (key lines)
seq_lens_list      = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]
token_offsets_list = [sl % P for sl in seq_lens_list]
num_pages_list     = [len(r.slot_indices) for r in reqs]

# Conditional page alloc: one int32 write when the last page fills
for i, req in enumerate(reqs):
    if token_offsets_list[i] == 0:
        new_page = self.kv_pool.alloc(1)[0]
        req.slot_indices.append(new_page)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, num_pages_list[i]] = new_page
    # else: existing last page has room — no alloc, no table write

kv_last_page_lens = token_offsets_t + 1           # range 1..P, not always 1

torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])  # GPU, no copy
total_pages_in_batch = sum(num_pages_list)
kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)

create_flashinfer_kv_indices_triton[(B,)](          # Triton, all B requests in parallel
    self.req_to_token_pool.req_to_token,
    req_pool_idx_t, num_pages_t, kv_indptr, None, kv_indices,
    self.req_to_token_pool.req_to_token.shape[1],
)

self._decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens,
                                   ..., P, ...)    # P=16, reused from __init__
ctx = DecodeKVCtx(wrapper=self._decode_wrapper, k_pool=kv_pool.k_pool,
                  v_pool=kv_pool.v_pool, last_page_indices=last_page_idx_t,
                  token_offsets=token_offsets_i64)
fb  = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)
logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)
self._decode_wrapper.end_forward()
```

The Python loop over `reqs` is now O(B) — it reads `len(input_ids) + len(output_ids)` per request, not the slot list. For the same B=16 batch with 500-token average history, the CPU sends five `[B]` int tensors totalling about 320 bytes to the GPU each step, and `kv_indices` has `ceil(8000/16) = 500` entries instead of 8000. The Triton kernel reads page data directly from the GPU-resident `req_to_token` table; no Python iteration over historical slot lists, no CPU-to-GPU copy of that data, ever.

---

## What Changes, What Does Not

Three things changed in `model_runner.py`. The `seq_len` formula is now `len(r.input_ids) + len(r.output_ids) - 1`, subtracting one because `r.output_ids[-1]` is the current input token — already in `output_ids` but not yet stored in the pool. The decode wrapper is no longer reconstructed on every step; it is created once in `ModelRunner.__init__` and reused. The `kv_indptr` buffer is pre-allocated at init and filled with `torch.cumsum` on the GPU rather than `itertools.accumulate` on the CPU.

Two new structures appear. `ReqToTokenPool` is a GPU-resident `[max_batch, max_pages_per_req]` int32 table that holds page indices on-device, populated at prefill and updated only when a new page is needed. `create_flashinfer_kv_indices_triton` is a Triton kernel that reads from that table and writes `kv_indices` on-GPU.

`kv_cache.py` changes to accommodate the larger page size: the `KVPool` pool tensors have a new `page_size` dimension, `PrefillKVCtx.store` pads and reshapes the prompt K/V into pages before writing, and `DecodeKVCtx.store` uses a 2D `(last_page, token_offset)` index instead of a flat `new_slots` index.

Everything else is unchanged. Prefill is still B=1 per request. `ForwardBatch` carries the same fields. `PagedBackend` in `model/backend.py` drops the `unsqueeze(1)` on the pool tensors — the page dimension is now native to the pool shape — but its structure and the `_prefill_forward` / `_decode_forward` routing are identical to Layer 8. `scheduler.py`, `server.py`, `request.py`, and `tokenizer.py` have no changes. Sampling and request lifecycle are identical.

The sections below explain each new piece in code order. Section 02 covers `ReqToTokenPool`: what it stores, how rows are assigned, and how prefill and decode update it. Section 03 covers the paged `KVPool` and how `PrefillKVCtx.store` packs prompt K/V into pages. Section 04 covers the decode step's conditional page allocation, `seq_len` arithmetic, and the GPU index build. Section 05 covers the Triton kernel itself.
