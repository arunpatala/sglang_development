# 07 — What Comes Next

Layer 8 eliminates the per-step float KV gather that grew unboundedly with batch history length. No K/V data is copied after the initial pool write during prefill. The only per-step overhead on the decode path is building `kv_indices` — a flat list of slot integers — and calling `begin_forward` once. This is a qualitative improvement: the bottleneck shifts from float memory bandwidth (Layer 7) to integer index construction and kernel scheduling (Layer 8).

But one structural inefficiency remains.

---

## The `page_size=1` Overhead

Every slot in the pool holds exactly one token. `kv_indices` therefore contains one entry per accumulated token across all active requests:

```python
kv_indices_list = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # one int per historical token
    kv_indices_list.append(new_slots[i])
```

For a batch with `T = total_kv_tokens` accumulated across all requests, `kv_indices` has `T + B` entries. `kv_indptr` has `B + 1` entries. `begin_forward` must ingest `T + B` integers at the start of every decode step. With `T = 10000`, that is 10000 `int32` entries — about 40 KB — transferred to the GPU once per step. The integer cost is two orders of magnitude cheaper than the float cost it replaced, but it is still `O(T)`.

FlashInfer's kernel also performs `T + B` pool-row lookups per layer — one per entry in `kv_indices`. With `page_size=1` and one entry per token, the number of random-access reads into the pool equals the total number of accumulated tokens. For large batches with long histories, this becomes a significant irregular-memory-access pattern.

---

## What Page Size Changes

Layer 9 groups tokens into fixed-size pages. With `page_size=16`, each page holds 16 consecutive tokens from a single request. A request with 51 accumulated tokens occupies `ceil(51/16) = 4` pages: three full pages of 16 tokens and one partial page with 3 tokens.

The pool shape changes to:

```python
# Layer 9 — pool shape per layer
k_pool[layer]: [total_pages, page_size, n_kv_heads, head_dim]
# e.g.         [3500,        16,         8,          128]
```

`kv_indices` now contains page indices rather than token indices. For the same 51-token request, `kv_indices` holds 4 entries instead of 51. For a batch with `T = 10000` total accumulated tokens and `page_size=16`, `kv_indices` has `ceil(10000/16) = 625` entries instead of 10000 — a 16× reduction.

`kv_last_page_lens` carries the actual fill count of each request's last page:

```python
# Layer 9 — kv_last_page_lens for a 51-token request
kv_last_page_len = 3   # 51 mod 16 = 3 tokens in the last page
```

FlashInfer uses `kv_last_page_lens` to know how many of the last page's slots contain real tokens and how many are empty padding rows. The empty slots are never attended over, so there is no correctness cost — only the full pages and the partial last page contribute to the attention output.

---

## What Stays the Same

`KVPool` survives as the global pre-allocated flat store, reshaped to accommodate page dimensions. `alloc` and `free` operate on pages rather than individual token slots, but the free-list mechanics are identical. `kv_pool.free(req.slot_indices)` returns all of a finished request's pages to the free list in one call, with no GC dependency.

`req.slot_indices` becomes `req.page_indices` — a list of page indices that grows by one entry every `page_size` decode steps, rather than by one entry every step. For `page_size=16`, a request that generates 1000 tokens requires only 63 page-index appends instead of 1000. The reduction in Python-list append overhead mirrors the reduction in `kv_indices` construction cost.

The same `BatchDecodeWithPagedKVCacheWrapper` API is used; the only argument that changes is `page_size` in the `begin_forward` call. The backend dispatch in `model/backend.py` is unchanged — `_decode_forward` passes the pool tensors (with their new shape) to `wrapper.forward` identically.

`scheduler.py`, `request.py`, `server.py`, and `tokenizer.py` are unchanged. The pattern continues: one mechanism changes (`page_size` in the pool and the alloc/free unit), one parameter changes (`page_size` in `begin_forward`), and the benchmark measures exactly the reduction in `kv_indices` construction and FlashInfer lookup overhead.
