# 06 — Integration with prefill_batch

The `RadixCache` lookup and the extend kernel are connected through three changes to `prefill_batch`: prefix page injection into `req_to_token_pool`, the eviction guard, and position IDs starting at `kv_committed_len`. These additions are surgical — the eight-step structure from Layer 11 is preserved, with new code inserted at the start of steps 2 and 3.

---

## Prefix Page Injection

```python
# model_runner.py — prefill_batch, step 2 (new in Layer 12)
for req in reqs:
    if req.prefix_page_indices:
        n_pfx = len(req.prefix_page_indices)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :n_pfx
        ] = torch.tensor(
            req.prefix_page_indices, dtype=torch.int32, device=DEVICE
        )
```

`prefix_page_indices` was set by `PrefillAdder._apply_prefix_match`. For a request with a 512-token prefix match (32 pages at `page_size=16`), this writes those 32 page indices into columns `0:32` of the request's `req_to_token_pool` row.

This write must happen before `compute_write_info`. The reason: `compute_write_info` calls `kv_pool.alloc` and writes new page indices into `req_to_token_pool` at positions `n_prev : n_prev + len(new_pages)`. `n_prev = len(slot_indices)`. If `slot_indices` is empty (fresh request), `n_prev = 0` and the new pages start at column 0, overwriting the prefix pages that were just injected. The correct behavior requires `slot_indices` to already contain the prefix page indices before `compute_write_info` is called.

But `PrefillAdder._apply_prefix_match` sets `req.slot_indices = prefix_page_indices` as a side effect:

```python
# scheduler.py — _apply_prefix_match
req.prefix_page_indices = page_indices
req.slot_indices        = list(page_indices)   # so compute_write_info sees n_prev = n_pfx
req.kv_committed_len    = matched_len
```

When `compute_write_info` runs, `slot_indices` already has `n_pfx` entries. New pages are written at columns `n_pfx : n_pfx + len(new_pages)`. The prefix and suffix page regions are correctly disjoint.

---

## Position IDs Starting at kv_committed_len

```python
# model_runner.py — prefill_batch, step 4
pos_list: List[int] = []
for req in reqs:
    for j in range(req.extend_input_len):
        pos_list.append(req.kv_committed_len + j)
```

For a prefix-cached request with `kv_committed_len = 512` and `extend_input_len = 512` (a 512-token suffix), positions are `[512, 513, ..., 1023]`. Rotary embeddings use these absolute positions to compute the correct angle for each token. If positions started at 0, the suffix tokens would have the wrong RoPE encoding — queries would not match keys at the right positions, corrupting the attention computation.

This is the same logic as the continuation-chunk case from Layer 11. Prefix caching is, in effect, a "free chunk": the prefix tokens have `kv_committed_len` advancing without actually processing them through the extend kernel. The position ID formula handles both cases uniformly.

---

## qo_indptr and kv_indptr with Prefix

```python
# model_runner.py — prefill_batch, step 5
for req in reqs:
    qo_indptr_list.append(qo_indptr_list[-1] + req.extend_input_len)   # suffix only
    total_committed = req.kv_committed_len + req.extend_input_len
    n_pages = len(req.slot_indices)    # prefix pages + new pages
    num_pages_list.append(n_pages)
    last_fill = total_committed % P
    kv_last_pg_list.append(last_fill if last_fill != 0 else P)
```

`qo_indptr` counts only the suffix tokens — the query positions in the extend kernel. For a 512-token prefix + 512-token suffix, `qo_indptr` covers 512 query tokens, not 1024.

`kv_indptr` counts all pages — both prefix pages and suffix pages. `n_pages = len(req.slot_indices)` is 64 (32 prefix + 32 suffix). The Triton kernel reads all 64 pages from `req_to_token_pool` (prefix pages written in step 2, suffix pages written by `compute_write_info`).

`kv_last_page_lens` is computed from `total_committed = kv_committed_len + extend_input_len = 512 + 512 = 1024`. `1024 % 16 = 0`, so the last page is full: `kv_last_pg = 16`.

FlashInfer's extend kernel then attends 512 query tokens over 1024 KV positions (64 pages), with a causal mask. Query token at position 512 attends over KV positions 0–512; query token at position 1023 attends over KV positions 0–1023. The prefix tokens' K/V was written by a prior request's forward pass — the current request reads them transparently, as if they had been computed locally.

---

## cache_finished_req in the Scheduler

In Layer 11, request completion freed the KV pool and table row directly:

```python
# Layer 11 — Scheduler.run, on FINISHED
self.kv_pool.free(req.slot_indices)
self.req_to_token_pool.free(req.req_pool_idx)
```

In Layer 12, this is replaced by:

```python
# Layer 12 — Scheduler.run, on FINISHED
if self.radix_cache is not None:
    self.radix_cache.cache_finished_req(req, self.req_to_token_pool, self.kv_pool)
else:
    self.kv_pool.free(req.slot_indices)
    self.req_to_token_pool.free(req.req_pool_idx)
```

`cache_finished_req` does all of: insert into tree, free overlap pages, free tail page, dec_lock_ref, free req_pool_idx. If prefix caching is disabled (`enable_prefix_caching=False`, `radix_cache=None`), the Layer 11 behavior is preserved exactly. The `else` branch ensures backward compatibility.

Section 07 traces the full loop for two requests sharing a system prompt to show all these pieces working end to end.
