# 01 — From Repeated Writes to Shared Pages

## From Layer 11 to Layer 12

Layer 11 established chunked prefill: long prompts are split into `chunked_prefill_size`-token slices, each processed in a separate `prefill_batch` call. The K/V for each slice is written into fresh pool pages, and the extend kernel reads the accumulating history from `kv_indices` on subsequent chunks. Every request — regardless of how much of its prompt it shares with other requests — computes and stores its own complete K/V.

This is correct but wasteful when many requests share a common prefix. Consider a chatbot with a 512-token system prompt repeated verbatim in every request. In Layer 11:

```python
# Layer 11 — prefill_batch for request 2 (same system prompt as request 1)
req.fill_ids         = req.input_ids          # all 1024 tokens including system prompt
req.extend_input_len = 1024
req.kv_committed_len = 0

compute_write_info(..., kv_committed_len=0, n_fill=1024)
# Allocates 64 new pages, writes ALL 1024 tokens' K/V
# The first 32 pages are identical to request 1's first 32 pages — wasted
```

For 100 concurrent requests with the same 512-token system prompt, the pool holds 100 × 32 = 3200 pages of system-prompt K/V. Only 32 pages are unique. The other 3168 pages consume pool space that could hold unique decode history, limiting effective batch size.

In Layer 12, the same scenario reads:

```python
# Layer 12 — PrefillAdder._apply_prefix_match sets:
req.prefix_page_indices = [p0, p1, ..., p31]   # the 32 cached pages
req.prefix_len          = 512
req.kv_committed_len    = 512                   # skip cached tokens
req.last_node           = <tree node for system prompt>

# fill_ids is set to input_ids[512:]  — the unique suffix only
req.fill_ids         = req.input_ids[512:]
req.extend_input_len = len(req.input_ids) - 512
```

`prefill_batch` injects the 32 cached page indices into `req_to_token_pool`, then calls `compute_write_info` with `kv_committed_len=512` — which sees the 32 existing pages already in the pool row and allocates only the pages needed for the suffix. The extend kernel processes the suffix tokens and attends over the full history (cached + new) via `kv_indices`. No K/V for the system prompt is recomputed.

---

## What Changes and What Does Not

One new file — `radix_cache.py` — is the entire change to the KV management layer. It contains the `TreeNode` struct, the `RadixCache` class with `match_prefix`, `insert`, `evict`, and `cache_finished_req`, and the `lock_ref` reference counting that prevents active pages from being evicted.

`scheduler.py` gains `_apply_prefix_match` in `PrefillAdder` — five lines that call `match_prefix` and set the four prefix-related fields on the request. `cache_finished_req` is called on request completion instead of the bare `kv_pool.free` + `req_to_token_pool.free` pair.

`model_runner.prefill_batch` gains two additions: a loop that writes `prefix_page_indices` into `req_to_token_pool` before `compute_write_info`, and an eviction guard that calls `radix_cache.evict` when the pool is low.

`decode_step`, `forward_batch.py`, `ExtendKVCtx`, `DecodeKVCtx`, `WriteInfo`, the Triton kernel, and all model files are entirely unchanged. The extend kernel sees page indices that happened to come from the tree rather than from fresh allocation — it cannot and does not need to distinguish the two.

---

## The Ownership Model

Pages in the pool can be owned in three states:

A page is **request-owned** while a request is actively using it during prefill or decode. `slot_indices` lists these pages and `kv_pool.free(slot_indices)` returns them when the request finishes.

A page is **tree-owned** after `cache_finished_req` inserts the request's sequence into the `RadixCache`. The tree holds the page indices in its node values. No explicit free-list entry is held — the tree owns the page until eviction.

A page is **locked** while a request is actively attending over it (from prefill setup to the request's completion). `lock_ref > 0` on every ancestor node from the deepest matching node to the root. A locked page cannot be evicted — `radix_cache.evict` skips nodes with `lock_ref > 0`. When the request finishes, `dec_lock_ref` walks the ancestor chain and decrements, potentially making those pages eviction-eligible.

This three-state model ensures that a page in the tree can be lent to a new request (via `match_prefix`) without being freed by a concurrent eviction while the request is mid-decode. Section 03 explains the lock mechanics in detail. Section 04 explains how `insert` and `cache_finished_req` transition pages from request-owned to tree-owned.
