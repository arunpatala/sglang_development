# 08 — What Comes Next

Layer 11 eliminates decode starvation by capping how many tokens any single request contributes per scheduler iteration. The extend kernel, the FlashInfer paged-prefill API, and the page-packing arithmetic are now established foundations. One class of waste remains.

---

## The Shared-Prefix Problem

Consider a chatbot server where every request begins with the same 512-token system prompt — a set of instructions, persona definitions, or few-shot examples. In Layer 11, each incoming request goes through the full `prefill_batch` pipeline for its entire prompt. The system prompt's K/V is computed independently for every request and written into separate pool pages. For 100 concurrent requests with the same system prompt:

- 100 × 512 = 51,200 token positions of K/V are computed.
- 100 × 32 pages (at `page_size=16`) are allocated.
- Only 32 pages' worth of K/V data is actually unique — the other 3168 pages are bit-for-bit identical duplicates.

The GPU work is proportional to 51,200 positions; the correct work would be proportional to 512 positions plus 100 × (unique suffix length). The pool fills with 99 redundant copies of the same data, reducing the effective cache capacity for the actual unique content.

---

## What Layer 12 Adds

Layer 12 adds a `RadixCache` — a compressed radix tree keyed on page-aligned token sequences — that maps shared prompt prefixes to their already-computed KV page indices. The tree is maintained on the CPU and stores one entry per unique cached prefix.

When a new request arrives with a 1024-token prompt whose first 512 tokens match a cached entry, `match_prefix` returns the 32 cached page indices and `matched_len = 512`. The scheduler's `PrefillAdder` calls `_apply_prefix_match` before setting `fill_ids`, advancing `kv_committed_len` to 512 and setting `fill_ids = input_ids[512:]`. The `prefill_batch` call for this request processes only 512 tokens — the unique suffix — and writes only 32 new pool pages. The 32 cached pages are injected directly into `req_to_token_pool` before the Triton kernel runs.

The extend kernel sees a 1024-position `kv_indptr` (32 cached + 32 new pages) and a 512-position `qo_indptr` (the suffix only). FlashInfer attends the suffix queries over the full 1024-position KV history — without ever recomputing the system prompt's K/V.

---

## What Changes in Layer 12

A single new file — `radix_cache.py` — contains the `RadixCache` class, the `TreeNode` dataclass with `lock_ref` and `last_access_time` fields, `match_prefix`, `insert`, `evict` (LRU via a min-heap), and `cache_finished_req`.

`scheduler.py` gains `_apply_prefix_match` in `PrefillAdder` and calls `cache_finished_req` on request completion instead of `kv_pool.free` directly.

`model_runner.prefill_batch` gains a step 2.5 that writes prefix page indices into `req_to_token_pool` for requests with a cache hit, and an eviction guard that calls `radix_cache.evict` when the pool is low before allocating new pages.

`decode_step`, `forward_batch.py`, the attention backend, and all model files are unchanged. The extend kernel does not know or care whether some of the pages in `kv_indices` were computed by a prior request — it reads them the same way regardless.

---

## The Pattern

Each layer in this series adds one mechanism, touches one or two files, and the benchmark measures exactly the improvement that mechanism was designed to produce:

- Layer 9: `ReqToTokenPool` + Triton kernel → O(B) per-step overhead, not O(Σ kv_tokens).
- Layer 11: `PrefillAdder` + `chunked_prefill_size` → bounded starvation window.
- Layer 12: `RadixCache` + `match_prefix` → shared-prefix KV reuse.
- Layer 13: `GPTQLinear` + `gptq_gemm` → 4× weight memory reduction.
- Layer 14: `SpecRunner` + draft/target + accept/reject → multiple committed tokens per target forward pass.

The KV pool layout, the `ForwardBatch` dispatch, and the FlashInfer API are unchanged from Layer 9 through Layer 14. Understanding one layer's implementation is understanding the skeleton on which all subsequent layers build.
