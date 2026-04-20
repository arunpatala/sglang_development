# 04 — WriteInfo and Page Packing

`compute_write_info` is the arithmetic engine of `prefill_batch`. It answers one question: given that a request has `kv_committed_len` tokens already in the pool, and `n_fill` new tokens are being added this round, which pages in the pool need to be written, and where does each token land within a page? The answer is a `WriteInfo` struct that `ExtendKVCtx.store()` uses to scatter the new K/V tensors into the correct pool locations.

---

## The Partial-Page Problem

In Layer 9, every prefill wrote a whole number of pages — `ceil(prompt_len / P)` pages, with the last page zero-padded to `P` tokens. Decode steps added at most one token per step at a specific within-page offset. There was never a case where new tokens had to be written into the middle of an already-partially-filled page.

Chunked prefill creates exactly this case. Suppose `page_size = 16` and a request's first chunk processed 24 tokens, writing `ceil(24/16) = 2` pages into the pool. The second page (positions 16–23) holds 8 real tokens; positions 8–15 are padding zeros. When the second chunk arrives with 20 more tokens, those tokens should continue filling the second page (positions 8–15) before allocating a new third page. If instead a new page were allocated immediately, the K/V pool would have an 8-position gap in the second page — positions 8–15 would hold zeros — and the attention kernel would attend over those zeros as if they were valid tokens.

`compute_write_info` detects and resolves this by computing `n_leftover = kv_committed_len % page_size`:

```python
# kv_cache.py — compute_write_info
P          = kv_pool.page_size
n_leftover = kv_committed_len % P   # tokens already occupying the last page

existing_page:  Optional[int] = None
existing_slots: int = 0

if n_leftover > 0 and slot_indices:
    space_in_last  = P - n_leftover
    existing_slots = min(space_in_last, n_fill)
    existing_page  = slot_indices[-1]

remaining = n_fill - existing_slots
new_pages: List[int] = []
if remaining > 0:
    new_pages = kv_pool.alloc(remaining)
    n_prev = len(slot_indices)
    slot_indices.extend(new_pages)
    pages_t = torch.tensor(new_pages, dtype=torch.int32, device=DEVICE)
    rtp.req_to_token[req_pool_idx, n_prev : n_prev + len(new_pages)] = pages_t

return WriteInfo(
    existing_page  = existing_page,
    n_leftover     = n_leftover,
    existing_slots = existing_slots,
    new_pages      = new_pages,
)
```

For `kv_committed_len = 24` and `n_fill = 20`: `n_leftover = 24 % 16 = 8`, so the last page has 8 tokens and 8 empty slots. `space_in_last = 8`. `existing_slots = min(8, 20) = 8` — the first 8 of the new tokens will complete the existing page. `remaining = 20 - 8 = 12` — 12 tokens need new pages: `ceil(12/16) = 1` new page is allocated. The result is `WriteInfo(existing_page=last_page_idx, n_leftover=8, existing_slots=8, new_pages=[new_page_idx])`.

---

## The WriteInfo Struct

```python
# kv_cache.py
class WriteInfo:
    existing_page:  Optional[int]   # pool page index to complete (None if n_leftover==0)
    n_leftover:     int             # tokens already in existing_page
    existing_slots: int             # tokens from this chunk that go into existing_page
    new_pages:      List[int]       # newly allocated page indices
```

`existing_page` is `None` when `kv_committed_len` is a multiple of `page_size` — the previous chunk ended cleanly on a page boundary, no partial page continuation is needed. `existing_slots` is the number of tokens from the current chunk that complete the existing page; when `n_leftover == 0`, this is zero. `new_pages` holds the freshly allocated page indices, whose entries in `req_to_token_pool` are updated in-place by `compute_write_info` before returning.

`compute_write_info` mutates `slot_indices` and `req_to_token_pool.req_to_token` as a side effect. This is intentional: both must reflect the complete page layout — old pages plus new pages — before the Triton `kv_indices` kernel runs. The kernel reads `req_to_token` from GPU memory and would produce wrong page addresses if the table were not updated before the kernel launch.

---

## How ExtendKVCtx.store Uses WriteInfo

`WriteInfo` is passed into `ExtendKVCtx` at construction time. Inside the forward pass, each attention layer calls `ctx.store(layer_idx, k, v)`. The store method uses `write_infos` to determine where each request's tokens land:

```python
# kv_cache.py — ExtendKVCtx.store (simplified for one request)
for i, wi in enumerate(self.write_infos):
    s, e = self.qo_indptr[i], self.qo_indptr[i + 1]
    k_i = k[:, :, s:e, :].squeeze(0).permute(1, 0, 2).contiguous()  # [L, n_kv, D]

    # 1. Fill the existing partial page
    if wi.existing_slots > 0:
        toks = k_i[:wi.existing_slots]   # [existing_slots, n_kv, D]
        pool[layer][wi.existing_page, wi.n_leftover : wi.n_leftover + wi.existing_slots] = toks

    # 2. Write overflow tokens into new pages (padded to page boundaries)
    if wi.new_pages:
        overflow = k_i[wi.existing_slots:]   # [remaining, n_kv, D]
        pad = len(wi.new_pages) * P - len(overflow)
        if pad > 0:
            overflow = F.pad(overflow, (0, 0, 0, 0, 0, pad))
        overflow_paged = overflow.view(len(wi.new_pages), P, n_kv, D)
        pool[layer][wi.new_pages] = overflow_paged
```

The two-part write preserves the pool's invariant: `slot_indices[j]` always holds the page containing tokens `j*P .. (j+1)*P - 1`. An existing partial page's already-written tokens are not touched; the new tokens fill the empty slots starting at `n_leftover`. Overflow tokens go into freshly allocated pages, padded to a full page with zeros in the tail. `kv_last_page_lens` tells FlashInfer the actual fill level of each request's last page, so the padding is never attended over.

---

## Why compute_write_info Runs on CPU

The page allocation (`kv_pool.alloc`) and `req_to_token` update happen on the CPU before the GPU kernel launches. This is the only feasible design: `kv_pool.alloc` modifies a Python list, and the allocated page indices need to be in `req_to_token` on the GPU before the Triton kernel reads them. Performing this step asynchronously would require GPU-side allocation, which is significantly more complex. For typical batch sizes and chunk counts, the CPU-side `compute_write_info` loop is microseconds — negligible relative to the GPU kernel launch and the extend forward pass.

Section 05 covers how `prefill_batch` assembles all the `WriteInfo` objects and translates them into the FlashInfer `begin_forward` arguments.
