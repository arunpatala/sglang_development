# Layer 9 — Summary

Layer 9 replaces the per-step Python slot-list assembly and CPU-to-GPU index transfer with a GPU-resident `ReqToTokenPool` table and a Triton kernel that builds `kv_indices` entirely on-device, and simultaneously raises `page_size` from 1 to 16, reducing index length, pool-row lookups, and page allocation frequency by 16×. The scheduler, server, request dataclass, tokenizer, `forward_batch.py`, and all model layers are unchanged; the `ForwardBatch` / `PagedBackend` dispatch pattern introduced in Layer 8 carries forward without modification.

---

## From Layer 8 to Layer 9

In Layer 8, every decode step assembled `kv_indices` in Python:

```python
# Layer 8 — decode_step (key lines)
new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

kv_indices_list: List[int] = []
for i, req in enumerate(reqs):
    kv_indices_list.extend(req.slot_indices)   # historical slots (O(kv_len) per req)
    kv_indices_list.append(new_slots[i])        # new token's slot

kv_indptr = torch.tensor(kv_indptr_list, dtype=torch.int32, device=DEVICE)
kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)   # CPU→GPU
kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)             # always 1

decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(self._workspace, "NHD")
decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens,
                             ..., page_size=1, ...)
```

For a B=16 batch with 500-token average history, this loop iterates 8000 times in Python and sends an 8000-element int32 array to the GPU on every decode step. With `page_size=1`, `kv_last_page_lens` is always ones, and FlashInfer performs 8000 pool-row lookups per layer.

In Layer 9, the same decode step reads:

```python
# Layer 9 — decode_step (key lines)
seq_lens_list      = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]
token_offsets_list = [sl % P for sl in seq_lens_list]

# Conditional page alloc: O(B), not O(Σ tokens)
for i, req in enumerate(reqs):
    if token_offsets_list[i] == 0:          # last page full → new page
        new_page = self.kv_pool.alloc(1)[0]
        req.slot_indices.append(new_page)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, num_pages_list[i]] = new_page
    # else: reuse last page — no alloc, no table write

kv_last_page_lens = token_offsets_t + 1    # range 1..P (variable)

torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])   # GPU, pre-allocated
kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)

create_flashinfer_kv_indices_triton[(B,)](
    self.req_to_token_pool.req_to_token,
    req_pool_idx_t, num_pages_t, kv_indptr, None, kv_indices,
    self.req_to_token_pool.req_to_token.shape[1],
)

self._decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens,
                                   ..., P, ...)   # P=16, not 1
```

The Python loop is now O(B) — one iteration per request regardless of history length. `kv_indices` has `ceil(total_tokens / 16)` entries instead of `total_tokens`. The CPU sends only `~5B` small int tensors to the GPU each step; the historical page data never crosses the PCIe bus again after prefill.

---

## ReqToTokenPool

`ReqToTokenPool` is a single `[max_batch, max_pages_per_req]` int32 tensor that lives permanently on GPU. Its `(row, col)` entry holds the physical page index for row `req_pool_idx` and page position `col`:

```python
# kv_cache.py — ReqToTokenPool
class ReqToTokenPool:
    def __init__(self, max_batch: int, max_context_len: int) -> None:
        self.req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device=DEVICE
        )
        self.free_slots: List[int] = list(range(max_batch))
```

`max_context_len` is `ceil(max_token_context / page_size)` — with `page_size=16` and a 4096-token limit this is 256 columns, compared to 4096 columns for a token-indexed table. A 128-request pool at 256 columns costs `128 × 256 × 4 = 131 KB` — negligible on GPU. `alloc()` pops a free row index; `free(idx)` pushes it back; both are O(1) Python list operations on the scheduler's CPU thread.

At prefill, `model_runner.prefill` writes all page indices in one GPU slice assignment:

```python
pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t
```

At decode, a new row entry is written only when the last page fills:

```python
if token_offsets_list[i] == 0:
    self.req_to_token_pool.req_to_token[
        req.req_pool_idx, num_pages_list[i]
    ] = new_page   # one scalar int32 write
```

On the 15 decode steps where `token_offset != 0`, the table does not change at all. The Triton kernel reads whichever rows it needs directly from `req_to_token` on the GPU — there is no "export to CPU, build list, copy back" cycle.

---

## Paged KVPool and PrefillKVCtx

The KVPool tensor shape changes to accommodate the page dimension natively:

```python
# kv_cache.py — KVPool (one tensor per layer)
self.k_pool = [
    torch.zeros(total_pages, page_size, n_kv_heads, head_dim,
                dtype=dtype, device=DEVICE)
    for _ in range(n_layers)
]
```

Layer 8's shape was `[total_slots, n_kv_heads, head_dim]`. Layer 9's `[total_pages, page_size, n_kv_heads, head_dim]` is what FlashInfer's paged kernel expects natively — the `unsqueeze(1)` that Layer 8 needed to insert a synthetic page-size=1 dimension is gone from `backend.py`. `KVPool.alloc(n_tokens)` now returns `ceil(n_tokens / page_size)` page indices instead of `n_tokens` slot indices. Slot 0 remains a zero-filled padding dummy for FlashInfer.

`PrefillKVCtx.store()` must pack the prompt's K/V into page-aligned chunks before writing:

```python
def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    # k: [1, n_kv_heads, prompt_len, head_dim]
    P   = self._kv_pool.page_size
    L   = k.shape[2]

    k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()   # [L, n_kv, D]
    pad   = self._n_pages * P - L
    if pad > 0:
        k_nhd = F.pad(k_nhd, (0, 0, 0, 0, 0, pad))       # [n_pages*P, n_kv, D]

    k_paged = k_nhd.view(self._n_pages, P, n_kv, D)      # [n_pages, P, n_kv, D]
    self._kv_pool.k_pool[layer_idx][self._page_t] = k_paged
```

The padding zeros in the last page's unused positions are never read by FlashInfer because `kv_last_page_lens` communicates the true fill level. `DecodeKVCtx.store()` replaces Layer 8's single-index `k_pool[layer][new_slots]` with a 2D advanced index:

```python
def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
    # k_fi: [B, n_kv_heads, head_dim]
    self.k_pool[layer_idx][self.last_page_indices, self.token_offsets] = k_fi
```

`last_page_indices [B]` identifies which page each request's new token lands in; `token_offsets [B]` identifies the position within that page. Both are set in `model_runner.decode_step` before `DecodeKVCtx` is constructed.

---

## The Decode Step: Conditional Allocation and GPU Indexing

The central formula in `decode_step` is the `seq_len` computation:

```python
seq_lens_list = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]
```

`r.output_ids[-1]` is the current input token — the one generated in the previous step and not yet stored in the KV cache. Subtracting 1 gives the number of tokens already in the pool, which equals the RoPE position of the token being processed this step. Without the `-1`, `token_offset` and `pos_ids` are both off by one, causing every write to land at the wrong within-page position.

`token_offset = seq_len % page_size` drives two decisions. First, if `token_offset == 0`, the current last page is exactly full and a new page must be allocated before the forward pass. A single scalar is written into `req_to_token`. For `page_size=16` this fires once every 16 steps per request instead of every step. Second, `token_offset` is the index passed to `DecodeKVCtx` for the 2D pool write.

`kv_indptr` is produced by `torch.cumsum(num_pages_t, ...)` writing into a pre-allocated `[max_batch+1]` buffer. The output stays on-GPU — no host buffer, no `torch.tensor([...], device=DEVICE)` call. `kv_last_page_lens = token_offsets_t + 1` replaces the all-ones constant from Layer 8; FlashInfer now knows exactly how many valid tokens the last page of each request contains.

---

## The Triton Kernel

`create_flashinfer_kv_indices_triton` writes the flat `kv_indices` tensor from the `req_to_token` table entirely on-device:

```python
# triton_utils.py
@triton.jit
def create_flashinfer_kv_indices_triton(
    req_to_token_ptr,       # [max_batch, max_pages_per_req]  int32
    req_pool_indices_ptr,   # [B]  which row per request
    page_kernel_lens_ptr,   # [B]  how many pages per request
    kv_indptr,              # [B+1]  cumulative page offsets
    kv_start_idx,           # None (sliding-window, unused here)
    kv_indices_ptr,         # [Σ num_pages]  output
    req_to_token_ptr_stride: tl.constexpr,
):
    BLOCK_SIZE: tl.constexpr = 512
    pid = tl.program_id(axis=0)   # one program per request
    ...
    num_loop = tl.cdiv(kv_end - kv_start, BLOCK_SIZE)
    for i in range(num_loop):
        offset = tl.arange(0, BLOCK_SIZE) + i * BLOCK_SIZE
        data   = tl.load(req_to_token_ptr + req_pool_index * stride + offset, mask=...)
        tl.store(kv_indices_ptr + kv_indices_offset + offset, data, mask=...)
```

The kernel is launched with `grid=(B,)` — one threadblock per request. Each block loads its row from `req_to_token` in 512-element chunks and stores consecutive entries into the appropriate segment of `kv_indices` (delimited by `kv_indptr`). All B blocks execute in parallel. The work is purely memory-bound: reading `n_pages` int32 values per request and writing them contiguously. With `page_size=16`, a 500-token history requires 32 reads and 32 writes per request, versus 500 in Layer 8.

The call site passes `req_pool_idx_t [B]` (which `req_to_token` row) and `num_pages_t [B]` (how many columns to read). Everything else — the actual page indices — already lives in `req_to_token` on-device. The CPU-to-GPU transfer at each decode step is exactly these two `[B]` tensors plus `seq_lens_t`, `token_offsets_t`, and `last_page_idx_t` — O(B) integers, independent of KV history length.

---

## The Full Loop

The scheduler and server are identical to Layer 8.

When a new request arrives, `prefill(req)` allocates `ceil(prompt_len / page_size)` pages from `KVPool`, stores the page indices on `req.slot_indices`, and writes them in one slice to `req_to_token[req_pool_idx, 0:n_pages]`. The `PrefillKVCtx` pads the prompt K/V to a multiple of `page_size`, reshapes into `[n_pages, P, n_kv, dim]`, and scatter-writes all pages at once. The first output token is sampled. From this point forward, the full KV history is on-device and never needs to leave.

On each scheduler iteration, `decode_step(_running)` begins with the corrected `seq_len` formula and `token_offset` check. For requests whose last page is full, one new page is allocated from `KVPool` and its index written as a scalar to `req_to_token`. This write is the only change to the GPU table until that page fills again — 15 steps later. `torch.cumsum` produces `kv_indptr` in the pre-allocated buffer; the Triton kernel runs B threadblocks in parallel, reads each request's row from `req_to_token`, and fills `kv_indices` — all on-device, sub-millisecond.

`_decode_wrapper.begin_forward(kv_indptr, kv_indices, kv_last_page_lens, ..., P)` plans the FlashInfer paged decode kernel with the real `page_size=P`. The `DecodeKVCtx` carries `last_page_indices [B]` and `token_offsets [B]` into the forward pass. In each of the 28 attention layers, `PagedBackend._decode_forward` calls `ctx.store(layer_idx, k_fi, v_fi)` — a single 2D indexed write placing each request's new token K/V at `(last_page, token_offset)` — then calls `wrapper.forward(q_fi, (k_pool[layer], v_pool[layer]))`. FlashInfer reads the full KV history for each request via `kv_indices`, producing attended output without any intermediate buffer. No K/V float data has moved across the PCIe bus since prefill.

After `end_forward()`, `sample_token` draws the next token for each request. Finished requests call `kv_pool.free(req.slot_indices)` — freeing all pages at once, in O(n_pages) not O(n_tokens) — and `req_to_token_pool.free(req_pool_idx)`, making the row immediately available to the next request.

---

## What Comes Next

Layer 9 reduces the per-step overhead from O(Σ kv_tokens) to O(B) in the index-building phase, but the scheduling policy remains first-come-first-served with no ability to interrupt a long prefill or share computation between requests with common prefixes. The natural next improvements are chunked prefill (splitting long prompts into fixed-size chunks so decode batches are not starved during expensive prefills), prefix caching (hashing and reusing KV pages for system prompts shared across requests, eliminating redundant prefill compute entirely), and speculative decoding (a small draft model proposes multiple tokens per step that the full model verifies in one forward pass). All three operate on top of the same `KVPool` / `ReqToTokenPool` / `ForwardBatch` foundation that Layer 9 establishes.
