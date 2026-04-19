# Layer 9 — Lesson Outline

## What This Lesson Covers

Layer 8 eliminated the per-step float gather by replacing `PerReqKVCache` with a pre-allocated `KVPool` and building a `kv_indices` integer array every decode step. Two inefficiencies remained. First, `kv_indices` was assembled in Python by iterating over each request's slot list and appending the new slot — an `O(Σ kv_tokens)` Python loop followed by a `torch.tensor(...)` call that transferred the entire array to the GPU. Second, `page_size=1` meant the pool had one row per token: a 1000-token batch history generated 1000 index entries, 1000 pool-row lookups per layer, and 1000 page allocations over the lifetime of that request.

Layer 9 fixes both in a single step. It introduces `ReqToTokenPool`, a GPU-resident `[max_batch, max_pages_per_req]` int32 table that stores page indices on-device from the moment of prefill, and a Triton kernel (`create_flashinfer_kv_indices_triton`) that reads that table and writes `kv_indices` entirely on-GPU with no Python iteration and no CPU-to-GPU copy. It simultaneously raises `page_size` to 16, reducing the `kv_indices` length, page alloc frequency, and index lookup count by 16×. The KV pool shape changes from `[total_slots, n_kv, dim]` to `[total_pages, page_size, n_kv, dim]`, prefill pads the prompt into aligned pages, and the decode step checks `token_offset % page_size` to allocate a new page only when the current last page is full.

The changes touch `kv_cache.py` (new `ReqToTokenPool`, new `KVPool` shape, paged `PrefillKVCtx.store`, 2D-indexed `DecodeKVCtx.store`), `model_runner.py` (page arithmetic, GPU cumsum, Triton call, pre-allocated buffers, decode wrapper reused from `__init__`), and `triton_utils.py` (new file with the Triton kernel). `forward_batch.py`, `model/backend.py`, `model/attention.py`, `model/decoder_layer.py`, `model/qwen3.py`, `scheduler.py`, `server.py`, `request.py`, and the tokenizer are unchanged or carry only docstring updates.

---

## Sections

### 01 — From Page-size-1 to Paged (`01_from_page_size_1_to_paged.md`)
- Layer 8's `decode_step` in full: Python slot-list iteration, `torch.tensor(list)` CPU→GPU transfer, `itertools.accumulate` on CPU, `decode_wrapper` created every step; `page_size=1` so `kv_last_page_lens` is always ones
- Layer 9's `decode_step` in full: `seq_lens - 1` formula, conditional page alloc, GPU `cumsum`, Triton kernel call, `decode_wrapper` reused from `__init__`, `kv_last_page_lens = token_offsets_t + 1`
- Quantifying the reduction: a B=16 batch with 500-token average history — Layer 8 sends 8000 integers to GPU each step; Layer 9 sends 16 (one seq_len per request), the table already lives on-device
- What is unchanged: `prefill()` still B=1; the `ForwardBatch` / `PagedBackend` pattern from Layer 8 is identical; sampling and request lifecycle identical

### 02 — ReqToTokenPool (`02_req_to_token_pool.md`)
- Layout: `[max_batch, max_pages_per_req]` int32 on GPU; `max_pages_per_req = ceil(max_token_context / page_size)` — 16× smaller columns vs a token-indexed table
- Row lifecycle: `rtp.alloc()` pops a free row index at prefill; `rtp.free(idx)` returns it at request completion; both are `O(1)` Python list operations
- Prefill write: `req_to_token[req_pool_idx, 0:n_pages] = pages_t` — one GPU slice assignment, `n_pages` int32 values, no per-token writes
- Decode write (conditional): when `token_offset == 0`, `req_to_token[req_pool_idx, num_pages] = new_page` — one scalar int32 write; otherwise, no write at all (the page index is already in the table)
- Why this enables the Triton kernel: all page data lives on-device; the kernel reads rows without any CPU-to-GPU transfer of historical slot data

### 03 — Paged KVPool and PrefillKVCtx (`03_paged_kv_pool_and_prefill.md`)
- Pool shape change: `[total_pages, page_size, n_kv_heads, head_dim]` vs Layer 8's `[total_slots, n_kv_heads, head_dim]`; the page_size dim is now native — FlashInfer reads it directly, no `unsqueeze(1)` needed
- `KVPool.alloc(n_tokens)`: returns `ceil(n_tokens / page_size)` page indices; `KVPool.free(page_indices)` returns them; slot 0 reserved as FlashInfer padding dummy
- `PrefillKVCtx.store()` four-step transform: `[1, n_kv, L, D]` → permute → `[L, n_kv, D]` → pad to `n_pages × P` → view → `[n_pages, P, n_kv, D]` → scatter-write `k_pool[layer][page_t] = k_paged`
- Why padding is safe: `kv_last_page_lens` tells FlashInfer the true fill level; zero-padded positions in the last page are never read
- `DecodeKVCtx.store()`: `k_pool[layer][last_page_indices, token_offsets]` — 2D advanced indexing writes one new-token row per request at the exact (page, within-page) position

### 04 — The Decode Step: Conditional Allocation and GPU Indexing (`04_decode_step_gpu_indexing.md`)
- `seq_len` formula: `len(r.input_ids) + len(r.output_ids) - 1` = tokens already in cache = position of current input token; the `-1` is required because `output_ids[-1]` is the current input, not yet stored
- `token_offset = seq_len % P`: the slot within the last page where the new token lands; drives both the conditional alloc and the `DecodeKVCtx` 2D write
- Conditional page alloc: `if token_offset == 0` allocates one new page, appends to `req.slot_indices`, writes a single scalar to `req_to_token`; otherwise the last page still has room and no alloc or table update is needed
- GPU tensors (O(B) transfer): `seq_lens_t`, `token_offsets_t`, `num_pages_t`, `req_pool_idx_t`, `last_page_idx_t` — all `[B]` int32; total ~5B integers per step vs O(Σ tokens) in Layer 8
- `kv_indptr` via `torch.cumsum(num_pages_t, ...)` into a pre-allocated buffer — remains on-device, no Python accumulate
- `kv_last_page_lens = token_offsets_t + 1`: range 1..P; replaces Layer 8's all-ones constant

### 05 — The Triton Kernel (`05_the_triton_kernel.md`)
- Why the kernel exists: `kv_indices` must be a flat `[Σ num_pages]` int32 tensor; reading it from `req_to_token` rows in Python would be `O(Σ num_pages)` Python ops + one CPU→GPU copy per step
- Kernel launch: `create_flashinfer_kv_indices_triton[(B,)](req_to_token, req_pool_idx_t, num_pages_t, kv_indptr, None, kv_indices, stride)` — grid = B threadblocks, one per request, all run in parallel
- Per-threadblock logic: load `req_pool_index = req_pool_indices[pid]`, load `kv_len = num_pages[pid]`, read `req_to_token[req_pool_index, 0:kv_len]` in 512-element BLOCK_SIZE chunks, write to `kv_indices[kv_indptr[pid]:]`
- What moves to GPU per step vs Layer 8: Layer 8 sent the full slot list (O(Σ kv_tokens) ints); Layer 9 sends B row indices + B page counts (~2B small ints); the kernel reads page data directly from on-device memory
- `kv_start_idx` parameter: always `None` here (used in SGLang for sliding-window attention); the kernel path degenerates to a straight row-copy when `None`

### 06 — The Full Loop (`06_the_full_loop.md`)
- End-to-end trace with B=2 concurrent requests, naming every component in execution order
- Step 1 — Prefill: `kv_pool.alloc(prompt_len)` returns `ceil(L/P)` page indices; `PrefillKVCtx` pads and scatter-writes all pages; `req_to_token[rpi, 0:n_pages] = pages_t`; first token sampled; `req.slot_indices` holds page index list
- Step 2 — `decode_step` entry: `seq_lens - 1` formula; conditional page alloc fires only when `token_offset == 0`; O(B) int tensors built and transferred
- Step 3 — GPU index build: `cumsum` produces `kv_indptr` in the pre-allocated buffer; Triton kernel runs B threadblocks in parallel, reads `req_to_token` rows, writes `kv_indices`; `begin_forward` plans the FlashInfer kernel with page_size=P
- Step 4 — Forward pass: 28 attention layers each call `PagedBackend._decode_forward`: `DecodeKVCtx.store()` writes new K/V at `(last_page, token_offset)`, `wrapper.forward(q_fi, (k_pool[layer], v_pool[layer]))` reads history
- Step 5 — Cleanup: `end_forward`; sampling; `slot_indices` unchanged unless new page was appended in step 2; finished requests call `kv_pool.free(slot_indices)` and `req_to_token_pool.free(req_pool_idx)`

### 07 — What Comes Next (`07_whats_next.md`)
- What Layer 9 still leaves on the table: the Python loop over `reqs` to read `len(input_ids) + len(output_ids)` is O(B), unavoidable, but grows with batch size; scheduling policy is still FCFS with no priority or preemption
- Chunked prefill: long prompts block the decode batch for many steps; splitting prefill into fixed-size chunks interleaved with decode reduces first-token latency variance
- Prefix caching / RadixAttention: requests sharing a common system prompt recompute identical KV data; caching those pages by hash eliminates redundant prefill compute
- Speculative decoding: one draft model generates multiple candidate tokens per step; the target model verifies them in a single forward pass; throughput improves when the draft is usually correct
- What stays the same through all of these: the `KVPool` / `ReqToTokenPool` page allocation model; `ForwardBatch`; FlashInfer paged kernel; the `scheduler.py` / `server.py` asyncio bridge

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 9 concepts to SGLang source: `ReqToTokenPool` → `srt/mem_cache/memory_pool.py`; `create_flashinfer_kv_indices_triton` → `srt/layers/attention/utils.py`; `page_size` → `srt/managers/schedule_batch.py`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `PAGE_SIZE = 16` constant | `model_runner.py` line 99: `PAGE_SIZE = 16` |
| Pool sizing in pages | `model_runner.py` line 134: `max_pages = int(free_bytes * kv_memory_fraction / (page_size * bytes_per_token))` |
| `KVPool` constructor call | `model_runner.py` line 136: `KVPool(total_pages=max_pages, page_size=page_size, ...)` |
| `ReqToTokenPool` constructor | `model_runner.py` line 148: `ReqToTokenPool(max_batch=_MAX_CONCURRENT_REQS, max_context_len=max_pages_per_req)` |
| Pre-allocated `kv_indptr` buffer | `model_runner.py` line 155: `self._kv_indptr_buf = torch.zeros(_MAX_CONCURRENT_REQS + 1, ...)` |
| `_decode_wrapper` created once | `model_runner.py` line 168: `self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(...)` |
| `n_pages` in prefill | `model_runner.py` line 193: `n_pages = math.ceil(prompt_len / P)` |
| Prefill page alloc | `model_runner.py` line 195: `page_indices = self.kv_pool.alloc(prompt_len)` |
| `req_to_token` prefill write | `model_runner.py` line 201: `self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t` |
| `seq_len` formula (corrected) | `model_runner.py` line 260: `seq_lens_list = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]` |
| `token_offset` | `model_runner.py` line 261: `token_offsets_list = [sl % P for sl in seq_lens_list]` |
| Conditional page alloc | `model_runner.py` line 269: `if token_offsets_list[i] == 0:` |
| Scalar `req_to_token` write | `model_runner.py` line 273: `self.req_to_token_pool.req_to_token[req.req_pool_idx, num_pages_list[i]] = new_page` |
| `kv_last_page_lens` | `model_runner.py` line 290: `kv_last_page_lens = token_offsets_t + 1` |
| GPU `cumsum` for `kv_indptr` | `model_runner.py` line 294: `torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])` |
| Triton kernel launch | `model_runner.py` line 301: `create_flashinfer_kv_indices_triton[(B,)](...)` |
| `pos_ids` from `seq_lens_t` | `model_runner.py` line 312: `pos_ids = seq_lens_t.unsqueeze(1).to(torch.long)` |
| `begin_forward` with `P` | `model_runner.py` line 320: `self._decode_wrapper.begin_forward(..., P, ...)` |
| `DecodeKVCtx` with 2D indices | `model_runner.py` line 332: `DecodeKVCtx(wrapper=..., last_page_indices=last_page_idx_t, token_offsets=token_offsets_i64)` |
| `ReqToTokenPool` class | `kv_cache.py` line 74: `class ReqToTokenPool:` |
| `req_to_token` tensor alloc | `kv_cache.py` line 97: `self.req_to_token = torch.zeros((max_batch, max_context_len), dtype=torch.int32, ...)` |
| `KVPool` class | `kv_cache.py` line 126: `class KVPool:` |
| Pool tensor shape (paged) | `kv_cache.py` line ~143: `torch.zeros(total_pages, page_size, n_kv_heads, head_dim, ...)` |
| `KVPool.alloc` (pages) | `kv_cache.py` line 185: `def alloc(self, n_tokens: int) -> List[int]:` |
| `PrefillKVCtx.store` scatter | `kv_cache.py` line 252: `self._kv_pool.k_pool[layer_idx][self._page_t] = k_paged` |
| `DecodeKVCtx.store` 2D write | `kv_cache.py` line 293: `def store(self, layer_idx, k_fi, v_fi)` → `k_pool[layer][last_page_indices, token_offsets]` |
| Triton kernel definition | `triton_utils.py` line 46: `def create_flashinfer_kv_indices_triton(...)` |
| Kernel BLOCK_SIZE | `triton_utils.py` line 66: `BLOCK_SIZE: tl.constexpr = 512` |
