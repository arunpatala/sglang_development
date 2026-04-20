# Layer 11 — Lesson Outline

## What This Lesson Covers

Layer 10 added `page_size=16` to the KV pool and introduced `ReqToTokenPool` to map request slots to page indices. `prefill(req)` still processed one request at a time — every call blocked until the full prompt was written into the pool. Long prompts therefore blocked the decode loop entirely until they finished: shorter requests waiting in `_running` received no decode steps, accumulating unbounded output latency ("decode starvation"). A 4096-token prompt forced 4096 token positions through one sequential `F.sdpa` call before the first decode step could execute.

Layer 11 fixes this with chunked prefill. `prefill(req)` is replaced by `prefill_batch(reqs)`: a single `BatchPrefillWithPagedKVCacheWrapper` call that processes a batch of requests — or a slice of one long request — in one GPU kernel. A new `chunked_prefill_size` parameter in `Scheduler` caps how many tokens any single request contributes per round; long prompts are split across multiple scheduler iterations. Between each chunk, the decode batch continues to run. The scheduler tracks exactly one in-flight chunked request via `self._chunked_req`, clearing it once the final chunk is processed.

The changes touch three files: `scheduler.py` (which adds the `_chunked_req` state machine and `PrefillAdder`), `model_runner.py` (which replaces `prefill` with `prefill_batch` using FlashInfer's extend wrapper and adds `WriteInfo` / `compute_write_info` for page-packing arithmetic), and `kv_cache.py` (which adds `ExtendKVCtx`, `ReqToTokenPool`, `WriteInfo`, and `compute_write_info`). `forward_batch.py` gains a third mode: `EXTEND` replaces the old `PREFILL` mode. `server.py`, `tokenizer.py`, `request.py`, and the model weights are unchanged.

The sections follow the new code top to bottom: the structural shift from single-request prefill to batched extend, the scheduler's chunked state machine, the `PrefillAdder` budget logic, the `WriteInfo` page-packing arithmetic, the `prefill_batch` call in `model_runner`, the attention backend changes (`EXTEND` mode replacing `PREFILL`), the full call trace, and what chunked prefill leaves for Layer 12 (prefix caching).

---

## Sections

### 01 — From Prefill to Extend (`01_from_prefill_to_extend.md`)
- Layer 10's `prefill(req)`: B=1, F.sdpa, one request at a time, blocking the decode loop for the full prompt length; `req.fill_ids = req.input_ids` set unconditionally
- Layer 11's `prefill_batch(reqs)`: packs N requests (or N slices) into one `[1, total_tokens]` input; processes all in a single FlashInfer `BatchPrefillWithPagedKVCacheWrapper` forward pass
- What drives the change: decode starvation — while `prefill(req)` runs for a 4096-token prompt, all decode requests wait idle; chunked prefill interleaves extend and decode on every scheduler tick
- What stays the same: page allocation via `kv_pool.alloc`, `ReqToTokenPool` slot mapping, sampling of the last token, `decode_step` unchanged

### 02 — The Scheduler's Chunked State Machine (`02_the_chunked_state_machine.md`)
- `Scheduler._chunked_req: Optional[Req]` — the one request that is mid-prefill; `None` means the scheduler can pick new requests from waiting
- Why only one in-flight chunked request: a single `chunked_req` serialises chunk boundaries cleanly and avoids budget contention between two partially-filled requests
- `PrefillAdder.build()` case 1: `_chunked_req is not None` → slice `req.fill_ids = req.input_ids[kv_committed_len : kv_committed_len + chunked_prefill_size]`, return immediately without looking at waiting queue
- `PrefillAdder.build()` case 2: peek waiting queue → if a request's `prompt_len > chunked_prefill_size`, take its first chunk, record as `new_chunked_req`, stop; otherwise drain up to `max_prefill_tokens` total
- State transitions: `PREFILLING` while chunks remain, `RUNNING` or `FINISHED` after last chunk, `_chunked_req = None` cleared in main `Scheduler.run` loop

### 03 — PrefillAdder and the Token Budget (`03_prefill_adder_budget.md`)
- `PrefillAdder`: a temporary helper object built at the start of each scheduler iteration from `(waiting, running_count, max_running_reqs, max_prefill_tokens, chunked_prefill_size, chunked_req)`
- `max_prefill_tokens` caps the total extend tokens across all new (non-chunked) requests per round; prevents the extend kernel from growing too large relative to the decode batch
- The two stopping conditions in case 2: `running_count + len(batch) >= max_running_reqs` (decode batch full) and `rem_tokens <= 0 and batch` (budget exhausted)
- Why `chunked_prefill_size and req.prompt_len > chunked_prefill_size` triggers the chunk path: a prompt larger than the chunk cap must not block the budget; first chunk occupies `chunked_prefill_size` tokens and the request leaves the waiting queue immediately
- `new_chunked_req` vs `_chunked_req`: `new_chunked_req` is set during `build()`, then copied to `_chunked_req` by `Scheduler.run` after `prefill_batch` returns

### 04 — WriteInfo and Page Packing (`04_write_info_page_packing.md`)
- The partial-page problem: when `kv_committed_len % page_size != 0`, the last committed page is only partially filled; the next chunk must continue writing into that existing page before allocating new ones
- `WriteInfo(new_pages, n_leftover, n_new_tokens_in_existing, slot_indices_ref)` — computed by `compute_write_info` once per request before the extend kernel runs
- `n_leftover = kv_committed_len % page_size`: tokens already occupying the current last page; `n_new_tokens_in_existing = min(page_size - n_leftover, extend_input_len)` fills the rest of that page
- New pages needed: `ceil((extend_input_len - n_new_tokens_in_existing) / page_size)` — allocated from `kv_pool.alloc` and appended to `req.slot_indices`
- `req_to_token` updated immediately by `compute_write_info` so the Triton kernel can read page→token mapping on GPU without a separate CPU→GPU copy step

### 05 — prefill_batch: The Extend Kernel (`05_prefill_batch.md`)
- Input packing: `all_ids = [fill_ids for req in reqs]` concatenated into `[1, total_tokens]`; position IDs start at `req.kv_committed_len` per request (not at 0)
- `qo_indptr [B+1]`: token boundary per request; `kv_indptr [B+1]`: page boundary per request (cumsum of `len(req.slot_indices)` after `compute_write_info`)
- `kv_last_page_lens [B]`: actual fill count of last page = `total_committed % page_size` or `page_size` if aligned
- Triton kernel `create_flashinfer_kv_indices_triton`: reads `req_to_token_pool` on-GPU and writes `kv_indices` without staging through CPU
- `begin_forward(qo_indptr, kv_indptr, kv_indices, kv_last_page_lens, ..., causal=True)` then `ExtendKVCtx` → forward → `end_forward`; for non-last-chunk requests `req.status = PREFILLING`; for last-chunk requests: sample first output token

### 06 — The Attention Backend: EXTEND Mode (`06_attention_extend_mode.md`)
- `forward_batch.py` gains `ForwardMode.EXTEND` (replaces `PREFILL`) and `ForwardMode.NOCACHE` (formerly the no-cache fallback)
- `PagedExtendBackend._extend_forward`: `kv_cache.store(layer_idx, k, v)` writes new tokens to pool via `WriteInfo`; Q reshaped to `[total_tokens, n_heads, head_dim]`; `extend_wrapper.forward(q_fi, (k_paged, v_paged), causal=True)` runs paged prefill; output reshaped back to `[1, n_heads, total_tokens, head_dim]`
- Why EXTEND subsumes both first-chunk and continuation: `kv_committed_len` carries the position information; the extend kernel attends over all pages (cached + new) correctly for any `kv_committed_len`
- Layer 10's `PrefillKVCtx` (F.sdpa, B=1 only) is gone; `ExtendKVCtx` handles all cases uniformly with paged prefill
- `model/qwen3.py` now builds `ForwardBatch` internally from kv_cache type detection; `decoder_layer.py` takes `forward_batch` instead of `(attention_mask, kv_cache)`

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace: one long request (prompt_len > chunked_prefill_size) alongside two running decode requests
- Step 1 — First chunk: `PrefillAdder` detects long prompt, takes tokens `[0:C]`, sets `new_chunked_req`; `prefill_batch([chunked_req])` runs; `req.status = PREFILLING`; `_chunked_req = req`; decode batch also runs
- Step 2 — Continuation chunks: scheduler routes to case 1; slices `[C:2C]`, `[2C:3C]`, etc.; each round the decode batch runs in the same scheduler iteration
- Step 3 — Last chunk: `req.is_last_chunk` → `prefill_batch` samples first token; `req.status = RUNNING`; `_chunked_req = None`; request joins decode batch
- Step 4 — Steady-state decode: `decode_step` processes all `_running` requests; `end_forward`; slot append; finished requests → `kv_pool.free` + `req_to_token_pool.free`

### 08 — What Comes Next (`08_whats_next.md`)
- The remaining waste: identical prefixes generate duplicate KV writes; two requests with the same system prompt both compute and store the same 512 tokens of KV data independently
- Layer 12 (prefix caching): `RadixCache` stores a compressed trie of prompt token sequences keyed to KV page indices; `match_prefix` returns the deepest matching node and its pages before prefill; `kv_committed_len` is set to `prefix_len` so `prefill_batch` only processes the unique suffix
- What changes in Layer 12: `radix_cache.py` (new), `model_runner.prefill_batch` (prefix page injection), `scheduler.py` (`PrefillAdder._apply_prefix_match`); `decode_step`, `spec_runner.py`, `forward_batch.py`, model weights unchanged

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 11 concepts to SGLang source: `prefill_batch` → `ModelRunner.forward_batch_fill` in `srt/model_executor/model_runner.py`; `_chunked_req` → `scheduler.py`'s `chunked_req`; `PrefillAdder` → `PrefillAdder` in `srt/managers/scheduler.py`; `compute_write_info` → `forward_info_fill` preparation logic

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `chunked_prefill_size` parameter | `scheduler.py` line 192: `chunked_prefill_size: int = 0` |
| `_chunked_req` state variable | `scheduler.py` line 205: `self._chunked_req: Optional[Req] = None` |
| `PrefillAdder` class | `scheduler.py` line 96: `class PrefillAdder:` |
| Case 1: continue in-flight chunk | `scheduler.py` line 130: `if self.chunked_req is not None:` |
| Case 2: new request chunked path | `scheduler.py` line 159: `if self.chunked_prefill_size and req.prompt_len > self.chunked_prefill_size:` |
| `ReqToTokenPool` class | `kv_cache.py` line 78: `class ReqToTokenPool:` |
| `KVPool` class (page-sized) | `kv_cache.py` line 110: `class KVPool:` |
| `WriteInfo` dataclass | `kv_cache.py` line 177: `class WriteInfo:` |
| `ExtendKVCtx` class | `kv_cache.py` line 196: `class ExtendKVCtx:` |
| `DecodeKVCtx` class | `kv_cache.py` line 286: `class DecodeKVCtx:` |
| `compute_write_info` | `kv_cache.py` line 321: `def compute_write_info(` |
| `prefill_batch` entry point | `model_runner.py` line 160: `def prefill_batch(self, reqs: List[Req]) -> None:` |
| Prefix-page injection (layer 12 preview) | `model_runner.py` line 176: alloc `req_pool_idx` for new requests |
| Triton kv_indices kernel | `model_runner.py` line 240: `create_flashinfer_kv_indices_triton[(B,)](` |
| `begin_forward` extend wrapper | `model_runner.py` line 251: `self._extend_wrapper.begin_forward(` |
| `end_forward` and PREFILLING branch | `model_runner.py` line 283: `self._extend_wrapper.end_forward()` |
| `ForwardMode.EXTEND` | `forward_batch.py` line 42: `EXTEND = auto()` |
| `PagedExtendBackend` | `model/backend.py` line 67: `class PagedExtendBackend:` |
| `_extend_forward` implementation | `model/backend.py` line 86: `def _extend_forward(` |
| `chunked_prefill_size` in config | `config.yml` — `chunked_prefill_size: 512` |
