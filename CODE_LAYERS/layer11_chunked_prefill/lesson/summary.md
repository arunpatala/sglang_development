# Layer 11 — Summary

Layer 11 replaces the blocking single-request `prefill(req)` with a batched `prefill_batch(reqs)` that can process multiple requests — or slices of one long request — in a single FlashInfer paged-prefill kernel, eliminating the decode starvation that occurs when long prompts monopolize the scheduler. The server API, `decode_step`, model weights, and `tokenizer.py` are unchanged.

---

## From Layer 9 to Layer 11

In Layer 9, each incoming request was prefilled individually, one request at a time:

```python
# Layer 9 — model_runner.prefill (B=1, blocks decode for full prompt length)
prompt_len = len(req.input_ids)
page_indices = self.kv_pool.alloc(prompt_len)
req.slot_indices = page_indices

ctx = PrefillKVCtx(page_indices, self.kv_pool)
fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
logits = self.model(ids, forward_batch=fb, position_ids=pos)

next_tok = sample_token(logits[0, -1], req.temperature)
req.output_ids.append(next_tok)
```

A 4096-token prompt held the scheduler loop for one complete 4096-position `F.sdpa` call. Every request already in `_running` was frozen for that entire duration.

In Layer 11, the same operation reads:

```python
# Layer 11 — model_runner.prefill_batch (batched extend, FlashInfer paged prefill)
write_infos = [compute_write_info(kv_pool, rtp, req.slot_indices,
                                  req.req_pool_idx, req.kv_committed_len,
                                  req.extend_input_len)
               for req in reqs]

all_ids = [tok for req in reqs for tok in req.fill_ids]
ids_t   = torch.tensor(all_ids, device=DEVICE).unsqueeze(0)   # [1, total_tokens]

self._extend_wrapper.begin_forward(
    qo_indptr_t, kv_indptr, kv_indices, kv_last_page_lens,
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, P,
    causal=True, q_data_type=DTYPE,
)
ctx    = ExtendKVCtx(wrapper=self._extend_wrapper, k_pool=..., v_pool=...,
                     qo_indptr=qo_indptr_list, write_infos=write_infos, page_size=P)
logits = self.model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
self._extend_wrapper.end_forward()
```

`total_tokens` is the sum of `extend_input_len` across all requests. When `chunked_prefill_size=512`, a 4096-token prompt contributes at most 512 tokens per round. The remaining 3584 tokens are processed in subsequent scheduler iterations, and `decode_step` runs between each chunk.

---

## The Scheduler's Chunked State Machine

`Scheduler` tracks at most one in-flight chunked request via `self._chunked_req`. On each scheduler iteration, it builds a `PrefillAdder` and calls `build()`:

```python
adder = PrefillAdder(
    waiting              = self._waiting,
    running_count        = len(self._running),
    max_running_reqs     = self.max_running_reqs,
    max_prefill_tokens   = self.max_prefill_tokens,
    chunked_prefill_size = self.chunked_prefill_size,
    chunked_req          = self._chunked_req,
)
prefill_batch = adder.build()
```

`build()` has two cases. Case 1 fires when `chunked_req is not None` — a request is mid-prefill and its next chunk is sliced and returned immediately without consulting the waiting queue:

```python
if self.chunked_req is not None:
    req   = self.chunked_req
    start = req.kv_committed_len
    size  = self.chunked_prefill_size or (req.prompt_len - start)
    end   = min(start + size, req.prompt_len)
    req.fill_ids         = req.input_ids[start:end]
    req.extend_input_len = end - start
    return [req]
```

Case 2 fires when `chunked_req is None`. New requests are drained from `_waiting` up to `max_prefill_tokens` total. If any request's `prompt_len > chunked_prefill_size`, it is chunked immediately — `new_chunked_req` is set, the first chunk is returned, and the loop stops without examining further waiting requests. Only one chunked request exists at a time.

After `prefill_batch` returns, the scheduler routes each request: `PREFILLING` requests stay in `_chunked_req`; `RUNNING` or `FINISHED` requests clear `_chunked_req` and join the decode batch or resolve.

---

## PrefillAdder and the Token Budget

`max_prefill_tokens` caps the total extend tokens contributed by all new (non-chunked) requests per round. This prevents the extend kernel from growing large enough to delay the decode step significantly. The loop in `build()` enforces two stopping conditions:

```python
while True:
    if self.running_count + len(batch) >= self.max_running_reqs:
        break
    if rem_tokens <= 0 and batch:
        break
    if self.waiting.empty():
        break

    req = self.waiting.queue[0]   # peek without removing

    if self.chunked_prefill_size and req.prompt_len > self.chunked_prefill_size:
        req = self.waiting.get_nowait()
        chunk_end = min(self.chunked_prefill_size, req.prompt_len)
        req.fill_ids         = req.input_ids[:chunk_end]
        req.extend_input_len = chunk_end
        self.new_chunked_req = req
        batch.append(req)
        break   # only one chunked request per round

    req = self.waiting.get_nowait()
    req.fill_ids         = req.input_ids
    req.extend_input_len = req.prompt_len
    batch.append(req)
    rem_tokens -= req.prompt_len
```

The first condition prevents the decode batch from growing beyond `max_running_reqs`. The second condition stops adding new requests once the token budget is spent, but only after at least one request is in the batch — a request larger than the remaining budget is deferred to the next round unless the batch is empty, in which case it is admitted regardless of its size.

---

## WriteInfo and Page Packing

Between chunks, `kv_committed_len` advances by `extend_input_len`. If the previous chunk's last page was not full — `kv_committed_len % page_size != 0` — the next chunk must continue writing into that partially-filled page before allocating new ones. `compute_write_info` performs this arithmetic:

```python
P          = kv_pool.page_size
n_leftover = kv_committed_len % P   # tokens already in last page

if n_leftover > 0 and slot_indices:
    space_in_last  = P - n_leftover
    existing_slots = min(space_in_last, n_fill)
    existing_page  = slot_indices[-1]

remaining = n_fill - existing_slots
if remaining > 0:
    new_pages = kv_pool.alloc(remaining)
    slot_indices.extend(new_pages)
    rtp.req_to_token[req_pool_idx, n_prev : n_prev + len(new_pages)] = pages_t

return WriteInfo(existing_page=existing_page, n_leftover=n_leftover,
                 existing_slots=existing_slots, new_pages=new_pages)
```

`compute_write_info` mutates `slot_indices` and `rtp.req_to_token` in-place so the Triton kernel that builds `kv_indices` can read the correct page layout from GPU memory without a separate CPU-side copy step. `WriteInfo` is then passed into `ExtendKVCtx` so `ctx.store()` knows where to write the new chunk's K/V tensors within the pool.

---

## prefill_batch: The Extend Kernel

All requests' `fill_ids` are concatenated into a single `[1, total_tokens]` tensor. Position IDs per token start at `req.kv_committed_len + j`, not at zero — the extend kernel needs to know each token's absolute position so the rotary embeddings are applied correctly:

```python
ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
pos_t = torch.tensor([req.kv_committed_len + j
                       for req in reqs
                       for j in range(req.extend_input_len)],
                     dtype=torch.long, device=DEVICE).unsqueeze(0)
```

The three FlashInfer arrays — `qo_indptr`, `kv_indptr`, `kv_indices` — describe the ragged layout. `qo_indptr` holds the token boundary per request; `kv_indptr` holds the page boundary per request (cumsum of `len(req.slot_indices)` after page allocation); `kv_indices` is filled on-GPU by a Triton kernel that reads `req_to_token_pool`:

```python
create_flashinfer_kv_indices_triton[(B,)](
    self.req_to_token_pool.req_to_token,
    req_pool_idx_t, num_pages_t, kv_indptr, None, kv_indices,
    self.req_to_token_pool.req_to_token.shape[1],
)
self._extend_wrapper.begin_forward(
    qo_indptr_t, kv_indptr, kv_indices, kv_last_page_lens,
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
    P, causal=True, q_data_type=DTYPE,
)
```

After `end_forward()`, requests whose `is_last_chunk` is false have their status set to `PREFILLING`; requests at their final chunk sample their first output token and transition to `RUNNING` or `FINISHED`.

---

## The Attention Backend: EXTEND Mode

`forward_batch.py` adds `ForwardMode.EXTEND` (replaces `PREFILL`). `PagedExtendBackend._extend_forward` in `model/backend.py` writes new K/V into the pool via `ctx.store()` then calls the extend wrapper:

```python
def _extend_forward(self, q, k, v, layer_idx, forward_batch):
    if forward_batch.kv_cache is not None:
        forward_batch.kv_cache.store(layer_idx, k, v)   # pool write

    B, n_q, T, D = q.shape
    q_fi = q.permute(0, 2, 1, 3).reshape(T, n_q, D)     # [T, n_q, D]
    k_paged = forward_batch.kv_cache.k_pool[layer_idx].unsqueeze(1)
    v_paged = forward_batch.kv_cache.v_pool[layer_idx].unsqueeze(1)

    attn_out = forward_batch.kv_cache.extend_wrapper.forward(
        q_fi, (k_paged, v_paged), causal=True
    )                                   # [T, n_q, D]
    return attn_out.reshape(B, T, n_q, D).permute(0, 2, 1, 3)
```

The key difference from Layer 9's `PrefillKVCtx` path is that there is no `F.sdpa` call for prefill any more. FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` handles the full ragged causal attention over paged KV for all requests in one kernel, whether they are fresh prompts or continuation chunks.

`Qwen3Model.forward` now detects the `kv_cache` type and builds the appropriate `ForwardBatch` internally, so `model_runner.py` can pass `kv_cache=ctx` without specifying a mode externally.

---

## The Full Loop

The scheduler loop runs continuously. Consider a 1024-token request arriving while two 50-token requests are already in `_running`, with `chunked_prefill_size=512`.

On the first scheduler iteration, `PrefillAdder.build()` sees a new 1024-token prompt that exceeds `chunked_prefill_size`. Tokens 0–511 are taken as `fill_ids`, `new_chunked_req` is set, and `prefill_batch([req])` runs. The 512-token extend kernel writes K/V for positions 0–511 into the pool, sets `req.kv_committed_len = 512`, and marks `req.status = PREFILLING`. After `prefill_batch` returns, `decode_step` runs for the two `_running` requests — they each advance one token while the first chunk was being processed.

On the second scheduler iteration, case 1 fires: `_chunked_req` is set, so `PrefillAdder.build()` slices tokens 512–1023 from `req.input_ids`. The pool's last page from the previous chunk may be partially filled; `compute_write_info` detects the leftover and completes that page before allocating new ones. The 512-token extend kernel runs again. `req.is_last_chunk` is true (we have reached `prompt_len`), so the first output token is sampled, `req.status = RUNNING`, `_chunked_req = None`, and the request joins the decode batch on the next iteration. `decode_step` again runs between the two iterations for the already-running requests.

From the third scheduler iteration onward, all three requests decode together. Finished requests call `kv_pool.free(req.slot_indices)` and `req_to_token_pool.free(req.req_pool_idx)`, immediately returning pages to the pool.

---

## What Comes Next

Layer 11 eliminates decode starvation but does not reuse any K/V data across requests. Two requests with the same 512-token system prompt each compute and store 512 × 28 × 2 × 8 × 128 × 2 = 7 MB of identical KV data independently. Layer 12 adds a `RadixCache` — a compressed trie keyed on token sequences — that maps shared prompt prefixes to their already-computed KV pages. When a new request arrives with a matching prefix, `match_prefix` returns those page indices, `kv_committed_len` is set to `prefix_len`, and `prefill_batch` only processes the unique suffix. The change touches `radix_cache.py` (new), `model_runner.prefill_batch` (prefix page injection before `compute_write_info`), and `scheduler.py` (`PrefillAdder` prefix lookup); `decode_step`, `forward_batch.py`, and the model layers are unchanged.
