# 01 — From Prefill to Extend

## From Layer 9 to Layer 11

Layer 9 established the paged KV pool with `page_size=16`, a GPU-resident `ReqToTokenPool`, and the Triton kernel for on-device `kv_indices` construction. Its `prefill(req)` method still processed one request at a time, blocking the entire scheduler loop until the full prompt was written into the pool:

```python
# Layer 9 — model_runner.prefill (B=1, one request, one F.sdpa call)
prompt_len = len(req.input_ids)
P          = self.page_size
n_pages    = math.ceil(prompt_len / P)

page_indices     = self.kv_pool.alloc(prompt_len)    # allocate n_pages
req.slot_indices = page_indices

req.req_pool_idx = self.req_to_token_pool.alloc()
pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t

ids  = torch.tensor([req.input_ids], device=DEVICE)
mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

ctx = PrefillKVCtx(page_indices, self.kv_pool)
fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
with torch.no_grad():
    logits = self.model(ids, forward_batch=fb, position_ids=pos)

next_tok = sample_token(logits[0, -1], req.temperature)
req.output_ids.append(next_tok)
```

A 4096-token prompt required one `F.sdpa` call over 4096 positions. The decode batch sat idle throughout. A server with 16 running requests generating 50 ms per decode step would stall those requests for the full prefill duration — potentially hundreds of milliseconds — before processing a single additional decode token.

In Layer 11, the same entry point is `prefill_batch(reqs)` and it handles any number of requests in one FlashInfer `BatchPrefillWithPagedKVCacheWrapper` call:

```python
# Layer 11 — model_runner.prefill_batch (batched extend, FlashInfer paged prefill)
write_infos = [
    compute_write_info(kv_pool, rtp, req.slot_indices, req.req_pool_idx,
                       req.kv_committed_len, req.extend_input_len)
    for req in reqs
]

all_ids = [tok for req in reqs for tok in req.fill_ids]
ids_t   = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
pos_t   = torch.tensor(
    [req.kv_committed_len + j
     for req in reqs
     for j in range(req.extend_input_len)],
    dtype=torch.long, device=DEVICE
).unsqueeze(0)

self._extend_wrapper.begin_forward(
    qo_indptr_t, kv_indptr, kv_indices, kv_last_page_lens,
    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
    P, causal=True, q_data_type=DTYPE,
)
ctx    = ExtendKVCtx(wrapper=self._extend_wrapper, k_pool=..., v_pool=...,
                     qo_indptr=qo_indptr_list, write_infos=write_infos, page_size=P)
logits = self.model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
self._extend_wrapper.end_forward()
```

The two key structural differences are: `fill_ids` can be a slice of `input_ids` (controlled by `chunked_prefill_size`), and multiple requests' tokens are packed into a single `[1, total_tokens]` input. The scheduler interleaves decode steps with each prefill chunk, so no single request can monopolize the GPU for more than `chunked_prefill_size` token positions.

---

## What Changes and What Does Not

Three files changed substantially. `model_runner.py` replaces `prefill(req)` with `prefill_batch(reqs)`, adds `compute_write_info` for page-packing arithmetic, and builds `qo_indptr` in addition to the decode-path `kv_indptr`. `scheduler.py` gains `PrefillAdder` (the request selection and chunking logic) and `self._chunked_req` (the state variable that tracks a partially-prefilled request). `kv_cache.py` replaces `PrefillKVCtx` with `ExtendKVCtx`, which handles batched ragged-sequence writes using `WriteInfo` objects.

`forward_batch.py` gains `ForwardMode.EXTEND` (previously `PREFILL`) and `ForwardMode.NOCACHE`. `model/backend.py` adds `_extend_forward` alongside `_decode_forward`. The model files (`qwen3.py`, `decoder_layer.py`) are mechanically updated to pass `ForwardBatch` through.

Everything else is unchanged. `decode_step` is identical to Layer 9. The server, tokenizer, sampling, and request lifecycle are unchanged. The `KVPool` shape, `ReqToTokenPool`, and Triton `kv_indices` kernel carry forward without modification. The `BatchDecodeWithPagedKVCacheWrapper` is untouched.

---

## Why Chunking Works

The key insight is that FlashInfer's `BatchPrefillWithPagedKVCacheWrapper` supports a `kv_committed_len` concept: the extend kernel can attend over pages that are already in the pool (from prior chunks) in addition to the pages being written in the current pass. This is expressed through `kv_indptr`, which counts all pages per request — prefix pages plus new pages — and `kv_last_page_lens`, which communicates the fill level of the last page.

When `chunked_prefill_size = 512` and a prompt is 1024 tokens, the first chunk writes tokens 0–511 into pool pages and advances `kv_committed_len` to 512. The second chunk writes tokens 512–1023. The extend kernel for the second chunk attends over all pages (both the 32 pages from chunk 1 and the 32 new pages from chunk 2) with a causal mask, giving the same result as a single 1024-position forward pass. No recomputation of the first chunk's K/V is needed.

This section traces from the structural shift. The next sections explain each new piece in code order: section 02 explains the scheduler's `_chunked_req` state machine; section 03 covers `PrefillAdder`'s token budget logic; section 04 explains `WriteInfo` and the page-packing arithmetic; section 05 covers `prefill_batch` step by step; section 06 explains `EXTEND` mode in the attention backend.
