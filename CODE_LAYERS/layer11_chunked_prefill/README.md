# Layer 10 — Batched Prefill + Chunked Prefill

Builds on Layer 9 (paged KV cache, `page_size=16`) by replacing the B=1 sequential prefill loop with:

1. **Batched prefill** — multiple requests packed into one EXTEND forward pass  
2. **Chunked prefill** — long prompts split across rounds, interleaved with decode

---

## What changed from Layer 9

| Component | Layer 9 | Layer 10 |
|-----------|---------|---------|
| Prefill batch size | B=1, one at a time | B=N, all new reqs together |
| Prefill kernel | F.sdpa (full causal) | `BatchPrefillWithPagedKVCacheWrapper` |
| Chunked prefill | ✗ | ✓ (`chunked_prefill_size` param) |
| `PrefillKVCtx` | Used for B=1 | Removed |
| `ExtendKVCtx` | — | New: handles first chunk, continuation, batched |
| Page packing | ✗ | ✓ (chunks continue filling partial pages) |
| `Req.fill_ids` | — | Token slice for this extend round |
| `Req.kv_committed_len` | — | Tokens already in KV pool |
| `Scheduler.chunked_req` | — | In-flight chunked request state |
| `PrefillAdder` | — | Builds extend batch (token budget, chunk logic) |

---

## Architecture

### Unified `ExtendKVCtx`

Replaces `PrefillKVCtx`. One type handles all cases:

```
Case 1: Full prefill, B=1     kv_committed_len=0, fill_ids=input_ids
Case 2: Batched prefill, B=N  kv_committed_len=0 for all, packed
Case 3: Continuation chunk    kv_committed_len>0, fill_ids=chunk_slice
```

All three go through `BatchPrefillWithPagedKVCacheWrapper(causal=True)`.

### Page packing across chunk boundaries

Without page packing, each chunk independently allocates pages:
```
P=16, chunk_size=24
Chunk 0 → 2 pages (0..15 full, 16..23 partial)
Chunk 1 → 2 MORE pages (not continuing page 2!)
= 4 pages for 48 tokens, kv_last_page_len is wrong
```

With page packing (Layer 10):
```
Chunk 0 → pages [1, 2]:  page 1 full (16 tokens), page 2 partial (8 tokens)
Chunk 1 → FILL page 2 first (8 more tokens), then allocate page 3 (16 tokens)
= 3 pages for 48 tokens, kv_last_page_len=16 correct
```

Invariant maintained: `len(slot_indices) == ceil(total_committed / P)`

### Scheduler flow

```
Round k:
  1. PrefillAdder.build()
     - If chunked_req set: continue it (return [chunked_req])
     - Else: pick N requests up to max_prefill_tokens budget
             If a request needs chunking: set chunked_req, return [req]
  2. model_runner.prefill_batch(batch)
     - For each req: compute_write_info() → page packing → rtp update
     - Pack all fill_ids → forward with ExtendKVCtx
     - After last chunk: sample first token, RUNNING/FINISHED
     - Mid-chunk: PREFILLING, stay in chunked_req
  3. model_runner.decode_step(running)  ← runs in parallel with chunked prefill!
```

Chunked requests are NOT in the decode batch during prefill — they join only after the last chunk completes.

### `causal=True` in both `begin_forward` and `forward`

FlashInfer 0.6.x has `causal` as a parameter on both `begin_forward` AND `forward`. Passing it only to `begin_forward` does not apply causal masking — it must be passed to `forward(q, kv, causal=True)` as well.

---

## Files changed

| File | Change |
|------|--------|
| `request.py` | `ReqStatus.PREFILLING`, `fill_ids`, `kv_committed_len`, `extend_input_len`, `is_last_chunk` |
| `kv_cache.py` | `ExtendKVCtx` (replaces `PrefillKVCtx`), `WriteInfo`, `compute_write_info()` |
| `model/attention.py` | `ExtendKVCtx` branch: paged prefill via `BatchPrefillWithPagedKVCacheWrapper` |
| `model_runner.py` | `prefill_batch(reqs)` replaces `prefill(req)`, uses `compute_write_info` |
| `scheduler.py` | `PrefillAdder`, `chunked_req` state machine, `chunked_prefill_size` param |
| `verify_batch.py` | 3 tests: batched prefill, chunked prefill, chunked→decode |

---

## Benchmark

Layer 10 throughput is comparable to Layer 9 (same `chunked_prefill_size=0` default,
so no chunking in the benchmark — the new infrastructure is in place but not exercised
by the constant-concurrency benchmark which uses short prompts):

| Metric | Layer 9 | Layer 10 |
|--------|---------|---------|
| Total wall time | 8.08s | 8.10s |
| Output tok/s | 219.9 | 219.4 |
| Total tok/s | 346.6 | 345.7 |
| TTFT avg/p95 | 172ms / 739ms | 188ms / 808ms |
| Latency avg/p95 | 1326ms / 2307ms | 1367ms / 2586ms |

The slight TTFT increase is because `prefill_batch` now uses the paged prefill wrapper
(which has more overhead than F.sdpa for short prompts) for ALL requests. The benefit
of chunked prefill becomes visible with **long prompts** where the old B=1 sequential
prefill blocked decode for hundreds of milliseconds.

---

## Verify

```bash
python verify_batch.py --model <path> --page-size 16 --chunk-size 24 --n-compare 8
# All 3 tests PASS ✓
```

**Test 1** — Batched prefill (B=4, one EXTEND pass): max_diff < 0.55  
**Test 2** — Chunked prefill (chunk_size=24, 5 chunks): max_diff < 0.40  
**Test 3** — Chunked prefill → 8 decode steps: max_diff < 0.40
