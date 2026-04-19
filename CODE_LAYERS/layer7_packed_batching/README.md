# Layer 6 — Packed Batching (Ragged KV + FlashInfer)

## What changes vs Layer 5 (continuous batching + F.sdpa)

| | Layer 5 | Layer 6 |
|---|---|---|
| Prefill | B=1, `PerReqKVCache`, `F.sdpa` | **Same — unchanged** |
| Decode batch structure | `BatchedKVCache`: left-pads all KVs to max_kv_len | `PackedKVCache`: concatenates KVs into one ragged tensor |
| Attention kernel (decode) | `F.scaled_dot_product_attention` | `FlashInfer.BatchPrefillWithRaggedKVCacheWrapper` |
| Decode mask | `[B, max_kv+1]` attention_mask passed to model | None — FlashInfer uses `kv_indptr` internally |
| Padding waste | O(max_kv_len) compute per padding column | Zero — only real tokens processed |
| Copy cost | O(total_kv_tokens) copy at each step | Same — copy still happens (see below) |
| GQA head expand | `repeat_kv()` expands 8→16 heads | FlashInfer handles GQA natively (no expand) |

## The key idea: ragged packing

Layer 5 batched decode with padding:
```
req0: [pad pad pad pad K0 K1 K2 K3 K4 K5 K6 K7 K8 K9]  → 14 cols, 4 wasted
req1: [pad pad pad pad pad pad pad pad K0 K1 K2 K3 K4 K5]  → same
req2: [pad pad pad pad pad pad pad pad pad pad K0 K1 K2 K3]  → same
```

Layer 6 packed (ragged) decode:
```
[K0..K9_req0 | K10_new | K0..K5_req1 | K6_new | K0..K3_req2 | K4_new]
 ↑─────────────────────↑              ↑─────────────────────↑
 kv_indptr[0..1] = 0,11              kv_indptr[1..2] = 11,18
```

`kv_indptr` = `[0, 11, 18, 23]` tells FlashInfer which slice of the
packed tensor belongs to each request. No zeros to compute attention over.

## What still has a copy cost

Building `pack_k` / `pack_v` requires gathering each request's KV tensors
into a new contiguous buffer at every decode step. This is an O(total_kv)
memory copy even though only 1 new token per request was added.

**Layer 7 (paged KV cache)** eliminates this copy by keeping KV in a
pre-allocated block table. FlashInfer reads the block table directly —
no gather needed. This is how production systems like SGLang work.

## Architecture (same as Layer 5, only decode internals change)

```
HTTP Client
    │  POST /v1/chat/completions
    ▼
server.py (FastAPI, asyncio event loop, main thread)
    │  tokenize → create Req(future) → scheduler.add_request()
    │  await future  ← blocks the coroutine (not the thread)
    ▼
scheduler.py (background daemon thread)
    │
    ├─ prefill new requests one at a time (B=1)
    │     model_runner.prefill(req) → PerReqKVCache + F.sdpa (unchanged)
    │
    └─ decode all running requests together (B=N)
          model_runner.decode_step(running_reqs)
              │
              ├─ build pos_ids [B, 1] (per-request)
              ├─ PackedKVCache(reqs, workspace)
              ├─ pack_kv.plan(heads, head_dim, dtype)  ← one-time FlashInfer setup
              ├─ model(last_toks, mask=None, kv_cache=pack_kv, pos=pos_ids)
              │     └─ attention.py:
              │           detect PackedKVCache → call pack_kv.forward_attn(layer, q, k, v)
              │               └─ pack hist_KV + new_KV → wrapper.forward() [FlashInfer]
              ├─ pack_kv.write_back()  ← append new token to each PerReqKVCache
              └─ pack_kv.end_forward()
    ▼
loop.call_soon_threadsafe(future.set_result, result)
    ▼
server.py coroutine resumes → returns HTTP response
```

## Files changed vs Layer 5

| File | Change |
|---|---|
| `kv_cache.py` | Added `PackedKVCache`, removed `BatchedKVCache` |
| `model/attention.py` | Added FlashInfer decode path alongside F.sdpa prefill path |
| `model_runner.py` | `decode_step` uses `PackedKVCache`; allocates `_workspace` once at init |

## Running

```bash
# Server (port 8105 same as layer5)
python server.py --model Qwen/Qwen3-0.6B

# Benchmark
python benchmark.py --concurrency 4 --n-requests 20
```
