# Layer 5 — Continuous Batching

## What changes vs Layer 3 (static batching)

| | Layer 3 | Layer 5 |
|---|---|---|
| Request model | Synchronous, blocking | Async (FastAPI + asyncio.Future) |
| Batch formation | Fixed batch, wait for all to finish | Dynamic — requests join/leave every step |
| Prefill | All prompts at once (padded together) | B=1 per request, as it arrives |
| Decode | All in batch until all finish | All running requests, every step |
| Head-of-line blocking | Yes — one long req stalls all | No — short reqs finish independently |
| KV cache | Single shared tensor `[B, heads, seq, dim]` | Per-request `PerReqKVCache` |

## Architecture

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
    │     model_runner.prefill(req) → populates req.kv_cache
    │
    ├─ decode all running requests together (B=N)
    │     model_runner.decode_step(running_reqs)
    │       → BatchedKVCache: pads & stacks per-request K/V
    │       → one model forward pass for all N requests
    │       → write_back: appends new K/V to each PerReqKVCache
    │
    └─ resolve finished requests
          loop.call_soon_threadsafe(req.future.set_result, result)
```

## The BatchedKVCache — padded KV for variable-length requests

At each decode step, running requests have different KV lengths because
they arrived at different times.  `F.sdpa` needs a rectangular tensor.

```
Step 10, 3 requests with KV lengths [8, 5, 3], max=8:

  KV cache before decode:
    Req 0: [K0...K7]          length 8
    Req 1: [K0...K4]          length 5 → padded [0,0,0,K0...K4]
    Req 2: [K0...K2]          length 3 → padded [0,0,0,0,0,K0,K1,K2]

  attention_mask [3, 9]:
    Req 0: [1,1,1,1,1,1,1,1, 1]   8 real + 1 new
    Req 1: [0,0,0,1,1,1,1,1, 1]   3 pad + 5 real + 1 new
    Req 2: [0,0,0,0,0,1,1,1, 1]   5 pad + 3 real + 1 new

  position_ids [3, 1]: [[8], [5], [3]]  ← per-request, not shared max

  After decode: write_back() appends only the new [1, n_kv, 1, dim]
  token to each PerReqKVCache — so storage stays compact.
```

## Key concepts demonstrated

- **Continuous batching**: requests enter and leave the decode batch
  independently — no waiting for a full batch to fill or finish
- **asyncio.Future bridge**: the HTTP layer awaits a future; the
  scheduler resolves it from a background thread via `call_soon_threadsafe`
- **Per-request KV cache**: each `Req` owns its own `PerReqKVCache`,
  enabling independent lifecycle management
- **BatchedKVCache**: temporary pad-and-stack view for one decode step,
  with `write_back()` to return new tokens to per-request storage
- **Per-request position_ids**: each request decodes at its own position
  (`kv_len_i`), not the shared `max_kv_len`, ensuring correct RoPE

## Files

| File | Role |
|------|------|
| `request.py` | `Req` dataclass + `ReqStatus` |
| `batch.py` | `Batch` + `ForwardMode` (PREFILL / DECODE) |
| `kv_cache.py` | `PerReqKVCache` + `BatchedKVCache` |
| `model_runner.py` | `prefill()` + `decode_step()` |
| `scheduler.py` | Event loop thread, waiting/running queues |
| `server.py` | FastAPI async server, Future bridge |
| `model/` | Qwen3ForCausalLM (unchanged from Layer 4) |
| `tokenizer.py` | Tokenizer (unchanged from Layer 4) |
| `benchmark.py` | Concurrent async requests benchmark |

## What's missing (Layer 6+)

- **Paged KV cache**: `BatchedKVCache.write_back()` still copies K/V
  tensors. With PagedAttention, requests share a global pool of fixed-size
  blocks — no copy, no padding waste.
- **FlashInfer ragged kernels**: replace `F.sdpa` + padding with
  `BatchDecodeWithPagedKVCacheWrapper` for true variable-length batches.
- **Chunked prefill**: long prompts currently monopolise one scheduler
  step. Chunking splits prefill across steps interleaved with decodes.
