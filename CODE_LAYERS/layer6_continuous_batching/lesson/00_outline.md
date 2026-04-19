# Layer 6 — Lesson Outline

## What This Lesson Covers

Layer 5 processed B requests in a single `generate_batch` call: prefill all prompts at once, decode until every request hits EOS, then return. A short request finishes after a handful of steps but cannot be returned until the longest request in the batch is done — head-of-line blocking. GPU slots occupied by finished requests sit idle while waiting for their neighbours.

Layer 6 replaces `generate_batch` with a continuous-batching scheduler. A background thread maintains two structures: a `waiting_queue` for requests that have arrived but not yet been prefilled, and a `running` list for requests actively decoding. On every loop iteration, it prefills any waiting requests one at a time (B=1 each), runs one decode step across all running requests simultaneously (B=N), then evicts any requests that just emitted EOS. A finished request is removed immediately — no other request waits for it.

The change touches five files: `scheduler.py`, `request.py`, `kv_cache.py`, `model_runner.py`, and `server.py`. The `model/` package is carried over from Layer 5 unchanged. `tokenizer.py` and `sampling.py` are likewise unchanged.

The sections follow execution order: the scheduler loop that drives everything, the `Req` object that carries per-request state, the two KV cache classes that solve the ragged-batch problem, prefill and decode as separate model runner methods, and the asyncio–thread-safety bridge.

---

## Sections

### 01 — The Scheduler Loop (`01_the_scheduler_loop.md`)
- Layer 5's `generate_batch` vs Layer 6's scheduler: the structural shift from one blocking call to a background daemon thread driving independent prefill and decode events
- `Scheduler.__init__`: `_waiting: queue.Queue[Req]`, `_running: List[Req]`, `max_running_reqs` capacity cap, `_loop` asyncio event loop handle
- `Scheduler.run(loop)`: the three-step event loop — (1) drain `_waiting`, prefill each, move to `_running`; (2) if `_running` non-empty, call `decode_step`; (3) if idle, sleep 1 ms to avoid spinning
- `add_request(req)`: the asyncio-thread-facing entry point — puts into `queue.Queue` without blocking
- Dynamic batch size: `_running` grows as requests arrive and shrinks as requests finish, with no fixed batch assembly step

### 02 — The Request (`02_the_request.md`)
- `ReqStatus` enum: `WAITING` → `RUNNING` → `FINISHED` lifecycle; transitions happen inside `ModelRunner.prefill` and `ModelRunner.decode_step`, never in the HTTP handler
- `Req` dataclass fields: `rid` (UUID), `input_ids`, `max_new_tokens`, `temperature`, `future`, `status`, `output_ids`, `kv_cache`
- `asyncio.Future` as the thread bridge: created by the server in the asyncio event loop, resolved by the scheduler thread via `loop.call_soon_threadsafe(req.future.set_result, result)` — the only safe way to set a Future from a non-asyncio thread
- Timing fields: `t_arrive`, `t_first_token`, `t_finish`; derived properties `ttft_ms` and `latency_ms` computed from these
- Why `kv_cache` lives on `Req`: each request owns its own `PerReqKVCache` so it can join and leave the decode batch independently

### 03 — Per-Request KV Cache (`03_per_request_kv_cache.md`)
- The ragged-batch problem: running requests have different KV lengths because they arrived at different times; Layer 5's single shared `KVCache [B, heads, seq, dim]` requires all B requests to be at the same sequence length
- `PerReqKVCache`: one per `Req`, dict of `layer_idx → [1, n_kv_heads, seq_len, head_dim]`; grows by one token per decode step via `torch.cat` along the sequence dimension
- `PerReqKVCache.update(layer_idx, new_k, new_v)`: concatenates new K/V and returns the full accumulated tensors; called by each attention layer during prefill
- `BatchedKVCache`: a temporary view built at the start of each decode step; pads each request's per-layer K/V to `max_kv_len` and stacks into `[B, heads, max_kv_len, dim]`
- `BatchedKVCache._init_layer`: lazy per-layer initialisation — `F.pad(..., (0, 0, pad, 0))` left-pads the sequence dimension; `torch.cat(ks, dim=0)` stacks into a single batch tensor
- `BatchedKVCache.write_back()`: after the forward pass, extracts the last-position slice `[i:i+1, :, -1:, :]` from each stacked layer and appends it to the corresponding `PerReqKVCache`

### 04 — Prefill (`04_prefill.md`)
- `ModelRunner.prefill(req)`: B=1 forward pass for one newly arrived request; no padding, no left-alignment — `attention_mask` is all ones, `position_ids` is `torch.arange(prompt_len)`
- `PerReqKVCache()` created fresh before the forward call; 28 attention layers each call `kv.update(layer_idx, k, v)` during the pass, populating all 28 slots
- `req.t_first_token = time.perf_counter()` recorded immediately after the forward call — this is the TTFT timestamp
- Sampling the first token: `_sample(logits[0, -1], req.temperature)` — single token from last position of the B=1 output
- EOS check: if the first token is EOS or `output_len >= max_new_tokens`, `req.status = FINISHED` immediately; the scheduler resolves the future without adding the request to `_running`

### 05 — The Decode Step (`05_the_decode_step.md`)
- `ModelRunner.decode_step(reqs)`: one forward pass for all N currently running requests; returns the list of newly finished requests
- `kv_lens = [r.kv_cache.get_seq_length() for r in reqs]`: each request reports its own current KV length; `max_kv = max(kv_lens)`
- `last_toks [B, 1]`: each request's most recently generated token as the decoder input
- `attn_mask [B, max_kv+1]` construction: `torch.zeros(B, max_kv+1)` then `attn_mask[i, max_kv - kv_len_i:] = 1` — real KV positions plus the new token slot; left zeros are padding that `_build_additive_mask` converts to `−inf`
- `pos_ids [B, 1]`: each request at its own absolute position `kv_len_i`, not the shared `max_kv + step` offset — preserves per-request RoPE correctness for requests at different stages
- `BatchedKVCache(reqs, max_kv)` passed to the model forward call; `write_back()` called immediately after to append new K/V to each `PerReqKVCache`
- Post-forward: sample one token per request, update `output_ids`, check EOS or `max_new_tokens`, collect `newly_finished`

### 06 — Thread Safety (`06_thread_safety.md`)
- Two threads: the asyncio event loop thread (handles HTTP, creates Futures, calls `add_request`) and the scheduler daemon thread (drives prefill, decode, and Future resolution)
- `queue.Queue` is the synchronisation primitive: the asyncio thread calls `put` without blocking; the scheduler thread calls `get_nowait` in a non-blocking drain loop — no explicit locks needed
- `_running` is only ever touched by the scheduler thread: no lock needed for the list itself
- `loop.call_soon_threadsafe(req.future.set_result, result)`: the correct way to resolve an asyncio Future from a non-asyncio thread; schedules the callback on the event loop without blocking the scheduler thread
- Why not `future.set_result` directly: asyncio Futures are not thread-safe; calling `set_result` from the wrong thread causes a race condition that may silently corrupt the event loop state

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace of one request from HTTP POST to JSON response, naming every component in execution order
- Step 1 — Server receives POST: `server.py` tokenizes with `Tokenizer`, creates `asyncio.Future`, builds `Req`, calls `scheduler.add_request(req)`, then `await req.future`
- Step 2 — Scheduler picks up the request: `_waiting.get_nowait()` in the prefill drain loop; `model_runner.prefill(req)` runs a B=1 forward; `req.kv_cache` populated; first token sampled; `req.status = RUNNING`; request appended to `_running`
- Step 3 — Decode iterations: each scheduler loop iteration calls `decode_step(_running)`; `BatchedKVCache` pads and stacks all running requests; one batched forward; `write_back()` updates each `PerReqKVCache`; newly finished requests collected
- Step 4 — Resolution: scheduler calls `_resolve(req)` on each finished request; `decode_output` decodes `output_ids` to text; result dict built; `loop.call_soon_threadsafe(req.future.set_result, result)` unblocks the waiting HTTP handler; response returned

### 08 — What Comes Next (`08_whats_next.md`)
- The remaining inefficiency in Layer 6: `BatchedKVCache` pads every request's KV to `max_kv_len` every decode step — a request with 10 tokens in a batch where another has 1000 performs 100× more attention work than necessary
- Memory fragmentation: each `PerReqKVCache` allocates contiguous tensors that grow one token at a time, causing repeated `torch.cat` allocations and GPU memory fragmentation
- Layer 7 (`layer7_paged_attention`): physical KV memory divided into fixed-size pages; each request is allocated pages as needed; no padding between requests, no contiguous reallocation
- What changes: `kv_cache.py` becomes a page-table-based cache; `model_runner.py` manages page allocation; the scheduler and `model/` package are untouched

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 6 concepts to SGLang source: `Scheduler` → `sglang/srt/managers/scheduler.py`; `Req` → `ScheduleBatch`; `PerReqKVCache` → `RadixCache`; `BatchedKVCache` → `ModelRunner.forward_batch`; `loop.call_soon_threadsafe` → SGLang's async–thread bridge in `AsyncEngine`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `add_request` entry point | `scheduler.py` line 84: `self._waiting.put(req)` |
| Scheduler event loop | `scheduler.py` line 103: `while True:` |
| Prefill drain | `scheduler.py` line 108: `while not self._waiting.empty() and len(self._running) < self.max_running_reqs:` |
| Decode step call | `scheduler.py` line 131: `newly_finished = self.model_runner.decode_step(self._running)` |
| Future resolution | `scheduler.py` line 167: `self._loop.call_soon_threadsafe(req.future.set_result, result)` |
| `ReqStatus` enum | `request.py` line 24: `class ReqStatus(Enum):` |
| `Req` dataclass | `request.py` line 31: `@dataclass class Req:` |
| `asyncio.Future` field | `request.py` line 43: `future: asyncio.Future` |
| `PerReqKVCache.update` | `kv_cache.py` line 52: `def update(self, layer_idx, new_k, new_v)` |
| `PerReqKVCache.get_seq_length` | `kv_cache.py` line 67: `return next(iter(self._k.values())).shape[2]` |
| `BatchedKVCache._init_layer` | `kv_cache.py` line 93: `rk = F.pad(rk, (0, 0, pad, 0))` |
| `BatchedKVCache.update` | `kv_cache.py` line 116: `self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)` |
| `BatchedKVCache.write_back` | `kv_cache.py` line 137: `new_k = full_k[i:i+1, :, -1:, :]` |
| `ModelRunner.prefill` | `model_runner.py` line 88: `def prefill(self, req: Req) -> None:` |
| Prefill forward call | `model_runner.py` line 102: `logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)` |
| TTFT timestamp | `model_runner.py` line 105: `req.t_first_token = time.perf_counter()` |
| `ModelRunner.decode_step` | `model_runner.py` line 125: `def decode_step(self, reqs: List[Req]) -> List[Req]:` |
| `attn_mask` construction | `model_runner.py` line 149: `attn_mask[i, max_kv - kv_len:] = 1` |
| Per-request `pos_ids` | `model_runner.py` line 156: `[[kv_len] for kv_len in kv_lens]` |
| `BatchedKVCache` created | `model_runner.py` line 161: `batch_kv = BatchedKVCache(reqs, max_kv)` |
| `write_back` call | `model_runner.py` line 171: `batch_kv.write_back()` |
