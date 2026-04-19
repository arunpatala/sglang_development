# Layer 6 — Summary

Layer 6 replaces the static `generate_batch` loop with a continuous-batching scheduler: finished requests are evicted from the decode batch immediately, and new requests enter with their own prefill injected at the next iteration. The `model/` package, `tokenizer.py`, and `sampling.py` are unchanged; the new code lives in `scheduler.py`, `request.py`, `kv_cache.py`, and `model_runner.py`.

---

## From Layer 5 to Layer 6

In Layer 5, every call to `generate_batch` assembled a fixed set of B requests, ran prefill for all of them simultaneously, then ran a decode loop until the last request emitted EOS:

```python
# Layer 5 — static batch: all requests enter together, none leaves until all finish
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
kv = KVCache()
logits = self.model(input_ids, attention_mask=attention_mask,
                    kv_cache=kv, position_ids=prefill_pos)
next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]

for _ in range(max_new_tokens - 1):
    if finished.all():
        break
    logits = self.model(current, attention_mask=attention_mask,
                        kv_cache=kv, position_ids=decode_pos)
    next_tokens = sample_batch(logits[:, -1, :], temperature)
    finished = finished | (next_tokens == self.eos_id)
```

In Layer 6, there is no `generate_batch`. The server hands each request to a scheduler and immediately awaits a `Future`:

```python
# Layer 6 — server: enqueue and await; the scheduler drives everything
req = Req(rid=str(uuid.uuid4()), input_ids=input_ids,
          max_new_tokens=params.max_new_tokens,
          temperature=params.temperature,
          future=loop.create_future())
scheduler.add_request(req)
result = await req.future   # unblocked when the scheduler resolves it
```

The scheduler runs on its own background thread. On every iteration it prefills any waiting requests one at a time, runs one decode step across all currently active requests together, then evicts finished ones. A short request that finishes in five decode steps is returned to its caller after those five steps — it does not wait for a co-running request that needs a thousand.

---

## The Scheduler Loop

The scheduler's event loop has three steps that repeat indefinitely:

```python
while True:
    # Step 1 — Prefill new requests
    while not self._waiting.empty() and len(self._running) < self.max_running_reqs:
        req = self._waiting.get_nowait()
        self.model_runner.prefill(req)
        if req.status == ReqStatus.FINISHED:
            self._resolve(req)          # EOS on first token — done immediately
        else:
            self._running.append(req)

    # Step 2 — Decode step for all running requests
    if self._running:
        newly_finished = self.model_runner.decode_step(self._running)
        for req in newly_finished:
            self._resolve(req)
        self._running = [r for r in self._running if r.status == ReqStatus.RUNNING]

    # Step 3 — Idle if nothing to do
    if not did_work:
        time.sleep(IDLE_SLEEP_S)
```

Step 1 drains the waiting queue up to `max_running_reqs`. Each new request gets its own B=1 prefill — no grouping, no padding against other prompts. Step 2 runs one batched decode step across all currently running requests simultaneously. Step 3 sleeps for 1 ms when both queues are empty to avoid busy-spinning. The batch size is not fixed: `_running` grows as new requests are prefilled and shrinks as requests finish, without waiting to assemble a full batch before proceeding.

---

## The Request

Each request is represented by a `Req` dataclass that carries all the state the scheduler and model runner need:

```python
@dataclass
class Req:
    rid:            str               # UUID for logging
    input_ids:      List[int]         # tokenized prompt
    max_new_tokens: int
    temperature:    float
    future:         asyncio.Future    # resolved when generation is done
    status:         ReqStatus = ReqStatus.WAITING
    output_ids:     List[int] = field(default_factory=list)
    kv_cache:       Optional[object] = None   # PerReqKVCache, set after prefill
    t_arrive:       float = field(default_factory=time.perf_counter)
    t_first_token:  float = 0.0
    t_finish:       float = 0.0
```

`ReqStatus` is an enum with three values: `WAITING` (in the queue, not yet prefilled), `RUNNING` (prefilled, actively decoding), and `FINISHED` (EOS emitted or `max_new_tokens` reached). Transitions happen inside `ModelRunner.prefill` and `ModelRunner.decode_step` — the HTTP handler never touches GPU state, and the scheduler thread never touches the HTTP layer.

The `future` field is the bridge between the two threads. It is created in the asyncio event loop by the server, and resolved by the scheduler thread via `loop.call_soon_threadsafe`. The HTTP handler awaits it: when the scheduler resolves it, the handler unblocks and returns the result. The timing fields `t_arrive`, `t_first_token`, and `t_finish` are set on the scheduler thread; derived properties `ttft_ms` and `latency_ms` are computed from them when building the result dict.

---

## Per-Request KV Cache

In Layer 5, all B requests shared a single `KVCache` with shape `[B, n_kv_heads, seq_len, head_dim]`, requiring every request to have the same sequence length. This is impossible in continuous batching: requests arrive at different times and have been decoding for different numbers of steps.

Layer 6 introduces two cache classes. `PerReqKVCache` stores one request's accumulated K and V tensors across all 28 decoder layers:

```python
class PerReqKVCache:
    def update(self, layer_idx, new_k, new_v):
        if layer_idx in self._k:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        else:
            self._k[layer_idx] = new_k
            self._v[layer_idx] = new_v
        return self._k[layer_idx], self._v[layer_idx]
```

Each request owns one `PerReqKVCache`. It grows by one token per decode step. Because it belongs to the request rather than the batch, it survives across decode iterations and is discarded only when the request finishes.

`BatchedKVCache` solves the problem of running a single forward pass across requests at different KV lengths. It is constructed fresh at the start of each decode step, pads each request's per-layer tensors to the current maximum KV length, and stacks them into rectangular `[B, n_kv_heads, max_kv_len, head_dim]` tensors that the model's SDPA call can process in one kernel. After the forward pass, `write_back()` extracts the last-position slice from each stacked tensor and appends it to the corresponding `PerReqKVCache` — the only per-step write that needs to happen.

```python
def write_back(self):
    for layer_idx in self._initialized:
        full_k = self._k[layer_idx]       # [B, n_kv, max_kv+1, dim]
        for i, req in enumerate(self._reqs):
            new_k = full_k[i:i+1, :, -1:, :]   # [1, n_kv, 1, dim]
            cache._k[layer_idx] = torch.cat([cache._k[layer_idx], new_k], dim=2)
```

---

## Prefill

`ModelRunner.prefill` runs a single B=1 forward pass for a newly arrived request. There is no left-padding and no shared `max_prompt_len` — the input is the request's exact token sequence:

```python
def prefill(self, req: Req) -> None:
    ids  = torch.tensor([req.input_ids], device=DEVICE)           # [1, L]
    mask = torch.ones(1, len(req.input_ids), dtype=torch.long, device=DEVICE)
    pos  = torch.arange(len(req.input_ids), device=DEVICE).unsqueeze(0)  # [1, L]

    kv = PerReqKVCache()
    logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)

    req.kv_cache      = kv
    req.t_first_token = time.perf_counter()
    next_tok = self._sample(logits[0, -1], req.temperature)
    req.output_ids.append(next_tok)

    if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
        req.status = ReqStatus.FINISHED
    else:
        req.status = ReqStatus.RUNNING
```

The forward call populates `kv` through the model's 28 attention layers, each calling `kv.update(layer_idx, k, v)`. `req.t_first_token` is recorded immediately after the forward call — this is the TTFT timestamp for this individual request, not a property of the batch. If the first generated token is EOS, the request transitions directly to `FINISHED` without ever entering `_running`.

---

## The Decode Step

`ModelRunner.decode_step` is the heart of the batched decode. The challenge is that each of the N running requests is at a different position in its generation — different KV lengths, different next positions for RoPE. Three pieces of construction precede the forward call:

```python
kv_lens  = [r.kv_cache.get_seq_length() for r in reqs]
max_kv   = max(kv_lens)

# [B, 1] — each request's most recently generated token
last_toks = torch.tensor([[r.output_ids[-1]] for r in reqs], device=DEVICE)

# [B, max_kv+1] — left zeros for padding, ones for real KV + new token slot
attn_mask = torch.zeros(B, max_kv + 1, dtype=torch.long, device=DEVICE)
for i, kv_len in enumerate(kv_lens):
    attn_mask[i, max_kv - kv_len:] = 1

# [B, 1] — each request at its own absolute position, not the shared max_kv
pos_ids = torch.tensor([[kv_len] for kv_len in kv_lens], device=DEVICE)
```

`attn_mask` is constructed row by row: request `i` has `kv_len_i` real KV positions plus one new token slot, with left zeros for the gap to `max_kv`. When `_build_additive_mask` processes this mask, those zeros become `−inf`, preventing the new query token from attending to padded positions. `pos_ids` gives every request its own next absolute position — `kv_len_i` — rather than a shared `max_kv + step` offset. Without this, two requests at different decode stages would get the same RoPE encoding for their new token, breaking attention.

After the forward pass, `batch_kv.write_back()` appends the new K/V to each `PerReqKVCache`, then the loop samples one token per request, updates `output_ids`, and collects any newly finished requests to return to the scheduler.

---

## Thread Safety

Layer 6 introduces concurrency that was not present in any prior layer: the HTTP handler and the scheduler loop run on different threads and must exchange `Req` objects safely.

`queue.Queue` handles the enqueue side. The asyncio thread calls `self._waiting.put(req)` from within the FastAPI handler; `queue.Queue.put` is thread-safe and non-blocking. The scheduler thread calls `self._waiting.get_nowait()` in its drain loop. No explicit lock is needed around the queue.

The `_running` list is only ever read and written by the scheduler thread. The asyncio thread never touches it. This is an intentional design constraint: the server communicates with the scheduler only through the queue and through `Future` resolution, never by directly inspecting or modifying the running batch.

Resolving a `Future` from the scheduler thread requires care. Calling `req.future.set_result(result)` directly from a non-asyncio thread is not thread-safe and can corrupt the event loop. The correct call is:

```python
self._loop.call_soon_threadsafe(req.future.set_result, result)
```

This schedules the `set_result` call as a callback on the asyncio event loop. The event loop thread executes it at the next iteration, which unblocks the `await req.future` in the HTTP handler. The scheduler thread does not wait for this — it schedules the callback and continues immediately.

---

## The Full Loop

Tracing one request from HTTP POST to JSON response shows how every component connects.

The HTTP POST arrives at `server.py`. The handler tokenizes the conversation with `Tokenizer`, creates an `asyncio.Future` on the current event loop, wraps everything in a `Req` with `status=WAITING`, and calls `scheduler.add_request(req)`. This places the request in `_waiting` and returns immediately. The handler then `await req.future` — it is now suspended, holding no GPU resources.

On the scheduler thread, the next iteration of the event loop finds a non-empty `_waiting` queue and room in `_running`. It calls `model_runner.prefill(req)`. A B=1 forward pass runs over the request's prompt: token IDs are embedded, RoPE is applied, 28 attention layers build up `req.kv_cache`, and the final `lm_head` projection produces `logits [1, L, vocab]`. The first token is sampled from `logits[0, -1]`, appended to `req.output_ids`, and `req.t_first_token` is recorded. If the first token is not EOS, `req.status = RUNNING` and the request is appended to `_running`.

On subsequent scheduler iterations, `decode_step(_running)` is called. `BatchedKVCache` pads and stacks all running requests' per-layer K/V tensors into rectangular `[N, heads, max_kv, dim]` batches. One batched forward call processes all N requests simultaneously. `write_back()` appends the new K/V slice to each `PerReqKVCache`. Each request's next token is sampled; requests that emit EOS are marked `FINISHED` and added to the `newly_finished` list returned to the scheduler.

The scheduler calls `_resolve(req)` for each finished request. `decode_output(req)` decodes `req.output_ids` to a string. A result dict is built with `text`, `prompt_tokens`, `completion_tokens`, `ttft_ms`, and `latency_ms` computed from the request's timing fields. `loop.call_soon_threadsafe(req.future.set_result, result)` schedules the resolution on the asyncio thread. On the asyncio thread, `await req.future` unblocks, and the handler returns the JSON response to the caller.

---

## What Comes Next

The remaining inefficiency in Layer 6 is padding inside `BatchedKVCache`. On each decode step, every request's historical K/V is padded to the current maximum KV length in the running batch. A request that arrived recently and has 10 tokens of KV cache is padded to the length of a request that has been running for 1000 steps — the attention kernel performs 100× more work than necessary for that row. As the running batch grows and requests diverge in age, padding waste compounds.

The second cost is memory fragmentation. `PerReqKVCache` grows by one `torch.cat` allocation per decode step per layer. Over a long generation, this produces 28 × `seq_len` separate allocations, none of which can be reused by another request. GPU memory cannot be compacted while tensors are live.

Layer 7 (`layer7_paged_attention`) addresses both problems by dividing physical KV memory into fixed-size pages and allocating pages to requests on demand. Requests of different lengths share the same physical memory pool without padding between them, and completed requests return their pages immediately for reuse. The scheduler and `model/` package are unchanged — the work moves into a page-table-based `kv_cache.py` and the model runner's page allocation logic.
