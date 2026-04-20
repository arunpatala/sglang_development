# 02 — The Chunked State Machine

The scheduler's primary job is to decide which requests to run each iteration. In Layer 9 this was straightforward: drain from `_waiting` up to `max_running_reqs`, prefill each individually, then run the decode batch. In Layer 11, prefill can span multiple scheduler iterations — a single request may be "in progress" across many rounds without blocking new requests from being prefilled alongside it. The `_chunked_req` variable and `PrefillAdder` together implement this multi-round tracking.

---

## The _chunked_req Variable

```python
# scheduler.py — Scheduler.__init__
self._chunked_req: Optional[Req] = None
```

`_chunked_req` holds a reference to the one request that is currently mid-prefill. `None` means all requests are either fully prefilled (in `_running`) or waiting in `_waiting` — there is no partially-processed prompt. At most one request can be mid-prefill at any time: allowing multiple simultaneous chunked requests would require a second state variable, complicate budget accounting, and risk starvation when two long prompts compete for the same slot budget.

The state transitions are:

When `PrefillAdder.build()` returns a request with `chunked_prefill_size and req.prompt_len > chunked_prefill_size`, the scheduler sets `self._chunked_req = req` (via `adder.new_chunked_req`). The request's status becomes `PREFILLING`. On the next scheduler iteration, `_chunked_req is not None` triggers case 1 of `build()` — the next chunk is taken and the request stays in `PREFILLING`. When `prefill_batch` processes the final chunk (detected by `req.is_last_chunk`), the request's status transitions to `RUNNING` or `FINISHED`, and the scheduler clears `_chunked_req = None`.

```python
# scheduler.py — Scheduler.run (routing after prefill_batch returns)
for req in prefill_batch:
    if req.status == ReqStatus.FINISHED:
        self._resolve(req)
        if req is self._chunked_req:
            self._chunked_req = None
    elif req.status == ReqStatus.RUNNING:
        self._running.append(req)
        if req is self._chunked_req:
            self._chunked_req = None
    elif req.status == ReqStatus.PREFILLING:
        pass   # chunked_req stays set; processed again next round
```

The `req is self._chunked_req` identity check (not equality) is important: if the request object is the same one being tracked, clearing `_chunked_req` is correct. If somehow a different `PREFILLING` request slipped in (which the current code prevents), clearing it prematurely would lose track of the in-flight chunk.

---

## PrefillAdder.build: Two Cases

`PrefillAdder` is a short-lived helper object constructed at the start of each scheduler iteration. `build()` returns the list of requests for the next `prefill_batch` call, having set `fill_ids` and `extend_input_len` on each.

```python
# scheduler.py — PrefillAdder.build() — Case 1
if self.chunked_req is not None:
    req   = self.chunked_req
    start = req.kv_committed_len
    size  = self.chunked_prefill_size or (req.prompt_len - start)
    end   = min(start + size, req.prompt_len)
    req.fill_ids         = req.input_ids[start:end]
    req.extend_input_len = end - start
    return [req]
```

Case 1 fires when a chunked request is already in flight. `kv_committed_len` is the number of tokens whose K/V is already in the pool from previous chunks. `start = kv_committed_len`, `end = min(start + chunked_prefill_size, prompt_len)`. The slice `input_ids[start:end]` is the next chunk. After `prefill_batch`, `kv_committed_len` will advance by `end - start`, and the cycle continues.

```python
# scheduler.py — PrefillAdder.build() — Case 2
batch: List[Req] = []
rem_tokens = self.max_prefill_tokens

while True:
    if self.running_count + len(batch) >= self.max_running_reqs:
        break
    if rem_tokens <= 0 and batch:
        break
    if self.waiting.empty():
        break

    req = self.waiting.queue[0]   # peek

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

Case 2 fires when `_chunked_req is None`. New requests are dequeued from `_waiting`. Requests small enough to fit within `max_prefill_tokens` are taken whole. A request larger than `chunked_prefill_size` triggers the chunk path: the first chunk is taken, `new_chunked_req` is set, and the loop stops — only one chunked request per round, to maintain the invariant.

---

## Why Only One Chunked Request at a Time

Allowing two chunked requests simultaneously would require the scheduler to allocate `chunked_prefill_size` tokens to each on every iteration. When both are near their last chunk, neither can be processed alone — the budget must be split. This creates a fixed-priority problem: if request A's last chunk is 300 tokens and request B's is 400 tokens, and the budget is 512, both cannot finish in one pass even though each individually could. Serializing chunked requests avoids this: the single `_chunked_req` always gets a full `chunked_prefill_size` token budget until it completes, after which the next request from `_waiting` is taken.

The cost is that a second long prompt arriving while one is being chunked waits in `_waiting` until the current chunked request finishes. For typical system-prompt lengths (256–1024 tokens) and chunk sizes (512), this delay is at most one chunk round — a few milliseconds.

---

## Connection to prefill_batch

After `adder.build()` returns, the scheduler calls `self.model_runner.prefill_batch(prefill_batch)`. Each request in the returned list has `fill_ids`, `extend_input_len`, and `kv_committed_len` correctly set. After `prefill_batch` returns, the scheduler copies `adder.new_chunked_req` to `self._chunked_req` if a new chunked request was started, then routes each request based on its updated status. Section 03 explains how `PrefillAdder` computes the token budget, and section 04 explains what `kv_committed_len` and `extend_input_len` drive inside `prefill_batch`.
