# 03 — PrefillAdder and the Token Budget

`PrefillAdder` sits between the waiting queue and `prefill_batch`. It enforces two limits simultaneously: the decode batch must not exceed `max_running_reqs`, and the total new tokens entering the extend kernel each round must not exceed `max_prefill_tokens`. Understanding how these two limits interact explains why `PrefillAdder` sometimes admits a request larger than the remaining budget and sometimes defers it.

---

## The Two Stopping Conditions

```python
# scheduler.py — PrefillAdder.build() — the while loop in Case 2
while True:
    if self.running_count + len(batch) >= self.max_running_reqs:
        break
    if rem_tokens <= 0 and batch:
        break
    if self.waiting.empty():
        break

    req = self.waiting.queue[0]   # peek without removing

    if req.prompt_len > rem_tokens and batch:
        break   # next request doesn't fit; defer
    ...
```

The first `break` fires when the decode batch is full. `running_count` is the number of requests already in `_running` at the start of this iteration; `len(batch)` is the count of requests being prefilled this round. Together they must stay below `max_running_reqs`.

The second `break` fires when `rem_tokens <= 0` and at least one request is already in the batch. The `and batch` guard is critical: if the batch is empty, `rem_tokens` being zero would silently skip all requests without admitting any — the scheduler would spin doing nothing. The guard ensures at least one request is always admitted even if its length exceeds `max_prefill_tokens`.

The `prompt_len > rem_tokens and batch` line at the bottom of the loop is a soft budget check: if the next request does not fit within the remaining budget, defer it to the next round, but only if the batch already has something. If the batch is empty and the request is large, it is admitted anyway. This prevents indefinite starvation of a single large request that can never fit in a non-empty budget window.

---

## max_prefill_tokens and Its Purpose

```python
# scheduler.py — Scheduler.__init__
chunked_prefill_size: int = 0,      # 0 = no chunking
max_prefill_tokens: int = 4096,     # token budget per extend round
```

`max_prefill_tokens` caps the total number of tokens across all requests in one `prefill_batch` call. Its purpose is to bound the latency of the extend kernel so that decode requests do not wait more than a bounded time for each extend round to complete.

Consider `max_prefill_tokens = 2048` with three waiting requests of 600, 800, and 900 tokens. The first round admits the 600-token and 800-token requests (1400 total, within budget). The 900-token request is deferred — it would push the total to 2300. The second round (with 1400 tokens now in the decode batch) admits the 900-token request.

If `max_prefill_tokens` is not set (or set very large), a burst of waiting requests could all be admitted at once, creating a very large extend kernel that blocks the decode loop for a long time — essentially reintroducing the starvation problem that chunked prefill was designed to solve.

---

## The Chunked Request Path

When `chunked_prefill_size > 0` and the peeked request's `prompt_len > chunked_prefill_size`, the loop takes the request out of `_waiting` immediately and computes its first chunk:

```python
if self.chunked_prefill_size and req.prompt_len > self.chunked_prefill_size:
    req = self.waiting.get_nowait()
    chunk_end = min(self.chunked_prefill_size, req.prompt_len)
    req.fill_ids         = req.input_ids[:chunk_end]
    req.extend_input_len = chunk_end
    self.new_chunked_req = req
    batch.append(req)
    break   # only one chunked request per round
```

The `break` at the end is important: once a chunked request is started, no further requests are added to this round's batch. This ensures the chunked request gets its full `chunked_prefill_size` budget rather than competing with other requests for the same token window.

`new_chunked_req` is not immediately stored into `self._chunked_req` — that happens in `Scheduler.run` after `prefill_batch` returns and the request's status has been confirmed as `PREFILLING`. If `prefill_batch` somehow completes the entire request in one call (because the first chunk was also the last), the request transitions to `RUNNING` or `FINISHED` and `_chunked_req` is never set.

---

## The is_last_chunk Signal

How does `prefill_batch` know whether a request is on its last chunk? The `extend_input_len` set by `PrefillAdder` determines how many tokens are processed. After `prefill_batch` updates `kv_committed_len += extend_input_len`, the check is:

```python
# model_runner.py — after the extend forward pass
req.kv_committed_len += req.extend_input_len

if not req.is_last_chunk:
    req.status = ReqStatus.PREFILLING
    continue
# else: last chunk — sample first token, set RUNNING or FINISHED
```

`req.is_last_chunk` is a property:

```python
@property
def is_last_chunk(self) -> bool:
    return self.kv_committed_len >= self.prompt_len
```

After `kv_committed_len` is advanced by `extend_input_len`, if it equals `prompt_len`, all tokens have been processed. This is the last chunk. The scheduler then clears `_chunked_req`.

The interaction between `PrefillAdder` (which sets `extend_input_len`) and `is_last_chunk` (which reads `kv_committed_len` after the update) is the complete state machine. Section 04 shows what happens inside `prefill_batch` with these values.
