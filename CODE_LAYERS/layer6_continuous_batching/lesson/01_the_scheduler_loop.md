# 01 — The Scheduler Loop

## From Layer 5 to Layer 6

In Layer 5, a single call to `generate_batch` drove the entire lifecycle of B requests. It prefilled all prompts together, ran a decode loop until the last request finished, and returned a list of result dicts. No request could leave the batch early:

```python
# Layer 5 — one blocking call per batch; all requests exit together
kv = KVCache()
logits = self.model(input_ids, attention_mask=attention_mask,
                    kv_cache=kv, position_ids=prefill_pos)
next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]

for _ in range(max_new_tokens - 1):
    if finished.all():
        break
    logits = self.model(current, attention_mask=attention_mask,
                        kv_cache=kv, position_ids=decode_pos)
    finished = finished | (next_tokens == self.eos_id)
```

In Layer 6, there is no `generate_batch`. The server enqueues each request and awaits an `asyncio.Future`. The scheduler loop — running on a background thread — drives prefill and decode independently:

```python
# Layer 6 — server: one request at a time, immediately released
req = Req(rid=uuid4().hex, input_ids=input_ids, ..., future=loop.create_future())
scheduler.add_request(req)
result = await req.future   # returns as soon as this request finishes
```

A request that generates 5 tokens has its result available after 5 decode iterations. It does not wait for a co-running request that needs 500.

---

## The Scheduler's Three-Step Loop

The full `Scheduler.run` method:

```python
def run(self, loop: asyncio.AbstractEventLoop) -> None:
    self._loop = loop
    while True:
        did_work = False

        # ── Step 1: Prefill new requests ──────────────────────────
        while (
            not self._waiting.empty()
            and len(self._running) < self.max_running_reqs
        ):
            req = self._waiting.get_nowait()
            self.model_runner.prefill(req)
            self._n_prefilled += 1
            did_work = True

            if req.status == ReqStatus.FINISHED:
                self._resolve(req)
            else:
                self._running.append(req)

        # ── Step 2: Decode step for all running requests ──────────
        if self._running:
            newly_finished = self.model_runner.decode_step(self._running)
            did_work = True

            for req in newly_finished:
                self._resolve(req)

            self._running = [
                r for r in self._running if r.status == ReqStatus.RUNNING
            ]

        # ── Step 3: Idle ──────────────────────────────────────────
        if not did_work:
            time.sleep(IDLE_SLEEP_S)
```

**Step 1** drains the waiting queue up to `max_running_reqs` capacity. Each request gets its own B=1 prefill — no grouping, no waiting for a full batch. After prefill, if the model produced EOS as the very first token, the request transitions directly to `FINISHED` and is resolved immediately without ever entering the decode batch. Otherwise it is appended to `_running`.

**Step 2** runs one batched decode step across all currently running requests. `decode_step` returns the list of requests that just emitted EOS. Each is resolved — its `asyncio.Future` is set — and removed from `_running`. The remaining requests continue in the next iteration.

**Step 3** sleeps for 1 ms when both queues are empty. Without this, the loop would spin at full CPU frequency between requests, wasting a core.

---

## Dynamic Batch Size

`_running` has no fixed size. It grows every time Step 1 moves a request from `_waiting` to `_running`, and shrinks every time Step 2 evicts a finished request. The only ceiling is `max_running_reqs`, which caps GPU memory usage.

This is the structural difference from static batching. In Layer 5, the batch size was decided once at the start of `generate_batch` and remained fixed until the last request finished. In Layer 6, the batch size at each decode step depends on how many requests happen to be active at that moment — which is determined by the mix of recent arrivals and recent completions. A system with highly variable request lengths will see the batch size oscillate; a system with uniform lengths will see it stabilise near `max_running_reqs`.

---

## `add_request` and `_resolve`

`add_request` is the only method the asyncio thread calls on the scheduler:

```python
def add_request(self, req: Req) -> None:
    self._waiting.put(req)
    logger.debug(f"queued rid={req.rid[:8]} prompt_len={req.prompt_len}")
```

`queue.Queue.put` is thread-safe. The asyncio thread calls it from within the FastAPI handler; the scheduler thread calls `get_nowait` from its drain loop. No explicit lock is needed. Section 06 explains the full thread-safety design.

`_resolve` is the mirror: it builds the result dict and posts the `Future` resolution back to the asyncio thread:

```python
def _resolve(self, req: Req) -> None:
    text = self.model_runner.decode_output(req)
    result = {
        "text":              text,
        "prompt_tokens":     req.prompt_len,
        "completion_tokens": req.output_len,
        "ttft_ms":           round(req.ttft_ms, 1),
        "latency_ms":        round(req.latency_ms, 1),
    }
    self._loop.call_soon_threadsafe(req.future.set_result, result)
```

`loop.call_soon_threadsafe` schedules the `set_result` callback on the asyncio event loop. The HTTP handler's `await req.future` unblocks on the next event loop tick. The `Req` fields that supply these values — `output_ids`, `t_arrive`, `t_first_token`, `t_finish` — are covered in section 02.
