# 07 — The Full Loop

The previous sections each explained one piece of the system in isolation. This section traces two requests — one short, one long — through the complete system from HTTP receipt to HTTP response, in the order events actually occur.

---

## Setup

Assume the server is running. The scheduler thread is spinning in its `while True` loop, `_waiting` is empty, `_running` is empty. Request A arrives first with a 12-token prompt and `max_new_tokens=5`. Request B arrives 10 ms later with a 30-token prompt and `max_new_tokens=100`.

---

## Step 1 — Request A Arrives (asyncio thread)

```python
# server.py — POST /v1/chat/completions for Request A
messages   = [{"role": "user", "content": "Hi"}]
prompt_str = tok.apply_chat_template(messages)
input_ids  = tok.encode(prompt_str)[0].tolist()   # 12 tokens

loop   = asyncio.get_event_loop()
future = loop.create_future()
req_A  = Req(rid="aa...", input_ids=input_ids, max_new_tokens=5, temperature=0.0, future=future)

scheduler.add_request(req_A)   # puts req_A into _waiting
result = await future           # HTTP handler suspends
```

`req_A.status = WAITING`. `req_A.t_arrive` is set to the current timestamp. The asyncio event loop is free to handle other HTTP requests.

---

## Step 2 — Scheduler Prefills Request A (scheduler thread)

On the next scheduler loop iteration, Step 1 drains `_waiting`:

```python
req = self._waiting.get_nowait()   # req_A
self.model_runner.prefill(req)
```

`prefill` runs a `[1, 12]` forward pass. The 28 attention layers populate `req_A.kv_cache` with K/V tensors of shape `[1, n_kv_heads, 12, head_dim]`. `logits[0, -1]` is sampled to produce the first output token — say, token ID 1234. `req_A.output_ids = [1234]`. `req_A.t_first_token` is recorded. Since 1234 is not EOS and `output_len=1 < max_new_tokens=5`, `req_A.status = RUNNING`.

The scheduler moves req_A to `_running`:
```python
self._running.append(req_A)
```

---

## Step 3 — Scheduler Decodes Request A Alone (scheduler thread)

Immediately after Step 1, Step 2 fires because `_running = [req_A]`. `decode_step([req_A])` runs:

- `kv_lens = [12]`, `max_kv = 12`
- `last_toks = [[1234]]` — shape `[1, 1]`
- `attn_mask = [[1, 1, ..., 1, 1]]` — shape `[1, 13]`, all ones (no padding needed for B=1 with no ragged lengths)
- `pos_ids = [[12]]` — shape `[1, 1]`, request A is at position 12
- `BatchedKVCache([req_A], 12)` is built; forward pass runs `[1, 1]` input
- `write_back()` grows `req_A.kv_cache` to length 13
- Next token sampled: say 5678. `req_A.output_ids = [1234, 5678]`. Not finished.

One more scheduler iteration: `req_A` is still alone in `_running`, `output_len=2`. The process repeats. After 5 decode steps, `req_A.output_len = 5 = max_new_tokens`. On that step, `req_A.status = FINISHED` and `req_A.t_finish` is recorded. `decode_step` returns `[req_A]`.

---

## Step 4 — Request A is Resolved (scheduler thread → asyncio thread)

```python
for req in newly_finished:   # [req_A]
    self._resolve(req)
```

`_resolve` decodes `req_A.output_ids` to a string, builds the result dict, and calls:

```python
self._loop.call_soon_threadsafe(req_A.future.set_result, result)
```

The event loop schedules `future.set_result(result)` for the next asyncio tick. `req_A` is removed from `_running`. `_running = []`.

On the asyncio thread: `await future` unblocks, `result` is returned to the HTTP handler, `ChatResponse(**result)` is assembled and sent. The HTTP connection for Request A closes. Total wall time: `req_A.latency_ms`.

---

## Step 5 — Request B Arrives While A Is Decoding (concurrent scenario)

In practice, Request B arrives 10 ms after A — while A is still in its decode loop. The sequence is:

1. **t=0 ms**: A arrives, scheduler prefills A (say, takes 5 ms). A enters `_running`.
2. **t=5 ms**: Scheduler starts decode steps for A.
3. **t=10 ms**: B arrives. `scheduler.add_request(req_B)` puts B in `_waiting`.
4. **t=10 ms (next scheduler loop iteration)**: Step 1 sees `_waiting` non-empty and `len(_running) = 1 < max_running_reqs`. B is dequeued and prefilled — a `[1, 30]` forward pass. B's `kv_cache` is populated to length 30. B enters `_running`. `_running = [req_A, req_B]`.
5. **t=~12 ms**: Step 2 fires for `_running = [req_A, req_B]`. `decode_step([req_A, req_B])` runs:
   - `kv_lens = [kv_len_A, 30]`. If A has had 3 decode steps, its KV length is `12 + 3 = 15`. `max_kv = 30`.
   - `attn_mask` row 0: 15 zeros on the left, then ones. Row 1: all ones (B is at `max_kv`).
   - `pos_ids = [[15], [30]]`.
   - `BatchedKVCache` pads A's cache to length 30 with zeros on the left, stacks with B's cache.
   - Forward pass: `[2, 1]` input. A and B are processed together.

---

## Step 6 — Request A Finishes, B Continues

On A's 5th decode step (the step that produces its 5th output token), `req_A.status = FINISHED`. `decode_step` returns `[req_A]`. The scheduler resolves A and removes it from `_running`. `_running = [req_B]`. B continues decoding alone, at whatever position it has reached.

Request B's HTTP handler is still suspended on `await req_B.future`. It will unblock only when B's `max_new_tokens=100` limit is reached or when B emits EOS — potentially seconds later. Request A's handler already returned. There was no head-of-line blocking: A was not waiting for B, and B did not hold up A.

---

## The Key Invariant

At every scheduler loop iteration, the state is fully consistent. `_running` contains exactly the requests with `status == RUNNING`. Each has its own `kv_cache` representing exactly its history. `decode_step` builds a clean `BatchedKVCache` from the current snapshot, runs one forward pass, calls `write_back`, and updates each request independently. The scheduler loop is deterministic and re-entrant: it does not matter how many requests are in `_running`, how long they have been there, or whether some arrived three iterations ago and others this iteration. The same code path handles all cases.
