# 02 — The Request

```python
@dataclass
class Req:
    rid:            str
    input_ids:      List[int]
    max_new_tokens: int
    temperature:    float
    future:         asyncio.Future

    status:     ReqStatus = ReqStatus.WAITING
    output_ids: List[int] = field(default_factory=list)
    kv_cache:   Optional[object] = None      # PerReqKVCache, set after prefill

    t_arrive:      float = field(default_factory=time.perf_counter)
    t_first_token: float = 0.0
    t_finish:      float = 0.0
```

In Layer 5, request state was scattered across local variables in `generate_batch`: `input_ids` in a tensor, `generated` in a list of lists, `finished` in a boolean mask. All of it was bound to a single call frame that lasted until the entire batch completed. In Layer 6, a `Req` object owns exactly one request's state from the moment the HTTP handler creates it to the moment the scheduler resolves it. It can be queued, moved between data structures, and discarded without interfering with any other request.

---

## Identity and Input

`rid` is a random UUID hex string. It appears in scheduler log lines — `rid={req.rid[:8]}` — so that individual requests can be tracked through prefill, decode steps, and resolution without reading GPU tensors. `input_ids` is a plain Python list of integers: the tokenized prompt, already formatted with the chat template by the server before `Req` is constructed. The scheduler and model runner never call the tokenizer again.

`max_new_tokens` and `temperature` are per-request generation parameters. In Layer 5, these were method arguments passed into `generate_batch` and shared across all B requests in the call. In Layer 6, each `Req` carries its own, so two simultaneously running requests can have different token budgets and sampling temperatures without any interference.

---

## The Async Bridge: `future`

`future` is an `asyncio.Future` created in the FastAPI handler before the request is enqueued:

```python
# server.py — in the POST /v1/chat/completions handler
loop   = asyncio.get_event_loop()
future = loop.create_future()

req = Req(rid=..., input_ids=..., ..., future=future)
scheduler.add_request(req)

result = await future    # HTTP handler suspends here
return ChatResponse(**result)
```

The handler suspends at `await future` and yields the event loop back to FastAPI — other HTTP requests can be handled while this one waits. When the scheduler later calls `loop.call_soon_threadsafe(req.future.set_result, result)`, the event loop picks up the callback, the `future` transitions from pending to resolved, and the suspended `await` unblocks, returning the result dict to the HTTP response. Section 06 covers the thread-safety mechanics of this bridge in detail.

---

## Mutable State

`status` starts as `ReqStatus.WAITING` (the enum has three values: `WAITING`, `RUNNING`, `FINISHED`). The scheduler updates it to `RUNNING` when `prefill` completes successfully, and to `FINISHED` when EOS is emitted or `max_new_tokens` is reached. The scheduler loop uses this field to decide which requests belong in `_running` and which should be evicted:

```python
self._running = [r for r in self._running if r.status == ReqStatus.RUNNING]
```

`output_ids` is a plain Python list that grows by one element per decode step. Both `prefill` and `decode_step` call `req.output_ids.append(next_tok)`. The scheduler reads `req.output_len` (a property that returns `len(self.output_ids)`) to check the token budget. `decode_output` reads `output_ids` at resolution time to decode the tokens to a string.

`kv_cache` starts as `None` and is assigned a `PerReqKVCache` instance at the end of `prefill`. It holds all 28 layers' accumulated key and value tensors for this request's full history. Section 03 covers `PerReqKVCache` in depth.

---

## Timing Properties

Three timestamps track the request's lifecycle. `t_arrive` is set automatically by `field(default_factory=time.perf_counter)` at construction time — no explicit assignment required. `t_first_token` is set by `prefill` immediately after the first token is sampled. `t_finish` is set by either `prefill` (if EOS occurs on the first token) or `decode_step` (when EOS is emitted mid-decode).

Three derived properties compute the metrics the HTTP response carries:

```python
@property
def ttft_ms(self) -> float:
    return (self.t_first_token - t_arrive) * 1000

@property
def latency_ms(self) -> float:
    return (self.t_finish - t_arrive) * 1000

@property
def prompt_len(self) -> int:
    return len(self.input_ids)
```

TTFT (time to first token) is the wall time from HTTP receipt to first generated token, covering queue wait plus prefill. Unlike Layer 5's TTFT — which measured only the single batched prefill and was shared across all B requests — each `Req` has its own TTFT that reflects the actual wait experienced by that specific request. A request that arrives to a busy queue will have a higher TTFT than one that is prefilled immediately.
