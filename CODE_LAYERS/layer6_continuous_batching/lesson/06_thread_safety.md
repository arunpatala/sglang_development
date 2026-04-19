# 06 — Thread Safety

GPU computation is synchronous and blocking — it cannot be interleaved with asyncio's cooperative switching. The scheduler therefore runs on a dedicated background thread, leaving the asyncio event loop free to handle incoming HTTP connections without stalling. That design choice makes thread safety a hard requirement: the two threads must communicate without corrupting each other's state. Layer 6 enforces this through exactly two primitives: `queue.Queue` for passing requests from the HTTP handler to the scheduler, and `loop.call_soon_threadsafe` for passing results back.

Layer 6 is the first layer in this series to use more than one thread. The FastAPI server runs an asyncio event loop on the main thread, handling HTTP connections with coroutine-based concurrency. The scheduler runs its `while True` loop on a background daemon thread, driving GPU computation. These two threads share exactly two pieces of state: `queue.Queue` and `asyncio.Future`. The design ensures they never share anything else.

---

## Two Concurrency Models, One Process

The asyncio event loop and the background scheduler thread represent two fundamentally different concurrency models. Asyncio is cooperative: coroutines voluntarily yield at `await` points, and the event loop runs only one coroutine at a time. The scheduler is a blocking loop: it calls GPU kernels synchronously and never yields. This is intentional — GPU computation cannot be interleaved with asyncio's cooperative switching, so it runs on its own thread where it is free to block.

The boundary between these two worlds must be carefully managed. Calling asyncio objects from the scheduler thread, or calling blocking GPU operations from the asyncio thread, would cause either deadlocks or corrupted event loop state. Layer 6 enforces the boundary through two primitives: `queue.Queue` for one-way communication from asyncio to scheduler, and `loop.call_soon_threadsafe` for one-way communication from scheduler to asyncio.

---

## `queue.Queue`: From Asyncio to Scheduler

```python
# In Scheduler.__init__
self._waiting: queue.Queue[Req] = queue.Queue()

# In Scheduler.add_request (called by the asyncio thread)
def add_request(self, req: Req) -> None:
    self._waiting.put(req)

# In Scheduler.run (called by the scheduler thread)
req = self._waiting.get_nowait()
```

`queue.Queue` is part of Python's standard library and is explicitly documented as thread-safe. It uses an internal `threading.Lock` to guard the underlying deque. `put` can be called from any thread; `get_nowait` can be called from any thread; they will never corrupt each other. No explicit lock, no `asyncio.Queue`, no shared list — just the one data structure designed for exactly this cross-thread producer-consumer pattern.

The asyncio thread (the HTTP handler) calls `add_request` which calls `put`. This is the only method the asyncio side ever calls on the scheduler. The scheduler thread calls `get_nowait` in its drain loop. They never call each other's methods; they communicate only through `_waiting`.

`_running` — the list of active decode requests — is only ever touched by the scheduler thread. The asyncio thread never reads it directly. (The `/health` endpoint does read `len(scheduler._running)`, but Python integer reads are atomic on CPython due to the GIL, so this is safe without additional locking.)

---

## `loop.call_soon_threadsafe`: From Scheduler to Asyncio

When the scheduler finishes a request, it needs to resolve the request's `asyncio.Future`. A `Future` belongs to an event loop and must only be set from within that loop's thread. Calling `future.set_result(value)` from the scheduler thread would raise a `RuntimeError` ("This event loop is already running") or silently corrupt the loop's state.

The correct mechanism is `loop.call_soon_threadsafe`:

```python
# In Scheduler._resolve (called by the scheduler thread)
def _resolve(self, req: Req) -> None:
    text   = self.model_runner.decode_output(req)
    result = {"text": text, "prompt_tokens": req.prompt_len, ...}
    self._loop.call_soon_threadsafe(req.future.set_result, result)
```

`call_soon_threadsafe(callback, *args)` is the one method that is safe to call on an event loop from outside its thread. It places a callback into the loop's thread-safe call queue, then writes to a self-pipe to wake the event loop if it is sleeping in `select`/`epoll`. On the next iteration of the event loop, the callback is dequeued and executed in the event loop's thread. At that point, `req.future.set_result(result)` runs in the correct thread, the `Future` transitions from pending to resolved, and `await future` in the HTTP handler unblocks.

The `loop` reference is passed to `Scheduler.run` at startup, captured as `self._loop`, and reused for every `_resolve` call. The event loop is created once by uvicorn before `startup()` runs, so the same loop instance is always available.

---

## The Full Threading Picture

```
Main thread (asyncio event loop)          Scheduler thread
────────────────────────────────          ─────────────────────────────────────
POST /v1/chat/completions
  tokenize prompt
  future = loop.create_future()
  req    = Req(..., future=future)
  scheduler.add_request(req)    ──put──→  _waiting.Queue
  result = await future         (suspend)
                                          get_nowait() ← _waiting.Queue
                                          model_runner.prefill(req)
                                          _running.append(req)
                                          model_runner.decode_step(_running)
                                          ...
                                          loop.call_soon_threadsafe(
                                            req.future.set_result, result)
                                    ↓
  (event loop wakes)
  future.set_result(result) ← callback runs in asyncio thread
  result = await future         (resume)
  return ChatResponse(**result)
```

The asyncio thread and the scheduler thread communicate in both directions, but through primitives that make the direction explicit and safe. `queue.Queue.put` (asyncio → scheduler) is lock-protected. `call_soon_threadsafe` (scheduler → asyncio) is thread-safe by design. No shared mutable state exists outside these two channels, so no lock contention, no race conditions, and no need for `asyncio.Lock` or `threading.Lock` in the application code.

---

## Why Not `asyncio.Queue`?

`asyncio.Queue` might seem like a natural choice — it is explicitly designed for passing data between coroutines. But it is not thread-safe: it assumes all access happens from within the event loop. Calling `asyncio.Queue.put_nowait` from the scheduler thread while the event loop runs on another thread would corrupt the queue's internal state. `queue.Queue` is the correct choice precisely because it is designed for cross-thread use, not for cross-coroutine use.

The same reasoning applies to `asyncio.Future.set_result`: it is coroutine-safe (callable from the event loop thread) but not thread-safe. `call_soon_threadsafe` is the bridge that moves the call back into the correct context.
