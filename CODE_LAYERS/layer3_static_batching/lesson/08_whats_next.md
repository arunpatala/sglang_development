# 08 — What Comes Next

Layer 3 makes inference dramatically faster in aggregate. Processing B requests in one forward pass pushes GPU utilisation from 5% to 80%, and total output throughput more than doubles on the benchmark. `kv_cache.py`, `tokenizer.py`, and the model weights are unchanged. The pattern holds: one capability was added, a measured improvement resulted, and the rest of the system was untouched.

But the benchmark also makes two costs concrete and unavoidable. TTFT at B=16 is 1306 ms — 45× higher than at B=1. And every request in a batch waits for the slowest one before any result is returned. These are not implementation oversights. They are the direct consequences of assembling a fixed-size batch before computation begins and holding it fixed until every request finishes.

---

## Head-of-Line Blocking, Precisely Stated

The core problem is that the static batch is a commitment. Once `generate_batch` is called with B requests, those B requests are bound together for the duration of their decode loop. A request that generates 5 tokens and emits EOS on step 5 still occupies a row in the `[B, 1]` decode tensor on steps 6 through 127, contributing pad tokens and consuming KV cache memory, while waiting for other requests to finish. Its result is not returned to the caller until `finished.all()` is `True`.

If request 0 in a batch of 16 generates 5 tokens and request 15 generates 128 tokens, request 0 waits 123 extra decode steps to have its result returned — 123 × TPOT ≈ 4 seconds of additional latency on top of the 5 decode steps it actually needed. In an interactive serving system, this is unacceptable tail latency.

---

## What Continuous Batching Changes

Continuous batching solves head-of-line blocking by making the batch dynamic. Instead of assembling B requests, running prefill on all of them, and looping until all finish, a continuous batching scheduler manages a set of in-flight requests that changes on every decode step. When a request finishes, it is removed from the set immediately and its result is returned. A new request from the waiting queue is selected, its prefill is run, and it joins the set for the next decode step.

The decode loop itself stays structurally similar: on every step, a forward pass runs over the currently active requests. The difference is that the set of active requests is not fixed — it fluctuates as requests enter and exit. A short request exits after 5 steps; a new request enters immediately, bringing fresh GPU work rather than idle padding rows.

The implementation challenge is mixing prefill and decode in the same forward call. When a new request joins mid-stream, its prefill — processing the full prompt — must happen before it can participate in decode steps. Continuous batching systems handle this either by running prefill in a separate pass before the next decode step (chunked prefill) or by inserting prefill tokens into the decode batch directly (prefill-decode fusion). Either approach requires the scheduler to know, for each token in the current batch, whether it is a prefill token or a decode token — information that did not need to be tracked in static batching.

---

## What Files Change

The model forward call does not change. `kv_cache.py` does not change. `tokenizer.py` does not change. The change is in the scheduling logic — the code that decides which requests enter the next forward pass, in what shape, and in what order.

In SGLang, this is the `Scheduler` class. It maintains a queue of waiting requests, a set of running requests, a memory budget for KV caches, and the logic to preempt, resume, and reorder requests based on available memory and throughput targets. The `ModelRunner` receives a batch description from the scheduler on every step and executes the forward pass without knowing or caring how the batch was assembled.

The practical lesson from Layer 3 is that the GPU utilisation problem and the head-of-line blocking problem are separable. Static batching solves the first without addressing the second. Continuous batching solves both — at the cost of a more complex scheduler. The next layer isolates and implements that scheduler, leaving the model, the cache, and the tokenizer exactly as they are now.
