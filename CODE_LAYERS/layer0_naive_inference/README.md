# Layer 0 — Naive Inference Server

The absolute minimum. No KV cache. No batching. One request at a time.

## What is in this layer

| File | Purpose |
|---|---|
| `server.py` | FastAPI server, HuggingFace `model.generate()`, `use_cache=False` |
| `test_client.py` | Measures latency, shows head-of-line blocking |

## Run it

```bash
# Terminal 1 — start the server
cd CODE_LAYERS/layer0
/home/arun/PROJECTS/sglang_development/.conda/bin/python server.py

# Wait for: "Starting Layer 0 server on 0.0.0.0:8100"

# Terminal 2 — run the tests
/home/arun/PROJECTS/sglang_development/.conda/bin/python test_client.py
```

## What to observe

**Test 1 (sequential):** Each request takes roughly the same amount of time.
The server does all the work for one request, then starts the next.

**Test 2 (concurrent):** When you fire N requests at the same time, the wall-clock
latency of the last request is approximately `N × single_request_latency`. There is
no parallelism — requests queue up and wait.

## The one-line upgrade to Layer 1

Open `server.py` and find this line in the `generate()` function:

```python
use_cache=False,  # <-- the defining choice of Layer 0
```

Change it to:

```python
use_cache=True,
```

Re-run the server and test client. You will see latency drop noticeably,
especially for longer sequences. That is the KV cache at work — HuggingFace
builds `past_key_values` internally and stops recomputing K and V for past tokens.

This teaches you Lesson 1 Section 1.3: the KV cache is just saved K/V tensors.

## What Layer 1 will add (properly)

- Expose the KV cache lifecycle explicitly (create, reuse, discard)
- Show what `past_key_values` looks like in memory
- Measure memory usage with and without caching
- Implement a manual decode loop so you can see the cache being passed step by step

## Architecture diagram

```
Client (curl / test_client.py)
        │
        │  POST /generate
        ▼
  FastAPI route (synchronous)
        │
        │  tokenizer(prompt)  →  input_ids
        │
        │  model.generate(input_ids, use_cache=False)
        │    - runs N forward passes (one per output token)
        │    - each pass recomputes K,V for ALL tokens from scratch
        │
        │  tokenizer.decode(output_ids)  →  text
        ▼
  JSON response {"text": ..., "latency_ms": ...}
```

## What is missing (intentionally)

| Missing feature | Problem it causes | Fixed in |
|---|---|---|
| KV cache | Redundant K,V recomputation | Layer 1 |
| Async / concurrent handling | Head-of-line blocking | Layer 2 |
| Batching | GPU underutilization during decode | Layer 2 |
| Custom KV pool | Per-request memory waste | Layer 4 |
| Prefix caching | Repeated prompts recomputed | Layer 5 |
