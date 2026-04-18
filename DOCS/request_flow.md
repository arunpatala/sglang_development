# SGLang Request Flow

A step-by-step trace of a single `/v1/chat/completions` request from the moment the client
sends it until the response is returned. Each step names the exact file and function.

---

## The Cast of Processes

```
┌──────────────────────────────────────────────────────────────────────┐
│  Process 1: HTTP Server  (same OS process as TokenizerManager)       │
│    FastAPI + Uvicorn async event loop                                │
│    http_server.py  ←→  serving_chat.py  ←→  tokenizer_manager.py    │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  ZeroMQ PUSH  (tokenized request)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Process 2: Scheduler  (one per tensor-parallel group)               │
│    Tight synchronous loop                                            │
│    scheduler.py  →  tp_worker.py  →  model_runner.py  (GPU)         │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  ZeroMQ PUSH  (token IDs)
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Process 3: DetokenizerManager                                       │
│    detokenizer_manager.py                                            │
└─────────────────────────────┬────────────────────────────────────────┘
                              │  ZeroMQ PUSH  (decoded text)
                              ▼
                    Back to Process 1 → HTTP response
```

---

## Step-by-Step Flow

---

### Step 1 — Client Sends HTTP POST

```
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "Qwen/Qwen3-0.6B",
  "messages": [{"role": "user", "content": "What is 2+2?"}],
  "max_tokens": 64,
  "stream": false
}
```

Uvicorn receives the TCP connection and hands the raw bytes to FastAPI.
FastAPI parses the JSON body into a `ChatCompletionRequest` Pydantic model
(`entrypoints/openai/protocol.py`), validating field types and defaults.

---

### Step 2 — FastAPI Route Handler

**File:** `srt/entrypoints/http_server.py` → `openai_v1_chat_completions()`

This is the first SGLang code that runs:

```python
@app.post("/v1/chat/completions", ...)
async def openai_v1_chat_completions(request: ChatCompletionRequest, raw_request: Request):
    global _request_counter
    _request_counter += 1
    # logs a preview of the last message
    # checks max prompt length guard (change #6)
    # checks in-memory response cache if --enable-response-cache (change #9)

    result = await raw_request.app.state.openai_serving_chat.handle_request(
        request, raw_request
    )
    # injects latency_ms into result.usage (change #8)
    return result
```

The handler itself is thin — it delegates immediately to `OpenAIServingChat`.

---

### Step 3 — OpenAIServingChat: Translate OpenAI Format → Internal Format

**File:** `srt/entrypoints/openai/serving_chat.py` → `handle_request()` → `_convert_to_internal_request()`

This is where the OpenAI message format is translated into SGLang's internal representation.

**3a. Apply chat template.**
`_process_messages()` takes the list of `{"role": ..., "content": ...}` dicts and runs them
through the HuggingFace tokenizer's `apply_chat_template`. For Qwen3 this produces a
formatted string like:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is 2+2?<|im_end|>
<|im_start|>assistant
```

The result is stored as `processed_messages.prompt_ids` (already as token IDs if the
template returns them, or as a string to be tokenized next).

**3b. Build sampling parameters.**
`request.to_sampling_params()` converts the OpenAI fields
(`temperature`, `top_p`, `max_tokens`, `stop`, `stream`, etc.) into SGLang's internal
`SamplingParams` object (`srt/sampling/sampling_params.py`).

**3c. Create `GenerateReqInput`.**
All of the above is packaged into a `GenerateReqInput` dataclass
(`srt/managers/io_struct.py`) — this is the internal request format that will travel
through the rest of the pipeline.

```python
adapted_request = GenerateReqInput(
    input_ids=processed_messages.prompt_ids,   # token ID list
    sampling_params=sampling_params,
    stream=request.stream,
    rid=request.rid,                           # UUID for tracking
    ...
)
```

---

### Step 4 — TokenizerManager: Tokenize and Send

**File:** `srt/managers/tokenizer_manager.py` → `generate_request()` → `_tokenize_one_request()` → `_send_one_request()`

The `GenerateReqInput` is passed to the `TokenizerManager` (which lives in the same
process). If the prompt arrived as text (not already token IDs), tokenization happens here:

```python
input_ids, _ = await self._tokenize_texts(input_text)
```

`_tokenize_texts()` calls the HuggingFace tokenizer and returns a flat `List[int]`.

A `ReqState` object is created and stored in `self._req_dict[rid]` — this is how the
TokenizerManager keeps track of in-flight requests and routes responses back to the
correct HTTP handler.

The tokenized request is then pushed over ZeroMQ to the Scheduler:

```python
self.send_to_scheduler.send_pyobj(tokenized_obj)  # ZMQ PUSH
```

The calling coroutine now suspends on an `asyncio.Event`, waiting for results.

---

### Step 5 — Scheduler: Receive and Queue

**File:** `srt/managers/scheduler.py` → `event_loop_overlap()` (or `event_loop_normal()`) → `recv_requests()` → `process_input_requests()`

The Scheduler is a separate OS process running a tight `while True` loop.

```python
while True:
    recv_reqs = self.recv_requests()          # reads ZMQ socket
    self.process_input_requests(recv_reqs)    # adds to waiting_queue
    batch = self.get_next_batch_to_run()      # schedule policy
    batch_result = self.run_batch(batch)      # GPU forward pass
    self.process_batch_result(batch, result)  # handle output, send to detokenizer
```

`process_input_requests()` deserializes the pickled `TokenizedGenerateReqInput`,
creates a `Req` object (defined in `schedule_batch.py`), and appends it to
`self.waiting_queue`.

---

### Step 6 — Scheduler: Schedule Policy — Prefix Matching

**File:** `srt/managers/schedule_policy.py` → `SchedulePolicy.get_next_batch_to_run()`

On the next loop iteration (or the same one if the queue was already being processed),
the schedule policy runs. The default policy is **Longest Prefix Match (LPM)**:

1. Sort waiting requests by how many of their leading token IDs match something already
   in the `RadixCache`.
2. Prefer requests with more cached prefix — they will need to compute fewer tokens.
3. Also respect `max_running_requests` and available KV cache slots.

The selected requests are promoted from `waiting_queue` into the `running_batch`.

---

### Step 7 — Scheduler: KV Cache Allocation

**File:** `srt/managers/schedule_batch.py` → `alloc_req_slots()`, `srt/mem_cache/radix_cache.py` → `match_prefix()`

For each newly added request:

**7a. Prefix lookup.**
`RadixCache.match_prefix(token_ids)` walks the radix tree and returns:
- `prefix_len`: how many tokens from the front already have KV pairs computed and cached
- `cached_kvs`: the GPU slot indices for those tokens (free to reuse!)

**7b. New slot allocation.**
The remaining tokens (`input_ids[prefix_len:]`) need new KV cache slots.
`TokenToKVPool.alloc(n)` allocates `n` contiguous-or-scattered slots from the free pool.
If there are not enough free slots, the scheduler evicts the least-recently-used unlocked
RadixCache nodes to reclaim memory.

**7c. Build the slot table.**
`ReqToTokenPool` is updated: the request's slot list is now `[cached_slots..., new_slots...]`.
The forward pass will write new KV pairs into the new slots and read old ones from the
cached slots.

---

### Step 8 — Scheduler: Assemble the Batch

**File:** `srt/managers/schedule_batch.py` → `ScheduleBatch`, `srt/managers/tp_worker.py`

A `ScheduleBatch` is assembled containing all requests to run this step.

The batch is tagged with a `ForwardMode`:
- `EXTEND` — for our new request (has prompt tokens to process)
- `DECODE` — for any already-running requests generating their next token
- `MIXED` — if chunked prefill is enabled and both are in the same batch

The `TpModelWorker` (`tp_worker.py`) receives the `ScheduleBatch` and converts it to a
`ModelWorkerBatch` — a leaner projection that strips scheduling metadata and converts
Python lists into numpy/tensor arrays.

---

### Step 9 — ModelRunner: Forward Pass on GPU

**File:** `srt/model_executor/model_runner.py` → `forward()`, `srt/model_executor/forward_batch_info.py`

`ModelWorkerBatch` is converted to a `ForwardBatch` — all GPU tensors:

| Tensor | Contents |
|---|---|
| `input_ids` | Token IDs being processed this step |
| `positions` | Absolute positions (for RoPE embeddings) |
| `req_to_token` | Maps each request to its KV slot indices |
| `out_cache_loc` | Where to write new KV pairs |

**For EXTEND (prefill):**
The full `forward()` is run in eager mode (normal Python dispatch). All transformer
layers execute sequentially. Each attention layer:
1. Computes Q, K, V projections for the new tokens
2. Reads K/V from cached slots (for the reused prefix)
3. Computes attention between new Q and all K/V (both cached and new)
4. Writes new K/V into `out_cache_loc`

This is compute-heavy and proportional to prompt length squared.

**For DECODE:**
If the batch size matches a pre-captured CUDA graph, the `CudaGraphRunner` replays it:

```python
cudaGraphLaunch(self.graphs[batch_size])
```

The entire 32-layer forward pass executes as a single GPU API call with near-zero CPU
overhead. This is 10-100x faster dispatch than eager mode.

After the final layer, the logits processor computes vocabulary-sized logit vectors.

---

### Step 10 — Sampling: Pick the Next Token

**File:** `srt/sampling/sampling_batch_info.py`, `srt/layers/logits_processor.py`

The logits for the last token position of each sequence go through the sampling pipeline:

1. **Penalties** — repetition, frequency, and presence penalties are applied by
   modifying logits based on what has already been generated.
2. **Temperature scaling** — `logits /= temperature` (default `1.0` now that we reverted change #7).
3. **Top-k** — zero out all but the top-k logits.
4. **Top-p** — zero out tokens below the nucleus threshold.
5. **Softmax + multinomial sample** — convert to probabilities and draw one token per sequence.

The result is a tensor of shape `[batch_size]` containing one new token ID per request.
This is returned to the Scheduler.

---

### Step 11 — Scheduler: Process Results

**File:** `srt/managers/scheduler.py` → `process_batch_result()`, `srt/managers/scheduler_output_processor_mixin.py`

For each request in the batch:

1. **Append token** to `req.output_ids`.
2. **Check stop conditions:**
   - Is this the EOS token?
   - Has `max_tokens` been reached?
   - Does the decoded text contain a user-specified stop string?
3. **Insert into RadixCache** — the newly computed KV pairs for this step are inserted
   into the radix tree so future requests with the same prefix can reuse them:
   ```python
   tree_cache.insert(req.prefix_indices + req.output_ids, kv_slot_indices)
   ```
4. **If finished:** remove from `running_batch`, send result to DetokenizerManager.
5. **If not finished:** keep in `running_batch` — it will decode the next token on the
   next loop iteration. Go to Step 9.

The output is packed into a `BatchTokenIDOutput` message and pushed over ZeroMQ:

```python
self.send_to_detokenizer.send_pyobj(BatchTokenIDOutput(...))
```

---

### Step 12 — DetokenizerManager: Token IDs → Text

**File:** `srt/managers/detokenizer_manager.py` → `event_loop()` → `handle_batch_token_id_out()` → `_decode_batch_token_id_output()`

The DetokenizerManager's `event_loop()` receives the `BatchTokenIDOutput`:

```python
def event_loop(self):
    while True:
        recv_obj = self.recv_from_scheduler.recv_pyobj()
        output = self._request_dispatcher(recv_obj)
        self.send_to_tokenizer.send_pyobj(output)
```

For each request in the batch, `_decode_batch_token_id_output()`:

1. Looks up the `DecodeStatus` for this `rid` (or creates one).
2. Appends new token IDs to the accumulated sequence.
3. Calls `tokenizer.batch_decode()` on the new tokens.
4. Handles the surrogate-offset trick: the "surrogate" window is a slightly larger
   context that ensures multi-byte UTF-8 characters are decoded completely before being
   sent — this avoids garbled mid-character streaming.
5. Trims any stop strings from the end if the request finished.

The decoded text delta is packaged into a `BatchStrOutput` and pushed over ZeroMQ
back to the TokenizerManager:

```python
self.send_to_tokenizer.send_pyobj(BatchStrOutput(...))
```

---

### Step 13 — TokenizerManager: Wake Up the HTTP Handler

**File:** `srt/managers/tokenizer_manager.py` → `handle_loop()` → `_wait_one_response()`

The TokenizerManager's `handle_loop()` coroutine receives `BatchStrOutput` from
DetokenizerManager. It looks up the `rid` in `self._req_dict`, appends the new text
to `req_state.out_list`, and sets `req_state.event`:

```python
req_state.out_list.append(out)
req_state.event.set()
```

The HTTP handler was suspended on `asyncio.Event.wait()`. Setting the event wakes it up.

**If streaming** (`stream=true`): the handler immediately yields the text delta as a
Server-Sent Event chunk and goes back to waiting for the next token.

**If not streaming** (`stream=false`): the handler waits until `req_state.finished` is
`True` (i.e., the final `BatchStrOutput` arrived), then proceeds to Step 14.

---

### Step 14 — OpenAIServingChat: Build the Response

**File:** `srt/entrypoints/openai/serving_chat.py` → `_create_chat_completion_response()`

The accumulated output text and metadata are wrapped into a `ChatCompletionResponse`
Pydantic model (defined in `entrypoints/openai/protocol.py`):

```python
ChatCompletionResponse(
    id="chatcmpl-abc123",
    object="chat.completion",
    created=<unix timestamp>,
    model="Qwen/Qwen3-0.6B",
    choices=[
        ChatCompletionResponseChoice(
            index=0,
            message=ChatMessage(role="assistant", content="4"),
            finish_reason="stop",
        )
    ],
    usage=UsageInfo(
        prompt_tokens=12,
        completion_tokens=1,
        total_tokens=13,
        latency_ms=142.7,   # injected by our change #8
    ),
)
```

---

### Step 15 — HTTP Response Returned

**File:** `srt/entrypoints/http_server.py` → `openai_v1_chat_completions()`

Back in our route handler, the result is returned. FastAPI serializes the Pydantic model
to JSON using `orjson` (faster than stdlib `json`). The response bytes are written to the
TCP socket and the client receives:

```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "choices": [{"message": {"role": "assistant", "content": "4"}, "finish_reason": "stop"}],
  "usage": {"prompt_tokens": 12, "completion_tokens": 1, "total_tokens": 13, "latency_ms": 142.7}
}
```

If `--enable-response-cache` is set, the result is stored in `_response_cache` before
returning, so an identical future request is served instantly from memory.

---

## Summary Timeline

```
Client
  │── POST /v1/chat/completions ──────────────────────────────────────► Step 1
                                                                         │
HTTP Server (Process 1)                                                  │
  ├── FastAPI parse → ChatCompletionRequest ──────────────────────────► Step 2
  ├── OpenAIServingChat: apply template, build GenerateReqInput ──────► Step 3
  ├── TokenizerManager: tokenize → List[int], create ReqState ────────► Step 4
  └── ZMQ PUSH (TokenizedGenerateReqInput) ───────────────────────────► Step 4
                                           │
Scheduler (Process 2)                      │
  ├── recv_requests(), add to waiting_queue ──────────────────────────► Step 5
  ├── schedule_policy: LPM, select batch ─────────────────────────────► Step 6
  ├── RadixCache.match_prefix(), alloc KV slots ──────────────────────► Step 7
  ├── Assemble ScheduleBatch → ForwardBatch ──────────────────────────► Step 8
  ├── ModelRunner.forward() — prefill on GPU ─────────────────────────► Step 9
  ├── Sampler: temp/top-p/top-k → token ID ───────────────────────────► Step 10
  ├── process_batch_result(): append token, check stop ───────────────► Step 11
  │   └── [not done? → back to Step 9, decode loop]
  └── ZMQ PUSH (BatchTokenIDOutput) ─────────────────────────────────► Step 11
                                   │
DetokenizerManager (Process 3)     │
  ├── recv, tokenizer.batch_decode() ─────────────────────────────────► Step 12
  └── ZMQ PUSH (BatchStrOutput) ─────────────────────────────────────► Step 12
                              │
HTTP Server (Process 1)       │
  ├── handle_loop wakes HTTP handler via asyncio.Event ───────────────► Step 13
  ├── OpenAIServingChat: assemble ChatCompletionResponse ─────────────► Step 14
  └── FastAPI → orjson serialize → TCP → Client ──────────────────────► Step 15
```

---

## Key Data Structures at Each Boundary

| Boundary | Object | File |
|---|---|---|
| Client → HTTP | `ChatCompletionRequest` (Pydantic) | `entrypoints/openai/protocol.py` |
| HTTP → TokenizerManager | `GenerateReqInput` | `managers/io_struct.py` |
| TokenizerManager → Scheduler | `TokenizedGenerateReqInput` (ZMQ) | `managers/io_struct.py` |
| Scheduler → GPU | `ForwardBatch` (GPU tensors) | `model_executor/forward_batch_info.py` |
| Scheduler → Detokenizer | `BatchTokenIDOutput` (ZMQ) | `managers/io_struct.py` |
| Detokenizer → TokenizerManager | `BatchStrOutput` (ZMQ) | `managers/io_struct.py` |
| TokenizerManager → HTTP | `ChatCompletionResponse` (Pydantic) | `entrypoints/openai/protocol.py` |
