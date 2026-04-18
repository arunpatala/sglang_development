# SGLang Inference Engine — Architecture Deep Dive

A textbook-style guide to how SGLang turns an HTTP request into generated tokens.
Read this top-to-bottom for the full picture, or jump to any section as a reference.

---

## Table of Contents

1. [The Big Picture](#1-the-big-picture)
2. [Process Architecture](#2-process-architecture)
3. [The HTTP Entrypoint](#3-the-http-entrypoint)
4. [TokenizerManager — the async front-door](#4-tokenizermanager--the-async-front-door)
5. [IPC: How Processes Talk to Each Other](#5-ipc-how-processes-talk-to-each-other)
6. [The Scheduler — Heart of the Engine](#6-the-scheduler--heart-of-the-engine)
7. [Memory Management: KV Cache Pools](#7-memory-management-kv-cache-pools)
8. [RadixCache — Prefix Reuse](#8-radixcache--prefix-reuse)
9. [The Batch Pipeline: Three Levels of Batching](#9-the-batch-pipeline-three-levels-of-batching)
10. [ModelRunner — Running the Forward Pass](#10-modelrunner--running-the-forward-pass)
11. [Attention Backends](#11-attention-backends)
12. [CUDA Graph Optimization](#12-cuda-graph-optimization)
13. [Sampling](#13-sampling)
14. [DetokenizerManager — Turning Tokens Back to Text](#14-detokenizermanager--turning-tokens-back-to-text)
15. [The Full Request Lifecycle, End to End](#15-the-full-request-lifecycle-end-to-end)
16. [Key Files Reference](#16-key-files-reference)

---

## 1. The Big Picture

SGLang is a high-throughput LLM serving framework. Its core design principle is to keep the GPU saturated at all times while minimizing the work the CPU has to do on the critical path. To achieve this, it separates concerns across multiple OS processes, uses a radix-tree KV cache to reuse already-computed attention states across requests, and captures GPU work into CUDA graphs so that dispatch overhead nearly disappears during the decode phase.

At the highest level, every request travels through this pipeline:

```
Client (HTTP/OpenAI API)
        │
        ▼
  HTTP Server (FastAPI, async)
        │
        ▼
  TokenizerManager (async process)
   – applies chat template
   – tokenizes input text → token IDs
        │  ZeroMQ PUSH
        ▼
  Scheduler (sync event loop per GPU group)
   – manages waiting queue
   – matches token IDs against RadixCache (prefix reuse)
   – allocates KV cache slots
   – forms a batch and dispatches it to the GPU
        │
        ▼
  TpModelWorker → ModelRunner
   – runs the transformer forward pass on GPU
   – CUDA graph replay (decode) or eager kernel (prefill)
        │
        ▼
  Scheduler (receives sampled token IDs)
   – appends new token to the sequence
   – checks stop conditions
   – if not done, loops back; if done, sends output
        │  ZeroMQ PUSH
        ▼
  DetokenizerManager (async process)
   – converts token IDs → text strings
   – streams partial results back
        │  ZeroMQ PUSH
        ▼
  TokenizerManager → HTTP response (streaming or batch)
```

Each box above is either a separate OS process or a distinct thread, deliberately separated so that tokenization, scheduling, GPU execution, and detokenization can all overlap in time.

---

## 2. Process Architecture

When you run `python -m sglang.launch_server`, the main process spawns several child processes and then stays alive as a watchdog. Understanding which process does what is the first step to understanding how to modify the engine.

### The Three Core Processes

**TokenizerManager** lives in the same async event loop that serves HTTP requests. It holds the FastAPI `app` object, receives raw text from clients, applies the chat template (e.g., Qwen3's `<|im_start|>` format), and tokenizes it. After tokenization it sends a `TokenizedGenerateReqInput` object over ZeroMQ to the Scheduler and then suspends, waiting for output.

**Scheduler** is a synchronous Python process that runs a tight loop — poll for new requests, schedule a batch, send it to the GPU, receive results, repeat. It is the central coordinator. It owns the KV cache, the RadixCache prefix tree, the waiting queue, and the running queue. Only one Scheduler process exists per tensor-parallel group (one GPU or one set of GPUs that share a tensor-parallel rank).

**DetokenizerManager** is a lightweight async process. It receives batches of token IDs from the Scheduler, calls the HuggingFace tokenizer to convert them to strings, and sends partial or complete text back to the TokenizerManager so it can stream responses to the client.

### Why Separate Processes?

Python's GIL prevents true CPU parallelism within a single process. By placing each manager in its own process, SGLang gets:

- True overlap between tokenization (CPU) and GPU execution
- Separation of async I/O (TokenizerManager, DetokenizerManager) from the tight synchronous loop of the Scheduler
- Fault isolation: a crash in one process does not immediately kill the others

The tradeoff is serialization overhead on the ZeroMQ pipes, which is why the data structures crossing process boundaries (`io_struct.py`) are kept lean — they carry token IDs and metadata, not tensor data.

---

## 3. The HTTP Entrypoint

**File:** `srt/entrypoints/http_server.py`

The HTTP server is a FastAPI application. When `launch_server` starts up, it creates the FastAPI `app`, attaches a lifespan handler that initializes a `TokenizerManager` and related objects into `app.state`, then hands control to Uvicorn.

Every route handler is a thin adapter. The `/v1/chat/completions` route, for example, receives a Pydantic `ChatCompletionRequest`, converts it into SGLang's internal `GenerateReqInput` structure, and delegates to `TokenizerManager.generate`. The route itself does almost no work — it is just a translation layer between the OpenAI API format and SGLang's internal types.

The server is also where we added our custom endpoints in the earlier exercises (`/hello`, `/stats/requests`, `/v1/models` enrichment). These additions are safe precisely because this layer is only a translation layer — adding routes here does not touch the GPU path.

---

## 4. TokenizerManager — the Async Front-Door

**File:** `srt/managers/tokenizer_manager.py`

The `TokenizerManager` is an `asyncio`-based class that lives inside the HTTP server process. Its responsibilities are:

1. **Chat template application.** Given a list of messages in OpenAI format, it calls the tokenizer's `apply_chat_template` to produce a single flat string (or token IDs directly, for models that have a structured template like Qwen3's thinking mode).

2. **Tokenization.** The string is tokenized into a list of integer token IDs. This list is what actually travels downstream — the raw text is never sent to the Scheduler.

3. **Request tracking.** Each request gets a UUID (`rid`). The TokenizerManager maintains a dictionary of in-flight requests keyed by `rid`. When the Scheduler sends back outputs, the manager looks up the correct `asyncio.Event` or streaming queue to wake up the HTTP handler.

4. **Streaming.** For streaming responses, the TokenizerManager receives incremental `BatchTokenIDOutput` messages from the DetokenizerManager and forwards them to the HTTP response stream one chunk at a time.

The TokenizerManager never touches CUDA or the model weights. It is deliberately kept "thin" so that adding features here (like our response cache) is low-risk.

---

## 5. IPC: How Processes Talk to Each Other

**File:** `srt/utils/network.py`, ZeroMQ sockets configured in `scheduler.py::init_ipc_channels`

All inter-process communication in SGLang uses **ZeroMQ** (ZMQ), a high-performance message-passing library. ZMQ provides:

- **PUSH/PULL pairs** for one-directional pipelines (TokenizerManager → Scheduler, Scheduler → DetokenizerManager)
- **DEALER sockets** for RPC-style round-trip calls

The socket addresses are Unix domain socket paths (IPC paths), assembled from `PortArgs` which is generated at server startup. Because ZMQ operates over file descriptors rather than the network stack, latency is in the microseconds range for small messages.

Data crossing these pipes is serialized with `pickle`. The objects in `io_struct.py` define the schema: `TokenizedGenerateReqInput` carries token IDs, sampling parameters, and metadata to the Scheduler; `BatchTokenIDOutput` carries arrays of new token IDs back. These objects are intentionally small — only the indices needed for the GPU, never the tensors themselves.

---

## 6. The Scheduler — Heart of the Engine

**File:** `srt/managers/scheduler.py`

The Scheduler is the most complex component. It runs a synchronous event loop and is responsible for all decisions about what runs on the GPU next. Understanding it is the key to understanding SGLang's performance.

### The Main Loop

The Scheduler's main loop runs continuously:

1. **Poll for new requests** from the ZMQ socket. Newly arrived `TokenizedGenerateReqInput` objects are deserialized and placed in `self.waiting_queue`.

2. **Get the schedule.** `schedule_policy.py` examines `self.waiting_queue` and `self.running_batch` to decide which requests to promote into the running batch. The default policy is **Longest Prefix Match (LPM)**: among all waiting requests, prefer those that share the longest common prefix with already-cached KV entries in the RadixCache. This maximizes cache reuse.

3. **Allocate KV cache.** For each request being added to the batch, the scheduler calls `RadixCache.match_prefix` to find how many of its leading tokens already have computed KV entries. Those tokens get their KV cache "for free". The remaining tokens (the unmatched suffix) need new KV cache slots, which are allocated from `TokenToKVPool`.

4. **Run the model.** The scheduler calls `self.tp_worker.forward_batch_generation(batch)`. This is a synchronous call that blocks until the GPU has produced new token logits and the sampler has chosen the next token ID for every sequence in the batch.

5. **Process results.** New token IDs are appended to each sequence. Stop conditions are checked (EOS token, max length, stop strings). Finished requests are removed from the running batch and their results are sent to the DetokenizerManager.

6. **Loop.** Non-finished requests stay in the running batch and participate in the next decode step.

### Prefill vs. Decode

These two phases have very different compute characteristics and the Scheduler handles them differently.

In the **prefill** (or "extend") phase, a new request's prompt tokens are processed all at once. This is a large, compute-bound matrix multiplication: every query token attends to every key/value token. The amount of work is proportional to the square of the prompt length. Because the RadixCache may have already computed KV for the common prefix, SGLang only needs to prefill the *unmatched suffix* — this is the "extend" operation (`ForwardMode.EXTEND`).

In the **decode** phase, the model generates one new token per request per forward pass. The query is a single vector per sequence, attending to the growing KV cache. This is memory-bandwidth-bound rather than compute-bound. Many requests can be batched together here efficiently.

The scheduler continuously interleaves prefill and decode. A large prompt may be split across multiple steps using **chunked prefill** (`chunked_prefill_size`), which limits the number of prefill tokens processed per step so that decode requests are not starved.

### Preemption and Retract

If KV cache runs out, the scheduler can *retract* a decode request: it evicts its KV cache entries back into the RadixCache (or discards them if they have no prefix value), returns the request to the waiting queue, and continues. When resources free up, the request is prefilled again — potentially hitting the RadixCache this time if someone else ran the same prefix.

---

## 7. Memory Management: KV Cache Pools

**Files:** `srt/mem_cache/memory_pool.py`, `srt/mem_cache/allocator.py`

Managing GPU memory for the KV cache is one of the hardest parts of a serving system, and SGLang uses a two-level design.

### Level 1: ReqToTokenPool

`ReqToTokenPool` is a CPU-side index table. It maps each request ID to a list of *slot indices* — positions in the KV cache tensor where that request's tokens live. When a request needs a new token appended, a new slot is allocated and its index is added to this list.

### Level 2: TokenToKVPool (KVCache)

`KVCache` is a large pre-allocated GPU tensor, dimensioned to hold the KV pairs for `max_total_num_tokens` tokens across all layers and heads. It is allocated once at startup based on a fraction of available GPU memory (`mem_fraction_static`, typically around 0.88 by default).

The key insight is that the KV cache is *not* stored request-by-request. Instead, it is a flat pool of "slots", and the mapping from request token positions to slot positions is maintained by `ReqToTokenPool`. This means any slot can hold any token's KV pair, which allows the cache to be shared and reused via the RadixCache prefix tree without copying data.

### PagedAttention

SGLang optionally uses **paging** (`page_size` argument). Instead of allocating one slot per token, it allocates memory in fixed-size pages (e.g., 16 tokens per page). This reduces fragmentation. The attention kernel receives a block table mapping sequence positions to pages, similar to virtual memory in an OS. When `page_size == 1`, paging is effectively disabled and each token gets its own slot.

---

## 8. RadixCache — Prefix Reuse

**Files:** `srt/mem_cache/radix_cache.py`, `srt/managers/schedule_policy.py`

The RadixCache is SGLang's signature optimization. It is a radix tree (also called a compact prefix tree) where every path from root to a node represents a sequence of token IDs, and every node stores a tensor of KV cache slot indices for the tokens along that path.

### How it Works

Imagine two requests:

- Request A: "What is the capital of France? Paris. What is..."
- Request B: "What is the capital of France? Berlin. What is..."

Both share the prefix "What is the capital of France?". After Request A is processed, the RadixCache holds the KV entries for those prefix tokens, stored in a tree node. When Request B arrives, `match_prefix` walks the tree and finds the longest matching path. It returns the KV slot indices for the shared prefix — Request B can skip computing attention for those tokens entirely.

The `TreeNode` class is the building block:

```python
class TreeNode:
    children: dict       # first token of child edge → child node
    key: RadixKey        # the token IDs on the edge leading to this node
    value: torch.Tensor  # KV cache slot indices stored here
    lock_ref: int        # how many active requests are using this node
    last_access_time     # for LRU eviction
```

Nodes are reference-counted. An active request holds a lock on every node along its prefix path. The cache can evict any unlocked node using LRU (or other configurable policies: FIFO, LFU, MRU, SLRU) when memory is scarce.

### Eviction

When the KV pool is under pressure, the Scheduler calls `RadixCache.evict(num_tokens)`. This uses a heap to find the least-recently-used unlocked leaf nodes, removes them from the tree, and returns their slot indices to the free pool. Unlike a request's active memory (which cannot be evicted), cached prefix nodes are always evictable.

### Radix Key

Since SGLang supports LoRA adapters and multi-tenant scenarios, the `RadixKey` carries not just token IDs but also an optional `extra_key` (e.g., LoRA adapter ID). This ensures that KV states computed with one adapter are never incorrectly reused for a different adapter's forward pass.

---

## 9. The Batch Pipeline: Three Levels of Batching

**Files:** `srt/managers/schedule_batch.py`, `srt/model_executor/forward_batch_info.py`

The source code itself documents this architecture clearly:

> `ScheduleBatch → ModelWorkerBatch → ForwardBatch`

This three-level structure separates scheduling concerns from GPU execution concerns.

### ScheduleBatch

`ScheduleBatch` is created and owned by the Scheduler. It contains high-level Python objects: the list of `Req` objects (each holding the full sequence state), the KV cache index tables, sampling parameters per request, and flags like whether this is a prefill or decode step. Most of this data lives on the CPU.

### ModelWorkerBatch

`ModelWorkerBatch` is a projection of `ScheduleBatch` that contains only the information the GPU worker needs. It strips out scheduling metadata and converts Python lists into compact arrays. The `TpModelWorker` in `tp_worker.py` is responsible for receiving a `ScheduleBatch` and preparing the `ModelWorkerBatch`.

### ForwardBatch

`ForwardBatch` is the GPU-resident version. It contains PyTorch tensors that are passed directly into the model's forward method: `input_ids` (the new token IDs being processed this step), `positions` (absolute positions for RoPE), `req_to_token` (the KV cache index table as a GPU tensor), `out_cache_loc` (where new KV pairs should be written), and the `ForwardMode` enum (EXTEND or DECODE).

The `ForwardMode` drives the attention kernel selection:

- `EXTEND`: new tokens that need full attention against all prior tokens (prefill)
- `DECODE`: one new token per sequence, attending to all cached KV (decode)
- `MIXED`: both in the same batch (chunked prefill)

---

## 10. ModelRunner — Running the Forward Pass

**File:** `srt/model_executor/model_runner.py`

The `ModelRunner` is the GPU-side engine. At startup it:

1. **Loads the model weights** using the model loader (`model_loader/`), respecting quantization formats (fp8, int4, etc.)
2. **Initializes the attention backend** (FlashInfer, Triton, FlashAttention, etc.)
3. **Allocates the KV cache** tensors on GPU (`init_memory_pool`)
4. **Captures CUDA graphs** for the decode phase (more on this below)

During inference, `ModelRunner.forward` is called with a `ForwardBatch`. It:

1. Prepares the input tensors (gathers embeddings for `input_ids`)
2. Calls the model's `forward` method, passing the `ForwardBatch` as context
3. Each transformer layer's attention module uses the batch's KV index tensors to read/write the correct slots in the pre-allocated KV cache
4. After the final layer, a logits processor computes per-token logits
5. The sampler (`sampling/`) applies temperature, top-p, top-k, and draws a sample
6. The resulting token IDs are returned to the Scheduler

The model implementations live in `srt/models/`. SGLang has its own implementations of popular architectures (Llama, Qwen, Mistral, DeepSeek, etc.) that are optimized specifically for this caching and batching scheme. Each model's attention layer calls `RadixAttention` from `srt/layers/radix_attention.py`, which knows how to read and write through the slot-based KV cache.

---

## 11. Attention Backends

**Directory:** `srt/layers/attention/`

Attention is the most compute-intensive operation and the one with the most room for optimization. SGLang abstracts attention behind a `BaseAttentionBackend` interface so that multiple kernel implementations can be swapped in at startup.

### FlashInfer (default)

FlashInfer is a library of highly optimized CUDA kernels for attention. It is SGLang's default backend and supports:

- **Paged KV cache**: attention directly over the block-table layout, avoiding the need to copy KV into a contiguous buffer
- **Ragged batching**: sequences of different lengths in the same batch with no padding
- **Fused decode**: a special decode kernel that is extremely fast for the single-token-per-sequence case

The FlashInfer backend is split into a JIT-compiled portion (kernels are compiled on first use and cached) and a pre-compiled CUBIN portion. This is why the first request after a fresh start is slower — the JIT is warming up.

### Triton Backend

The Triton backend provides attention kernels written in Triton, NVIDIA's Python-based GPU programming language. It is more portable than FlashInfer (works on AMD GPUs via HIP) and is useful as a fallback. We used `--attention-backend triton` as a workaround during setup before FlashInfer's JIT cache was properly configured.

### FlashAttention

The FlashAttention backend uses the well-known FlashAttention-2/3 kernels, which use tiling and recomputation to make attention IO-efficient. It is used when FlashInfer is not available or when specific model architectures need it.

### Backend Selection

The backend is chosen by `attention_registry.py` based on `ServerArgs.attention_backend`. During the `ModelRunner` initialization, the chosen backend is instantiated and stored; each `RadixAttention` layer holds a reference to it and calls its `forward_extend` or `forward_decode` method depending on `ForwardMode`.

---

## 12. CUDA Graph Optimization

**File:** `srt/model_executor/cuda_graph_runner.py`

CUDA graph capture is one of the most impactful optimizations in modern LLM inference. To understand why it matters, consider what happens during a decode step without graphs:

Every call to `model.forward()` triggers hundreds of individual CUDA kernel launches. Each launch has a small but non-zero overhead on the CPU side (a few microseconds per kernel). For a model with 32 layers and many operations per layer, the launch overhead can add up to milliseconds per step — which at decode speeds (potentially hundreds of steps per second) becomes a significant fraction of total time.

**CUDA graphs** eliminate this overhead. The idea is simple: record all CUDA operations (the kernel launches, their arguments, the memory pointers) into a "graph" once, then replay the entire graph with a single API call. The CPU overhead drops from O(num_kernels) to O(1) per step.

### Capture and Replay

At startup, `CudaGraphRunner` runs through the decode forward pass once with warm-up inputs, recording everything into a CUDA graph. This is done for several different batch sizes (since the graph is fixed for a given batch size, SGLang captures graphs for a range of common sizes and pads inputs to the nearest captured size at runtime).

During inference, when a decode batch arrives:
1. Its size is rounded up to the nearest captured batch size
2. The graph's input buffers are filled with the actual token IDs and KV indices
3. `cudaGraphLaunch` replays the entire forward pass in microseconds

### When Graphs Cannot Be Used

CUDA graphs require fixed memory addresses. Prefill (EXTEND mode) cannot use graphs because the number of input tokens varies per request. The first step of a new request always runs in "eager" mode (normal Python kernel dispatch). Only subsequent decode steps use graphs.

---

## 13. Sampling

**Files:** `srt/sampling/sampling_params.py`, `srt/sampling/sampling_batch_info.py`, `srt/layers/logits_processor.py`

After the model produces logits (one float per vocabulary token per sequence in the batch), the sampler selects the next token.

`SamplingParams` holds the user-specified parameters:

- **temperature**: scales logits before softmax. `temperature=1.0` is unmodified; lower values make the distribution more peaked (more deterministic); higher values flatten it (more random).
- **top-p** (nucleus sampling): keep only the smallest set of tokens whose cumulative probability exceeds `p`, then sample from that set.
- **top-k**: keep only the `k` highest-probability tokens.
- **min-p**: discard tokens with probability below `min_p * max_probability`.
- **repetition_penalty**, **frequency_penalty**, **presence_penalty**: modify logits based on what has already been generated to reduce repetition.

These operations are applied as a pipeline in `logits_processor.py`. Because many requests are batched together with potentially different parameters, the penalty and sampling operations work on batched tensors using gather/scatter operations to apply per-request parameters efficiently.

---

## 14. DetokenizerManager — Turning Tokens Back to Text

**File:** `srt/managers/detokenizer_manager.py`

The DetokenizerManager receives `BatchTokenIDOutput` messages from the Scheduler containing arrays of newly generated token IDs (one per active request). Its job is to call the HuggingFace tokenizer's `decode` method to convert token IDs back to text strings.

This sounds simple but has a subtlety: the tokenizer's decoding is *stateful* because many tokenizers use byte-pair encoding where a token may be split across what appears to be a natural word boundary. For streaming, you cannot just decode each new token in isolation — you may get garbled output. The `find_printable_text` utility handles this by buffering tokens until it can be sure the decoded string is complete (i.e., does not end mid-character).

The manager also handles **stop-string matching** in token space (checking if the generated text contains any of the user-specified stop sequences) and routes partial outputs back to the TokenizerManager via ZMQ.

---

## 15. The Full Request Lifecycle, End to End

Let's trace a single request — `"What is 2+2?"` with `max_tokens=10` — through the entire system.

**Step 1 — HTTP.** The client sends a POST to `/v1/chat/completions`. FastAPI parses the JSON into a `ChatCompletionRequest` Pydantic object. Our custom code increments `_request_counter`, logs a preview, checks the response cache (miss, since this is the first call), and delegates to `TokenizerManager.generate`.

**Step 2 — Tokenization.** TokenizerManager applies the Qwen3 chat template to produce a formatted prompt string, then tokenizes it into, say, `[151644, 8948, 198, ...]` — a list of integers. It wraps these into a `TokenizedGenerateReqInput` with the sampling params, a fresh UUID `rid`, and pushes it over ZMQ to the Scheduler.

**Step 3 — Scheduling.** The Scheduler receives the message, creates a `Req` object, and places it in `waiting_queue`. On the next scheduling tick it calls `match_prefix` on the RadixCache. This is a cold start, so there is no cached prefix — the full prompt needs to be prefilled. The scheduler allocates KV slots for all prompt tokens and promotes the request to the running batch with `ForwardMode.EXTEND`.

**Step 4 — Prefill.** A `ScheduleBatch` is assembled, projected to a `ModelWorkerBatch`, and converted to a `ForwardBatch` with GPU tensors. The ModelRunner calls the model's `forward` pass in eager mode (no CUDA graph for prefill). All 32 transformer layers run, each computing attention over the full prompt and writing KV pairs into the allocated GPU slots. The logits for the last prompt token are sampled to produce the first output token — say token `220` (a space) or `17` (the digit "4"). This token ID is returned to the Scheduler.

**Step 5 — Decode loop.** The Scheduler appends the new token to the request's output sequence and sends a `BatchTokenIDOutput` with it to the DetokenizerManager. Since `max_tokens=10` is not yet reached and no EOS was produced, the request stays in the running batch. On the next tick, the batch runs with `ForwardMode.DECODE`. The batch size is small, so the CudaGraphRunner selects the pre-captured graph for that size, fills the graph's input buffers with the single new token ID and the updated KV index, and replays the graph in microseconds. This produces the second output token.

**Steps 5a–5j.** The decode loop repeats up to 10 times (or until EOS). Each step is a CUDA graph replay.

**Step 6 — Detokenization.** After each decode step, the DetokenizerManager receives the new token ID, appends it to the sequence buffer for this `rid`, and decodes the accumulated tokens into a text string. If streaming is enabled, each partial string is sent back to the TokenizerManager immediately. Otherwise, the full text is sent at the end.

**Step 7 — HTTP response.** The TokenizerManager receives the final `BatchStrOutput`, constructs a `ChatCompletionResponse` Pydantic object (with our injected `latency_ms` in `usage`), serializes it to JSON, and returns it to the client. If this was a non-streaming request and the response cache is enabled, the result is stored in `_response_cache` for future identical queries.

---

## 16. Key Files Reference

| File | Role |
|---|---|
| `srt/entrypoints/http_server.py` | FastAPI app, all HTTP routes |
| `srt/managers/tokenizer_manager.py` | Async front-door, tokenization, streaming |
| `srt/managers/scheduler.py` | Main scheduling loop, cache management, model dispatch |
| `srt/managers/schedule_batch.py` | `Req`, `ScheduleBatch`, `ModelWorkerBatch` definitions |
| `srt/managers/schedule_policy.py` | LPM and other batch selection policies |
| `srt/managers/tp_worker.py` | Bridge between Scheduler and ModelRunner |
| `srt/managers/detokenizer_manager.py` | Token IDs → text, streaming output |
| `srt/managers/io_struct.py` | All inter-process message types |
| `srt/model_executor/model_runner.py` | Loads model, runs forward pass, owns KV memory |
| `srt/model_executor/forward_batch_info.py` | `ForwardBatch`, `ForwardMode` |
| `srt/model_executor/cuda_graph_runner.py` | CUDA graph capture and replay for decode |
| `srt/mem_cache/memory_pool.py` | `ReqToTokenPool`, `KVCache` GPU allocation |
| `srt/mem_cache/radix_cache.py` | Prefix tree for KV reuse |
| `srt/mem_cache/allocator.py` | Slot allocator for KV pool |
| `srt/layers/attention/` | All attention backend implementations |
| `srt/layers/attention/flashinfer_backend.py` | FlashInfer kernels |
| `srt/layers/attention/triton_backend.py` | Triton kernels |
| `srt/layers/radix_attention.py` | Attention layer that uses the slot-based KV cache |
| `srt/layers/logits_processor.py` | Post-model logit processing |
| `srt/sampling/sampling_params.py` | User-facing sampling parameters |
| `srt/server_args.py` | All server configuration (CLI flags → `ServerArgs`) |
| `srt/models/` | Per-architecture model implementations (Qwen, Llama, etc.) |

---

*This document covers the standard single-node, single-GPU inference path. SGLang also supports tensor parallelism (TP), pipeline parallelism (PP), data parallelism (DP attention), speculative decoding, LoRA multi-adapter serving, disaggregated prefill/decode, and multimodal inputs — each of which adds additional complexity on top of this foundation.*
