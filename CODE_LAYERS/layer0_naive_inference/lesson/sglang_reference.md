# SGLang Reference — Layer 0 Concepts in the Real Codebase

This file maps each concept from the Layer 0 lesson to where and how it is implemented in the SGLang source tree. The SGLang repo lives at `REPOS/sglang/python/sglang/srt/`. All paths below are relative to that root.

The overarching difference to keep in mind throughout: our Layer 0 server is a single synchronous Python process — one function does everything in sequence. SGLang is a **multi-process pipeline** where the HTTP server, tokenizer, scheduler, model executor, and detokenizer all run in separate processes and communicate over ZMQ. The simple call chain `tokenize → generate → decode` becomes a message passing system with async I/O, continuous batching, and tensor parallelism.

---

## 1. Tokenizer Loading

**Layer 0:** `AutoTokenizer.from_pretrained(model_path)` — one line, loads synchronously at startup.

**SGLang:** `managers/tokenizer_manager.py` — `TokenizerManager` (line 215)

The tokenizer is owned by the `TokenizerManager` process, which also owns all pre- and post-processing. It calls `get_tokenizer` from `utils/hf_transformers/tokenizer.py`, which wraps `AutoTokenizer.from_pretrained` with additional logic for fast vs slow tokenizers, special token handling, and tokenizer config overrides. The tokenizer is not directly accessible to the model runner — it lives in a separate process and communicates results via ZMQ.

---

## 2. Model Weight Loading

**Layer 0:** `AutoModelForCausalLM.from_pretrained(..., torch_dtype=torch.bfloat16).to("cuda")`

**SGLang:** `model_loader/loader.py` — `DefaultModelLoader` (line 306), `model_executor/model_runner.py` — `ModelRunner` (line 289)

SGLang does not use `AutoModelForCausalLM.from_pretrained`. Instead it instantiates the model architecture directly from its own registry (each supported model has a custom class under `models/`), then loads weights by iterating over `safetensors` shards using `safetensors_weights_iterator` from `model_loader/weight_utils.py`. This allows SGLang to:

- Load weights directly onto GPU without a CPU staging step
- Support tensor parallelism by sharding weights across multiple GPUs at load time
- Apply quantisation during the load pass rather than as a post-processing step

The `DefaultModelLoader` handles the standard case. Specialised loaders exist for GGUF, sharded checkpoints, remote models, and quantised formats.

---

## 3. Chat Template and `apply_chat_template`

**Layer 0:**
```python
tokenizer.apply_chat_template(messages, tokenize=False,
    add_generation_prompt=True, enable_thinking=False)
```

**SGLang:** `entrypoints/openai/serving_chat.py` — `OpenAIServingChat._process_messages` (line 356)

The call is essentially the same — SGLang also calls `tokenizer.apply_chat_template` — but it goes through several layers first:

`TemplateManager` (`managers/template_manager.py`) resolves which chat template to use: it may take the template baked into the tokenizer's `chat_template` field, override it with a custom template passed via `--chat-template`, or fall back to a FastChat-style conversation template from `parser/conversation.py`. 

For the standard case (HuggingFace tokenizer with an embedded Jinja template), the actual call in `serving_chat.py` around line 518 is:
```python
prompt_ids = self.tokenizer_manager.tokenizer.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, **extra_template_kwargs
)
```

Notice `tokenize=True` — SGLang tokenizes in the same call rather than doing it separately, and receives token IDs directly rather than an intermediate formatted string.

---

## 4. The Generate Loop (Autoregressive Decoding)

**Layer 0:** `model.generate(input_ids, max_new_tokens=64, use_cache=False)` — a blocking loop inside HuggingFace's `GenerationMixin`, one request at a time.

**SGLang:** `managers/scheduler.py` — `Scheduler` (line 316), `model_executor/model_runner.py` — `ModelRunner.forward_decode` / `forward_extend`

This is where the architecture diverges most dramatically. SGLang has no equivalent of `model.generate`. Instead:

The `Scheduler` runs a **continuous event loop** (`event_loop_normal` or `event_loop_overlap`). On each iteration it calls `get_next_batch_to_run` to assemble a batch from all currently active requests — both new requests being prefilled and existing requests in their decode phase — and sends that batch to `ModelRunner.forward`. This is **continuous batching**: rather than waiting for one request to finish before starting the next, the scheduler interleaves prefill and decode steps across many requests simultaneously.

The `ModelRunner` (`model_executor/model_runner.py`) runs the actual GPU forward pass. It distinguishes between two modes:
- `forward_extend` (prefill): process all prompt tokens in parallel for a new request
- `forward_decode`: generate one new token per active request, but for a whole batch of requests at once

The output is logits → sampling → new token IDs, handled by `TpModelWorker.forward_batch_generation` in `managers/tp_worker.py`. The results are sent to the `DetokenizerManager` process via ZMQ, which converts token IDs back to text and streams the response to the waiting HTTP client.

---

## 5. The HTTP Server and Routes

**Layer 0:**
```python
app = FastAPI()

@app.post("/generate")
def generate(req: GenerateRequest): ...

uvicorn.run(app, host="0.0.0.0", port=8100)
```

**SGLang:** `entrypoints/http_server.py`

SGLang also uses FastAPI and Uvicorn, but the `lifespan` function (line 296) wires up a much larger set of components at startup: the `TokenizerManager`, serving handlers for each API flavour, and process health monitoring. The relevant routes:

| Route | Line | Handler |
|---|---|---|
| `POST /v1/chat/completions` | 1528 | `OpenAIServingChat.handle_request` |
| `POST /v1/completions` | 1491 | `OpenAIServingCompletion.handle_request` |
| `GET /health` / `GET /health_generate` | 515 | Fires a minimal generation request to confirm the pipeline is alive |
| `GET /get_model_info` | 590 | Returns model metadata |

There is no `/generate` equivalent at the public API level for chat — SGLang speaks the OpenAI `/v1/chat/completions` path natively. Internally, the serving handler converts the OpenAI request into a `GenerateReqInput` (`managers/io_struct.py`, line 134) and sends it to the `TokenizerManager` via an async queue.

All route handlers are `async def` — SGLang's HTTP layer is fully async, whereas Layer 0's are synchronous `def` functions.

---

## 6. Request and Response Schemas

**Layer 0:**
```python
class GenerateRequest(BaseModel):
    messages: list[Message]
    max_new_tokens: int = 64
    temperature: float = 1.0

class GenerateResponse(BaseModel):
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float
```

**SGLang:** `entrypoints/openai/protocol.py` — `ChatCompletionRequest` (line 554), `ChatCompletionResponse`

SGLang implements the full OpenAI protocol. `ChatCompletionRequest` has dozens of fields covering sampling parameters, tool calls, structured output, logprobs, function calling, reasoning effort, and more. The response schema matches the OpenAI API exactly, including the `choices` list, `usage` object with `prompt_tokens` / `completion_tokens`, and streaming `data: [DONE]` events.

The internal representation used between the tokenizer and scheduler is a separate, leaner `GenerateReqInput` dataclass (`managers/io_struct.py`, line 134) that the serving handlers translate into after validating the public request.

---

## 7. Server Configuration

**Layer 0:** `config.yml` with a three-level priority (CLI > file > defaults), manually parsed.

**SGLang:** `server_args.py` — `ServerArgs` dataclass (line 285)

`ServerArgs` is a large dataclass with over 100 fields covering the model path, dtype, quantisation format, tensor parallelism degree, memory limits, attention backend, speculative decoding settings, chunked prefill, and much more. It is populated from CLI arguments using `argparse` and passed through to the `Engine`, `Scheduler`, and `ModelRunner`. There is no file-based config layer in core SGLang — configuration is entirely CLI-driven, though wrapper scripts can supply defaults.

---

## 8. Health Check

**Layer 0:** `GET /health` returns `{"status": "ok", "layer": 0}` immediately.

**SGLang:** `entrypoints/http_server.py` lines 515–565

SGLang's `/health_generate` does more: it fires an actual minimal generation request through the full pipeline — tokenizer → scheduler → model → detokenizer — and waits for the detokenizer to report activity. If the pipeline is deadlocked or a subprocess has crashed, the health check will time out and return a non-200 status. The simpler `GET /health` path may return 200 without a generation round-trip depending on the `SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION` environment variable.

---

## Summary of Architectural Differences

| Aspect | Layer 0 | SGLang |
|---|---|---|
| Process model | Single process | HTTP + TokenizerManager + Scheduler + DetokenizerManager (separate processes) |
| IPC | In-process function calls | ZMQ message passing |
| Concurrency | Synchronous, one request at a time | Async HTTP, continuous batching across many requests |
| Generate loop | `model.generate()` (HuggingFace) | Custom `Scheduler` event loop + `ModelRunner.forward` |
| Model loading | `AutoModelForCausalLM.from_pretrained` | Custom loaders with direct safetensors iteration |
| Tensor parallelism | None | `TpModelWorker`, sharded weights across GPUs |
| API surface | Custom `/generate` endpoint | Full OpenAI-compatible `/v1/chat/completions` |
| Configuration | `config.yml` + CLI | `ServerArgs` dataclass, CLI only |
