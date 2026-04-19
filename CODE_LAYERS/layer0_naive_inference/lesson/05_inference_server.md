# 05 — The Inference Server

## Why a Server at All?

The generate loop in section 04 is a script: it runs once and exits. A production inference system needs to accept requests from multiple clients, potentially over a network, repeatedly and indefinitely, without reloading the model for each request. Wrapping the loop in an HTTP server solves all of these requirements. The model loads once at startup and stays resident in GPU memory. Each HTTP request triggers one invocation of the generate loop. The client can be a browser, a command-line tool, another microservice, or a benchmark script — anything that can send an HTTP POST.

## FastAPI and How a Request Gets Served

The Layer 0 server uses **FastAPI** to define the API and **Uvicorn** to run it. FastAPI lets you declare a Python function as an HTTP endpoint by decorating it with the HTTP method and path you want it to handle:

```python
@app.post("/generate")
def generate(req: GenerateRequest):
    ...
```

When a client sends a `POST` request to `/generate`, FastAPI parses the JSON body, validates it against the `GenerateRequest` schema, and calls this function with the result as a typed Python object. The function runs the generate loop and returns a `GenerateResponse` object, which FastAPI serialises back to JSON and sends to the client. From the client's perspective the interaction is simple: send JSON, receive JSON. From the server's perspective, all the type checking and serialisation is handled automatically.

Uvicorn sits underneath FastAPI and handles the actual network layer — accepting TCP connections, reading HTTP bytes off the wire, and handing parsed requests to FastAPI. Starting the server is a single call:

```python
uvicorn.run(app, host="0.0.0.0", port=8100)
```

`host="0.0.0.0"` means the server listens on all network interfaces, so it is reachable from other machines on the network, not just localhost. Once running, any HTTP client — `curl`, `requests`, a browser — can talk to it.

## Configuration: CLI, File, and Defaults

The server supports three sources of configuration, applied in priority order from highest to lowest: command-line arguments, the `config.yml` file, and Python hardcoded defaults. This mirrors the pattern used by SGLang's own `server_args_config_parser.py`.

```yaml
# config.yml
model: Qwen/Qwen3-0.6B
dtype: bfloat16
device: cuda

host: "0.0.0.0"
port: 8100
log_level: warning

use_cache: false
max_new_tokens: 64
temperature: 1.0

benchmark_num_requests: 20
benchmark_max_new_tokens: 128
benchmark_seed: 42
```

Running `python server.py` loads this file automatically. Running `python server.py --port 8200` overrides only the port; all other settings come from the file. Running `python server.py --use-cache` sets `use_cache=True` regardless of what the file says. This layered approach means you can keep sensible defaults in the file for your development machine and override just the parameters that differ in a particular experiment without editing any code.


## The Request and Response Schema

FastAPI uses Pydantic models to define what a valid request looks like and what the server promises to return. The Layer 0 schema is:

```python
class Message(BaseModel):
    role: str        # "system" | "user" | "assistant"
    content: str

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

`GenerateRequest` mirrors the OpenAI Chat Completions API's `messages` field directly. A client that already speaks the OpenAI schema can send requests to this server without modification, because both use the same `[{"role": ..., "content": ...}]` list structure. The server does not implement the full OpenAI API (there is no `/v1/chat/completions` path, no streaming, no model selection), but the message format is identical.

`GenerateResponse` returns the generated text together with token counts and server-side latency. The `latency_ms` field measures time from request arrival to response — it is server-side wall time, not round-trip network time. The test client records its own wall time separately to capture the network overhead as well.

## The Routes

The server exposes three endpoints.

`POST /generate` is the main inference route. It accepts a `GenerateRequest`, runs the full pipeline from section 04 (apply chat template, tokenize, generate, decode), and returns a `GenerateResponse`. The implementation is a direct translation of the generate loop:

```python
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    messages = [m.model_dump() for m in req.messages]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
    prompt_tokens = input_ids.shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=req.max_new_tokens,
            do_sample=(req.temperature > 0 and req.temperature != 1.0),
            temperature=req.temperature if req.temperature != 1.0 else None,
            use_cache=cfg.use_cache,
        )

    new_ids = output_ids[0, prompt_tokens:]
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return GenerateResponse(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=len(new_ids),
        latency_ms=...,
    )
```

`GET /health` returns `{"status": "ok", "layer": 0}`. Load balancers, container orchestrators, and the benchmark client all use this endpoint to confirm the server is reachable before sending real requests. It has no side effects and costs essentially nothing.

`GET /stats` returns `{"gpu_memory_mb": ...}`. This reports how much GPU VRAM is currently allocated according to PyTorch's memory tracker. It is useful for quick sanity checks: after loading Qwen3-0.6B at bfloat16, you should see roughly 1200 MB allocated. If you see much more, something has leaked tensors.

## The Blocking Problem

The server processes one request at a time. When two requests arrive simultaneously, the second one waits for the first to finish completely before any work begins on it. If ten clients fire at once, the tenth waits for all nine ahead of it. This is **head-of-line blocking**, and it is why `test_client.py` shows wall latency growing roughly linearly with the number of concurrent clients.

The synchronous design makes this bottleneck visible and measurable, which is exactly the point — later chapters fix it.
