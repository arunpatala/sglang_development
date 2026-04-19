# Layer 0 — Summary

This chapter builds the simplest possible LLM inference system from scratch. No tricks, no optimisations — just the fundamental pipeline that every production serving system is built on top of. By the end you have a working HTTP server that loads a real model, accepts conversations, generates text, and reports performance numbers that the rest of the curriculum will improve upon.

---

## Large Language Models

An LLM is a neural network trained on a massive corpus of text with one objective: given a sequence of tokens, predict what comes next. From this deceptively simple task, a model trained at sufficient scale develops a wide range of capabilities — answering questions, writing and debugging code, summarising documents, translating between languages, and holding extended conversations. None of these are explicitly programmed; they emerge from the training process itself.

For this chapter, the model is treated as a black box: text goes in, text comes out. The internal machinery will be opened up as the curriculum progresses. What matters right now is the interface and the cost of running it.

The model used throughout is **Qwen3-0.6B**. At 600 million parameters stored in `bfloat16` (2 bytes each), the weights occupy roughly 1.2 GB of GPU VRAM. This is why the model is loaded once when the server starts and kept resident in memory — loading from disk takes several seconds and is far too expensive to repeat for every request.

---

## Tokens

Models do not read text the way humans do. Internally, everything is integers. The tokenizer converts a string into a sequence of **token IDs**, each of which maps to a subword fragment in the model's vocabulary. Common words like "the" or "model" are a single token; longer or rarer words are split into pieces — "tokenizer" becomes "token" + "izer", and a Python identifier like `AutoModelForCausalLM` might become five or six fragments. Qwen3 has a vocabulary of 151,936 tokens, which keeps the vocabulary manageable while still being able to represent any string exactly.

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")

ids = tokenizer.encode("Hello, world!")   # string → list of ints
text = tokenizer.decode(ids)              # list of ints → string
```

This round-trip — encode on the way in, decode on the way out — happens at the boundary of every inference call. Token counts are the correct unit for measuring compute cost, which is why the server reports `prompt_tokens` and `completion_tokens` rather than word counts, and why `max_new_tokens` is a token budget, not a word limit.

Beyond regular text fragments, the tokenizer also defines **special tokens** that carry structural meaning. The most important is the end-of-sequence token (EOS): the model emits it when it considers its response complete, and the generate loop stops automatically. Qwen3 also uses `<|im_start|>` and `<|im_end|>` to mark the boundaries of each speaker's turn.

---

## Messages and the Chat Template

A base language model is a completion engine — give it a prefix and it extends it. To get it to behave as an assistant, it must be fine-tuned on structured conversations, and at inference time you must format the input the same way those training conversations were formatted, or the model's behaviour is undefined.

The standard format is a list of messages, each with a role and content:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
    {"role": "assistant", "content": "Paris."},
    {"role": "user",      "content": "What is its population?"},
]
```

The `"system"` role carries the developer's instruction that shapes the model's behaviour. The `"user"` role is the human. The `"assistant"` role is the model's prior replies. Critically, the model has no persistent memory between requests — every call must include the full conversation history as a flat list. If you omit the prior assistant turns, the model has no way to know the context of a follow-up question.

`apply_chat_template` serialises this list into the string format the model was trained on:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

The result is a string with `<|im_start|>` / `<|im_end|>` markers around each turn, ending with `<|im_start|>assistant\n` — the opening of the model's next reply. `add_generation_prompt=True` adds this final marker so the model knows it is now generating, not reading. When the model finishes generating, it emits `<|im_end|>` to close its turn; calling `tokenizer.decode(..., skip_special_tokens=True)` strips all these structural markers from the output so the caller receives clean text.

---

## Model Loading and the Generate Loop

Models and tokenizers are hosted on **HuggingFace Hub** — a repository platform for machine learning models, similar to GitHub but for weights and configs. Passing a path like `"Qwen/Qwen3-0.6B"` to `from_pretrained` downloads the files on first use and caches them locally for subsequent runs.

The complete inference pipeline looks like this:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

messages = [{"role": "user", "content": "What is 2 + 2?"}]
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
)
input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
prompt_len = input_ids.shape[1]

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=64, use_cache=False)

print(tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True))
```

`torch_dtype=torch.bfloat16` loads weights at half precision, halving memory use with negligible quality impact. `.to("cuda")` moves them to GPU VRAM. `model.eval()` disables dropout so outputs are deterministic. `torch.no_grad()` turns off gradient tracking — only needed during training — saving memory and time on every forward pass.

`model.generate` is the autoregressive loop: run a forward pass, pick the most likely next token, append it to the sequence, repeat. It stops when the model generates EOS or `max_new_tokens` is reached. The output tensor contains both the prompt and the completion concatenated; slicing from `prompt_len` extracts only the new tokens.

---

## The Inference Server

Wrapping the generate loop in an HTTP server allows any client to send requests over a network, indefinitely, without the model being reloaded. **FastAPI** makes this straightforward: decorate a Python function with an HTTP method and path, define the request and response shapes with Pydantic, and FastAPI handles the JSON parsing, validation, and serialisation automatically. **Uvicorn** handles the underlying network layer.

```python
@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # apply template, tokenize, generate, decode
    return GenerateResponse(text=..., prompt_tokens=..., ...)
```

The server reads its configuration from `config.yml` — model path, port, dtype, generation defaults — with command-line arguments taking precedence. This means you can run `python server.py --port 8200` to override a single setting without touching any code.

Three endpoints are exposed. `POST /generate` is the main inference route. `GET /health` returns a quick liveness check used by the benchmark client before it starts sending requests. `GET /stats` reports current GPU memory usage, useful for sanity-checking that the model loaded correctly.

One important limitation: the server processes one request at a time. A second request that arrives while the first is running must wait. With ten simultaneous clients, the tenth waits for all nine ahead of it. This is head-of-line blocking, and it is clearly visible in `test_client.py`'s concurrent test.

---

## Metrics and Benchmarking

Measuring performance precisely requires agreed-upon metrics and a reproducible workload. The three numbers that matter most for an LLM server are:

**Output throughput** (tokens per second): how many generated tokens the server produces per second across all requests. This is the primary capacity metric — it tells you how much generative work the system can sustain.

**Latency** (milliseconds): end-to-end time for a single request. Because requests vary in prompt length and output length, the median (p50) and tail (p99) both matter.

**Request rate** (requests per second): how many complete requests finish per unit time. For a sequential server like Layer 0, this is just `1 / average_latency`.

`benchmark.py` uses the **ShareGPT** dataset — real user conversations — sampled with a fixed seed for reproducibility. It sends 20 requests sequentially, records timing per request, and prints a summary. Running it against every layer with the same seed and dataset makes the numbers directly comparable.

**Baseline on RTX 4060 Ti with Qwen3-0.6B:**

| Metric | Value |
|---|---|
| Output throughput | 114 tok/s |
| Average latency | 1418 ms |
| Request rate | 0.70 req/s |

These are the numbers every subsequent chapter is measured against.

---

## What Comes Next

Layer 0 is the base engine: correct, complete, and unoptimised. Each chapter that follows makes one targeted improvement — speeding up the generate loop, batching multiple requests together, handling concurrency without blocking, managing memory more carefully, and eventually quantising the model to fit more throughput into the same hardware. Every change is measured with the same benchmark, so progress is always visible.
