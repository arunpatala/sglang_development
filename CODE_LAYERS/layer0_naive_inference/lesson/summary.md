# Layer 0 — Summary

## What This Chapter Built

A minimal, end-to-end LLM inference system: load a model, format a conversation, generate text, serve it over HTTP, and measure performance. Every subsequent layer in this curriculum improves on what is built here.

---

## Large Language Models

An LLM is a neural network trained to predict the next token from text. From that single objective it develops the ability to answer questions, write code, summarise documents, and hold conversations. The model is a black box with a simple interface for this chapter: text goes in, text comes out.

Qwen3-0.6B has 600 million parameters. At `bfloat16` (2 bytes per parameter) that is roughly 1.2 GB of GPU VRAM just for the weights — which is why the model is loaded once at startup and kept resident in memory for the lifetime of the server.

---

## Tokens

Models do not read characters or whole words; they read **tokens** — integer IDs that map to subword fragments. Common words are a single token; rarer or longer words are split into pieces (e.g. "tokenizer" → "token" + "izer"). Qwen3 has a vocabulary of 151,936 tokens.

The tokenizer converts text to IDs (`encode`) and IDs back to text (`decode`). This round-trip happens at the boundary of every inference call. Token counts, not word counts, are the correct unit for measuring compute cost and for interpreting `prompt_tokens`, `completion_tokens`, and `max_new_tokens`.

**Special tokens** carry structural meaning rather than text. The most important for inference is the end-of-sequence token (EOS): the model emits it when it considers its response complete, and the generate loop stops. Qwen3 also uses `<|im_start|>` and `<|im_end|>` to mark speaker turns.

---

## Messages and the Chat Template

A base model is a text completion engine — give it a prefix and it continues it. To behave as an assistant the model must be fine-tuned on structured conversations and then receive the same structure at inference time.

The standard format is a list of messages with roles:

```python
messages = [
    {"role": "system",    "content": "You are a helpful assistant."},
    {"role": "user",      "content": "What is the capital of France?"},
]
```

`"system"` is the developer's instruction, `"user"` is the human, `"assistant"` is the model's reply. The model has no persistent memory — every request must include the full conversation history.

`apply_chat_template` serialises the messages list into the formatted string the model was trained on, with `<|im_start|>` / `<|im_end|>` turn markers. `add_generation_prompt=True` appends the opening of the assistant's turn so the model knows to generate a reply. When it finishes, it emits `<|im_end|>` to close its turn; `skip_special_tokens=True` strips these markers from the decoded output.

---

## Model Loading and the Generate Loop

```python
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
            torch_dtype=torch.bfloat16).to("cuda")
model.eval()

formatted  = tokenizer.apply_chat_template(messages, tokenize=False,
                 add_generation_prompt=True, enable_thinking=False)
input_ids  = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")

with torch.no_grad():
    output_ids = model.generate(input_ids, max_new_tokens=64, use_cache=False)

print(tokenizer.decode(output_ids[0, input_ids.shape[1]:], skip_special_tokens=True))
```

Models and tokenizers are hosted on HuggingFace Hub and downloaded on first use. `from_pretrained` reads the architecture config and weight files; `.to("cuda")` moves weights to GPU VRAM. `model.eval()` disables dropout for deterministic output. `torch.no_grad()` skips gradient tracking, which is only needed during training. `model.generate` runs the autoregressive loop — one forward pass per token — until EOS or `max_new_tokens`. The output tensor includes the prompt; slicing from `prompt_len` extracts only the newly generated tokens.

---

## The Inference Server

Wrapping the generate loop in an HTTP server allows multiple clients to send requests over a network without the model being reloaded each time. FastAPI declares Python functions as endpoints:

```python
@app.post("/generate")
def generate(req: GenerateRequest): ...
```

FastAPI validates the incoming JSON against a Pydantic schema, calls the function, and serialises the return value back to JSON. Uvicorn handles the network layer underneath.

The server reads configuration from `config.yml` with CLI arguments taking precedence, then the file, then hardcoded defaults. It exposes three endpoints: `POST /generate` for inference, `GET /health` for liveness checks, and `GET /stats` for GPU memory usage.

The server processes one request at a time. A second request arriving while the first is running must wait — this is head-of-line blocking, and it causes wall latency to grow linearly with concurrent clients.

---

## Metrics and Benchmarking

| Metric | What it measures |
|---|---|
| Output throughput (tok/s) | Generated tokens per second — the primary capacity metric |
| Total throughput (tok/s) | Prompt + completion tokens per second |
| Request rate (req/s) | Completed requests per second |
| Latency (ms) | End-to-end time for a single request; p50 and p99 matter |

`benchmark.py` samples 20 conversations from the ShareGPT dataset with a fixed seed and sends them sequentially, producing a reproducible result that can be compared directly across layers.

**Baseline on RTX 4060 Ti with Qwen3-0.6B:**
- Output throughput: **114 tok/s**
- Average latency: **1418 ms**
- Request rate: **0.70 req/s**

Every subsequent chapter is measured against these numbers.
