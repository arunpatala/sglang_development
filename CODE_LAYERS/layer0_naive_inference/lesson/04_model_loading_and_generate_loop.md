# 04 — Model Loading and the Generate Loop

## The Complete Minimal Script

Before examining how the server works, it is useful to have the entire inference pipeline in one place — from loading weights to printing a response — without any server machinery around it. Here is that script in full:

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_path = "Qwen/Qwen3-0.6B"

# --- Load ---
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")
model.eval()

# --- Format and tokenize ---
messages = [{"role": "user", "content": "What is 2 + 2?"}]
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
prompt_len = input_ids.shape[1]

# --- Generate ---
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=64,
        use_cache=False,
    )

# --- Decode ---
new_ids = output_ids[0, prompt_len:]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

This is the complete pipeline. Every production LLM inference server is fundamentally this same loop, wrapped in scheduling, batching, memory management, and network plumbing. Understanding each line here is the prerequisite for understanding what all of that adds.

## Line-by-Line Walkthrough

### Loading the Tokenizer

```python
tokenizer = AutoTokenizer.from_pretrained(model_path)
```

Models and their tokenizers are hosted on HuggingFace Hub — a repository platform for machine learning models, similar in spirit to GitHub but for model weights and configs. Each model lives at a path like `Qwen/Qwen3-0.6B`, where `Qwen` is the organisation and `Qwen3-0.6B` is the repository. The repository contains all the files the model needs: weight files, architecture config, and the tokenizer.

`AutoTokenizer.from_pretrained` downloads (or reads from local cache) `tokenizer_config.json` to determine the tokenizer class, then loads `tokenizer.json`, which contains the vocabulary — the mapping from token strings to integer IDs. The resulting `tokenizer` object is stateless with respect to any particular conversation — the same tokenizer is reused for every request the server handles.

### Loading the Model Weights

```python
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
).to("cuda")
```

`AutoModelForCausalLM.from_pretrained` downloads (or reads from local cache) `config.json` to determine the model architecture (number of layers, heads, hidden dimension, etc.), then loads the weight tensors from `model.safetensors`. For Qwen3-0.6B these weights are approximately 1.2 GB on disk. The `torch_dtype=torch.bfloat16` argument instructs HuggingFace to load the weights directly in 16-bit brain float format rather than the default 32-bit float. This halves memory consumption with negligible impact on output quality for inference — bfloat16 preserves the same exponent range as float32, only reducing mantissa precision.

After the weights are loaded into CPU memory, `.to("cuda")` transfers every tensor to GPU VRAM. This is a synchronous operation that blocks until the transfer is complete. In the server, all of this happens once at startup; subsequent requests reuse the weights already resident on the GPU.

### Setting Eval Mode

```python
model.eval()
```

PyTorch models have a training mode and an evaluation mode. In training mode, layers like dropout randomly zero out activations during the forward pass to prevent overfitting. In evaluation mode they are disabled, giving deterministic outputs. `model.eval()` is therefore essential for inference: without it, different calls to the same model with the same input can produce different outputs due to stochastic dropout. It also signals to certain norm layers to use running statistics rather than batch statistics.

### Formatting and Tokenizing

```python
formatted = tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
prompt_len = input_ids.shape[1]
```

These three lines were covered in depth in section 03. The result is `input_ids`: a `torch.LongTensor` of shape `[1, T]` where `T` is the number of prompt tokens. The batch dimension of size 1 is there because `model.generate` expects a batch — in later layers we will pass tensors with a first dimension larger than 1 to process multiple requests simultaneously. Saving `prompt_len` is needed later to slice off the prompt from the model's output.

### Disabling Gradient Tracking

```python
with torch.no_grad():
    output_ids = model.generate(...)
```

PyTorch normally records every operation in a computational graph so that gradients can be computed during backpropagation — this is how the model learns during training, by measuring how wrong its predictions were and adjusting weights accordingly. This bookkeeping has real memory and time cost. During inference there is no backpropagation, so `torch.no_grad()` disables the graph entirely. This is a context manager that applies to all operations inside its block. Forgetting it does not cause wrong output, but it wastes memory proportional to sequence length and makes every forward pass measurably slower.

### The Generate Call

```python
output_ids = model.generate(
    input_ids,
    max_new_tokens=64,
    use_cache=False,
)
```

`model.generate` is the autoregressive decode loop. It runs a forward pass through the model, picks the most likely next token, appends it to the sequence, and repeats — stopping when the model generates an end-of-sequence token or when `max_new_tokens` new tokens have been produced. The `use_cache` argument controls an important optimisation that we will look at closely in the next chapter.

`output_ids` is a tensor of shape `[1, prompt_len + completion_len]`. It contains both the prompt token IDs and the newly generated token IDs concatenated into a single sequence.

### Decoding the Output

```python
new_ids = output_ids[0, prompt_len:]
print(tokenizer.decode(new_ids, skip_special_tokens=True))
```

The slice `output_ids[0, prompt_len:]` extracts only the tokens the model generated, discarding the prompt. The first index `0` selects the first (and only) element of the batch. `tokenizer.decode` converts the integer sequence back to a UTF-8 string. `skip_special_tokens=True` strips any end-of-sequence or role-boundary tokens from the output so the caller receives clean text rather than `The answer is 4.<|im_end|>`.

The server returns `prompt_tokens` and `completion_tokens` as separate fields in its response. These counts come directly from `input_ids.shape[1]` and `len(new_ids)` respectively, and they are the correct unit for billing, capacity planning, and throughput measurement.

## This Is the Base Engine

The loop above is a working inference engine. It loads a real model, formats a real conversation, and generates a real response. The next section wraps it in an HTTP server so it can handle requests over the network. From there, each chapter in this curriculum takes this base and makes it meaningfully faster or more capable — making the generate loop faster, then adding batching, then concurrency, and so on. The foundation you have here is the thing being optimised; understanding it clearly is what makes every subsequent improvement legible.
