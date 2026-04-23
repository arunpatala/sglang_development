# LoRAX: Serve 1000 Fine-Tuned Models on One GPU — Here's How

**Source:** https://medium.com/@saimudhiganti/lorax-serve-1000-fine-tuned-models-on-one-gpu-heres-how-62336a64de4b
**Author:** Sai Krishna Reddy Mudhiganti (Samsung AI Innovation team)
**Published:** July 16, 2025 — 3 min read
**Level:** L1 — Shortest useful introduction; Docker quickstart in 3 minutes
**Why here:** The most concise L1 entry point for understanding what LoRAX is and why multi-adapter serving matters. Sets up the mental model (one base model, many adapters, per-request selection) and gives a working Docker example in under 10 minutes. Read this before all other references.

---

## The Problem

Fine-tuning large language models is easier than ever. Serving them efficiently? That's where LoRAX steps in.

You fine-tuned a LLaMA 8B model using LoRA, producing a small adapter file. Now you have:

- Dozens (or hundreds) of these fine-tuned adapters.
- A single powerful GPU.
- And you don't need to spin up a new server for every model.

Normally, you'd have to load a full model for each adapter — expensive and wasteful. LoRAX fixes this by letting you:

- ✅ Load the base model once
- ✅ Serve many adapters dynamically
- ✅ Run it all on one GPU

## Meet LoRAX

LoRAX (LoRA eXchange) is an open-source inference server designed to serve thousands of LoRA adapters on a single base model.

Key features:

- **Dynamic Adapter Loading** — Load adapters only when needed (just-in-time).
- **Shared Base Model** — One GPU holds the base model, all adapters are light.
- **OpenAI-Compatible API** — Works like OpenAI's API, so integration is easy.
- **Docker-Ready** — No complex setup. Just pull and run.

Works with LLaMA, Mistral, Qwen, CodeLLaMA and others. Works with adapters trained via PEFT or Ludwig. Supports quantization (bitsandbytes, GPTQ, AWQ).

## Setup: Serving a Fine-Tuned Model with LoRAX (Using Docker)

Assumes:
- You fine-tuned `meta-llama/Llama-2-8b-hf` using LoRA
- Adapter saved locally at `./lorax_data/llama-8b-sentiment`
- NVIDIA GPU (Ampere or newer), CUDA 11.8+

### Run LoRAX in Docker

```bash
MODEL_ID="meta-llama/Llama-2-8b-hf"
VOLUME_DIR="$PWD/lorax_data"

mkdir -p "$VOLUME_DIR"
docker run --gpus all --shm-size 1g -p 8080:80 \
  -v "$VOLUME_DIR:/data" \
  ghcr.io/predibase/lorax:main \
  --model-id "$MODEL_ID"
```

This loads the base model ONCE. LoRAX will hot-load any adapter you ask for — no restart needed.

### Inference via REST API

**Base model only:**
```bash
curl 127.0.0.1:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{"inputs": "Explain quantum physics in simple terms.", "parameters": {"max_new_tokens": 50}}'
```

**Using your LoRA adapter:**
```bash
curl 127.0.0.1:8080/generate \
  -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "inputs": "How do I know if a stock is undervalued?",
    "parameters": {
      "max_new_tokens": 50,
      "adapter_id": "/data/llama-8b-sentiment"
    }
  }'
```

You can swap adapters per request. Just change `adapter_id`.

### OpenAI-Compatible Python API

```python
from openai import OpenAI

client = OpenAI(
    api_key="EMPTY",
    base_url="http://127.0.0.1:8080/v1"
)
response = client.chat.completions.create(
    model="/data/llama-8b-sentiment",  # local adapter path or HF ID
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "How does inflation affect consumers?"}
    ],
    max_tokens=64
)
print(response.choices[0].message.content)
```

You can plug LoRAX into your own apps like you would with OpenAI's GPT API — no code rewrite needed.

## Why This Matters

| Traditional Setup | With LoRAX |
|---|---|
| 10 adapters = 10 full models = $$$ | 1 base model + 10 tiny LoRA adapters |
| GPU bills scale linearly with adapters | GPU cost nearly constant with adapter count |
| Restart required to switch models | Per-request adapter_id selection |

You can even run hundreds of adapters from disk and LoRAX will smartly load/cache them.

## Key Takeaway for Layer 20

Layer 20 implements the *minimal* version of this concept: one static adapter, no dynamic loading. LoRAX is the production version of the same idea scaled to thousands of adapters with LRU caching, SGMV kernels, and serverless deployment.
