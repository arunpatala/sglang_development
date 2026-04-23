# LoRA Serving — SGLang Official Documentation

**Source:** https://sgl-project.github.io/advanced_features/lora.html
**Project:** SGLang (sgl-project)
**Accessed:** April 2026
**Level:** L2 — Official production framework documentation
**Why here:** SGLang is the framework Layer 20 is built on. This documentation describes the production LoRA system that Layer 20 is a simplified version of: the `--lora-paths` argument, `max_loras_per_batch` memory pool, `lora_eviction_policy`, `lora_backend` (Triton / ChunkedSGMV), dynamic load/unload API, and tensor parallelism strategy. Directly informs what would need to change to promote Layer 20's single-adapter implementation to full production readiness.

---

## Overview

SGLang enables use of LoRA adapters with a base model. By incorporating techniques from [S-LoRA](https://arxiv.org/pdf/2311.03285) and [Punica](https://arxiv.org/pdf/2310.18547), SGLang can efficiently support multiple LoRA adapters for different sequences within a single batch.

---

## Server Arguments for Multi-LoRA Serving

| Argument | Description |
|---|---|
| `--enable-lora` | Enable LoRA support. Auto-set to True if `--lora-paths` is provided. |
| `--enable-lora-overlap-loading` | Asynchronous H2D transfers overlapped with GPU compute — useful when loading large adapters is the bottleneck. |
| `--lora-paths` | List of adapters to load. Format: `name=path` or JSON `{"lora_name":str,"lora_path":str,"pinned":bool}`. |
| `--max-loras-per-batch` | Maximum adapters used per batch. Affects GPU memory reservation. Default: 8. |
| `--max-loaded-loras` | Limit on adapters loaded in CPU memory. Must be ≥ `max-loras-per-batch`. |
| `--lora-eviction-policy` | `lru` (default) or `fifo` — controls which adapter is evicted when GPU pool is full. |
| `--lora-backend` | `triton` or `csgmv` (Chunked SGMV). Faster Cutlass/CUDA backends planned. |
| `--max-lora-rank` | Maximum LoRA rank to support. Needed when dynamically loading larger-rank adapters. |
| `--lora-target-modules` | Modules where LoRA is applied (`q_proj`, `k_proj`, etc). Set to `all` for all supported modules. |
| `--max-lora-chunk-size` | Chunk size for ChunkedSGMV backend. Default: 16. |
| `--tp-size` | Tensor parallel GPUs. SGLang implements the S-LoRA tensor sharding strategy for LoRA computation. |

---

## Usage Examples

### Serving a Single Adapter

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    --max-loras-per-batch 2 \
    --log-level warning
```

**Native API request** (per-sequence adapter selection):

```python
import requests

json_data = {
    "text": [
        "List 3 countries and their capitals.",  # uses lora0
        "List 3 countries and their capitals.",  # uses base model
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    "lora_path": ["lora0", None],  # None = base model
}
response = requests.post("http://localhost:{port}/generate", json=json_data)
```

**Output differences observed:**
- With `lora0`: structured output (`France, Paris\nJapan, Tokyo\nBrazil, Brasília`)
- With base model: enumerated list (`1. United States - Washington D.C. 2. Japan - Tokyo...`)

**OpenAI-compatible API** (uses `model` field as adapter selector):

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:{port}/v1")

# Use adapter
response = client.chat.completions.create(
    model="model_name:lora0",  # syntax: base_model:adapter_name
    messages=[{"role": "user", "content": "List 3 countries"}]
)

# Use base model
response = client.chat.completions.create(
    model="model_name",  # no adapter suffix
    messages=[{"role": "user", "content": "List 3 countries"}]
)
```

### Serving Multiple Adapters

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
               lora1=Nutanix/Meta-Llama-3.1-8B-Instruct_SFT_lora_4_alpha_16_humaneval_raw_json \
    --max-loras-per-batch 2
```

Mixed batches route automatically:

```python
json_data = {
    "text": ["List 3 countries.", "Write a Python function."],
    "lora_path": ["lora0", "lora1"],  # each input uses a different adapter
}
```

### Dynamic LoRA Loading (Load/Unload at Runtime)

Instead of specifying adapters at startup, load/unload via API endpoints:

```python
# Load a new adapter
requests.post("http://localhost:{port}/load_lora_adapter", json={
    "lora_name": "my_new_lora",
    "lora_path": "huggingface_user/adapter_repo",
})

# Unload an adapter
requests.post("http://localhost:{port}/unload_lora_adapter", json={
    "lora_name": "my_new_lora",
})
```

**Important:** When using dynamic loading, specify `--max-lora-rank` and `--lora-target-modules` at startup. Otherwise SGLang infers them from `--lora-paths` and dynamically loaded adapters must have identical or smaller shapes.

**Example startup for dynamic loading:**

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --cuda-graph-max-bs 2 \
    --max-loras-per-batch 2 \
    --max-lora-rank 256 \
    --lora-target-modules q_proj k_proj v_proj o_proj gate_proj up_proj down_proj
```

---

## Memory Architecture: How SGLang Allocates GPU Memory for LoRA

SGLang pre-allocates a **LoRA GPU memory pool** at startup:

- Pool size is determined by `max_loras_per_batch × max_lora_rank × target_modules × model_hidden_size × 2`
- Adapters are loaded into this pool on demand (from CPU memory or disk)
- When pool is full and a new adapter is needed, the `lora_eviction_policy` determines which adapter to evict

**Layer 20 comparison:**
- Layer 20 allocates one adapter's tensors directly to GPU at startup — zero dynamic allocation, zero eviction
- SGLang's pool supports `max_loras_per_batch` *simultaneously active* adapters with LRU eviction for the rest

---

## LoRA Backend Kernels

| Backend | Description | Best for |
|---|---|---|
| `triton` | Triton-based SGMV | General use, good compatibility |
| `csgmv` | Chunked SGMV — processes tokens in chunks | Higher throughput for large batches |
| Future | Cutlass / CUDA kernels | Maximum performance |

The ChunkedSGMV backend `--max-lora-chunk-size` controls token chunk size per kernel invocation. Larger chunks improve efficiency but require more scratch memory.

---

## Tensor Parallelism with LoRA

SGLang follows the S-LoRA tensor sharding strategy:

- `q_proj`, `k_proj`, `v_proj`, `o_proj`: column/row parallel split matching the attention heads
- `gate_proj`, `up_proj`, `down_proj` (MLP): column/row split matching the MLP hidden dimension
- Each GPU shard holds a corresponding slice of both the base weight and the LoRA A/B matrices

---

## Relevance to Layer 20

Layer 20 `lora.py` and `model_runner.py` implement the simplest possible version of what this documentation describes:

| Feature | Layer 20 | SGLang Production |
|---|---|---|
| Number of adapters | 1 static | N dynamic (via `--lora-paths`) |
| Loading | At startup, stays in GPU | JIT load from CPU/disk, LRU eviction |
| Mixed batch | Float mask `lora_delta * lora_mask` | SGMV kernel (Triton / ChunkedSGMV) |
| Tensor parallelism | Not supported | S-LoRA sharding strategy |
| Dynamic load/unload | Not supported | `/load_lora_adapter`, `/unload_lora_adapter` |
