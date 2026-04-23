# LoRA Adapters — vLLM Official Documentation

**Source:** https://docs.vllm.ai/en/latest/features/lora.html
**Project:** vLLM (vllm-project)
**Accessed:** April 2026
**Level:** L2 — Official production framework documentation
**Why here:** vLLM is the other major production LLM serving framework with comprehensive multi-LoRA support. Comparing vLLM's approach to SGLang's (and Layer 20's) reveals different design choices: `LoRARequest` object per inference call, `--lora-modules` at server startup, dynamic runtime loading via REST API, LoRAResolver plugin system, and in-place adapter reloading for RL training loops.

---

## Overview

LoRA adapters can be used with any vLLM model that implements `SupportsLoRA`. Adapters are efficiently served on a per-request basis with minimal overhead.

---

## Offline Inference with LoRA

### Download and load an adapter

```python
from huggingface_hub import snapshot_download
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")

# Enable LoRA support at model init
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)
```

### Inference with LoRARequest

```python
sampling_params = SamplingParams(temperature=0, max_tokens=256, stop=["[/assistant]"])

prompts = [
    "[user] Write a SQL query ... [/user] [assistant]",
    "[user] Write another SQL query ... [/user] [assistant]",
]

outputs = llm.generate(
    prompts,
    sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    # Parameters: (human_name, unique_int_id, adapter_path)
)
```

**Key design:** `LoRARequest` is passed per `generate()` call, not per token. All prompts in the batch above use the same adapter. For mixed batches (different adapters per request), see `examples/offline_inference/multilora_inference.py`.

---

## Serving LoRA Adapters via OpenAI-Compatible API

### Server startup

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

### Query available models

```bash
curl localhost:8000/v1/models | jq .
# Returns both the base model and "sql-lora" as separate model entries
```

### Request routing by model name

```bash
# Use LoRA adapter
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sql-lora", "prompt": "San Francisco is a", "max_tokens": 7}'

# Use base model (same server)
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Llama-3.2-3B-Instruct", "prompt": "San Francisco is a", "max_tokens": 7}'
```

The `model` field in the request selects the adapter. Requests are processed in parallel with base model requests.

---

## Dynamic LoRA Serving at Runtime

> ⚠️ **Security warning:** Enable only in isolated, fully trusted environments.

```bash
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
```

### Load an adapter

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "sql_adapter", "lora_path": "/path/to/sql-lora-adapter"}'
# Response: "Success: LoRA adapter 'sql_adapter' added successfully"
```

### Unload an adapter

```bash
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "sql_adapter"}'
# Response: "Success: LoRA adapter 'sql_adapter' removed successfully"
```

### In-Place LoRA Reloading

Update adapter weights while keeping the same name — critical for asynchronous RL training loops where adapters are continuously updated:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "my-adapter", "lora_path": "/path/to/adapter/v2", "load_inplace": true}'
```

---

## LoRAResolver Plugin System

Instead of pre-specifying adapters or calling the REST API, vLLM supports **LoRAResolver plugins** — resolvers that dynamically look up an adapter path given a model name. On every request, when an unknown model name is seen, the resolver is consulted.

### Built-in resolvers (require `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`)

- **Local filesystem:** set `VLLM_PLUGINS=lora_filesystem_resolver`, `VLLM_LORA_RESOLVER_CACHE_DIR=/local/dir`
- **HuggingFace Hub:** set `VLLM_PLUGINS=lora_hf_hub_resolver`, `VLLM_LORA_RESOLVER_HF_REPO_LIST=user/repo,user/repo2`

### Custom S3 resolver example

```python
import os, s3fs
from vllm.lora.request import LoRARequest
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry

class S3LoRAResolver(LoRAResolver):
    def __init__(self):
        self.s3 = s3fs.S3FileSystem()
        self.s3_path_format = os.getenv("S3_PATH_TEMPLATE")
        self.local_path_format = os.getenv("LOCAL_PATH_TEMPLATE")

    async def resolve_lora(self, base_model_name, lora_name):
        s3_path = self.s3_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
        local_path = self.local_path_format.format(base_model_name=base_model_name, lora_name=lora_name)
        await self.s3._get(s3_path, local_path, recursive=True, maxdepth=1)
        return LoRARequest(
            lora_name=lora_name,
            lora_path=local_path,
            lora_int_id=abs(hash(lora_name)),
        )

# Register
s3_resolver = S3LoRAResolver()
LoRAResolverRegistry.register_resolver("s3_resolver", s3_resolver)
```

---

## New `--lora-modules` Format

Old format (name=path only):
```bash
--lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

New format (with base model lineage):
```bash
--lora-modules '{"name": "sql-lora", "path": "jeeejeee/llama32-3b-text2sql-spider", "base_model_name": "meta-llama/Llama-3.2-3B-Instruct"}'
```

The new format enables tracking which base model each adapter was trained on (useful for multi-base-model deployments).

---

## Comparison: vLLM vs SGLang Multi-LoRA

| Feature | vLLM | SGLang |
|---|---|---|
| Adapter selection per request | `LoRARequest(name, id, path)` object | `"lora_path"` field in request JSON |
| Server API | `model` = adapter name in completion | `lora_path` = adapter name in `/generate` |
| Dynamic loading | REST API + LoRAResolver plugins | `/load_lora_adapter` endpoint |
| Kernel backend | Punica BGMV (decode-optimized) | Triton / ChunkedSGMV (prefill-optimized) |
| In-place adapter reload | `load_inplace=true` | Not documented |

---

## Relevance to Layer 20

Layer 20 `server.py` implements a simplified version of vLLM's `model` field routing:

```python
# Layer 20: lora_id in ChatRequest
class ChatRequest(BaseModel):
    lora_id: Optional[str] = None  # None = base model, "adapter_name" = LoRA

# vLLM equivalent: model field in OpenAI request
{"model": "sql-lora", "prompt": "..."}   # LoRA
{"model": "base_model", "prompt": "..."} # base
```

The Layer 20 `lora_mask` construction in `model_runner.py` is the minimal equivalent of vLLM's per-request LoRA routing, without the `LoRARequest` abstraction layer.
