# Section 05 — Server and Config Integration

## Overview

LoRA support propagates through four integration points: the config file, the server argument parsing, the API request schema, and the model runner startup. Each of these is a small, targeted change.

---

## `config.yml`: Single `lora_path`

The multi-LoRA config section was simplified from a pool configuration to a single path:

**Before (multi-LoRA pool config — removed):**
```yaml
# Multi-LoRA Serving
lora_paths:
  - "phh/Qwen3-0.6B-TLDR-Lora"
  - "another/adapter"
max_loras_per_batch: 4
max_lora_rank: 64
lora_mem_fraction: 0.1
```

**After (single static adapter):**
```yaml
# LoRA (Layer 20) — single static adapter, null to disable
lora_path: "phh/Qwen3-0.6B-TLDR-Lora"
```

Setting `lora_path: null` (or omitting the key entirely) disables LoRA — the model runner skips loading any adapter and `lora_adapter` is `None` throughout.

---

## `server.py`: Three Changes

### 1. Config loading

`_load_config()` is unchanged — it reads `lora_path` from `config.yml` automatically because it loads the entire YAML dict. The new module-level variable extracts it:

```python
LORA_PATH = _CFG.get("lora_path", None)   # Layer 20
```

A CLI override could be added later:
```python
# (not yet implemented, but the pattern is clear)
p.add_argument("--lora-path", type=str, default=None, dest="lora_path")
```

### 2. `ChatRequest` schema

`lora_id` is added as an optional field:

```python
class ChatRequest(BaseModel):
    model:       str            = MODEL_PATH
    messages:    list[Message]
    max_tokens:  int            = 128
    temperature: float          = 0.0
    lora_id:     Optional[str]  = None   # Layer 20: name of adapter, or None for base
```

The field is `Optional[str]` with default `None`. Any non-null string activates the adapter (the string itself is the "adapter name" — it is not validated against a registry in this minimal implementation). Setting `lora_id: null` (or omitting it) uses the base model.

The choice of `Optional[str]` rather than `bool` prepares for multi-LoRA: in the full system, `lora_id` would match one of many registered adapter names. Using a string now means the API does not need to change later.

### 3. `Req` construction and startup

```python
@app.post("/v1/chat/completions")
async def chat(req_body: ChatRequest):
    ...
    req = Req(
        rid            = uuid.uuid4().hex,
        input_ids      = input_ids,
        max_new_tokens = req_body.max_tokens,
        temperature    = req_body.temperature,
        lora_id        = req_body.lora_id,   # Layer 20: pass through
        future         = future,
    )
    ...

@app.on_event("startup")
async def startup():
    ...
    runner = ModelRunner(
        model_path            = MODEL_PATH,
        page_size             = PAGE_SIZE,
        enable_prefix_caching = ENABLE_PREFIX_CACHING,
        use_gptq              = USE_GPTQ,
        kv_memory_fraction    = KV_MEMORY_FRACTION,
        lora_path             = LORA_PATH,   # Layer 20: load adapter at startup
    )
    ...
```

---

## `request.py`: `lora_id` on `Req`

The `Req` dataclass is extended with `lora_id`:

```python
@dataclass
class Req:
    rid:            str
    input_ids:      List[int]
    max_new_tokens: int
    temperature:    float
    lora_id:        Optional[str] = None   # Layer 20
    future:         Optional[asyncio.Future] = None
    ...
```

`lora_id` flows from the HTTP request through the scheduler queue to the model runner without any transformation — the scheduler passes requests through unmodified, and the model runner reads `req.lora_id` when building the mask.

---

## `model_runner.py`: Loading the Adapter at Startup

```python
class ModelRunner:
    def __init__(
        self,
        model_path: str,
        page_size: int = PAGE_SIZE,
        enable_prefix_caching: bool = True,
        use_gptq: bool = False,
        kv_memory_fraction: float = _KV_MEMORY_FRACTION,
        lora_path: Optional[str] = None,   # Layer 20
    ) -> None:
        ...
        # After model and KV pool are loaded:

        # ── LoRA adapter (optional) ──────────────────────────────────────
        self.lora_adapter = None
        if lora_path:
            from lora import LoRAAdapter
            self.lora_adapter = LoRAAdapter(lora_path, dtype=DTYPE, device=DEVICE)

        logger.info(
            f"ModelRunner ready  "
            f"lora={'on' if self.lora_adapter else 'off'}"
        )
```

The adapter is loaded **once at startup** and stored on `self.lora_adapter`. It is never reloaded or swapped during the server's lifetime — this is the "static single adapter" invariant. All forward passes use the same `self.lora_adapter` object.

The import of `LoRAAdapter` is deferred to this block (not at module top level) to avoid circular imports when `model_runner.py` is imported without LoRA configured.

---

## API Usage Examples

### Base model request (no LoRA)

```bash
curl -X POST http://localhost:8114/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "What is 2+2?"}],
    "max_tokens": 32
  }'
```

The `lora_id` field is absent — Pydantic fills it as `None`. The mask for this request will be all `0.0`.

### LoRA adapter request

```bash
curl -X POST http://localhost:8114/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "Summarize: The quick brown fox jumps."}],
    "max_tokens": 32,
    "lora_id": "tldr"
  }'
```

`lora_id = "tldr"` is a non-null string → the mask for this request will be all `1.0`.

### Mixed batch (handled automatically)

Both requests above could arrive concurrently. The scheduler batches them together. `prefill_batch` sees:
```python
reqs = [req_base (lora_id=None), req_lora (lora_id="tldr")]
```

And builds:
```python
mask_vals = [0.0, 0.0, ..., 0.0,  # base req tokens
             1.0, 1.0, ..., 1.0]  # lora req tokens
lora_mask = tensor(mask_vals).view(1, -1, 1)
```

The model sees one unified forward pass; the mask handles the routing.

---

## What `lora_id` Does NOT Do

In this minimal implementation, `lora_id` is a presence/absence signal — any non-null string activates the loaded adapter. It does not:
- Select among multiple adapters (only one is loaded)
- Validate the adapter name against a registry
- Route to different adapter configurations

In the full multi-LoRA system, `lora_id` would be a key into the `LoRARegistry` that maps names to GPU memory pool slots.

---

## Startup Sequence

```
1. uvicorn.run() starts
2. FastAPI startup event fires:
   a. ModelRunner.__init__():
      - Load tokenizer
      - Load Qwen3ForCausalLM weights from disk (~2.4 GB BF16)
      - Allocate KVPool (85% of remaining VRAM)
      - Allocate ReqToTokenPool, FlashInfer workspaces
      - Load LoRAAdapter from "phh/Qwen3-0.6B-TLDR-Lora"
        * snapshot_download() → local cache
        * Read adapter_config.json
        * Stream 2.2 MB safetensors → 112 tensors on GPU
   b. Scheduler starts in background thread
3. Server is ready to accept requests
```

Total extra startup time for LoRA: ~0.5 s (mostly HF Hub resolution + safetensors streaming). GPU memory: +2.2 MB (negligible).
