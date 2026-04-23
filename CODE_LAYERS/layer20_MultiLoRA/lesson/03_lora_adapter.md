# Section 03 — `lora.py`: Loading the Adapter

## Overview

`lora.py` contains one class: `LoRAAdapter`. It is responsible for:
1. **Resolving** the adapter path (HF Hub ID or local directory)
2. **Parsing** `adapter_config.json` to get the rank, alpha, and target modules
3. **Loading** A and B weight tensors from `adapter_model.safetensors`
4. **Applying** the LoRA delta on demand via `apply()`

Nothing in `lora.py` knows about batching, masking, or the model architecture. It is a pure weight store + compute function.

---

## The HuggingFace PEFT Checkpoint Format

When PEFT saves a LoRA adapter, it writes two files:

### `adapter_config.json`

```json
{
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  "lora_dropout": 0.0,
  "bias": "none",
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  ...
}
```

This is where `LoRAAdapter.__init__` reads `r`, `lora_alpha`, and `target_modules`.

### `adapter_model.safetensors`

Contains one A and one B tensor per targeted module per layer. The keys follow this pattern:

```
base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight   [8, 1024]
base_model.model.model.layers.{i}.self_attn.q_proj.lora_B.weight   [2048, 8]
base_model.model.model.layers.{i}.self_attn.v_proj.lora_A.weight   [8, 1024]
base_model.model.model.layers.{i}.self_attn.v_proj.lora_B.weight   [1024, 8]
```

For `Qwen3-0.6B` with 28 layers and `target_modules=["q_proj", "v_proj"]`:
- Total keys: 28 × 2 modules × 2 (A+B) = **112 tensors**
- Total size: ~2.2 MB at BF16

---

## `_resolve_adapter_path()`

```python
def _resolve_adapter_path(adapter_path: str) -> Path:
    path = Path(adapter_path)
    if path.is_dir() and (path / "adapter_config.json").exists():
        return path
    logger.info(f"Resolving HF Hub adapter: {adapter_path}")
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    resolved = snapshot_download(adapter_path, local_files_only=offline)
    return Path(resolved)
```

Accepts either:
- A local path: `"/path/to/my-adapter"` — returned directly if `adapter_config.json` exists
- A HuggingFace Hub ID: `"phh/Qwen3-0.6B-TLDR-Lora"` — downloaded via `snapshot_download`

The `HF_HUB_OFFLINE=1` environment variable forces using the local cache without network access — useful in air-gapped deployments.

---

## `LoRAAdapter.__init__`

```python
class LoRAAdapter:
    def __init__(
        self,
        adapter_path: str,
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
    ) -> None:
        path = _resolve_adapter_path(adapter_path)

        with open(path / "adapter_config.json") as f:
            cfg = json.load(f)

        self.r       = cfg["r"]
        self.alpha   = cfg.get("lora_alpha", self.r)
        self.scaling = self.alpha / self.r              # 32/8 = 4.0
        target_modules = set(cfg.get("target_modules", _SUPPORTED_MODULES))
        self.target_modules = target_modules & _SUPPORTED_MODULES  # only known modules

        self.A_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self.B_weights: Dict[int, Dict[str, torch.Tensor]] = {}
        self._load_weights(path, dtype, device)
```

Three important design choices:
1. `scaling` is computed once at init — not recomputed per `apply()` call
2. `target_modules` is intersected with `_SUPPORTED_MODULES` — if the adapter targets a module we don't support (e.g., `embed_tokens`), we silently ignore it
3. A and B weights are stored in nested dicts keyed by `layer_idx` then `module_name` — O(1) lookup in `apply()`

---

## `_load_weights()`

```python
def _load_weights(self, path: Path, dtype: torch.dtype, device: str) -> None:
    weight_file = path / "adapter_model.safetensors"
    if weight_file.exists():
        from safetensors import safe_open
        def _iter():
            with safe_open(str(weight_file), framework="pt", device="cpu") as f:
                for key in f.keys():
                    yield key, f.get_tensor(key).to(dtype)
    else:
        weight_file = path / "adapter_model.bin"
        def _iter():
            ckpt = torch.load(str(weight_file), map_location="cpu")
            for key, tensor in ckpt.items():
                yield key, tensor.to(dtype)

    for name, tensor in _iter():
        parts = name.split(".")
        if "lora_A" not in parts and "lora_B" not in parts:
            continue

        # Find layer index from "...layers.{i}...."
        try:
            layers_idx = parts.index("layers")
            layer_idx  = int(parts[layers_idx + 1])
        except (ValueError, IndexError):
            continue

        # Find module name from supported set
        module_name = None
        for mod in self.target_modules:
            if mod in parts:
                module_name = mod
                break
        if module_name is None:
            continue

        ab = "A" if "lora_A" in parts else "B"
        t  = tensor.to(device)

        if ab == "A":
            self.A_weights.setdefault(layer_idx, {})[module_name] = t
        else:
            self.B_weights.setdefault(layer_idx, {})[module_name] = t
```

**Key parsing logic:**

Given key: `base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight`

```
parts = ["base_model", "model", "model", "layers", "3", "self_attn", "q_proj", "lora_A", "weight"]
                                                     ↑
                                     parts.index("layers") = 3
                                     layer_idx = int(parts[4]) = 3

module_name: scan supported modules:
  - "q_proj" in parts? → yes → module_name = "q_proj"

ab: "lora_A" in parts → ab = "A"
```

Result: `A_weights[3]["q_proj"] = tensor([8, 1024], device="cuda", dtype=bfloat16)`

**Streaming via safetensors:**

`safe_open()` streams one tensor at a time without loading the entire file into CPU RAM. For a 2.2 MB adapter this makes no practical difference, but the same pattern in the full multi-LoRA system allows loading adapters larger than available CPU memory.

**Fallback to `.bin`:**

Older PEFT checkpoints use `adapter_model.bin` (PyTorch pickle format). The fallback loads the entire dict into CPU RAM with `torch.load()`, then iterates over it. This is safe for the small adapter files used here.

---

## `has_layer()` and `apply()`

```python
def has_layer(self, layer_idx: int, module_name: str) -> bool:
    return (
        layer_idx in self.A_weights
        and module_name in self.A_weights[layer_idx]
        and layer_idx in self.B_weights
        and module_name in self.B_weights[layer_idx]
    )

def apply(
    self,
    x:           torch.Tensor,   # [..., in_dim]
    layer_idx:   int,
    module_name: str,
) -> Optional[torch.Tensor]:     # [..., out_dim] or None
    if not self.has_layer(layer_idx, module_name):
        return None
    A = self.A_weights[layer_idx][module_name]   # [rank, in_dim]
    B = self.B_weights[layer_idx][module_name]   # [out_dim, rank]
    return (x @ A.T) @ B.T * self.scaling
```

`apply()` is called from `Qwen3Attention.forward()` and `Qwen3MLP.forward()` for each projection. The return value is `None` for untargeted modules — the caller is responsible for checking:

```python
delta = ada.apply(hidden_states, self.layer_idx, "q_proj")
if delta is not None:
    q = q + delta * mask
```

**GEMM shapes** for `q_proj` at layer 0, with a batch of 4 tokens (NOCACHE path, B=1):

```
x    :  [1, 4, 1024]        (B=1, 4 tokens, hidden_size=1024)
A.T  :  [1024, 8]           (transposed from [8, 1024])
x@A.T:  [1, 4, 8]           (low-rank intermediate)
B.T  :  [8, 2048]           (transposed from [2048, 8])
delta:  [1, 4, 2048]        (output dim = q_dim)
```

Total FLOPs for this apply: `4 × 1024 × 8 + 4 × 8 × 2048 = 98,304`. Compare to the base `q_proj` GEMM: `4 × 1024 × 2048 = 8,388,608`. The LoRA GEMM is **~1.2%** of the base GEMM cost.

---

## What `_SUPPORTED_MODULES` Contains

```python
_SUPPORTED_MODULES = {
    "q_proj", "k_proj", "v_proj", "o_proj",   # attention
    "gate_proj", "up_proj", "down_proj",        # MLP
}
```

This is the set of modules that this layer's model wiring can handle. If an adapter targets something outside this set (e.g., `"embed_tokens"`, `"lm_head"`), those entries are silently dropped at init time:

```python
self.target_modules = target_modules & _SUPPORTED_MODULES
```

For `phh/Qwen3-0.6B-TLDR-Lora`:
- `target_modules` from config: `{"q_proj", "v_proj"}`
- After intersection: `{"q_proj", "v_proj"}` (no change — both are in `_SUPPORTED_MODULES`)
