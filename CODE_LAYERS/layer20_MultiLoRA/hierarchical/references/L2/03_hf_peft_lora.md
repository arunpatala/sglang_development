# LoRA Developer Guide — HuggingFace PEFT

**Source:** https://huggingface.co/docs/peft/en/developer_guides/lora
**Project:** HuggingFace PEFT
**Accessed:** April 2026
**Level:** L2 — Official production library documentation
**Why here:** PEFT is the ground-truth LoRA implementation used in `verify_lora.py` to verify Layer 20's custom implementation. Understanding the PEFT checkpoint format (`adapter_config.json`, `adapter_model.safetensors`), weight naming conventions, and adapter loading/applying code is essential for writing `LoRAAdapter._load_weights()` correctly and interpreting verification test results.

---

## Overview

LoRA is a low-rank decomposition method to reduce the number of trainable parameters, which speeds up fine-tuning large models and uses less memory.

```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                          # rank
    lora_alpha=32,                # scaling: alpha/r applied to B·A product
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)

peft_model = get_peft_model(base_model, config)
peft_model.print_trainable_parameters()
# trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0623
```

---

## LoraConfig Key Parameters

| Parameter | Description |
|---|---|
| `r` | Rank of the low-rank decomposition matrices A and B. Lower = fewer parameters. |
| `lora_alpha` | Scaling factor. The effective scaling applied is `lora_alpha / r`. |
| `target_modules` | Layer names to apply LoRA to (`q_proj`, `v_proj`, `all-linear`, etc). |
| `lora_dropout` | Dropout probability applied to LoRA input. Default: 0.0 |
| `bias` | Whether to train bias parameters (`none`, `all`, `lora_only`). |
| `task_type` | Task type for configuration helpers (`CAUSAL_LM`, `SEQ_2_SEQ_LM`, etc). |
| `init_lora_weights` | Weight initialization strategy (see below). |
| `use_rslora` | Use rank-stabilized LoRA (`lora_alpha/sqrt(r)` instead of `lora_alpha/r`). |
| `use_dora` | Use DoRA (Weight-Decomposed Low-Rank Adaptation). |

---

## Weight Initialization Strategies

### Default: Kaiming-Uniform + Zeros (identity transform at init)

```python
# A: Kaiming-uniform  →  ensures nonzero gradients from the start
# B: zeros            →  output is zero at init, so BA = 0; no initial perturbation
config = LoraConfig(...)  # default
```

This is what `phh/Qwen3-0.6B-TLDR-Lora` uses (trained from this initialization). Layer 20's `LoRAAdapter._load_weights()` loads the trained A and B matrices after fine-tuning.

### Gaussian initialization

```python
config = LoraConfig(init_lora_weights="gaussian", ...)
# A: Gaussian distribution  (used by Diffusers)
# B: zeros
```

### Other advanced strategies

- **PiSSA** (`init_lora_weights="pissa"`) — SVD on pre-trained weights; faster convergence
- **OLoRA** (`init_lora_weights="olora"`) — QR decomposition initialization; better stability
- **EVA** (`init_lora_weights="eva"`) — Data-driven SVD on input activations; adaptive rank allocation
- **LoftQ** — Minimizes quantization error for QLoRA training

---

## The LoRA Math (as implemented in PEFT)

For a linear layer with weight `W₀ ∈ ℝ^{d×k}`:

```
h = x · W₀ᵀ  +  x · Aᵀ · Bᵀ · (lora_alpha / r)
```

In PEFT source code (`peft/tuners/lora/layer.py`):

```python
# Default forward (no merged weights)
result = F.linear(x, weight)                          # x · W₀ᵀ
lora_output = self.lora_B(self.lora_A(self.lora_dropout(x)))  # x · Aᵀ · Bᵀ
result = result + lora_output * self.scaling          # scaling = lora_alpha / r
```

Layer 20 replicates this exactly in `LoRAAdapter.apply()`:

```python
def apply(self, x, layer_idx, module_name):
    A = self.A_weights[layer_idx][module_name]  # [rank, in_dim]
    B = self.B_weights[layer_idx][module_name]  # [out_dim, rank]
    return (x @ A.T) @ B.T * self.scaling       # identical math
```

---

## Checkpoint Format: What `verify_lora.py` Reads

When you call `peft_model.save_pretrained("path/")` or download from HuggingFace Hub, PEFT creates:

```
adapter_config.json          # LoraConfig serialized as JSON
adapter_model.safetensors    # A and B weight tensors
README.md                    # optional
```

### `adapter_config.json` structure

```json
{
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 16,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

This is parsed by `LoRAAdapter.__init__()` to set `self.rank`, `self.lora_alpha`, `self.scaling`, and `self.target_modules`.

### `adapter_model.safetensors` key format

Each key follows the pattern:
```
base_model.model.model.layers.{layer_idx}.{module_path}.lora_A.weight  →  A matrix [r, in_dim]
base_model.model.model.layers.{layer_idx}.{module_path}.lora_B.weight  →  B matrix [out_dim, r]
```

Example keys for Qwen3:
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight  →  A for q_proj, layer 0
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight  →  B for q_proj, layer 0
base_model.model.model.layers.0.mlp.gate_proj.lora_A.weight     →  A for gate_proj, layer 0
```

Layer 20 `LoRAAdapter._load_weights()` parses these keys to extract `layer_idx` and `module_name`.

---

## Loading a Trained Adapter for Inference

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Wrap with PEFT adapter
peft_model = PeftModel.from_pretrained(base_model, "phh/Qwen3-0.6B-TLDR-Lora")
peft_model.eval()

# Forward pass uses the adapter automatically
outputs = peft_model(**inputs)
```

This is what `verify_lora.py` does for ground-truth comparison.

---

## Adapter Merging and Unmerging

For inference-only deployment, adapters can be merged into base weights (zero overhead):

```python
merged_model = peft_model.merge_and_unload()
# Returns a standard AutoModelForCausalLM with weights = W₀ + B·A·scaling
# No more PEFT overhead, but cannot unmerge
```

For temporary merging (with ability to unmerge):

```python
peft_model.merge_adapter()    # merge into base weights
peft_model.unmerge_adapter()  # restore separate weights
```

Layer 20 uses neither — it keeps A and B separate and computes the delta at runtime (equivalent to unmerged PEFT inference). This is correct for a multi-adapter system where the base model is shared.

---

## LoRA Variants Available in PEFT

| Variant | Key parameter | Innovation |
|---|---|---|
| Standard LoRA | `r`, `lora_alpha` | Rank decomposition; baseline |
| rsLoRA | `use_rslora=True` | `lora_alpha/sqrt(r)` scaling; more stable at high r |
| DoRA | `use_dora=True` | Decomposes weight update into magnitude + direction |
| QLoRA | `load_in_4bit=True` | 4-bit NF4 quantized base + LoRA adapters |
| PiSSA | `init_lora_weights="pissa"` | SVD init from pre-trained weights; faster convergence |
| OLoRA | `init_lora_weights="olora"` | QR decomposition init; better stability |
| EVA | `init_lora_weights="eva"` | Data-driven SVD; adaptive rank per layer |

---

## Target Module Coverage for QLoRA-style Fine-Tuning

```python
# Target ALL linear layers (as in QLoRA paper)
config = LoraConfig(target_modules="all-linear", ...)

# Target specific modules only (as in Layer 20's phh/Qwen3-0.6B-TLDR-Lora)
config = LoraConfig(
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    ...
)
```

Per-layer rank overrides (for heterogeneous rank assignments):
```python
config = LoraConfig(
    r=8,                              # default rank
    rank_pattern={"q_proj": 16,       # attention Q uses higher rank
                  "v_proj": 16},      # attention V uses higher rank
    ...
)
```

---

## Relevance to Layer 20

The three most important items from this documentation for Layer 20:

1. **Checkpoint format** — `adapter_config.json` and `adapter_model.safetensors` key parsing in `LoRAAdapter._load_weights()` follows the exact PEFT naming convention described here.

2. **Math identity** — `(x @ A.T) @ B.T * scaling` in `LoRAAdapter.apply()` is identical to PEFT's `lora_B(lora_A(x)) * scaling`. This is verified by `test_lora_matches_peft` in `verify_lora.py`.

3. **Verification target** — `PeftModel.from_pretrained(base_model, "phh/Qwen3-0.6B-TLDR-Lora")` is the ground-truth reference used in `verify_lora.py:test_lora_matches_peft()`.
