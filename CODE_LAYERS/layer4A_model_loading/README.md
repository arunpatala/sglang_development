# Layer 4 — Custom Model Init

**Previous layer:** Layer 3 used `AutoModelForCausalLM.from_pretrained`.  
**This layer:** We implement the Qwen3 architecture ourselves and load weights from `model.safetensors` directly, replacing HuggingFace's opaque model factory.

The computation is **identical** to Layer 3. Throughput should match within ~5%.  
What changes is **ownership** — every tensor, every weight, every forward-pass step is in code we can read and modify.

---

## File layout

```
layer4_custom_model/
    model/                      ← Python package: our Qwen3 implementation
        __init__.py             ← exports Qwen3ForCausalLM, Qwen3Config
        config.py               ← Qwen3Config dataclass (reads config.json)
        norm.py                 ← RMSNorm
        rope.py                 ← RotaryEmbedding, rotate_half, apply_rotary_pos_emb
        attention.py            ← Qwen3Attention (GQA + per-head QK norm + KV cache)
        mlp.py                  ← Qwen3MLP (SwiGLU)
        decoder_layer.py        ← Qwen3DecoderLayer (pre-norm + residuals)
        qwen3.py                ← Qwen3Model, Qwen3ForCausalLM, load_weights()
    kv_cache.py                 ← clean KVCache (no HF interface needed)
    tokenizer.py                ← unchanged from Layer 3
    model_runner.py             ← BatchedModel (same generate_batch, new model)
    server.py                   ← FastAPI, port 8104
    benchmark.py                ← batch-size sweep, same as Layer 3
```

---

## The two-line diff in model_runner.py

```python
# Layer 3
from transformers import AutoModelForCausalLM
self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")

# Layer 4
from model import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

Everything else in `model_runner.py` is unchanged — including the forward call signature:

```python
# Layer 3 (HuggingFace output object)
out = self.model(input_ids=ids, attention_mask=mask, past_key_values=kv, use_cache=True)
past_kv = out.past_key_values   # must re-assign every step
logits  = out.logits[:, -1, :]

# Layer 4 (our model: returns logits, mutates KVCache in-place)
logits = self.model(ids, attention_mask=mask, kv_cache=kv)
logits = logits[:, -1, :]       # kv is already updated, no re-assign needed
```

---

## Architecture: Qwen3-0.6B

| Hyperparameter | Value |
|---|---|
| `hidden_size` | 1024 |
| `num_hidden_layers` | 28 |
| `num_attention_heads` (Q) | 16 |
| `num_key_value_heads` (KV) | 8 → GQA ratio 2 |
| `head_dim` | 128 |
| `intermediate_size` | 3072 |
| `rope_theta` | 1,000,000 |
| `tie_word_embeddings` | True |

### Qwen3-specific details

- **Per-head QK RMSNorm** (`q_norm`, `k_norm`, weight shape `[128]`): applied to each head's Q and K vectors before RoPE. Not present in Llama or Qwen2 — it stabilises attention at scale.
- **Grouped Query Attention**: 8 KV heads serve 16 Q heads. `repeat_kv` expands K/V from 8→16 heads using `expand` (no copy).
- **Tied weights**: `lm_head.weight` shares memory with `embed_tokens.weight`. After `load_weights()` completes, `self.lm_head.weight = self.model.embed_tokens.weight`.

---

## Extensibility surfaces

The `model/` package is structured so each optimisation touches exactly one file:

| Optimisation | File(s) |
|---|---|
| Fuse QKV projections → one matmul | `attention.py` + `qwen3.py::load_weights` |
| Fuse gate+up → gate_up_proj | `mlp.py` + `qwen3.py::load_weights` |
| Flash Attention / SDPA backend swap | `attention.py` forward only |
| Tensor parallelism | Replace `nn.Linear` in `attention.py` / `mlp.py` |
| Paged KV cache | `kv_cache.py` + `attention.py` |
| FP8 / int8 quantisation | `qwen3.py::load_weights` (quantise on load) |

---

## Running

```bash
# Terminal 1 — start server
cd CODE_LAYERS/layer4_custom_model
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python server.py

# Terminal 2 — run benchmark
cd CODE_LAYERS/layer4_custom_model
python benchmark.py
```
