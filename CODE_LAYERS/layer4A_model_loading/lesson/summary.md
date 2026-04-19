# Layer 4A — Summary

Layer 4A takes ownership of two things that HuggingFace previously handled invisibly: **reading the config** and **loading the weights**. The forward computation is still HuggingFace's — `self._model` is an `AutoModelForCausalLM` under the hood. Layer 4B replaces it with our own layers. The value of doing loading first, separately, is that `from_pretrained`, `load_weights`, and `Qwen3Config` are explained once here and carried forward unchanged into every subsequent layer.

---

## From Layer 3 to Layer 4A

In Layer 3, two lines handed everything to HuggingFace:

```python
# Layer 3
from transformers import AutoModelForCausalLM
self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
```

The forward call returned a `ModelOutput` namedtuple. The KV cache had to be extracted and re-assigned on every decode step:

```python
# Layer 3 — forward call
out = self.model(input_ids=ids, attention_mask=mask,
                 past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values   # re-assign every step
logits  = out.logits[:, -1, :]
```

In Layer 4A, the init becomes:

```python
# Layer 4A
from model import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

The forward call returns `(logits, past_kv)` directly. `KVCache` — Layer 3's HF-compatible cache — is passed as `past_key_values` and updated in-place by HF's attention layers, so the returned second value can be discarded:

```python
# Layer 4A — forward call
kv = KVCache()
logits, _ = self.model(ids, attention_mask=mask,
                       past_key_values=kv, position_ids=pos)
logits = logits[:, -1, :]   # kv already updated in-place, _ discarded
```

Everything else — tokenizer, left-padding, cumsum position IDs, pad injection, mask extension, finished mask, `sample_batch` — is byte-for-byte identical to Layer 3. The new code lives entirely in `model/config.py` and `model/qwen3.py`.

---

## How HuggingFace Hosts Models

Every model on HuggingFace is identified by a model ID of the form `owner/name` — for example, `Qwen/Qwen3-0.6B`. This maps to a git repository on `huggingface.co`. The repository contains several files: `config.json` (architecture hyperparameters), `tokenizer.json` (BPE vocabulary and merge rules), `model.safetensors` (all weights, ~1.1 GB for Qwen3-0.6B), and a few others like `generation_config.json`. Larger models split weights across numbered shard files.

`_resolve_model_path` handles both a local directory and a Hub model ID:

```python
def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path    # fast path — already on disk

    from huggingface_hub import snapshot_download
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    return Path(snapshot_download(model_path, local_files_only=offline))
```

If the argument is a directory that already contains `config.json`, it is returned immediately — no network call. Otherwise `snapshot_download` mirrors the repository to `~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<commit_hash>/` and returns that path. Subsequent calls with the same model ID return the cached copy without re-downloading. `HF_HUB_OFFLINE=1` makes `snapshot_download` raise an error rather than attempt any network request — used in air-gapped clusters where the cache has been pre-populated.

---

## `Qwen3Config` — a Plain Dataclass

`from_pretrained` begins by reading the model's hyperparameters. `config.json` in the repository contains roughly 25 keys — architecture fields we need, plus metadata for HuggingFace's own machinery (`architectures`, `model_type`, `transformers_version`, rope scaling sub-objects). `Qwen3Config.from_json` reads the file and extracts only the 12 fields the dataclass declares:

```python
@dataclass
class Qwen3Config:
    vocab_size: int = 151_936
    hidden_size: int = 1_024
    num_hidden_layers: int = 28
    intermediate_size: int = 3_072
    num_attention_heads: int = 16   # Q heads
    num_key_value_heads: int = 8    # KV heads (GQA)
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads   # 16 // 8 = 2
```

HuggingFace's `PretrainedConfig` adds roughly a thousand lines of serialisation, hub-download, and legacy-compatibility machinery. All we need are the numeric hyperparameters, so a dataclass suffices. `from_json` uses `d.get(key, default)` for every field — unknown keys in the JSON are silently ignored, and a partial config (useful in unit tests) populates with sensible defaults. `num_kv_groups` is a derived property rather than a stored field to prevent the two head counts from going out of sync. In Layer 4B it will be used explicitly to expand KV heads via `repeat_kv`.

---

## `from_pretrained` — Six Steps

`Qwen3ForCausalLM.from_pretrained` runs six steps:

```python
@classmethod
def from_pretrained(cls, model_path, dtype=torch.bfloat16):
    model_dir  = _resolve_model_path(model_path)                    # Step 1 — local or Hub
    config     = Qwen3Config.from_json(model_dir / "config.json")   # Step 2 — our dataclass
    hf_config  = AutoConfig.from_pretrained(model_dir)              # Step 3 — HF skeleton
    hf_model   = AutoModelForCausalLM.from_config(hf_config)        #
    model      = cls(config, hf_model)                              #
    model      = model.to(dtype)                                    # Step 4 — cast BEFORE copy
    model.load_weights(_iter())                                     # Step 5 — stream weights
    return model.to("cuda").eval()                                  # Step 6
```

Step 2 reads our `Qwen3Config` — we carry this object throughout. Step 3 builds the HF model skeleton: `AutoConfig.from_pretrained` reads `config.json` into HF's own `PretrainedConfig`, `AutoModelForCausalLM.from_config` constructs the full Qwen3 architecture with random weights. Crucially, `from_config` calls `tie_weights()` internally, making `lm_head.weight` and `embed_tokens.weight` point to the same storage before any checkpoint values are written. `cls(config, hf_model)` wraps both into our `Qwen3ForCausalLM`.

Casting to `bfloat16` before copying weights (Step 4) means `copy_` writes `bfloat16` directly into `bfloat16` parameters — one memory operation. Casting after copying would require two passes over the parameter memory.

---

## The `safetensors` Format

Model weights are stored in `model.safetensors` rather than the old pickle-based `.bin` format. The file layout is a short JSON header followed by flat binary tensor data:

```
[8 bytes: header length]
[JSON header: name → {dtype, shape, byte_offsets}]
[raw tensor bytes, densely packed]
```

Because every tensor's position is encoded in the header, `safe_open` can open the file as a memory-mapped region and access any tensor by reading its byte offset from the header — no deserialisation, no temporary buffer, no code execution. `safetensors` is safe to load from untrusted sources in a way that pickle `.bin` files are not.

For Qwen3-0.6B, the file contains 290 tensors. The names match HuggingFace's module hierarchy exactly — `model.layers.0.self_attn.q_proj.weight`, `model.layers.0.mlp.gate_proj.weight`, and so on — which means they also match the keys in `dict(self._model.named_parameters())`, making the lookup in `load_weights` a direct dictionary access.

---

## `load_weights` — Streaming One Tensor at a Time

`load_weights` receives a generator of `(name, tensor)` pairs and copies each into the matching HF parameter:

```python
def load_weights(self, weights):
    params = dict(self._model.named_parameters())
    for name, tensor in weights:
        if name == "lm_head.weight" and self.config.tie_word_embeddings:
            continue                    # already tied by HF's from_config
        if name in params:
            params[name].data.copy_(tensor)
```

The generator is built with `safe_open`:

```python
def _iter():
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key).to(dtype)
```

`f.get_tensor(key)` returns a tensor backed by the memory-mapped file — no copy from disk until the bytes are accessed. `.to(dtype)` casts to `bfloat16` in a new CPU tensor. `copy_` writes that into the model parameter's buffer in-place. The previous tensor then goes out of scope and is freed. Peak memory during loading is at most one extra tensor — the largest is `embed_tokens.weight` at ~310 MB.

The `lm_head.weight` key is skipped because `from_config` already called `tie_weights()`, making `lm_head` and `embed_tokens` point to the same storage. Copying `embed_tokens.weight` from the checkpoint also updates `lm_head` automatically. Loading `lm_head.weight` separately would overwrite that storage again with identical values — a redundant 310 MB copy.

---

## What Comes Next

Layer 4A owns loading but not the forward pass. `self._model` inside `Qwen3ForCausalLM` is still HF's implementation — 28 layers of RMSNorm, RoPE, Attention, MLP, and the causal mask that HF builds internally.

Layer 4B replaces `self._model` with our own `Qwen3Model`. The `from_pretrained`, `load_weights`, and `Qwen3Config` code from Layer 4A carries over unchanged — the loading story is told once and done. The new work in 4B is the forward computation: `RMSNorm`, `RotaryEmbedding`, `Qwen3Attention` (GQA + per-head QK norm + SDPA), `Qwen3MLP` (SwiGLU), `Qwen3DecoderLayer`, and the explicit `_build_additive_mask`. The `KVCache` interface also shifts from HF's signature (`update(key, value, layer_idx)`) to our own (`update(layer_idx, key, value)`) so our attention layers can call it directly.
