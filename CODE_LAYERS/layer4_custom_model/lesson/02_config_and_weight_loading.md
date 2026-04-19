# 02 — Config and Weight Loading

## The Entry Point

The generate loop in section 01 calls `self.model(...)`. Before any request arrives, `BatchedModel.__init__` populates `self.model` with a single call:

```python
self.model = Qwen3ForCausalLM.from_pretrained(
    model_path,
    dtype=torch.bfloat16,
)
```

In Layer 3, the equivalent line was `AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")`. HuggingFace resolved the model path, read `config.json`, built the architecture, loaded the weights, moved to CUDA, and returned an opaque object. In Layer 4, `from_pretrained` is a classmethod we wrote — every step is visible.

---

## `Qwen3Config` — a Plain Dataclass

`from_pretrained` begins by reading the model's hyperparameters. The config is represented as a `@dataclass` in `model/config.py`:

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

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
```

HuggingFace's `PretrainedConfig` adds roughly a thousand lines of serialisation, hub-download, and legacy-compatibility machinery. None of that is needed here. A dataclass gives us typed fields with defaults, a single `from_json` classmethod that reads only the keys we care about, and a `num_kv_groups` property derived from the two head counts. The `num_kv_groups` value (16 ÷ 8 = 2) tells `Qwen3Attention` how many Q heads share each KV head, which `repeat_kv` uses in section 06.

---

## `from_pretrained` — Five Steps

`Qwen3ForCausalLM.from_pretrained` runs five steps in sequence:

```python
@classmethod
def from_pretrained(cls, model_path: str, dtype: torch.dtype = torch.bfloat16):
    model_dir = _resolve_model_path(model_path)          # Step 1
    config    = Qwen3Config.from_json(model_dir / "config.json")  # Step 2
    model     = cls(config)                              # Step 3
    model     = model.to(dtype)                          # Step 4 — cast before copy
    model.load_weights(_iter_safetensors(model_dir / "model.safetensors", dtype))
    model     = model.to("cuda").eval()                  # Step 5
    return model
```

`_resolve_model_path` accepts either a local directory or a HuggingFace Hub model ID. If the path is a directory containing `config.json`, it is returned directly. Otherwise `snapshot_download` is called, respecting `HF_HUB_OFFLINE=1` for air-gapped environments.

`Qwen3Config.from_json` opens `config.json` and calls `cls(vocab_size=d.get(...), ...)`, populating only the fields the dataclass declares. Unknown keys in the JSON are silently ignored.

`cls(config)` constructs the full `Qwen3ForCausalLM` on CPU using PyTorch's default `float32` init. No weights are meaningful yet — all parameters hold random values from `nn.init`.

The dtype cast on Step 4 happens before weights are copied, not after. This is deliberate: if the model were cast after copying, `copy_` would first write `float32` values into `float32` parameters, then recast the entire tensor — two memory operations. Casting first ensures `copy_` writes `bfloat16` directly into `bfloat16` parameters — one operation.

Step 5 moves the fully initialised model to CUDA and sets `eval()` mode, disabling dropout and batch norm tracking (neither is present in Qwen3, but `eval()` is idiomatic for inference).

---

## `load_weights` — Streaming One Tensor at a Time

HuggingFace's default weight loading reads all tensors into a temporary buffer before copying them into the model parameters — a double-memory spike. For Qwen3-0.6B this is manageable, but for a 70B model with 140 GB of weights it is a serious problem.

`load_weights` avoids this by accepting an iterator of `(name, tensor)` pairs and processing them one at a time:

```python
def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> None:
    params = dict(self.named_parameters())
    loaded: set[str] = set()

    for name, tensor in weights:
        if name == "lm_head.weight" and self.config.tie_word_embeddings:
            continue                    # handled after the loop
        if name in params:
            params[name].data.copy_(tensor)
            loaded.add(name)

    if self.config.tie_word_embeddings:
        self.lm_head.weight = self.model.embed_tokens.weight
```

`dict(self.named_parameters())` builds a flat name-to-parameter mapping. The weight key names in `model.safetensors` match the module hierarchy exactly — `"model.layers.0.self_attn.q_proj.weight"`, `"model.layers.0.mlp.gate_proj.weight"`, and so on — so a simple `name in params` lookup suffices. `params[name].data.copy_(tensor)` writes the checkpoint tensor into the parameter's data buffer in-place, without allocating a new tensor.

The iterator that feeds `load_weights` is built with `safe_open` from the `safetensors` library:

```python
def _iter():
    with safe_open(str(weights_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            yield key, f.get_tensor(key).to(dtype)
```

`safe_open` reads one tensor from the file at a time, without loading the entire file into memory. `f.get_tensor(key)` memory-maps the relevant bytes and returns a CPU tensor. `.to(dtype)` casts it to `bfloat16` before yielding. By the time `copy_` runs, the tensor is already the right dtype and the previous tensor has gone out of scope and been freed. Peak memory during loading is at most one extra tensor at a time — the smallest possible overhead.

---

## Tied Weights

After the loop, if `tie_word_embeddings` is true in the config, one assignment completes the weight setup:

```python
self.lm_head.weight = self.model.embed_tokens.weight
```

This makes `lm_head.weight` and `embed_tokens.weight` the same Python object pointing to the same underlying storage. The model now has one copy of the embedding matrix in GPU memory instead of two. For Qwen3-0.6B, `vocab_size × hidden_size = 151_936 × 1024 = ~155M` parameters at `bfloat16` takes roughly 310 MB. Tying saves that entire allocation. For models like Qwen3-72B where `hidden_size = 7168`, the saving is over 2 GB.

The `continue` in `load_weights` skips copying `"lm_head.weight"` from the checkpoint for the same reason: the checkpoint contains a copy of the embedding matrix stored under the `lm_head.weight` key, but copying it would be redundant since the tie assignment handles it. Loading it first and then pointing `lm_head.weight` at `embed_tokens.weight` would orphan the just-copied tensor.

After `from_pretrained` returns, `self.model` holds a fully initialised `Qwen3ForCausalLM` with 622M parameters correctly loaded, `lm_head` and `embed_tokens` sharing memory, all tensors in `bfloat16` on CUDA. The sections that follow explain what happens inside each forward call.
