# 02 — Config Loading

## How HuggingFace Hosts Models

Every model on HuggingFace is identified by a model ID of the form `owner/name` — for example, `Qwen/Qwen3-0.6B`. This ID maps to a git repository on `huggingface.co`. Visiting the page at `https://huggingface.co/Qwen/Qwen3-0.6B` shows a model card describing the architecture, training data, and benchmarks, followed by a "Files and versions" tab that lists every file in the repository.

For Qwen3-0.6B, the repository contains:

```
config.json             # architecture hyperparameters
generation_config.json  # default generation settings (temperature, top-p, etc.)
tokenizer.json          # full BPE vocabulary and merge rules
tokenizer_config.json   # tokenizer class name and chat template
vocab.json              # token-to-ID mapping (redundant with tokenizer.json for BPE)
merges.txt              # BPE merge pairs
model.safetensors       # all model weights in a single file (~1.1 GB)
```

Larger models split weights across multiple shard files: `model-00001-of-00004.safetensors`, `model-00002-of-00004.safetensors`, and so on, plus a `model.safetensors.index.json` that maps each tensor name to the shard file containing it.

Under the hood, model files are stored on Cloudflare R2 object storage and served via a CDN. The `huggingface_hub` Python package handles all download, caching, and authentication — a single call to `snapshot_download("Qwen/Qwen3-0.6B")` mirrors the entire repository to disk.

---

## `_resolve_model_path` — Local or Hub

`from_pretrained` accepts either a local path or a Hub model ID. The helper function `_resolve_model_path` decides which:

```python
def _resolve_model_path(model_path: str) -> Path:
    path = Path(model_path)
    if path.is_dir() and (path / "config.json").exists():
        return path

    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path, local_files_only=offline))
```

If the argument is a directory that already contains `config.json`, it is returned immediately — no network calls. This is the fast path for workflows where the model has already been downloaded. Otherwise, `snapshot_download` is called with the Hub model ID.

`snapshot_download` downloads all files in the repository to a cache directory on disk. By default this is `~/.cache/huggingface/hub/`. The exact path on disk is determined by a hash of the model ID and the commit revision:

```
~/.cache/huggingface/hub/
  models--Qwen--Qwen3-0.6B/
    blobs/              # raw file contents, named by their SHA-256 hash
    refs/
      main              # text file containing the latest commit hash
    snapshots/
      <commit_hash>/    # symlinks into blobs/, one per file
        config.json
        model.safetensors
        tokenizer.json
        ...
```

`snapshot_download` returns the path to the `snapshots/<commit_hash>/` directory. Subsequent calls with the same model ID check the `refs/main` file to see if the remote commit has changed; if not, the cached snapshot is returned without downloading anything.

The `HF_HUB_OFFLINE=1` environment variable makes `snapshot_download` raise an error rather than attempt any network requests — useful in air-gapped clusters where the cache has been pre-populated. Setting `HF_HOME` or `HUGGINGFACE_HUB_CACHE` redirects the cache to a different directory, which is common on HPC systems with shared network storage.

---

## `config.json` — What's Inside

Once `model_dir` is resolved, `from_pretrained` reads two different things from it. First, `config.json`:

```json
{
  "architectures": ["Qwen3ForCausalLM"],
  "model_type": "qwen3",
  "vocab_size": 151936,
  "hidden_size": 1024,
  "num_hidden_layers": 28,
  "num_attention_heads": 16,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 3072,
  "hidden_act": "silu",
  "rms_norm_eps": 1e-6,
  "rope_theta": 1000000.0,
  "attention_bias": false,
  "tie_word_embeddings": true,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.51.0",
  ...
}
```

The actual file contains roughly 25 keys. Many of them — `architectures`, `model_type`, `torch_dtype`, `transformers_version`, the rope scaling sub-object — are metadata that only HuggingFace's own loading machinery uses. Our `Qwen3Config.from_json` reads `config.json` once and extracts only the numeric and boolean fields that matter for the computation.

---

## `Qwen3Config` — a Plain Dataclass

The config is represented in `model/config.py` as a `@dataclass`:

```python
@dataclass
class Qwen3Config:
    vocab_size: int = 151_936

    hidden_size: int = 1_024
    num_hidden_layers: int = 28
    intermediate_size: int = 3_072

    num_attention_heads: int = 16   # Q heads
    num_key_value_heads: int = 8    # KV heads (GQA: each KV head shared by 2 Q heads)
    head_dim: int = 128

    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0

    attention_bias: bool = False
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    @property
    def num_kv_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads
```

HuggingFace's `PretrainedConfig` adds roughly a thousand lines of serialisation, hub-download, and legacy-compatibility machinery. None of that is needed here — we just read the JSON and populate typed fields. The `@dataclass` decorator generates `__init__`, `__repr__`, and `__eq__` automatically, so `Qwen3Config` can be printed, compared, and instantiated by keyword without any boilerplate.

The `from_json` classmethod is twelve lines:

```python
@classmethod
def from_json(cls, path: str | Path) -> "Qwen3Config":
    with open(path) as f:
        d = json.load(f)
    return cls(
        vocab_size=d.get("vocab_size", 151_936),
        hidden_size=d.get("hidden_size", 1_024),
        num_hidden_layers=d.get("num_hidden_layers", 28),
        intermediate_size=d.get("intermediate_size", 3_072),
        num_attention_heads=d.get("num_attention_heads", 16),
        num_key_value_heads=d.get("num_key_value_heads", 8),
        head_dim=d.get("head_dim", 128),
        rms_norm_eps=d.get("rms_norm_eps", 1e-6),
        rope_theta=d.get("rope_theta", 1_000_000.0),
        attention_bias=d.get("attention_bias", False),
        tie_word_embeddings=d.get("tie_word_embeddings", True),
        hidden_act=d.get("hidden_act", "silu"),
    )
```

Every field uses `d.get(key, default)` rather than `d[key]`. This means the dataclass can be instantiated from a partial config — useful in unit tests where only a subset of fields are relevant. Unknown keys in `config.json` are silently ignored; the full Qwen3 config file (with its rope scaling sub-objects, sliding window fields, and version metadata) loads cleanly without extra parsing logic.

---

## The `num_kv_groups` Property

Qwen3-0.6B uses Grouped Query Attention (GQA): 16 query heads share 8 key/value head pairs, with 2 Q heads per KV head. The `num_kv_groups` property encodes this ratio:

```python
@property
def num_kv_groups(self) -> int:
    return self.num_attention_heads // self.num_key_value_heads   # 16 // 8 = 2
```

In Layer 4 this value is not used directly by our code — HF reads the GQA ratio from its own config object. In Layer 5, our `Qwen3Attention` implementation will use `num_kv_groups` explicitly to expand the KV heads via `repeat_kv` before computing attention scores. It is declared as a property rather than a stored field because it is entirely determined by the two head counts — having both the raw values and the derived ratio stored separately would risk them going out of sync.

---

## Building the HF Model Skeleton

With `config` in hand, `from_pretrained` needs to construct the neural network before any weights can be copied into it. In Layer 4, the forward computation is still HuggingFace's — so the skeleton must be an HF model:

```python
from transformers import AutoConfig, AutoModelForCausalLM

hf_config = AutoConfig.from_pretrained(model_dir)       # HF's PretrainedConfig
hf_model  = AutoModelForCausalLM.from_config(hf_config) # architecture, random weights
model     = cls(config, hf_model)                       # our wrapper
```

`AutoConfig.from_pretrained(model_dir)` reads `config.json` a second time — into HF's own `PretrainedConfig` object. This is needed because `AutoModelForCausalLM.from_config` expects a `PretrainedConfig`, not our dataclass. The two config objects co-exist: HF's is used to build the architecture; ours is used for everything we control (tied weights check, dtype handling, and later GQA ratios in Layer 5).

`AutoModelForCausalLM.from_config` constructs the full `Qwen3ForCausalLM` architecture — all 28 decoder layers with their attention, MLP, and norm modules — initialised with random weights from `nn.init`. No checkpoint is loaded; this is a skeleton. Crucially, `from_config` calls `tie_weights()` internally, which makes `lm_head.weight` and `embed_tokens.weight` point to the same storage before any real weights are written. This is important for the loading step covered in section 03.

`cls(config, hf_model)` wraps both into our `Qwen3ForCausalLM`. At this point the model exists in memory as randomly initialised `float32` tensors on CPU. The next step — cast to `bfloat16` before copying weights — is covered at the start of section 03.
