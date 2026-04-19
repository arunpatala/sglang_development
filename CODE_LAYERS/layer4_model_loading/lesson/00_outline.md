# Layer 4 — Lesson Outline

## What This Lesson Covers

Layer 3 called `AutoModelForCausalLM.from_pretrained` — HuggingFace resolved the path, parsed `config.json`, built the architecture, loaded the weights, and returned a `ModelOutput` object. We owned none of it.

Layer 4 takes ownership of exactly two things: **reading the config** and **loading the weights**. Before either can happen, the model needs to be located: `_resolve_model_path` accepts either a local directory or a HuggingFace Hub model ID (`Qwen/Qwen3-0.6B`), calling `snapshot_download` to mirror the repository into `~/.cache/huggingface/hub/` when needed.

Config loading replaces HuggingFace's `PretrainedConfig` with `Qwen3Config`, a plain `@dataclass` with a `from_json()` classmethod. It reads only the numeric fields from `config.json`; the 13 other keys HuggingFace includes are silently ignored. An HF model skeleton is then built via `AutoModelForCausalLM.from_config` — architecture constructed, weights random, `tie_weights()` already called.

Weight loading streams tensors one at a time from `model.safetensors` using `safe_open`. The `safetensors` format stores a JSON header mapping each tensor name to its byte offset, making `mmap`-backed zero-copy access possible. Weights are cast to `bfloat16` before the copy loop begins, so each `copy_` is a single same-dtype operation. Peak memory during loading is the size of one tensor.

The forward pass is still HuggingFace's Qwen3 implementation — `self._model` is an `AutoModelForCausalLM` under the hood. The change in `model_runner.py` is two lines at init time and the `logits, _ = self.model(...)` tuple unpack. Everything else — tokenizer, left-padding, position IDs, `KVCache`, finished mask, `sample_batch`, `server.py`, `benchmark.py` — is carried over from Layer 3 unchanged. The new code lives entirely in `model/config.py` and `model/qwen3.py`.

Layer 5 takes the next step: replacing `self._model` with our own `Qwen3Model` (RMSNorm, RoPE, Attention, MLP, DecoderLayer) so we own the forward pass too.

Progression:
- **Layer 3**  → HF loads + HF forward + HF `past_key_values`
- **Layer 4** → **we load** + HF forward + `KVCache` (HF-compatible)
- **Layer 5** → we load + **we forward** + our in-place `KVCache`

---

## Sections

### 01 — The Model Runner (`01_the_decode_loop.md`)
- Layer 3's `AutoModelForCausalLM.from_pretrained` vs Layer 4's `Qwen3ForCausalLM.from_pretrained` — the two-line diff in `model_runner.py`
- Layer 3 had `past_kv = out.past_key_values` re-assignment on every decode step; Layer 4 passes `KVCache()` (Layer 3's HF-compatible cache) as `past_key_values` — HF calls `kv.update()` in-place, no re-assign needed
- `logits, _ = self.model(...)` — forward returns `(logits, past_kv)` tuple; the second value is discarded because `kv` is already updated
- Everything else in `generate_batch` — tokenizer, left-padding, cumsum position IDs, pad injection, mask extension, finished mask, `sample_batch` — is byte-for-byte identical to Layer 3
- `verify.py` confirms numerical parity with HuggingFace outputs (trivially exact since we use the same HF forward); `verify_batch.py` checks B=4 batched vs 4×B=1 individual runs

### 02 — Config Loading (`02_config_loading.md`)
- HuggingFace Hub: model IDs (`Qwen/Qwen3-0.6B`), repository structure (`config.json`, `tokenizer.json`, `model.safetensors`), Cloudflare R2 storage, CDN delivery
- `_resolve_model_path`: local directory fast-path vs `snapshot_download`; cache layout (`~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<hash>/`); `HF_HUB_OFFLINE=1` for air-gapped clusters
- `config.json` contents: the ~25 keys HuggingFace includes and the ~12 we extract
- `Qwen3Config` as a plain `@dataclass` with `from_json()`: reads only the numeric fields; no `PretrainedConfig` inheritance, no hub-download machinery, no legacy compatibility shims; `d.get(key, default)` tolerates partial configs
- `num_kv_groups` property: 16 Q heads ÷ 8 KV heads = 2; derived at runtime rather than stored to avoid redundancy; used explicitly in Layer 5's `repeat_kv`
- Building the HF model skeleton: `AutoConfig.from_pretrained` → `AutoModelForCausalLM.from_config` → `cls(config, hf_model)`; `from_config` calls `tie_weights()` internally before returning

### 03 — Weight Loading (`03_weight_loading.md`)
- `safetensors` format: JSON header with tensor name → dtype/shape/byte-offset, raw binary data, `mmap`-backed access — no pickle, no code execution, no double-buffer spike
- What is in `model.safetensors`: 290 tensors for Qwen3-0.6B; embed + 28×(norm, q/k/v/o, q_norm, k_norm, gate/up/down) + final norm + lm_head; `lm_head.weight` and `embed_tokens.weight` both appear as independent 310 MB entries
- Cast before copy: `model.to(dtype)` converts skeleton parameters to `bfloat16` before `load_weights` runs — `copy_` then writes same-dtype in one pass instead of two
- Streaming iterator with `safe_open`: yields `(name, tensor)` one at a time; each tensor freed after `copy_`; peak additional memory = one tensor (~310 MB for embed, ~2 MB for a layer weight)
- `load_weights()`: `dict(self._model.named_parameters())` flat lookup; `params[name].data.copy_(tensor)` in-place with no allocation; checkpoint names match HF hierarchy exactly
- Tied weights: HF's `from_config` already called `tie_weights()` — `lm_head.weight` and `embed_tokens.weight` share storage; the `continue` skips the redundant second copy of 310 MB from the checkpoint

### 04 — What Comes Next (`04_whats_next.md`)
- Layer 4 owns loading but not the forward pass — `self._model` is still HF's `Qwen3ForCausalLM` under the hood
- Layer 5 replaces `self._model` with our own `Qwen3Model`: RMSNorm, SwiGLU MLP, RotaryEmbedding, Qwen3Attention (GQA + per-head QK norm + SDPA), Qwen3DecoderLayer, and the additive causal-plus-padding mask
- The `from_pretrained` / `load_weights` / `Qwen3Config` code in Layer 4 carries over to Layer 5 unchanged — the loading story is told once, here
- `KVCache` in 5 switches from the HF-compatible interface (`update(key, value, layer_idx)`) to our own interface (`update(layer_idx, key, value)`) because our attention layers call it directly
- Layer 6 (`layer6_continuous_batching`) adds a `Scheduler`, `Request`, and `Batch` object on top of 5's model; the `model/` package and weight loading are not touched again

---

## Supporting Files

- `summary.md` — prose overview of the config and weight loading concepts
- `sglang_reference.md` — maps layer 4 concepts to SGLang source: `Qwen3Config` → `sglang/srt/models/qwen3.py` config handling; `load_weights()` iterator style → SGLang's `stacked_params_mapping`; `from_pretrained` → SGLang's `ModelRunner.load_model`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Model init diff | `model_runner.py` line 50: `self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)` |
| KVCache created | `model_runner.py` line 99: `kv = KVCache()` |
| Prefill forward | `model_runner.py` line 101: `logits, _ = self.model(input_ids, ..., past_key_values=kv, position_ids=prefill_pos)` |
| Decode forward | `model_runner.py` line 143: `logits, _ = self.model(current, ..., past_key_values=kv, position_ids=decode_pos)` |
| `Qwen3Config` dataclass | `model/config.py` line 23: `@dataclass class Qwen3Config:` |
| `Qwen3Config.from_json` | `model/config.py` line 62: `cls(vocab_size=d.get(...), ...)` |
| `from_pretrained` steps | `model/qwen3.py` line 131: `@classmethod def from_pretrained(...)` |
| HF model skeleton | `model/qwen3.py` line 159: `hf_config = AutoConfig.from_pretrained(model_dir)` |
| Cast before copy | `model/qwen3.py` line 163: `model = model.to(dtype)` |
| Weight streaming | `model/qwen3.py` line 171: `with safe_open(str(weights_path), framework="pt", device="cpu") as f:` |
| `load_weights` iterator | `model/qwen3.py` line 97: `def load_weights(self, weights: Iterable[...])` |
| Weight copy | `model/qwen3.py` line 119: `params[name].data.copy_(tensor)` |
| Tied weights skip | `model/qwen3.py` line 116: `if name == "lm_head.weight" and self.config.tie_word_embeddings: continue` |
| Forward delegate to HF | `model/qwen3.py` line 86: `out = self._model(input_ids=..., past_key_values=..., use_cache=True)` |
