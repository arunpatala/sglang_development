# Layer 4 — Lesson Outline

## What This Lesson Covers

Layer 3 called `AutoModelForCausalLM.from_pretrained` — HuggingFace built the architecture, owned the forward pass, and returned a `ModelOutput` object whose `past_key_values` had to be re-assigned every decode step. Layer 4 replaces that with our own `Qwen3ForCausalLM`, implemented across a new `model/` package. The forward call now returns a logits tensor directly and mutates a `KVCache` object in-place, so the generate loop needs no re-assignment.

The change in `model_runner.py` is two lines. Everything else — tokenizer, left-padding, position IDs, finished mask, `sample_batch`, `server.py`, `benchmark.py` — is carried over from Layer 3 unchanged. The new code lives entirely inside `model/`: eight files that implement `Qwen3Config`, `RMSNorm`, `RotaryEmbedding`, `Qwen3MLP`, `Qwen3Attention`, `Qwen3DecoderLayer`, `Qwen3Model`, and `Qwen3ForCausalLM`.

The sections follow `model_runner.py` for section 01, then drill into the `model/` package in execution order: config and weight loading, the decoder block and its norms, rotary embeddings, the additive mask, and attention with GQA and per-head QK norm.

---

## Sections

### 01 — The Model Runner (`01_the_decode_loop.md`)
- Layer 3's `AutoModelForCausalLM.from_pretrained` vs Layer 4's `Qwen3ForCausalLM.from_pretrained` — the two-line diff in `model_runner.py`
- What `ModelOutput` hid: `out.past_key_values` re-assignment every step, `use_cache=True` flag, opaque forward internals
- Layer 4's forward call returns logits directly; `kv_cache` is mutated in-place by each attention layer — no re-assign needed
- Everything else in `generate_batch` — tokenizer, left-padding, prefill position IDs, finished mask, decode position IDs — is identical to Layer 3
- `verify.py` and `verify_batch.py` confirm numerical parity with HuggingFace outputs within bfloat16 tolerance

### 02 — Config and Weight Loading (`02_config_and_weight_loading.md`)
- `Qwen3Config` as a plain `@dataclass` with `from_json()`: reads only the numeric fields from `config.json`; no `PretrainedConfig` inheritance, no hub-download machinery
- `from_pretrained()` five steps: resolve path (`_resolve_model_path`) → `Qwen3Config.from_json` → `cls(config)` on CPU → `.to(dtype)` before copying weights → stream tensors via `safe_open` → `.to("cuda").eval()`
- `load_weights()` SGLang-style iterator: `dict(self.named_parameters())`, `params[name].data.copy_(tensor)` — streaming one tensor at a time avoids a peak-memory double-spike
- Tied weights: after the loop, `self.lm_head.weight = self.model.embed_tokens.weight`; why this matters (~600 MB saved for large models) and how `tie_word_embeddings` in `config.json` controls it

### 03 — RMSNorm and the Decoder Layer (`03_rmsnorm_and_decoder_layer.md`)
- `RMSNorm.forward`: cast to `float32`, compute `rsqrt(mean(x²) + eps)`, multiply learned `weight`, cast back — why float32 is needed for numerical stability at bfloat16 training precision
- Pre-norm architecture in `Qwen3DecoderLayer`: `hidden = residual + self_attn(input_layernorm(hidden))` then `hidden = residual + mlp(post_attention_layernorm(hidden))`
- Why pre-norm (normalise before the sublayer) vs post-norm (normalise after): pre-norm prevents gradient explosion at depth; Qwen3 uses pre-norm throughout
- `Qwen3MLP` SwiGLU: `down_proj(silu(gate_proj(x)) * up_proj(x))` — `gate_proj` produces a learned soft gate; `up_proj` provides the content; element-wise multiply before `down_proj`
- RMSNorm appears in four roles: `input_layernorm`, `post_attention_layernorm`, final `model.norm`, and the per-head `q_norm`/`k_norm` inside attention

### 04 — Rotary Position Embedding (`04_rope.md`)
- `inv_freq`: `1 / (theta ^ (2i / dim))` for `i in 0..dim/2-1`, registered as a non-persistent buffer (shape `[dim/2=64]`); `rope_theta=1_000_000` for Qwen3-0.6B
- Forward: outer product `inv_freq ⊗ position_ids → freqs [B, q_len, 64]`; concatenate to `[B, q_len, 128]`; `.cos()` and `.sin()` give the rotation matrices, each `[B, q_len, 128]`
- `rotate_half(x)`: splits `[x1 | x2]` at `dim//2`, returns `[-x2, x1]` — the half-rotation that implements the 2D rotation in each head-dim pair
- `apply_rotary_pos_emb`: `q_rot = q * cos + rotate_half(q) * sin`; `cos.unsqueeze(1)` broadcasts over the head dimension
- Why RoPE encodes relative position: the dot product `Q_i · K_j` depends only on `(i − j)`, so the attention logit naturally encodes distance without requiring learned position embeddings
- RoPE is computed once per forward pass in `Qwen3Model.forward` and the same `(cos, sin)` pair is passed to all 28 decoder layers — no per-layer recomputation

### 05 — The Additive Attention Mask (`05_additive_mask.md`)
- Layer 3 passed a binary `attention_mask` to HuggingFace, which built its own additive mask internally; Layer 4 builds it explicitly in `_build_additive_mask` and passes the result to every layer
- Causal part: `torch.zeros(q_len, kv_len)` then `masked_fill(triu(diagonal=kv_len−q_len+1), −inf)` — the diagonal offset aligns future positions correctly with the cached past
- Decode case (`q_len=1`): `triu(diagonal=kv_len)` selects nothing, so the causal tensor remains all-zeros — the single query token can attend to every cached key
- Padding part: `(1.0 − attention_mask.float()) * NEG_INF` → shape `[B, 1, 1, kv_len]` — zeros where tokens are real, `−inf` where they are padding
- Combined: `causal [1, 1, q_len, kv_len] + pad [B, 1, 1, kv_len]` broadcasts to `[B, 1, q_len, kv_len]`; passed directly into SDPA as `attn_mask` (section 06)

### 06 — Attention: GQA and Per-Head QK Norm (`06_attention.md`)
- Q/K/V projection: `q_proj [hidden → 16×128]`, `k_proj`/`v_proj [hidden → 8×128]`; reshape to `[B, q_len, n_heads, head_dim]`, then transpose to `[B, n_heads, q_len, head_dim]`
- Per-head QK `RMSNorm` (`q_norm`, `k_norm`): applied to each head's Q and K individually before RoPE — Qwen3-specific, not present in Llama or Qwen2; `weight` shape is `[head_dim=128]`
- RoPE application: `q, k = apply_rotary_pos_emb(q, k, cos, sin)` using the `(cos, sin)` pair precomputed once in `Qwen3Model.forward`
- KV cache update: `k, v = kv_cache.update(self.layer_idx, k, v)` — `layer_idx` routes to the correct slot; `update()` appends new tokens and returns the full accumulated tensors
- Grouped Query Attention: 8 KV heads serve 16 Q heads; `repeat_kv` uses `expand + reshape` (no data copy) to go `[B, 8, kv_len, 128] → [B, 16, kv_len, 128]`
- Attention formula: `softmax(Q · Kᵀ / √head_dim) · V`; score matrix shape `[B, n_heads, q_len, kv_len]`; additive mask from section 05 applied before softmax sets future/padding positions to `−inf`
- `F.scaled_dot_product_attention(q, k, v, attn_mask=additive_mask, scale=head_dim**-0.5)` — dispatches to FlashAttention on CUDA; never materialises the full score matrix in HBM

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace of one `generate_batch` call, connecting all prior sections in execution order
- Step 1 — Tokenise (identical to Layer 3): `prepare_batch` → `input_ids`, `attention_mask`, `prompt_lens`
- Step 2 — Prefill: `prefill_pos` from `cumsum`; `KVCache()` created; `Qwen3ForCausalLM.forward` → `Qwen3Model.forward` → embed → RoPE (section 04) → additive mask (section 05) → 28 × `Qwen3DecoderLayer` (section 03) each calling `Qwen3Attention` (section 06) → `kv_cache.update()` → final `RMSNorm` → `lm_head`; `sample_batch` → `next_tokens [B]`; TTFT recorded
- Step 3 — Decode loop: pad injection, mask extension, per-request `decode_pos`, same `Qwen3ForCausalLM.forward` with `q_len=1` (causal mask all-zeros), `sample_batch`, `finished |=`, `finished.all()` exit
- Step 4 — Results: `decode_batch` → texts; per-request dict assembled with `latency_ms`, `ttft_ms`, `tpot_ms`

### 08 — What Comes Next (`08_whats_next.md`)
- Layer 4's `generate_batch` still holds the entire batch until the longest request finishes — head-of-line blocking from Layer 3 is unchanged
- Layer 5 (`layer5_continuous_batching`) adds a `Scheduler`, `Request`, and `Batch` object: finished requests are evicted mid-loop and new requests inserted in their place, so no request waits for another
- What files change: `scheduler.py`, `batch.py`, `request.py`; the `generate_batch` loop in `model_runner.py` is replaced by a `step()` method that the scheduler drives
- What stays the same: the `model/` package and `kv_cache.py` are carried over from Layer 4 unchanged — the architecture and weight loading are not touched
- The key engineering challenge Layer 5 addresses: a newly inserted mid-flight request needs its prefill mixed with ongoing decode steps in the same forward pass

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps layer 4 concepts to SGLang source: `Qwen3Config` → `sglang/srt/models/qwen3.py` config handling; `load_weights()` iterator style → SGLang's `stacked_params_mapping`; `repeat_kv` → `QKVParallelLinear`; `_build_additive_mask` → SGLang's `InputMetadata` + `RadixAttention`; `from_pretrained` → SGLang's `ModelRunner.load_model`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Model init diff | `model_runner.py` line 46: `self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)` |
| Prefill forward (returns logits) | `model_runner.py` line 110: `logits = self.model(input_ids, attention_mask=..., kv_cache=kv, position_ids=prefill_pos)` |
| First sample | `model_runner.py` line 120: `next_tokens = sample_batch(logits[:, -1, :], temperature)  # [B]` |
| Decode forward | `model_runner.py` line 160: `logits = self.model(current, attention_mask=..., kv_cache=kv, position_ids=decode_pos)` |
| `Qwen3Config.from_json` | `model/config.py` line 62: `cls(vocab_size=d.get(...), ...)` |
| `from_pretrained` steps | `model/qwen3.py` line 248: `@classmethod def from_pretrained(...)` |
| Weight streaming | `model/qwen3.py` line 280: `with safe_open(str(weights_path), framework="pt", device="cpu") as f:` |
| `load_weights` copy | `model/qwen3.py` line 229: `params[name].data.copy_(tensor)` |
| Tied weights | `model/qwen3.py` line 236: `self.lm_head.weight = self.model.embed_tokens.weight` |
| `RMSNorm.forward` | `model/norm.py` line 27: `norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)` |
| Pre-norm + residual (attn) | `model/decoder_layer.py` line 43: `hidden_states = self.input_layernorm(hidden_states)` |
| SwiGLU forward | `model/mlp.py` line 39: `return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))` |
| Per-head QK norm | `model/attention.py` line 93: `q = self.q_norm(q)` |
| KV cache update | `model/attention.py` line 108: `k, v = kv_cache.update(self.layer_idx, k, v)` |
| `repeat_kv` call | `model/attention.py` line 112: `k = repeat_kv(k, self.num_kv_groups)` |
| SDPA call | `model/attention.py` line 118: `attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attention_mask, scale=self.scale)` |
| `inv_freq` precompute | `model/rope.py` line 41: `inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, ...) / dim))` |
| `apply_rotary_pos_emb` | `model/rope.py` line 73: `q_rot = (q * cos) + (rotate_half(q) * sin)` |
| RoPE computed once | `model/qwen3.py` line 108: `cos, sin = self.rotary_emb(hidden, position_ids)` |
| `_build_additive_mask` | `model/qwen3.py` line 128: `def _build_additive_mask(attention_mask, q_len, kv_len, ...)` |
| Causal fill | `model/qwen3.py` line 154: `causal = causal.masked_fill(mask_upper, NEG_INF)` |
| Padding additive mask | `model/qwen3.py` line 167: `pad = (1.0 - pad) * NEG_INF` |
