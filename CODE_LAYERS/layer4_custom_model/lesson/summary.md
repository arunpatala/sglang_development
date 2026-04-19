# Layer 4 — Summary

Layer 4 replaces `AutoModelForCausalLM.from_pretrained` with our own `Qwen3ForCausalLM`, implementing the full Qwen3 architecture in a `model/` package we can read and modify. The computation is identical to Layer 3 — the `generate_batch` loop, tokenizer, left-padding, position IDs, and finished mask are all carried over unchanged. What changes is ownership: every tensor, every weight, every forward-pass step is now in code we control.

---

## From Layer 3 to Layer 4

In Layer 3, two lines in `model_runner.py` handed the architecture entirely to HuggingFace:

```python
# Layer 3
from transformers import AutoModelForCausalLM
self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).to("cuda")
```

The forward call returned a `ModelOutput` namedtuple. Getting logits required indexing into it, and the KV cache had to be re-assigned from the output on every decode step:

```python
# Layer 3 — forward call
out = self.model(input_ids=ids, attention_mask=mask,
                 past_key_values=kv, use_cache=True)
past_kv = out.past_key_values   # must re-assign every step
logits  = out.logits[:, -1, :]
```

In Layer 4, the import and init become:

```python
# Layer 4
from model import Qwen3ForCausalLM
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

The forward call returns a logits tensor directly. The `KVCache` is mutated in-place by each attention layer, so there is nothing to re-assign:

```python
# Layer 4 — forward call
logits = self.model(ids, attention_mask=mask,
                    kv_cache=kv, position_ids=pos)
logits = logits[:, -1, :]       # kv already updated, no re-assign
```

Everything else in `generate_batch` — the prefill, the decode loop, `sample_batch`, the finished mask — is byte-for-byte identical to Layer 3. The `model/` package is the only new code. Inside `Qwen3ForCausalLM.forward`, input IDs are embedded, RoPE rotation matrices are precomputed from position IDs, an additive causal-plus-padding mask is built, and the result is passed through 28 identical decoder layers before a final projection to vocabulary size. The sections below explain each of those steps in that order: config and weight loading, the decoder block, rotary embeddings, the additive mask, and attention.

---

## Config and Weight Loading

`Qwen3Config` is a plain `@dataclass` with a `from_json()` classmethod:

```python
@dataclass
class Qwen3Config:
    vocab_size: int = 151_936
    hidden_size: int = 1_024
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    ...
```

HuggingFace's `PretrainedConfig` adds roughly a thousand lines of serialisation, hub-download, and legacy-compatibility machinery. All we need are the numeric hyperparameters, so a dataclass suffices.

`from_pretrained()` runs five steps:

```python
@classmethod
def from_pretrained(cls, model_path, dtype=torch.bfloat16):
    model_dir = _resolve_model_path(model_path)           # Step 1 — local dir or HF Hub
    config    = Qwen3Config.from_json(model_dir / "config.json")  # Step 2 — read hyperparams
    model     = cls(config)                               # Step 3 — build on CPU
    model     = model.to(dtype)                           # Step 4 — cast BEFORE copy
    model.load_weights(_iter_safetensors(model_dir, dtype))       # Step 5 — stream weights
    return model.to("cuda").eval()
```

Casting to `bfloat16` before copying weights (Step 4) means `copy_` writes `bfloat16` directly into `bfloat16` parameters — one memory operation instead of two. Streaming weights one tensor at a time via `safe_open` avoids the double-memory spike that would occur if all weights were buffered first.

`load_weights()` receives an iterator of `(name, tensor)` pairs and copies each tensor into the matching parameter:

```python
params = dict(self.named_parameters())
for name, tensor in weights:
    if name in params:
        params[name].data.copy_(tensor)
```

After the loop, if `tie_word_embeddings` is true, `lm_head.weight` is pointed at `embed_tokens.weight` so they share memory. For a model the size of Qwen3-0.6B this saves around 600 MB of GPU memory; for larger models the saving scales with vocabulary size times hidden dimension.

---

## RMSNorm and the Decoder Layer

Each `layer(hidden, cos, sin, additive_mask, kv_cache)` call in `Qwen3Model.forward` is a `Qwen3DecoderLayer` — one transformer block — executing the normalise-attend-add-normalise-MLP-add pattern 28 times in sequence.

Every normalisation in the model — four uses per layer plus the final norm — goes through the same `RMSNorm`:

```python
def forward(self, x: torch.Tensor) -> torch.Tensor:
    dtype = x.dtype
    x = x.float()
    norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return (self.weight * norm).to(dtype)
```

The cast to `float32` before computing `rsqrt` is not optional. At `bfloat16` precision, squaring small activations frequently underflows to zero, making the normalisation numerically unstable. The cast-compute-cast-back pattern is standard in production transformer implementations.

Each of the 28 `Qwen3DecoderLayer` blocks follows a pre-norm architecture: the input is normalised before it enters the sublayer, and the sublayer output is added back to the un-normalised residual:

```python
residual = hidden_states
hidden_states = self.input_layernorm(hidden_states)
hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, kv_cache)
hidden_states = residual + hidden_states

residual = hidden_states
hidden_states = self.post_attention_layernorm(hidden_states)
hidden_states = self.mlp(hidden_states)
hidden_states = residual + hidden_states
```

The MLP is a SwiGLU feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`. Two parallel projections replace the single up-projection of a standard FFN. `gate_proj` produces a learned gating signal passed through `silu`; `up_proj` provides the content vector. Element-wise multiplication acts as a soft gate before `down_proj` projects back to `hidden_size`.

---

## Rotary Position Embedding

RoPE encodes position by rotating each query and key vector by an angle that depends on the token's absolute position. The rotation is constructed from a set of frequencies precomputed once at model initialisation:

```python
inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
# shape: [dim/2 = 64]
```

At each forward pass, `RotaryEmbedding` computes an outer product of `inv_freq` with `position_ids`:

```python
freqs = (inv_freq[None, :, None] @ position_ids[:, None, :]).transpose(1, 2)
# inv_freq: [1, 64, 1]  ×  position_ids: [B, 1, q_len]  →  freqs: [B, q_len, 64]
emb = torch.cat([freqs, freqs], dim=-1)   # [B, q_len, 128]
cos = emb.cos()                           # [B, q_len, 128]
sin = emb.sin()                           # [B, q_len, 128]
```

The rotation is applied with `rotate_half`, which splits each vector into two equal halves `[x1 | x2]` and returns `[-x2, x1]`:

```python
q_rot = (q * cos) + (rotate_half(q) * sin)   # [B, n_heads, q_len, head_dim]
k_rot = (k * cos) + (rotate_half(k) * sin)   # [B, n_kv_heads, q_len, head_dim]
```

The key property of this rotation is that the dot product `Q_i · K_j` depends only on the relative distance `(i − j)` between the two positions. The attention logit naturally encodes how far apart two tokens are without any learned position embeddings. The `(cos, sin)` pair is computed once per forward pass in `Qwen3Model.forward` and passed to all 28 decoder layers, avoiding redundant recomputation.

---

## The Additive Attention Mask

In Layer 3, a binary `attention_mask` was passed to HuggingFace's model, which built its internal additive mask in code we could not see. In Layer 4, `_build_additive_mask` constructs it explicitly in one place and passes the result to every attention layer:

```python
additive_mask = _build_additive_mask(
    attention_mask=attention_mask,
    q_len=q_len,
    kv_len=past_len + q_len,
    dtype=hidden.dtype,
    device=hidden.device,
)
```

The mask combines two parts. The causal part prevents any query token from attending to future key positions:

```python
causal = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
mask_upper = torch.triu(torch.ones(..., dtype=torch.bool), diagonal=kv_len - q_len + 1)
causal = causal.masked_fill(mask_upper, NEG_INF)   # [1, 1, q_len, kv_len]
```

During decode, `q_len` is 1. `triu(diagonal=kv_len)` selects nothing, so the causal tensor stays all-zeros — the single query attends to every key in the cache without restriction.

The padding part converts the binary mask to additive form:

```python
pad = (1.0 - attention_mask.to(dtype)) * NEG_INF   # [B, 1, 1, kv_len]
```

Adding the two parts and broadcasting gives the final `[B, 1, q_len, kv_len]` tensor. This is the `attention_mask` passed into every `Qwen3Attention.forward` call and handed directly to SDPA.

---

## Attention: GQA and Per-Head QK Norm

`Qwen3Attention.forward` opens by projecting the hidden state into queries, keys, and values:

```python
q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
# q: [B, q_len, 16, 128]   k/v: [B, q_len, 8, 128]
```

Before RoPE is applied, each head's Q and K vectors are normalised independently:

```python
q = self.q_norm(q)   # [B, q_len, 16, 128] — normalises last dim per head
k = self.k_norm(k)   # [B, q_len,  8, 128]
```

This per-head QK `RMSNorm` is specific to Qwen3. It is not present in Llama or Qwen2. The weight for each norm is a `[head_dim=128]` vector, so the normalisation scales independently per head dimension rather than per sequence position. It stabilises attention logits at large model scales where per-head Q/K magnitudes can diverge.

RoPE is then applied using the `(cos, sin)` pair computed once by `Qwen3Model` and passed down to every layer:

```python
q, k = apply_rotary_pos_emb(q, k, cos, sin)
# q: [B, q_len, 16, 128]   k: [B, q_len, 8, 128]
```

The rotated K and V are appended to the cache so the decode step can attend to all prior tokens:

```python
k, v = kv_cache.update(self.layer_idx, k, v)
# k: [B, kv_len, 8, 128]   v: [B, kv_len, 8, 128]
```

With the cache updated, 8 KV heads must serve 16 Q heads. `repeat_kv` handles this without copying data — it uses `expand` (a view) followed by `reshape` to go from `[B, 8, kv_len, 128]` to `[B, 16, kv_len, 128]`.

The attention score for query position `i` attending to key position `j` is:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √head_dim ) · V
```

`Q · Kᵀ` produces a `[B, n_heads, q_len, kv_len]` score matrix. Dividing by `√head_dim` (= `√128` ≈ 11.3) prevents the dot products from growing large enough to push softmax into regions where gradients vanish. The additive mask is added before softmax, setting future and padding positions to `−inf` so they contribute zero weight. The weighted sum over V then produces the attended output.

In code this is a single call:

```python
attn_out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,   # [B, 1, q_len, kv_len] additive
    scale=self.scale,           # 1 / √head_dim
)  # [B, n_heads, q_len, head_dim]
```

PyTorch's `scaled_dot_product_attention` dispatches to FlashAttention on CUDA when the inputs are contiguous and in a supported dtype, fusing the `Q · Kᵀ`, scale, mask, softmax, and `· V` steps into a single kernel that never materialises the full score matrix in HBM. The output is transposed and reshaped back to `[B, q_len, hidden]` before the output projection `o_proj`.

---

## The Full Loop

Now that all the parts have been explained, it is worth tracing the full lifecycle — from server startup through a single `generate_batch` call — to see how they connect.

Before any request arrives, `BatchedModel.__init__` loads the model. `Qwen3ForCausalLM.from_pretrained` first resolves the model path, then calls `Qwen3Config.from_json` to read `config.json` into a plain dataclass — no HuggingFace machinery involved. It constructs `Qwen3ForCausalLM(config)` on CPU, casts it to `bfloat16`, then streams weight tensors from `model.safetensors` one at a time via `safe_open`, copying each into the matching parameter with `params[name].data.copy_(tensor)`. After the loop, `lm_head.weight` is pointed at `embed_tokens.weight` to share memory. The model is then moved to CUDA and set to eval mode. From this point on `self.model` is our own `Qwen3ForCausalLM` — every subsequent forward call goes through code we can read.

The call arrives with B conversations. `tokenizer.prepare_batch` formats each with `apply_chat_template`, tokenises with `padding=True` and left-alignment, and returns `input_ids [B, max_prompt_len]`, `attention_mask [B, max_prompt_len]`, and `prompt_lens_list`. The prefill position IDs are computed from `attention_mask` with the `cumsum` formula from Layer 3 — unchanged.

A fresh `KVCache()` is created and `Qwen3ForCausalLM.forward` is called for the prefill. The skeleton of `Qwen3Model.forward` makes the execution order explicit:

```python
def forward(self, input_ids, attention_mask, kv_cache, position_ids):
    past_len = kv_cache.get_seq_length() if kv_cache is not None else 0

    hidden = self.embed_tokens(input_ids)                    # [B, q_len, hidden]
    cos, sin = self.rotary_emb(hidden, position_ids)         # each [B, q_len, head_dim]
    additive_mask = _build_additive_mask(
        attention_mask, q_len, past_len + q_len, hidden.dtype, hidden.device
    )                                                        # [B, 1, q_len, kv_len]

    for layer in self.layers:                                # 28 ×
        hidden = layer(hidden, cos, sin, additive_mask, kv_cache)

    return self.norm(hidden)                                 # [B, q_len, hidden]
```

Four things happen before the layer loop: the token IDs are embedded, the RoPE `(cos, sin)` pair is computed once from `position_ids`, the additive causal-plus-padding mask is built once, and `past_len` is read from the cache so the mask covers the correct `kv_len`.

Each of the 28 `Qwen3DecoderLayer` blocks receives the same `cos`, `sin`, `additive_mask`, and `kv_cache` reference — no recomputation per layer. Inside each block, pre-norm is applied, then `Qwen3Attention` projects Q/K/V, applies per-head QK norm, applies RoPE, calls `kv_cache.update(layer_idx, k, v)`, expands K/V for GQA with `repeat_kv`, and runs SDPA. The attention output is added back to the residual, a second pre-norm is applied, the SwiGLU MLP runs, and its output is added back to the residual.

After all 28 layers the final `RMSNorm` and `lm_head` produce `logits [B, max_prompt_len, vocab_size]`. The `[:, -1, :]` slice extracts last-position logits, `sample_batch` draws `next_tokens [B]`, and TTFT is recorded.

The decode loop runs exactly as in Layer 3. Finished requests receive `pad_id` via `torch.where`; the attention mask grows by one column of ones; per-request `decode_pos` is `prompt_lens + decode_step`. The same `Qwen3ForCausalLM.forward` is called with `current [B, 1]` — this time `q_len=1`, so the causal mask is all-zeros and the single query token attends to every key in the growing cache without restriction. `sample_batch` draws the next `[B]` tokens, `finished` is updated with `|=`, and the loop exits when `finished.all()` is true.

After the loop, `tokenizer.decode_batch` converts each request's token list to text. One result dict is assembled per request with individual token counts and the shared `ttft_ms` and `tpot_ms` from the batched computation. The caller receives the same response format as Layer 3.

---

## What Comes Next

Layer 4 does not change the scheduling primitive: `generate_batch` still holds the entire batch until the longest request finishes, and head-of-line blocking remains. Layer 5 (`layer5_continuous_batching`) addresses this by introducing a `Scheduler`, `Request`, and `Batch` object. Finished requests are evicted from the batch mid-loop; new requests enter with their own prefill injected at the next step. The `model/` package and `kv_cache.py` from Layer 4 are carried over unchanged — the architecture is not touched. The work moves into the scheduler that decides, at each step, which requests participate in the next forward pass and what shape the input tensor takes.
