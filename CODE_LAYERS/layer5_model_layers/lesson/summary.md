# Layer 5 — Summary

Layer 4 took ownership of config reading and weight loading while keeping HuggingFace's Qwen3 as the forward computation. Layer 5 completes the picture: `self._model` — the HF `AutoModelForCausalLM` that did the actual computation — is replaced with our own `Qwen3Model`. The forward call returns a plain logits tensor, the KV cache is passed directly to our attention layers, and every step of the computation is now in code we can read and modify.

The `from_pretrained`, `load_weights`, and `Qwen3Config` code from Layer 4 carries over to Layer 5 unchanged — loading is told once and done. The new code lives in five files: `norm.py`, `mlp.py`, `rope.py`, `attention.py`, and `decoder_layer.py`, composed into `Qwen3Model` and `Qwen3ForCausalLM` in `model/qwen3.py`.

---

## From Layer 4 to Layer 5

In Layer 4, the forward call used HuggingFace's interface. `past_key_values` was the HF-compatible cache handle, and the call returned a `(logits, past_kv)` tuple whose second value was discarded because the `KVCache` updated itself in-place:

```python
# Layer 4 — forward call
kv = KVCache()
logits, _ = self.model(ids, attention_mask=mask,
                       past_key_values=kv, position_ids=pos)
logits = logits[:, -1, :]
```

In Layer 5, our attention layers call `kv_cache.update(layer_idx, k, v)` directly. There is no HF adapter, no tuple to unpack — the call returns logits and nothing else:

```python
# Layer 5 — forward call
kv = KVCache()
logits = self.model(ids, attention_mask=mask,
                    kv_cache=kv, position_ids=pos)
logits = logits[:, -1, :]
```

That two-word change in the call signature — `past_key_values=kv` → `kv_cache=kv`, and unpacking `logits, _` → assigning `logits` — is the entire visible diff in `model_runner.py`. Everything else in `generate_batch` is unchanged from Layer 4 and Layer 3.

Inside `Qwen3ForCausalLM.forward`, input IDs are embedded, RoPE rotation matrices are precomputed from position IDs, an additive causal-plus-padding mask is built, and the result is passed through 28 identical decoder layers before a final projection to vocabulary size. The sections below explain each of those steps in execution order.

---

## RMSNorm and the Decoder Layer

Each call in `Qwen3Model.forward` passes through a `Qwen3DecoderLayer` — one transformer block — executing the normalise-attend-add-normalise-MLP-add pattern 28 times in sequence.

Every normalisation in the model goes through the same `RMSNorm`:

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

The MLP is a SwiGLU feed-forward network: `down_proj(silu(gate_proj(x)) * up_proj(x))`. `gate_proj` produces a learned gating signal through `silu`; `up_proj` provides the content vector. Element-wise multiplication acts as a soft gate before `down_proj` projects back to `hidden_size`.

---

## Rotary Position Embedding

RoPE encodes position by rotating each query and key vector by an angle that depends on the token's absolute position. The rotation frequencies are precomputed once at model initialisation:

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

The key property is that the dot product `Q_i · K_j` depends only on the relative distance `(i − j)`. The `(cos, sin)` pair is computed once per forward pass in `Qwen3Model.forward` and passed to all 28 decoder layers — no per-layer recomputation.

---

## The Additive Attention Mask

In Layer 4, a binary `attention_mask` was passed to HuggingFace, which built its internal additive mask in code we could not see. In Layer 5, `_build_additive_mask` constructs it explicitly in one place and passes the result to every attention layer.

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

Adding the two parts and broadcasting gives the final `[B, 1, q_len, kv_len]` tensor. This is passed directly into SDPA as `attn_mask`.

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
q = self.q_norm(q)   # [B, q_len, 16, 128]
k = self.k_norm(k)   # [B, q_len,  8, 128]
```

This per-head QK `RMSNorm` is specific to Qwen3 — not present in Llama or Qwen2. It stabilises attention logits at large model scales where per-head Q/K magnitudes can diverge. After normalisation, RoPE is applied and the K/V are written to the cache:

```python
q, k = apply_rotary_pos_emb(q, k, cos, sin)
k, v = kv_cache.update(self.layer_idx, k, v)   # k: [B, kv_len, 8, 128]
```

With the cache updated, 8 KV heads must serve 16 Q heads. `repeat_kv` handles this without copying data — it uses `expand` (a view) followed by `reshape`:

```
[B, 8, kv_len, 128] → [B, 16, kv_len, 128]
```

Attention is then computed with a single SDPA call:

```python
attn_out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,   # [B, 1, q_len, kv_len] additive
    scale=self.scale,           # 1 / √head_dim
)
```

`scaled_dot_product_attention` dispatches to FlashAttention on CUDA, fusing the `Q · Kᵀ`, scale, mask, softmax, and `· V` steps into a single kernel that never materialises the full score matrix in HBM.

---

## The Full Loop

Before any request arrives, `BatchedModel.__init__` loads the model. `Qwen3ForCausalLM.from_pretrained` handles path resolution, config reading, HF skeleton construction, weight streaming, and the `lm_head` tie — all covered in Layer 4 and unchanged here.

A call arrives with B conversations. `tokenizer.prepare_batch` formats and left-pads them, returning `input_ids [B, max_prompt_len]`, `attention_mask`, and `prompt_lens_list`. The prefill position IDs are computed from `attention_mask` with the `cumsum` formula from Layer 3.

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

Four things happen before the layer loop: token IDs are embedded, the RoPE `(cos, sin)` pair is computed once from `position_ids`, the additive mask is built once, and `past_len` is read from the cache so the mask covers the correct `kv_len`. Each of the 28 `Qwen3DecoderLayer` blocks receives the same `cos`, `sin`, `additive_mask`, and `kv_cache` reference — no recomputation per layer. After all 28 layers, the final `RMSNorm` and `lm_head` produce `logits [B, max_prompt_len, vocab_size]`. The `[:, -1, :]` slice extracts last-position logits, `sample_batch` draws `next_tokens [B]`, and TTFT is recorded.

The decode loop runs exactly as in Layer 4 and Layer 3. Finished requests receive `pad_id` via `torch.where`; the attention mask grows by one column of ones; per-request `decode_pos` is `prompt_lens + decode_step`. The same `Qwen3ForCausalLM.forward` is called with `current [B, 1]` — `q_len=1`, so the causal mask is all-zeros and the single query token attends to every key in the growing cache. The loop exits when `finished.all()` is true.

---

## What Comes Next

Layer 5 does not change the scheduling primitive: `generate_batch` still holds the entire batch until the longest request finishes. Head-of-line blocking remains — a 5-token request is held until a co-batched 1000-token request completes.

Layer 6 (`layer6_continuous_batching`) addresses this by introducing a `Scheduler`, `Request`, and `Batch` object. Finished requests are evicted from the batch mid-loop; new requests enter with their own prefill injected at the next step. The `model/` package and `kv_cache.py` from Layer 5 are carried over unchanged — the architecture is not touched. The work moves into the scheduler that decides, at each step, which requests participate in the next forward pass and what shape the input tensor takes.
