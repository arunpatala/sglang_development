# 06 — Attention: GQA and Per-Head QK Norm

## The Call Site

Inside each `Qwen3DecoderLayer.forward`, after `input_layernorm` normalises the hidden state, attention is computed:

```python
hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, kv_cache)
```

`self.self_attn` is a `Qwen3Attention` module. It receives the normalised `hidden [B, q_len, hidden_size]`, the `(cos, sin)` pair precomputed once by `Qwen3Model` (section 04), the additive attention mask precomputed once by `_build_additive_mask` (section 05), and the `KVCache` object shared across all 28 layers. It returns an output of the same shape `[B, q_len, hidden_size]`, which the decoder layer then adds back to the residual.

`Qwen3Attention` runs seven steps in sequence. The code below is the complete `forward` method:

```python
def forward(self, hidden_states, cos, sin, attention_mask, kv_cache):
    B, q_len, _ = hidden_states.shape

    # 1. Project Q / K / V
    q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
    k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
    v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

    # 2. Per-head QK RMSNorm (Qwen3-specific)
    q = self.q_norm(q)
    k = self.k_norm(k)

    # 3. Transpose to [B, n_heads, q_len, head_dim]
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)

    # 4. Apply RoPE
    q, k = apply_rotary_pos_emb(q, k, cos, sin)

    # 5. KV cache update
    if kv_cache is not None:
        k, v = kv_cache.update(self.layer_idx, k, v)

    # 6. GQA: expand KV heads to match Q head count
    k = repeat_kv(k, self.num_kv_groups)   # [B, 16, kv_len, 128]
    v = repeat_kv(v, self.num_kv_groups)

    # 7. Scaled dot-product attention
    attn_out = F.scaled_dot_product_attention(
        q, k, v,
        attn_mask=attention_mask,   # [B, 1, q_len, kv_len] additive
        scale=self.scale,           # 1 / √head_dim
    )   # [B, n_heads, q_len, head_dim]

    attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
    return self.o_proj(attn_out)
```

---

## Step 1 — Q / K / V Projection

```python
q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
# q: [B, q_len, 16, 128]   k/v: [B, q_len, 8, 128]
```

`q_proj` maps `hidden_size = 1024` to `num_heads × head_dim = 16 × 128 = 2048`. `k_proj` and `v_proj` map to `num_kv_heads × head_dim = 8 × 128 = 1024`. The `.view` reshapes the projected output into the per-head structure without a copy. At this point the tensor still has the sequence dimension before the head dimension — that is corrected in step 3 after per-head normalisation.

---

## Step 2 — Per-Head QK RMSNorm

```python
q = self.q_norm(q)   # [B, q_len, 16, 128] — normalises last dim
k = self.k_norm(k)   # [B, q_len,  8, 128]
```

`q_norm` and `k_norm` are `RMSNorm` instances with `dim = head_dim = 128`. They apply `RMSNorm.forward` (section 03) to the last dimension — independently for each head, each token, and each batch element. The learned `weight` for each norm is a `[128]` vector, not `[hidden_size]` — the scale operates at head-dimension granularity, not hidden-size granularity.

This per-head QK normalisation is specific to Qwen3. Llama and Qwen2 do not have it. Its purpose is to stabilise the magnitudes of Q and K before the dot product. Without normalisation, Q and K magnitudes can grow large during training, causing the dot products `Q · Kᵀ` to produce extremely large values. Even with the `1/√head_dim` scale factor, large dot products push softmax into saturation regions where gradients effectively vanish. Normalising Q and K to unit-RMS before applying RoPE keeps the pre-softmax logits in a moderate range regardless of model scale.

---

## Step 3 — Transpose and Step 4 — RoPE

After normalisation, the tensors are transposed to the layout expected by the attention kernel:

```python
q = q.transpose(1, 2)   # [B, 16, q_len, 128]
k = k.transpose(1, 2)   # [B,  8, q_len, 128]
v = v.transpose(1, 2)   # [B,  8, q_len, 128]
q, k = apply_rotary_pos_emb(q, k, cos, sin)
```

`apply_rotary_pos_emb` applies the rotation matrices from section 04 to Q and K. V is not rotated — position information is encoded only in the keys and queries, not in the values. The rotation modifies the angle of each head-dim vector pair according to the token's absolute position, which (as section 04 established) makes the Q-K dot product depend on relative distance.

---

## Step 5 — KV Cache Update

```python
if kv_cache is not None:
    k, v = kv_cache.update(self.layer_idx, k, v)
```

`kv_cache.update` appends the new token's K and V to the stored tensors for this layer and returns the full accumulated tensors:

```python
def update(self, layer_idx, new_k, new_v):
    if layer_idx not in self._k:
        self._k[layer_idx] = new_k
        self._v[layer_idx] = new_v
    else:
        self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=-2)
        self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=-2)
    return self._k[layer_idx], self._v[layer_idx]
```

`self.layer_idx` is stored on the `Qwen3Attention` instance at construction time — `Qwen3DecoderLayer` passes its index `i` when creating each attention module. This is how all 28 layers share the same `KVCache` object without colliding: layer 0 writes to `_k[0]`, layer 1 to `_k[1]`, and so on.

After `update`, `k` and `v` have `kv_len = past_len + q_len` positions in their sequence dimension — the full history of tokens seen by this layer, not just the new ones. This is what makes it possible for the new query token to attend to every key in the sequence history.

---

## Step 6 — Grouped Query Attention

```python
k = repeat_kv(k, self.num_kv_groups)   # [B, 8, kv_len, 128] → [B, 16, kv_len, 128]
v = repeat_kv(v, self.num_kv_groups)
```

`num_kv_groups = num_attention_heads // num_key_value_heads = 16 // 8 = 2`. Each of the 8 KV heads must serve 2 Q heads. `repeat_kv` performs this without copying data:

```python
def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )
```

`.expand` creates a view that makes each of the 8 head slices appear twice along a new dimension, without allocating new memory. `.reshape` then flattens the head and repetition dimensions from `[B, 8, 2, S, D]` to `[B, 16, S, D]`. The result looks like 16 distinct KV heads, but internally only 8 sets of weights and 8 slots in the KV cache exist. GQA was introduced to reduce the KV cache memory footprint — at decode time, a batch of B requests with sequences of length L requires `B × L × n_kv_heads × head_dim × 2 (K+V) × dtype_bytes` bytes of cache. Halving `n_kv_heads` from 16 to 8 halves that cost, which matters significantly as sequence length grows.

---

## Step 7 — Scaled Dot-Product Attention

With Q, K, and V all at shape `[B, 16, q_len or kv_len, 128]`, the attention computation runs:

```python
attn_out = F.scaled_dot_product_attention(
    q, k, v,
    attn_mask=attention_mask,   # [B, 1, q_len, kv_len] additive
    scale=self.scale,           # 1 / √head_dim = 1 / √128 ≈ 0.0884
)   # [B, 16, q_len, 128]
```

The formula being computed is:

```
Attention(Q, K, V) = softmax( Q · Kᵀ / √head_dim + mask ) · V
```

`Q · Kᵀ` produces `[B, 16, q_len, kv_len]` raw attention scores. Dividing by `√128 ≈ 11.3` prevents the dot products from growing so large that softmax saturates — a single attention head with `head_dim = 128` computes a dot product of two unit-norm vectors of 128 dimensions, which grows as `√128` on average. The additive mask from section 05 (value 0 for real positions, `−inf` for future and padding positions) is added before softmax, making masked positions contribute zero weight. The softmax output is a `[B, 16, q_len, kv_len]` weight matrix; multiplying by `V [B, 16, kv_len, 128]` produces `attn_out [B, 16, q_len, 128]`.

`F.scaled_dot_product_attention` dispatches to FlashAttention on CUDA when the input tensors are contiguous and in a supported dtype. FlashAttention implements the same mathematical formula but tiles the computation to avoid materialising the full `[B, 16, q_len, kv_len]` score matrix in GPU high-bandwidth memory. For long sequences this reduces memory access by an order of magnitude and improves throughput correspondingly. The API call is identical — PyTorch selects the implementation automatically.

After `attn_out` is computed, `.transpose(1, 2).contiguous().view(B, q_len, -1)` merges the 16 heads back into a single `[B, q_len, 2048]` tensor, and `o_proj` projects from `2048` back to `hidden_size = 1024`. The output is what `Qwen3DecoderLayer` adds to its residual.
