# 05 — The Additive Attention Mask

## From Implicit to Explicit

In Layer 3, `generate_batch` passed a binary `attention_mask` directly to HuggingFace's model:

```python
# Layer 3 — binary mask passed to HuggingFace
out = self.model(
    input_ids=input_ids,
    attention_mask=attention_mask,   # [B, kv_len]  1=real, 0=pad
    past_key_values=past_kv,
    use_cache=True,
)
```

HuggingFace converted this binary mask into an additive mask internally, inside code we could not easily inspect. The conversion happened in its attention implementation: binary 1 became 0.0 (attend), binary 0 became `−inf` (ignore after softmax). A causal upper-triangular mask was also constructed and merged in. All of this was invisible.

In Layer 4, `_build_additive_mask` in `model/qwen3.py` constructs the full additive mask explicitly, before the 28-layer loop:

```python
additive_mask = _build_additive_mask(
    attention_mask=attention_mask,   # [B, kv_len] binary
    q_len=q_len,
    kv_len=past_len + q_len,
    dtype=hidden.dtype,
    device=hidden.device,
)   # result: [B, 1, q_len, kv_len]
```

The same `additive_mask` tensor is passed to all 28 attention layers. The cost of constructing it is paid once per forward call, not once per layer.

---

## The Causal Part

The causal part of the mask prevents any query token from attending to key tokens that appear later in the sequence:

```python
NEG_INF = torch.finfo(dtype).min

causal = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
if q_len > 1:
    mask_upper = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
    mask_upper = torch.triu(mask_upper, diagonal=kv_len - q_len + 1)
    causal = causal.masked_fill(mask_upper, NEG_INF)

causal = causal.unsqueeze(0).unsqueeze(0)   # [1, 1, q_len, kv_len]
```

`torch.triu(diagonal=d)` selects positions strictly above diagonal `d`. For a `[q_len, kv_len]` matrix, position `(i, j)` is in the upper triangle when `j >= i + d`. The diagonal offset `d = kv_len - q_len + 1` is chosen so that query at row `i` (which corresponds to absolute sequence position `past_len + i`) cannot see key at column `j` when `j > past_len + i`. In other words, each query can attend to all past cached positions plus its own position, but not to future tokens in the current chunk.

For a concrete example, suppose `past_len = 3` and `q_len = 4`, giving `kv_len = 7`. The diagonal offset is `7 - 4 + 1 = 4`. The `triu` call marks positions where `j >= i + 4`:

```
         j: 0  1  2  3  4  5  6     (cache + new tokens)
    i=0:    .  .  .  .  X  X  X     query 0 can see j=0,1,2,3 only
    i=1:    .  .  .  .  .  X  X     query 1 can see j=0,1,2,3,4
    i=2:    .  .  .  .  .  .  X     query 2 can see j=0..5
    i=3:    .  .  .  .  .  .  .     query 3 can see all 7
```

`X` positions are filled with `NEG_INF`; `.` positions remain 0 (attend freely). This enforces causality while allowing every query to attend to all previously cached keys.

---

## The Decode Case

During decode, `q_len = 1`. The guard `if q_len > 1` is false, so the causal tensor stays all-zeros without any `triu` computation:

```python
causal = torch.zeros(1, kv_len, dtype=dtype, device=device)
# shape [1, kv_len] — all zeros, no masking needed
causal = causal.unsqueeze(0).unsqueeze(0)   # [1, 1, 1, kv_len]
```

A single query token can attend to every key in the cache. There is no future token to mask — the sequence history is already fixed in the cache. The short-circuit avoids the `triu` allocation entirely, which is a minor efficiency win but more importantly clarifies the intent: the causal mask exists to prevent prefill tokens from seeing each other out of order, not to restrict decode steps.

---

## The Padding Part

The padding part converts the binary `attention_mask` into additive form:

```python
if attention_mask is not None:
    pad = attention_mask.to(dtype)        # 1.0 for real, 0.0 for padding
    pad = (1.0 - pad) * NEG_INF          # 0.0 for real, −inf for padding
    pad = pad[:, None, None, :]           # [B, 1, 1, kv_len]
```

`(1.0 - attention_mask) * NEG_INF` inverts the binary mask: positions that were 1 (real tokens) become 0.0 (attend), and positions that were 0 (padding) become `NEG_INF` (ignore). The unsqueeze chain adds a batch-broadcast dimension for heads and query positions: `[B, kv_len]` → `[B, 1, 1, kv_len]`. When added to the causal tensor `[1, 1, q_len, kv_len]`, broadcasting expands both to `[B, 1, q_len, kv_len]`.

`torch.finfo(dtype).min` is used for `NEG_INF` rather than a hardcoded value like `-1e9`. For `bfloat16`, the minimum representable value is approximately `-3.39e38`. After `exp` in softmax, `exp(-3.39e38)` is effectively zero on any hardware. Using `finfo.min` is safer than `-1e9` for `float32` and mandatory for `bfloat16`, where `-1e9` is representable but may not be far enough from 0 in extreme cases.

---

## The Combined Mask

```python
return causal + pad   # [B, 1, q_len, kv_len]
```

Adding the two parts produces the final mask. Both causal future positions and padding positions receive `NEG_INF` (or `NEG_INF + NEG_INF = NEG_INF`); real past positions receive 0. This tensor is passed directly as `attn_mask` to `F.scaled_dot_product_attention` in every attention layer. PyTorch's SDPA expects an additive mask in exactly this format: values that should be masked go to `−inf`; values that should be attended get 0. No further conversion is needed.

The single `[B, 1, q_len, kv_len]` tensor broadcasts over all 16 attention heads inside SDPA. The head dimension is 1 here; SDPA expands it to 16 during the `Q · Kᵀ` computation. This means the same padding and causal structure applies equally to all heads, which is the correct behaviour — each head sees the same sequence, just through different learned projections.

Section 06 explains how `Qwen3Attention` receives this mask and uses it inside `F.scaled_dot_product_attention`.
