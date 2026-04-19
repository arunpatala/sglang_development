# 04 — Rotary Position Embedding

## Where RoPE Enters the Forward Pass

In `Qwen3Model.forward`, before the 28-layer loop begins, two lines compute the rotation matrices that will be used by every attention layer:

```python
cos, sin = self.rotary_emb(hidden, position_ids)   # each [B, q_len, head_dim]
```

`self.rotary_emb` is a `RotaryEmbedding` module stored on `Qwen3Model`. Its output — a `(cos, sin)` pair — is passed unchanged into every `Qwen3DecoderLayer`, which forwards it into `Qwen3Attention`. The computation happens once, not 28 times. This section explains what `RotaryEmbedding` computes and why it works.

---

## The Frequency Buffer

`RotaryEmbedding.__init__` precomputes a set of inverse frequencies and stores them as a non-persistent buffer:

```python
def __init__(self, config: Qwen3Config) -> None:
    super().__init__()
    dim   = config.head_dim    # 128
    theta = config.rope_theta  # 1_000_000.0

    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    self.register_buffer("inv_freq", inv_freq, persistent=False)
    # inv_freq shape: [dim/2 = 64]
```

`torch.arange(0, dim, 2)` generates even indices `[0, 2, 4, ..., 126]`, giving 64 values for `head_dim = 128`. Dividing by `dim` gives `[0/128, 2/128, ..., 126/128]`. The formula `1 / (theta ^ (2i / dim))` produces 64 frequencies that decrease geometrically from `1/theta^0 = 1.0` down to `1/theta^(126/128) ≈ 1e-5.9`. Lower-indexed dimensions rotate fast; higher-indexed dimensions rotate slowly.

The choice of `theta = 1_000_000` extends the effective context length compared to the original `theta = 10_000` used in Llama. A higher theta means the slow-rotating high-index dimensions take longer to complete a full cycle, allowing the model to represent much longer position ranges without ambiguity. Qwen3-0.6B is trained on sequences up to 32,768 tokens; `rope_theta = 1_000_000` provides headroom for that range.

`register_buffer` with `persistent=False` means `inv_freq` is part of the module's state and moves with it when `.to("cuda")` is called, but it is not saved in the checkpoint — it is recomputed from `config` each time the model is loaded.

---

## Computing cos and sin

At each forward pass, `RotaryEmbedding.forward` converts position indices into rotation angles:

```python
@torch.no_grad()
def forward(self, x, position_ids):
    B, q_len = position_ids.shape

    # Outer product: [B, 64, 1] × [B, 1, q_len] → [B, q_len, 64]
    inv = self.inv_freq[None, :, None].float().expand(B, -1, 1)
    pos = position_ids[:, None, :].float()
    freqs = (inv @ pos).transpose(1, 2)        # [B, q_len, 64]

    emb = torch.cat([freqs, freqs], dim=-1)    # [B, q_len, 128]
    cos = emb.cos().to(x.dtype)                # [B, q_len, 128]
    sin = emb.sin().to(x.dtype)                # [B, q_len, 128]
    return cos, sin
```

The outer product multiplies each of the 64 frequencies by each of the `q_len` position indices, giving a `[B, q_len, 64]` matrix of angles. Each entry `freqs[b, t, i]` is `inv_freq[i] * position_ids[b, t]` — the angle for token at position `t` in batch element `b` for frequency dimension `i`.

Concatenating `freqs` with itself to get `[B, q_len, 128]` is an implementation detail of the half-rotation trick: the same angles are needed for both halves of the `rotate_half` operation. Taking `.cos()` and `.sin()` converts angles to rotation matrices. The cast `.to(x.dtype)` moves from `float32` (used for the trig computation) back to `bfloat16` to match the query and key tensors they will be multiplied with.

---

## Applying the Rotation

The rotation is applied inside `Qwen3Attention.forward` after Q/K projection and per-head QK norm, using two functions from `model/rope.py`:

```python
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([-x2, x1], dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)   # [B, 1, q_len, head_dim] — broadcast over heads
    sin = sin.unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)   # [B, n_heads,    q_len, head_dim]
    k_rot = (k * cos) + (rotate_half(k) * sin)   # [B, n_kv_heads, q_len, head_dim]
    return q_rot, k_rot
```

`rotate_half` splits the last dimension at the midpoint and returns `[-x2, x1]`. This implements a 2D rotation in each consecutive pair of dimensions. For a 2D vector `(x1, x2)`, the rotation by angle `θ` is `(x1 cos θ - x2 sin θ, x1 sin θ + x2 cos θ)`. The formula `x * cos + rotate_half(x) * sin` implements exactly this for every pair simultaneously across all `head_dim / 2` pairs.

`cos.unsqueeze(1)` inserts a head dimension so the `[B, 1, q_len, head_dim]` tensor broadcasts over all Q heads (16) and all KV heads (8) without needing explicit expansion.

---

## Why RoPE Encodes Relative Position

The reason for all this geometry is a property of the dot product. After rotation, the dot product between a query at position `i` and a key at position `j` satisfies:

```
Q_rotated[i] · K_rotated[j]  =  f(Q[i], K[j], i - j)
```

The result depends on the content of Q and K and on the *difference* `i − j`, not on `i` or `j` individually. A query at position 10 attending to a key at position 8 sees the same relative offset as a query at position 100 attending to a key at position 98. This is the core property that absolute position embeddings (adding a fixed vector to the token embedding) cannot provide — they encode absolute positions, not relative distances.

The practical consequence is that the attention pattern generalises well to sequence lengths not seen during training, as long as the frequencies' cycles haven't fully wrapped around. The large `rope_theta` value keeps the slow-rotating dimensions from completing a cycle within the training context length.

---

## Position IDs and Left Padding

`position_ids` is not always sequential. As section 01 noted, left-padded batches require explicit `position_ids` computed from `attention_mask` to ensure each real token gets the same RoPE position it would in an independent B=1 run. `Qwen3Model.forward` accepts `position_ids` as an argument and falls back to sequential positions only when it is `None` (valid only for B=1, no padding). The `cos` and `sin` tensors produced by `RotaryEmbedding` directly reflect whatever `position_ids` the caller provides — the embedding module has no opinion about whether positions are sequential. This is what makes the same RoPE implementation work correctly for both prefill (where `position_ids` is `[B, max_prompt_len]` with the `cumsum` correction) and decode (where `position_ids` is `[B, 1]` with `prompt_lens + decode_step`).
