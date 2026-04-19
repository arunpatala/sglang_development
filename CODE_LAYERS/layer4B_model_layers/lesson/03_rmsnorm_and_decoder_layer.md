# 03 — RMSNorm and the Decoder Layer

## The Skeleton of Each Forward Pass

After `from_pretrained` returns (section 02), every call to `self.model(...)` enters `Qwen3ForCausalLM.forward`, which delegates to `Qwen3Model.forward`. The inner model runs a loop over 28 identical blocks:

```python
for layer in self.layers:                    # 28 × Qwen3DecoderLayer
    hidden = layer(hidden, cos, sin, additive_mask, kv_cache)
```

`Qwen3DecoderLayer` is what that `layer(...)` call resolves to. Its forward method is the entire computation that transforms a `[B, q_len, hidden]` tensor into another tensor of the same shape, one block at a time. Understanding the decoder layer is understanding 28/30 of the model's total computation — the remaining 2/30 being the embedding lookup and the final projection to vocabulary size.

---

## RMSNorm

Every normalisation step in the model uses `RMSNorm` from `model/norm.py`. It appears in four distinct roles:

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (self.weight * norm).to(dtype)
```

The formula normalises each vector `x` in the last dimension by its root-mean-square: `x / sqrt(mean(x²) + eps)`. A learned per-dimension scale `weight` (initialised to ones) is then multiplied in, giving the model a way to undo the normalisation where beneficial.

The cast to `float32` on the second line is not optional. At `bfloat16` precision, `x.pow(2)` squares values that may already be close to the smallest representable number, pushing them to zero before the mean is even computed. `rsqrt` of zero produces `inf`, and the multiplication with `x` produces `nan`. The pattern cast-compute-cast-back is standard in every production transformer implementation — the norm is computed in `float32` for numerical safety, then the result is cast back to `bfloat16` for the rest of the computation.

The four roles where `RMSNorm` appears are: `input_layernorm` (pre-attention, one per decoder layer), `post_attention_layernorm` (pre-MLP, one per decoder layer), `model.norm` (final norm after all 28 layers, before the lm_head projection), and `q_norm`/`k_norm` inside `Qwen3Attention` (per-head query and key normalisation — explained in section 06). The first two roles appear directly in `Qwen3DecoderLayer`.

---

## Pre-Norm Architecture

`Qwen3DecoderLayer.forward` implements the pre-norm residual pattern used throughout modern transformers:

```python
def forward(self, hidden_states, cos, sin, attention_mask, kv_cache):
    # ── Self-attention sublayer ───────────────────────────────────
    residual      = hidden_states
    hidden_states = self.input_layernorm(hidden_states)
    hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask, kv_cache)
    hidden_states = residual + hidden_states

    # ── MLP sublayer ──────────────────────────────────────────────
    residual      = hidden_states
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)
    hidden_states = residual + hidden_states

    return hidden_states
```

The key structural detail is that normalisation happens *before* the sublayer, not after. The input `hidden_states` is saved as `residual`, then normalised, then passed into `self_attn`. The attention output is added back to the *un-normalised* residual. The same pattern repeats for the MLP sublayer.

This is called pre-norm (or pre-LN) to distinguish it from the original Transformer's post-norm design, which applied the norm after the residual addition. The practical difference is stability: in post-norm, the residual path carries unnormalised values that grow as layers deepen, making gradients unreliable early in training. In pre-norm, the residual path always adds a raw copy of the previous layer's output, and the normalised branch handles the sublayer computation. Gradients flow through the residual path without passing through any normalisation, making training at depth much more stable. All modern large language models — including Qwen3 — use pre-norm.

---

## SwiGLU MLP

The MLP sublayer is `Qwen3MLP`:

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
```

For Qwen3-0.6B, `hidden_size = 1024` and `intermediate_size = 3072`. A standard FFN would project up from 1024 to 3072 with one linear layer, apply an activation, and project back down. SwiGLU replaces the single up-projection with two parallel projections of the same size.

`gate_proj(x)` produces a `[B, q_len, 3072]` tensor that is passed through `F.silu` — the Sigmoid Linear Unit, defined as `x * sigmoid(x)`. This acts as a learned soft gate: values near zero are suppressed; large positive values pass through approximately unchanged. `up_proj(x)` produces another `[B, q_len, 3072]` tensor that provides the content. The element-wise multiplication `silu(...) * up_proj(x)` applies the gate to the content, selectively amplifying or suppressing each of the 3072 dimensions. `down_proj` then projects the gated output back to `hidden_size = 1024`.

The SwiGLU variant outperforms standard `ReLU` or `GELU` FFNs in practice. The gating structure lets the network route information more selectively: different tokens can activate different parts of the 3072-dimensional intermediate space depending on what the gate learns to open. The cost is that SwiGLU requires two up-projections instead of one — `gate_proj` and `up_proj` together have twice the parameters of a standard FFN's up-projection, which is why Qwen3 uses `intermediate_size = 3072` rather than `4096` (the typical ratio for standard FFNs): the total parameter count stays comparable.

---

## What the Layer Passes On

Each `Qwen3DecoderLayer` receives `hidden [B, q_len, hidden]` and returns a tensor of exactly the same shape. The `cos`, `sin`, `additive_mask`, and `kv_cache` arguments pass through to `Qwen3Attention` without modification — the decoder layer itself does not compute or alter them. This means `Qwen3Model.forward` can compute `cos`, `sin`, and `additive_mask` once before the loop and reuse the same tensors across all 28 layers, as shown in the skeleton in section 01. How `cos` and `sin` are produced is covered in section 04; how the additive mask is built is covered in section 05; how `Qwen3Attention` uses all three is covered in section 06.
