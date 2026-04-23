# Section 04 — Wiring LoRA into the Forward Pass

## Overview: Five Files Changed

The LoRA delta must be applied at specific points inside the model's forward pass. This requires threading two pieces of state — `lora_mask` and `lora_adapter` — from the top-level `Qwen3ForCausalLM.forward()` call down to each individual projection in each of the 28 decoder layers.

Five files change:

| File | Change |
|---|---|
| `forward_batch.py` | Add `lora_mask` and `lora_adapter` fields to `ForwardBatch` |
| `model/qwen3.py` | Accept `lora_mask`/`lora_adapter` kwargs; populate `ForwardBatch` |
| `model/attention.py` | Apply delta after q/k/v projections and after o_proj |
| `model/mlp.py` | Accept `forward_batch`; apply delta after gate/up/down projections |
| `model/decoder_layer.py` | Pass `layer_idx` to MLP; pass `forward_batch` to `mlp()` |

---

## `forward_batch.py`: Extending ForwardBatch

`ForwardBatch` is the dataclass that carries per-call state through the entire model stack. It already holds `mode`, `kv_cache`, and `attention_mask`. Two optional fields are added:

```python
@dataclass
class ForwardBatch:
    mode:           ForwardMode
    kv_cache:       object
    attention_mask: Optional[torch.Tensor]
    lora_mask:    Optional[torch.Tensor] = None   # [B, q_len, 1]
    lora_adapter: Optional[Any]         = None   # LoRAAdapter | None
```

Both fields default to `None`. All existing code that creates `ForwardBatch` without these fields continues to work unchanged. This is the only addition to `forward_batch.py`.

---

## `model/qwen3.py`: Accepting LoRA Arguments

### `Qwen3Model.forward()`

Two new keyword arguments are added to the signature:

```python
def forward(
    self,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor | None,
    kv_cache=None,
    position_ids:   torch.Tensor | None = None,
    lora_mask:    torch.Tensor | None = None,   # NEW
    lora_adapter=None,                          # NEW
) -> torch.Tensor:
```

All three `ForwardBatch` construction sites (EXTEND, DECODE, NOCACHE) are updated to populate the new fields:

```python
# EXTEND (paged prefill):
forward_batch = ForwardBatch(
    mode=ForwardMode.EXTEND,
    kv_cache=kv_cache,
    attention_mask=None,
    lora_mask=lora_mask,      # passed through
    lora_adapter=lora_adapter,
)

# DECODE (paged decode):
forward_batch = ForwardBatch(
    mode=ForwardMode.DECODE,
    kv_cache=kv_cache,
    attention_mask=None,
    lora_mask=lora_mask,
    lora_adapter=lora_adapter,
)

# NOCACHE (verify path, F.sdpa):
forward_batch = ForwardBatch(
    mode=ForwardMode.NOCACHE,
    kv_cache=None,
    attention_mask=additive_mask,
    lora_mask=lora_mask,
    lora_adapter=lora_adapter,
)
```

The NOCACHE path is critical — `verify_lora.py` uses it as the ground-truth reference, so it must also propagate the LoRA state.

### `Qwen3ForCausalLM.forward()`

Two new kwargs are added and forwarded to `self.model`:

```python
def forward(
    self,
    input_ids:      torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    kv_cache=None,
    position_ids:   torch.Tensor | None = None,
    lora_mask=None,      # NEW
    lora_adapter=None,   # NEW
) -> torch.Tensor:
    hidden = self.model(
        input_ids, attention_mask, kv_cache, position_ids,
        lora_mask=lora_mask, lora_adapter=lora_adapter,   # forwarded
    )
    return self.lm_head(hidden)
```

---

## `model/attention.py`: LoRA Deltas for Q/K/V/O

`Qwen3Attention.forward()` already had five steps. LoRA inserts two new half-steps:

### Step 1 → 1b: After Q/K/V projections

```python
# ── 1. Project Q / K / V ──────────────────────────────────────────────
q = self.q_proj(hidden_states)   # [B, q_len, q_dim]
k = self.k_proj(hidden_states)   # [B, q_len, kv_dim]
v = self.v_proj(hidden_states)   # [B, q_len, kv_dim]

# ── 1b. LoRA deltas for Q / K / V ─────────────────────────────────────
if forward_batch.lora_adapter is not None:
    ada  = forward_batch.lora_adapter
    mask = forward_batch.lora_mask   # [B, q_len, 1]
    dq = ada.apply(hidden_states, self.layer_idx, "q_proj")   # [B, q_len, q_dim] or None
    dk = ada.apply(hidden_states, self.layer_idx, "k_proj")   # None for this adapter
    dv = ada.apply(hidden_states, self.layer_idx, "v_proj")   # [B, q_len, kv_dim] or None
    if dq is not None: q = q + dq * mask
    if dk is not None: k = k + dk * mask
    if dv is not None: v = v + dv * mask

# Step 2 (view/reshape) comes AFTER LoRA:
q = q.view(B, q_len, self.num_heads,    self.head_dim)
k = k.view(B, q_len, self.num_kv_heads, self.head_dim)
v = v.view(B, q_len, self.num_kv_heads, self.head_dim)
```

**Critical ordering:** the `.view()` reshape happens *after* the LoRA delta is added. If the reshape occurred before, the delta would need to be computed on the already-reshaped `q` (shape `[B, q_len, n_heads, head_dim]`), but the LoRA B matrix is `[q_dim, rank]` — it expects the flat `q_dim` dimension. Adding delta before reshape is the correct ordering and exactly matches PEFT's implementation.

### Step 5 → 5b: After O-proj

```python
# ── 5. Merge heads and output projection ──────────────────────────────
attn_flat = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
out = self.o_proj(attn_flat)

# ── 5b. LoRA delta for o_proj ──────────────────────────────────────────
if forward_batch.lora_adapter is not None:
    do = forward_batch.lora_adapter.apply(attn_flat, self.layer_idx, "o_proj")
    if do is not None:
        out = out + do * forward_batch.lora_mask
```

Note: the input to `o_proj`'s LoRA is `attn_flat` (the merged attention output, shape `[B, q_len, q_dim]`), not `hidden_states`. This is correct — `o_proj` maps from the attention output space, not the input hidden space.

For `phh/Qwen3-0.6B-TLDR-Lora`, `o_proj` is not in `target_modules`, so `do` is `None` and this block adds zero overhead.

---

## `model/mlp.py`: LoRA Deltas for Gate/Up/Down

Two changes to `Qwen3MLP`:

### 1. Accept `layer_idx` in `__init__`

```python
class Qwen3MLP(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:  # added layer_idx
        super().__init__()
        self.layer_idx = layer_idx   # stored for apply() lookups
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj   = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)
```

### 2. Accept `forward_batch` and apply deltas

```python
def forward(
    self,
    x: torch.Tensor,
    forward_batch: Optional["ForwardBatch"] = None,   # added
) -> torch.Tensor:
    gate = self.gate_proj(x)
    up   = self.up_proj(x)

    # ── LoRA deltas for gate_proj and up_proj ──────────────────────────
    if forward_batch is not None and forward_batch.lora_adapter is not None:
        ada  = forward_batch.lora_adapter
        mask = forward_batch.lora_mask   # [B, q_len, 1]
        delta_gate = ada.apply(x, self.layer_idx, "gate_proj")
        delta_up   = ada.apply(x, self.layer_idx, "up_proj")
        if delta_gate is not None: gate = gate + delta_gate * mask
        if delta_up   is not None: up   = up   + delta_up   * mask

    hidden = F.silu(gate) * up          # SwiGLU activation
    out    = self.down_proj(hidden)

    # ── LoRA delta for down_proj ──────────────────────────────────────
    if forward_batch is not None and forward_batch.lora_adapter is not None:
        delta_down = forward_batch.lora_adapter.apply(hidden, self.layer_idx, "down_proj")
        if delta_down is not None:
            out = out + delta_down * forward_batch.lora_mask

    return out
```

**Why `gate_proj`/`up_proj` deltas come before the SwiGLU activation:** the activation gate modifies the signal nonlinearly; the LoRA delta must be added to the linear projection output before the nonlinearity. This matches the mathematical formulation `δ(gate_proj(x) + LoRA_gate(x))`.

**Why `down_proj` delta input is `hidden` (not `x`):** `down_proj` maps from `intermediate_size` back to `hidden_size`. Its LoRA A matrix is `[rank, intermediate_size]` — the input must be the post-activation intermediate hidden, not the pre-activation input `x`.

For this adapter (MLP not targeted): all three `apply()` calls return `None` — the entire LoRA block is entered once for the pointer check, then immediately skipped.

---

## `model/decoder_layer.py`: Two One-Line Changes

```python
class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx  = layer_idx
        self.self_attn  = Qwen3Attention(config, layer_idx)
        self.mlp        = Qwen3MLP(config, layer_idx)   # CHANGED: added layer_idx
        self.input_layernorm    = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # Attention sub-layer
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, forward_batch)
        hidden_states = residual + hidden_states

        # MLP sub-layer
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states, forward_batch)   # CHANGED: pass forward_batch
        hidden_states = residual + hidden_states

        return hidden_states
```

Both changes are minimal:
1. `Qwen3MLP(config, layer_idx)` — adds the layer index at construction time
2. `self.mlp(hidden_states, forward_batch)` — passes `forward_batch` so the MLP can read `lora_adapter` and `lora_mask`

The `forward_batch` was already passed to `self.self_attn` (attention always needed it for KV cache dispatch), so this is symmetric.

---

## The Complete LoRA Signal Path

```
Qwen3ForCausalLM.forward(lora_mask=M, lora_adapter=A)
  │
  └─ Qwen3Model.forward(lora_mask=M, lora_adapter=A)
       │
       │  ForwardBatch(mode=..., kv_cache=..., lora_mask=M, lora_adapter=A)
       │
       └─ for layer in self.layers:
            Qwen3DecoderLayer.forward(hidden, cos, sin, forward_batch)
              │
              ├─ Qwen3Attention.forward(hidden, cos, sin, forward_batch)
              │    ├─ q = q_proj(x)
              │    ├─ q = q + adapter.apply(x, i, "q_proj") * mask   ← LoRA
              │    ├─ k = k_proj(x)
              │    ├─ (k: returns None, no-op)
              │    ├─ v = v_proj(x)
              │    ├─ v = v + adapter.apply(x, i, "v_proj") * mask   ← LoRA
              │    ├─ ... attention, RoPE, paged kernel ...
              │    ├─ out = o_proj(attn_flat)
              │    └─ (o: returns None, no-op)
              │
              └─ Qwen3MLP.forward(hidden, forward_batch)
                   ├─ gate = gate_proj(x)
                   ├─ (gate: returns None, no-op)
                   ├─ up = up_proj(x)
                   ├─ (up: returns None, no-op)
                   ├─ hidden = silu(gate) * up
                   ├─ out = down_proj(hidden)
                   └─ (down: returns None, no-op)
```

For this adapter (`q_proj` and `v_proj` only): 56 LoRA `apply()` calls across 28 layers, of which only 56 result in non-None (q and v) and the remaining 112 return None immediately (k, o, gate, up, down).

---

## Why Not Pass lora_mask/lora_adapter Directly to Each Layer?

Alternative: skip `ForwardBatch` and add `lora_mask` and `lora_adapter` as direct arguments to each `forward()` signature.

This was not done because `ForwardBatch` is already the carrier for all per-call state (mode, kv_cache, attention_mask). Adding more fields to `ForwardBatch` is consistent with the existing pattern and avoids proliferating keyword arguments across the call chain. The attention module already receives the full `ForwardBatch` for KV cache dispatch — it is natural for LoRA state to travel the same way.
