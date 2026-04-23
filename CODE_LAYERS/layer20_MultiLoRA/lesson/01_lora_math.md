# Section 01 — LoRA Math and Why Rank Decomposition Works

## The Core Idea

Fine-tuning a large language model by updating all of its weights is expensive: Qwen3-0.6B has ~600 M parameters; a full fine-tune requires storing gradients and optimizer state for all of them. But empirically, the *change* from a base model to a fine-tuned model lives in a surprisingly low-dimensional subspace. LoRA exploits this directly.

Instead of adding a full `[d_out, d_in]` weight update matrix, LoRA learns two small matrices:

```
ΔW = B @ A          where A ∈ ℝ^{r × d_in},  B ∈ ℝ^{d_out × r},  r ≪ min(d_in, d_out)
```

The effective weight during inference is:

```
W_eff = W  +  B @ A * scaling
        ────  ────────────────
        frozen  LoRA delta
```

This is equivalent to splitting the forward pass into two parts and adding their outputs:

```python
base_output  = x @ W.T                       # [B, q_len, d_out]
lora_output  = (x @ A.T) @ B.T * scaling     # [B, q_len, d_out]
total_output = base_output + lora_output
```

Neither `W`, `A`, nor `B` are ever merged — the residual add is computed fresh on every forward pass. This means one GPU holds the frozen base model once, and many adapters can be applied without touching the base weights.

---

## The Scaling Factor

The scaling is `lora_alpha / r`, not `1 / r`. This gives a hyperparameter (`lora_alpha`) that adjusts the overall magnitude of the adapter's effect without changing the rank. A higher `alpha` → larger deltas; a lower `alpha` → softer adaptation. At training time this absorbs into the effective learning rate for the adapter.

For the adapter used in this layer:

```
r          =  8
lora_alpha =  32
scaling    =  32 / 8  =  4.0
```

A scaling of 4.0 means the adapter's output is amplified 4× relative to a naïve rank-8 projection. This is why `test_adapter_changes_output` observes a max-diff of ~3.4 — the adapter has a meaningful, non-trivial effect on the logits.

---

## Why Rank Decomposition Works

The weight matrices in transformer attention and MLP layers are large but not independently random. After pre-training, the model uses a relatively compact "internal vocabulary" of transformations. Fine-tuning shifts which transformations are most heavily used — and that shift lives in a subspace whose intrinsic rank is much lower than `d`.

**The original LoRA paper (Hu et al., 2021) showed:**
- For GPT-3 175B, adapting to downstream tasks required as few as `r=4` components
- The updates to different attention layers were approximately low-rank in practice (not just by construction)
- `r=4` or `r=8` generally sufficed; going higher gave diminishing returns

**Why `q_proj` and `v_proj` are targeted (not `k_proj`, `o_proj`, MLP):**

The `phh/Qwen3-0.6B-TLDR-Lora` adapter targets only `q_proj` and `v_proj`. This follows a finding from the LoRA paper: adapting Q and V (the "query" and "value" that determine what to attend to and what to extract) is sufficient for many tasks; K primarily acts as a lookup key and is more stable. MLP layers are adapted in some setups but not here.

This is visible in `adapter_config.json`:

```json
{
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "v_proj"],
  ...
}
```

And in our `LoRAAdapter.apply()` — any call for `k_proj`, `o_proj`, or MLP modules returns `None`:

```python
def apply(self, x, layer_idx, module_name):
    if not self.has_layer(layer_idx, module_name):
        return None      # k_proj, o_proj, gate/up/down → skipped
    A = self.A_weights[layer_idx][module_name]   # [8, 1024]
    B = self.B_weights[layer_idx][module_name]   # [1024 or 2048, 8]
    return (x @ A.T) @ B.T * self.scaling
```

---

## Concrete Numbers: Qwen3-0.6B + rank 8

Qwen3-0.6B architecture:
- `hidden_size = 1024`
- `num_attention_heads = 16`, `head_dim = 128` → `q_dim = 2048`
- `num_key_value_heads = 8` → `kv_dim = 1024`
- `num_hidden_layers = 28`

For each targeted module:

| Module | A shape | B shape | delta params | base params | overhead |
|---|---|---|---|---|---|
| `q_proj` | `[8, 1024]` | `[2048, 8]` | 8,192 + 16,384 = 24,576 | 1024×2048 = 2,097,152 | 1.17% |
| `v_proj` | `[8, 1024]` | `[1024, 8]` | 8,192 + 8,192 = 16,384 | 1024×1024 = 1,048,576 | 1.56% |

Across 28 layers:
- Total LoRA parameters: 28 × (24,576 + 16,384) = **1,146,880**
- Total base model parameters: ~600,000,000
- **LoRA overhead: ~0.19%**

In bytes (BF16, 2 bytes each): 1,146,880 × 2 = **~2.2 MB** for the entire adapter.

---

## What Happens When the Adapter is Not Targeted

For layers/modules not in `target_modules`, `LoRAAdapter.apply()` returns `None`. The calling code checks before adding:

```python
dq = ada.apply(hidden_states, self.layer_idx, "q_proj")   # returns [B, q_len, q_dim]
dk = ada.apply(hidden_states, self.layer_idx, "k_proj")   # returns None
dv = ada.apply(hidden_states, self.layer_idx, "v_proj")   # returns [B, q_len, kv_dim]

if dq is not None: q = q + dq * mask    # applied for q_proj
if dk is not None: k = k + dk * mask    # SKIPPED for k_proj
if dv is not None: v = v + dv * mask    # applied for v_proj
```

There is zero computation overhead for untargeted modules — `has_layer()` is a dictionary lookup (`O(1)`), and `None` returned means no GEMM is launched.

---

## LoRA vs Full Fine-tuning vs Other PEFT Methods

| Method | Trainable params | Inference overhead | Multi-task support |
|---|---|---|---|
| Full fine-tune | 100% | None (merged) | One model per task |
| LoRA | 0.1–1% | One extra GEMM per layer per module | N adapters share 1 base model |
| Prefix tuning | <0.1% | Extra KV positions | One model per task |
| IA³ | <0.01% | Element-wise scale | N adapters share 1 base model |

LoRA is the dominant PEFT method for inference serving because:
1. The adapter can remain separate (no weight merging needed)
2. Multiple adapters can share a single loaded base model
3. The overhead is proportional to `r`, which is typically 4–64

The production multi-LoRA system (S-LoRA, documented in `sglang_multi_lora_implementation.md`) exploits property 2 at scale: dozens of adapters in a GPU pool, with requests dynamically routed to the correct adapter.

---

## Relationship to Weight Merging

One common deployment choice is to *merge* the LoRA weights into the base model at the end of training:

```python
W_merged = W + B @ A * scaling    # one-time operation
```

This eliminates inference overhead entirely — the merged model is just a regular model. The downside is that you now have one model per adapter and cannot switch between base and adapter at request time.

This layer intentionally does **not** merge — we keep `W` frozen and compute the residual add per forward pass, which is required to serve both base-model and LoRA requests from a single loaded model.
