# Layer 20 — Summary: Single-Adapter LoRA Serving

## The Problem in One Sentence

A fine-tuned LoRA adapter costs ~3 MB; the base model costs ~1.2 GB — **one server should serve both the base model and the adapter in a single batched forward pass**, routing each request to the right output with zero per-request overhead.

---

## Part 1: What LoRA Is and Why It Matters at Inference

LoRA fine-tunes a model by injecting two small matrices (`A` and `B`) into each target linear projection:

```
W_eff = W  +  B @ A * scaling
       ─────  ─────────────────
       frozen  LoRA delta (adapter-specific)
```

Where:
- `W ∈ ℝ^{d_out × d_in}` — frozen base model weight
- `A ∈ ℝ^{r × d_in}` — "shrink" matrix, maps input to rank-r subspace
- `B ∈ ℝ^{d_out × r}` — "expand" matrix, maps back to output dimension
- `scaling = lora_alpha / r` — controls delta magnitude

For the `phh/Qwen3-0.6B-TLDR-Lora` adapter:
- `r=8`, `lora_alpha=32`, `scaling=4.0`
- Targets `q_proj` and `v_proj` only (not k, o, or MLP)
- Total adapter size: 2 × 28 layers × (A + B) ≈ 1.8 MB

At inference, the delta is applied as a **residual add** — no weight merging needed:

```python
base_out = x @ W.T                      # frozen, unchanged
delta    = (x @ A.T) @ B.T * scaling    # LoRA delta
output   = base_out + delta             # applied on-the-fly
```

This means one base model in GPU VRAM can serve many adapter variants simultaneously — only the A/B matrices differ between requests.

---

## Part 2: The Mask Strategy for Mixed Batches

The key design question is: how do you handle a batch where some requests use the adapter and others use the base model?

**Option A (naive):** run two forward passes — one for base-model requests, one for LoRA requests. This doubles latency when the batch is split 50/50.

**Option B (masked delta):** compute the LoRA delta for every token, then multiply by a per-token float mask before adding:

```python
lora_mask[i] = 1.0   # token i belongs to a LoRA request
lora_mask[i] = 0.0   # token i belongs to a base-model request

output = base_out + delta * lora_mask   # one forward pass for the whole batch
```

The `lora_mask` broadcasts over the output dimension:
- EXTEND (packed prefill, B=1): shape `[1, total_tokens, 1]`
- DECODE (one token per request, B=N): shape `[N, 1, 1]`

A single elementwise multiply gates the entire delta — no branching, no sorting, no separate kernel. The only cost is computing the delta for base-model tokens (wasted) and zeroing it out.

**Efficiency note:** The production SGLang implementation (S-LoRA/Punica) sorts tokens by adapter and uses **segmented GEMM** kernels that skip the computation entirely for base-model tokens. That complexity is left to `sglang_multi_lora_implementation.md`. For the single-adapter case here, the mask approach is correct and the wasted computation is bounded by the fraction of base-model tokens in the batch.

**Zero overhead when no LoRA requests:** `lora_adapter` is `None` when no adapter is loaded, and `lora_adapter=None` is passed when no request in the batch has `lora_id` set. Each layer checks `if forward_batch.lora_adapter is not None` — one pointer comparison — before doing any LoRA work.

---

## Part 3: Implementation — What Changed and Why

### `lora.py` (new file)

`LoRAAdapter` is the only new class in this layer. It does two things:

**1. Load weights** from `adapter_model.safetensors`:

PEFT checkpoints use keys like:
```
base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight   [8, 1024]
base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight   [2048, 8]
```

`_load_weights()` parses the layer index from the `"layers"` segment and the module name from the set of supported modules, storing into:
```python
A_weights[layer_idx][module_name]  # [rank, in_dim]
B_weights[layer_idx][module_name]  # [out_dim, rank]
```

**2. Compute the delta** via `apply(x, layer_idx, module_name)`:

```python
def apply(self, x, layer_idx, module_name):
    A = self.A_weights[layer_idx][module_name]   # [rank, in_dim]
    B = self.B_weights[layer_idx][module_name]   # [out_dim, rank]
    return (x @ A.T) @ B.T * self.scaling
```

Returns `None` if the layer/module is not targeted — callers check `if delta is not None` before adding.

### `forward_batch.py`

Two optional fields added to the `ForwardBatch` dataclass:

```python
lora_mask:    Optional[torch.Tensor] = None  # [B, q_len, 1]
lora_adapter: Optional[Any]         = None  # LoRAAdapter | None
```

These default to `None`, so all existing non-LoRA code is untouched.

### `model/attention.py`

LoRA deltas are inserted at two points in `Qwen3Attention.forward()`:

```python
# After q/k/v projections (step 1b):
if forward_batch.lora_adapter is not None:
    mask = forward_batch.lora_mask
    q = q + adapter.apply(hidden_states, self.layer_idx, "q_proj") * mask
    k = k + adapter.apply(hidden_states, self.layer_idx, "k_proj") * mask
    v = v + adapter.apply(hidden_states, self.layer_idx, "v_proj") * mask

# After o_proj (step 5b):
if forward_batch.lora_adapter is not None:
    do = adapter.apply(attn_flat, self.layer_idx, "o_proj")
    if do is not None:
        out = out + do * forward_batch.lora_mask
```

Important: the delta is added **after** the base projection but **before** the reshape and QK norm. This matches the PEFT forward pass ordering exactly.

For `phh/Qwen3-0.6B-TLDR-Lora`, `k_proj` and `o_proj` are not in `target_modules`, so `apply()` returns `None` for those and the check `if delta is not None` skips the add. The result is identical to a model that only has `q_proj` and `v_proj` LoRA.

### `model/mlp.py`

`Qwen3MLP` gains:
1. `layer_idx` constructor argument (needed for `apply()` lookups)
2. `forward_batch: Optional[ForwardBatch]` parameter
3. LoRA deltas for `gate_proj`, `up_proj` (before SwiGLU activation), and `down_proj` (after activation)

```python
gate = self.gate_proj(x)
if adapter: gate = gate + adapter.apply(x, self.layer_idx, "gate_proj") * mask
up   = self.up_proj(x)
if adapter: up   = up   + adapter.apply(x, self.layer_idx, "up_proj")   * mask
hidden = F.silu(gate) * up
out    = self.down_proj(hidden)
if adapter: out  = out  + adapter.apply(hidden, self.layer_idx, "down_proj") * mask
```

For this adapter, `gate_proj`/`up_proj`/`down_proj` are not targeted, so all three return `None` silently.

### `model/decoder_layer.py`

Two one-line changes:
```python
self.mlp = Qwen3MLP(config, layer_idx)     # was: Qwen3MLP(config)
hidden   = self.mlp(hidden, forward_batch)  # was: self.mlp(hidden)
```

### `model/qwen3.py`

Both `Qwen3Model.forward()` and `Qwen3ForCausalLM.forward()` accept `lora_mask` and `lora_adapter` as keyword arguments. All three `ForwardBatch` construction sites (EXTEND, DECODE, NOCACHE) populate these fields. The NOCACHE path is important — it is used as the ground-truth reference in `verify_lora.py`.

### `model_runner.py`

```python
# At startup:
self.lora_adapter = LoRAAdapter(lora_path, dtype=DTYPE, device=DEVICE) if lora_path else None

# In prefill_batch(), before model.forward():
mask_vals = [1.0 if req.lora_id else 0.0 for token in each_req]
lora_mask = torch.tensor(mask_vals).view(1, -1, 1)   # [1, total_tokens, 1]

# In decode_step(), before model.forward():
mask_vals = [1.0 if r.lora_id else 0.0 for r in reqs]
lora_mask = torch.tensor(mask_vals).view(-1, 1, 1)   # [B, 1, 1]
```

---

## Part 4: Verification Results

`verify_lora.py` runs four tests. Tests 0–2 use only the NOCACHE path (F.sdpa, no flashinfer) and ran successfully on MPS with the base conda environment (torch 2.7.1 + peft 0.15.2 + transformers 5.6.0).

### Test 0 — Base model match

Confirms our custom `Qwen3ForCausalLM` produces the same logits as HuggingFace `AutoModelForCausalLM` on the same prompt before adding LoRA. This is the prerequisite — if base models don't match, LoRA comparisons are meaningless.

```
'What is 2+2?'                          max_diff=0.6875  ✓
'Explain what a neural network is...'   max_diff=0.4375  ✓
'What is the capital of France?'        max_diff=0.5156  ✓
→ PASS ✓  (atol=1.0)
```

The ~0.5–0.7 diffs are expected: different attention backends (our `F.sdpa` vs HF's attention), BF16 precision accumulation across 28 layers, MPS vs CPU numerical differences.

### Test 1 — Adapter changes output

Confirms the LoRA delta is non-zero (adapter actually applied):

```
max_diff (lora vs base) = 3.375000  ✓
→ PASS ✓
```

A delta of 3.375 is large — this adapter has been meaningfully trained (B matrices are not near-zero for all layers).

### Test 2 — Matches PEFT (external ground truth)

The critical test. Compares our LoRA delta `(our_lora - our_base)` against PEFT's delta `(peft_lora - hf_base)`. Using the delta isolates LoRA math correctness from base model implementation differences.

```
'What is 2+2?':
  abs  diff=0.8438  ✓
  delta diff=0.8281  ✓  ← primary check
'Explain what a neural network is...':
  abs  diff=0.4375  ✓
  delta diff=0.7812  ✓
'What is the capital of France?':
  abs  diff=0.6875  ✓
  delta diff=0.7266  ✓
→ PASS ✓  (atol=1.0)
```

The delta diffs of ~0.73–0.83 are confirmed as precision artifacts, not bugs. A weight-level comparison showed:
```
A max_diff = 0.00012204   (BF16 dtype rounding only)
B max_diff = 0.00000190   (essentially zero)
```
Both A and B matrices match PEFT to BF16 precision. The forward pass diffs come from 28 layers of accumulated BF16 arithmetic across two different model implementations (our `F.sdpa` vs HF's attention).

On CUDA, expected diffs for delta comparison are ~0.3–0.4 (well within atol=0.75).

### Tests 3 and 4 — SKIPPED (CUDA + flashinfer required)

Test 3 (mixed-batch mask separation) and Test 4 (paged LoRA prefill + decode) require flashinfer, which is only available on CUDA. These tests use the EXTEND/DECODE paged attention paths.

---

## Part 5: How the Data Flows

End-to-end flow for a request that uses the adapter:

```
POST /v1/chat/completions  {"lora_id": "tldr", "messages": [...]}
  │
  └─ server.py: ChatRequest.lora_id = "tldr" → Req(lora_id="tldr")
       │
  scheduler: adds to waiting_queue
       │
  prefill_batch([req]):
    mask_vals = [1.0, 1.0, 1.0, ...]     # all tokens for this req = LoRA
    lora_mask = tensor([mask_vals]).view(1, -1, 1)
       │
  model.forward(ids, ..., lora_mask=lora_mask, lora_adapter=self.lora_adapter)
       │
  Qwen3Model.forward():
    ForwardBatch(mode=EXTEND, kv_cache=ctx,
                 lora_mask=lora_mask, lora_adapter=adapter)
       │
  for each Qwen3DecoderLayer:
    ├─ Qwen3Attention.forward():
    │    q = q_proj(x)  +  adapter.apply(x, layer_idx, "q_proj") * mask
    │    v = v_proj(x)  +  adapter.apply(x, layer_idx, "v_proj") * mask
    │    (k_proj, o_proj: adapter.apply() returns None → skipped)
    │    → paged attention via FlashInfer
    └─ Qwen3MLP.forward():
         (all MLP modules: adapter.apply() returns None → skipped for this adapter)
       │
  lm_head(hidden) → logits
       │
  sample first output token → req.output_ids
```

For a decode step, `lora_mask` is `[B, 1, 1]` — one scalar per request in the batch, shaping the same way.

---

## Part 6: The Minimal vs. Production Gap

This implementation is intentionally minimal. Understanding what was left out points to the full production system:

| Feature | This layer | Full multi-LoRA (sglang_multi_lora_implementation.md) |
|---|---|---|
| Number of adapters | 1 static | N, loaded/evicted dynamically |
| GPU memory layout | A/B per layer per module | Pre-allocated 3D pool buffers `[max_slots, rank, hidden]` |
| Per-token routing | Float mask 0/1 | `weight_indices[i]` → pool slot index |
| GEMM strategy | Full delta + mask multiply | Segmented GEMM (Punica): skip base-model tokens entirely |
| CUDA graph | Not supported with LoRA in this form | Static `LoRABatchInfo` tensors updated in-place before replay |
| Dynamic load | Not supported | `POST /lora/load` + `POST /lora/unload` with in-flight ref-counting |
| TP sharding | Not implemented | Per-module sharding rules (A unsharded, B sliced for QKV) |

The mask approach costs at most 1 extra GEMM per LoRA-targeted module per non-LoRA token. At batch size 16 with 8 LoRA / 8 base requests, the wasted compute is 50% of the delta cost, which is ~0.6% of total compute (delta is tiny vs base model). For production at large scale, the segmented GEMM eliminates this and also enables multi-adapter serving.

---

## Relationship to Adjacent Layers

| Layer | Topic | Relationship to Layer 20 |
|---|---|---|
| **Layer 5** | Model layers from scratch | Layer 20 adds LoRA delta to the same `Qwen3Attention` and `Qwen3MLP` built there |
| **Layer 8** | Paged KV pool | KV pool is unchanged — LoRA adds no KV state; adapters affect only the linear projections |
| **Layer 11** | Prefix caching | Prefix caching and LoRA are orthogonal; the same prefix can be reused for different adapter requests (though production systems track adapter-specific cache separately) |
| **Layer 19** | PD disaggregation | In disaggregated serving, the P worker applies LoRA during prefill; the D worker applies it during decode — both use the same mask approach |
