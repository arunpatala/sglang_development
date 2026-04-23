# Section 02 — The Mask-Based Mixed-Batch Strategy

## The Problem

A deployed LLM server receives a continuous stream of requests. Some requests want the base model; some want a LoRA adapter. Both types arrive simultaneously and are batched together for throughput.

The naive approach — run a separate forward pass for each set — is expensive:

```
Batch: [req1_base, req2_lora, req3_base, req4_lora]

Option A (naive):
  pass 1: [req1_base, req3_base]  → 2 active requests, wasted GPU
  pass 2: [req2_lora, req4_lora]  → 2 active requests, wasted GPU
  total: 2 forward passes, half GPU utilisation per pass
```

The GPU is most efficient when the batch is large. Splitting by adapter halves batch size and halves throughput.

---

## The Mask Solution

Instead of routing, compute the LoRA delta for **every token in the batch**, then multiply by a per-token binary mask before adding:

```python
base_output  = x @ W.T                        # computed for all tokens
lora_delta   = (x @ A.T) @ B.T * scaling      # computed for all tokens
output       = base_output + lora_delta * mask # mask gates which tokens get delta
```

Where:
```
mask[i] = 1.0   →  token i belongs to a LoRA request → delta is added
mask[i] = 0.0   →  token i belongs to a base request  → delta is zeroed out
```

The entire batch runs in one forward pass. The mask is a float tensor (not bool) to allow the multiply to be fused and to work cleanly in BF16.

---

## Shape Conventions

The mask must broadcast correctly over the output tensor at each projection:

### EXTEND mode (packed prefill)

The `ids_t` tensor is `[1, total_tokens]` — all requests' tokens packed into one sequence with `B=1`. The mask must be `[1, total_tokens, 1]`:

```
Request 1 (LoRA,  3 tokens): mask = [1.0, 1.0, 1.0]
Request 2 (base,  2 tokens): mask = [0.0, 0.0]
Request 3 (LoRA,  4 tokens): mask = [1.0, 1.0, 1.0, 1.0]

→ lora_mask = [1.0, 1.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]
→ shape: [1, 9, 1]
```

The trailing `1` broadcasts over the output feature dimension (e.g., `q_dim=2048`), so the add becomes:

```python
q = q + dq * lora_mask   # [1, 9, 2048] + [1, 9, 2048] * [1, 9, 1]
```

The broadcasting makes the mask a per-token scalar — efficient and simple.

### DECODE mode (one token per request)

The `ids_t` tensor is `[B, 1]` — one token per active request. The mask is `[B, 1, 1]`:

```
Request 1 (LoRA): mask = [[1.0]]
Request 2 (base): mask = [[0.0]]
Request 3 (LoRA): mask = [[1.0]]

→ lora_mask = [[[1.0]], [[0.0]], [[1.0]]]
→ shape: [3, 1, 1]
```

Broadcasting again makes this a per-request scalar during decode.

---

## Building the Mask in model_runner.py

### Prefill (EXTEND)

```python
# Step 7 of prefill_batch(), model_runner.py:336
if self.lora_adapter is not None:
    mask_vals = []
    for req in reqs:
        mask_vals.extend(
            [1.0 if req.lora_id else 0.0] * req.extend_input_len
        )
    lora_mask = torch.tensor(mask_vals, dtype=DTYPE, device=DEVICE).view(1, -1, 1)
else:
    lora_mask = None
```

- `req.lora_id` is a string (the adapter name) or `None` (base model).
- `req.extend_input_len` is the number of new tokens being prefilled for this request (may be a chunk, not the full prompt).
- The mask is built as a flat list and then reshaped to `[1, total_tokens, 1]`.

### Decode

```python
# decode_step(), model_runner.py
if self.lora_adapter is not None:
    mask_vals = [1.0 if r.lora_id else 0.0 for r in running_reqs]
    lora_mask = torch.tensor(mask_vals, dtype=DTYPE, device=DEVICE).view(-1, 1, 1)
else:
    lora_mask = None
```

---

## Zero Overhead When No LoRA Is Configured

If `lora_path` is not set in `config.yml`, `self.lora_adapter` is `None`. The mask is never built:

```python
lora_mask = None    # when lora_adapter is None
```

Inside each layer, the check is:

```python
if forward_batch.lora_adapter is not None:
    ...    # entire block skipped when adapter=None
```

This is one pointer comparison per layer — the cost is negligible. There are no branches inside the LoRA block itself; the entire block is skipped.

---

## Efficiency Cost of the Mask Approach

The mask approach wastes compute proportional to base-model tokens in the batch.

**Worst case (50/50 split):**

```
LoRA tokens:  50%  → delta needed
Base tokens:  50%  → delta computed and zeroed out

Wasted compute: 50% of the LoRA GEMM cost
```

For a single targeted module at rank 8, the LoRA GEMMs are:
- `x @ A.T`:  `[total_tokens, 1024] × [1024, 8]`  = `O(total_tokens × 1024 × 8)`
- `(.) @ B.T`: `[total_tokens, 8] × [8, 2048]` = `O(total_tokens × 8 × 2048)`

The base projection is `O(total_tokens × 1024 × 2048)`. The ratio is roughly:

```
LoRA GEMM cost / base GEMM cost ≈ (1024×8 + 8×2048) / (1024×2048)
                                 = (8,192 + 16,384) / 2,097,152
                                 ≈ 1.2%
```

Even wasting 50% of this 1.2% is only ~0.6% overhead — negligible in practice for the single-adapter case.

**The production fix (Punica / S-LoRA segmented GEMM):**

In the full multi-LoRA system, `LoRABatchInfo` contains the sorted token-to-adapter mapping, and a Triton segmented GEMM kernel skips base-model tokens entirely. The `weight_indices` tensor maps each token to a pool slot index (or -1 for base model), enabling the kernel to gather only the relevant A/B rows.

For this layer, the simpler mask approach is correct, and the wasted FLOPs are bounded and small.

---

## Why Float, Not Bool

The mask could be a `bool` tensor with `torch.where()` instead of multiplication:

```python
# Alternative (slower on GPU):
output = torch.where(mask_bool.unsqueeze(-1), base + delta, base)
```

Using a float mask and simple multiplication (`* mask`) is preferred because:
1. It fuses cleanly with the preceding GEMM output in most backends
2. BF16 multiply by 0.0 is defined and exact (no denormal issues at these magnitudes)
3. No dtype conversion needed — `mask` is already BF16 to match the activations
4. The compiler/kernel can potentially fuse `base + delta * mask` into a single FMA pass

---

## Chunked Prefill and the Mask

When `chunked_prefill_size` is set (default 512 tokens), a long prompt may be split across multiple `prefill_batch` calls. The mask is rebuilt independently for each chunk:

```python
# Each chunk sees the correct lora_id for its requests
mask_vals.extend([1.0 if req.lora_id else 0.0] * req.extend_input_len)
```

`req.extend_input_len` is the chunk length (not the total prompt length), so the mask is always `[1, chunk_len, 1]` regardless of how many chunks the prompt spans. The LoRA delta accumulates correctly in the KV cache across chunks because the KV positions are packed contiguously by the paged attention system.
