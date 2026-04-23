# Section 06 — Verification: `verify_lora.py`

## Overview

`verify_lora.py` tests the LoRA implementation at four levels of increasing complexity. Tests 0–2 run on any device (CUDA, MPS, CPU) using only the NOCACHE path (`F.scaled_dot_product_attention`). Tests 3–4 require CUDA and FlashInfer and test the full paged attention path.

```bash
# Run all tests (MPS or CUDA):
python verify_lora.py

# Explicit arguments:
python verify_lora.py --model Qwen/Qwen3-0.6B --lora phh/Qwen3-0.6B-TLDR-Lora --atol 1.0
```

---

## Why Ground-Truth Comparison Against PEFT?

It would be easy to run the model and observe that output changes when LoRA is applied. But "output changes" only proves the adapter is connected — it does not prove the math is correct. The B matrix could be applied transposed; the scaling could be inverted; the layers could be offset by one.

HuggingFace PEFT is the canonical LoRA implementation. If our `(x @ A.T) @ B.T * scaling` matches PEFT's output for the same weights, the math is correct by construction. The comparison is at the logit level — the final `[vocab_size]` vector — which is sensitive to any error anywhere in the forward pass.

---

## Reference Functions

### `our_nocache_logit(model, token_ids, adapter)`

Runs our `Qwen3ForCausalLM` in NOCACHE mode and returns the last-token logit:

```python
def our_nocache_logit(model, token_ids, adapter=None):
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    pos_t = torch.arange(len(token_ids), device=DEVICE).unsqueeze(0)
    mask  = torch.ones(1, len(token_ids), dtype=torch.long, device=DEVICE)

    lora_mask = None
    if adapter is not None:
        lora_mask = torch.ones(1, len(token_ids), 1, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        lg = model(ids_t, attention_mask=mask, kv_cache=None,
                   position_ids=pos_t, lora_mask=lora_mask, lora_adapter=adapter)
    return lg[0, -1, :]
```

When `adapter=None`: standard base model forward. When `adapter` is set: `lora_mask` is all-ones (all tokens use the adapter) — equivalent to a single-request LoRA batch with no base-model tokens.

### `peft_logit(peft_model, token_ids)`

Runs HuggingFace `PeftModel` and returns the last-token logit:

```python
def peft_logit(peft_model, token_ids):
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = peft_model(input_ids=ids_t)
    return out.logits[0, -1, :]
```

### `hf_base_logit(hf_model, token_ids)`

Runs `AutoModelForCausalLM` (no LoRA) for the base model reference.

---

## Test 0 — Base Model Match (Prerequisite)

**What it checks:** Our custom `Qwen3ForCausalLM` (NOCACHE path, `F.sdpa`) produces logits within `atol` of HuggingFace `AutoModelForCausalLM` on the same prompt, with no LoRA.

**Why this comes first:** If the base models disagree, LoRA comparisons are unreliable. Test 2 compares LoRA deltas, which requires both base models to be numerically close.

```python
def test_base_match(our_model, hf_model, tok, prompts):
    for p in prompts:
        ids    = tokenize(p)
        our_lg = our_nocache_logit(our_model, ids, adapter=None)
        hf_lg  = hf_base_logit(hf_model, ids)
        diff   = (our_lg - hf_lg).abs().max().item()
        ok     = diff < ATOL
```

**Expected diffs and why they're non-zero:**
- Our model uses `F.scaled_dot_product_attention` with a precomputed causal+padding additive mask.
- HuggingFace's Qwen3 uses its own attention implementation with a different masking path.
- Both use BF16, accumulating rounding error across 28 layers.
- On MPS (Apple Silicon), BF16 arithmetic differs from CUDA in low bits.

Observed on MPS: `max_diff ≈ 0.44–0.69`. With `atol=1.0`, all prompts pass.

---

## Test 1 — Adapter Changes Output

**What it checks:** Applying the adapter actually changes the logits (delta is non-zero and meaningful).

```python
def test_adapter_changes_output(our_model, adapter, tok, prompt):
    ids    = tokenize(prompt)
    base_lg = our_nocache_logit(our_model, ids, adapter=None)
    lora_lg = our_nocache_logit(our_model, ids, adapter=adapter)
    diff    = (lora_lg - base_lg).abs().max().item()
    ok      = diff > 1e-3   # any non-trivial change
```

**Expected result:** A large, clear difference. For `phh/Qwen3-0.6B-TLDR-Lora`, the observed `max_diff ≈ 3.375`. This is expected because:
- `scaling = 4.0` (amplifies the adapter's output)
- The adapter was actually trained (B matrices are not near-zero)
- q/v projections directly shape the attention pattern in every layer

A diff near zero would indicate the adapter is not connected — typically caused by a key name mismatch in `_load_weights()` or a missing `apply()` call in the forward pass.

---

## Test 2 — Matches PEFT (Core Correctness)

**What it checks:** Our LoRA math matches HuggingFace `PeftModel` on the same weights.

**The naive approach (absolute comparison) is fragile:**

```python
# This can fail due to base model differences:
diff = (our_lora_lg - peft_lg).abs().max()
```

Even if our LoRA math is perfectly correct, our base model produces slightly different numbers than HF's base model (as Test 0 shows). The LoRA output inherits this offset.

**The delta comparison isolates LoRA correctness:**

```python
def test_lora_matches_peft(our_model, adapter, hf_model, peft_model, tok, prompts):
    for p in prompts:
        ids = tokenize(p)

        our_base_lg = our_nocache_logit(our_model, ids, adapter=None)
        our_lora_lg = our_nocache_logit(our_model, ids, adapter=adapter)
        hf_base_lg  = hf_base_logit(hf_model, ids)
        peft_lg     = peft_logit(peft_model, ids)

        # (a) Absolute: affected by base model differences
        diff_abs   = (our_lora_lg - peft_lg).abs().max().item()

        # (b) Delta: LoRA contribution only
        our_delta  = our_lora_lg - our_base_lg    # what LoRA added in our model
        peft_delta = peft_lg     - hf_base_lg     # what LoRA added in PEFT
        diff_delta = (our_delta - peft_delta).abs().max().item()

        ok = diff_delta < ATOL   # primary check
```

**Why the delta comparison works:**

Both models apply the same LoRA weights with the same math. The base models differ slightly (as Test 0 shows), but the *change* induced by the adapter depends only on:
- The adapter weights (A, B — confirmed identical to PEFT at weight level)
- The input activations at each layer (slightly different between our model and HF's)

The second point causes the delta diff to be slightly larger than zero. Observed: `diff_delta ≈ 0.73–0.83` on MPS.

**Weight-level confirmation:**

A diagnostic check compared A and B tensors directly:
```
A max_diff = 0.00012204   (BF16 rounding: exact binary rounding of float32 → bfloat16)
B max_diff = 0.00000190   (essentially machine epsilon for bfloat16)
```

Both match to BF16 precision. The forward pass diffs are not from wrong weights — they are from BF16 arithmetic accumulation across 28 layers in two different implementations.

**Observed results on MPS (atol=1.0):**

```
'What is 2+2?'
  abs  diff=0.8438  ✓
  delta diff=0.8281  ✓  ← primary check

'Explain what a neural network is in one sentence.'
  abs  diff=0.4375  ✓
  delta diff=0.7812  ✓

'What is the capital of France?'
  abs  diff=0.6875  ✓
  delta diff=0.7266  ✓

→ PASS ✓
```

On CUDA, expected delta diffs are ~0.3–0.4 (tighter FP arithmetic, matching atol=0.75).

---

## Test 3 — Mixed-Batch Mask Separation (CUDA + FlashInfer)

**What it checks:** In a single EXTEND forward pass with two requests (one LoRA, one base), each request receives the correct output — LoRA request gets the adapter delta, base request does not.

```python
# One EXTEND pass with mixed mask:
# req0 (LoRA):  mask = [1.0, 1.0, ..., 1.0]
# req1 (base):  mask = [0.0, 0.0, ..., 0.0]
logits, qo_indptr = do_extend_lora(model, kv_pool, rtp, workspace, [req0, req1], cfg, adapter)

# Extract per-request last-token logits:
paged_lora = logits[0, qo_indptr[1] - 1, :]   # last token of req0
paged_base = logits[0, qo_indptr[2] - 1, :]   # last token of req1

# Compare against individual NOCACHE references:
diff_lora = (paged_lora - ref_lora_nocache).abs().max()   # should be small
diff_base = (paged_base - ref_base_nocache).abs().max()   # should be small
```

This test proves the mask correctly gates the delta — a bug where the mask is all-ones would make `paged_base` match `ref_lora` instead of `ref_base`.

**Requires:** CUDA + FlashInfer (paged attention is not available on MPS).

---

## Test 4 — Paged LoRA Prefill + Decode (CUDA + FlashInfer)

**What it checks:** The LoRA adapter works correctly through the full paged attention path, including across multiple decode steps.

Steps:
1. Prefill prompt with `use_lora=True` → compare last-token logits to NOCACHE+LoRA reference
2. Sample first output token
3. Decode N steps with `dec_mask = [[1.0]]` (always LoRA) → compare each step to NOCACHE+LoRA reference on the growing sequence

```python
# Decode step with LoRA:
dec_mask = torch.ones(1, 1, 1, dtype=DTYPE, device=DEVICE)  # [B=1, q_len=1, 1]
lg = our_model(
    cur_tok_t,
    attention_mask=None,
    kv_cache=ctx,         # DecodeKVCtx — reads from paged pool
    position_ids=pos_t,
    lora_mask=dec_mask,   # 1.0 → apply adapter
    lora_adapter=adapter,
)
```

The test confirms that LoRA applies consistently across prefill and decode modes — both paths flow through the same `Qwen3Attention.forward()` and `Qwen3MLP.forward()` code, so if the wiring is correct in the model, it works in all modes.

**Requires:** CUDA + FlashInfer.

---

## Running the Tests

### MPS (Mac with Apple Silicon)

```bash
# Tests 0, 1, 2 run. Tests 3, 4 skipped.
python verify_lora.py

# Expected output:
# Test 0: PASS ✓
# Test 1: PASS ✓
# Test 2: PASS ✓
# Test 3: SKIPPED (requires CUDA + flashinfer)
# Test 4: SKIPPED (requires CUDA + flashinfer)
# OVERALL: PASS ✓
```

### CUDA

```bash
# All 5 tests run.
python verify_lora.py --atol 0.75

# Expected output:
# Test 0: PASS ✓
# Test 1: PASS ✓
# Test 2: PASS ✓
# Test 3: PASS ✓
# Test 4: PASS ✓
# OVERALL: PASS ✓
```

### Troubleshooting

| Symptom | Likely cause |
|---|---|
| Test 0 FAIL with large diff (>2.0) | Model weights not loaded, wrong model path |
| Test 1 FAIL (diff < 1e-3) | `apply()` not called, wrong layer index, B matrix is zero |
| Test 2 FAIL on delta (diff > atol) | Wrong A/B transpose, wrong scaling, wrong module name mapping |
| Test 3 FAIL on base request | Mask has wrong shape — all-ones instead of per-token |
| Test 3 FAIL on lora request | Mask has wrong values — `lora_id` not passed to `prefill_batch` |
| Test 4 FAIL at decode | `lora_mask` shape for decode is wrong (should be `[B, 1, 1]` not `[1, B, 1]`) |

---

## Device-Flexible Model Loading

`from_pretrained()` in `model/qwen3.py` hardcodes `.to("cuda")` at the end. The verify script bypasses this with a custom loader:

```python
def load_our_model(model_path: str, device: str, dtype: torch.dtype):
    model_dir = _resolve_model_path(model_path)
    config    = Qwen3Config.from_json(model_dir / "config.json")
    model     = Qwen3ForCausalLM(config).to(dtype)
    # ... load weights ...
    return model.to(device).eval()   # device-flexible
```

Similarly, `load_hf_base_model` and `load_peft_model` both use `m.to(device)` rather than the CUDA-hardcoded path. This is what allows the script to run on MPS without modification.
