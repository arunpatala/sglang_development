# Layer 20 — Lesson Outline: Single-Adapter LoRA Serving

## What This Lesson Covers

Layer 19 introduced PD disaggregation: routing prefill and decode to different GPU pools for better resource utilisation. Layer 20 asks a different question — what if different requests need different *model behaviours*? Not different hardware, but different **fine-tuned variants** of the same base model?

LoRA (Low-Rank Adaptation) fine-tunes a model by injecting small rank-decomposed weight matrices (`A` and `B`) into each linear projection. At inference time, the base weights stay frozen; the adapter adds a learned delta:

```
output = x @ W.T  +  (x @ A.T) @ B.T * scaling
```

At rank=8 with `alpha=32`, one adapter for Qwen3-0.6B adds ~3 MB — trivial compared to the ~1.2 GB base model. The key inference challenge is serving multiple adapters in a **single batched forward pass** where different requests use different adapters.

This layer implements the minimal case: **one static adapter loaded at startup**, where each request independently selects the base model or the adapter. The core mechanism is a per-token float mask that gates the LoRA delta:

```
output = base_output + delta * lora_mask   (mask = 1.0 or 0.0 per token)
```

This allows mixed batches — some requests using the adapter, others using the base model — in a single forward pass with no routing complexity. The full multi-LoRA system (pool management, LRU eviction, segmented GEMM kernels) is documented in `sglang_multi_lora_implementation.md` as a reference; this layer focuses on understanding and implementing the correctness-critical path.

---

## Sections

### 01 — LoRA Math and Why Rank Decomposition Works (`01_lora_math.md`)

- LoRA represents the weight delta as `ΔW = B @ A` where `A ∈ ℝ^{r×d}` and `B ∈ ℝ^{d_out×r}` with rank `r ≪ min(d, d_out)`
- For Qwen3-0.6B with `r=8`: `q_proj` delta is `[8, 1024]` (A) + `[2048, 8]` (B) = 24,576 parameters vs `[2048, 1024]` = 2,097,152 base parameters — **0.6% overhead per projection**
- Scaling: `lora_alpha / r` (not 1/r) gives a hyperparameter to control delta magnitude without changing rank
- Why rank works: fine-tuning updates occupy a low-dimensional subspace of the full weight space; LoRA exploits this by directly parameterising that subspace
- The `phh/Qwen3-0.6B-TLDR-Lora` adapter used in this layer: `r=8`, `lora_alpha=32`, `scaling=4.0`, targets `q_proj` and `v_proj` only

### 02 — The Mask-Based Mixed-Batch Strategy (`02_mask_strategy.md`)

- **The problem**: a batch of N requests may use different adapters (or no adapter). Launching one forward pass per adapter is linear cost O(N_adapters); we want O(1).
- **The mask solution**: one scalar `lora_mask[i] ∈ {0.0, 1.0}` per request token. The LoRA delta is computed for every token, then multiplied by the mask before adding to the base output.
- Shape conventions:
  - EXTEND (packed prefill): `lora_mask` is `[1, total_tokens, 1]`
  - DECODE (one token per request): `lora_mask` is `[B, 1, 1]`
  - The trailing `1` broadcasts over the output dimension
- Why compute delta for base-model tokens and zero it out: the wasted FLOPs are proportional to `n_base_tokens / n_total_tokens`. For a batch where half use LoRA, we waste 50% of the delta compute. This is the efficiency cost of the minimal implementation — the production segmented GEMM (Punica) eliminates it.
- Zero overhead when no LoRA requests: `lora_adapter=None` is checked once per layer; the entire delta path is skipped with a single `if` check.

### 03 — `lora.py`: Loading the Adapter (`03_lora_adapter.md`)

- `LoRAAdapter` reads `adapter_config.json` to get `r`, `lora_alpha`, `target_modules`
- Streams A and B tensors from `adapter_model.safetensors` (or `.bin`)
- Parses HuggingFace PEFT key format: `base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight`
- Stores in nested dicts: `A_weights[layer_idx][module_name]`, `B_weights[layer_idx][module_name]`
- `apply(x, layer_idx, module_name)` returns `(x @ A.T) @ B.T * scaling` or `None` if that layer/module is not targeted — callers check `None` before adding
- Supports both `.safetensors` and `.bin` checkpoints; resolves HF Hub IDs via `snapshot_download`

### 04 — Wiring LoRA into the Forward Pass (`04_wiring.md`)

Five files are changed to thread `lora_mask` and `lora_adapter` through the model stack:

1. **`forward_batch.py`** — two new optional fields: `lora_mask` and `lora_adapter`
2. **`model/qwen3.py`** — `Qwen3Model.forward()` and `Qwen3ForCausalLM.forward()` accept `lora_mask`/`lora_adapter` kwargs; all three `ForwardBatch` construction sites populate them
3. **`model/attention.py`** — after `q_proj`/`k_proj`/`v_proj` projections and after `o_proj`, adds `delta * mask` when adapter is active
4. **`model/mlp.py`** — accepts `forward_batch` argument; adds `delta * mask` for `gate_proj`, `up_proj`, `down_proj`; stores `layer_idx` for `apply()` lookups
5. **`model/decoder_layer.py`** — passes `layer_idx` to `Qwen3MLP.__init__`; passes `forward_batch` to `self.mlp()`

The LoRA delta is added **after** the base linear projection and **before** any downstream operation (e.g., before the `view()` reshape in attention, before `F.silu()` in MLP). This matches PEFT's implementation exactly.

### 05 — Server and Config Integration (`05_server_config.md`)

- `config.yml`: single `lora_path` field (replaces the multi-adapter pool config); `null` disables LoRA entirely
- `server.py`: `ChatRequest` gains `lora_id: Optional[str]`; any non-null string activates the adapter for that request; `Req.lora_id` carries this flag through the scheduler
- `model_runner.py`: `__init__` accepts `lora_path`; loads `LoRAAdapter` at startup and stores on `self.lora_adapter`; `prefill_batch` and `decode_step` each build `lora_mask` from `req.lora_id` before the `model.forward()` call

Example API usage:
```bash
# Base model (no LoRA)
curl -X POST http://localhost:8114/v1/chat/completions \
  -d '{"messages": [{"role": "user", "content": "What is 2+2?"}]}'

# LoRA adapter
curl -X POST http://localhost:8114/v1/chat/completions \
  -d '{"messages": [...], "lora_id": "tldr"}'
```

### 06 — Verification: `verify_lora.py` (`06_verification.md`)

Four correctness tests, ordered from prerequisite to full coverage:

| Test | What it checks | Requires |
|---|---|---|
| **Test 0** Base model match | Our custom model vs HuggingFace `AutoModelForCausalLM` (NOCACHE path) | torch + transformers |
| **Test 1** Adapter changes output | LoRA delta ≠ 0 — adapter is actually applied | torch + our lora.py |
| **Test 2** Matches PEFT | Our LoRA delta matches `PeftModel` delta (external ground truth) | torch + peft |
| **Test 3** Mixed-batch separation | In a `[lora, base]` batch, each token gets the correct output | CUDA + flashinfer |
| **Test 4** Paged LoRA | EXTEND+DECODE with LoRA mask matches NOCACHE reference | CUDA + flashinfer |

**Test 2 uses the delta comparison** `(our_lora - our_base) vs (peft_lora - hf_base)` to isolate LoRA math correctness from base model numerical differences. This is more meaningful than comparing absolute logits.

**Verified on MPS (Mac, base conda env)**:
- Test 0: PASS (max_diff ≤ 0.69 with atol=1.0)
- Test 1: PASS (LoRA delta = 3.375 — clearly non-zero)
- Test 2: PASS (delta diff ≤ 0.83, confirmed by weight comparison: A max_diff=0.00012, B max_diff=0.0000019 — pure BF16 rounding)
- Tests 3–4: SKIPPED (no flashinfer on MPS), will run on CUDA

---

## Supporting Files

- `summary.md` — narrative walkthrough of all six sections with worked examples and test results
- `01_lora_math.md` — LoRA rank decomposition, scaling, FLOP numbers, weight merging vs. residual add
- `02_mask_strategy.md` — float mask shape conventions, how it is built in model_runner, efficiency analysis
- `03_lora_adapter.md` — PEFT checkpoint format, `_load_weights()` key parsing, `apply()` GEMM shapes
- `04_wiring.md` — all five changed files with exact code and ordering rationale
- `05_server_config.md` — config.yml, ChatRequest schema, startup sequence, API examples
- `06_verification.md` — all four tests, ground-truth comparison strategy, results, troubleshooting table
- `sglang_multi_lora_implementation.md` — full reference for production multi-LoRA: pool management, segmented GEMM kernels, LRU eviction, CUDA graph support, dynamic load/unload

## New Implementation Files

| File | Role |
|---|---|
| `lora.py` | `LoRAAdapter`: weight loading, `apply()` method |
| `forward_batch.py` | +`lora_mask`, `lora_adapter` fields |
| `model/attention.py` | +LoRA deltas for q/k/v/o projections |
| `model/mlp.py` | +LoRA deltas for gate/up/down; +`layer_idx`, `forward_batch` arg |
| `model/decoder_layer.py` | +`layer_idx` to MLP init; +`forward_batch` to `mlp()` call |
| `model/qwen3.py` | +`lora_mask`/`lora_adapter` kwargs in both forward() methods |
| `model_runner.py` | +`lora_path` arg; `LoRAAdapter` loaded at startup; mask built per batch |
| `server.py` | +`lora_id` in `ChatRequest`; passed to `Req` |
| `config.yml` | Simplified to `lora_path: "phh/Qwen3-0.6B-TLDR-Lora"` |
| `verify_lora.py` | 4-test correctness suite with PEFT ground-truth comparison |

---

## Key Code Anchors (this layer)

| Concept | File | Symbol / Line |
|---|---|---|
| LoRA math: `apply()` | `lora.py` | `LoRAAdapter.apply()` |
| Weight loading from PEFT format | `lora.py` | `LoRAAdapter._load_weights()` |
| `lora_mask` + `lora_adapter` on ForwardBatch | `forward_batch.py` | `ForwardBatch` dataclass |
| QKV LoRA deltas | `model/attention.py` | `Qwen3Attention.forward()` — step 1b |
| O-proj LoRA delta | `model/attention.py` | `Qwen3Attention.forward()` — step 5b |
| MLP LoRA deltas | `model/mlp.py` | `Qwen3MLP.forward()` |
| ForwardBatch populated with LoRA state | `model/qwen3.py` | `Qwen3Model.forward()` |
| Adapter loaded at startup | `model_runner.py` | `ModelRunner.__init__()` |
| `lora_mask` built for prefill | `model_runner.py` | `ModelRunner.prefill_batch()` — Step 7 |
| `lora_mask` built for decode | `model_runner.py` | `ModelRunner.decode_step()` |
| `lora_id` in API request | `server.py` | `ChatRequest.lora_id` |
| PEFT ground-truth comparison | `verify_lora.py` | `test_lora_matches_peft()` |
| Mixed-batch separation test | `verify_lora.py` | `test_mixed_batch_separation()` |
