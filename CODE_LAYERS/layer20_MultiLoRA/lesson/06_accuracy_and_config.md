# 06 — Accuracy, Configuration, and Practical Tradeoffs

## What This Section Covers

The previous five sections built the understanding: why KV quantization matters, how FP8 is encoded, how SGLang and vLLM implement it, and what research has shown is possible. This final section is the practical guide: how to configure KV quantization for real deployments, how to generate calibrated scales, how to measure accuracy, and how it interacts with HiCache and weight quantization.

---

## SGLang Configuration: The Two Flags

KV cache quantization in SGLang is controlled by exactly two CLI flags:

| Flag | Default | Type | Effect |
|---|---|---|---|
| `--kv-cache-dtype` | `"auto"` | string | Sets the KV storage dtype. Options: `auto`, `fp8_e4m3`, `fp8_e5m2`, `bf16`, `fp4_e2m1` |
| `--quantization-param-path` | `None` | string | Path to JSON file with per-layer `k_scale` and `v_scale` values |

### Mode 1: Auto-detect from FP8 checkpoint (recommended)

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 \
  --kv-cache-dtype auto \
  --tp 8
```

Works when using an FP8 checkpoint from HuggingFace that includes:
1. `config.json` field `"kv_cache_quant_algo": "FP8"` (tells SGLang to activate FP8 KV)
2. A `kv_cache_scales.json` in the checkpoint directory (provides per-layer scales)

SGLang's `configure_kv_cache_dtype()` reads (1) and `load_kv_cache_scales()` reads (2) automatically. No manual flags needed beyond `--kv-cache-dtype auto`.

Common checkpoints that work with this mode:
- `meta-llama/Meta-Llama-3.1-8B-Instruct-FP8`
- `meta-llama/Meta-Llama-3.1-70B-Instruct-FP8`
- `neuralmagic/Meta-Llama-3.1-8B-Instruct-FP8`
- `Qwen/Qwen2.5-72B-Instruct-FP8`
- `mistralai/Mistral-7B-Instruct-v0.3-FP8`

### Mode 2: FP8 KV with explicit calibrated scales

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --kv-cache-dtype fp8_e4m3 \
  --quantization-param-path /mnt/scales/llama70b_kv_scales.json \
  --tp 8
```

Use when you have a BF16 checkpoint and want FP8 KV. The scales file must be generated separately (see calibration section below).

### Mode 3: FP8 KV without calibration (convenience, accuracy risk)

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct \
  --kv-cache-dtype fp8_e4m3 \
  --tp 8
```

SGLang logs a warning and uses scale=1.0 for all layers. This works correctly only if the model's KV activations happen to fall within ±448 — which is not guaranteed. For most Llama and Mistral models at typical input distributions, this is approximately safe. For models with larger activation ranges (some code-focused or long-context fine-tunes), saturation can cause visible quality degradation.

### Mode 4: FP8 KV + HiCache together

```bash
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 \
  --kv-cache-dtype auto \
  --enable-hierarchical-cache \
  --hicache-ratio 2.0 \
  --hicache-storage-backend file \
  --hicache-storage-backend-extra-config '{"storage_dir": "/mnt/nvme/kvcache"}' \
  --tp 8
```

FP8 KV and HiCache are fully composable. The FP8 tensors are written to the GPU pool → evicted to the CPU pinned buffer → written to NVMe, all in FP8. PCIe transfers move 2× fewer bytes per token than BF16.

---

## Generating Calibrated Scales with llm-compressor

When you do not have an FP8 checkpoint with pre-included scales, generate them offline using [llm-compressor](https://github.com/vllm-project/llm-compressor) (formerly neural-magic/llm-compressor):

```python
from llmcompressor.modifiers.quantization import QuantizationModifier
from llmcompressor.transformers import SparseAutoModelForCausalLM, oneshot

# Load the base BF16 model
model = SparseAutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3.1-70B-Instruct",
    device_map="auto",
    torch_dtype="bfloat16",
)

# Define quantization recipe
recipe = QuantizationModifier(
    targets="Linear",
    scheme="FP8",
    ignore=["lm_head"],          # don't quantize output projection
    kv_cache_scheme={
        "type": "float8",
        "strategy": "tensor",    # per-tensor scale (one per layer)
        # "strategy": "token"    # per-token scale (finer, higher overhead)
    }
)

# Run calibration on a sample dataset (512–2048 samples is sufficient)
from datasets import load_dataset
ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:512]")

oneshot(
    model=model,
    recipe=recipe,
    dataset=ds,
    num_calibration_samples=512,
    max_seq_length=2048,
    output_dir="./llama70b-fp8-kv",  # saves config.json + kv_cache_scales.json
)
```

The output directory contains:
- `config.json` with `"kv_cache_quant_algo": "FP8"` added
- `kv_cache_scales.json` with per-layer `k_scale`/`v_scale` floats
- All original model weight files (calibration does not change weights — only scales are new)

**Calibration tips:**
- Use 512–2048 samples from a representative dataset — samples from your actual production prompt distribution are ideal
- A diverse mix (instructions, code, multi-turn chat) is better than domain-specific samples if your deployment serves diverse inputs
- Calibration takes 10–30 minutes for a 70B model with 8×H100 (just one forward pass per sample, no gradients)
- The resulting scales are static — if your input distribution shifts significantly, recalibrate

---

## Accuracy Characterization

### Benchmarks with calibrated FP8 scales

For models with calibrated per-layer scales (from llm-compressor or included in the checkpoint):

| Model | Benchmark | BF16 | FP8 KV (calibrated) | Degradation |
|---|---|---|---|---|
| Llama-3.1-8B-Instruct | MMLU | 68.1 | 67.8 | 0.4% |
| Llama-3.1-70B-Instruct | MMLU | 82.6 | 82.1 | 0.6% |
| Llama-3.1-8B-Instruct | HumanEval | 72.6 | 72.2 | 0.5% |
| Mistral-7B-v0.3 | LongBench | 45.3 | 44.9 | 0.9% |

(Figures approximate; consult model cards for exact numbers.)

### Benchmarks without calibration (scale=1.0)

Without calibration, results depend heavily on the model's activation range:
- Models with typical activation ranges (most instruction-tuned Llama, Qwen, Mistral): 0.5–2% degradation — usually acceptable
- Models with larger ranges (some math/code fine-tunes, MoE models): 2–5% degradation — not recommended without calibration

### Long-context accuracy

Long-context workloads are slightly more sensitive to KV quantization because errors accumulate over many tokens. For 32K+ context:
- FP8 per-tensor with calibrated scales: typically < 1% degradation on LongBench, RULER, and SCROLLS
- FP8 without calibration: 1–3% degradation; more visible in tasks requiring precise cross-document retrieval

---

## Workload Decision Guide

```
Is your workload accuracy-critical (medical, legal, financial)?
  → BF16 KV (no quantization). Use HiCache (Layer 17) for capacity.
  → OR: deploy FP8 with calibrated scales and run regular eval checks

Do you have a calibrated FP8 checkpoint already?
  → Yes: --kv-cache-dtype auto. Done.
  → No: calibrate with llm-compressor (30 min), then --kv-cache-dtype fp8_e4m3 --quantization-param-path

Is your input distribution diverse (multi-language, code + text)?
  → Consider vLLM with fp8_per_token_head: dynamic scales adapt to any distribution, no calibration

Are you serving long contexts (32K+) and VRAM-constrained?
  → FP8 KV + HiCache: 2× compression per tier, composable
  → If still not enough: explore KIVI or KVQuant (research, requires custom kernels)

Is decode throughput the primary metric?
  → FP8 KV: memory-bandwidth-bound decode runs faster with 2× smaller KV
  → Stack with SageAttention2 for additional compute acceleration (not in serving stacks yet)

Is this a development/testing environment?
  → --kv-cache-dtype fp8_e4m3 (no --quantization-param-path): convenient, usually fine
  → Watch for warning "Using default scale factor 1.0" in SGLang logs
```

---

## Interaction with Weight Quantization

KV cache quantization and model weight quantization are independent and composable. The table below characterizes common combinations:

| Weight dtype | KV dtype | VRAM usage | Throughput | Accuracy risk | Recommended for |
|---|---|---|---|---|---|
| BF16 | BF16 | 100% (baseline) | 1× | None | Development, accuracy-critical |
| BF16 | FP8 | ~60% | 1.5–2× decode | Low (with calibration) | Production baseline |
| FP8 weights | FP8 KV | ~30% | 2–3× | Low | Maximum throughput at full quality |
| INT4 GPTQ | FP8 KV | ~20% | 2–4× | Moderate | Research, long-context with budget GPU |
| INT4 GPTQ | BF16 | ~40% | 1.5× | Low | VRAM-constrained, accuracy priority |

"FP8 weights" here refers to model weights stored in FP8 (e.g., `--quantization fp8` in SGLang), which is separate from `--kv-cache-dtype fp8_e4m3`. These are different flags for different tensors:
- `--quantization fp8` → quantizes the linear layer weight matrices
- `--kv-cache-dtype fp8_e4m3` → quantizes the K/V activations stored in the paged KV pool

---

## Monitoring FP8 KV in Production

KV quantization itself does not expose dedicated Prometheus metrics (unlike HiCache which has `sglang:hicache_host_used_tokens` etc.). You monitor its effects through standard metrics:

```promql
# VRAM utilization — should drop ~50% vs BF16 KV (check after enabling)
sglang:kv_available_tokens / (sglang:kv_available_tokens + sglang:kv_used_tokens)

# Throughput improvement — decode-bound workloads should show improvement
rate(sglang:num_tokens_generated_total[1m])

# Cache hit rate — unaffected by quantization dtype
sglang:cache_hit_rate

# Batch size — should increase as VRAM frees up
sglang:running_requests
```

**Accuracy monitoring:** there is no built-in accuracy metric in Prometheus. For production deployments with FP8 KV:
1. Run your accuracy benchmark suite (MMLU, HumanEval, domain-specific evals) before and after enabling FP8 KV
2. Set up periodic inference quality probes (fixed prompts with expected outputs, checked for semantic similarity)
3. Watch for user-reported quality regressions in the first 24–48 hours after enabling

---

## Common Issues and Fixes

### Issue: "Using default scale factor 1.0" warning

**Cause:** `--kv-cache-dtype fp8_e4m3` set but no `--quantization-param-path` provided.
**Fix:** Add `--quantization-param-path /path/to/scales.json` or switch to a checkpoint with included scales (`--kv-cache-dtype auto`).

### Issue: Accuracy degradation after enabling FP8 KV

**Cause:** Scale saturation — model's KV activations exceed ±448 with the current scale.
**Fix:**
1. Check if you have calibrated scales; if not, generate them with llm-compressor
2. Switch to `fp8_e5m2` (wider range, ±57344) at the cost of precision
3. Switch back to BF16 KV and use HiCache instead for capacity

### Issue: OOM despite FP8 KV enabled

**Cause:** Context length or batch size still exceeds FP8-reduced VRAM budget.
**Fix:** Add HiCache (`--enable-hierarchical-cache`) to spill excess pages to CPU RAM.

### Issue: vLLM sleep-mode accuracy regression

**Cause:** `init_fp8_kv_scales()` resets calibrated scales to 1.0 after GPU wakeup (vLLM bug, `gpu_model_runner.py:885`).
**Fix:** Disable sleep mode when using calibrated FP8 KV (`--disable-sleep-mode` in vLLM) until the bug is fixed upstream.

### Issue: FP4 (`fp4_e2m1`) is causing model errors

**Cause:** FP4 is experimental in SGLang; model may not have calibrated FP4 scales.
**Fix:** Use `fp8_e4m3` instead. FP4 is not recommended for production.

---

## What Layer 18 Explicitly Defers

- **Pre-RoPE KV quantization in production**: KVQuant's technique requires splitting `apply_rotary_pos_emb()` from the cache write — an architectural change not yet in SGLang or vLLM mainline
- **Sub-FP8 production kernels** (KIVI INT2, KVQuant INT3): require custom CUDA dequant kernels not yet merged into FlashInfer or CUTLASS
- **Training-aware KV quantization (QAT)**: all methods here are post-training; QAT for KV cache is an active research area
- **Blackwell NVFP4 in production**: vLLM has partial support (WIP); not yet in SGLang
- **SageAttention2 integration**: the library is standalone; production integration into FlashInfer or CUTLASS for SGLang/vLLM attention backends is an open item
- **MoE-specific KV quantization**: mixture-of-experts models have different expert routing patterns that affect KV cache layout and quantization statistics — not covered here

---

## Summary

- Two SGLang flags cover all of KV cache quantization: `--kv-cache-dtype` and `--quantization-param-path`
- Recommended path: use `--kv-cache-dtype auto` with an FP8 checkpoint that includes scales; calibrate with llm-compressor if using a BF16 base checkpoint
- Accuracy impact with calibrated scales: < 1% on standard benchmarks for Llama, Qwen, Mistral; monitor long-context retrieval tasks most carefully
- FP8 KV and HiCache are orthogonal and composable: enable both for maximum VRAM efficiency
- FP8 KV and weight quantization are independent: FP8 weights + FP8 KV is the recommended stack for maximum throughput
- Monitor via standard SGLang metrics: VRAM utilization, throughput, and batch size improvement; accuracy monitoring requires separate eval pipelines
- vLLM's per-token-head dynamic mode (`fp8_per_token_head`) trades ~3% scale overhead for no calibration requirement — useful for diverse input distributions

---

## Relationship to Adjacent Layers

| Layer | Topic | How Layer 18 connects |
|---|---|---|
| **Layer 12** | `RadixCache`, `MHATokenToKVPool` | Layer 18 quantizes values written into the same pool structure |
| **Layer 17** | HiCache tiered storage | FP8 KV halves bytes-per-token at every tier; PCIe transfers in HiCache tier-2 loads are 2× cheaper |
| **Layer 19** | PD disaggregation (Mooncake) | FP8 KV tensors transferred via RDMA during disaggregation are 2× smaller → lower transfer latency |
