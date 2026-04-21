# Layer 18 — KV Cache Quantization Benchmarks

Two scripts that answer the same question from different angles:
**does FP8 KV cache actually help on your hardware, and by how much?**

---

## Scripts

| Script | What it does | Requires GPU? | Requires model download? |
|---|---|---|---|
| `bench_memory.py` | Pure math: token capacity, concurrency limits, bandwidth | No | No (just config.json) |
| `bench_server.py` | Live A/B test: SGLang server BF16 vs FP8 | Yes | Yes |

---

## Quick Start

### Step 1 — Memory analysis (run this first, no server needed)

```bash
cd CODE_LAYERS/layer18_KVCacheQuantization/benchmarks

# Qwen3-1.7B (this layer's model)
python bench_memory.py --model Qwen/Qwen3-1.7B --kv-fraction 0.75

# Your specific GPU setup
python bench_memory.py \
    --model Qwen/Qwen3-1.7B \
    --kv-fraction 0.75 \
    --context-lengths 512 1024 2048 4096 8192 16384
```

This tells you **before running anything**:
- How many KV tokens fit at BF16, FP8, FP4 on your GPU
- How many concurrent requests you can serve at each context length
- The memory bandwidth reduction at decode time
- How FP8 KV interacts with HiCache (Layer 17) PCIe offload

### Step 2 — Install aiohttp (needed for bench_server.py)

```bash
pip install aiohttp
```

### Step 3 — Live server benchmark

```bash
# Qwen3-1.7B: safe to test FP8 KV (has per-head QK RMSNorm stability)
python bench_server.py \
    --model Qwen/Qwen3-1.7B \
    --baseline-kv-dtype auto \
    --compare-kv-dtype fp8_e4m3 \
    --n-requests 24 \
    --concurrency 4 \
    --max-tokens 200 \
    --mem-fraction 0.75

# Stress the KV pool harder (where quantization pays off most)
python bench_server.py \
    --model Qwen/Qwen3-1.7B \
    --baseline-kv-dtype auto \
    --compare-kv-dtype fp8_e4m3 \
    --n-requests 48 \
    --concurrency 8 \
    --max-tokens 400 \
    --mem-fraction 0.75
```

Results are printed to the console and written to `results.md`.

---

## What to Expect on RTX 4060 Ti (16 GB, sm89)

### Memory analysis output (Qwen3-1.7B)

```
KV dtype        Bytes/tok    KV pool tokens   vs BF16
bf16 (baseline)      7,168        ~85,000      1.00×  PRODUCTION
fp8_e4m3             3,584        ~170,000     2.00×  PRODUCTION
int8_per_token_head  3,695        ~165,000     1.94×  PRODUCTION
nvfp4                1,792        ~340,000     4.00×  research (Blackwell only)
```

FP8 KV doubles the token capacity → doubles concurrent users at the same context length.

### Server benchmark — where you will and won't see speedup

| Scenario | Expected result | Reason |
|---|---|---|
| 4 concurrent, 200 tokens | ~equal throughput | Compute-bound; KV is not the bottleneck yet |
| 8 concurrent, 400 tokens | FP8 1.2–1.6× faster | KV bandwidth starts dominating decode time |
| 16 concurrent, 2048 tokens | FP8 1.5–2× faster or BF16 OOMs | KV pool pressure; FP8 fits 2× more in VRAM |
| Qwen2.5 + FP8 KV | Broken output | Outlier K heads saturate FP8 range at scale=1.0 |

---

## Model Compatibility with FP8 KV Cache

| Model | FP8 KV safe? | Notes |
|---|---|---|
| `Qwen/Qwen3-0.6B` | ✓ Yes | Per-head QK RMSNorm stabilises K outliers |
| `Qwen/Qwen3-1.7B` | ✓ Yes | Same |
| `Qwen/Qwen2.5-7B-Instruct` | ✗ No | Two outlier K heads; repeating output without calibration |
| `Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4` | ✗ No | Same Qwen2.5 architecture issue |
| `neuralmagic/Qwen2.5-7B-Instruct-FP8-dynamic` | ✓ (weights only) | FP8 weights, BF16 KV (use `auto`) |
| `meta-llama/Meta-Llama-3.1-8B-Instruct-FP8` | ✓ Yes | Embedded calibrated KV scales; use `--compare-kv-dtype auto` |

---

## The Core Concept: When Does FP8 KV Help?

FP8 KV cache reduces memory bandwidth, not compute. The GPU must load the entire KV
history from HBM on every decode step:

```
Decode bandwidth per step = num_layers × 2 × num_kv_heads × head_dim × dtype_bytes × num_active_tokens
```

For Qwen3-1.7B (28 layers, 8 KV heads, 128 head_dim) at 1024 active tokens:
- BF16: 28 × 2 × 8 × 128 × 2 × 1024 = 117 MB loaded per decode step
- FP8:  same × 1 byte              =  59 MB loaded per decode step → 2× less bandwidth

At low batch sizes, decode is compute-bound (tensor cores busy). FP8 KV only helps when
the GPU is memory-bandwidth-bound — i.e., when the KV pool is large relative to compute.
The crossover point on RTX 4060 Ti is roughly at **concurrency ≥ 6 and context ≥ 1024 tokens**.

---

## Flags Reference

### `bench_memory.py`

```
--model             HuggingFace model ID (default: Qwen/Qwen3-1.7B)
--kv-fraction       Fraction of free VRAM for KV pool (default: 0.75)
--page-size         KV page size in tokens (default: 16)
--context-lengths   Space-separated list of context lengths to tabulate
--num-layers        Manual override (skip model download)
--num-kv-heads      Manual override
--head-dim          Manual override
--vram-gb           Manual GPU VRAM total in GB
--weight-gb         Manual model weight size in GB
```

### `bench_server.py`

```
--model             HuggingFace model ID (default: Qwen/Qwen3-1.7B)
--quantization      Weight quant method (e.g. gptq_marlin)
--baseline-kv-dtype KV dtype for run 1 (default: auto → BF16)
--compare-kv-dtype  KV dtype for run 2 (default: fp8_e4m3)
--port              SGLang server port (default: 30080)
--mem-fraction      --mem-fraction-static (default: 0.75)
--startup-timeout   Seconds to wait for server ready (default: 120)
--n-requests        Total requests per run (default: 24)
--concurrency       Concurrent requests in flight (default: 4)
--max-tokens        Max output tokens per request (default: 200)
--prompt-kind       short / long / mixed (default: mixed)
--output            Markdown output file (default: results.md)
--no-compare        Only run baseline (skip comparison)
```

---

## Files

```
benchmarks/
  README.md          This file
  bench_memory.py    KV capacity math — no server, no weights
  bench_server.py    SGLang server A/B benchmark
  utils.py           Shared prompts, BenchResult dataclass, table formatting
  results.md         Generated after bench_server.py completes
  server_auto.log    SGLang server log for baseline run
  server_fp8_e4m3.log  SGLang server log for comparison run
```
