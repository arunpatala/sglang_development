# Multi-LoRA Inference — From Fine-Tuning Economics to Production Serving

**Level:** L1 + L2 — Concept, economics, architecture, and deployment. No paper-level math; no kernel internals.

**What this file is:** A single coherent article synthesising all L1 and L2 source material into a progressive narrative. Sections build from "why does serving many fine-tuned models cost so much?" to "how do I launch LoRA serving in SGLang and vLLM today?" Sections above L2 (Punica SGMV kernel CUDA implementation, S-LoRA Unified Paging formal analysis, dLoRA merge/unmerge credit algorithm, CaraServe CPU-assisted prefill design, Loquetier virtualized module internals, InfiniLoRA disaggregated LoRA server) are deliberately left out.

**Sources synthesised:**
- L1/01 — Sai Mudhiganti: LoRAX: Serve 1000 Fine-Tuned Models on One GPU (July 2025)
- L1/02 — João Moura (AWS): LoRA Serving on Amazon SageMaker (January 2024)
- L1/03 — Neel Shah (Towards AI): The Architectural Paradigm of Multi-Adapter Inference (March 2026)
- L2/01 — SGLang: LoRA Serving Official Documentation (April 2026)
- L2/02 — vLLM: LoRA Adapters Official Documentation (April 2026)
- L2/03 — HuggingFace PEFT: LoRA Developer Guide (April 2026)

**Omitted (above L2):** Punica SGMV kernel CUDA implementation, S-LoRA Unified Paging formal analysis, dLoRA credit-based merge/unmerge algorithm, CaraServe CPU-assisted prefill, Loquetier virtualised module design, ServerlessLoRA backbone sharing internals, Predictive-LoRA LSTM predictor training, InfiniLoRA disaggregated LoRA server architecture, DoRA and rsLoRA variant mathematics, EVA/PiSSA/OLoRA initialisation strategies.

---

## Section Plan

| § | Title | Sources | Reading time |
|---|-------|---------|------|
| 1 | [The Fine-Tuning Explosion and the Serving Crisis](#1-the-fine-tuning-explosion-and-the-serving-crisis) | L1/01, L1/02, L1/03 | 3 min |
| 2 | [LoRA in a Nutshell: How Adapters Stay Small](#2-lora-in-a-nutshell-how-adapters-stay-small) | L2/03, L1/02, L1/03 | 4 min |
| 3 | [The Serving Problem: One Base Model, One Thousand Adapters](#3-the-serving-problem-one-base-model-one-thousand-adapters) | L1/02, L1/03 | 4 min |
| 4 | [The SGMV Kernel: Batching Across Different Adapters](#4-the-sgmv-kernel-batching-across-different-adapters) | L1/02, L1/03 | 4 min |
| 5 | [Three Pillars of Production Multi-Adapter Serving](#5-three-pillars-of-production-multi-adapter-serving) | L1/03 | 5 min |
| 6 | [The Live Benchmark: 50 Adapters at the Cost of One](#6-the-live-benchmark-50-adapters-at-the-cost-of-one) | L1/02 | 3 min |
| 7 | [Serving LoRA Adapters in SGLang](#7-serving-lora-adapters-in-sglang) | L2/01 | 5 min |
| 8 | [Serving LoRA Adapters in vLLM](#8-serving-lora-adapters-in-vllm) | L2/02 | 4 min |
| 9 | [The HuggingFace PEFT Checkpoint Format](#9-the-huggingface-peft-checkpoint-format) | L2/03 | 4 min |
| 10 | [Advanced Production Features](#10-advanced-production-features) | L1/03, L2/02 | 3 min |
| 11 | [The KV Cache vs Adapter VRAM Trade-off](#11-the-kv-cache-vs-adapter-vram-trade-off) | L1/02, L1/03 | 3 min |
| 12 | [Decision Framework: When Does Multi-LoRA Serving Pay Off?](#12-decision-framework-when-does-multi-lora-serving-pay-off) | L1/01, L1/02, L1/03 | 3 min |

**Total reading time:** ~45 minutes

---

## 1. The Fine-Tuning Explosion and the Serving Crisis

Fine-tuning large language models has never been easier. Datasets are accessible, tooling is mature, and the PEFT (Parameter-Efficient Fine-Tuning) ecosystem has made the process inexpensive. A motivated ML team can produce dozens of task-specific fine-tuned models in a single sprint.

The problem arrives the moment those models go to production.

A 7-billion-parameter model in half-precision (FP16) occupies approximately **14 gigabytes of GPU VRAM** just for its static weights — before inference, before KV cache, before anything. An enterprise SaaS provider offering custom-tuned models for each of 1,000 customers would, in the traditional deployment model, require 1,000 GPU instances. At $2–$5 per H100 hour, that is millions of dollars monthly in GPU infrastructure — with most instances sitting idle at any given time because most customers are not querying at the same moment.

The industry's response to the training cost was Parameter-Efficient Fine-Tuning. LoRA and its variants reduced the trainable parameter count to under 1% of the model. A full LLaMA-3.1-8B fine-tune produces 16 GB of weights; a LoRA adapter fine-tune produces 50–200 MB of adapter weights. The 8B parameters of the base model are never touched.

But LoRA solved the training cost without solving the serving cost. Until recently, most inference frameworks required adapters to be **merged** into the base model before serving — effectively recreating the full-weight model for each adapter. You gained on training storage and training compute, and then spent it all back on inference infrastructure.

> "Fine-tuning large language models is easier than ever. Serving them efficiently? That's where LoRAX steps in." — Sai Mudhiganti, 2025

This is the serving crisis: not a shortage of fine-tuned models, but the inability to serve many of them cheaply from shared hardware.

---

## 2. LoRA in a Nutshell: How Adapters Stay Small

To understand multi-adapter serving, you need to understand what an adapter actually is and why it can be so small.

### The Rank Decomposition Idea

For any linear layer with weight matrix `W₀ ∈ ℝ^{d×k}`, fine-tuning typically learns a small correction `ΔW`. The LoRA hypothesis is that this correction has low intrinsic dimensionality — it can be well-approximated by a pair of much smaller matrices:

```
ΔW ≈ B · A

where A ∈ ℝ^{r×k}  (r ≪ k)
      B ∈ ℝ^{d×r}  (r ≪ d)
      r is the "rank", typically 4–64
```

Instead of storing and updating all `d × k` parameters of `ΔW`, LoRA stores and trains only the `r × k` parameters of A and the `d × r` parameters of B. For a typical 4096×4096 weight matrix with rank 8, that is 8,192 + 32,768 = 40,960 parameters instead of 16,777,216 — a 410× reduction.

### The Forward Pass

During inference, the LoRA contribution is computed as:

```
h = W₀ · x  +  B · A · x · (α / r)
```

where:
- `W₀`: frozen base model weights (never changed, never loaded twice regardless of how many adapters you have)
- `A`: the down-projection matrix (maps input to rank-r space)
- `B`: the up-projection matrix (maps rank-r space back to output dimension)
- `α / r`: a scaling factor (analogous to a learning rate; `α` is a hyperparameter, often set to `r` or `2r`)

In PyTorch notation (using transposed convention):

```python
h = x @ W0.T  +  (x @ A.T) @ B.T * (lora_alpha / r)
```

This is exactly what `LoRAAdapter.apply()` in Layer 20 computes.

### Weight Initialisation

The initialisation strategy is deliberate:
- **A is initialised with Kaiming-uniform (random)** — ensures gradients flow from the first training step
- **B is initialised to zeros** — so `B·A = 0` at initialisation, meaning the model starts training from the exact pre-trained behaviour with no initial perturbation

This means fine-tuning with LoRA begins from the same point as fine-tuning without it — you aren't "restarting" the model, you're adding a trainable correction layer on top of frozen weights.

### What's In an Adapter Checkpoint

An adapter checkpoint from HuggingFace PEFT consists of two files:

```
adapter_config.json          ← LoRA configuration
adapter_model.safetensors    ← A and B weight tensors
```

The `adapter_config.json` contains:
```json
{
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  "r": 8,
  "lora_alpha": 32,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                     "gate_proj", "up_proj", "down_proj"],
  "bias": "none"
}
```

The weight keys in `adapter_model.safetensors` follow the pattern:
```
base_model.model.model.layers.{layer_idx}.{module}.lora_A.weight  →  A matrix [r, in_dim]
base_model.model.model.layers.{layer_idx}.{module}.lora_B.weight  →  B matrix [out_dim, r]
```

This is the format that `LoRAAdapter._load_weights()` in Layer 20 parses. PEFT's key naming convention is the standard that all adapter checkpoints on HuggingFace Hub follow.

---

## 3. The Serving Problem: One Base Model, One Thousand Adapters

With the LoRA math understood, the serving problem becomes precise. You have one base model (14 GB) and N adapters (50–200 MB each). You want to serve requests, where each request specifies which adapter (or no adapter — just the base model) to use. How do you handle this at scale?

### Approach 1: Merge Weights (Naïve)

The simplest approach: before serving, merge each adapter into its own copy of the base model.

```
W = W₀ + B · A · (α/r)
```

Then serve each merged model independently. This works correctly — the forward pass becomes a single `x @ W.T` — but it completely undoes the storage savings of LoRA. You are back to 14 GB per adapter. For 100 adapters, that is 1.4 TB of VRAM across 100 GPU instances.

Furthermore, merged serving limits batch sizes: requests for different adapters cannot be combined into the same batch, because each GPU instance now holds a different model.

### Approach 2: Separate Computation (Efficient)

A better decomposition:

```
h = W₀ · x  +  B · A · x · (α/r)
  = (base contribution)  +  (adapter contribution)
```

Compute the base contribution once for the entire batch — all tokens, regardless of which adapter they need. Then compute the adapter contribution separately, per-adapter. Add the two.

This has two immediate benefits:

1. **The base model is held once** — regardless of how many adapters are active. 14 GB of base weights, loaded once.
2. **The base forward pass can be batched across all requests simultaneously** — token A needs adapter 1, token B needs adapter 2, but both execute the same `W₀ · x` computation in the same matrix multiply.

The adapter contribution computation is the remaining problem: how do you efficiently compute `B_n · A_n · x` for each token in a batch where different tokens need different adapter pairs?

> "There are two methods to compute h using LoRA... approach 2) seems to be more suitable for a multi-task/tenancy scenario, as we can compute the left term once, the right term for every adapter n, and sum them as needed." — João Moura (AWS), 2024

### Why the Adapter Computation Is the Hard Part

A naive implementation iterates over adapters and launches a separate matrix multiply for each group of tokens sharing that adapter. Each kernel launch has overhead. With 50 adapters and diverse access patterns, the GPU launches 50 separate matrix multiplications per layer, each tiny, each with its own scheduling overhead. GPU utilisation collapses.

This is precisely the problem that the Punica project solved with its SGMV kernel.

---

## 4. The SGMV Kernel: Batching Across Different Adapters

The key insight from Punica (MLSys 2024) is that the adapter computation can be fused across all adapters in a batch — not executed separately per adapter.

### Segmented Gather Matrix-Vector Multiplication (SGMV)

For a batch with tokens using different adapters:

```
Token 0: adapter (A₁, B₁)  →  delta₀ = x₀ @ A₁.T @ B₁.T * scale
Token 1: adapter (A₂, B₂)  →  delta₁ = x₁ @ A₂.T @ B₂.T * scale
Token 2: adapter (A₁, B₁)  →  delta₂ = x₂ @ A₁.T @ B₁.T * scale
Token 3: (base only)        →  delta₃ = 0
```

SGMV executes all of these in a single kernel launch:
1. **Gather:** map each token to its adapter weight matrices
2. **Segment:** group tokens sharing the same adapter (tokens 0 and 2 above both use A₁/B₁)
3. **Multiply:** compute all segments' matrix products in one fused operation, sharing CUDA thread blocks efficiently across segments

The result: GPU utilisation stays high regardless of how many distinct adapters are in the batch. SGMV treats heterogeneous adapter computation as a single parallelisable operation, not as N separate operations.

### What the Benchmark Says

Punica's evaluation on a single A100 80GB with LLaMA-7B:

> "Punica achieves 12× higher throughput in serving multiple LoRA models compared to state-of-the-art LLM serving systems while only adding 2ms latency per token." — Punica (MLSys 2024)

The 12× throughput gain comes from two sources:
1. The base model GEMM is computed once for the full batch (not once per adapter)
2. SGMV eliminates per-adapter kernel launch overhead — all adapter deltas are computed in one fused launch

The 2ms latency overhead is nearly constant regardless of the number of adapters in the batch. You pay 2ms extra per token for the SGMV computation; the actual adapter count doesn't change that cost significantly.

### SGMV vs BGMV

Later work introduced a variant called BGMV (Batched Gather Matrix-Vector Multiplication), optimised specifically for the decode phase:

| Kernel | Optimised for | Used by |
|---|---|---|
| SGMV | Prefill (batch of tokens, long sequences) | LoRAX, SGLang Triton backend |
| BGMV | Decode (one token per request) | vLLM |

Both eliminate the per-kernel-launch overhead for multi-adapter batching. SGMV is better for long prompt processing; BGMV is better for single-token autoregressive steps.

### The Masked Alternative (Layer 20)

Layer 20 implements a simpler — but correct — alternative to SGMV:

```python
out = base_output + lora_delta * lora_mask   # mask ∈ {0.0, 1.0} per token
```

This always computes `lora_delta` for every token (even base-model tokens, whose results are zeroed by the mask), while SGMV skips computation for base-model tokens entirely. For a batch where 10% of tokens need a LoRA adapter, the masked approach wastes 90% of the extra matmuls. SGMV skips the wasted computation.

The masked approach is significantly simpler to implement (10 lines vs ~500 lines of custom CUDA) and completely correct for the single-adapter use case. SGMV is the production upgrade for multi-adapter batching at scale.

---

## 5. Three Pillars of Production Multi-Adapter Serving

With Punica's SGMV kernel solving the compute problem, the remaining challenges are operational: how do you manage an adapter pool that may contain thousands of adapters without running out of VRAM or creating unacceptable cold-start latency?

LoRAX (Predibase's open-source inference server, originally forked from HuggingFace TGI) assembled all of these components into a production-ready architecture. Its design rests on three pillars.

### Pillar 1: Dynamic Adapter Loading

Rather than loading all adapters at server startup, LoRAX loads adapters **just-in-time**:

1. Server starts with only the base model in VRAM
2. When a request specifies `adapter_id="my-fine-tuned-model"`, the loader checks its registry
3. If the adapter is already in VRAM: immediate processing
4. If not: fetch from HuggingFace Hub, S3, or local disk; queue the request; process once loaded

The fetch is non-blocking: while adapter A is loading, requests for other already-loaded adapters continue processing uninterrupted. A LoRA adapter is typically 10–200 MB, so loading takes hundreds of milliseconds — far less than the minutes required for a full model. The server feels responsive even during cold starts.

> "The time taken to load an adapter is a function of its size. Since LoRA adapters are typically between 10MB and 200MB, the load time is measured in hundreds of milliseconds." — Neel Shah (Towards AI), 2026

### Pillar 2: Tiered Weight Caching

Memory management is the critical limiting factor for a high-density inference server. A three-tier cache hierarchy manages adapter weight lifetime:

| Tier | Storage | Policy | Speed |
|---|---|---|---|
| **Hot** | GPU VRAM | Currently active | Immediate |
| **Warm** | Host DRAM | LRU eviction from GPU | ~10ms reload |
| **Cold** | NVMe / S3 | Full adapter catalog | ~100ms–seconds |

When GPU VRAM is full and a new adapter is needed, the Least Recently Used (LRU) adapter is evicted from VRAM to DRAM. If DRAM is also full, adapters move from DRAM to disk. A warm start (DRAM → VRAM) is nearly instantaneous compared to a cold start (disk or network download → VRAM).

This tiered hierarchy prevents OOM errors while enabling a server to hold, in principle, thousands of adapters — as many as fit on disk — across all three tiers simultaneously. The GPU tier is limited by remaining VRAM after the base model and KV cache are allocated.

### Pillar 3: Continuous Multi-Adapter Batching

The third pillar extends continuous batching — the technique that keeps GPUs busy by processing requests as they arrive rather than waiting for full batch windows — to work across multiple adapters.

In standard continuous batching, the engine maintains a running batch of requests and slots new arrivals in as compute is available. Multi-adapter continuous batching extends this with a **heterogeneous scheduler**:

1. A fair scheduler marks a subset of adapters as "active" at any given time
2. Requests from active adapters are drained from their queues and combined into a single heterogeneous batch
3. The SGMV kernel computes all adapter deltas in one pass, with a mathematical mask ensuring each token uses the correct adapter's weights
4. After a configurable time window, the scheduler rotates: a different set of adapters becomes active, ensuring all adapters eventually get served

> "Requests from active adapters are drained from their queues and combined into a single batch. A mathematical mask is applied during the computation of activations to ensure that each input sequence is processed by the correct adapter weights." — Neel Shah (Towards AI), 2026

The mask mentioned here is the production version of Layer 20's `lora_mask` approach — extended to N adapters and powered by SGMV instead of dense matmuls.

---

## 6. The Live Benchmark: 50 Adapters at the Cost of One

The theoretical claims from Punica are compelling. The most concrete empirical evidence comes from an end-to-end benchmark run by João Moura (AWS Solutions Architect) on Amazon SageMaker, published January 2024.

### Setup

- **Model:** `mistralai/Mistral-7B-Instruct-v0.1`
- **Adapter:** `vineetsharma/qlora-adapter-Mistral-7B-Instruct-v0.1-gsm8k` (grade school math, rank-8)
- **Replication:** 50 copies of this adapter uploaded under different S3 prefixes (simulating 50 distinct adapters)
- **Hardware:** AWS `ml.g5.xlarge` — one NVIDIA A10G 24 GB GPU
- **Concurrency:** 20 parallel clients, 300 total requests

### What the Test Shows

The benchmark compares two traffic patterns:
1. **Single adapter:** all 300 requests use the same adapter
2. **Random access:** each request randomly selects one of the 50 adapters

The result:

```
Single Adapter:
  Total Time: 42.34s  |  Average Latency: 2.80s  |  Throughput: 7.09 req/s

50 Adapters (random access):
  Total Time: 42.60s  |  Average Latency: 2.82s  |  Throughput: 7.04 req/s
```

**Virtually identical performance.** Serving 50 different adapters randomly in parallel costs almost the same as serving one adapter repeatedly.

> "We would effectively now be serving 50 different models on a single A10G." — João Moura, 2024

This is the central empirical claim of multi-adapter serving, validated end-to-end on production hardware. The Punica/SGMV kernel's 12× throughput claim is confirmed in a realistic workload.

### Why It Works: The Arithmetic

The base model GEMM dominates compute. On a Mistral-7B with a 4096-token context, each forward pass is dominated by the 32 attention layers × 32 heads × 4096 dimensions = billions of multiply-adds in `W₀ · x`. The LoRA delta (`B · A · x`) is a tiny fraction of this: rank-8 adds `2 × 4096 × 8 = 65,536` multiply-adds per layer, against `4096 × 4096 × 32 = 536 million` for the base layer. The LoRA overhead is 0.012% of the base model computation.

SGMV's job is to compute that 0.012% correctly per token without the overhead of separate kernel launches. Once it succeeds, the adapter count becomes irrelevant to throughput.

---

## 7. Serving LoRA Adapters in SGLang

SGLang's LoRA serving system incorporates both S-LoRA and Punica techniques, providing a production-ready multi-adapter serving stack built directly on the same framework as Layer 20's minimal implementation.

### Enabling LoRA: The Key Arguments

```bash
python3 -m sglang.launch_server \
    --model-path meta-llama/Meta-Llama-3.1-8B-Instruct \
    --enable-lora \
    --lora-paths lora0=algoprog/fact-generation-llama-3.1-8b-instruct-lora \
    --max-loras-per-batch 2 \
    --lora-eviction-policy lru
```

| Argument | What it controls |
|---|---|
| `--enable-lora` | Enable LoRA support (auto-set if `--lora-paths` is provided) |
| `--lora-paths` | Adapters to load at startup; format: `name=path_or_hf_id` |
| `--max-loras-per-batch` | Maximum adapters active simultaneously in GPU pool. Affects VRAM reservation. Default: 8 |
| `--max-loaded-loras` | Cap on adapters in CPU memory (must be ≥ `max-loras-per-batch`) |
| `--lora-eviction-policy` | `lru` (default) or `fifo` — which adapter leaves GPU pool when full |
| `--lora-backend` | `triton` or `csgmv` (Chunked SGMV) — the kernel used for adapter delta computation |
| `--max-lora-rank` | Maximum rank to reserve GPU buffer for. Needed for dynamic loading of unknown adapters |
| `--lora-target-modules` | Projection modules to enable LoRA on; set `all` for all supported modules |

### Serving Single vs Multiple Adapters

**Single adapter, per-request selection (Native API):**

```python
import requests

json_data = {
    "text": [
        "List 3 countries and their capitals.",  # uses lora0
        "List 3 countries and their capitals.",  # uses base model
    ],
    "sampling_params": {"max_new_tokens": 32, "temperature": 0},
    "lora_path": ["lora0", None],  # None → base model
}
response = requests.post("http://localhost:{port}/generate", json=json_data)
```

**Multiple adapters in the same batch:**

```python
json_data = {
    "text": ["List 3 countries.", "Write a Python function."],
    "lora_path": ["lora0", "lora1"],  # different adapter per sequence
}
```

This is the mixed-batch capability Layer 20 implements at the request level: different tokens in the same batch can use different adapters (or no adapter). SGLang uses SGMV/Triton kernels for the adapter computation; Layer 20 uses the dense float-mask approach.

**OpenAI-compatible API (uses `model` field as adapter selector):**

```python
from openai import OpenAI
client = OpenAI(api_key="EMPTY", base_url="http://localhost:{port}/v1")

# Use adapter: model="base_model_name:adapter_name"
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct:lora0",
    messages=[{"role": "user", "content": "List 3 countries"}]
)

# Use base model: model="base_model_name"
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[{"role": "user", "content": "List 3 countries"}]
)
```

The `model:adapter` colon syntax routes to the named adapter. No adapter suffix routes to the base model. This is how Layer 20's `lora_id` field maps to the production API.

### Dynamic Loading and Unloading at Runtime

Rather than specifying all adapters at startup, SGLang supports loading and unloading adapters while the server runs:

```python
# Load a new adapter (e.g., just pushed a new fine-tune)
requests.post(f"http://localhost:{port}/load_lora_adapter", json={
    "lora_name": "new_fine_tune",
    "lora_path": "huggingface_user/my-new-adapter",
})

# Unload an adapter (free its VRAM/DRAM slot)
requests.post(f"http://localhost:{port}/unload_lora_adapter", json={
    "lora_name": "old_adapter",
})
```

**Important:** When planning to dynamically load adapters after server startup, specify `--max-lora-rank` and `--lora-target-modules` at launch. Without these, SGLang infers the maximums from `--lora-paths` at startup and may reject later-loaded adapters with larger ranks or different target modules.

### Memory Architecture: What `max-loras-per-batch` Controls

SGLang pre-allocates a GPU memory pool for adapter weights at startup:

```
GPU VRAM allocation:
  Base model weights:       fixed (e.g., 16 GB for LLaMA-8B)
  KV cache:                 configurable (largest remaining)
  LoRA GPU pool:            max_loras_per_batch × max_lora_rank × module_dims × 2
```

The LoRA GPU pool holds up to `max_loras_per_batch` adapter A/B matrices simultaneously. When a new adapter is needed and the pool is full, the LRU adapter's slot is reclaimed. The evicted adapter's weights move to CPU DRAM (warm cache), not discarded.

Layer 20 is the degenerate case: `max_loras_per_batch = 1`, never evicted, loaded once at startup.

### Tensor Parallelism

SGLang follows S-LoRA's tensor parallelism strategy for multi-GPU deployments:

- Column-parallel modules (`q_proj`, `k_proj`, `v_proj`, `gate_proj`, `up_proj`): LoRA B is column-sharded; LoRA A is replicated
- Row-parallel modules (`o_proj`, `down_proj`): LoRA A is row-sharded; LoRA B is replicated

This ensures each GPU can compute its portion of the adapter delta independently, with the same all-reduce cost as the base model's TP communication.

---

## 8. Serving LoRA Adapters in vLLM

vLLM's multi-LoRA implementation uses a different API design from SGLang — request-level `LoRARequest` objects for offline inference, and the `model` field in API requests for online serving.

### Offline Inference with `LoRARequest`

```python
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from huggingface_hub import snapshot_download

# Download adapter locally
sql_lora_path = snapshot_download(repo_id="jeeejeee/llama32-3b-text2sql-spider")

# Instantiate with LoRA support enabled
llm = LLM(model="meta-llama/Llama-3.2-3B-Instruct", enable_lora=True)

# Generate with a specific adapter
sampling_params = SamplingParams(temperature=0, max_tokens=256)
outputs = llm.generate(
    prompts=["Write a SQL query to answer: ..."],
    sampling_params=sampling_params,
    lora_request=LoRARequest("sql_adapter", 1, sql_lora_path),
    # Arguments: (human_name, unique_int_id, adapter_path)
)
```

The `LoRARequest` object is passed per `generate()` call. For mixed batches where different prompts need different adapters, see `examples/offline_inference/multilora_inference.py`.

### Online Serving with the OpenAI-Compatible API

```bash
vllm serve meta-llama/Llama-3.2-3B-Instruct \
    --enable-lora \
    --lora-modules sql-lora=jeeejeee/llama32-3b-text2sql-spider
```

Querying with the adapter:
```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "sql-lora", "prompt": "Write a SQL query...", "max_tokens": 256}'
```

The `model` field selects the adapter — the same field used for base model selection in standard OpenAI API calls. Using the base model name routes to the base model; using a registered adapter name routes to that adapter. Requests are processed in parallel.

### Dynamic Loading at Runtime

```bash
# Enable runtime LoRA loading
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Load a new adapter
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "new_adapter", "lora_path": "/path/to/adapter"}'

# Unload an adapter
curl -X POST http://localhost:8000/v1/unload_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "new_adapter"}'
```

### LoRAResolver Plugins (Fully Automatic Loading)

vLLM supports LoRAResolver plugins that automatically resolve and load adapters on demand — no explicit load call needed. When a request arrives with an unknown `model` name, the resolver is consulted:

```python
from vllm.lora.resolver import LoRAResolver, LoRAResolverRegistry
from vllm.lora.request import LoRARequest

class S3LoRAResolver(LoRAResolver):
    async def resolve_lora(self, base_model_name, lora_name):
        local_path = self.download_from_s3(lora_name)
        return LoRARequest(lora_name=lora_name, lora_path=local_path,
                           lora_int_id=abs(hash(lora_name)))

LoRAResolverRegistry.register_resolver("s3_resolver", S3LoRAResolver())
```

With a resolver registered, every new model name is automatically fetched and loaded. This is the production realisation of LoRAX's "dynamic adapter loading" pillar, implemented as an extensible plugin.

### In-Place Adapter Reloading

For Reinforcement Learning pipelines where adapters are continuously updated and re-served without downtime:

```bash
curl -X POST http://localhost:8000/v1/load_lora_adapter \
  -H "Content-Type: application/json" \
  -d '{"lora_name": "rl-adapter", "lora_path": "/path/to/updated/adapter", "load_inplace": true}'
```

`load_inplace=true` replaces the existing weights in-place — the adapter name stays registered, but its weights are swapped to the new checkpoint. The server never needs to restart.

---

## 9. The HuggingFace PEFT Checkpoint Format

Every adapter on HuggingFace Hub — including `phh/Qwen3-0.6B-TLDR-Lora` used in Layer 20 — follows the PEFT checkpoint format. Understanding this format is essential for loading adapters correctly and for verifying adapter implementations.

### Files in a PEFT Adapter Checkpoint

```
adapter_config.json
adapter_model.safetensors   (sometimes adapter_model.bin for older checkpoints)
README.md                   (optional)
```

### `adapter_config.json` — The Configuration

```json
{
  "base_model_name_or_path": "Qwen/Qwen3-0.6B",
  "peft_type": "LORA",
  "r": 8,
  "lora_alpha": 32,
  "lora_dropout": 0.05,
  "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
  "bias": "none",
  "task_type": "CAUSAL_LM"
}
```

Key fields:
- `r`: rank — controls adapter size and expressiveness
- `lora_alpha`: the scaling numerator; effective scale = `lora_alpha / r`
- `target_modules`: which projection layers have LoRA applied

These are the fields that `LoRAAdapter.__init__()` reads from `adapter_config.json`.

### `adapter_model.safetensors` — The Weight Tensors

Keys follow a strict naming convention:

```
base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_A.weight  →  shape [r, in_dim]
base_model.model.model.layers.{layer_idx}.self_attn.q_proj.lora_B.weight  →  shape [out_dim, r]
base_model.model.model.layers.{layer_idx}.mlp.gate_proj.lora_A.weight     →  shape [r, in_dim]
base_model.model.model.layers.{layer_idx}.mlp.gate_proj.lora_B.weight     →  shape [out_dim, r]
...
```

The structure is:
- `base_model.model.` — PEFT prefix (always present)
- `model.layers.{layer_idx}.` — model architecture path
- `{component}.{module_name}.` — e.g., `self_attn.q_proj.`
- `lora_A.weight` or `lora_B.weight`

`LoRAAdapter._load_weights()` splits each key on `.`, finds `lora_A` or `lora_B`, extracts the layer index and module name, and stores the tensor in a nested dict `{layer_idx: {module_name: tensor}}`.

### Loading and Using a PEFT Adapter for Inference

The standard PEFT inference workflow:

```python
from transformers import AutoModelForCausalLM
from peft import PeftModel

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")

# Wrap with adapter
peft_model = PeftModel.from_pretrained(base_model, "phh/Qwen3-0.6B-TLDR-Lora")
peft_model.eval()

# Inference — adapter is applied automatically
outputs = peft_model(**inputs)
```

This is the ground truth that `verify_lora.py` in Layer 20 compares against.

### Merging Adapters for Zero-Overhead Inference

If you only need a single adapter and don't need to switch between adapters:

```python
# Merge adapter weights into base model permanently
merged_model = peft_model.merge_and_unload()
# merged_model is now a standard model with W = W₀ + B·A·scale
# Zero inference overhead — one GEMM, not two
# Cannot be unmerged
```

Merging is appropriate for production deployments with a fixed adapter. It is inappropriate for multi-adapter serving (Layer 20, LoRAX, SGLang production) because it destroys the base model's separability from the adapter.

### Target Module Coverage

| Coverage | Configuration | Use case |
|---|---|---|
| Minimal | `target_modules=["q_proj", "v_proj"]` | Fastest training; only attention value projection |
| Attention | `target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]` | Full attention coverage |
| Full | `target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` | All attention + MLP; strongest adaptation |
| QLoRA-style | `target_modules="all-linear"` | Every linear layer; maximum parameter budget |

`phh/Qwen3-0.6B-TLDR-Lora` uses full coverage (7 modules, rank-8). This is why `LoRAAdapter._load_weights()` must handle keys for all 7 module types.

---

## 10. Advanced Production Features

### Structured Generation (JSON Mode)

LLMs fine-tuned for extraction tasks — invoice parsing, entity recognition, API call generation — need their outputs to be valid JSON, not just probably-valid JSON. LoRAX integrates the Outlines library to enforce this:

During token sampling, structured generation constrains the probability distribution so only valid tokens (tokens that keep the response on a valid JSON parse path) have nonzero probability. The result is 100% structural validity — the model can never produce malformed JSON.

The value compounds with LoRA fine-tuning: the adapter learns which fields to extract and what their content should be, while structured generation guarantees the output format. Combined results from the LoRAX team:

| Mode | Content accuracy | Structural validity |
|---|---|---|
| Base model | 50% | 90% |
| LoRA adapter | 71% | 92% |
| Adapter + structured generation | **80%** | **99.9%** |

The adapter improves content; structured generation eliminates formatting failures. Both are necessary for a production extraction pipeline.

### Lookahead LoRA: Speculative Decoding Without a Draft Model

Standard speculative decoding uses a small "draft" model to predict the next 3–5 tokens, which the larger target model then verifies in parallel. It accelerates generation at the cost of maintaining and serving a separate draft model.

LoRAX's Lookahead LoRA trains the adapter to do both: predict the next token (standard generation) and predict the following 2–3 tokens (draft prediction). The adapter itself becomes the speculative decoder. No separate model is needed.

Reported throughput improvement: **2–3× over standard LoRA adapters**. This makes Lookahead LoRA particularly valuable for latency-sensitive applications like code completion and real-time chat, where Inter-Token Latency (ITL) is the critical metric.

### Dynamic Adapter Management in vLLM (LoRAResolver + in-place reload)

For production RL pipelines where adapters are updated continuously:

1. **LoRAResolver** loads new adapters automatically when their names first appear in requests — no explicit `/load_lora_adapter` call required. Resolvers can pull from local directories, HuggingFace Hub, or custom backends like S3.

2. **In-place reload** (`load_inplace=true`) swaps adapter weights without changing the adapter name — ongoing requests referencing that adapter name see the updated weights seamlessly.

Together, these enable a pipeline where a training loop pushes new adapter checkpoints and the serving layer picks them up without any downtime or redeployment.

---

## 11. The KV Cache vs Adapter VRAM Trade-off

Multi-adapter serving introduces a VRAM trade-off that does not exist for single-model serving: every adapter slot in the GPU pool consumes memory that would otherwise be available for the KV cache.

### The Competition

On an A10G 24 GB GPU serving Mistral-7B:

```
Base model weights: ~14 GB
Available for KV + adapters: ~10 GB
```

With each rank-8 adapter occupying ~50 MB for a 7B model, the adapter pool costs are:

| Active adapters | Adapter pool | KV cache budget | Max batch size (approx.) |
|---|---|---|---|
| 1 | ~50 MB | ~9.95 GB | ~100 requests |
| 8 | ~400 MB | ~9.6 GB | ~96 requests |
| 32 | ~1.6 GB | ~8.4 GB | ~84 requests |
| 128 | ~6.4 GB | ~3.6 GB | ~36 requests |

At 128 active adapters, you have 64% of the original KV cache budget. Batch size drops, GPU utilisation drops, throughput drops.

> "Loading more adapters to GPU for concurrent execution means you will have less DRAM available to store the KV cache... the optimal configuration will be defined by the specific requirements of your workload." — João Moura, 2024

### The Break-Even Point

The trade-off has a break-even point that depends on your adapter access pattern. With diverse adapter access (each request uses a different adapter), more concurrent adapters in the GPU pool means fewer cold starts and lower average latency. With concentrated access (most requests use one of two adapters), keeping a small pool with large KV cache is better: batch sizes are higher and the two hot adapters are always warm.

Practical guidance:

- **If your adapter access distribution is heavily skewed** (80% of requests use 5% of adapters): keep `max_loras_per_batch` at 8–16. The hot adapters are always resident, cold adapters load on demand, and the large KV cache handles the high volume on the hot adapters.

- **If your adapter access is nearly uniform** (each adapter gets equal traffic): larger pool sizes make sense, because every adapter has a reasonable chance of being requested in the next batch window.

- **For Layer 20's single-adapter use case**: the pool always has exactly one adapter. The full remaining VRAM goes to KV cache. No trade-off exists.

---

## 12. Decision Framework: When Does Multi-LoRA Serving Pay Off?

### The 4-Check Framework

**Check 1 — How many adapters do you actually need?**

If you need fewer than 4 adapters with predictable access patterns, consider separate dedicated servers with merged weights. Each merged model has zero inference overhead and the operational simplicity of single-model serving. The multi-adapter infrastructure pays off at 8+ adapters, and becomes clearly worthwhile at 50+.

**Check 2 — What is your adapter access distribution?**

Collect a day of production traffic and plot adapter request frequency. If your access is highly skewed (top 10 adapters handle 95% of traffic), you need a hot pool of ~10 adapters with LRU eviction for the long tail. If access is uniform, you need a larger GPU pool. Neither case requires serving 1,000 adapters simultaneously — just the active concurrency set.

**Check 3 — What is your KV cache budget?**

Calculate the KV cache requirement for your typical batch:

```python
kv_per_token = n_layers × n_kv_heads × head_dim × 2 × dtype_bytes
kv_per_request = kv_per_token × avg_sequence_length
kv_budget = gpu_vram - model_weights - adapter_pool_size
max_concurrent_requests ≈ kv_budget / kv_per_request
```

If your adapter pool reduces `max_concurrent_requests` below your target, either reduce `max_loras_per_batch` or provision a larger GPU.

**Check 4 — Do you need dynamic loading or can you pre-load at startup?**

Pre-loading all adapters at startup (like Layer 20 does, but extended to N adapters via `--lora-paths`) works when your adapter catalog is fixed and fits in CPU DRAM. Dynamic loading (via `/load_lora_adapter` or LoRAResolver) is needed when adapters are created continuously (e.g., user-specific fine-tunes, RL training loops) or when the catalog is too large for full pre-loading.

### The Hardware Decision

| Scenario | Recommendation |
|---|---|
| 1 adapter, static | Merge into base model; zero overhead; standard serving |
| 2–4 adapters, stable | Pre-load at startup with `--lora-paths`; minimal pool |
| 5–100 adapters, stable catalog | SGLang or vLLM with `max_loras_per_batch=8–16`; LRU eviction |
| 100+ adapters, dynamic catalog | Dynamic loading via REST API or LoRAResolver; tier to CPU/disk |
| RL training pipeline | vLLM with in-place reload (`load_inplace=true`) |
| Serverless / per-request isolation | LoRAX with backbone sharing; ServerlessLoRA-style architecture |

### The Economic Case

For the typical SaaS scenario that motivated this technology — 1,000 customer-specific fine-tunes of a 7B model:

| Architecture | GPU instances | Monthly cost (H100 at $3/hr) |
|---|---|---|
| Dedicated per adapter | 1,000 | ~$2.16M/month |
| Multi-LoRA (single server, 100 concurrent) | 10–20 | ~$22K–$43K/month |
| Savings | — | **~98% cost reduction** |

The savings assume 24/7 serving with comparable request rates. Real workloads have uneven traffic; the actual savings come from not provisioning for peak on each adapter independently.

> "The transition from 'One Model per GPU' to '1000 Adapters per GPU' is more than just a hardware optimisation; it is a fundamental shift in how we architect AI systems." — Neel Shah (Towards AI), 2026

---

## Key Quotes

> "Fine-tuning large language models is easier than ever. Serving them efficiently? That's where LoRAX steps in." — Sai Mudhiganti, July 2025

> "There are two methods to compute h using LoRA... approach 2) seems to be more suitable for a multi-task/tenancy scenario, as we can compute the left term once, the right term for every adapter n, and sum them as needed." — João Moura (AWS), January 2024

> "Punica achieves 12× higher throughput in serving multiple LoRA models compared to state-of-the-art LLM serving systems while only adding 2ms latency per token." — Punica (MLSys 2024)

> "Loading more adapters to GPU for concurrent execution means you will have less DRAM available to store the KV cache... the optimal configuration will be defined by the specific requirements of your workload." — João Moura (AWS), January 2024

> "The transition from 'One Model per GPU' to '1000 Adapters per GPU' is more than just a hardware optimisation; it is a fundamental shift in how we architect AI systems." — Neel Shah (Towards AI), March 2026

> "To the best of our knowledge, this is the highest reported throughput for DeepSeek-V3 serving at that time." (The same architecture SGLang uses for multi-LoRA runs the DeepSeek-V3 96 H100 deployment) — SGLang Team, May 2025

---

## What Is Left Out and Why

### Left out: Punica SGMV kernel CUDA implementation

Section 4 explains what SGMV does and why it is faster than naive per-adapter kernel launches. The Punica paper (L3/02) contains the actual CUDA implementation: thread block assignment to segments, gather operation for non-contiguous adapter pages, handling of variable-rank adapters in the same launch. These implementation details require CUDA programming knowledge and belong at L3.

### Left out: S-LoRA Unified Paging formal analysis

Section 7 describes how SGLang's `max_loras_per_batch` and `lora_eviction_policy` work conceptually. S-LoRA (L3/03) provides the formal analysis: proof that Unified Paging eliminates fragmentation, the queuing model for adapter loading latency, and the tensor parallelism sharding derivation. These belong at L3.

### Left out: dLoRA merge/unmerge credit algorithm

Section 12's decision framework includes the insight that merging is better for skewed adapter distributions. dLoRA (L4/01) formalises this as a credit-based algorithm with configurable thresholds and adaptive merging based on live traffic. The full algorithm and its convergence analysis belong at L4.

### Left out: CaraServe CPU-assisted prefill

CaraServe (L4/02) addresses cold-start latency by overlapping CPU prefill execution with adapter loading. This is a niche optimisation for systems where adapter cold starts are frequent and CPU prefill is fast enough to be useful. It is not covered here because neither SGLang nor vLLM currently implement it, and it requires understanding the CPU-GPU synchronisation model to implement.

### Left out: LoRA variant mathematics (DoRA, rsLoRA, PiSSA, EVA)

Section 9 mentions that PEFT supports variants like `use_rslora=True` and `use_dora=True`. The mathematical derivations of each variant (rsLoRA's `alpha/sqrt(r)` scaling analysis, DoRA's weight decomposition into magnitude and direction, PiSSA's SVD-based initialisation proof) are covered in L3 sources and require engaging with the original LoRA math at depth.

### Left out: InfiniLoRA disaggregated LoRA server

InfiniLoRA (L4/06) extends PD disaggregation to LoRA computation, running adapter deltas on a dedicated LoRA Server cluster that communicates with base model workers via GPU-initiated RDMA. This is the frontier of multi-LoRA architecture but requires understanding both PD disaggregation (Layer 19) and multi-LoRA serving (Layer 20) before it makes sense. It is L4 material.

### Left out: Loquetier unified fine-tuning and serving

Loquetier (L4/03) addresses the case where the same GPU cluster must simultaneously train new adapters and serve existing ones. The Virtualised Module design and fused training+serving kernel are advanced systems engineering material that belongs at L4 and requires understanding gradient computation in the presence of batched inference.
