# LoRA Serving on Amazon SageMaker — Serve 100's of Fine-Tuned LLMs For the Price of 1

**Source:** https://medium.com/@joaopcmoura/lora-serving-on-amazon-sagemaker-serve-100s-of-fine-tuned-llms-for-the-price-of-1-85034ef889c5
**Author:** João Moura (AI Solutions Architect, AWS)
**Published:** January 26, 2024 — 15 min read
**Level:** L1 — End-to-end practitioner tutorial with benchmark results
**Why here:** Explains the economics and systems reasoning behind multi-adapter serving (Punica, S-LoRA), then shows a complete deployment on Amazon SageMaker. The benchmark section is especially valuable: 50 adapters randomly accessed in parallel achieves *virtually the same throughput* as single-adapter requests — directly demonstrating the central claim of multi-LoRA serving.

---

## The Dawn of Multi-Adapter Serving

### LoRA in a Nutshell

The input is forwarded through both the base model and the low-rank adapter at each targeted layer, and the outputs are summed:

```
h = W₀ · x  +  B · A · x · (α/r)
```

- `W₀`: frozen pre-trained weights
- `A ∈ ℝ^{r×d}`, `B ∈ ℝ^{d×r}`: trainable low-rank matrices, `r ≪ d`
- `α`: scaling hyperparameter; `r`: rank

### The Two-Path Computation Choice

**Method 1 — Merge weights first:** Compute `(W₀ + B·A) · x`. Only works for a single adapter; limits batch size to 1.

**Method 2 — Compute separately:** Compute `W₀ · x` once for the full batch, then `B_n · A_n · x` per adapter n. This enables the base model forward pass to be batched independently of the adapter forward pass.

### Systems Optimization: Punica and S-LoRA

Method 2 raises the question: how to batch the adapter forward pass efficiently?

**Punica** (MLSys 2024) introduced the **Segmented Gather Matrix-Vector Multiplication (SGMV)** kernel. For usage patterns where each request targets a different adapter, Punica achieves **12× the throughput** of state-of-the-art inference servers like vLLM, keeping latency nearly constant with the number of concurrent adapters.

**S-LoRA** (MLSys 2024) added:
- Unified memory pool to reduce fragmentation
- Novel tensor parallel strategy for LoRA computation
- Highly optimized CUDA kernels for heterogeneous batching

## Production-Ready: LoRAX Server

LoRAX (originally forked from HuggingFace TGI v0.9.4) added adapter-specific features on top of a production-ready LLM serving stack:

1. **Heterogeneous continuous batching** — schedules requests for different adapters in the same batch; powered by Punica
2. **Dynamic adapter loading** — add `adapter_id` to request, LoRAX downloads from HuggingFace Hub or S3 just-in-time, without blocking other requests
3. **Adapter offloading between GPU and CPU memory** — adapters loaded to GPU as needed, offloaded according to scheduling policy when GPU memory is saturated

A simple mask ensures the correct adapter is applied to each request in the batch when computing activations:

```
output = W₀ · x  +  mask * (B_n · A_n · x)
```

This is exactly the mechanism Layer 20 implements in `lora.py` and the masked forward pass.

## Tutorial: Deploy LoRAX on SageMaker

### Prerequisites

- AWS account, AWS CLI configured
- SageMaker IAM role

### Build Custom Container

LoRAX is fork-compatible with SageMaker since it exposes `/invocations` and `/ping`. Create `sagemaker_entrypoint.sh`:

```bash
#!/bin/bash
if [[ -z "${HF_MODEL_ID}" ]]; then
  echo "HF_MODEL_ID must be set"; exit 1
fi
export MODEL_ID="${HF_MODEL_ID}"
if [[ -n "${SM_NUM_GPUS}" ]]; then export NUM_SHARD="${SM_NUM_GPUS}"; fi
if [[ -n "${ADAPTER_BUCKET}" ]]; then export PREDIBASE_MODEL_BUCKET="${ADAPTER_BUCKET}"; fi
lorax-launcher --port 8080
```

Build and push to ECR:
```bash
algorithm_name="lorax"
tag="sagemaker"
region="us-east-1"
account=$(aws sts get-caller-identity --query Account --output text)
image_uri="${account}.dkr.ecr.${region}.amazonaws.com/${algorithm_name}:${tag}"

cd sagemaker_lorax/ && docker build -t ${algorithm_name}:${tag} .
aws ecr get-login-password --region ${region} | docker login --username AWS --password-stdin ...
docker push ${image_uri}
```

### Deploy the Endpoint (g5.xlarge, single A10G 24GB)

```python
from sagemaker import Model

config = {
    'HF_MODEL_ID': "mistralai/Mistral-7B-Instruct-v0.1",
    'SM_NUM_GPUS': "1",
    'MAX_INPUT_LENGTH': "1024",
    'MAX_TOTAL_TOKENS': "4096",
    'ADAPTER_BUCKET': sagemaker_session_bucket,
}
lorax_model = Model(image_uri=image_uri, role=role, env=config)
lorax_predictor = lorax_model.deploy(
    endpoint_name='sm-lorax',
    initial_instance_count=1,
    instance_type="ml.g5.xlarge",
    container_startup_health_check_timeout=800,
)
```

### Invoke Base Model vs Adapter

```python
prompt = '[INST] Natalia sold clips to 48 friends in April, then half as many in May. Total? [/INST]'

# Base model
payload_base = {"inputs": prompt, "parameters": {"max_new_tokens": 64}}

# Adapter from S3
payload_adapter = {
    "inputs": prompt,
    "parameters": {
        "max_new_tokens": 64,
        "adapter_id": "lorax/mistral-adapters/10",
        "adapter_source": "s3"
    }
}
```

**Results:**
- Base model: `"Let's break down the problem: 1. In April, Natalia sold 48 clips…"` (did not converge to final answer within 64 tokens)
- Adapter (GSM8K-trained): `"48/2 = 24 clips in May. Total: 48 + 24 = 72 clips. #### 72"` ✓

### Benchmark: Single Adapter vs 50 Random Adapters

- 300 total requests, 20 parallel threads, same prompt
- All 50 adapters pre-loaded to disk before benchmark

```
Single Adapter:
  Total Time: 42.34s
  Average Latency: 2.80s
  Throughput: 7.09 req/s

50 Adapters (random access):
  Total Time: 42.60s
  Average Latency: 2.82s
  Throughput: 7.04 req/s
```

**Amazing: virtually identical performance.** This demonstrates Punica/SGMV's core claim — multi-adapter serving overhead is negligible when using segmented kernel batching.

## Performance Considerations

Loading more adapters to GPU for concurrent execution means less DRAM for the KV cache. This creates a fundamental tradeoff:

> **Bigger batches = better resource utilization** vs **Higher adapter diversity = more DRAM used**

The optimal configuration depends on:
- Traffic patterns (skewed vs random adapter access)
- Sequence lengths (KV cache growth)
- GPU VRAM budget
- Required SLO

## Relevance to Layer 20

Layer 20 resolves this tradeoff with the simplest possible choice: **one static adapter, always in GPU memory.** No LRU eviction, no cold-start, no SGMV kernel. The mask approach (`output + delta * mask`) is the same mathematical idea as the production LoRAX mask shown in Fig.3 of the original LoRAX blog post, cited in this article.
