# Spotlight: Perplexity AI Serves 400 Million Search Queries a Month Using NVIDIA Inference Stack

**Source:** https://developer.nvidia.com/blog/spotlight-perplexity-ai-serves-400-million-search-queries-a-month-using-nvidia-inference-stack/
**Author:** Amr Elmeleegy (NVIDIA)
**Published:** December 5, 2024
**Note:** As of March 18, 2025, NVIDIA Triton Inference Server is now part of NVIDIA Dynamo Platform (renamed to NVIDIA Dynamo Triton).
**Level:** L1 — Production case study; real-world scale; disaggregation as roadmap item
**Why here:** Answers the question "is disaggregated serving production-ready?" — yes, Perplexity AI is actively deploying it at 435M queries/month. Provides concrete production details: 20+ simultaneous models, H100 SXM, Kubernetes architecture, tensor parallelism choices, and cost savings ($1M/year). Also documents the active collaboration with NVIDIA's Triton engineering team to deploy disaggregated serving.

---

## Scale Context

Perplexity AI is an AI-powered search engine serving:
- **435 million+ queries per month** (at article publication time, up from 400M in title)
- **20+ AI models simultaneously** (Llama-3.1 8B, 70B, 405B, embedding models, classifiers)
- **Kubernetes cluster** with pods running H100 GPUs managed by NVIDIA Triton Inference Server

Each query represents **multiple AI inference requests** (classifier → search → summarisation → follow-up questions).

---

## Production Architecture

### Multi-Model Routing

1. Incoming user request
2. **Classifier models** (< 1B parameters) determine user intent
3. Request routed to the appropriate model pod
4. Each pod: one or more H100 GPUs + Triton Inference Server instance

### Scheduling Algorithm

- **In-house front-end scheduler**: routes traffic to the appropriate pod based on load and usage
- Scheduling algorithm directly impacts **inter-token latency** — particularly the worst percentile (P99)
- Perplexity constantly investigates new scheduler optimisations including better accounting for sequence length variation

**Internal benchmark comparison (at 16 QPS):**

| Load balancing strategy | ITL P99 |
|---|---|
| Round-robin | Highest |
| Least requests | Middle |
| Power of two random choices | Lowest |

### Parallelism Choices

For user-facing Llama models (8B, 70B, 405B):
- **Tensor parallelism (TP)**: increased to 4 or 8 GPUs → lower serving cost for latency-sensitive requests within fixed GPU budget
- **Data or pipeline parallelism**: used for maximising throughput in less latency-sensitive settings

**Cost impact of TP:** Sharding Llama-3.1-8B with TP=4 across 4 H100 GPUs → up to **3× lower cost per million tokens** for latency-sensitive requests.

### SLA-Driven Deployment

- **Embedding models** (< 1B params): lowest possible latency, small batch sizes, multiple models per H100
- **User-facing models**: deeper analysis of TTFT, tokens/second/user, cost per million queries
- A/B testing across configurations to find optimal SLA compliance / GPU utilisation balance

---

## Disaggregated Serving: Active Deployment

> "The inference team at Perplexity adopts a comprehensive, full-stack approach to their road map, consistently optimizing and enhancing every layer of the stack — from applications and use cases to inference serving middleware and hardware accelerators."

> "In terms of inference serving middleware, the team is actively collaborating with the NVIDIA Triton engineering team to **deploy disaggregating serving, a groundbreaking technique that separates the prefill and decode inference phases of an LLM workflow onto separate NVIDIA GPUs.** This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token. Additionally, this technique gives Perplexity the flexibility to use **different NVIDIA GPU products for each inference phase** given its specific hardware resource requirements."

**Key confirmations from this quote:**
1. Disaggregated serving is being **actively deployed** (not evaluated) by Perplexity
2. It "significantly boosts overall system throughput while meeting SLAs"
3. It enables **hardware heterogeneity**: different GPU SKUs for prefill vs decode phases (the core Splitwise insight)
4. Partnership with NVIDIA Triton → now NVIDIA Dynamo platform

---

## Cost Savings Reported

**Related-Questions feature:**
- Powers follow-up question suggestions after each search query
- By hosting internally on cloud H100 GPUs instead of third-party APIs: **approximately $1 million annually saved**

---

## Infrastructure Stack

| Component | Technology |
|---|---|
| GPU hardware | NVIDIA HGX H100 (4-GPU and 8-GPU systems) |
| Inference server | NVIDIA Triton Inference Server (now Dynamo Triton) |
| Model runtime | NVIDIA TensorRT-LLM + proprietary CUDA kernels |
| Container orchestration | Kubernetes |
| Model registry | NVIDIA NGC |
| Enterprise software | NVIDIA AI Enterprise |

---

## Why This Matters for Layer 19

**Before this article:** disaggregated serving might seem like a research concept.
**After this article:** it is a production system serving 435M queries/month at a major AI company.

**Specific insights for Layer 19:**
- The **hardware heterogeneity** insight (different GPUs for prefill vs decode) is confirmed as a production motivation, not just a theoretical optimisation.
- **TTFT, tokens/second/user, and cost per million queries** are the three SLA metrics Perplexity actually tracks — exactly the metrics DistServe's goodput framework is designed to optimise.
- The collaboration with **NVIDIA Triton/Dynamo** team is the origin story for NVIDIA Dynamo's disaggregated serving architecture (L3/04).
- The **custom CUDA kernels + TensorRT-LLM** combination shows that inference stack optimisation is multi-layered — disaggregation is the system-level piece, not the whole picture.

---

## Key Takeaways for Layer 19

- Perplexity AI at 435M queries/month is the production proof that disaggregated serving works at scale — read this before doubting whether disaggregation is worth the engineering investment.
- Hardware heterogeneity (different GPU SKUs for prefill and decode) is not a theoretical proposal; it is a stated production benefit already being leveraged.
- The P99 ITL improvement from better load balancing (power of two choices > least requests > round-robin) is a preview of the KV-aware router sophistication described in Dynamo's architecture (L3/04).
- The $1M/year cost saving from hosting a single feature (Related-Questions) on GPUs illustrates the financial scale at which disaggregation's 15–40% infrastructure cost reduction becomes significant.
