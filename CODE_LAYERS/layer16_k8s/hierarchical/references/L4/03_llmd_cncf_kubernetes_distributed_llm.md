# llm-d: Kubernetes-Native Distributed LLM Inference (CNCF Sandbox)

**Sources:**
- Website: https://llm-d.ai/
- GitHub: https://github.com/llm-d/llm-d/
- CNCF blog: https://www.cncf.io/blog/2026/03/24/welcome-llm-d-to-the-cncf-evolving-kubernetes-into-sota-ai-infrastructure/
**Author:** IBM Research, Red Hat, Google Cloud (donated to CNCF)
**Date:** May 2025 (launched); March 2026 (CNCF Sandbox)
**Level:** L4 — "What comes after Layer 16"
**Why here:** llm-d is the production-scale evolution of the same architecture Layer 16 teaches. Layer 16: one router + service discovery + cache-aware routing. llm-d: disaggregated serving + hierarchical KV cache + inference-aware routing + scale-to-zero, all on Kubernetes. Reading llm-d's design explains why each Layer 16 decision was made.

---

## What is llm-d?

llm-d is a Kubernetes-native, high-performance distributed LLM inference framework. It fills the gap between:
- **Low-level**: vLLM/SGLang (efficient inference on a single pod or pod group)
- **High-level**: KServe (abstract model serving platform)

**llm-d's role**: middleware between inference engine and orchestration layer, adding:
1. Disaggregated Prefill/Decode (P/D) serving
2. Hierarchical KV cache offloading (GPU → CPU → NVMe)
3. Inference-aware, prefix-cache-aware routing
4. Scale-to-zero autoscaling

**CNCF Sandbox** (March 2026): donated by IBM Research, Red Hat, and Google Cloud. Backed by NVIDIA, CoreWeave, AMD, Cisco, Hugging Face, Intel, Lambda, Mistral AI.

---

## What llm-d adds on top of vLLM/SGLang

| Layer 16 capability | llm-d extension |
|---|---|
| Cache-aware routing (single router) | Inference-aware routing via Gateway API Inference Extension (EPP) |
| Service discovery (pod watcher) | `InferencePool` CRD manages backend pod groups |
| Round-robin / least-load / cache-aware | Prefix-cache-aware routing maintaining near-zero TTFT under load |
| HPA on queue depth | Scale-to-zero autoscaling + workload variant autoscaler |
| Single GPU per pod | LWS multi-node replicas; wide expert parallelism |
| No KV cache offloading | Hierarchical KV offloading: GPU → CPU memory → NVMe |
| No P/D disaggregation | Separate prefill and decode node pools |

---

## Architecture

```
Client
  ↓
Kubernetes Gateway API (with Inference Extension)
  ↓ EPP picks best pod
InferencePool → vLLM/SGLang pods (prefill or decode)
  ↓
Hierarchical KV Cache (GPU → CPU → Storage)
  ↓
LeaderWorkerSet for multi-node groups
```

**Core components:**
- **vLLM** as default model server and inference engine
- **Kubernetes Gateway API Inference Extension (GIE)** as control plane API and load balancing orchestrator
- **LeaderWorkerSet (LWS)** for multi-node replicas and wide expert parallelism
- **NIXL** for high-performance KV cache transfer between nodes

---

## Key Features (v0.5, February 2026)

### Prefill/Decode Disaggregation

Separate prefill and decode node pools allow them to scale independently:
- Prefill is compute-intensive (processes the prompt)
- Decode is memory-bandwidth-intensive (generates tokens one at a time)
- With shared pools: bottleneck alternates; with disaggregated pools: each scales to its bottleneck

### Hierarchical KV Cache Offloading

KV cache is offloaded across tiers based on access frequency:
- **GPU VRAM** (hot): active sessions
- **CPU memory** (warm): recently completed sessions
- **NVMe / object storage** (cold): historical sessions for prefix reuse

Enables massive prefix cache reuse without consuming GPU VRAM for idle contexts.

### Inference-Aware Routing

The Endpoint Picker (EPP) routes each request to the pod that:
- Has the longest prefix match for the request (KV cache reuse)
- Has the lowest queue depth
- Has available KV cache memory
- Matches the request criticality level

**Benchmark (Qwen3-32B, 8×vLLM pods, 16×NVIDIA H100):** Near-zero TTFT up to ~120k tok/s; standard Kubernetes Service degrades rapidly under load.

### Scale-to-Zero

Workers scale to zero when idle and provision automatically on new traffic. Includes a workload variant autoscaler that handles the cold-start period.

---

## Benchmarks (v0.5, Feb 2026)

| Metric | Value |
|---|---|
| Decode throughput per B200 GPU | ~3.1k tok/s (wide expert parallelism) |
| Max output tokens on 16×16 B200 topology | ~50k tok/s |
| TTFT reduction vs. round-robin | Order-of-magnitude |
| Per-output-token latency improvement (DeepSeek V3.1) | 40% (v0.4 on H200) |

---

## Hardware support

llm-d's core design principle is vendor-neutral:
- NVIDIA A100+
- AMD MI250+
- Google TPU v5e+
- Intel GPU
- Kubernetes 1.29+

Accelerators are selected declaratively via Kubernetes `nodeSelector` and Device Resource Allocation (DRA).

---

## Quick Start

```bash
# Prerequisites: kubectl, helm, yq, git, HuggingFace token

# Install llm-d with inference scheduling well-lit path
cd quickstarts/guides/inference-scheduling
# Follow README for your cluster type

# Validate
helm list -n ${NAMESPACE}
kubectl get all -n ${NAMESPACE}

# Make an inference request
curl http://${GATEWAY_IP}/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "llama-3.1-8b", "prompt": "Hello", "max_tokens": 50}'
```

---

## How this relates to Layer 16

Layer 16 teaches the foundational pattern: one SGLang router + Kubernetes service discovery + cache-aware routing. This pattern is exactly what llm-d extends to production scale.

| Layer 16 teaches | llm-d builds on top |
|---|---|
| K8s service discovery (`--service-discovery`) | `InferencePool` CRD with GIE EPP |
| RBAC for pod watching (`get/list/watch`) | Same RBAC, applied to more resources |
| Cache-aware routing (SGLang's radix tree) | Distributed prefix-cache-aware routing |
| Single router replica (HA issue) | gRPC mesh state sync (issue #10839) → llm-d's active-active HA |
| HPA on queue depth | Workload variant autoscaler + scale-to-zero |
| LWS for multi-node workers | LWS integrated into deployment topology |

The gap between Layer 16 and llm-d is **disaggregation** and **scale**: Layer 16 uses one worker type; llm-d separates prefill and decode workers and coordinates KV cache across all of them.

---

## CNCF Sandbox status (2026)

**Sandbox** = CNCF's early-stage designation. Production stability is not yet fully validated for all use cases. Recommendation: validate quantitative gains through benchmarking in staging before production rollout.

Graduation trajectory: Sandbox → Incubating → Graduated (same path as Prometheus, Kubernetes, Helm).
