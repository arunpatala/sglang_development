# Introducing Gateway API Inference Extension

**Source:** https://kubernetes.io/blog/2025/06/05/introducing-gateway-api-inference-extension/
**Docs:** https://gateway-api-inference-extension.sigs.k8s.io/
**GitHub:** https://github.com/kubernetes-sigs/gateway-api-inference-extension
**Authors:** Daneyon Hansen (Solo.io), Kaushik Mitra (Google), Jiaxin Shan (Bytedance), Kellen Swain (Google)
**Date:** June 5, 2025
**Level:** L3–L4 — Ecosystem context, "what comes after Layer 16"
**Why here:** The official Kubernetes standard for LLM-aware routing. This is where the ecosystem is heading: model-aware, criticality-based routing standardized as a K8s API. Understanding GIE explains where SGLang's `--service-discovery` flag fits — and what replaces it at scale.

---

## The Problem

Modern LLM inference sessions are:
- **Long-running and resource-intensive**: a single GPU-backed model server may keep multiple inference sessions active
- **Partially stateful**: in-memory KV cache (token cache) is maintained per-session
- **Model-identity-aware**: a request for "gpt-4-chat" must reach a server running that model

Traditional load balancers (round-robin, path-based HTTP routing) lack:
- Model identity awareness
- Request criticality (interactive chat vs. batch jobs)
- Real-time pod metrics (queue depth, KV cache pressure, loaded LoRA adapters)
- Standardized approach across vendors

---

## Gateway API Inference Extension

Built on the existing Kubernetes Gateway API, adding inference-specific routing capabilities while retaining familiar Gateways and HTTPRoutes.

By adding an inference extension to your existing gateway, you transform it into an **Inference Gateway** — enabling "model-as-a-service" hosting of GenAI/LLMs.

**Key objectives:**
- Enable model-aware routing
- Support per-request criticalities (interactive vs. batch)
- Facilitate safe model roll-outs
- Optimize load balancing on real-time model metrics (queue depth, memory pressure, loaded adapters)

---

## New CRDs

### `InferencePool`

Defines a pool of pods (model servers) running on shared compute. Platform admin-controlled.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferencePool
metadata:
  name: llama-pool
  namespace: production
spec:
  targetPortNumber: 8080
  selector:
    matchLabels:
      app: vllm-llama
  extensionRef:
    name: endpoint-selection-extension   # ← EPP sidecar
```

Similar to a Kubernetes `Service` but specialized for AI/ML serving — aware of model-serving protocol, queue depth, and KV cache state.

### `InferenceModel`

A user-facing model endpoint managed by AI/ML workload owners.

```yaml
apiVersion: inference.networking.x-k8s.io/v1alpha2
kind: InferenceModel
metadata:
  name: customer-support-bot
  namespace: ai-workloads
spec:
  modelName: customer-support        # ← public model name in API requests
  criticality: Critical              # ← Interactive, NonCritical, or Critical
  poolRef:
    name: llama-pool
```

Maps a public model name (e.g., `"customer-support"`) to the actual model in an `InferencePool`, with traffic-splitting and prioritization policy.

**Separation of concerns:**
- `InferenceModel`: AI/ML owners manage *what* is served
- `InferencePool`: Platform operators manage *where and how* it's served

---

## Request Flow

```
Client → Gateway (Envoy/NGINX/Istio) → HTTPRoute matches InferencePool
       → Endpoint Selection Extension (EPP) → picks best pod
       → Routes to specific pod IP
```

1. **Gateway routing**: Client sends `POST /v1/completions`. Gateway examines HTTPRoute and identifies the matching `InferencePool` backend.
2. **Endpoint selection**: Instead of round-robin, the Gateway consults the **Endpoint Selection Extension (EPP)** — a sidecar that examines live pod metrics:
   - Queue lengths per pod
   - KV cache memory usage per pod
   - Loaded LoRA adapters
   - Request criticality
3. **Inference-aware scheduling**: EPP picks the optimal pod and returns it to the Gateway, which forwards the request to that specific pod IP (not the Service VIP).

This extra step is invisible to the client — it still sees a single request with a single response.

---

## Benchmarks (10 vLLM replicas on H100 GPUs, ShareGPT workload)

### Key results

**Comparable throughput**: ESE delivered throughput on par with standard Kubernetes Service across all tested QPS.

**Lower latency at high QPS:**
- p90 per-output-token latency: significantly lower at 500+ QPS
- p90 overall latency: lower at 400–500+ QPS

**Why:** Model-aware routing avoids hotspots. Standard load balancing sends requests to all pods uniformly — some pods may be overloaded while others are idle. ESE routes to the least-loaded pod based on real queue depth and KV cache state.

---

## Supported Gateway implementations (2025–2026)

| Gateway | Status |
|---|---|
| GKE L7 Gateway | Supported |
| Istio (v1.27+) | Supported |
| NGINX Gateway Fabric | Supported |
| Agentgateway (llm-d) | Default in llm-d |
| Envoy Gateway | Roadmap |

SGLang and vLLM are both listed as supported model servers.

---

## Roadmap

1. Prefix-cache aware load balancing for remote caches
2. LoRA adapter pipelines for automated rollout
3. Fairness and priority between workloads in the same criticality band
4. HPA support for scaling based on aggregate, per-model metrics
5. Heterogeneous accelerators (latency- and cost-aware routing across GPU types)
6. Disaggregated serving (independently scaling prefill and decode pools)

---

## How this relates to Layer 16

| Layer 16 (SGLang router) | Gateway API Inference Extension |
|---|---|
| Application-layer routing (`python -m sglang_router.launch_router`) | L7 gateway-layer routing |
| `--service-discovery` watches pods | `InferencePool` is the pod pool abstraction |
| `cache_aware` policy in Rust | Endpoint Selection Extension (EPP) |
| Label selector (`--selector app=sglang-worker`) | `InferencePool.spec.selector` |
| Criticality: not supported | `InferenceModel.spec.criticality` |
| Deployed as a pod in the cluster | Deployed as a gateway extension |

Layer 16's SGLang router solves the same problem as GIE's EPP — but at the application layer, specific to SGLang. GIE is the emerging Kubernetes-native standard that works with any compliant model server.

---

## Getting started

```bash
# Deploy vLLM model server
MODEL_SERVER=vllm
INFERENCE_POOL_NAME=${MODEL_SERVER}-qwen3-32b
MODEL_NAME=Qwen/Qwen3-32B

# Install inference extension CRDs
kubectl apply -f https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/latest/download/install.yaml

# Test
IP=$(kubectl get gateway/inference-gateway -o jsonpath='{.status.addresses[0].value}')
curl -i ${IP}:80/v1/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "Qwen/Qwen3-32B", "prompt": "Hello", "max_tokens": 50}'
```
