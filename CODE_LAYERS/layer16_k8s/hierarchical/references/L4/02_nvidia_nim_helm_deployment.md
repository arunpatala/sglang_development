# NVIDIA NIM — Helm and Kubernetes Deployment

**Source:** https://docs.nvidia.com/nim/large-language-models/latest/deployment/kubernetes-deployment/helm-k8s.html
**Author:** NVIDIA
**Level:** L3–L4 — Industry reference for Helm-based LLM serving
**Why here:** NIM on Kubernetes is the production reference for how NVIDIA recommends deploying LLM servers via Helm. Pattern is identical to Layer 16's worker deployment (single GPU pod + PVC + NGC secret) but with higher-level abstractions. Good comparison point to understand what Layer 16 does manually vs. what NIM automates.

---

## Overview

NVIDIA NIM (NVIDIA Inference Microservices) are optimized containers for LLM inference. The NIM Helm chart (`nim-llm`) deploys a GPU-backed LLM server with:
- NGC API key authentication for image pull and model download
- PVC for model weight caching
- GPU resource limits
- Readiness/liveness probes
- LoRA adapter support

---

## Prerequisites

- Kubernetes cluster with GPU-capable nodes
- Configured `kubectl` and Helm 3+
- NGC API key (for NVCR image registry and model artifacts)
- Storage class supporting PersistentVolumes

---

## Fetch the Helm Chart

```bash
export HELM_CHART_VERSION="<version_number>"
helm fetch "https://helm.ngc.nvidia.com/nim/charts/nim-llm-${HELM_CHART_VERSION}.tgz" \
  --username='$oauthtoken' \
  --password="${NGC_API_KEY}"
tar -xzf "nim-llm-${HELM_CHART_VERSION}.tgz"
```

Inspect defaults and README:
```bash
helm show readme nim-llm/
helm show values nim-llm/
```

---

## Key Helm Values

| Value | Description |
|---|---|
| `image.repository` | NIM container image to deploy |
| `image.tag` | NIM container image tag |
| `model.ngcAPISecret` | K8s Secret containing NGC API key |
| `imagePullSecrets` | Secret for pulling from NVCR |
| `persistence.enabled` | Enable PVC for model cache |
| `persistence.storageClass` | K8s StorageClass for the PVC |
| `persistence.size` | PVC size (set based on model size + cache usage) |
| `resources.limits.nvidia.com/gpu` | GPU count per pod |
| `env` | Optional runtime environment variables |

**Layer 16 parallel:** `persistence.size` advice matches Layer 16's `values.yaml` PVC sizing guidance: measure actual model size and add headroom for HF cache.

---

## Cache and Temporary Directories

Several env vars are derived from `model.nimCache` (default: `/model-store`):

| Env Variable | Value | Purpose |
|---|---|---|
| `NIM_CACHE_PATH` | `/model-store` | Primary model cache |
| `HF_HOME` | `/model-store/huggingface/hub` | HuggingFace cache |
| `OUTLINES_CACHE_DIR` | `/model-store/outlines` | Structured output grammar cache |

---

## Minimal Deployment Example

### Create Secrets

```bash
export NGC_API_KEY=<your_ngc_api_key>

# Image pull secret (for nvcr.io)
kubectl create secret docker-registry ngc-secret \
  --docker-server=nvcr.io \
  --docker-username='$oauthtoken' \
  --docker-password="${NGC_API_KEY}"

# NGC API key secret (for model download at runtime)
kubectl create secret generic nvidia-nim-secrets \
  --from-literal=NGC_API_KEY="${NGC_API_KEY}"
```

### Create `values.yaml`

```yaml
image:
  repository: nvcr.io/nim/meta/llama-3.1-8b-instruct
  tag: "1.8.4"

model:
  ngcAPISecret: "nvidia-nim-secrets"

persistence:
  enabled: true
  storageClass: "standard"         # Adjust to available StorageClass
  accessMode: ReadWriteOnce
  size: 50Gi                       # Adjust based on model size

resources:
  limits:
    nvidia.com/gpu: 1              # 1 GPU for 8B model

imagePullSecrets:
  - name: "ngc-secret"
```

### Install

```bash
helm install my-nim nim-llm/ -f values.yaml
```

---

## Verify Deployment

```bash
# Check pods are running
kubectl get pods -l app.kubernetes.io/instance=my-nim

# Check service
kubectl get svc -l app.kubernetes.io/instance=my-nim

# Port-forward for local testing
kubectl port-forward svc/my-nim-nim-llm 8000:8000

# Test readiness
curl -sS http://127.0.0.1:8000/v1/health/ready
```

---

## LoRA Adapters

```yaml
# Add to values.yaml to enable LoRA
env:
  - name: NIM_PEFT_SOURCE
    value: /loras
extraVolumes:
  lora-adapter:
    persistentVolumeClaim:
      claimName: nvidia-nim-lora-pvc
extraVolumeMounts:
  lora-adapter:
    mountPath: /loras
```

Adapter directory structure:
```
/loras/
  adapter_name/
    adapter_config.json
    adapter_model.safetensors
```

---

## Multi-node via LeaderWorkerSet

For models too large for a single node (e.g., Llama-3.1 405B), NIM uses LWS for multi-node deployment. This is beyond Layer 16's scope. Key configuration:

```yaml
# values.yaml (multi-node NIM)
multiNode:
  enabled: true
  leaderWorkerSet:
    size: 2    # 1 leader + 1 worker = 2 nodes, 16 GPUs total
  gpuPerNode: 8
```

See `L3/05_lws_vllm_multi_node_deployment.md` for the LWS deployment pattern.

---

## Comparison: NIM Helm vs. Layer 16 YAML

| Aspect | NIM Helm chart | Layer 16 K8s manifests |
|---|---|---|
| Authentication | NGC API key (nvcr.io) | HuggingFace token (dockerhub or ghcr.io) |
| Model source | NGC Model Catalog (NVIDIA-optimized) | HuggingFace Hub |
| PVC | `persistence.*` values | Explicit PVC YAML |
| GPU resource | `resources.limits.nvidia.com/gpu` | Same — identical pattern |
| Probes | Preconfigured in chart | Explicit in YAML, tunable |
| Service discovery | Not included — static single pod | SGLang router's `--service-discovery` |
| Abstraction level | High (everything in values.yaml) | Low (explicit control over every resource) |

Layer 16 intentionally uses low-level YAML to teach the patterns that tools like NIM's Helm chart hide. Once you understand the YAML, the Helm chart is just a template on top of the same structures.
