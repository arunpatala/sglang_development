# Layer 16 — SGLang Router on Kubernetes (Helm Chart)

This directory contains a Helm chart that deploys the full SGLang router + worker stack on Kubernetes, as covered in the Layer 16 lesson.

```
k8s/
└── sglang/                  ← the Helm chart
    ├── Chart.yaml
    ├── values.yaml          ← all configuration with comments
    └── templates/
        ├── _helpers.tpl         common template functions
        ├── namespace.yaml       Namespace
        ├── rbac.yaml            ServiceAccount + Role + RoleBinding
        ├── secret.yaml          HuggingFace token Secret
        ├── pvc.yaml             PersistentVolumeClaim (model weights)
        ├── worker-deployment.yaml   GPU worker pods
        ├── router-deployment.yaml   CPU router pod (service discovery)
        ├── router-service.yaml      ClusterIP Service + session affinity
        ├── ingress.yaml             Nginx Ingress with X-Session-ID routing
        ├── hpa.yaml                 HorizontalPodAutoscaler (workers)
        ├── pdb.yaml                 PodDisruptionBudget (workers + router)
        ├── servicemonitor.yaml      Prometheus ServiceMonitor
        └── prometheusrule.yaml      Alert rules (4 essential alerts)
```

**Lesson sections this chart implements:**

| Section | File |
|---|---|
| 03 — Prerequisites & RBAC | `rbac.yaml`, `secret.yaml` |
| 04 — Worker Deployment | `worker-deployment.yaml`, `pvc.yaml` |
| 05 — Router Deployment | `router-deployment.yaml`, `router-service.yaml` |
| 07 — Observability | `servicemonitor.yaml`, `prometheusrule.yaml` |
| 08 — High Availability | `pdb.yaml`, `hpa.yaml`, session affinity in `router-service.yaml`, `ingress.yaml` |

---

## Prerequisites

```bash
# 1. Kubernetes cluster with GPU nodes
kubectl get nodes -l nvidia.com/gpu.present=true

# 2. NVIDIA GPU Operator (if not already installed)
helm repo add nvidia https://helm.ngc.nvidia.com/nvidia && helm repo update
helm install gpu-operator nvidia/gpu-operator -n gpu-operator --create-namespace

# 3. Helm 3
helm version

# 4. (Optional) Prometheus Operator — for ServiceMonitor and alerts
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install prometheus prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false

# 5. (Optional) Nginx Ingress — for X-Session-ID routing
helm repo add ingress-nginx https://kubernetes.github.io/ingress-nginx
helm install ingress-nginx ingress-nginx/ingress-nginx -n ingress-nginx --create-namespace
```

---

## Quickstart

### 1. Minimal install (no external tools needed)

```bash
# Clone / navigate to this directory
cd CODE_LAYERS/layer16_k8s/k8s

helm install sglang ./sglang \
  --set model.name="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --set model.hfToken.value="hf_xxxxxxxxxxxxxxxxxxxx" \
  --set worker.replicas=2

# Watch pods come up (workers take 2-10 min to load the model)
kubectl get pods -n production -w

# Test the router
kubectl port-forward svc/sglang-sglang-router -n production 30000:30000 &
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"meta-llama/Meta-Llama-3.1-8B-Instruct","messages":[{"role":"user","content":"Hello"}],"max_tokens":50}'
```

### 2. Production install (with observability + HA)

```bash
helm install sglang ./sglang \
  --set model.name="meta-llama/Meta-Llama-3.1-8B-Instruct" \
  --set model.hfToken.existingSecret="my-hf-token" \
  --set worker.replicas=4 \
  --set router.replicas=2 \
  --set ha.sessionAffinity.type=ClientIP \
  --set ha.pdb.enabled=true \
  --set ha.hpa.enabled=true \
  --set ha.hpa.maxReplicas=8 \
  --set observability.serviceMonitor.enabled=true \
  --set observability.prometheusRule.enabled=true
```

### 3. Using a values file (recommended for production)

```bash
# Copy and edit the values file
cp sglang/values.yaml my-values.yaml
# edit my-values.yaml ...

helm install sglang ./sglang -f my-values.yaml

# Upgrade after changes
helm upgrade sglang ./sglang -f my-values.yaml
```

---

## Key Configuration

### Changing the Model

```bash
helm upgrade sglang ./sglang \
  --set model.name="Qwen/Qwen2.5-7B-Instruct" \
  --set model.storage="30Gi"
```

### Scaling Workers

```bash
# Static scaling
helm upgrade sglang ./sglang --set worker.replicas=6

# Or use kubectl directly (router discovers new pods automatically)
kubectl scale deployment sglang-sglang-worker -n production --replicas=6

# Autoscaling (requires prometheus-adapter)
helm upgrade sglang ./sglang \
  --set ha.hpa.enabled=true \
  --set ha.hpa.minReplicas=2 \
  --set ha.hpa.maxReplicas=8 \
  --set ha.hpa.targetRequestsPerPod=20
```

### HuggingFace Token (Gated Models)

**Option A — provide in values (development only):**
```bash
helm install sglang ./sglang --set model.hfToken.value="hf_xxx"
```

**Option B — pre-create the secret (production):**
```bash
kubectl create secret generic my-hf-token \
  --namespace production \
  --from-literal=token="hf_xxx"

helm install sglang ./sglang \
  --set model.hfToken.existingSecret="my-hf-token"
```

### Routing Policy

```bash
# Cache-aware (default) — best for workloads with repeated prefixes
helm install sglang ./sglang --set router.policy=cache_aware

# Round-robin — uniform load, no prefix awareness
helm install sglang ./sglang --set router.policy=round_robin

# Power-of-two — pick least-loaded of 2 random workers
helm install sglang ./sglang --set router.policy=power_of_two
```

---

## X-Session-ID Routing (Custom Session Affinity)

X-Session-ID gives you semantic control over which router pod handles a request. Since each router pod has its own radix tree, routing the same "session" to the same router pod means the radix tree will route those requests to the same GPU — maximising KV cache hits.

**When to use it:** when you know which requests share prefixes (same system prompt, same user conversation, same document).

### Enable in the Helm chart

```bash
helm upgrade sglang ./sglang \
  --set ingress.enabled=true \
  --set ingress.host="llm.example.com" \
  --set ingress.sessionId.enabled=true \
  --set ingress.sessionId.header="X-Session-ID"
```

### Client code

```python
import openai

# All requests with the same X-Session-ID:
#   → same router pod (Nginx hash)
#   → same worker/GPU (router's radix tree)
#   → KV cache hit for shared prefix

# Pattern 1: route by system prompt (all users with same prompt → same GPU)
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": "system-prompt-customer-support-v2"}
)

# Pattern 2: route by user (per-user conversation stays on one GPU)
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": f"user-{user_id}"}
)

# Pattern 3: route by document (batch processing same doc → one GPU)
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": f"doc-{document_id}"}
)
```

**What happens when a router pod crashes:**
Nginx reassigns those sessions to the surviving router pod. The surviving router's radix tree is cold for those sessions — the first few requests have cache misses — then it warms up and cache hits resume.

---

## Why Two Router Replicas Don't Sync

Each router pod maintains its own **independent radix tree in RAM**. There is no cross-pod synchronisation. This is intentional — implementing distributed consensus across radix trees is complex and not currently implemented in the SGLang gateway.

The consequence: with round-robin across 2 routers, the same prefix may land on different routers, which route to different workers, causing cache misses.

The solutions (in order of simplicity):

| Solution | Trade-off |
|---|---|
| `sessionAffinity: ClientIP` | Simple; breaks behind NAT |
| `X-Session-ID` + Nginx | Best cache efficiency; requires ingress + client change |
| `replicas: 1` | Perfect cache coherence; 30s downtime on crash |

See `lesson/08_high_availability.md` for the full analysis.

---

## Verification

```bash
# All pods running?
kubectl get pods -n production

# Workers discovered by router?
kubectl logs -n production -l app=sglang-router | grep "Adding pod"

# Metrics available?
kubectl port-forward svc/sglang-sglang-router -n production 29000:29000 &
curl -s http://localhost:29000/metrics | grep discovery_workers_discovered

# Alerts loaded? (requires prometheus-operator)
kubectl get prometheusrule -n production

# Test session affinity (same X-Session-ID → same router pod)
for i in 1 2 3 4 5; do
  curl -s http://llm.example.com/v1/chat/completions \
    -H "X-Session-ID: test-session-abc" \
    -H "Content-Type: application/json" \
    -d '{"model":"...","messages":[{"role":"user","content":"hi"}],"max_tokens":5}' | \
    jq '.id'
done
# All requests should complete successfully (same router, same GPU)
```

---

## Uninstall

```bash
helm uninstall sglang -n production

# PVC is NOT deleted by default (model weights are kept)
# To delete: kubectl delete pvc sglang-sglang-model-cache -n production
```
