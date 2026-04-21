# Layer 16 — Summary

Layer 16 takes the routing gateway from Layer 15 and makes it Kubernetes-aware. No new routing policies are added — `RoundRobinPolicy`, `LeastLoadPolicy`, and `PrefixCacheAwarePolicy` carry forward unchanged. The single change is replacing the static `--worker-urls` list in `config.yml` with a Kubernetes service discovery loop: the router connects to the K8s API at startup, watches for pods matching a label selector, and adds or removes them from its worker pool in real time as pods come and go.

---

## From Layer 15 to Layer 16

In Layer 15, the router was configured with static backend URLs:

```yaml
# Layer 15 config.yml
router:
  host: "0.0.0.0"
  port: 8200
  workers:
    - url: "http://localhost:8114"
      name: "engine-a"
    - url: "http://localhost:8115"
      name: "engine-b"
  policy: prefix_cache_aware
```

This works on a developer laptop. In Kubernetes, it breaks immediately:
- A pod restart gives the pod a new IP. The old `http://10.0.1.14:8000` is dead.
- `kubectl scale deployment sglang-worker --replicas=6` creates four new pods. The router's list still has two entries.
- A rolling update replaces pods one at a time. The router's list contains a mix of old and new IPs throughout the rollout.

In Layer 16, the static list is replaced with a watcher:

```bash
# Layer 16: no --worker-urls, service discovery instead
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

The router starts with zero workers, connects to the K8s API, lists all pods currently matching `app=sglang-worker,component=inference` in the `production` namespace, adds each one that is Running and Ready, then watches for future events. When a new pod becomes Ready, it is added. When a pod begins terminating, it is removed before it shuts down.

---

## The Architecture in Kubernetes

```
Internet / internal clients
        ↓
  [K8s Service: sglang-router]          ← stable ClusterIP DNS, port 30000
        ↓                               ← routes to one of the router replicas
  [Router Deployment: 2 replicas]       ← sglang-router image, CPU only, no weights
        │  watches K8s API (pod labels)
        │  adds/removes workers as pods appear/disappear
        ↓
  [Worker Deployment: N GPU pods]       ← lmsysorg/sglang:latest, 1 GPU each
        ↓
  [PVC: model-cache]                    ← model weights, survives pod restarts
```

Three things are new compared to Layer 15's local setup:
1. **K8s Service** in front of the router — provides a stable DNS name regardless of which replica is running.
2. **RBAC objects** — allow the router pod to call the K8s API.
3. **Pod labels** on worker pods — the discovery contract between worker Deployment and router.

---

## Prerequisites and RBAC

The router process calls the Kubernetes API from inside the cluster. It needs:

**A ServiceAccount:**
```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: sglang-router
  namespace: production
```

**A Role with minimal permissions — only pods, only read:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-router
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list", "watch"]
```

**A RoleBinding:**
```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-router
  namespace: production
subjects:
- kind: ServiceAccount
  name: sglang-router
  namespace: production
roleRef:
  kind: Role
  name: sglang-router
  apiGroup: rbac.authorization.k8s.io
```

The `list` verb is needed for the initial bootstrap (loading currently-running pods on startup). The `watch` verb enables the streaming event connection. The `get` verb is needed for individual pod lookups. Nothing else. Using a `ClusterRole` instead of a `Role` would allow the router to watch pods across all namespaces — a wider blast radius than needed.

The router pod picks up the ServiceAccount token automatically from the mounted secret at `/var/run/secrets/kubernetes.io/serviceaccount/`. In `service_discovery.rs`, `Client::try_default()` (line 223) detects this automatically — no kubeconfig file is needed.

---

## Worker Deployment

Worker pods are GPU pods running the SGLang engine. The key elements:

**Labels the router will match:**
```yaml
template:
  metadata:
    labels:
      app: sglang-worker
      component: inference
      model: llama-3-8b
```

**GPU resource request:**
```yaml
resources:
  requests:
    nvidia.com/gpu: "1"
    memory: "32Gi"
    cpu: "8"
  limits:
    nvidia.com/gpu: "1"
    memory: "64Gi"
    cpu: "16"
```

**Shared memory volume** (required for tensor parallelism — from vLLM's K8s guide at `REPOS/vllm/docs/deployment/k8s.md`):
```yaml
volumes:
- name: shm
  emptyDir:
    medium: Memory
    sizeLimit: "2Gi"
# in container:
volumeMounts:
- name: shm
  mountPath: /dev/shm
```

**Readiness probe — the critical gate:**
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120   # model loading: measure once, set with buffer
  periodSeconds: 5
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 10
```

The readiness probe is the gate the router relies on. In `service_discovery.rs`, `is_healthy()` (line 196) returns `true` only when `is_ready == true` (Ready condition = "True") **and** `status == "Running"`. A pod that is starting up (Running but not Ready) or stuck in CrashLoopBackOff is never added to the worker pool. Set `initialDelaySeconds` conservatively — if it is too low, K8s kills the pod before the model finishes loading.

**Model weights via PVC:**
```yaml
volumes:
- name: model-cache
  persistentVolumeClaim:
    claimName: sglang-models-pvc
volumeMounts:
- mountPath: /root/.cache/huggingface
  name: model-cache
```

**HuggingFace token for gated models:**
```yaml
env:
- name: HF_TOKEN
  valueFrom:
    secretKeyRef:
      name: hf-token-secret
      key: token
```

**SGLang engine command with prefix caching enabled:**
```yaml
command: ["python", "-m", "sglang.launch_server"]
args:
  - "--model"
  - "meta-llama/Meta-Llama-3.1-8B-Instruct"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "8000"
  - "--enable-prefix-caching"   # activates RadixCache from Layer 12
```

---

## Router Deployment

The router pod needs no GPU. It is a Rust process with minimal resource requirements:

```yaml
resources:
  requests:
    cpu: "2"
    memory: "4Gi"
  limits:
    cpu: "4"
    memory: "8Gi"
```

**Router command — the key flags:**
```yaml
command: ["python", "-m", "sglang_router.launch_router"]
args:
  - "--service-discovery"
  - "--selector"
  - "app=sglang-worker"
  - "component=inference"
  - "--service-discovery-namespace"
  - "production"
  - "--service-discovery-port"
  - "8000"
  - "--policy"
  - "cache_aware"
  - "--cache-threshold"
  - "0.5"
  - "--balance-abs-threshold"
  - "32"
  - "--host"
  - "0.0.0.0"
  - "--port"
  - "30000"
  - "--prometheus-port"
  - "29000"
```

Note the absence of `--worker-urls` — this is the entire change from Layer 15's launch command. The router builds its worker list from the cluster.

**ServiceAccount attachment:**
```yaml
spec:
  serviceAccountName: sglang-router   # must match the RBAC objects
```

Without this, the pod runs as `default` which has no permission to list or watch pods. The K8s API call fails at startup and the router begins with zero workers.

**K8s Service for stable DNS:**
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-router
  namespace: production
spec:
  selector:
    app: sglang-router
  ports:
  - name: http
    port: 30000
    targetPort: 30000
  - name: metrics
    port: 29000
    targetPort: 29000
  type: ClusterIP
```

**Router readiness probe** — waits until at least one worker is registered:
```yaml
readinessProbe:
  httpGet:
    path: /readiness
    port: 30000
  initialDelaySeconds: 10
  periodSeconds: 5
```

---

## How Service Discovery Works

The discovery logic lives in `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs`. It does four things:

**1. Connect to the K8s API** (line 223):
```rust
let client = Client::try_default().await?;
```
`try_default()` detects the in-cluster service account token automatically. No kubeconfig needed.

**2. Open a watcher stream** (line 313):
```rust
let watcher_stream = watcher(pods.clone(), watcher_config).applied_objects();
```
This is a long-lived HTTP connection to the K8s API server. Events (pod Added, Modified, Deleted) stream in as they happen — not polled.

**3. Filter by label selector** (lines 88–96):
```rust
pod.metadata.labels
    .as_ref()
    .is_some_and(|labels| selector.iter().all(|(k, v)| labels.get(k) == Some(v)))
```
All key=value pairs in `--selector` must match. `app=sglang-worker component=inference` requires both labels.

**4. On each event, gate by health then add or remove** (lines 349–530):
```rust
if pod.metadata.deletion_timestamp.is_some() {
    handle_pod_deletion(...)    // submit Job::RemoveWorker
} else {
    handle_pod_event(...)       // submit Job::AddWorker if is_healthy()
}
```

`is_healthy()` (line 196) requires both `Ready` condition = `"True"` **and** pod phase = `"Running"`. A pod that is Pending, Terminating, or CrashLoopBackOff never enters the worker pool.

When a pod becomes Ready, the router builds its URL from the pod IP and the configured port (line 200–203):
```rust
pub fn worker_url(&self, port: u16) -> String {
    format!("http://{}:{}", self.ip, port)
}
```
The pod IP is used directly — no DNS lookup, no K8s Service in the path between router and worker.

**Reconnection:** if the K8s API connection drops, the watcher retries with exponential backoff (lines 308–393): 1s → 2s → 4s → … → 300s maximum. The router continues serving existing workers during reconnection. The `tracked_pods: HashSet` (line 274) deduplicates events when the watcher reconnects and replays history.

The full lifecycle of one worker pod:

```
kubectl apply -f worker-deployment.yaml
  → Pod scheduled on GPU node (Pending)
  → Model downloaded from PVC (ContainerCreating)
  → Engine starts, radix cache initialised
  → /health returns 200 → readiness probe passes
  → Pod condition Ready = True
  → watcher delivers Modified event (pod now Ready)
  → handle_pod_event() → is_healthy() = true
  → Job::AddWorker submitted → worker registered
  → Router starts sending traffic to this pod

kubectl scale deployment sglang-worker --replicas=0
  → Pod gets deletionTimestamp
  → watcher delivers Modified event (deletionTimestamp set)
  → handle_pod_deletion() → Job::RemoveWorker submitted
  → Worker removed from pool (in-flight requests finish normally)
  → Pod terminates
```

---

## Observability

The router emits metrics at `--prometheus-port 29000`. Mount a `ServiceMonitor` (Prometheus Operator) targeting that port with label `monitoring: enabled`.

**Discovery health — the new metrics in Layer 16:**
```promql
# Workers the router currently knows about (gauge)
smg_discovery_workers_discovered{source="kubernetes"}

# Registrations — watch for "failed" rising
sum by (result) (rate(smg_discovery_registrations_total{source="kubernetes"}[5m]))
```

**Routing latency and throughput:**
```promql
# Request rate by endpoint
sum(rate(smg_http_requests_total[5m])) by (path)

# P99 latency
histogram_quantile(0.99,
  sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
/ sum(rate(smg_http_responses_total[5m]))
```

**Worker health:**
```promql
# Healthy workers total — alert at 0
sum(smg_worker_pool_size)

# Per-worker in-flight requests (mirrors Worker.load from Layer 15)
smg_worker_requests_active

# Circuit breaker state: 0=Closed, 1=Open, 2=HalfOpen
smg_worker_cb_state
```

**Four alerts that must fire:**

```yaml
groups:
- name: layer16-k8s
  rules:
  - alert: NoHealthyWorkers
    expr: sum(smg_worker_pool_size) == 0
    for: 1m
    severity: critical
    annotations:
      summary: "Router has no healthy workers — all traffic failing"

  - alert: DiscoveryZeroWorkers
    expr: smg_discovery_workers_discovered{source="kubernetes"} == 0
    for: 2m
    severity: warning
    annotations:
      summary: "Discovery sees no pods — RBAC or label selector may be broken"

  - alert: HighErrorRate
    expr: >
      sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
      / sum(rate(smg_http_responses_total[5m])) > 0.05
    for: 5m
    severity: critical

  - alert: CircuitBreakerOpen
    expr: count(smg_worker_cb_state == 1) > 0
    for: 2m
    severity: warning
    annotations:
      summary: "At least one worker circuit breaker is open"
```

`DiscoveryZeroWorkers` is a Layer 16-specific alert — it catches misconfigured RBAC or label selectors before they cause `NoHealthyWorkers`. If discovery sees zero pods but worker pods exist, the ServiceAccount probably lacks the `list`/`watch` permission, or the `--selector` does not match the pods' labels.

---

## High Availability: Multiple Router Replicas and the Cache-Tree Problem

### Why 2 Replicas?

A single router replica is a single point of failure. If it crashes or the node it runs on fails, all traffic returns 503s for ~30–60 seconds until Kubernetes restarts the pod. Two replicas behind the ClusterIP Service eliminate this: if one crashes, the Service immediately routes 100% of traffic to the surviving replica. Zero client-visible downtime.

The router is stateless from the client's perspective (it forwards requests, it doesn't hold session data). Scaling replicas purely for fault tolerance is cheap — the router uses no GPU, minimal CPU, minimal RAM.

### Why the Replicas Do NOT Sync

Each router replica independently maintains its own **radix tree in memory**. There is no cross-pod synchronisation, no shared cache, no gossip between replicas. This is intentional — the complexity of distributed consensus across radix trees is not implemented in the current SGLang gateway.

```
Router Pod 1 (its own RAM):              Router Pod 2 (its own RAM):
  radix tree:                              radix tree:
  "You are a customer support..." → A       (empty, or diverged)
  "Summarise this document..."    → B
```

When the K8s Service round-robins between them, requests with the same prefix may land on different routers, routing to different workers, causing cache misses on prefixes that were already cached:

```
Request 1 → Router Pod 1 → knows prefix is on Worker A → cache hit ✓
Request 2 → Router Pod 2 → no idea → picks Worker B by load → cache miss ✗
                                       Worker B recomputes the full prefix
```

**The cost:** 10–20% cache hit rate reduction vs. a single router. The routing still works correctly — it just doesn't hit the KV cache as often as it could.

**The fix:** ensure requests that share a prefix always hit the **same router pod**. That router's radix tree will be warm for those requests, and it will consistently route them to the right worker.

### Option A — ClientIP Session Affinity (Simple)

The K8s Service hashes the client's IP to always pick the same router pod:

```yaml
# Router Service with session affinity
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600    # remember affinity for 1 hour
```

Works well when different clients have different IPs. Breaks down when many clients share an IP (corporate NAT, VPN gateway) — they all land on the same router pod, creating an imbalance.

### Option B — X-Session-ID Header (Recommended)

You define a **semantic routing key** in your application code — something that groups requests by their shared prefix. Send it as the `X-Session-ID` header. The ingress controller hashes this header to pick a consistent router pod, bypassing the K8s Service's default round-robin.

**What to use as the key:**

| Your Scenario | Good Routing Key | Why |
|---|---|---|
| All users share the same system prompt | `system_prompt_id` | All users hit the same router → same worker already has the system prompt cached |
| Per-user long conversations | `user_id` | Each user's conversation history stays on one router → one worker |
| Batch job processing the same document | `document_id` | All chunks of the doc go to one GPU → context prefix cached |
| Different tenants have different system prompts | `tenant_id` | Each tenant routes to a dedicated router shard |

**Ingress configuration (Nginx):**

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sglang-router
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_session_id"
    # ↑ hashes the X-Session-ID header value to pick the router pod consistently
spec:
  rules:
  - host: llm.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: sglang-router
            port:
              number: 30000
```

**Client code — how to populate X-Session-ID:**

```python
import openai

# Case 1: all users share the same system prompt
# → use the system prompt's ID as the routing key
# → all users land on the same router → same worker already has it cached
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": "system-prompt-customer-support-v2"}
)

# Case 2: per-user conversations
# → use user_id so each user's chat history stays on one router/worker
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": f"user-{user_id}"}
)

# Case 3: batch job over a shared document
# → all chunks route to the same GPU, document prefix cached once
client = openai.Client(
    base_url="http://llm.example.com",
    default_headers={"X-Session-ID": f"doc-{document_id}"}
)

# Then use the client normally — the header is sent automatically
response = client.chat.completions.create(
    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
    messages=[
        {"role": "system", "content": "You are a helpful customer support agent..."},
        {"role": "user",   "content": "My order hasn't arrived yet."}
    ]
)
```

**What happens inside the stack:**

```
Client sends X-Session-ID: "system-prompt-customer-support-v2"
  ↓
Nginx hashes "system-prompt-customer-support-v2" → always picks Router Pod 1
  ↓
Router Pod 1 receives request
Its radix tree: "You are a helpful customer support agent..." → Worker A
  ↓
Router sends to Worker A → KV cache hit (system prompt already cached)
  ↓
Low latency response ✓
```

All future requests with the same `X-Session-ID` go to Router Pod 1, which consistently routes them to Worker A. If Router Pod 1 crashes, Nginx reassigns those clients to Router Pod 2 — brief cold-cache period while Pod 2's tree warms up, then back to normal.

### Option C — Single Router with Tight Liveness Probe

```yaml
spec:
  replicas: 1
livenessProbe:
  periodSeconds: 5
  failureThreshold: 2    # restart after 10 seconds of failure
```

Total crash-to-recovery time: ~20 seconds. Acceptable for internal tooling; not for user-facing APIs.

### Decision Guide

| Scenario | Recommended Setup |
|---|---|
| Development / testing | `replicas: 1` — keep it simple |
| Internal batch workload | `replicas: 1` + tight liveness probe |
| External API, heterogeneous clients | `replicas: 2` + `sessionAffinity: ClientIP` |
| You control client code, same prefix groups | `replicas: 2` + `X-Session-ID` header at Nginx |
| Maximum cache efficiency is the SLO | `replicas: 1` — single radix tree, no cache dilution |

**Horizontal Pod Autoscaler for workers:**

Worker pods can be autoscaled independently of the router. The router discovers new pods automatically via the watcher — no manual configuration change needed when pods are added or removed:

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-worker-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sglang-worker
  minReplicas: 2
  maxReplicas: 16
  metrics:
  - type: Pods
    pods:
      metric:
        name: smg_worker_requests_active   # from Prometheus adapter
      target:
        type: AverageValue
        averageValue: "20"                 # scale when avg in-flight > 20
```

When the HPA adds a new worker pod, the pod becomes Ready → watcher fires → `Job::AddWorker` submitted → router starts routing to the new pod within seconds. No restart, no config change.

---

## What Layer 16 Does Not Cover

**Router-to-router radix tree synchronisation:** the cache-tree consistency problem (§ High Availability) is an open problem. Synchronising three router replicas' radix trees in real time requires consensus or gossip — not implemented in the current SGLang gateway. The gossip mesh (`smg_mesh`) infrastructure exists in the codebase but is not wired to the radix trees.

**Global live KV cache index:** llm-d (Red Hat) solves a related problem differently — worker pods emit `KVCacheUpdatedEvent` and `KVCacheEvictedEvent` to the router, giving it an exact view of what each pod has cached. This requires changes to the inference engine. It is a separate architecture from the router-side radix tree used here.

**Cross-region routing:** SkyWalker (EUROSYS 2026) extends prefix-aware routing to multi-region deployments with KV cache block transfer between replicas across regions. The principles are the same; the infrastructure is substantially more complex.

**Helm chart packaging:** the YAML manifests in `k8s_manifests/` are flat files. A production team would typically package them as a Helm chart (see `REPOS/vllm/examples/online_serving/chart-helm/` for a complete vLLM Helm chart as a reference). Helm adds templating, values files, and upgrade lifecycle management — useful but outside the scope of this layer.
