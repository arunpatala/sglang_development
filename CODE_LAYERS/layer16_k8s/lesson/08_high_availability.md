# 08 — High Availability

## What This Section Covers

With a single router pod, the system has a **single point of failure**: if that pod crashes or the node it runs on fails, 100% of traffic fails until Kubernetes restarts the pod (typically 30–60 seconds of downtime).

This section covers:
1. The naive fix: just run two routers (and why it introduces a new problem)
2. The cache-tree consistency problem
3. The practical options and their tradeoffs
4. Horizontal Pod Autoscaler (HPA) for workers

---

## Why One Router Is Risky

Kubernetes restarts crashed pods, but restart takes time:

```
Router pod crashes
  │
  ▼ (Kubernetes detects failure, typically 5-30s depending on liveness probe)
  ▼
  New pod scheduled and starts
  │
  ▼ (initialDelaySeconds for probes, ~5s for router)
  ▼
  Pod becomes Ready
  │
  ▼ (Service starts routing to new pod)

Total downtime: 30-60 seconds typical
```

For a production LLM API, 30–60 seconds of downtime is unacceptable. Any automated retries or clients see 503s for that entire window.

---

## The Naive Fix: Two Router Replicas

The obvious solution is to run two router pods:

```yaml
spec:
  replicas: 2
```

With two replicas, the Kubernetes Service load-balances across both. If one crashes, the Service immediately directs all traffic to the surviving pod. Zero client downtime.

**But this creates a new problem.**

---

## The Cache-Tree Consistency Problem

Each router pod has its own **independent radix tree** in memory. The radix tree is the data structure that maps "which prompt prefixes have been cached on which worker". It grows as requests come in.

```
Router Pod 1 (radix tree):
  "You are a helpful assistant. User: What is..." → Worker A
  "You are a helpful assistant. User: Explain..." → Worker B
  "System prompt: Always respond in JSON..." → Worker C

Router Pod 2 (radix tree):
  empty (just started)
  OR diverged state (different requests hit each router)
```

When the Kubernetes Service load-balances between Router Pod 1 and Router Pod 2, requests with the same prefix may land on different routers, which have different views of where those prefixes are cached:

```
Request: "You are a helpful assistant. User: What is the capital of France?"
  ↓ K8s Service picks Router Pod 1 (50% chance)
  → Router Pod 1: radix tree knows this prefix is on Worker A → routes to Worker A ✓ cache hit

Same request again:
  ↓ K8s Service picks Router Pod 2 (50% chance)
  → Router Pod 2: radix tree is empty → routes to Worker B (random/load-based)
  → Worker B: cache miss, must recompute the system prompt prefix
  → Higher latency, reduced cache hit rate
```

**The consequence:** with N router replicas, the effective cache hit rate drops because the same prefix gets split across multiple routers' routing decisions. In the worst case (all routers empty, all requests first-time), the prefix cache affinity routing provides no benefit — you're paying for the cache-aware routing complexity but getting round-robin behavior in terms of cache utilization.

This is called the **cache-tree consistency problem**: there is no single source of truth for which worker has cached which prefix. Each router has an independent, eventually-stale view.

---

## Option 1: Session Affinity at the K8s Service Level

The simplest fix is to use Kubernetes session affinity to ensure each client always goes to the same router:

```yaml
# In the Router Service
spec:
  sessionAffinity: ClientIP        # route requests from same client IP to same router pod
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600         # remember affinity for 1 hour
```

```bash
kubectl patch svc sglang-router -n production \
  --type='json' \
  -p='[{"op": "add", "path": "/spec/sessionAffinity", "value": "ClientIP"}]'
```

**What this does:** The K8s Service uses the client's IP address to deterministically pick the same router pod every time. Client A always goes to Router Pod 1, Client B always goes to Router Pod 2.

**Benefit:** Each router's radix tree stays consistent for its assigned clients. Cache hit rates are preserved.

**Limitation:** If Router Pod 1 crashes, clients previously assigned to it get reassigned to Router Pod 2 — which has a cold cache for their requests. The cache warms up over the next few minutes.

**Limitation 2:** Client IP affinity doesn't work well behind a NAT (many clients appear as the same IP) or when clients come from many different IPs but repeatedly send the same prompts.

```yaml
# Full router service with session affinity
# k8s_manifests/06_router_service.yaml (updated)
apiVersion: v1
kind: Service
metadata:
  name: sglang-router
  namespace: production
spec:
  selector:
    app: sglang-router
  ports:
  - name: inference
    port: 30000
    targetPort: 30000
  type: ClusterIP
  sessionAffinity: ClientIP          # ← add this
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

---

## Option 2: Sticky Sessions at the Application Level

Instead of IP-based affinity, route based on the request content (which session or user is making the request). Clients include a header like `X-Session-ID` or `X-User-ID`. A reverse proxy or ingress controller uses this header to route consistently.

For example, with Nginx Ingress:

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sglang-router
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_session_id"
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

This hashes the `X-Session-ID` header value and consistently routes to the same upstream (router pod). Works across NAT, more granular than IP-based.

**Clients must include the header:**

```python
import openai
client = openai.Client(
    base_url="http://llm.example.com",
    api_key="...",
    default_headers={"X-Session-ID": "user-12345"}
)
```

---

## Option 3: One Router, High-Quality Liveness Probe (Pragmatic Choice)

For many deployments, one router with a well-configured liveness probe is acceptable:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 30000
  initialDelaySeconds: 10
  periodSeconds: 5         # check every 5 seconds
  failureThreshold: 2      # restart after 2 consecutive failures (10 seconds)
  successThreshold: 1
```

If the router hangs or crashes, Kubernetes detects it within 10 seconds and restarts it. The new router reconnects to the K8s API and discovers workers within a few seconds. Total downtime: ~15-30 seconds.

For internal or batch workloads (not real-time APIs), this is often fine. The cache hit rate is optimal (single radix tree), and the cost is simpler configuration.

---

## Option 4: Two Replicas + Session Affinity (Recommended for Production)

The recommended production configuration combines two router replicas with ClientIP session affinity:

```yaml
# Router Deployment with 2 replicas
spec:
  replicas: 2
```

```yaml
# Router Service with session affinity
spec:
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600
```

**What you get:**
- If one router pod crashes: the surviving pod immediately takes all traffic (zero client-visible downtime). The surviving router's radix tree is warm for its assigned clients; reassigned clients experience a brief cache cold-start.
- Rolling deployments: update one router at a time, zero downtime.
- Load distribution: roughly half of clients go to each router.

**What you give up:**
- Requests from the same client that happen to route to the other router (e.g., after a failover) temporarily have cold cache. This increases latency for a few requests while the cache warms up.
- Two routers use 2x the CPU and memory (still cheap compared to GPU workers).

---

## Part 2: Autoscaling Workers (HPA)

The **Horizontal Pod Autoscaler (HPA)** automatically adjusts the number of worker replicas based on metrics.

### Basic CPU-Based HPA

```yaml
# k8s_manifests/09_hpa.yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-worker-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sglang-worker           # which deployment to scale
  minReplicas: 2                  # never go below 2 workers
  maxReplicas: 8                  # never exceed 8 workers (8 GPUs limit)
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70    # scale up when average CPU > 70%
```

```bash
kubectl apply -f k8s_manifests/09_hpa.yaml

# Check HPA status
kubectl get hpa -n production
# NAME                  REFERENCE              TARGETS   MINPODS   MAXPODS   REPLICAS
# sglang-worker-hpa     Deployment/sglang-worker  45%/70%   2         8         2
```

**Limitation:** CPU is a poor metric for GPU LLM workloads. The GPU is at 100% while the CPU is idle (tokenization is fast). Scaling on CPU means the HPA never triggers during actual GPU saturation.

### Better: Custom Metrics-Based HPA

Scale based on the number of in-flight requests or queue length — actual load signals:

```yaml
# HPA using Prometheus custom metrics (requires prometheus-adapter)
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sglang-worker-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sglang-worker
  minReplicas: 2
  maxReplicas: 8
  metrics:
  - type: External
    external:
      metric:
        name: sglang_num_running_reqs  # from worker Prometheus metrics
        selector:
          matchLabels:
            namespace: production
      target:
        type: AverageValue
        averageValue: "10"    # scale up when avg running requests per pod > 10
```

**Setting up Prometheus Adapter** (needed to expose Prometheus metrics to K8s HPA):

```bash
helm install prometheus-adapter \
  prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --set prometheus.url=http://prometheus-kube-prometheus-prometheus.monitoring.svc
```

---

## Part 3: Pod Disruption Budget

A **PodDisruptionBudget (PDB)** prevents Kubernetes from taking down too many pods at once during voluntary disruptions (node maintenance, rolling upgrades):

```yaml
# k8s_manifests/10_pdb.yaml
---
# Ensure at least 1 worker is always available
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: sglang-worker-pdb
  namespace: production
spec:
  minAvailable: 1               # always keep at least 1 worker pod running
  selector:
    matchLabels:
      app: sglang-worker

---
# Ensure the router is always available
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: sglang-router-pdb
  namespace: production
spec:
  minAvailable: 1               # always keep at least 1 router running
  selector:
    matchLabels:
      app: sglang-router
```

```bash
kubectl apply -f k8s_manifests/10_pdb.yaml
```

Without a PDB, a `kubectl drain` command (used for node maintenance) might take down all your worker pods simultaneously if they happen to be on the same node. The PDB tells the drain process to proceed one pod at a time, ensuring minimum availability.

---

## Part 4: Anti-Affinity Rules

Spread pods across nodes to reduce correlated failures (if a node goes down, you don't lose all your replicas):

```yaml
# In the worker Deployment's pod spec
spec:
  affinity:
    podAntiAffinity:
      preferredDuringSchedulingIgnoredDuringExecution:  # soft constraint (try to, but ok if can't)
      - weight: 100
        podAffinityTerm:
          labelSelector:
            matchLabels:
              app: sglang-worker
          topologyKey: kubernetes.io/hostname    # spread across different nodes
```

**`preferredDuring...`** (soft): Kubernetes tries to spread pods across nodes, but will violate the rule if there are insufficient nodes.

**`requiredDuring...`** (hard): Kubernetes NEVER schedules two worker pods on the same node. Use this if you have enough GPU nodes and want a hard guarantee.

```yaml
# Hard anti-affinity (only use if you have enough GPU nodes)
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: sglang-worker
        topologyKey: kubernetes.io/hostname
```

**For routers:** Same pattern. Two router replicas should run on different nodes:

```yaml
# In the router Deployment
spec:
  affinity:
    podAntiAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
      - labelSelector:
          matchLabels:
            app: sglang-router
        topologyKey: kubernetes.io/hostname
```

---

## Summary: HA Configuration Checklist

```
Workers:
  ✓ replicas: 2 (minimum 2, so one can restart without downtime)
  ✓ PodDisruptionBudget: minAvailable: 1
  ✓ Anti-affinity: prefer different nodes
  ✓ HPA: minReplicas: 2, maxReplicas: N

Router:
  ✓ replicas: 2
  ✓ Session affinity: ClientIP (preserve cache-tree effectiveness)
  ✓ PodDisruptionBudget: minAvailable: 1
  ✓ Anti-affinity: require different nodes (routers are cheap, use hard constraint)
  ✓ Liveness probe: periodSeconds: 5, failureThreshold: 2 (fast detection)
```

---

## Decision Guide: How Many Router Replicas?

| Scenario | Recommended Setup |
|---|---|
| Development / testing | 1 replica, no session affinity. Keep it simple. |
| Internal batch workload, tolerate 30s downtime | 1 replica, tight liveness probe (5s period, 2 failures) |
| External API, zero-downtime requirement | 2 replicas + ClientIP session affinity |
| Very high traffic, many client IPs | 2-3 replicas + application-level sticky sessions (header-based) |

The cache-tree consistency problem is real but manageable. Session affinity reduces it to near-zero for the common case. The cost is that failover causes a temporary cache cold-start for affected clients, which degrades latency for a few requests until the surviving router's cache warms up again.

For most deployments, **2 replicas + ClientIP session affinity** is the right balance of simplicity and reliability.
