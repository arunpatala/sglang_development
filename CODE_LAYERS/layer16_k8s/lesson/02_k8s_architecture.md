# 02 — The Architecture in Kubernetes

## The Big Picture

In Layer 15, everything ran on one machine:

```
Client
  ↓
Router (port 8200)     ← router.py
  ↓           ↓
Engine A    Engine B   ← server.py instances
(port 8114) (port 8115)
```

In Layer 16, the same logical structure runs in Kubernetes, but each component is a separate Deployment and the routing uses Kubernetes networking:

```
Client (outside the cluster)
  ↓
[Ingress / LoadBalancer Service]        ← public entry point, optional
  ↓
[K8s Service: sglang-router]            ← stable hostname, ClusterIP
  ↓  ↓  ↓                              ← load balanced across router pods
[Router Pod 1] [Router Pod 2]           ← Deployment: sglang-router
    │               │
    └───────────────┘
            │  watches K8s API, discovers worker pods by label
            ↓
[Worker Pod A] [Worker Pod B] [Worker Pod C] [Worker Pod D]
← Deployment: sglang-worker (GPU pods) ←
            │
            ↓
[PVC: model-cache]                      ← persistent volume, shared across pods
```

Let's walk through each component from bottom to top.

---

## Component 1: The PVC (Persistent Volume Claim)

**What it is:** A request for storage that survives pod restarts.

**Why you need it:** The model weights (e.g. Llama-3 8B = ~16GB of files) are large. If you download them from HuggingFace every time a pod starts, startup takes 10–30 minutes and burns network bandwidth. A PVC downloads the weights once and keeps them on disk.

```yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sglang-models-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce      # one node can mount this at a time
  resources:
    requests:
      storage: 50Gi      # enough for one 8B model in fp16 + overhead
  storageClassName: standard
```

Worker pods mount this at `/root/.cache/huggingface` — the default HuggingFace cache directory. The engine finds the model files there without downloading anything.

**One PVC per node** is the simple setup. If all worker pods run on the same node (common for small clusters), they share one PVC. For multi-node setups, you need either `ReadWriteMany` (requires a network filesystem like NFS or EFS) or one PVC per node with `ReadWriteOnce`.

---

## Component 2: Worker Pods (GPU pods)

**What they are:** The actual inference engines — `server.py` from Layer 15, packaged into a container.

**What they do:** Accept requests, run the LLM forward pass, return completions. Each pod runs on a GPU node and requests one GPU.

**Key label — the discovery contract:**
```yaml
labels:
  app: sglang-worker        # the router matches on this
  component: inference      # and this
  model: llama-3-8b        # informational, not required for discovery
```

These labels are how the router finds the workers. Any pod without these labels is invisible to the router.

**The readiness probe — the health gate:**
```yaml
readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 120
  periodSeconds: 5
```

A pod only becomes "Ready" (and only gets added to the router's worker list) after the readiness probe passes. During model loading (which can take 1–5 minutes), the pod is Running but not Ready — the router ignores it. This prevents the router from sending requests to a pod that is still warming up.

**How many workers:** Start with as many GPUs as you have. Each pod gets one GPU. The HPA (Horizontal Pod Autoscaler) can add more automatically based on load.

---

## Component 3: The Router Pods (CPU pods)

**What they are:** The `sglang-router` process — the same routing logic from Layer 15, now running in a container.

**What they do:** Accept incoming requests, pick a worker using the routing policy (cache-aware, round-robin, etc.), forward the request, return the response.

**No GPU needed.** The router holds no model weights, does no computation. It is a lightweight Rust/Python process that proxies HTTP requests. A small CPU pod with 2 cores and 4GB RAM is sufficient for thousands of requests per minute.

**Key difference from Layer 15:** instead of reading worker URLs from `config.yml`, the router connects to the Kubernetes API and watches for pods with the right labels.

```
Router pod starts
  → connects to K8s API
  → lists all Running+Ready pods with labels app=sglang-worker,component=inference
  → adds each one as a worker
  → watches for future changes (new pods, deleted pods)
  → forwards requests using cache-aware policy
```

**Why run multiple router replicas:** one router pod is a single point of failure. If it crashes or the node it runs on fails, all traffic stops until Kubernetes restarts it (~30 seconds). Two replicas means one can fail without any downtime. Three replicas gives rolling updates with no downtime.

---

## Component 4: K8s Services

A **Service** is a stable DNS name that points to a set of pods. Pods have ephemeral IPs; Services have stable IPs and hostnames.

**Router Service** — what clients call:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-router
  namespace: production
spec:
  selector:
    app: sglang-router       # routes to pods with this label
  ports:
  - name: http
    port: 30000
    targetPort: 30000
  type: ClusterIP            # accessible inside the cluster only
```

DNS name: `sglang-router.production.svc.cluster.local:30000`

Any pod inside the cluster can reach the router at that hostname. Kubernetes automatically load-balances across all healthy router pods.

**Worker pods do NOT have a Service in front of them for routing.** The router connects directly to each worker pod's IP. This is intentional: a K8s Service between the router and workers would hide the individual pod IPs and make cache-aware routing impossible (the router needs to know which specific pod it's talking to for the radix tree to work correctly).

> **Key insight:** Router → K8s Service → Workers would break prefix-cache routing. The router needs direct pod-IP connections to each worker.

**Metrics Service** (separate, for Prometheus scraping):
```yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-router-metrics
  namespace: production
  labels:
    monitoring: enabled      # Prometheus ServiceMonitor selector
spec:
  selector:
    app: sglang-router
  ports:
  - name: metrics
    port: 29000
    targetPort: 29000
  type: ClusterIP
```

---

## How Kubernetes Networking Works (Basics)

Every pod gets a cluster-internal IP address (e.g. `10.244.2.14`). Pods can reach each other by IP. A pod in namespace `production` can connect directly to `10.244.2.14:8000` on another pod in the same (or different) namespace.

The Kubernetes CNI (Container Network Interface) plugin — Flannel, Calico, or Cilium depending on your cluster — handles the routing so these IPs work across nodes.

```
Node 1 (GPU)          Node 2 (GPU)          Node 3 (CPU)
┌──────────────┐      ┌──────────────┐      ┌──────────────┐
│ Worker Pod A │      │ Worker Pod B │      │ Router Pod   │
│ 10.244.2.14  │      │ 10.244.3.7   │      │ 10.244.1.2   │
│ :8000        │      │ :8000        │      │ :30000       │
└──────────────┘      └──────────────┘      └──────────────┘
       ↑                      ↑                      │
       └──────────────────────┴──────────────────────┘
              (direct pod-to-pod IP connections)
```

The router running on Node 3 connects directly to `10.244.2.14:8000` and `10.244.3.7:8000`. No Service in between. The Kubernetes network layer handles cross-node connectivity transparently.

---

## Namespaces

A **namespace** is a logical partition inside a Kubernetes cluster. Different teams or environments (dev, staging, production) use different namespaces to isolate their resources.

In Layer 16, everything runs in the `production` namespace:
- Worker Deployment
- Router Deployment
- Services
- PVC
- RBAC objects (Role, RoleBinding, ServiceAccount)

The router is configured with `--service-discovery-namespace production`, which tells it to only watch pods in that namespace. This prevents it from accidentally routing to a dev or staging worker running in a different namespace.

---

## The Namespace and Role Boundary

The RBAC system enforces that the router can only *read* pods, and only in its own namespace:

```
Router Pod (namespace: production)
  → has ServiceAccount: sglang-router
  → ServiceAccount has RoleBinding → Role (namespace: production)
  → Role allows: GET, LIST, WATCH on pods in production namespace only
  → Cannot see pods in: default, staging, kube-system
  → Cannot modify pods anywhere
```

If you need the router to watch workers in a different namespace, you need a `ClusterRole` + `ClusterRoleBinding` (cluster-wide permissions). Avoid this — it gives the router visibility into every pod in the cluster.

---

## Putting It All Together: Request Flow

Here is what happens when a client sends a request in the Layer 16 setup:

```
1. Client sends: POST http://sglang-router.production.svc.cluster.local:30000/v1/chat/completions

2. K8s DNS resolves "sglang-router.production.svc.cluster.local" → ClusterIP of the Service

3. K8s Service forwards to one of the healthy Router Pods (e.g. Router Pod 1 at 10.244.1.2)

4. Router Pod 1 receives the request.
   Its radix tree shows Worker Pod A (10.244.2.14) has the best prefix match.
   Worker A's load is 3; Worker B's load is 8. Load is balanced (diff < 32).
   Policy: route to Worker A.

5. Router opens direct HTTP connection to 10.244.2.14:8000 (Worker Pod A's IP).
   Forwards the full request.

6. Worker Pod A processes the request:
   - RadixCache hits on the system prompt prefix (KV cache reuse)
   - Generates completion tokens
   - Returns JSON response

7. Router receives response from Worker A.
   Forwards it back to the client.
   Decrements Worker A's in-flight counter.
   Inserts the prompt into the radix tree for Worker A.

Total router overhead: ~0.5ms (same as Layer 15, unchanged)
```

---

## What Gets Added Compared to Layer 15

| Component | Layer 15 | Layer 16 |
|---|---|---|
| Worker processes | `python -m sglang.launch_server` on localhost | Kubernetes Deployment (GPU pods) |
| Router process | `python -m sglang_router.launch_router` on localhost | Kubernetes Deployment (CPU pods) |
| Worker discovery | Static URLs in `config.yml` | K8s service discovery by pod label |
| Worker address | `localhost:8114`, `localhost:8115` | Pod IPs from K8s API |
| Stable client endpoint | `localhost:8200` | K8s ClusterIP Service |
| Health checking | Router polls `/health` every 10s | Router polls + K8s readiness probe (both) |
| Scaling | Manual: start another process | `kubectl scale` or HPA |
| Model weights | Local file on disk | PVC mounted in each pod |
| Permissions | None (same machine) | RBAC ServiceAccount + Role |

The routing algorithm, the radix trie, the two-guard strategy, the `WorkerLoadGuard` pattern — all unchanged. The infrastructure around them changes.
