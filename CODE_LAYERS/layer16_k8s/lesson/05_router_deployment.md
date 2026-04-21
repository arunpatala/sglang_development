# 05 — Router Deployment YAML

## What This Section Covers

The router is the gateway process — the same `sglang_router.launch_router` from Layer 15, now running in a Kubernetes pod. Unlike workers, it needs no GPU, uses much less memory, and needs the ServiceAccount we set up in Section 03 so it can call the Kubernetes API to discover workers.

We'll build the complete YAML piece by piece, then show the Service and the final verified deployment.

---

## Part 1: Router Deployment Metadata

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-router
  namespace: production
  labels:
    app: sglang-router
spec:
  replicas: 1    # start with 1; see Section 08 for why 2+ is tricky
```

Start with 1 replica. Running multiple replicas of the router introduces the "cache-tree consistency problem" (covered in Section 08). For initial deployment, one router replica is correct.

---

## Part 2: The ServiceAccount — Critical Difference From Workers

```yaml
spec:
  selector:
    matchLabels:
      app: sglang-router
  template:
    metadata:
      labels:
        app: sglang-router
    spec:
      serviceAccountName: sglang-router    # ← THE KEY DIFFERENCE
```

This single line is what gives the router permission to call the Kubernetes API. Without it, the router runs under the `default` ServiceAccount, which has no permissions, and the service discovery code fails with a 403 Forbidden error when trying to list pods.

Worker pods do NOT need a ServiceAccount (they use the default). Only the router needs API access.

---

## Part 3: Container Image and Command

```yaml
      containers:
      - name: router
        image: lmsysorg/sglang:latest
        imagePullPolicy: Always

        command: ["python", "-m", "sglang_router.launch_router"]
        args:
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "30000"              # main inference routing port
          - "--metrics-port"
          - "29000"              # Prometheus metrics port (covered in Section 07)
          - "--policy"
          - "cache_aware"        # prefix-cache-aware routing from Layer 15
          - "--service-discovery"         # enable K8s service discovery (disables --worker-urls)
          - "--service-discovery-port"
          - "8000"               # port to connect to on each discovered worker pod
          - "--selector"
          - "app=sglang-worker"
          - "--selector"
          - "component=inference"   # only route to pods with BOTH labels
          - "--service-discovery-namespace"
          - "production"            # only watch pods in this namespace
```

**`--service-discovery`**: Turns on the Kubernetes watcher mode. When this flag is present, `--worker-urls` is ignored (they're mutually exclusive). The router starts with zero workers and waits for the K8s watcher to deliver the initial pod list.

**`--policy cache_aware`**: The same routing algorithm from Layer 15. The service discovery mechanism just updates the *list of workers* the policy operates over. The policy itself is unchanged.

**`--selector app=sglang-worker --selector component=inference`**: Each `--selector` argument adds one label requirement. A pod must satisfy all of them. You can add as many as needed (e.g. `model=llama-3-8b` to route only to pods serving that model).

**`--service-discovery-namespace production`**: The router only watches pods in the `production` namespace. This prevents accidental cross-namespace routing if you have the same label on dev/staging pods.

---

## Part 4: Resource Requests

```yaml
        resources:
          requests:
            memory: "512Mi"     # router is lightweight: no model weights
            cpu: "1"            # one core for request handling
          limits:
            memory: "2Gi"
            cpu: "4"
```

The router is a proxy process — it does not hold model weights in memory. Its memory usage is dominated by the radix trie (which grows with request history) and the connection pool to workers. 512MB request with a 2GB limit is generous.

No GPU. No toleration for GPU taints. The router should run on a CPU node, leaving GPU nodes available for workers.

---

## Part 5: Probes

```yaml
        readinessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 5    # router starts in seconds, not minutes
          periodSeconds: 3

        livenessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
```

The router's `initialDelaySeconds` is much lower than the worker's (5 vs 120) because the router starts in a few seconds — there's no model to load. The `/health` endpoint on port 30000 returns 200 as soon as the router is listening.

**Important subtlety:** The router becomes `Ready` (readiness probe passes) quickly, before any workers are discovered. During the discovery window (a few seconds after the router starts), the router may return 503s for inference requests because it has no workers yet. This is normal and brief.

---

## Part 6: Port Declarations

```yaml
        ports:
        - name: inference
          containerPort: 30000    # client requests go here
        - name: metrics
          containerPort: 29000    # Prometheus scrapes this (Section 07)
```

---

## Full Router Deployment YAML

```yaml
# k8s_manifests/05_router_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-router
  namespace: production
  labels:
    app: sglang-router
spec:
  replicas: 1
  selector:
    matchLabels:
      app: sglang-router
  template:
    metadata:
      labels:
        app: sglang-router
      annotations:
        prometheus.io/scrape: "true"    # tells Prometheus to scrape this pod
        prometheus.io/port: "29000"
        prometheus.io/path: "/metrics"
    spec:
      serviceAccountName: sglang-router   # RBAC identity (Section 03)

      containers:
      - name: router
        image: lmsysorg/sglang:latest
        imagePullPolicy: Always

        command: ["python", "-m", "sglang_router.launch_router"]
        args:
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "30000"
          - "--metrics-port"
          - "29000"
          - "--policy"
          - "cache_aware"
          - "--service-discovery"
          - "--service-discovery-port"
          - "8000"
          - "--selector"
          - "app=sglang-worker"
          - "--selector"
          - "component=inference"
          - "--service-discovery-namespace"
          - "production"

        ports:
        - name: inference
          containerPort: 30000
        - name: metrics
          containerPort: 29000

        resources:
          requests:
            memory: "512Mi"
            cpu: "1"
          limits:
            memory: "2Gi"
            cpu: "4"

        readinessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 5
          periodSeconds: 3

        livenessProbe:
          httpGet:
            path: /health
            port: 30000
          initialDelaySeconds: 10
          periodSeconds: 10
          failureThreshold: 3
```

---

## Part 7: The Router Service

The Deployment creates the router pod. The Service creates a stable endpoint that clients use to reach it:

```yaml
# k8s_manifests/06_router_service.yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-router
  namespace: production
  labels:
    app: sglang-router
spec:
  selector:
    app: sglang-router         # forwards to pods with this label
  ports:
  - name: inference
    port: 30000                # port clients connect to
    targetPort: 30000          # port the pod is listening on
  - name: metrics
    port: 29000
    targetPort: 29000
  type: ClusterIP              # only accessible within the cluster
```

**DNS name inside the cluster:** `sglang-router.production.svc.cluster.local:30000`

Any pod inside the cluster (e.g. your application pods) can send requests here. K8s resolves this hostname to the ClusterIP and load-balances across all healthy router pods.

**For external access (optional):** Change `type: ClusterIP` to `type: LoadBalancer` (creates a cloud load balancer) or `type: NodePort` (exposes on every node's IP at a fixed port). For production, prefer an Ingress controller with TLS termination.

---

## Part 8: Making It Externally Accessible

If you need to call the router from outside the cluster (e.g. from your laptop for testing):

**Option A: NodePort (simple, for testing)**

```yaml
spec:
  type: NodePort
  ports:
  - name: inference
    port: 30000
    targetPort: 30000
    nodePort: 30000            # fixed port on every node's IP
```

```bash
# Get any node's IP
kubectl get nodes -o wide | awk '{print $7}' | head -2

# Call the router
curl http://<node-ip>:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3.1-8B-Instruct", "messages": [{"role": "user", "content": "hello"}]}'
```

**Option B: kubectl port-forward (for development)**

```bash
# Forward the router service to your local machine
kubectl port-forward svc/sglang-router -n production 30000:30000

# Now you can call the router as if it's running locally
curl http://localhost:30000/v1/chat/completions ...
```

This is the simplest way to test the cluster from your laptop without modifying the Service type.

---

## Deploying the Router

```bash
# Apply deployment and service
kubectl apply -f k8s_manifests/05_router_deployment.yaml
kubectl apply -f k8s_manifests/06_router_service.yaml

# Watch router pod start (should be ready in ~10 seconds)
kubectl get pods -n production -l app=sglang-router -w

# NAME                       READY   STATUS    RESTARTS
# sglang-router-abc123       0/1     Running   0
# sglang-router-abc123       1/1     Running   0    ← ready in ~5 seconds

# Check router logs — look for worker discovery messages
kubectl logs -n production -l app=sglang-router --follow

# Expected log output:
# [INFO] Service discovery enabled, watching namespace: production
# [INFO] Found 2 pods matching selector app=sglang-worker,component=inference
# [INFO] Adding worker: 10.244.2.14:8000
# [INFO] Adding worker: 10.244.3.7:8000
# [INFO] Router ready. Workers: 2
```

---

## Verifying the Full Stack

At this point, workers and router are deployed. Verify end-to-end:

```bash
# 1. Check both workers are Running and Ready
kubectl get pods -n production -l app=sglang-worker
# Both should show 1/1 READY

# 2. Check router is Running and Ready
kubectl get pods -n production -l app=sglang-router
# Should show 1/1 READY

# 3. Check router service exists
kubectl get svc -n production sglang-router

# 4. Port-forward and send a test request
kubectl port-forward svc/sglang-router -n production 30000:30000 &

curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Meta-Llama-3.1-8B-Instruct",
    "messages": [
      {"role": "user", "content": "What is 2+2?"}
    ],
    "max_tokens": 50
  }'

# 5. Check worker discovery in router logs
kubectl logs -n production -l app=sglang-router | grep "Adding worker"

# 6. Test scaling: add a third worker and watch it get discovered
kubectl scale deployment sglang-worker -n production --replicas=3
kubectl logs -n production -l app=sglang-router -f
# Should see: "Adding worker: 10.244.X.X:8000" within ~10 seconds of the new pod becoming Ready
```

---

## What Happens When a Worker Pod Dies

```bash
# Simulate a worker failure by deleting a pod
kubectl delete pod sglang-worker-abc12 -n production

# The Deployment restarts it immediately (that's the point of Deployments)
# Watch the router logs:
kubectl logs -n production -l app=sglang-router -f
# Should see:
# [WARN] Worker 10.244.2.14:8000 removed (pod deleted/terminating)
# [INFO] Worker pool: 1 worker(s)
# ... (pod restarts, gets new IP, becomes Ready)
# [INFO] Adding worker: 10.244.2.99:8000
# [INFO] Worker pool: 2 worker(s)
```

The router handles this automatically. No manual intervention, no config changes, no router restart needed.

---

## Summary of All Files Applied So Far

```
k8s_manifests/
├── 00_namespace.yaml         → Namespace: production
├── 01_rbac.yaml              → ServiceAccount + Role + RoleBinding for router
├── 02_secret.yaml            → HuggingFace token secret
├── 03_pvc.yaml               → PersistentVolumeClaim for model weights
├── 04_worker_deployment.yaml → GPU worker Deployment (replicas: 2)
├── 05_router_deployment.yaml → CPU router Deployment (replicas: 1)
└── 06_router_service.yaml    → ClusterIP Service for router
```

Apply order matters:

```bash
kubectl apply -f k8s_manifests/00_namespace.yaml
kubectl apply -f k8s_manifests/01_rbac.yaml
kubectl apply -f k8s_manifests/02_secret.yaml
kubectl apply -f k8s_manifests/03_pvc.yaml
kubectl apply -f k8s_manifests/04_worker_deployment.yaml
kubectl apply -f k8s_manifests/05_router_deployment.yaml
kubectl apply -f k8s_manifests/06_router_service.yaml
```

Or apply everything in one command if you trust the order (K8s processes dependencies eventually):

```bash
kubectl apply -f k8s_manifests/
```
