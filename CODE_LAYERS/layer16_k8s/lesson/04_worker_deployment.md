# 04 — Worker Deployment YAML

## What This Section Covers

The worker pods are the GPU pods running the SGLang inference engine. This is the Kubernetes Deployment that replaces manually running `python -m sglang.launch_server` on a GPU machine. We'll build the YAML piece by piece, explaining each part, then show the full file at the end.

---

## Part 1: The PVC (Model Storage)

Before the worker Deployment, create the storage for model weights:

```yaml
# k8s_manifests/03_pvc.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: sglang-models-pvc
  namespace: production
spec:
  accessModes:
    - ReadWriteOnce    # one node mounts at a time (fine for single-node GPU cluster)
  resources:
    requests:
      storage: 100Gi   # Llama-3 8B in bf16 = ~16GB; leave room for multiple models
  storageClassName: standard
```

```bash
kubectl apply -f k8s_manifests/03_pvc.yaml
```

The first time a worker pod starts, HuggingFace downloads the model into the PVC. All subsequent pod restarts skip the download and start immediately.

---

## Part 2: Deployment Metadata and Replica Count

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
  labels:
    app: sglang-worker
spec:
  replicas: 2    # start with 2, scale up with kubectl scale or HPA
```

`replicas: 2` means two GPU pods running the same model. The router discovers both and distributes requests between them.

---

## Part 3: The Selector and Pod Template Labels — The Discovery Contract

This is the most important part for service discovery:

```yaml
spec:
  selector:
    matchLabels:
      app: sglang-worker          # Deployment manages pods with this label
  template:
    metadata:
      labels:
        app: sglang-worker        # must match selector.matchLabels
        component: inference      # additional label for fine-grained selection
        model: llama-3-8b        # which model this pod serves
```

**The connection to the router:**

The router is started with `--selector app=sglang-worker component=inference`. The router's discovery code checks every pod to see if it has **all** labels in the selector. A pod with `app=sglang-worker` but missing `component=inference` is ignored.

```
Router selector: app=sglang-worker AND component=inference
Worker pod labels: app=sglang-worker, component=inference, model=llama-3-8b

Does app=sglang-worker match?   YES ✓
Does component=inference match? YES ✓
→ Pod is included in worker pool
```

Choose your labels carefully. They are the contract between the worker Deployment and the router. If you change them in one place, update the other.

---

## Part 4: Container Image and Command

```yaml
    spec:
      containers:
      - name: worker
        image: lmsysorg/sglang:latest    # official SGLang image
        command: ["python", "-m", "sglang.launch_server"]
        args:
          - "--model"
          - "meta-llama/Meta-Llama-3.1-8B-Instruct"
          - "--host"
          - "0.0.0.0"              # listen on all interfaces (not just localhost)
          - "--port"
          - "8000"
          - "--enable-prefix-caching"    # activates RadixCache from Layer 12
          - "--trust-remote-code"        # needed for some models
```

**`--host 0.0.0.0`**: This is critical. Without it, the server only listens on `localhost` (127.0.0.1) inside the container. The router connects to the pod's IP (e.g. `10.244.2.14`), which is a different interface. You must use `0.0.0.0` to listen on all interfaces.

**`--enable-prefix-caching`**: Activates the RadixCache inside the engine. The router's prefix-cache-aware routing policy routes requests to the right engine, but the engine itself needs this flag to actually reuse the cached KV entries. Without it, the routing is prefix-aware but the engine still recomputes everything.

---

## Part 5: Environment Variables

```yaml
        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret    # the Secret from section 03
              key: token
        - name: CUDA_VISIBLE_DEVICES
          value: "0"                   # use GPU 0 (if multiple GPUs per node)
        - name: NCCL_DEBUG
          value: "WARN"               # reduce NCCL log noise
```

**`HF_TOKEN`**: The HuggingFace token for downloading gated models. Loaded from the Secret created in Section 03. The `HF_TOKEN` environment variable is checked automatically by the `transformers` and `huggingface_hub` libraries.

**`CUDA_VISIBLE_DEVICES`**: Controls which GPU(s) the process can see. For single-GPU-per-pod setups (the standard), set to `"0"`. Kubernetes allocates one GPU to the pod; this env var ensures the process uses it.

---

## Part 6: GPU Resource Request

```yaml
        resources:
          requests:
            nvidia.com/gpu: "1"    # request 1 GPU
            memory: "32Gi"         # system RAM for the process
            cpu: "8"               # CPU for tokenization and overhead
          limits:
            nvidia.com/gpu: "1"    # limit to 1 GPU (must equal request for GPU)
            memory: "64Gi"
            cpu: "16"
```

**GPU requests and limits must be equal.** Unlike CPU and memory (where requests and limits can differ), Kubernetes GPU scheduling requires `requests == limits`. If they differ, the pod may not schedule correctly.

**System RAM vs GPU memory:** `memory: 32Gi` is the *system RAM* for the process, not GPU VRAM. The model weights live in GPU HBM (High Bandwidth Memory), managed by CUDA. System RAM holds the Python process, tokenizer, and request buffers. 32–64GB system RAM is typical for an 8B model.

**CPU for tokenization:** The SGLang engine tokenizes requests on the CPU before running the GPU forward pass. 8–16 cores is sufficient for high-throughput serving.

---

## Part 7: Port Declaration

```yaml
        ports:
        - name: http
          containerPort: 8000     # what the container listens on
```

This is informational — it doesn't control networking, just documents the port. The router connects to `pod_ip:8000` regardless of whether this port is declared.

---

## Part 8: The Readiness Probe — The Critical Gate

```yaml
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120   # wait 2 minutes before first probe
          periodSeconds: 5           # check every 5 seconds
          failureThreshold: 3        # 3 consecutive failures = not ready
          successThreshold: 1        # 1 success = ready
```

**Why this matters so much for Layer 16:**

The router's service discovery code (`is_healthy()` in `service_discovery.rs`) only adds a pod to the worker pool when the pod's `Ready` condition is `"True"`. The `Ready` condition becomes `True` only when the readiness probe passes.

Timeline for a worker pod:

```
0:00  Pod scheduled on GPU node
0:05  Container starts, Python process initialises
0:10  Model files found in PVC (or HuggingFace download begins)
1:45  Model loading complete, SGLang engine starts
1:50  /health endpoint returns 200
1:55  Readiness probe passes (initialDelaySeconds=120, then 5s probe interval)
2:00  Pod condition: Ready = True
2:01  K8s watcher fires event: pod modified, now Ready
2:01  Router receives event, calls is_healthy() → true
2:01  Router adds pod to worker pool
2:02  Router starts sending traffic to this pod
```

**Setting `initialDelaySeconds` correctly:**

Too low: Kubernetes tries to probe `/health` before the engine is up. The probe fails. Kubernetes marks the pod NotReady. After `failureThreshold` consecutive failures, Kubernetes restarts the container — you're stuck in a crash loop.

Too high: The model loaded 2 minutes ago, but Kubernetes is still waiting to probe it. No traffic for 2 extra minutes.

**How to measure:** remove the readiness probe from your YAML, deploy, watch the pod, and time how long it takes for `curl http://<pod-ip>:8000/health` to return 200. Add 30 seconds as buffer. That's your `initialDelaySeconds`.

```bash
# Measure model loading time
kubectl exec -it sglang-worker-xxxxx -- \
  sh -c "time curl localhost:8000/health"
```

---

## Part 9: The Liveness Probe

```yaml
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120   # same delay as readiness
          periodSeconds: 10
          failureThreshold: 5        # more tolerant than readiness
```

**Readiness vs Liveness:**
- **Readiness**: "Is this pod ready to receive traffic?" When it fails, the pod is removed from the routing pool (no new traffic), but it is NOT restarted. Useful for temporary overload.
- **Liveness**: "Is this pod alive at all?" When it fails (after `failureThreshold` checks), Kubernetes **restarts the container**. Catches deadlocks and infinite loops.

Set liveness `failureThreshold` higher than readiness (5 vs 3) to avoid restarting pods that are momentarily slow due to a large batch.

---

## Part 10: Volumes — Shared Memory and Model Cache

```yaml
        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface    # where HF libraries look for models
        - name: shm
          mountPath: /dev/shm                    # shared memory for CUDA

      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: sglang-models-pvc           # the PVC from Part 1
      - name: shm
        emptyDir:
          medium: Memory                         # backed by RAM, not disk
          sizeLimit: "2Gi"
```

**Why shared memory (`/dev/shm`):** CUDA tensor parallelism (running a model split across multiple GPUs) uses shared memory for inter-GPU communication via NVIDIA NCCL. By default, Docker and Kubernetes give containers a 64MB `/dev/shm` limit. This is too small for LLM workloads. The `emptyDir: {medium: Memory}` volume allocates a RAM-backed tmpfs with a larger limit.

For a single-GPU-per-pod setup (no tensor parallelism), this is not strictly required but is good practice since some SGLang internals use shared memory for large buffer allocation.

---

## Part 11: Node Affinity and Tolerations

```yaml
      nodeSelector:
        nvidia.com/gpu.present: "true"    # only schedule on GPU nodes

      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"              # allow scheduling on GPU-tainted nodes
```

**`nodeSelector`:** Ensures the pod only schedules on nodes with GPUs. Without this, Kubernetes might try to schedule the pod on a CPU node where the GPU resource request will fail.

**`tolerations`:** Many GPU clusters **taint** their GPU nodes to prevent non-GPU workloads from landing on them (GPU nodes are expensive). A taint says "don't schedule here unless the pod tolerates this". Adding this toleration allows your GPU pods to land on tainted GPU nodes. CPU pods (like the router) should NOT have this toleration — they should run on cheaper CPU nodes.

---

## Full Worker Deployment YAML

```yaml
# k8s_manifests/04_worker_deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
  labels:
    app: sglang-worker
spec:
  replicas: 2
  selector:
    matchLabels:
      app: sglang-worker
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      serviceAccountName: default    # workers don't need K8s API access

      containers:
      - name: worker
        image: lmsysorg/sglang:latest
        imagePullPolicy: Always      # ensure latest version on each restart

        command: ["python", "-m", "sglang.launch_server"]
        args:
          - "--model"
          - "meta-llama/Meta-Llama-3.1-8B-Instruct"
          - "--host"
          - "0.0.0.0"
          - "--port"
          - "8000"
          - "--enable-prefix-caching"
          - "--trust-remote-code"

        env:
        - name: HF_TOKEN
          valueFrom:
            secretKeyRef:
              name: hf-token-secret
              key: token
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NCCL_DEBUG
          value: "WARN"

        ports:
        - name: http
          containerPort: 8000

        resources:
          requests:
            nvidia.com/gpu: "1"
            memory: "32Gi"
            cpu: "8"
          limits:
            nvidia.com/gpu: "1"
            memory: "64Gi"
            cpu: "16"

        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 5
          failureThreshold: 3
          successThreshold: 1

        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 120
          periodSeconds: 10
          failureThreshold: 5

        volumeMounts:
        - name: model-cache
          mountPath: /root/.cache/huggingface
        - name: shm
          mountPath: /dev/shm

      volumes:
      - name: model-cache
        persistentVolumeClaim:
          claimName: sglang-models-pvc
      - name: shm
        emptyDir:
          medium: Memory
          sizeLimit: "2Gi"

      nodeSelector:
        nvidia.com/gpu.present: "true"

      tolerations:
      - key: "nvidia.com/gpu"
        operator: "Exists"
        effect: "NoSchedule"
```

---

## Deploying and Verifying

```bash
# Apply the Deployment
kubectl apply -f k8s_manifests/04_worker_deployment.yaml

# Watch pods come up (takes 2-10 minutes depending on model size)
kubectl get pods -n production -l app=sglang-worker -w

# Expected progression:
# NAME                     READY   STATUS              RESTARTS
# sglang-worker-abc12      0/1     ContainerCreating   0
# sglang-worker-abc12      0/1     Running             0        ← model loading
# sglang-worker-abc12      1/1     Running             0        ← ready! router will add this

# Check logs if a pod is stuck
kubectl logs -n production sglang-worker-abc12 --follow

# Verify the engine is serving
kubectl exec -n production sglang-worker-abc12 -- \
  curl -s localhost:8000/health
# Expected: {"status": "ok"}

# Check GPU usage inside the pod
kubectl exec -n production sglang-worker-abc12 -- nvidia-smi
```

---

## Scaling Workers

```bash
# Scale up to 4 workers
kubectl scale deployment sglang-worker -n production --replicas=4

# Watch new pods become ready (they get discovered by the router automatically)
kubectl get pods -n production -l app=sglang-worker -w

# Scale back down to 2
kubectl scale deployment sglang-worker -n production --replicas=2
# K8s sends SIGTERM to the terminating pods
# Router removes them before they shut down (deletionTimestamp triggers removal)
```

When you scale up, each new pod goes through the same lifecycle: Running → readiness probe passes → Ready = True → router discovers it → traffic starts flowing. No router restart needed.

When you scale down, K8s sets `deletionTimestamp` on the terminating pods. The router's watcher fires, removes the pods from the worker pool, and routes all future requests to the remaining workers. In-flight requests to the terminating pod complete normally (the pod has a graceful shutdown period).
