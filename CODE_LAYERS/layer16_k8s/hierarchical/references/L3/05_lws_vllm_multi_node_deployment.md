# LeaderWorkerSet (LWS) — Overview and vLLM Deployment

**Sources:**
- Overview: https://lws.sigs.k8s.io/docs/overview/
- vLLM deployment: https://docs.vllm.ai/en/stable/deployment/frameworks/lws/
**Author:** Kubernetes SIG Apps
**Level:** L3–L4 — Multi-node GPU deployment
**Why here:** LWS is the Kubernetes-native primitive for multi-node tensor-parallel LLM inference. Layer 16 uses single-GPU-per-pod workers and does not require LWS — but understanding it marks where Layer 16 ends and multi-node deployment begins. Referenced in `04_worker_deployment.md` as "what comes after".

---

## What is LeaderWorkerSet?

LeaderWorkerSet (LWS) is a Kubernetes API for deploying a group of pods as a single replicated unit. It addresses common deployment patterns for AI/ML inference workloads — especially multi-host inference where an LLM is sharded across multiple GPUs on multiple nodes.

**Key concept: "super-pod"** — a single LWS replica consists of:
- **1 leader pod**: runs the model server (vLLM/SGLang), exposes the HTTP port
- **N-1 worker pods**: join the Ray/NCCL collective, no external port

All pods in the group are created and destroyed atomically.

**Adopters (2025–2026):** vLLM, SGLang, NVIDIA NIM, NVIDIA Dynamo, llm-d, Amazon EKS, GKE — the industry standard for multi-node LLM deployment on K8s.

---

## Feature Overview

- **Group of pods as a unit**: a `LeaderWorkerSet` replica is a "super pod" with a leader and N workers
- **Unique pod identity**: each pod gets a unique index (0 to N-1); leader is always index 0
- **Parallel creation**: all pods in the group have the same lifecycle and are created in parallel
- **All-or-nothing restart**: `restartPolicy: RecreateGroupOnPodRestart` — if any pod fails, the entire group restarts; prevents partial tensor-parallel hangs
- **Automatic env injection**: `LWS_GROUP_SIZE` and `LWS_LEADER_ADDRESS` are injected into all pods automatically

---

## Layer 16 vs. LWS

| Layer 16 (single-GPU-per-pod) | LWS (multi-node) |
|---|---|
| `Deployment` with `replicas: N` | `LeaderWorkerSet` with `replicas: N` |
| Each pod = 1 GPU, 1 vLLM/SGLang instance | Each LWS replica = M pods = M×8 GPUs, 1 model instance |
| Llama-3 8B, one GPU | Llama-3.1 405B, 16 GPUs across 2 nodes |
| `nvidia.com/gpu: "1"` | `nvidia.com/gpu: "8"` per pod |
| Service routes to any pod | Service routes only to leader pod |

---

## vLLM + LWS Deployment: 2 nodes × 8 GPUs

### Prerequisites
- Kubernetes cluster with ≥ 2 nodes, each with 8 GPUs
- LWS controller installed: `kubectl apply --server-side -f https://github.com/kubernetes-sigs/lws/releases/download/v0.4.0/manifests.yaml`

### YAML (lws.yaml)

```yaml
apiVersion: leaderworkerset.x-k8s.io/v1
kind: LeaderWorkerSet
metadata:
  name: vllm
spec:
  replicas: 1                # 1 LWS replica = 1 model instance
  leaderWorkerTemplate:
    size: 2                  # 1 leader + 1 worker = 2 nodes total
    restartPolicy: RecreateGroupOnPodRestart

    leaderTemplate:
      metadata:
        labels:
          role: leader
      spec:
        containers:
          - name: vllm-leader
            image: docker.io/vllm/vllm-openai:latest
            env:
              - name: HF_TOKEN
                value: <your-hf-token>
            command:
              - sh
              - -c
              - >
                bash /vllm-workspace/examples/online_serving/multi-node-serving.sh leader
                --ray_cluster_size=$(LWS_GROUP_SIZE);
                vllm serve meta-llama/Meta-Llama-3.1-405B-Instruct
                --port 8080
                --tensor-parallel-size 8
                --pipeline_parallel_size 2
            resources:
              limits:
                nvidia.com/gpu: "8"
                memory: 1124Gi
            ports:
              - containerPort: 8080
            readinessProbe:
              httpGet:
                path: /health
                port: 8080
              initialDelaySeconds: 1800   # 30 min — 405B model takes time to load
              periodSeconds: 30
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 15Gi

    workerTemplate:
      spec:
        containers:
          - name: vllm-worker
            image: docker.io/vllm/vllm-openai:latest
            command:
              - sh
              - -c
              - >
                bash /vllm-workspace/examples/online_serving/multi-node-serving.sh worker
            resources:
              limits:
                nvidia.com/gpu: "8"
                memory: 1124Gi
            volumeMounts:
              - name: dshm
                mountPath: /dev/shm
        volumes:
          - name: dshm
            emptyDir:
              medium: Memory
              sizeLimit: 15Gi
---
# Service only exposes leader pod
apiVersion: v1
kind: Service
metadata:
  name: vllm-leader
spec:
  ports:
    - name: http
      port: 8080
      protocol: TCP
      targetPort: 8080
  selector:
    leaderworkerset.sigs.k8s.io/name: vllm
    role: leader              # ← only leader pods, not workers
  type: ClusterIP
```

### Deploy and verify

```bash
export HF_TOKEN=<your-hf-token>
kubectl apply -f lws.yaml

# Check pods
kubectl get pods
# NAME        READY   STATUS    RESTARTS   AGE
# vllm-0      1/1     Running   0          2s   ← leader
# vllm-0-1    1/1     Running   0          2s   ← worker

# Test
kubectl port-forward svc/vllm-leader 8080:8080
curl http://localhost:8080/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "meta-llama/Meta-Llama-3.1-405B-Instruct", "prompt": "Hello", "max_tokens": 50}'
```

---

## Automatic Environment Variables

LWS injects these into every pod in the group:

| Variable | Value | Usage |
|---|---|---|
| `LWS_GROUP_SIZE` | Number of pods in the group (e.g., `2`) | `--ray_cluster_size=$(LWS_GROUP_SIZE)` |
| `LWS_LEADER_ADDRESS` | Leader pod's hostname | Worker uses to join Ray cluster |
| `LWS_WORKER_INDEX` | Pod index (0 = leader, 1..N-1 = workers) | Determine role in startup script |

No manual pod IP wiring needed — LWS handles group membership and leader discovery automatically.

---

## When to use LWS

| Scenario | Use |
|---|---|
| Single GPU per model instance (Llama-3 8B on A100) | `Deployment` — Layer 16 pattern |
| 8 GPUs per model on 1 node (Llama-3 70B on 8×A100) | `Deployment` with `nvidia.com/gpu: "8"` |
| 16 GPUs across 2 nodes (Llama-3.1 405B) | `LeaderWorkerSet` — LWS |
| 32 GPUs across 4 nodes (Llama-3.1 405B, 2 replicas) | `LeaderWorkerSet`, `replicas: 2` |

The threshold is **multi-node**: if the model fits on a single node (any number of GPUs), use a regular `Deployment`. If it spans nodes, use LWS.
