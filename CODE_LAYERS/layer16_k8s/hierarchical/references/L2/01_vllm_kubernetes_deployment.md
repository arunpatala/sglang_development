# Using Kubernetes — vLLM Deployment Guide

**Source:** https://docs.vllm.ai/en/stable/deployment/k8s.html
**Author:** vLLM Project
**Level:** L2 — Practitioner deployment
**Why here:** The production reference for deploying any GPU inference engine on Kubernetes with PVC, Secret, probes, and GPU resource requests. Layer 16's `04_worker_deployment.md` follows this pattern directly. The readiness probe section explains the `initialDelaySeconds` methodology used throughout Layer 16.

---

## Overview

Deploying vLLM on Kubernetes is a scalable and efficient way to serve machine learning models. This guide walks you through deploying vLLM using native Kubernetes.

You can also deploy vLLM to Kubernetes using:
- Helm
- NVIDIA Dynamo
- llm-d
- KServe
- LeaderWorkerSet (LWS)
- vllm-project/production-stack

---

## Deployment with GPUs

### 1. Create PVC and Secret

```yaml
# PVC for model weights
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: mistral-7b
  namespace: default
spec:
  accessModes:
    - ReadWriteOnce
  volumeMode: Filesystem
  resources:
    requests:
      storage: 50Gi
---
# HuggingFace token secret
apiVersion: v1
kind: Secret
metadata:
  name: hf-token-secret
  namespace: default
type: Opaque
data:
  token: "YOUR_HF_TOKEN_BASE64_ENCODED"
```

### 2. Create Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: mistral-7b
  namespace: default
  labels:
    app: mistral-7b
spec:
  replicas: 1
  selector:
    matchLabels:
      app: mistral-7b
  template:
    metadata:
      labels:
        app: mistral-7b
    spec:
      volumes:
        - name: cache-volume
          persistentVolumeClaim:
            claimName: mistral-7b
        # vLLM needs to access the host's shared memory for tensor parallel inference
        - name: shm
          emptyDir:
            medium: Memory
            sizeLimit: "2Gi"
      containers:
        - name: mistral-7b
          image: vllm/vllm-openai:latest
          command: ["/bin/sh", "-c"]
          args: [
            "vllm serve mistralai/Mistral-7B-Instruct-v0.3 --trust-remote-code --enable-chunked-prefill --max_num_batched_tokens 1024"
          ]
          env:
            - name: HUGGING_FACE_HUB_TOKEN
              valueFrom:
                secretKeyRef:
                  name: hf-token-secret
                  key: token
          ports:
            - containerPort: 8000
          resources:
            limits:
              cpu: "10"
              memory: 20G
              nvidia.com/gpu: "1"
            requests:
              cpu: "2"
              memory: 6G
              nvidia.com/gpu: "1"
          volumeMounts:
            - mountPath: /root/.cache/huggingface
              name: cache-volume
            - name: shm
              mountPath: /dev/shm
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 10
          readinessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 60
            periodSeconds: 5
```

### 3. Create Service

```yaml
apiVersion: v1
kind: Service
metadata:
  name: mistral-7b
  namespace: default
spec:
  ports:
    - name: http-mistral-7b
      port: 80
      protocol: TCP
      targetPort: 8000
  selector:
    app: mistral-7b
  sessionAffinity: None
  type: ClusterIP
```

### 4. Deploy and Test

```bash
kubectl apply -f deployment.yaml
kubectl apply -f service.yaml

# Test the deployment
curl http://mistral-7b.default.svc.cluster.local/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mistralai/Mistral-7B-Instruct-v0.3",
    "prompt": "San Francisco is a",
    "max_tokens": 7,
    "temperature": 0
  }'
```

---

## Troubleshooting

### Startup Probe or Readiness Probe Failure

If the probe `failureThreshold` is too low for model startup time, Kubernetes kills the container. Indications:

1. Container log contains `"KeyboardInterrupt: terminated"`
2. `kubectl get events` shows `Container $NAME failed startup probe, will be restarted`

**Fix:** Increase `failureThreshold` to allow more time. Identify an ideal threshold by removing probes and measuring actual startup time before the model is ready to serve.

This methodology — profile once, set `initialDelaySeconds` accordingly — is the same approach used in Layer 16's `04_worker_deployment.md`.

---

## Key points for Layer 16

| vLLM pattern | Layer 16 equivalent |
|---|---|
| `nvidia.com/gpu: "1"` in limits and requests | `04_worker_deployment.md` GPU resource block |
| `emptyDir: {medium: Memory, sizeLimit: "2Gi"}` at `/dev/shm` | Shared memory volume for tensor parallel |
| `initialDelaySeconds: 60` readiness probe | Match to actual model load time |
| PVC at `/root/.cache/huggingface` | Model weight caching across restarts |
| HF token from `secretKeyRef` | Never put credentials in ConfigMap or image |
