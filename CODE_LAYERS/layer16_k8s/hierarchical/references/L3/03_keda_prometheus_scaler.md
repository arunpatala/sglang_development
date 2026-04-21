# KEDA — Kubernetes Event-Driven Autoscaling: Prometheus Scaler

**Source:** https://keda.sh/docs/latest/scalers/prometheus/
**GitHub:** https://github.com/kedacore/keda
**Author:** KEDA Project (CNCF)
**Version:** 2.19 (latest as of 2026)
**Level:** L3 — Technical autoscaling reference
**Why here:** KEDA is the recommended alternative to prometheus-adapter for GPU LLM autoscaling. Layer 16's lesson/08 mentions KEDA as the path for GPU-specific scaling. This is the definitive reference for `ScaledObject` configuration with Prometheus as the trigger source.

---

## What KEDA does

KEDA (Kubernetes Event-Driven Autoscaling) sits between a metric source and the Kubernetes HPA controller, enabling:
- **Scale-to-zero**: pods can scale to 0 replicas when idle
- **Scale-from-zero**: automatically provisions pods when traffic arrives
- **Arbitrary metrics**: any Prometheus query, Redis queue length, Azure Service Bus, etc.

**Key difference from prometheus-adapter:**
- prometheus-adapter exposes metrics through the Kubernetes *Custom Metrics API* (per-object metrics)
- KEDA uses the Kubernetes *External Metrics API* (cluster-scoped)
- Both work with HPA; KEDA requires fewer components and natively supports scale-to-zero

---

## Install KEDA

```bash
helm repo add kedacore https://kedacore.github.io/charts
helm repo update
helm install keda kedacore/keda --namespace keda --create-namespace
```

---

## Prometheus Scaler — Trigger Specification

```yaml
triggers:
- type: prometheus
  metadata:
    # Required
    serverAddress: http://prometheus.monitoring.svc:9090
    query: avg(smg_worker_requests_active{namespace="production"})
    threshold: '20'
    # Optional
    activationThreshold: '1'   # Minimum value to start scaling (default: 0)
    namespace: production       # For namespaced Prometheus queries (Thanos)
    ignoreNullValues: "false"  # Return error when target lost (default: true = ignore)
    timeout: "5000"            # HTTP timeout in ms
```

### Parameters

| Parameter | Description |
|---|---|
| `serverAddress` | Address of Prometheus server |
| `query` | PromQL query. Must return a vector/scalar single element |
| `threshold` | Value at which scaling begins |
| `activationThreshold` | Minimum value to exit scale-to-zero (default: 0) |
| `namespace` | For namespaced queries (Thanos, etc.) |
| `ignoreNullValues` | If `false`, scaler errors when Prometheus target is lost |
| `timeout` | HTTP client timeout override |

---

## ScaledObject for SGLang Workers (Layer 16)

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: sglang-worker-scaler
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sglang-worker
  pollingInterval: 30          # Seconds between metric checks
  cooldownPeriod: 300          # Seconds after last trigger before scale-to-zero
  minReplicaCount: 1           # Never scale to 0 for LLM workloads
  maxReplicaCount: 10
  advanced:
    horizontalPodAutoscalerConfig:
      behavior:
        scaleDown:
          stabilizationWindowSeconds: 300   # 5 min before scaling down
          policies:
            - type: Pods
              value: 1
              periodSeconds: 120           # Remove at most 1 pod per 2 min
        scaleUp:
          stabilizationWindowSeconds: 60
          policies:
            - type: Pods
              value: 2
              periodSeconds: 60           # Add at most 2 pods per min
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://kube-prometheus-stack-prometheus.monitoring.svc:9090
      query: >
        avg(smg_worker_requests_active{
          namespace="production",
          job="sglang-workers"
        })
      threshold: '20'            # Scale up when avg active requests > 20/pod
      activationThreshold: '1'  # First pod wakes from 0 at 1 active request
```

**Why `minReplicaCount: 1`:** GPU pods take 2–7 minutes to start (model loading). Scale-to-zero is only practical when you can tolerate that cold start latency. For production LLM serving, keep at least 1 replica.

---

## ScaledObject for vLLM Workers

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: vllm-worker-scaler
  namespace: production
spec:
  scaleTargetRef:
    name: vllm-deployment
  minReplicaCount: 1
  maxReplicaCount: 5
  triggers:
  - type: prometheus
    metadata:
      serverAddress: http://prometheus.monitoring.svc:9090
      query: avg(vllm:num_requests_waiting{namespace="production"})
      threshold: '3'              # Scale up when avg > 3 waiting per pod
      activationThreshold: '0.5'
```

---

## Authentication with TriggerAuthentication

For Prometheus endpoints requiring authentication:

```yaml
apiVersion: v1
kind: Secret
metadata:
  name: keda-prom-secret
  namespace: keda
data:
  bearerToken: "BASE64_ENCODED_TOKEN"
---
apiVersion: keda.sh/v1alpha1
kind: TriggerAuthentication
metadata:
  name: keda-prom-creds
  namespace: production
spec:
  secretTargetRef:
    - parameter: bearerToken
      name: keda-prom-secret
      key: bearerToken
---
# Reference in ScaledObject
triggers:
- type: prometheus
  metadata:
    serverAddress: https://prometheus.example.com
    query: avg(smg_worker_requests_active)
    threshold: '20'
    authModes: "bearer"
  authenticationRef:
    name: keda-prom-creds
```

---

## Cloud Managed Prometheus

### Google Managed Prometheus
```yaml
triggers:
- type: prometheus
  metadata:
    serverAddress: https://monitoring.googleapis.com/v1/projects/MY_PROJECT/location/global/prometheus
    query: avg(smg_worker_requests_active{namespace="production"})
    threshold: '20'
  authenticationRef:
    kind: ClusterTriggerAuthentication
    name: google-workload-identity-auth
```

### Amazon Managed Service for Prometheus (AMP)
```yaml
triggers:
- type: prometheus
  metadata:
    awsRegion: us-east-1
    serverAddress: "https://aps-workspaces.us-east-1.amazonaws.com/workspaces/WORKSPACE_ID"
    query: avg(smg_worker_requests_active)
    threshold: '20'
    identityOwner: operator
  authenticationRef:
    name: keda-trigger-auth-aws-credentials
```

---

## Tuning guidance for GPU workloads

| Parameter | Recommended | Reason |
|---|---|---|
| `pollingInterval` | 30s | Balance responsiveness vs Prometheus load |
| `cooldownPeriod` | 300s | 5 min — prevents thrashing after a burst |
| `stabilizationWindowSeconds` (down) | 300s | GPU pods are expensive; don't scale down prematurely |
| `stabilizationWindowSeconds` (up) | 60s | Respond to load quickly |
| Scale-up policy `value` | 2 per 60s | Rate-limit pod addition; each GPU pod needs provisioning |
| Scale-down policy `value` | 1 per 120s | Cautious removal; in-flight requests complete first |
| `minReplicaCount` | 1 | Avoid cold start latency for user-facing services |

**Why conservative scale-down:** A request in-flight when a pod is removed fails unless the router has retry logic. SGLang's router removes pods from the pool when their `deletionTimestamp` is set, but in-flight requests are drained before the pod terminates. The `terminationGracePeriodSeconds` on the pod spec should be long enough for the longest expected request to complete.
