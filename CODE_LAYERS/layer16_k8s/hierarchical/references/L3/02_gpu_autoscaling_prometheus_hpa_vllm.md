# GPU Autoscaling on Kubernetes: Prometheus Metrics to HPA with vLLM on GKE

**Source:** https://medium.com/@akrem.issaoui1/gpu-autoscaling-on-kubernetes-from-prometheus-metrics-to-hpa-with-vllm-on-gke-48578e5753d1
**Author:** Akrem Issaoui
**Date:** April 2026
**Level:** L3 — Technical walkthrough
**Why here:** End-to-end guide for the complete HPA pipeline that Layer 16's `07_observability.md` and `08_high_availability.md` describe: vLLM metrics → Prometheus → prometheus-adapter → Kubernetes Custom Metrics API → HPA. Quantifies the "CPU is useless for GPU autoscaling" problem.

---

## The Core Problem

When you deploy a traditional web server, Kubernetes gives you CPU and memory metrics out of the box. For GPU-based LLM workloads, this model breaks entirely.

**A vLLM pod holding a 4B-parameter model in VRAM consumes the same CPU and memory whether it's idle or serving 100 concurrent users.** The signal that matters — how many users are waiting in line — lives inside the application, exposed via Prometheus.

> "CPU-based autoscaling doesn't work for LLM workloads. A GPU pod at 0% CPU utilization and 100% GPU utilization look identical to standard Kubernetes metrics."

---

## The Signal Chain

```
vLLM exports metrics → Prometheus scrapes → prometheus-adapter
→ Kubernetes Custom Metrics API → HPA reads queue depth
→ Scales GPU replicas
```

The adapter translates: Prometheus stores `vllm:num_requests_waiting`; HPA doesn't speak Prometheus. The adapter runs as a pod, queries Prometheus on a schedule, and re-exposes results through `custom.metrics.k8s.io`.

---

## Phase 1: Deploy kube-prometheus-stack

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install kube-prometheus-stack prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace
```

**GKE Autopilot caveat:** Several components (node-exporter as privileged DaemonSet) don't work. Disable them:
```bash
--set nodeExporter.enabled=false \
--set prometheus.prometheusSpec.nodeSelector."cloud\.google\.com/gke-nodepool"=default-pool
```

---

## Phase 2: Configure ServiceMonitor for vLLM

```yaml
# Service exposing vLLM metrics on a named port
apiVersion: v1
kind: Service
metadata:
  name: vllm-metrics
  labels:
    app: vllm
    monitoring: enabled
spec:
  selector:
    app: vllm
  ports:
    - name: metrics         # ← must match ServiceMonitor endpoint port
      port: 8000
      targetPort: 8000
---
# ServiceMonitor — tell Prometheus Operator to scrape this
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: vllm
  labels:
    release: kube-prometheus-stack   # ← must match Prometheus Operator label
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
    - port: metrics          # ← must match Service port name
      path: /metrics
      interval: 15s
```

**Three-part alignment:** Named port + namespace selector + `release` label must all match. The "colon vs underscore trap": real metric name is `vllm:kv_cache_usage_perc` (with colon), not `vllm_kv_cache_usage_perc` (with underscore).

---

## Phase 3: Install prometheus-adapter

```bash
helm install prometheus-adapter prometheus-community/prometheus-adapter \
  --namespace monitoring \
  --set prometheus.url=http://kube-prometheus-stack-prometheus.monitoring.svc \
  --set prometheus.port=9090
```

Configure the adapter to expose `vllm:num_requests_waiting` as a K8s custom metric:

```yaml
# values.yaml for prometheus-adapter
rules:
  custom:
    - seriesQuery: 'vllm:num_requests_waiting{namespace!="",pod!=""}'
      resources:
        overrides:
          namespace:
            resource: namespace
          pod:
            resource: pod
      name:
        matches: "vllm:num_requests_waiting"
        as: "vllm_requests_waiting"
      metricsQuery: 'avg(<<.Series>>{<<.LabelMatchers>>})'
```

Verify the metric is visible:
```bash
kubectl get --raw "/apis/custom.metrics.k8s.io/v1beta1" | jq '.resources[].name'
```

---

## Phase 4: Configure HPA

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: vllm-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: vllm
  minReplicas: 1
  maxReplicas: 5
  metrics:
    - type: Pods
      pods:
        metric:
          name: vllm_requests_waiting
        target:
          type: AverageValue
          averageValue: "3"   # Scale up when avg > 3 waiting requests per pod
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300   # Wait 5 min before scaling down
      policies:
        - type: Pods
          value: 1
          periodSeconds: 60
    scaleUp:
      stabilizationWindowSeconds: 60
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60
```

---

## Phase 5: Add PrometheusRule alerts

```yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: vllm-alerts
  labels:
    release: kube-prometheus-stack
spec:
  groups:
    - name: vllm
      rules:
        - alert: HighRequestQueue
          expr: avg(vllm:num_requests_waiting) > 10
          for: 5m
          labels:
            severity: warning
          annotations:
            summary: "vLLM request queue depth high"

        - alert: KVCachePressure
          expr: max(vllm:kv_cache_usage_perc) > 90
          for: 10m
          labels:
            severity: critical
          annotations:
            summary: "KV cache usage > 90%"
```

---

## GKE Autopilot: GPU node provisioning lag

On GKE Autopilot, scaling from 1 to 3 GPU replicas triggers node provisioning:
1. HPA requests new pod
2. Autopilot provisions an L4 GPU node (~3–5 min)
3. Pod schedules on new node
4. Model loads into VRAM (~1–2 min for a 7B model)
5. Readiness probe passes → pod enters service

**Total: ~5–7 min from HPA trigger to serving.** Set `stabilizationWindowSeconds: 300` for scale-down to avoid thrashing.

---

## Mapping to Layer 16

| This guide | Layer 16 |
|---|---|
| vLLM `vllm:num_requests_waiting` | SGLang `smg_worker_requests_active` |
| prometheus-adapter for Custom Metrics API | `k8s/sglang/templates/hpa.yaml` pattern |
| KEDA (alternative approach) | Simpler — no prometheus-adapter needed |
| GKE Autopilot provisioning lag | Same `initialDelaySeconds` sizing challenge |

**Key insight:** The prometheus-adapter approach requires a chain of 5 components (Prometheus, adapter, Custom Metrics API, HPA, Cluster Autoscaler). KEDA (see `L3/03_keda_prometheus_scaler.md`) simplifies this to 2 components: Prometheus + KEDA.
