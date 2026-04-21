# 07 — Observability: Metrics and Alerts

## What This Section Covers

Once the router and workers are running in Kubernetes, you need to be able to answer questions like:
- Is the router healthy right now?
- How many requests per second is it handling?
- Are any workers overloaded?
- How long are requests taking?
- How many workers has the router discovered?

This is **observability** — the ability to understand the internal state of a system by looking at its external outputs. In Layer 16, we use **Prometheus** to collect metrics and set up alerts.

---

## What Is Prometheus?

Prometheus is the standard monitoring system for Kubernetes workloads. It works by:

1. **Scraping**: Prometheus periodically calls each service's `/metrics` endpoint (an HTTP endpoint that returns metric values as plain text)
2. **Storing**: It stores these metric values as time series (value over time)
3. **Querying**: You query the stored data using PromQL (Prometheus Query Language)
4. **Alerting**: You define rules — "if this metric exceeds this threshold, fire an alert"

The router exposes a `/metrics` endpoint on port `29000`. Prometheus calls it every 15 seconds and stores the values.

---

## Part 1: What Metrics the Router Exposes

The router (via `--metrics-port 29000`) exposes metrics in Prometheus exposition format. Here is what a raw scrape looks like:

```
# HELP discovery_workers_discovered Number of workers currently discovered
# TYPE discovery_workers_discovered gauge
discovery_workers_discovered{source="kubernetes"} 3

# HELP discovery_registrations_total Total number of worker registrations
# TYPE discovery_registrations_total counter
discovery_registrations_total{source="kubernetes",status="success"} 5
discovery_registrations_total{source="kubernetes",status="failed"} 0
discovery_registrations_total{source="kubernetes",status="duplicate"} 2

# HELP discovery_deregistrations_total Total number of worker deregistrations
# TYPE discovery_deregistrations_total counter
discovery_deregistrations_total{source="kubernetes",reason="pod_deleted"} 2

# HELP http_requests_total Total number of HTTP requests
# TYPE http_requests_total counter
http_requests_total{method="POST",path="/v1/chat/completions",status="200"} 1842
http_requests_total{method="POST",path="/v1/chat/completions",status="503"} 3

# HELP http_request_duration_seconds HTTP request duration in seconds
# TYPE http_request_duration_seconds histogram
http_request_duration_seconds_bucket{le="0.1"} 10
http_request_duration_seconds_bucket{le="1.0"} 450
http_request_duration_seconds_bucket{le="5.0"} 1800
http_request_duration_seconds_bucket{le="30.0"} 1840
http_request_duration_seconds_bucket{le="+Inf"} 1842
http_request_duration_seconds_sum 2180.5
http_request_duration_seconds_count 1842
```

**Gauge** — a current value that can go up or down. Example: `discovery_workers_discovered = 3`.
**Counter** — a value that only increases. Example: `http_requests_total = 1842` (total since process started).
**Histogram** — records the distribution of a value across buckets. Example: request duration.

---

## Part 2: Checking Metrics Manually

Before setting up Prometheus, you can check the metrics directly:

```bash
# Port-forward the metrics port
kubectl port-forward svc/sglang-router -n production 29000:29000 &

# View raw metrics
curl http://localhost:29000/metrics

# Check specific metrics with grep
curl -s http://localhost:29000/metrics | grep discovery_workers_discovered
# discovery_workers_discovered{source="kubernetes"} 2

curl -s http://localhost:29000/metrics | grep http_requests_total
# http_requests_total{...,status="200"} 842
# http_requests_total{...,status="503"} 0
```

If `discovery_workers_discovered` is 0 and you have workers running, the service discovery is not working (check RBAC, selector labels, namespace).

If `http_requests_total{status="503"}` is non-zero and growing, the router has no workers or all workers are failing health checks.

---

## Part 3: Setting Up Prometheus Scraping

### Option A: Annotation-Based Scraping (Simple)

The router Deployment in Section 05 already has annotations:

```yaml
template:
  metadata:
    annotations:
      prometheus.io/scrape: "true"
      prometheus.io/port: "29000"
      prometheus.io/path: "/metrics"
```

If your Prometheus is configured with the standard Kubernetes annotation scraping config (common in `kube-prometheus-stack`), it will automatically discover and scrape pods with these annotations. No extra configuration needed.

### Option B: ServiceMonitor (Recommended for Production)

A `ServiceMonitor` is a custom Kubernetes resource that tells Prometheus Operator exactly which services to scrape:

```yaml
# k8s_manifests/07_servicemonitor.yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sglang-router
  namespace: production
  labels:
    release: prometheus   # must match your Prometheus Operator's selector
spec:
  selector:
    matchLabels:
      app: sglang-router   # must match the Service's labels
  endpoints:
  - port: metrics          # the named port in the Service (port 29000)
    path: /metrics
    interval: 15s          # scrape every 15 seconds
    scrapeTimeout: 10s
```

Apply it:

```bash
kubectl apply -f k8s_manifests/07_servicemonitor.yaml
```

This tells Prometheus to call `sglang-router.production.svc.cluster.local:29000/metrics` every 15 seconds.

### Installing Prometheus (if not already installed)

```bash
# The kube-prometheus-stack installs Prometheus + Grafana + AlertManager
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm repo update

helm install prometheus prometheus-community/kube-prometheus-stack \
  --namespace monitoring \
  --create-namespace \
  --set prometheus.prometheusSpec.serviceMonitorSelectorNilUsesHelmValues=false
```

The last flag is important: it makes Prometheus pick up ServiceMonitors from all namespaces, not just the one where Prometheus itself is running.

---

## Part 4: Useful PromQL Queries

Once metrics are being scraped, use these queries in Prometheus (or Grafana):

### Request Throughput (requests per second)

```promql
# Requests per second over the last 5 minutes
rate(http_requests_total{namespace="production"}[5m])
```

### Error Rate

```promql
# Fraction of requests returning 5xx errors
rate(http_requests_total{status=~"5.."}[5m])
/
rate(http_requests_total[5m])
```

If this is above 0.01 (1%), something is wrong.

### P50 / P95 / P99 Latency

```promql
# Median request latency
histogram_quantile(0.50, rate(http_request_duration_seconds_bucket[5m]))

# 95th percentile (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# 99th percentile (p99)
histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))
```

P99 latency is the most useful for LLM serving: it tells you the worst-case experience for 1% of requests. Prefix cache hits are fast (<1s); cache misses or long prompts can be slow (10-60s).

### Worker Count

```promql
# Number of workers currently in the router's pool
discovery_workers_discovered{source="kubernetes", namespace="production"}
```

This should equal the number of Ready worker pods. If it's lower, some pods are excluded (health check failure or label mismatch). If it drops to 0, all requests will fail with 503.

### Worker Registration Events

```promql
# Rate of new workers being added (should spike when you scale up)
rate(discovery_registrations_total{status="success"}[5m])

# Rate of workers being removed (should spike when you scale down)
rate(discovery_deregistrations_total[5m])
```

---

## Part 5: Alerting Rules

Create a PrometheusRule to define alerts:

```yaml
# k8s_manifests/08_alerts.yaml
apiVersion: monitoring.coreos.com/v1
kind: PrometheusRule
metadata:
  name: sglang-router-alerts
  namespace: production
  labels:
    release: prometheus   # must match your Prometheus Operator's ruleSelector
spec:
  groups:
  - name: sglang.router
    interval: 30s
    rules:

    # Alert 1: No workers discovered
    - alert: SGLangRouterNoWorkers
      expr: |
        discovery_workers_discovered{source="kubernetes"} == 0
      for: 2m        # alert fires if condition is true for 2 consecutive minutes
      labels:
        severity: critical
      annotations:
        summary: "SGLang router has no workers"
        description: |
          The router has 0 discovered workers. All inference requests will return 503.
          Check: kubectl get pods -n production -l app=sglang-worker
          Check: kubectl logs -n production -l app=sglang-router | grep -E "error|Error"

    # Alert 2: High error rate
    - alert: SGLangHighErrorRate
      expr: |
        rate(http_requests_total{status=~"5.."}[5m])
        /
        rate(http_requests_total[5m])
        > 0.05
      for: 5m        # only alert if sustained for 5 minutes (not transient spikes)
      labels:
        severity: warning
      annotations:
        summary: "SGLang router error rate above 5%"
        description: "Error rate is {{ $value | humanizePercentage }} over the last 5 minutes."

    # Alert 3: High P99 latency
    - alert: SGLangHighLatency
      expr: |
        histogram_quantile(0.99,
          rate(http_request_duration_seconds_bucket[5m])
        ) > 60
      for: 10m
      labels:
        severity: warning
      annotations:
        summary: "SGLang router P99 latency exceeds 60 seconds"
        description: "P99 latency is {{ $value }}s. Workers may be overloaded."

    # Alert 4: Worker count dropped (scale-down was not intentional)
    - alert: SGLangWorkerCountDrop
      expr: |
        delta(discovery_workers_discovered{source="kubernetes"}[10m]) < -1
      for: 0m        # alert immediately on first occurrence
      labels:
        severity: warning
      annotations:
        summary: "SGLang worker count dropped by more than 1"
        description: "Worker count changed by {{ $value }}. Check for pod failures."
```

```bash
kubectl apply -f k8s_manifests/08_alerts.yaml
```

**How alerts work:**
1. Prometheus evaluates each rule every `interval` seconds
2. If the `expr` is true, the alert enters `Pending` state
3. If it stays true for the `for` duration, it fires (goes to `Alerting` state)
4. AlertManager receives the alert and sends it to your notification channel (Slack, PagerDuty, email, etc.)

---

## Part 6: Grafana Dashboard

If you have Grafana installed (included in `kube-prometheus-stack`), create a dashboard:

```bash
# Port-forward Grafana
kubectl port-forward -n monitoring svc/prometheus-grafana 3000:80 &

# Open http://localhost:3000
# Default credentials: admin / prom-operator
```

**Recommended panels for an SGLang router dashboard:**

```
Panel 1: Workers Discovered (gauge, current value)
  Query: discovery_workers_discovered{source="kubernetes"}
  Thresholds: 0 = red, 1 = yellow, 2+ = green

Panel 2: Request Rate (graph)
  Query: rate(http_requests_total[5m])

Panel 3: Error Rate (graph)
  Query: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100
  Unit: percent (0-100)

Panel 4: Latency Percentiles (graph)
  Queries:
    p50: histogram_quantile(0.5, rate(http_request_duration_seconds_bucket[5m]))
    p95: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))
    p99: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m]))

Panel 5: Worker Registrations (graph)
  Query: rate(discovery_registrations_total{status="success"}[5m])
  Spikes here = new workers discovered (scale-up events)
```

---

## Part 7: Worker-Level Metrics

The SGLang inference engine itself also exposes metrics on port 8000:

```bash
# Check worker metrics directly
kubectl exec -n production sglang-worker-abc12 -- \
  curl -s localhost:8000/metrics | head -50

# Key worker metrics:
# sglang:num_running_reqs   — currently processing requests
# sglang:num_waiting_reqs   — queued but not yet started
# sglang:cache_hit_rate     — fraction of tokens served from KV cache
# sglang:gpu_cache_usage    — fraction of GPU cache in use
```

These are per-worker metrics. The router exposes aggregate/routing-level metrics; the worker exposes GPU and model-level metrics.

**Important worker metric: `sglang:cache_hit_rate`**

This tells you whether the prefix-cache-aware routing policy is working. If hit rate is low (<10%) with repeated similar prompts, the routing may not be directing same-prefix requests to the same worker consistently. A well-tuned deployment should show 50–80% cache hit rates for chat workloads with shared system prompts.

```promql
# Average cache hit rate across all workers
avg(sglang:cache_hit_rate)

# GPU cache usage (if this approaches 1.0, workers are near capacity)
max(sglang:gpu_cache_usage)
```

---

## Part 8: Structured Logging

In addition to metrics, the router emits structured logs. Useful patterns to search for:

```bash
# All discovery events (adds + removes)
kubectl logs -n production -l app=sglang-router | grep -E "Adding pod|Removing pod"

# Service discovery errors
kubectl logs -n production -l app=sglang-router | grep -E "Error in Kubernetes watcher|Retrying"

# Router startup sequence
kubectl logs -n production -l app=sglang-router | grep -E "Starting K8s|Router ready|Workers:"

# All errors
kubectl logs -n production -l app=sglang-router --since=1h | grep -i error
```

**Common log patterns and what they mean:**

```
[INFO] Starting K8s service discovery | selector: 'app=sglang-worker,component=inference'
→ Router starting, service discovery enabled, correct selector loaded

[INFO] Adding pod: sglang-worker-abc12 | type: Some(Regular) | url: http://10.244.2.14:8000
→ New worker discovered and being added to the pool

[INFO] Removing pod: sglang-worker-abc12 | url: http://10.244.2.14:8000
→ Worker pod terminated, removed from pool

[ERROR] Error in Kubernetes watcher: 403 Forbidden
→ RBAC problem: ServiceAccount doesn't have list/watch permissions on pods

[WARN] Retrying in 2 seconds with exponential backoff
→ K8s API temporarily unreachable, will retry

[WARN] Kubernetes watcher exited, restarting in 60 seconds
→ Normal: the watch connection closed (K8s API servers periodically close watch connections)
```

---

## Summary of Observability Setup

| Component | What to Install | What It Does |
|---|---|---|
| Router `/metrics` endpoint | Built-in (enabled with `--metrics-port 29000`) | Exposes metrics in Prometheus format |
| Prometheus | `kube-prometheus-stack` Helm chart | Scrapes and stores metrics |
| ServiceMonitor | `07_servicemonitor.yaml` | Tells Prometheus to scrape the router |
| PrometheusRule | `08_alerts.yaml` | Fires alerts on error rate, zero workers, etc. |
| Grafana | Included in `kube-prometheus-stack` | Dashboards for visualizing metrics |

The most critical metric is `discovery_workers_discovered`. If it drops to 0, set up an immediate alert — the system cannot serve any requests until at least one worker is discovered.
