# Monitor LLM Inference in Production (2026): Prometheus & Grafana

**Source:** https://glukhov.org/observability/monitoring-llm-inference-prometheus-grafana/
**Author:** Rost Glukhov
**Date:** 2026
**Level:** L2–L3 — Observability practitioner
**Why here:** Best general-audience guide to Prometheus monitoring for LLM inference engines (vLLM, TGI, llama.cpp). Explains the ServiceMonitor pattern, key metric names, PromQL examples, and common failure modes. Directly supports Layer 16's `07_observability.md`.

---

## Why monitor LLM inference differently

LLM inference looks like "just another API" — until latency spikes, queues back up, and GPUs sit at 95% memory with no obvious explanation. Traditional API metrics (RPS, p95 latency, error rate) are necessary but not sufficient:

### 1. Latency has two meanings
- **E2E latency**: time from request received → final token returned.
- **Inter-token latency**: time per token during decode (critical for streaming UX).

### 2. Throughput is in tokens, not requests
A "fast" service returning 5 tokens is not comparable to one returning 500 tokens. Your "RPS" should often be "tokens/sec".

### 3. The queue is the product
If you run continuous batching, queue depth is what you sell. Queue duration and queue size tell you whether you're meeting user expectations.

### 4. Cache pressure is an outage precursor
KV cache exhaustion often shows up as sudden latency spikes and timeouts. vLLM exposes KV cache usage as a gauge.

---

## Metrics checklist

### Golden signals (LLM-flavored)
- **Traffic**: requests/sec, tokens/sec
- **Errors**: error rate, timeouts, OOMs, 429s (rate limiting)
- **Latency**: p50/p95/p99 request duration; prefill vs decode latency; inter-token latency
- **Saturation**: GPU utilization, memory usage, KV cache usage, queue size

### Key metric names by engine

**vLLM** (`vllm:` prefix):
- `vllm:num_requests_running`
- `vllm:num_requests_waiting`
- `vllm:kv_cache_usage_perc`

**Hugging Face TGI**:
- `tgi_queue_size` (gauge)
- `tgi_request_duration` (histogram, e2e latency)
- `tgi_request_queue_duration` (histogram)
- `tgi_request_mean_time_per_token_duration` (histogram)

**SGLang** (from Layer 16's router, `smg_` prefix):
- `smg_http_request_duration_seconds` (histogram)
- `smg_worker_requests_active` (per-worker in-flight)
- `smg_worker_pool_size` (healthy worker count)
- `smg_discovery_workers_discovered` (service discovery gauge)

---

## Prometheus configuration: scraping inference servers

```yaml
global:
  scrape_interval: 5s
  evaluation_interval: 15s

scrape_configs:
  - job_name: "vllm"
    metrics_path: /metrics
    static_configs:
      - targets: ["vllm:8000"]

  - job_name: "sglang-router"
    metrics_path: /metrics
    static_configs:
      - targets: ["sglang-router:29000"]   # --prometheus-port

  - job_name: "llama_cpp"
    metrics_path: /metrics
    static_configs:
      - targets: ["llama:8080"]
```

---

## PromQL examples

### Request rate (RPS)
```promql
sum(rate(tgi_request_count[5m]))
```

### Error rate (%)
```promql
1 - (
  sum(rate(tgi_request_success[5m]))
  /
  sum(rate(tgi_request_count[5m]))
)
```

### p95 latency (histogram)
```promql
histogram_quantile(
  0.95,
  sum by (le) (rate(tgi_request_duration_bucket[5m]))
)
```

### p99 queue time
```promql
histogram_quantile(
  0.99,
  sum by (le) (rate(tgi_request_queue_duration_bucket[5m]))
)
```

### vLLM KV cache utilization
```promql
max(vllm:kv_cache_usage_perc)
```

### SGLang worker pool empty (critical alert)
```promql
sum(smg_worker_pool_size) == 0
```

---

## Kubernetes: Prometheus Operator + ServiceMonitor

### 1. Expose inference deployment with a named port

```yaml
apiVersion: v1
kind: Service
metadata:
  name: sglang-router-metrics
  labels:
    app: sglang-router
    monitoring: enabled
spec:
  selector:
    app: sglang-router
  ports:
    - name: metrics        # ← name is critical for ServiceMonitor
      port: 29000
      targetPort: 29000
```

### 2. Create a ServiceMonitor

```yaml
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sglang-router
  labels:
    release: kube-prometheus-stack   # ← must match Prometheus Operator label
spec:
  selector:
    matchLabels:
      monitoring: enabled
  endpoints:
    - port: metrics          # ← must match port name above
      path: /metrics
      interval: 5s
```

**Three-part alignment problem**: named port name + ServiceMonitor `port` field + Prometheus Operator `release` label must all match. This is the most common misconfiguration.

---

## Alerting: SLO-style rules

```yaml
groups:
  - name: llm-inference
    rules:
      - alert: LLMHighP95Latency
        expr: >
          histogram_quantile(0.95,
            sum by (le) (rate(tgi_request_duration_bucket[5m]))
          ) > 3
        for: 10m
        labels:
          severity: page
        annotations:
          summary: "p95 latency > 3s (10m)"

      - alert: LLMQueueBacklog
        expr: max(tgi_queue_size) > 50
        for: 5m
        labels:
          severity: warning

      - alert: LLMHighErrorRate
        expr: >
          sum(rate(tgi_request_count[5m])) - sum(rate(tgi_request_success[5m]))
          / sum(rate(tgi_request_count[5m])) > 0.01
        for: 5m
        labels:
          severity: critical

      - alert: LLMKVCachePressure
        expr: max(vllm:kv_cache_usage_perc) > 90
        for: 15m
        labels:
          severity: warning
```

---

## Troubleshooting: common Prometheus failures

### 1. Prometheus target is "DOWN"
- Wrong port or scheme (http vs https)?
- Kubernetes: Is the Service selecting pods? Is the ServiceMonitor `release` label correct?

Quick test: `curl -sS http://vllm:8000/metrics | head`

### 2. Panels are empty
- Wrong metric name (server version changed metric names)
- Dashboard expects `_bucket` suffix for histograms but metric is a gauge/counter
- Scrape interval too long for short window (`[1m]` with 30s scrape is noisy)

Fix: Use Grafana Explore to search metric prefixes (e.g., `vllm:`, `smg_`)

### 3. Histogram percentiles look "flat" or wrong

Correct pattern:
```promql
histogram_quantile(0.95,
  sum by (le) (rate(metric_bucket[5m]))
)
```
Must use `rate()` on `_bucket` suffix, then `sum by (le)`, then `histogram_quantile()`.

### 4. Cardinality explosion (Prometheus memory spikes)
- Root cause: high-cardinality labels like `prompt`, `user_id`, or request IDs
- Fix: Remove high-cardinality labels; use logs/traces for per-request debugging

---

## Minimal "day-1" dashboard and alert set

**Dashboard panels (6 required):**
1. p95 request latency
2. p95 mean time per token
3. Queue size
4. p95 queue duration
5. Error rate
6. KV cache usage %

**Alerts (5 required):**
- p95 request latency > X for 10m
- p99 queue duration > Y for 10m
- Error rate > 1% for 5m
- KV cache usage > 90% for 15m
- Prometheus target DOWN (always)
