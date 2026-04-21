# SGLang Model Gateway — Full Documentation

**Source:** https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/sgl_model_gateway.md
**Raw:** https://raw.githubusercontent.com/sgl-project/sglang/main/docs/advanced_features/sgl_model_gateway.md
**Author:** SGLang Project
**Level:** L4 — Advanced production reference
**Why here:** The definitive reference for the SGLang Model Gateway (sgl-model-gateway), which is what Layer 16's router is built on. Contains the complete service discovery configuration (Section 13), pod labelling guide, RBAC requirements, all 40+ Prometheus metrics, and production recommendations for HA deployment.

---

## Overview

SGLang Model Gateway is a high-performance model-routing gateway for large-scale LLM deployments. It:
- Centralizes worker lifecycle management
- Balances traffic across HTTP, gRPC, and OpenAI-compatible protocols
- Provides enterprise-ready reliability: retries with jitter, circuit breakers, rate limiting, health checks
- Exports 40+ Prometheus metrics at `--prometheus-port` (default: 29000)
- Supports native K8s service discovery (`--service-discovery`)

**Architecture:**
- **Control Plane**: Worker Manager (discover workers via `/server_info`), Job Queue, Load Monitor, Health Checker, Tokenizer Registry
- **Data Plane**: HTTP routers (regular & PD), gRPC router, OpenAI proxy

---

## Service Discovery (Kubernetes) — Section 13

Enable automatic worker discovery via Kubernetes pod selectors:

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker role=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

### PD Mode Discovery

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --service-discovery \
  --prefill-selector app=sglang component=prefill \
  --decode-selector app=sglang component=decode \
  --service-discovery-namespace production
```

Prefill pods can expose bootstrap ports via the `sglang.ai/bootstrap-port` annotation.

### Pod Labelling for Service Discovery

Worker pods must have consistent labels that match the router's `--selector`:

```yaml
# Worker Deployment (Regular Mode)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sglang-worker
  namespace: production
spec:
  replicas: 4
  selector:
    matchLabels:
      app: sglang-worker
      component: inference
  template:
    metadata:
      labels:
        app: sglang-worker
        component: inference
        model: llama-3-8b
    spec:
      containers:
        - name: worker
          image: lmsysorg/sglang:latest
          ports:
            - containerPort: 8000
              name: http
            - containerPort: 20000
              name: grpc
```

**Best practice:** "Use Kubernetes Service Discovery: Let the gateway automatically discover and manage workers."

### RBAC Requirements

```yaml
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: sglang-gateway
  namespace: production
rules:
  - apiGroups: [""]
    resources: ["pods"]
    verbs: ["get", "list", "watch"]
---
apiVersion: rbac.authorization.k8s.io/v1
kind: RoleBinding
metadata:
  name: sglang-gateway
  namespace: production
subjects:
  - kind: ServiceAccount
    name: sglang-gateway
    namespace: production
roleRef:
  kind: Role
  name: sglang-gateway
  apiGroup: rbac.authorization.k8s.io
```

---

## All 40+ Prometheus Metrics (grouped)

### HTTP Request Metrics
| Metric | Type | Description |
|---|---|---|
| `smg_http_requests_total` | Counter | Total HTTP requests by path/method |
| `smg_http_request_duration_seconds` | Histogram | Request latency distribution |
| `smg_http_responses_total` | Counter | Responses by status code |
| `smg_http_rate_limit_total` | Counter | Rate limit decisions (allowed/rejected) |

### Worker Pool Metrics
| Metric | Type | Description |
|---|---|---|
| `smg_worker_pool_size` | Gauge | Count of healthy workers |
| `smg_worker_requests_active` | Gauge | Per-worker in-flight request count |
| `smg_worker_connections_active` | Gauge | Active connections per worker |
| `smg_worker_health_checks_total` | Counter | Health check results by worker/result |
| `smg_worker_cb_state` | Gauge | Circuit breaker: 0=Closed, 1=Open, 2=HalfOpen |
| `smg_worker_cb_transitions_total` | Counter | Circuit breaker state transitions |
| `smg_worker_retries_total` | Counter | Retry attempts per worker |
| `smg_worker_retries_exhausted_total` | Counter | Retries that exhausted all attempts |

### Service Discovery Metrics
| Metric | Type | Description |
|---|---|---|
| `smg_discovery_registrations_total` | Counter | Pods registered: `{source="kubernetes", result="success\|failed\|duplicate"}` |
| `smg_discovery_workers_discovered` | Gauge | Currently known workers from K8s |
| `smg_discovery_sync_duration_seconds` | Histogram | Time for reconciliation cycle |

### Inference Metrics (gRPC mode)
| Metric | Type | Description |
|---|---|---|
| `smg_router_ttft_seconds` | Histogram | Time to first token by model |
| `smg_router_tpot_seconds` | Histogram | Time per output token by model |
| `smg_router_tokens_total` | Counter | Token throughput by model/direction |
| `smg_router_generation_duration_seconds` | Histogram | Total generation time |

---

## Essential PromQL Queries

### Request rate and latency
```promql
# Request rate by endpoint
sum(rate(smg_http_requests_total[5m])) by (path, method)

# P99 latency
histogram_quantile(0.99, sum(rate(smg_http_request_duration_seconds_bucket[5m])) by (le))

# Error rate
sum(rate(smg_http_responses_total{status=~"5.."}[5m])) / sum(rate(smg_http_responses_total[5m]))
```

### Worker health
```promql
# Healthy workers (alert if 0)
sum(smg_worker_pool_size)

# Workers with open circuit breaker
count(smg_worker_cb_state == 1)

# Health check failure rate
sum(rate(smg_worker_health_checks_total{result="failure"}[5m])) by (worker_id)
```

### Service discovery
```promql
# Workers currently discovered from K8s
smg_discovery_workers_discovered{source="kubernetes"}

# Discovery failures
sum(rate(smg_discovery_registrations_total{result="failed"}[5m]))
```

---

## Production Alerting Rules (from SGLang docs)

```yaml
groups:
  - name: sglang-gateway
    rules:
      - alert: NoHealthyWorkers
        expr: sum(smg_worker_pool_size) == 0
        for: 1m
        severity: critical

      - alert: DiscoveryZeroWorkers
        expr: smg_discovery_workers_discovered{source="kubernetes"} == 0
        for: 2m
        severity: warning

      - alert: HighErrorRate
        expr: >
          sum(rate(smg_http_responses_total{status=~"5.."}[5m]))
          / sum(rate(smg_http_responses_total[5m])) > 0.05
        for: 5m
        severity: critical

      - alert: CircuitBreakerOpen
        expr: count(smg_worker_cb_state == 1) > 0
        for: 2m
        severity: warning
```

---

## High Availability Section

### Multiple Router Replicas

The gateway supports HA via multiple replicas. As of early 2026, radix-tree state synchronization across replicas is in development (see issue #10839 / PR #14108). Each replica independently:
- Runs its own watcher loop
- Builds its own worker list
- Maintains its own radix tree

### HA CLI flags

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker \
  --service-discovery-namespace production \
  --ha-server-port 9100 \          # Router mesh port for state sync
  --ha-server-peers "router-1:9100,router-2:9100,router-3:9100"
```

### Production Recommendations

1. **Multiple replicas behind a load balancer or Service** — run 2–3 router replicas; tolerate 10–20% cache degradation vs. single router
2. **Session affinity** — configure ingress to stick users to a single router replica (preserves cache hit rate per user)
3. **Prometheus scraping** — expose `--prometheus-port` on a separate Service; use ServiceMonitor with `monitoring: enabled` label
4. **Readiness probe** — router's `/readiness` returns 503 until at least one healthy worker is registered; use as K8s readiness probe

---

## Load Balancing Policies

| Policy | Flag | Best for |
|---|---|---|
| Round robin | `--policy round_robin` | Uniform workloads, debugging |
| Least load | `--policy least_load` | Uneven request sizes |
| Power of two | `--policy power_of_two` | Good balance, less tree overhead |
| Cache-aware | `--policy cache_aware` | Repeated/similar prompts, prefix caching workloads |

### Cache-aware parameters
```bash
--policy cache_aware \
--cache-threshold 0.5 \           # Minimum prefix match ratio to use cache routing
--balance-abs-threshold 32 \       # Absolute load difference for imbalance detection
--balance-rel-threshold 1.0001 \   # Relative load ratio for imbalance detection
--eviction-interval-secs 60 \      # LRU eviction cycle
--max-tree-size 16777216           # Max nodes per radix tree (16M)
```

---

## Deployment Modes

```bash
# Standard HTTP routing with K8s discovery (Layer 16 pattern)
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000 \
  --prometheus-port 29000

# gRPC routing (high throughput, lower latency)
python -m sglang_router.launch_router \
  --worker-urls grpc://worker1:20000 grpc://worker2:20000 \
  --model-path meta-llama/Llama-3.1-8B-Instruct \
  --reasoning-parser deepseek-r1 \
  --tool-call-parser json \
  --host 0.0.0.0 --port 8080

# PD disaggregation with service discovery (production LLM serving)
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --service-discovery \
  --prefill-selector app=sglang-worker component=prefill \
  --decode-selector app=sglang-worker component=decode \
  --service-discovery-namespace production
```
