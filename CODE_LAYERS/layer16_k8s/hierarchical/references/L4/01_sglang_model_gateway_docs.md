# SGLang Model Gateway — Full Documentation

**Source:** https://docs.sglang.io/advanced_features/sgl_model_gateway.html
**Author:** SGLang / LMSYS
**Level:** L4 — Production + systems
**Why here:** Primary production reference. Layer 15's `router.py` is a direct Python teaching-scale port of this Rust gateway. All policy names, parameters, and architecture decisions in Layer 15 trace back to this document.

---

## Overview

SGLang Model Gateway is a high-performance model-routing gateway for large-scale LLM deployments. It:
- Centralizes worker lifecycle management
- Balances traffic across heterogeneous protocols (HTTP, gRPC, OpenAI-compatible)
- Provides enterprise-ready control over history storage, MCP tooling, and privacy-sensitive workflows

---

## Architecture

### Control Plane

- **Worker Manager**: Discovers capabilities (`/server_info`, `/get_model_info`), tracks load, registers/removes workers.
- **Job Queue**: Serializes add/remove requests, exposes status via `/workers/{worker_id}`.
- **Load Monitor**: Feeds cache-aware and power-of-two policies with live worker load statistics.
- **Health Checker**: Continuously probes workers and updates readiness, circuit breaker state, and metrics.
- **Tokenizer Registry**: Manages dynamically registered tokenizers.

### Data Plane

- **HTTP routers** (regular & PD): Implement `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/classify`, and admin endpoints.
- **gRPC router**: Streams tokenized requests directly to SRT gRPC workers in Rust — tokenizer, reasoning parser, and tool parser in-process.
- **OpenAI router**: Proxies OpenAI-compatible endpoints while keeping chat history and MCP sessions local.

---

## Quick Start

```bash
# Separate Launch (HTTP) — mirrors Layer 15's config.yml setup
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8000
python -m sglang.launch_server --model meta-llama/Meta-Llama-3.1-8B-Instruct --port 8001

python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 --port 30000
```

---

## Load Balancing Policies

| Policy | Description | Layer 15 equivalent |
|---|---|---|
| `random` | Uniform random selection | — (not exposed) |
| `round_robin` | Cycles through workers in order | `RoundRobinPolicy` |
| `power_of_two` | Samples two workers, picks the lighter one | `LeastLoadPolicy` |
| `cache_aware` | Combines cache locality with load balancing (default) | `PrefixCacheAwarePolicy` |
| `bucket` | Divides workers into load buckets with dynamic boundaries | — (not in Layer 15) |

### Cache-Aware Policy Tuning

```bash
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.5 \
--eviction-interval-secs 120 \
--max-tree-size 67108864
```

| Parameter | Default | Description | Layer 15 `config.yml` |
|---|---|---|---|
| `--cache-threshold` | `0.3` | Min prefix match ratio for cache hit | `cache_threshold: 0.5` |
| `--balance-abs-threshold` | `64` | Absolute load difference before rebalancing | `balance_abs_threshold: 32` |
| `--balance-rel-threshold` | `1.5` | Relative load ratio before rebalancing | (not in Layer 15) |
| `--eviction-interval-secs` | `120` | Cache eviction cadence in seconds | (not in Layer 15) |
| `--max-tree-size` | `67108864` | Maximum nodes in cache tree | (not in Layer 15) |

---

## Reliability and Flow Control

### Retries (not in Layer 15)

```bash
--retry-max-retries 5 \
--retry-initial-backoff-ms 50 \
--retry-max-backoff-ms 30000 \
--retry-backoff-multiplier 1.5 \
--retry-jitter-factor 0.2
```

Retryable status codes: 408, 429, 500, 502, 503, 504.

### Circuit Breaker (not in Layer 15)

Per-worker circuit breakers prevent cascading failures:

```bash
--cb-failure-threshold 5 \
--cb-success-threshold 2 \
--cb-timeout-duration-secs 30 \
--cb-window-duration-secs 60
```

States: `Closed` (normal) → `Open` (failures) → `HalfOpen` (testing recovery).

Layer 15's health check loop (`_health_loop`) is the simplified version: binary healthy/unhealthy, no three-state machine.

---

## Worker Management API

```bash
# List workers
curl http://localhost:30000/workers

# Response
{
  "workers": [{
    "id": "2f3a0c3e-...",
    "url": "http://0.0.0.0:31378",
    "model_id": "mistral",
    "is_healthy": true,
    "load": 0,
    "worker_type": "regular"
  }]
}
```

↔ Layer 15's `/health` endpoint returns similar per-worker state.

---

## Deployment Modes

### Separate Launch (HTTP) — Layer 15 mode

```bash
python -m sglang.launch_server --model ... --port 8000
python -m sglang.launch_server --model ... --port 8001
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware
```

### Prefill-Decode Disaggregation — Layer 16 concept

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```

Prefill and decode workers are separate pools. The router orchestrates the two-phase handoff. This is the "what comes next" from `lesson/09_whats_next.md`.

---

## Observability

- **40+ Prometheus metrics** at `--prometheus-port` (default 29000)
- **OpenTelemetry distributed tracing**
- **Structured logging** with request ID propagation

Key metrics:
- `sgl_request_total` — total requests by policy
- `sgl_request_duration_seconds` — latency histogram
- `sgl_worker_load` — per-worker queue depth
- `sgl_cache_hit_rate` — prefix cache hit ratio

Layer 15 replaces all of these with Python `logging.getLogger()`.

---

## Source Code Mapping

| SGLang gateway file | Layer 15 `router.py` |
|---|---|
| `src/policies/round_robin.rs` | `RoundRobinPolicy` (lines 216–231) |
| `src/policies/power_of_two.rs` | `LeastLoadPolicy` (lines 242–251) |
| `src/policies/cache_aware.rs` | `PrefixCacheAwarePolicy` (lines 279–331) |
| `src/core/worker.rs` | `Worker` dataclass (lines 108–112) |
| `src/routers/http/router.rs` | `Router.route()` (lines 354–436) |
| Health checker background loop | `_health_loop` (lines 439–446) |
| Circuit breaker | (not in Layer 15) |
| Prometheus metrics | (not in Layer 15) |
| gRPC router | (not in Layer 15) |
