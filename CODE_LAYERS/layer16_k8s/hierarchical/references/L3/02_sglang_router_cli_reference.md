# SGLang Router — CLI and Configuration Reference

**Source:** https://sgl-project.github.io/advanced_features/router.html (redirects to docs.sglang.io)
**Also:** https://docs.sglang.io/advanced_features/sgl_model_gateway.html
**Author:** SGLang / LMSYS
**Level:** L3 — Mechanism level
**Why here:** Primary production reference for the router and policies that Layer 15 directly mirrors. Policy names, parameter names, and defaults in `config.yml` come from this document.

---

## Overview

The SGLang Router is a high-performance request distribution system that routes inference requests across multiple SGLang runtime instances. Features:

- **Cache-Aware Load Balancing**: Optimizes cache utilization while maintaining balanced load distribution.
- **Fault Tolerance**: Automatic worker health monitoring, failure detection, and recovery.
- **Kubernetes Integration**: Native service discovery and pod management.
- **Prometheus Metrics**: Built-in observability and monitoring.

---

## Installation

```bash
pip install sglang-router
```

---

## Quick Start

```bash
# Co-launch router and workers
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000

# Separate launch (router only)
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --host 0.0.0.0 \
  --port 30000
```

---

## Deployment Modes

1. **Co-launch Mode**: Router and workers launch together (simplest for single-node deployments).
2. **Separate Launch Mode**: Router and workers run as independent processes (Layer 15 default).
3. **Kubernetes Mode**: Workers discovered via label selectors instead of static URLs.

---

## Configuration Reference

### Core Settings

| Parameter | Type | Default | Layer 15 `config.yml` |
|---|---|---|---|
| `--host` | str | `127.0.0.1` | `router.host: "0.0.0.0"` |
| `--port` | int | `30000` | `router.port: 8200` |
| `--worker-urls` | list | `[]` | `router.workers[*].url` |
| `--policy` | str | `cache_aware` | `router.policy: prefix_cache_aware` |
| `--max-concurrent-requests` | int | `64` | (not in Layer 15) |
| `--request-timeout-secs` | int | `600` | (not in Layer 15) |
| `--max-payload-size` | int | `256MB` | (not in Layer 15) |

### Cache-Aware Routing Parameters

| Parameter | Type | Default | Layer 15 `config.yml` |
|---|---|---|---|
| `--cache-threshold` | float | `0.5` | `router.cache_threshold: 0.5` |
| `--balance-abs-threshold` | int | `32` | `router.balance_abs_threshold: 32` |

### Policy Values

| SGLang `--policy` value | Layer 15 `config.yml` value | Class |
|---|---|---|
| `random` | (not exposed) | Random selection |
| `round_robin` | `round_robin` | `RoundRobinPolicy` |
| `cache_aware` | `prefix_cache_aware` | `PrefixCacheAwarePolicy` |
| `power_of_two` | `least_load` | `LeastLoadPolicy` |

### Kubernetes Integration

| Parameter | Description |
|---|---|
| `--service-discovery` | Enable Kubernetes pod discovery |
| `--service-discovery-namespace` | Kubernetes namespace to watch |
| `--bootstrap-port-annotation` | Annotation for bootstrap ports |

### Observability

| Parameter | Default | Description |
|---|---|---|
| `--prometheus-port` | `29000` | Prometheus metrics port |
| `--prometheus-host` | `127.0.0.1` | Prometheus metrics host |
| `--log-dir` | `None` | Directory for log files |
| `--log-level` | `info` | Logging level |

---

## Source Code Mapping

The SGLang router is implemented in Rust:

| File | Layer 15 equivalent |
|---|---|
| `sgl-model-gateway/src/policies/round_robin.rs` | `router.py` `RoundRobinPolicy` |
| `sgl-model-gateway/src/policies/power_of_two.rs` | `router.py` `LeastLoadPolicy` |
| `sgl-model-gateway/src/policies/cache_aware.rs` | `router.py` `PrefixCacheAwarePolicy` |
| `sgl-model-gateway/src/core/worker.rs` | `router.py` `Worker` dataclass |
| `sgl-model-gateway/src/routers/http/router.rs` | `router.py` `Router.route()` |

---

## Debug Mode

```bash
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --log-level debug \
  --log-dir ./router_logs
```

Layer 15's equivalent: `log_level: debug` in `config.yml` (passed to Python's `logging.basicConfig`).
