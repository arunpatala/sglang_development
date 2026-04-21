# SGLang Router CLI Reference — Kubernetes Integration

**Source:** https://docs.sglang.ai/advanced_features/router.html (also: https://sgl-project.github.io/advanced_features/router.html)
**Author:** SGLang Project
**Level:** L3 — Technical CLI reference
**Why here:** The definitive reference for all `--service-discovery` flags used in Layer 16's `05_router_deployment.md`. Explains the complete parameter table, PD mode service discovery, and confirms that `--worker-urls` and `--service-discovery` are mutually exclusive.

---

## Overview

SGLang Router is a high-performance request distribution system that routes inference requests across multiple SGLang runtime instances. It features:
- Cache-Aware Load Balancing
- Fault tolerance
- Kubernetes native service discovery and pod management

---

## Kubernetes Service Discovery Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--service-discovery` | flag | False | Enable Kubernetes service discovery |
| `--selector` | list | [] | Label selector for workers (key1=value1 key2=value2) |
| `--service-discovery-namespace` | str | None | Kubernetes namespace to watch |
| `--service-discovery-port` | int | 80 | Port for discovered pods |
| `--prefill-selector` | list | [] | Label selector for prefill servers in PD mode |
| `--decode-selector` | list | [] | Label selector for decode servers in PD mode |
| `--bootstrap-port-annotation` | str | `sglang.ai/bootstrap-port` | Annotation for bootstrap port |

**Key rule:** `--worker-urls` and `--service-discovery` are mutually exclusive. Enabling `--service-discovery` replaces the static URL list entirely.

---

## Standard Mode with Service Discovery

```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --selector app=sglang-worker component=inference \
  --service-discovery-namespace production \
  --service-discovery-port 8000 \
  --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 32 \
  --host 0.0.0.0 --port 30000 \
  --prometheus-port 29000
```

This is the exact command used in Layer 16's `05_router_deployment.md`.

---

## PD Mode with Service Discovery

For deployments with separate prefill and decode server pools:

```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --service-discovery \
  --prefill-selector app=prefill-server tier=gpu \
  --decode-selector app=decode-server tier=cpu \
  --service-discovery-namespace production \
  --service-discovery-port 8000
```

---

## Load Balancing Policies

### Round Robin
```bash
--policy round_robin
```

### Least Load
```bash
--policy least_load
```

### Power of Two Choices
```bash
--policy power_of_two
```
Samples two workers and routes to the less loaded one.

### Cache-Aware Load Balancing (Default — Layer 16)
```bash
--policy cache_aware \
--cache-threshold 0.5 \
--balance-abs-threshold 32 \
--balance-rel-threshold 1.0001
```

**How it works:**
1. **Load Assessment**: Checks if the system is balanced
2. **Routing Decision**:
   - **Balanced System**: Uses cache-aware routing → routes to worker with highest prefix match if match > `cache_threshold`
   - **Imbalanced System**: Uses shortest queue routing to the least busy worker
3. **Cache Management**: Maintains approximate radix trees per worker; periodically evicts LRU entries

---

## Cache-Aware Policy Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--cache-threshold` | float | 0.5 | Minimum prefix match ratio to use cache routing |
| `--balance-abs-threshold` | int | 32 | Absolute load difference threshold for imbalance |
| `--balance-rel-threshold` | float | 1.0001 | Relative load ratio threshold |
| `--eviction-interval-secs` | int | 60 | Seconds between cache eviction cycles |
| `--max-tree-size` | int | 16777216 | Maximum nodes in routing tree |

---

## Fault Tolerance Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `--worker-startup-timeout` | int | 300 | Timeout for worker startup |
| `--worker-startup-check-interval` | int | 10 | Interval between startup checks |
| `--retry-failed-requests` | flag | False | Retry failed requests on other workers |

---

## Data Parallelism Aware Routing

```bash
python -m sglang_router.launch_router \
  --dp-awareness \
  --dp-controller-host localhost \
  --dp-controller-port 9000
```

Enables fine-grained control over data parallel replicas, coordinating with SGLang's DP controller for optimized request distribution.

---

## Multi-Model Support

One router instance can handle many model families:

```bash
python -m sglang_router.launch_router \
  --model-registry \
  --policy cache_aware
```

- Policy registry allows multiple radix trees and different policies per model family.
- OpenAI-compatible endpoints (`/v1/chat/completions`, `/v1/embeddings`, `/v1/rerank`) are all supported.
- Workers are registered per-model; routing decisions are model-aware.

---

## SGLang Kubernetes Service Discovery History

This feature originated from GitHub issue #3073 (January 2025):
- **Motivation**: "Service discovery will enable the router to dynamically identify and connect to backend services running in a Kubernetes cluster"
- **Closed**: March 2025, when the feature was merged
- **Implementation**: `REPOS/sglang/sgl-model-gateway/src/service_discovery.rs`

The feature replaced `--worker-urls` entirely — the router starts with zero static URLs and builds its worker list from the K8s API.
