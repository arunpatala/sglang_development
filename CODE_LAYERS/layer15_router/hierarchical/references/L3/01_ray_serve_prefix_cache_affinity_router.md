# Prefix-Aware Routing — Ray Serve

**Source:** https://docs.ray.io/en/releases-2.54.0/serve/llm/user-guides/prefix-aware-routing.html
**Author:** Ray / Anyscale
**Version:** Ray 2.54.0 (alpha API)
**Level:** L3 — Mechanism level (pseudocode, invariants, configuration)
**Why here:** Best concise technical description of the two-guard policy design (load check → prefix match). Includes runnable Python config and a full parameter reference. The `imbalanced_threshold` / `match_rate_threshold` pair directly maps to Layer 15's `balance_abs_threshold` / `cache_threshold`.

---

## Overview

LLM inference can benefit significantly from cache locality optimization. When one replica processes multiple prompts that share a prefix, the engine can reuse previously computed KV-cache entries, reducing computation overhead and improving response times. This is known as Automatic Prefix Caching (APC) in vLLM.

The `PrefixCacheAffinityRouter` routes requests with similar prefixes to the same replicas, maximizing KV cache hit rates.

> **Warning**: This API is in alpha and may change before becoming stable.

---

## When to Use Prefix-Aware Routing

Use prefix-aware routing when:
- Your workload has many requests with shared prefixes (same system prompts or few-shot examples).
- You're using vLLM with Automatic Prefix Caching enabled.
- Cache hit rate is more important than perfect load balance in balanced scenarios.

---

## How It Works — Three-Tier Routing Strategy

### 1. Load Balance Check

First, it evaluates whether the current load is balanced across replicas by comparing queue lengths. If the difference between the highest and lowest queue lengths is below `imbalanced_threshold`, it proceeds with prefix cache-aware routing.

↔ **Layer 15**: `abs(max_load - min_load) > balance_abs_threshold` → skip cache, route to least-loaded.

### 2. Prefix Matching Strategy (when load is balanced)

Uses a prefix tree to find replicas that have previously processed similar input text:
- **High match rate (≥ `match_rate_threshold`)**: Routes to replicas with the highest prefix match rate for better cache hit rates.
- **Low match rate (< `match_rate_threshold`)**: Falls back to replicas with the lowest prefix cache utilization to increase utilization.
- **No prefix data**: Uses the default Power of Two Choices selection.

↔ **Layer 15**: `match_len / len(prompt) >= cache_threshold` → pick best-match worker; otherwise pick smallest trie (most free cache).

### 3. Imbalanced Load Fallback

When load is imbalanced (queue length difference exceeds `imbalanced_threshold`), the router **prioritizes load balancing over cache locality** and falls back to Power of Two Choices.

↔ **Layer 15**: `LeastLoadPolicy._pick_worker` used as fallback when imbalanced.

### Prefix Tree Management

The router maintains a **distributed prefix tree actor** that:
- Tracks input text prefixes processed by each replica.
- Supports automatic eviction of old entries (LRU-like) to manage memory usage.
- Persists across router instances using Ray's detached actor pattern.

↔ **Layer 15**: One `RadixTrie` per worker in `PrefixCacheAwarePolicy._tries`. Layer 15's trie is in-process; Ray's is a distributed actor.

---

## Deployment Example

```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
from ray.serve.llm.request_router import PrefixCacheAffinityRouter

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "qwen-0.5b",
        "model_source": "Qwen/Qwen2.5-0.5B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {
            "min_replicas": 4,
            "max_replicas": 4,
        },
        "request_router_config": {
            "request_router_class": PrefixCacheAffinityRouter,
            "request_router_kwargs": {
                "imbalanced_threshold": 5,       # Load balance aggressiveness
                "match_rate_threshold": 0.15,    # Require 15% match for cache routing
                "do_eviction": True,             # Enable memory management
                "eviction_threshold_chars": 500_000,
                "eviction_target_chars": 400_000,
                "eviction_interval_secs": 30,
            },
        },
    },
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
)

app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

---

## Configuration Parameters

### Core Routing Parameters

| Parameter | Default | Description |
|---|---|---|
| `imbalanced_threshold` | `infinity` | Queue length gap threshold. Lower → prioritize load balance over cache. ↔ `balance_abs_threshold` in `config.yml` |
| `match_rate_threshold` | `0.1` | Min prefix match rate (0.0–1.0) required for cache-affinity routing. ↔ `cache_threshold` in `config.yml` |

### Memory Management Parameters

| Parameter | Default | Description |
|---|---|---|
| `do_eviction` | `False` | Enable automatic eviction of old prefix tree entries. |
| `eviction_threshold_chars` | `400,000` | Max characters before eviction is triggered. |
| `eviction_target_chars` | `360,000` | Target characters after eviction. |
| `eviction_interval_secs` | `10` | Interval between eviction checks. |

Layer 15's `RadixTrie` does not implement eviction (teaching scope). Production systems require it.

---

## Best Practices

- Enable vLLM APC: Set `enable_prefix_caching=True` in `engine_kwargs` for the router to have any effect.
- Tune thresholds: Adjust `imbalanced_threshold` and `match_rate_threshold` based on workload characteristics.
- Monitor cache hit rates: Track vLLM's cache hit metrics to verify the router is improving performance.
- Start conservative: Begin with default settings and tune incrementally.

---

## Parameter Mapping: Ray Serve ↔ Layer 15

| Ray Serve | Layer 15 `config.yml` | SGLang Gateway |
|---|---|---|
| `imbalanced_threshold` | `balance_abs_threshold: 32` | `--balance-abs-threshold` |
| `match_rate_threshold` | `cache_threshold: 0.5` | `--cache-threshold` |
| Power of Two Choices fallback | `LeastLoadPolicy` | `power_of_two` policy |
| Distributed prefix tree actor | `RadixTrie` per worker (in-process) | Per-worker tree in Rust |
