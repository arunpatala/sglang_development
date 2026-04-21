# Master KV Cache Aware Routing with llm-d for Efficient AI Inference

**Source:** https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
**Author:** Red Hat Developer
**Date:** October 7, 2025
**Level:** L2 — Definitions + motivation
**Why here:** Practitioner walkthrough of llm-d KV cache routing with concrete configuration examples. The 87.4% cache hit rate benchmark and the TTFT improvement table give concrete numbers to motivate `PrefixCacheAwarePolicy`.

---

## Summary

In the era of large-scale AI inference, ensuring efficiency across distributed environments is essential. llm-d is a Kubernetes-native framework for scalable, intelligent LLM inference. One of its most powerful capabilities is KV cache aware routing, which reduces latency and improves throughput by directing requests to pods that already hold relevant context in GPU memory.

---

## Why Stateless Inference Fails to Reuse Cache

In traditional deployments, even if KV caches are enabled inside the model server (like vLLM), the gateway is unaware of the cache state. This leads to:
- Round-robin routing or explicit sticky sessions
- Frequent cache misses
- Repeated computation for common prefixes

---

## How llm-d Enables Cache-Aware Routing

llm-d introduces the **Gateway API Inference Extension (GAIE)** with an Endpoint Picker Plugin (EPP):

- **Session-aware routing**: Maintains request consistency for optimal cache reuse.
- **Prefix-aware scoring**: Routes requests based on prompt similarity and cache warmth.

```yaml
# plugins.yaml configuration
plugins:
  - name: "cache-aware-router"
    type: "external_processor"
    config:
      discovery:
        label_selector: "llm-d.ai/inferenceServing=true"
      cache:
        type: "in-memory-lru"
        max_size: 10000
      routing:
        algorithm: "prefix-aware"
        session_affinity: true
```

---

## vLLM Pod Configuration for Optimal Prefix Caching

```yaml
args:
  - "--enable-prefix-caching"          # Enable KV-cache prefix reuse
  - "--block-size=16"                   # Optimal block size for cache efficiency
  - "--gpu-memory-utilization=0.7"     # Reserve memory for cache storage
  - "--max-model-len=4096"             # Match expected prompt lengths
  - "--kv-cache-dtype=auto"            # Automatic cache data type optimization
```

---

## Performance Results

**Total queries: 4,776 | Cache hits: 4,176 | Cache hit rate: 87.4%**

| Scenario | Without cache routing | With KV cache routing | Improvement |
|---|---|---|---|
| Cold inference | 2,850 ms TTFT | 2,850 ms TTFT | Baseline |
| Warm cache hit | 2,850 ms TTFT | ~285 ms TTFT | **10× faster** |
| Multi-turn conversation | 2,200 ms avg TTFT | 310 ms avg TTFT | **7× faster** |

---

## Why This Matters

The 87.4% cache hit rate translates into tangible business value:
- **Cost reduction**: Cached tokens cost 10× less than uncached tokens.
- **Latency**: Multi-turn conversations are 7× faster on average.
- **Throughput**: Fewer GPU cycles wasted on redundant prefill.

Use cases where this matters most:
- Customer service bots with long shared system prompts
- Code generation assistants with shared context
- Multi-tenant SaaS: shared prompt patterns benefit all users

---

## Routing Strategy Hierarchy

1. **Round-robin**: Simple, ignores cache state.
2. **Session-based (sticky sessions)**: Keeps one user on one pod. Helps multi-turn but misses cross-user cache reuse.
3. **Approximate prefix-cache**: Estimates cache state from routing history. Works at low scale.
4. **Precise prefix-cache aware (llm-d)**: Direct introspection into vLLM KV cache via KVEvents. Most effective.

---

## Relevance to Layer 15

The 87.4% cache hit rate at 87.4% is the production target that `PrefixCacheAwarePolicy` in `router.py` approximates at teaching scale:
- llm-d's `kvcache.Index` ↔ Layer 15's `RadixTrie` per worker
- llm-d's `Precise Prefix-Cache Scorer` ↔ Layer 15's `PrefixCacheAwarePolicy._pick_worker`
- llm-d's load-awareness gate ↔ Layer 15's `balance_abs_threshold` guard
