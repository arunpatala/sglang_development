# KV-Cache Wins You Can See: From Prefix Caching in vLLM to Distributed Scheduling with llm-d

**Source:** https://llm-d.ai/blog/kvcache-wins-you-can-see
**Author:** llm-d project team
**Date:** 2025
**Level:** L2 — Definitions + motivation
**Why here:** Quantifies the cost of cache-blind routing in distributed deployments. The 57× TTFT gap between approximate and precise scheduling is the strongest motivation for `PrefixCacheAwarePolicy`. Benchmark details directly back `cache_threshold` and `balance_abs_threshold` design choices.

---

## Summary

The llm-d project provides a series of "well-lit paths" for deploying LLMs in production. This blog covers the transition from approximate to precise prefix-cache aware scheduling, demonstrating order-of-magnitude performance improvements.

**Key takeaways:**
- KV-cache hit rates directly impact cost: 10× difference between cached ($0.30/M) and uncached ($3.00/M) tokens.
- vLLM's prefix caching **breaks in distributed deployments**: standard load balancers scatter related requests across pods.
- Precise prefix-cache aware scheduling delivers **57× faster TTFT** and **double the throughput** on identical hardware.

---

## The Most Important Metric in Production AI

> "The KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost." — Manus, Context Engineering for AI Agents

In a single-instance environment, vLLM's Automatic Prefix Caching cuts redundant work. But the moment you scale to distributed, multi-replica environments, these optimizations fall apart.

---

## Inside vLLM: Prefix Caching in a Single Instance

**The KV-cache:**
- Self-attention computes Key (K) and Value (V) tensors for every input token.
- These tensors are stored in the KV-cache — the model's short-term memory.
- For subsequent token generation (decode), the model pulls existing values from memory rather than recomputing.

**Automatic Prefix Caching (APC):**
- vLLM identifies when requests share the same token sequence prefix.
- It reuses the same memory pages via hash-based block matching.
- Result: TTFT drops from 4.3 seconds to 0.6 seconds on a ~10,000 token prompt (second request).

---

## The Challenge of Scale-Out

When scaling to a distributed cluster, the unified KV-cache becomes **disaggregated**. Each vLLM pod manages its own cache in isolation. Standard load balancers spread traffic using cache-blind metrics, scattering related requests and destroying cache locality.

**The cascade of failures:**
1. Cache Miss: The warm cache benefit on Pod A is lost when the next request goes to Pod B.
2. Duplicated Work: The most expensive computation is performed twice.
3. Increased Latency: Higher TTFT.
4. Wasted GPU Resources: Hardware is tied up re-doing work instead of serving new requests.

This isn't a rare event in production — it's the **default behavior** of any distributed deployment with a stateless load balancer.

---

## llm-d: Precise Prefix-Cache Aware Scheduling

llm-d creates a **global view of the cluster's KV-cache**, allowing it to treat the disaggregated memory as a single, manageable pool.

### How It Works: KVEvents

Each vLLM pod continuously emits `KVEvents` — live feed of all physical cache changes:
1. `kvevents.Pool`: Consumes the event stream, maintains a KV-Block Index (map of block-hashes to pod + memory medium).
2. `kvcache.Index`: Higher-level index used by the scheduler. Maps logical token sequences (prefixes) to pods.

### The Precise Prefix-Cache Scorer

For every incoming request, the scorer:
1. Retrieves the most extended cached token sequence from `PrefixStore`.
2. Outputs a "cache affinity score" for each pod — directly representing the computational work that can be saved.
3. Combines the cache affinity score with load awareness to make the final routing decision.

---

## Performance Results

**Benchmark: 8 vLLM pods (16 H100 GPUs total), B2B SaaS workload**
- 150 enterprise customers, each with 6,000-token system prompts
- 5 concurrent users per customer, 1,200-token queries
- Load ramping from 3 QPS to 60 QPS

| Strategy | Output toks/s | TTFT p90 (s) | TTFT mean (s) | vLLM Wait Queue (mean) |
|---|---|---|---|---|
| `precise-scheduling` | **8730.0** | **0.542** | **0.298** | **0.1** |
| `approximate-scheduling` | 6944.4 | 31.083 | 13.316 | 8.1 |
| `load-scheduling` | 4428.7 | 94.865 | 46.987 | 28.9 |

**`precise-scheduling` is 57× faster TTFT than `approximate-scheduling`.**

### Why Such a Large Gap?

Cache-blind schedulers constantly duplicate and evict the same prefixes across different pods — "cache thrashing." `precise-scheduling` avoids this entirely by routing requests for cache-hits consistently, resulting in virtually no queues and stable throughput.

---

## Scheduling Strategy Hierarchy

1. **Random/Round-Robin**: Works for symmetric workloads with minimal cache reuse.
2. **Load-Aware**: Prevents overload by routing based on pod serving capacity.
3. **Approximate Prefix-Cache**: Introduces cache-awareness but estimates can become unreliable at high scale.
4. **Precise Prefix-Cache Aware**: Most effective for dynamic, high-scale workloads where maximizing cache-hit ratio is the primary driver.

---

## Relevance to Layer 15

The benchmark validates Layer 15's design choices:
- **`RoundRobinPolicy`**: Corresponds to "Random/Round-Robin" — the baseline that destroys cache locality.
- **`LeastLoadPolicy`**: Corresponds to "Load-Aware" — better than round-robin for unequal request sizes.
- **`PrefixCacheAwarePolicy`**: Corresponds to "Precise Prefix-Cache Aware" — the `RadixTrie` in `router.py` is the router-side approximation of llm-d's global `kvcache.Index`.
- The `balance_abs_threshold` guard in `PrefixCacheAwarePolicy` directly mirrors llm-d's load-awareness gate.
