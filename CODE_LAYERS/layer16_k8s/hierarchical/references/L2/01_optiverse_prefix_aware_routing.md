# Prefix-Aware Routing: Cache-Conscious Request Distribution

**Source:** https://www.optiversetech.com/blog/prefix-aware-routing
**Author:** Huang Tzu Lin (OptiVerse Technology)
**Date:** April 2026
**Level:** L2 — Definitions + motivation
**Why here:** Best standalone explainer on prefix-aware routing. Covers the cache-blind → cache-aware transition, RadixAttention context, and compares SGLang, vLLM, and Ray Serve implementations side by side. Essential background for `PrefixCacheAwarePolicy` in `router.py`.

---

## Summary

This post explains why traditional routing strategies (round-robin, least-loaded, power-of-two-choices) waste significant computation when applied to LLM inference, and what prefix-aware routing looks like in practice.

---

## How Traditional Routing Works

Three strategies dominate traditional web service load balancing:

- **Round-robin**: Each request goes to the next replica in a fixed cyclic order. Simple and predictable.
- **Least-loaded**: Routes to the replica with the shortest queue. Adapts to uneven processing times.
- **Power-of-two-choices**: Picks two replicas at random, routes to the less busy one. Near-optimal load distribution with minimal overhead.

All three share a fundamental assumption: **requests are stateless**. Any replica can handle any request equally well.

LLM inference is different.

---

## The Wasted Computation Problem

Every LLM request begins with a **prefill phase**: the model processes the entire input in a single forward pass, computing attention key and value vectors for every input token and storing them in the KV cache. Prefill is compute-bound and scales with input length.

**Example — multi-turn conversation:**

A travel agent sends five messages in a conversation. Each message includes the full conversation history. With round-robin routing across five replicas:
- Turn 1 goes to Replica A, Turn 2 to Replica B, Turn 3 to Replica C...
- Each replica processes the full input from scratch — no KV cache from previous turns.

Total prefill: **5,200 tokens, all computed from scratch**.

If every turn had been routed to the same replica:
- Turn 2 needs only 300 new tokens (not 700).
- Turn 5 needs only 300 new tokens (not 1,700).
- Total prefill drops to **1,700 tokens — a 67% reduction**.

At a prefill rate of ~10,000 tokens/sec, Turn 5 alone saves ~140ms TTFT.

---

## What Prefixes Are Shared

Two patterns dominate in production workloads:

1. **Multi-turn conversations**: Each turn includes the full conversation history as its prefix. The prefix grows with every turn. Chatbots, customer support copilots, coding assistants.

2. **Shared system prompts**: Many users of the same application share an identical system prompt. In production workloads — chatbots, coding assistants, enterprise copilots — prefix overlap rates routinely reach **40-80% of input tokens**.

---

## Prefix-Aware Routing: How It Works

The router maintains an index of which prefixes are cached on which replicas. When a new request arrives, the router:
1. Extracts the request's token prefix.
2. Queries the index for the replica with the longest matching cached prefix.
3. Routes the request there.
4. If no replica has a match, falls back to least-loaded.

> "Think of it as sorting mail by ZIP code. Traditional routing is like handing each letter to the next available mail carrier, regardless of the letter's destination. Prefix-aware routing is like giving the Kyoto letters to the carrier who already knows the Kyoto neighborhoods. The map is the KV cache. The ZIP code is the prefix."

---

## Three Production Implementations

### SGLang RadixAttention

SGLang maintains all cached KV entries in a **radix tree** — a compact prefix tree where each edge can represent a multi-token sequence.

When a request arrives, the runtime walks the tree from the root, matching the request's token prefix against existing edges. If the request shares its first 1,400 tokens with a previously processed request, the tree walk matches those 1,400 tokens and stops at the divergence point. KV cache blocks for the matched portion are reused. Only the unmatched suffix goes through prefill.

Results: up to **5× throughput improvement**, up to **6.4× on workloads with high prefix variation**.

### vLLM Automatic Prefix Caching (APC)

vLLM uses **hash-based block matching** instead of a tree. Each KV cache block (16 tokens by default) is associated with a hash of its token sequence. New requests trigger a hash lookup. If a matching block exists, it's reused without recomputation.

### Ray Serve PrefixCacheAffinityRouter

Ray Serve operates at the cluster routing level (distributing across multiple inference engine replicas). It implements a **two-guard strategy**:

1. **Load balance check**: Compare queue lengths across replicas. If the difference between the highest and lowest queue lengths is below `imbalanced_threshold`, proceed with prefix cache-aware routing.
2. **Prefix matching**: Among balanced replicas, find the one with the highest prefix match rate. If match rate ≥ `match_rate_threshold` (default: 0.1), route to that replica. Otherwise, route to the replica with the smallest prefix tree (most free cache).

This is exactly the two-guard pattern used in Layer 15's `PrefixCacheAwarePolicy`:
- `balance_abs_threshold` ↔ `imbalanced_threshold`
- `cache_threshold` ↔ `match_rate_threshold`

---

## Trade-offs

**Cache vs load tension**: If all requests with a popular system prompt route to the same replica, that replica becomes a hotspot while others sit idle. This is the fundamental tension in prefix-aware routing. Production systems address this with hybrid strategies.

**Workload dependency**: Prefix-aware routing helps most when prefix overlap is high. For batch inference over diverse, unrelated queries — each with a unique prefix — traditional routing may be simpler and equally effective.

**Cache eviction pressure**: GPU memory is finite. Under high traffic with many diverse prefixes, the eviction policy may remove cached entries before they can be reused.

**Stale routing decisions**: The router's prefix index may lag behind actual cache state on replicas. A cached prefix may be evicted between the routing decision and request arrival.

> "Prefix-aware routing is an optimization for high-prefix-overlap workloads, not a universal replacement for traditional load balancing. The best production systems treat it as a routing preference that can be overridden when load balance requires it."

---

## Source Notes

- Zheng et al. "SGLang: Efficient Execution of Structured Language Model Programs." NeurIPS 2024. [arxiv.org/abs/2312.07104](https://arxiv.org/abs/2312.07104)
- vLLM Automatic Prefix Caching docs: [docs.vllm.ai/en/stable/design/prefix_caching/](https://docs.vllm.ai/en/stable/design/prefix_caching/)
- Ray Serve PrefixCacheAffinityRouter docs: [docs.ray.io](https://docs.ray.io/en/releases-2.54.0/serve/llm/user-guides/prefix-aware-routing.html)
- DualMap paper (Feb 2026): [arxiv.org/abs/2602.06502](https://arxiv.org/abs/2602.06502)
- llm-d KV-Cache benchmark: 87% cache hit rate, 40% per-output-token latency reduction on DeepSeek V3.1 with H200 GPUs.
