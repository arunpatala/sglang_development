# SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference

**Source:** https://arxiv.org/abs/2505.24095
**Authors:** Heming Xia et al. (LMSYS / UC Berkeley)
**Venue:** EUROSYS 2026, April 27–30, Edinburgh, Scotland
**arXiv:** arXiv:2505.24095v2 [cs.DC] — submitted May 2025, revised November 2025
**Level:** L4 — Production + systems
**Why here:** The benchmark baseline set matches Layer 15's three policies exactly. Shows that SGLang Router (the production version of Layer 15's `router.py`) is the recognized baseline for prefix-aware routing at the single-region level. Extends the design to cross-region traffic aggregation.

---

## Abstract

Serving LLMs efficiently in multi-region setups remains a challenge. SkyWalker introduces a multi-region load balancer for LLM inference that aggregates regional diurnal traffic patterns through cross-region traffic handling. It preserves KV-cache locality and load balancing, ensuring cost efficiency without sacrificing performance.

**Results:** 1.12–2.06× higher throughput, 1.74–6.30× lower latency vs. existing load balancers, while reducing total serving cost by 25%.

---

## Why Round-Robin Fails for LLM Inference

> "We find two replicas under round-robin can have memory usage difference up to **2.64×**." — SkyWalker §2.3

LLM inference load is **unpredictable**:
- Output length is non-deterministic (auto-regressive generation).
- Traditional policies (round-robin, least-load-first) blindly push requests to replicas without accounting for resource consumption.
- Long-running requests block all subsequent ones in the queue — **head-of-line blocking**.
- Load imbalance between replicas reaches 2.64× under round-robin.

This directly motivates Layer 15's `LeastLoadPolicy` and `PrefixCacheAwarePolicy` over `RoundRobinPolicy`.

---

## Benchmark Baseline Set

SkyWalker's evaluation compares the exact same policies as Layer 15:

| System | Policy | Layer 15 equivalent |
|---|---|---|
| GKE Gateway | Network load balancer (round-robin per connection) | `RoundRobinPolicy` |
| Round Robin (RR) | Stateless, distributes requests cyclically | `RoundRobinPolicy` |
| Least Load (LL) | Tracks outstanding requests per replica | `LeastLoadPolicy` |
| Consistent Hashing (CH) | Ring hash based on session ID | (not in Layer 15) |
| SGLang Router (SGL) | **Prefix-aware load balancer** | `PrefixCacheAwarePolicy` |

**SGLang Router is the recognized production baseline for prefix-aware routing.** SkyWalker extends it to cross-region traffic.

---

## SkyWalker Design

### Cache-Aware Cross-Region Traffic Handler

Routes requests to regions with matching prefix caches, similar to `PrefixCacheAwarePolicy` but across geographic regions rather than replicas.

### Selective Pushing-Based Load Balancing

When load is imbalanced, pushes excess requests from overloaded replicas to underloaded ones, rather than just routing away from hotspots. This is a more aggressive version of Layer 15's `balance_abs_threshold` guard.

### KV Cache Transfer (Beyond Layer 15)

When routing to a non-cache-holding replica, SkyWalker can transfer the KV cache blocks from the cache-holding replica to the target replica. This eliminates the need for recomputation even when cache affinity is overridden by load balance.

This is the "what comes next" beyond Layer 15's stateless prefix routing: **active KV cache migration** between replicas.

---

## Evaluation Results

**Hardware:** Real-world workloads, multiple regions.
**Key metric:** Throughput improvement vs. round-robin baseline.

| Baseline | Throughput vs SkyWalker | Latency vs SkyWalker |
|---|---|---|
| Round Robin | 1.12–2.06× worse | 1.74–6.30× higher |
| Least Load | Still worse | Still higher |
| SGLang Router | Worse in cross-region | Higher in cross-region |

SkyWalker outperforms SGLang Router by routing cross-region requests to the region with the most matching prefix cache, even when that region is not the geographically nearest.

---

## Practical Implications for Layer 15

1. **Round-robin is the weakest baseline**: 2.64× memory imbalance confirms `RoundRobinPolicy` is a teaching baseline, not a production choice.

2. **SGLang Router (= Layer 15's `PrefixCacheAwarePolicy`) is the production baseline**: SkyWalker uses SGLang Router as its single-region baseline — confirming Layer 15's policy design is production-representative.

3. **Least Load is not enough**: Even with perfect load balance, cache-blind routing wastes significant compute. `LeastLoadPolicy` is better than round-robin but worse than `PrefixCacheAwarePolicy` on prefix-heavy workloads.

4. **KV transfer is the next frontier**: Layer 15's `RadixTrie` only routes to the worker that has the prefix. SkyWalker adds active KV cache migration — the next step in prefix-aware design.

---

## Connection to Layer 15 lesson/09_whats_next.md

SkyWalker demonstrates two "what comes next" directions beyond Layer 15:
- **Cross-region routing**: Extending the `RadixTrie` to a global index across regions.
- **KV cache transfer**: Instead of routing requests to where the cache is, move the cache to where the request needs to go (P/D disaggregation at the KV block level).
