# Preble: Efficient Distributed Prompt Scheduling for LLM Serving

**Source:** https://arxiv.org/abs/2407.00023
**Authors:** Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang
**Venue:** ICLR 2025
**arXiv:** arXiv:2407.00023v2 [cs.DC] — submitted May 2024, revised October 2024
**Level:** L4 — Production + systems
**Why here:** Theoretical and empirical foundation for `PrefixCacheAwarePolicy`. Preble's E2 (Exploitation + Exploration) algorithm is the rigorous version of Layer 15's cache-threshold/balance-threshold design. 1.5×–14.5× improvement over SOTA systems.

---

## Abstract

Prompts to LLMs have evolved beyond simple user questions. Today's practices include domain-specific instructions, tool usage examples, and long context (textbook chapters) in prompts. Many parts of prompts are repetitive across requests. Recent works propose to cache and reuse KV state of prompts — but they are all confined to a single-GPU optimization, while production LLM serving systems are distributed by nature.

Preble proposes the first distributed LLM serving platform that targets and optimizes for **prompt sharing**. It co-optimizes KV state reuse and computation load-balancing with a new scheduling algorithm and a hierarchical scheduling mechanism.

**Results:** 1.5×–14.5× improvement in average latency, 2×–10× in p99 latency over SOTA serving systems.

---

## Key Insight: Single-GPU Caching Doesn't Scale

Single-GPU prefix caching (SGLang RadixAttention, vLLM APC) only helps when both requests land on the same GPU. In distributed deployments:
- Current systems use round-robin or least-loaded routing.
- These strategies scatter related requests across different GPUs.
- Each GPU recomputes the full prefix from scratch — no reuse across GPUs.
- The opportunity for inter-request KV sharing is entirely lost.

This is exactly the problem that motivates `PrefixCacheAwarePolicy` in `router.py`.

---

## The E2 Algorithm (Exploitation + Exploration)

Preble's central contribution is the **E2 scheduling algorithm** — a principled approach to the exploitation/exploration tradeoff in distributed KV cache scheduling.

### Exploitation
Route requests to the GPU that already holds the longest matching KV prefix. This maximizes cache hits and minimizes prefill computation.

↔ Layer 15: `PrefixCacheAwarePolicy._pick_worker` — find the worker whose `RadixTrie` has the longest match for the current prompt.

### Exploration  
Occasionally route requests to underloaded GPUs even if they don't have the best prefix match. This prevents hotspots and ensures load balance.

↔ Layer 15: `balance_abs_threshold` guard — if load imbalance exceeds the threshold, route to the least-loaded worker instead of the best-match worker.

### The Tension

Pure exploitation causes hotspots: popular prefixes all go to one GPU, which becomes overloaded. Pure exploration (round-robin) destroys cache reuse. E2 combines both:
- **Exploit** (cache-affinity routing) when load is balanced.
- **Explore** (least-loaded routing) when load diverges above a threshold.

This is the exact two-guard pattern in `PrefixCacheAwarePolicy`:
```python
if abs(max_load - min_load) > balance_abs_threshold:
    return least_load_pick(workers)  # Explore: break hotspot
best_worker, best_match = find_best_prefix_match(workers, prompt)
if best_match / len(prompt) >= cache_threshold:
    return best_worker  # Exploit: route to cache-hit worker
return smallest_trie_pick(workers)  # Fallback: route to most free cache
```

---

## Architecture

Preble uses a **two-level scheduling system**:

### Global Scheduler
- Performs **request-level** scheduling decisions across all GPUs.
- Maintains a **global radix tree**: maps prompt prefixes to GPU assignments.
- Uses E2 algorithm for routing decisions.
- Scalable to 70–391 GPUs with a single scheduler node.

↔ Layer 15's `PrefixCacheAwarePolicy` with one `RadixTrie` per worker — the same logical structure, but router-side rather than a separate scheduler process.

### Local Scheduler (per GPU)
- Performs **iteration-level** scheduling within a single GPU.
- Manages memory allocation, request prioritization, and eviction.
- Not visible to the router layer.

↔ The SGLang engine's internal RadixAttention KV cache management.

---

## Global Radix Tree Design

The global radix tree in Preble tracks which GPU holds which prefix:
- **Node**: Represents a prefix sequence (can span multiple tokens per edge).
- **Leaf metadata**: GPU ID + in-flight request count.
- **LRU eviction**: When GPU memory is full, removes least-recently-used prefix nodes.

Layer 15's `RadixTrie` is a simplified router-side approximation:
- Uses **character-level** (text) matching instead of token-level matching.
- No eviction (teaching scope).
- One trie per worker (not one shared global trie).

The text-level approximation is valid because the router doesn't have access to the tokenizer — it operates on raw prompt text.

---

## Evaluation Highlights

**Workloads:** Real-world LLM workloads (chat, code, document QA) with high prefix sharing.
**Models:** Two open-source LLMs on two GPU clusters.
**Baselines:** vLLM (round-robin + APC), SGLang (round-robin + RadixAttention).

| Metric | Preble vs SOTA |
|---|---|
| Average latency | 1.5×–14.5× improvement |
| p99 latency | 2×–10× improvement |
| Effective throughput | Significant improvement at high QPS |

The 14.5× improvement occurs on workloads with high prefix overlap (shared system prompts + multi-turn) — the same class of workload where Layer 15's `PrefixCacheAwarePolicy` provides the most benefit.

---

## Relevance to Layer 15

| Preble concept | Layer 15 `router.py` |
|---|---|
| E2 exploit phase (cache-affinity routing) | `PrefixCacheAwarePolicy` + `RadixTrie.match_len` |
| E2 explore phase (load-balance routing) | `balance_abs_threshold` guard → `LeastLoadPolicy` |
| Global radix tree | Per-worker `RadixTrie` in `_tries` dict |
| `cache_threshold` (E2 min-match ratio) | `cache_threshold: 0.5` in `config.yml` |
| `balance_abs_threshold` (load gap trigger) | `balance_abs_threshold: 32` in `config.yml` |
| Global scheduler process | `Router` class (FastAPI process) |
| Local scheduler (per GPU) | SGLang engine's internal RadixAttention |

**Reading note:** Preble is the theoretical paper; SGLang's `sgl-model-gateway` is the Rust implementation that Layer 15 directly ports to Python.
