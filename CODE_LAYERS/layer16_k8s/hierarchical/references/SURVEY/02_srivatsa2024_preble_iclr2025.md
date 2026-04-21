# Preble: Efficient Distributed Prompt Scheduling for LLM Serving (ICLR 2025)

**Source:** https://openreview.net/forum?id=meKEKDhdnx
**PDF:** https://openreview.net/pdf?id=meKEKDhdnx
**Authors:** Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang
**Venue:** ICLR 2025 (Poster)
**Keywords:** LLM prefix caching, LLM serving, Distributed systems for ML
**TL;DR:** First distributed LLM serving platform targeting prompt sharing; improves SOTA by up to 14.5× on average latency.
**Level:** L4 — Survey / academic paper
**Why here:** The conceptual and experimental foundation for all distributed prefix-cache-aware routing. Layer 15's `PrefixCacheAwarePolicy` is the teaching-scale Python port of Preble's E2 algorithm. The `cache_threshold` and `balance_abs_threshold` config values are calibrated against Preble's results.

---

## Problem Statement

Prompts have evolved beyond simple user questions. Today's LLM prompts include:
- Domain-specific instructions (hundreds of tokens)
- Tool usage examples and schemas
- Long context (textbook chapters, conversation history)

As a result, **many parts of prompts are repetitive across requests**. Recent works (SGLang RadixAttention, vLLM APC) cache KV state of repeated prefixes — but only within a single GPU. Production LLM serving is distributed by nature. The opportunity for cross-request KV sharing is entirely lost when related requests land on different GPUs.

---

## Key Contributions

1. **First study of LLM workloads with long and shared prompts**: Documents the wide existence of prefix sharing in production LLM workloads (RAG, chat, tool-augmented generation).

2. **E2 (Exploitation + Exploration) scheduling algorithm**: Co-designs model computation load-balancing and prefix-cache sharing. The core algorithm that Layer 15 implements in simplified form.

3. **Preble system**: Distributed LLM serving system built on vLLM and SGLang. Implements E2 at cluster scale. Two-level scheduling: global request-level + per-GPU iteration-level.

---

## E2 Algorithm — Technical Details

The E2 algorithm makes routing decisions based on two factors for each candidate GPU:
1. **Cache hit ratio**: What fraction of the request's prompt prefix is already cached on this GPU?
2. **Worker load**: How many in-flight requests does this GPU have?

### Phase 1: Exploitation (cache-affinity routing)

If a GPU has a long matching prefix in its radix tree, route the request there. The benefit: the GPU reuses cached KV state, skipping expensive prefill computation for the shared portion.

**Threshold**: Only exploit if the cache hit ratio exceeds a minimum threshold (analogous to `cache_threshold`). Below the threshold, the benefit of cache affinity doesn't justify the routing overhead.

### Phase 2: Exploration (load-balancing routing)

If all GPUs have comparable load, prefer the one with the best prefix match (exploitation). If load is highly imbalanced, route to the least-loaded GPU regardless of prefix match (exploration).

**Threshold**: Only switch to load-balance routing if the load gap exceeds a threshold (analogous to `balance_abs_threshold`). Small load differences are acceptable to preserve cache affinity.

---

## The Two-Level Scheduling System

```
Incoming requests
       ↓
Global Scheduler (Preble E2)
  ├── Maintains global radix tree
  ├── Tracks per-GPU load
  └── Routes requests: exploit or explore
       ↓              ↓
   GPU-1 Local    GPU-2 Local    ... GPU-N Local
   Scheduler      Scheduler          Scheduler
   (iteration-    (iteration-       (iteration-
    level)         level)            level)
```

**Global scheduler** (request-level, Layer 15 analog):
- One process per cluster.
- Maintains the global radix tree.
- Routes each request to a GPU using E2.

**Local scheduler** (iteration-level, inside the engine):
- One per GPU.
- Manages token generation, batching, memory allocation.
- Not part of the router layer.

This maps to Layer 15's separation of concerns:
- `Router.route()` ↔ global scheduler
- SGLang engine internals ↔ local scheduler
- `RadixTrie` per worker ↔ global radix tree (simplified)

---

## Evaluation

**Workloads:**
- Multi-turn chat (Anthropic Claude production distribution)
- RAG (retrieval-augmented generation with shared document prefixes)
- Tool-augmented generation (shared tool schemas)
- Code generation (shared system prompts and examples)

**Models:** Two open-source LLMs (Llama-class)

**Clusters:** Two GPU clusters (single-server multi-GPU + multi-server)

**Results:**
- Average latency: **1.5×–14.5× improvement** over vLLM and SGLang with round-robin routing.
- p99 latency: **2×–10× improvement**.
- Highest gains on workloads with high prefix overlap (multi-turn chat with long history).

---

## Why Prefix Overlap Is High in Production

From the paper's workload analysis:
- Chat applications: 40–60% token overlap across consecutive turns of the same session.
- Tool-augmented: 70–80% token overlap (shared tool schemas + system prompts).
- RAG: 30–50% overlap (shared retrieved documents).

These overlap rates explain why `PrefixCacheAwarePolicy` can meaningfully outperform `RoundRobinPolicy` and `LeastLoadPolicy` in practice.

---

## Hot Entry Replication (Beyond Layer 15)

When a prefix becomes "hot" (many requests map to the same GPU), Preble **replicates** the cached KV entries onto other GPUs. This allows multiple GPUs to serve requests for the same popular prefix, preventing hotspots.

Layer 15 does not implement replication — it relies on `balance_abs_threshold` to break away from hotspots by routing to the least-loaded worker instead. Replication would require the router to coordinate KV cache state across engines, which is out of scope for the teaching layer.

---

## BibTeX

```bibtex
@inproceedings{srivatsa2024preble,
  title={Preble: Efficient Distributed Prompt Scheduling for LLM Serving},
  author={Srivatsa, Vikranth and He, Zijian and Abhyankar, Reyna and Li, Dongming and Zhang, Yiying},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025},
  url={https://arxiv.org/abs/2407.00023}
}
```
