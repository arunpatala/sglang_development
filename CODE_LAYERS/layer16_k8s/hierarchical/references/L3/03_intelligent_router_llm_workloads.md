# Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing

**Source:** https://arxiv.org/abs/2408.13510
**Authors:** IBM Research
**arXiv:** arXiv:2408.13510v2 [cs.DC] — submitted August 2024, revised January 2025
**Level:** L3 — Mechanism level
**Why here:** Empirical study comparing Round Robin, Decode Balancer, Join-Shortest-Queue, and RL-based routing on real LLM workloads. Establishes that round-robin is the baseline to beat and quantifies the latency improvements from smarter routing — directly supporting Layer 15's motivation.

---

## Abstract

LLM workloads have distinct prefill and decode phases with different compute and memory requirements. Existing scheduling algorithms treat LLM workloads as monolithic jobs without considering the distinct characteristics of the two phases. This leads to sub-optimal scheduling and increased response latency.

Key findings:
- Better load balancing across LLM instances can improve end-to-end latency **more than** optimizing the instance-level scheduler.
- An RL-based intelligent router achieves **11% lower end-to-end latency** than existing approaches on public datasets and **7.8% lower** on real workload data from Cloud Provider X.

---

## Why Mixing Request Types Matters

LLM requests have distinct phases:
- **Prefill**: Processes the full input prompt. Compute-bound. Cost scales with input length.
- **Decode**: Auto-regressively generates output tokens. Memory-bound. Duration scales with output length.

Routing a long-decode request to an instance already serving many long-decode requests creates a bottleneck where short-prefill requests are blocked behind long-running decode phases. The "mixing effects" of different workload types on the same instance are a primary source of latency variance.

---

## Routing Algorithms Compared

### Baseline: Round Robin (A.1.5)

Each of two model instances is used alternately by the router. Simple, predictable, ignores workload characteristics.

↔ Layer 15's `RoundRobinPolicy`.

### Join Shortest Queue (A.2.1)

Each arriving request is routed to the model with the least number of prompt and decode tokens yet to be processed. Equivalent to join-min-load.

↔ Layer 15's `LeastLoadPolicy` (which uses in-flight request count as the load signal rather than token count).

### Decode Balancer

Assumes the total number of output tokens is known in advance (oracle), routes to minimize decode-phase queue depth. Upper bound on token-count-aware routing.

### Least Work Left

Routes to the instance with the least total remaining work (prompt + decode tokens remaining). Requires output length prediction.

### RL-Based Workload-Aware Router

Uses a trainable response-length predictor and a novel formulation for estimating the impact of mixing workloads. Three variants:
- **Baseline RL**: 3.15% improvement over round-robin.
- **Workload Aware RL**: Aware of workload types.
- **Workload Guided RL**: Best configuration — uses predicted output lengths and mixing cost estimates.

---

## Key Results

| Routing Algorithm | Avg E2E Latency (s) | Improvement over RR |
|---|---|---|
| Round Robin | 248.41 | — |
| Baseline RL | 240.58 | 3.15% |
| Workload Aware RL | ~232 | ~7% |
| Workload Guided RL | ~220 | ~11% |

**Key insight**: Even Join-Shortest-Queue (the direct precursor to `LeastLoadPolicy`) outperforms round-robin for decode-heavy workloads. The improvement is workload-dependent: for balanced prefill/decode, round-robin and join-shortest-queue perform similarly. For decode-heavy workloads, the gap widens.

---

## Dedicated Small-Large Split (Alternative to Cache Routing)

One alternative to prefix-cache-aware routing: dedicate one instance to "light-decode" requests and another to "heavy-decode" requests. This avoids mixing effects without needing prefix state, at the cost of requiring output length prediction.

Layer 15's `PrefixCacheAwarePolicy` is orthogonal: it routes by prefix match, not by predicted output length.

---

## Generalizability

Experiments on A100s with Llama 3.1 8B show the same relative ordering of algorithms holds across hardware:
- Round Robin → weakest baseline
- RL-based workload routing → strongest (requires training)
- Join-Shortest-Queue → practical middle ground (no training, reasonable improvement)

---

## Relevance to Layer 15

This paper provides the empirical foundation for why Layer 15 needs more than round-robin:

1. **`RoundRobinPolicy`**: Validated as the correct baseline. Known weakness: ignores decode-phase queue depth.
2. **`LeastLoadPolicy`**: The "Join Shortest Queue" baseline. Proven improvement over round-robin for heterogeneous workloads.
3. **`PrefixCacheAwarePolicy`**: Goes further by incorporating KV cache state as a routing signal — not studied in this paper but the natural next step.

The paper also validates the Layer 15 design choice to track in-flight request count per worker (in `Worker.in_flight`) rather than total token count — a simpler heuristic that still improves over round-robin for most workloads.
