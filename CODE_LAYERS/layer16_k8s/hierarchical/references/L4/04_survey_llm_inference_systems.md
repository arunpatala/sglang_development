# A Survey of LLM Inference Systems

**Source:** https://arxiv.org/html/2506.21901v1
**Authors:** James Pan, Guoliang Li (Tsinghua University)
**arXiv:** arXiv:2506.21901 [cs.DC]
**Level:** L4 — Production + systems
**Why here:** The multi-replica section (§5.2.2) positions Round Robin, Power-of-Two, Preble, and PD disaggregation in the inference stack. Provides the academic framing for why Layer 15 needs all three policies. Connects the router layer to the rest of the LLM inference stack.

---

## Abstract

This survey reviews techniques for LLM inference systems, starting from operators and algorithms for request processing, then moving to model optimization and execution (kernel design, batching, scheduling), and ending with memory management (paged memory, eviction, quantization, cache persistence).

Covered systems: vLLM, SGLang, Mooncake, DeepFlow, Preble, DistServe, SplitWise, TetriInfer.

---

## The Multi-Replica Load Balancing Landscape (§5.2.2)

Multi-replica runtimes use load balancers to evenly distribute workload across multiple workers. For multi-replica runtimes that support cache persistence, persisted entries are distributed across different workers — leading to techniques for **distributed cache management**.

### Monolithic Runtimes (Layer 15 scope)

**Preble** (Srivatsa et al., 2024) demonstrates how cache persistence can be extended to a multi-replica scenario:
- A central scheduler maintains a **global radix tree**.
- Load balancing is performed based on **cache hit ratio in addition to worker load**.
- Since the persisted cache is distributed across workers, some workers have higher hit rates.
- To avoid overloading these hot workers, hot entries are replicated onto other workers.

This is the research implementation that Layer 15's `PrefixCacheAwarePolicy` approximates.

### Disaggregated Runtimes (Beyond Layer 15)

**DistServe** (Zhong et al., 2024): Reduces KV cache transfer cost by assigning inference devices to either prefill tasks or decode tasks based on physical location. GPUs on the same machine use NVLink for low-latency KV transfer.

**TetriInfer** (Hu et al., 2024): Addresses decode-side load balancer overload by adopting **power-of-two load balancing** (Mitzenmacher 2001).

↔ TetriInfer's use of power-of-two for the decode worker pool is the production application of Layer 15's `LeastLoadPolicy`.

**SplitWise** and **Mooncake**: Select the decode worker at the same time as the prefill worker, allowing the KV cache to stream asynchronously onto the decode worker as it is being generated.

---

## The Three Load Balancing Policies (from the survey's framework)

The survey identifies three tiers of load balancing sophistication:

### 1. Stateless (Round-Robin)
- No worker state tracked.
- Each request goes to the next worker in rotation.
- **Problem**: Ignores KV cache state, output length variance, and GPU load.
- ↔ Layer 15's `RoundRobinPolicy`.

### 2. Load-Aware (Power of Two Choices)
- Samples two workers at random, routes to the less loaded one.
- **Why two and not all**: Mitzenmacher (2001) proved d=2 gives exponential improvement over d=1, while d=3 gives only constant improvement over d=2. Two choices is the sweet spot.
- Used in TetriInfer for decode-side balancing.
- ↔ Layer 15's `LeastLoadPolicy`.

### 3. Cache-Aware (Preble E2 / SGLang Router)
- Combines cache hit ratio with load balance.
- Maintains a per-worker prefix index (global radix tree or per-worker tries).
- Routes to the worker with the best prefix match, subject to load balance guard.
- ↔ Layer 15's `PrefixCacheAwarePolicy`.

---

## The Inference Stack (Figure 1 mapping)

The survey's Figure 1 shows the LLM inference stack. Layer 15 implements the routing layer (between client and engine):

```
Client
  ↓
[Layer 15: Router / Gateway] ← routing policies
  ↓         ↓
Engine A  Engine B
  ↓         ↓
[Batching & Scheduling] ← vLLM/SGLang internal
  ↓
[KV Cache Management] ← RadixAttention, PagedAttention
  ↓
[GPU Kernel Execution]
```

The survey covers all layers. Layer 15 focuses exclusively on the routing layer at the top.

---

## Mitzenmacher Reference (Key Citation)

The survey explicitly cites: *Mitzenmacher (2001) power-of-two load balancing* when describing TetriInfer's decode-side load balancer. This confirms that Mitzenmacher 2001 is the canonical theoretical reference for `LeastLoadPolicy`.

> "TetriInfer addresses this issue by adopting power-of-two load balancing [Mitzenmacher 2001]."

↔ The same paper is listed in Layer 15's SURVEY references (`01_mitzenmacher2001_power_of_two_choices.md`).

---

## Remaining Challenges (§6)

The survey identifies open problems relevant to Layer 15:

- **Load prediction**: More accurate models for predicting request duration at routing time.
- **Adaptive mechanisms**: Dynamic adjustment of batch size, number of workers, and routing thresholds based on observed workload.
- **Elastic resource provisioning**: Handling workload shifts without fixed worker pools.

These point to the natural extensions of Layer 15: dynamic `balance_abs_threshold`, output-length-aware routing, and autoscaling worker pools.
