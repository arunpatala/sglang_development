# LLM Router / Gateway — Combined Reference

**What this file is:** A synthesis of orientation articles on LLM routing, combined into a single progressive narrative. The reading order moves from "what problem does routing solve?" → "what is an LLM gateway?" → "why standard load balancers fail for LLM inference" → "how prefix-aware routing works" → "what the numbers say" → "how production systems implement it."

**Sources synthesized:**
- Portkey Blog: LLM proxy vs. AI gateway; the "control plane" framing
- PkgPulse: LiteLLM vs. Portkey vs. OpenRouter; real gateway products
- Portkey AI Gateway intro: production scale (10B tokens/day); middleware architecture
- OptiVerse: prefix-aware routing explainer; two-guard strategy; worked example
- llm-d Blog: 57× TTFT gap; cost of cache-blind routing at scale
- vLLM Router release: SGLang lineage; policy comparison table
- Red Hat / llm-d: 87.4% cache hit rate; routing strategy hierarchy

---

## 1. The Problem: Routing LLM Traffic at Scale

As more teams build with large language models, getting requests to the right backend efficiently becomes a core infrastructure problem. A single LLM instance reaches its compute or memory limit quickly. The solution is obvious: run multiple instances and distribute requests across them.

But **how** you distribute requests turns out to matter enormously. The wrong policy wastes GPU compute, increases latency, and raises costs — all on the same hardware. The right policy can reduce time-to-first-token by **57× on identical hardware** without adding a single GPU.

This document explains why, and how.

---

## 2. Two Patterns: LLM Proxy vs. AI Gateway

Before understanding routers, it helps to understand the spectrum of infrastructure they live in.

### The LLM Proxy

An LLM proxy is a lightweight middleware layer that sits between an application and one or more LLM backends. It does one job: forward and shape requests. Common extras — caching, logging, token counting — are useful but secondary.

**Use when:** prototyping, single-provider setups, or when governance is not a concern.

### The AI Gateway

An AI gateway is a production-grade infrastructure layer designed to manage, govern, and optimise all LLM traffic across an organisation. It includes everything a proxy does, plus:

| Feature | LLM Proxy | AI Gateway |
|---------|-----------|------------|
| Request routing | ✔ | ✔ |
| Basic caching | ✔ | ✔ (advanced, configurable) |
| Logging | ✔ (limited) | ✔ (structured, queryable) |
| Access control | ✘ | ✔ (role-based, multi-tenant) |
| Guardrails & moderation | ✘ | ✔ |
| Budget controls | ✘ | ✔ (per team/app/model) |
| Rate limiting | ✘ | ✔ |
| Observability | ✘ | ✔ (latency, error, usage analytics) |

> "An AI gateway is a **control plane for LLM usage**. Every request, regardless of model provider, team, or use case, flows through the gateway." — Portkey

The key framing: a gateway is not just a reverse proxy — it is the policy enforcement point for your entire LLM traffic.

### Where a simple instance router sits

A minimal instance router — one that routes requests across replicas of the same model — sits at the **proxy end of this spectrum**: it routes based on a configurable policy, performs health checks, and does basic logging. It has no guardrails, audit logs, or multi-provider abstraction. The SGLang Model Gateway adds circuit breakers, Prometheus metrics, and Kubernetes service discovery on the way toward a full gateway.

---

## 3. Real Gateways: What the Ecosystem Looks Like

Three products dominate open-source LLM gateway deployments:

| | LiteLLM | Portkey | SGLang Router |
|---|---|---|---|
| **Scope** | 100+ LLM providers, OpenAI-compatible API | Enterprise, 25+ providers, semantic caching | Single-model-family, same-model backends |
| **Routing strategies** | round-robin, least-busy, latency-based, cost-based | Weighted, conditional, fallback | Round-robin, least-load, prefix-cache-aware |
| **Proxy latency** | ~10–20ms | ~5ms | sub-ms (Rust) |
| **Open source** | Yes (MIT) | Partial | Yes (Apache) |
| **Scale** | Self-hosted | 10B+ tokens/day | Cluster-scale |

The key distinction in this space is **what is being routed**:
- LiteLLM / Portkey / OpenRouter route across **different model providers** (OpenAI vs Anthropic vs Gemini).
- SGLang Router / vLLM Router route across **different instances of the same model** (engine-0 vs engine-1, both serving Llama-3 70B).

These are orthogonal routing problems. Instance routing and model-provider routing can be composed — a model-provider gateway sits upstream of an instance router.

---

## 4. Why Standard Load Balancers Fail for LLM Inference

Three strategies dominate traditional web service load balancing:

- **Round-robin**: Each request goes to the next replica in a fixed cyclic order.
- **Least-loaded**: Routes to the replica with the shortest queue.
- **Power-of-two-choices**: Picks two replicas at random, routes to the less busy one.

All three share a fundamental assumption: **requests are stateless**. Any replica can handle any request equally well.

LLM inference violates this assumption.

### The KV cache: why LLM instances have state

Every LLM request begins with a **prefill phase**: the model processes the entire input prompt, computing attention Key and Value vectors for every input token. These vectors are stored in the **KV cache** — the model's short-term memory during a conversation.

For a 10,000-token prompt, this prefill computation takes ~4.3 seconds (measured on a single vLLM instance). But if the same prompt arrives again, the KV cache can be reused — TTFT drops to 0.6 seconds. **Same hardware, 7× faster.**

This single-instance optimisation works. The problem starts when you scale out.

### The scale-out problem: cache scatter

When you add more replicas, the KV cache becomes **disaggregated** — each replica has its own isolated cache. A stateless load balancer (round-robin, least-load) scatters related requests across replicas, destroying the cache locality that made the single-instance optimisation work.

**Concrete example — multi-turn conversation with round-robin:**

A user sends five messages in a conversation. Each message includes the full conversation history. With round-robin across five replicas:
- Turn 1 → Replica A (computes 600 tokens from scratch)
- Turn 2 → Replica B (computes 900 tokens from scratch — including 600 already computed by A)
- Turn 5 → Replica E (computes 1,700 tokens from scratch)
- **Total prefill: 5,200 tokens**

If every turn had gone to the same replica:
- Turn 2 → Replica A (only 300 new tokens)
- Turn 5 → Replica A (only 300 new tokens)
- **Total prefill: 1,700 tokens — a 67% reduction**

At a prefill rate of ~10,000 tokens/sec, Turn 5 alone saves ~140ms TTFT.

The cascade of failures from cache-blind routing:

1. **Cache miss**: The warm cache on Replica A is unused when Turn 2 goes to Replica B.
2. **Duplicated work**: The same prefill computation runs twice across two replicas.
3. **Increased TTFT**: Users wait longer for every response.
4. **Wasted GPU resources**: Hardware is tied up re-doing work instead of serving new requests.

> "This isn't a rare event in production — it's the **default behavior** of any distributed deployment with a stateless load balancer." — llm-d team

---

## 5. The Routing Strategy Ladder

There is a natural progression of routing strategies, each building on the previous:

### Rung 1: Round-robin

Simple, predictable, stateless. Requests cycle through replicas in order. Works well for symmetric workloads where prefix overlap is negligible (batch inference over diverse, unrelated queries).

**Failure mode:** Destroys KV cache locality for any workload with shared prefixes. All cache investments are wasted across replicas.

**Production name:** `RoundRobin` / `random` — the baseline, present in every router as a comparison point.

### Rung 2: Least-load (power-of-two choices)

Routes to the replica with the smallest queue (or picks two at random and routes to the less busy). Adapts to unequal request processing times.

**Improvement over round-robin:** Prevents hotspots when some requests are much longer than others. Reduces queue buildup.

**Failure mode:** Still stateless with respect to the KV cache. Does not improve cache hit rates.

**Production name:** `PowerOfTwo` / `least-busy` — based on Mitzenmacher's proof that d=2 random choices gives exponentially better load balance than d=1.

### Rung 3: Session affinity (sticky sessions)

Routes all requests from the same user/session to the same replica. Ensures multi-turn conversations reuse the KV cache from previous turns.

**Improvement:** Captures multi-turn cache reuse. Works for single-user workloads.

**Failure mode:** Misses cross-user cache reuse (shared system prompts). Creates hotspots when some users are much heavier than others. No load balancing.

### Rung 4: Prefix-cache-aware routing

The router maintains an index of which prefixes are cached on which replicas. When a new request arrives:
1. Extract the request's token prefix.
2. Query the index for the replica with the longest matching cached prefix.
3. Route to that replica (maximise cache hits).
4. Override with load-balance if a replica is significantly more loaded than others.

**Improvement:** Captures both multi-turn reuse and cross-user shared-prefix reuse. Balances cache affinity against load balance.

**Production name:** `CacheAware` (SGLang), `PrefixCacheAffinityRouter` (Ray Serve), `precise-scheduling` (llm-d).

---

## 6. Prefix-Aware Routing: How It Works

### The prefix index

The router maintains a **per-replica prefix index** — a data structure that tracks which token sequences each replica has cached. When a request arrives, the router computes a prefix match score for each replica and routes to the best match.

**SGLang uses a radix tree:** a compact prefix tree where each edge represents a multi-token sequence. The router walks the tree from the root, matching the request's token prefix. If the first 1,400 tokens match an existing entry, the tree walk identifies those tokens and stops at the divergence point. Only the unmatched suffix needs prefill.

**vLLM uses hash-based block matching:** each KV cache block (16 tokens) is associated with a hash of its token sequence. New requests trigger a hash lookup to identify reusable blocks.

Both approaches serve the same function: identify which replica has the most relevant cached context.

> "Think of it as sorting mail by ZIP code. Traditional routing hands each letter to the next available mail carrier, regardless of destination. Prefix-aware routing gives the Kyoto letters to the carrier who already knows the Kyoto neighborhoods. The map is the KV cache. The ZIP code is the prefix." — OptiVerse

### Two patterns that dominate production workloads

1. **Multi-turn conversations**: Each turn includes the full conversation history. The prefix grows with every turn. Chatbots, customer support copilots, coding assistants.

2. **Shared system prompts**: Many users of the same application share an identical system prompt. In production workloads — chatbots, coding assistants, enterprise copilots — prefix overlap routinely reaches **40–80% of input tokens**.

Both patterns are captured by routing requests with the same prefix to the same replica.

### The two-guard strategy

Pure prefix-cache-aware routing has a failure mode: if all requests with a popular system prompt route to the same replica, that replica becomes a **hotspot** while others sit idle.

Production systems address this with a two-guard strategy, used by both SGLang's `CacheAware` policy and Ray Serve's `PrefixCacheAffinityRouter`:

**Guard 1 — load balance check:**
Compare queue depths across replicas. If the most-loaded replica's queue exceeds the least-loaded replica's queue by more than a configured threshold (`balance_abs_threshold`), override prefix affinity and route to the least-loaded replica. Prevents hotspots.

**Guard 2 — prefix match check:**
Among load-balanced replicas, find the one with the highest prefix match ratio. If the match ratio meets or exceeds a minimum threshold (`cache_threshold`, default: 0.5), route to that replica. Otherwise, route to the replica with the smallest prefix tree (most available cache space for new entries).

```
for each incoming request:
    if max_queue - min_queue > balance_abs_threshold:
        → route to least-loaded replica  (load guard wins)
    else:
        if best_match_ratio >= cache_threshold:
            → route to best-prefix replica  (cache affinity wins)
        else:
            → route to replica with smallest trie  (cache capacity wins)
```

This design is validated empirically in the Preble paper (ICLR 2025): the **E2 (Exploitation + Exploration)** algorithm — exploit cache hits when load is balanced, explore underloaded replicas when load diverges.

| Parameter | SGLang name | Ray Serve equivalent | Preble equivalent |
|---|---|---|---|
| Load balance threshold | `balance_abs_threshold` | `imbalanced_threshold` | load divergence threshold |
| Min match ratio | `cache_threshold` | `match_rate_threshold` | minimum match ratio |

---

## 7. Quantifying the Cost of Cache-Blind Routing

The transition from cache-blind to prefix-aware routing is not a marginal improvement. The numbers are order-of-magnitude.

### The KV cache economics

Cached token processing costs roughly **10× less** than uncached:
- Uncached (full prefill): ~$3.00 per million tokens
- Cached (KV cache hit): ~$0.30 per million tokens

An 87.4% cache hit rate (measured by llm-d on a SaaS workload with 150 enterprise customers, 6,000-token system prompts) translates directly into a ~8× reduction in compute cost for the prefill phase.

### The 57× TTFT gap

The most striking benchmark comes from llm-d's production measurement:

**Setup:** 8 vLLM pods (16 H100 GPUs total), B2B SaaS workload — 150 enterprise customers, 6,000-token system prompts, 5 concurrent users per customer, 1,200-token queries, load ramping 3–60 QPS.

| Strategy | Output toks/s | TTFT p90 | TTFT mean | Wait queue |
|---|---|---|---|---|
| Precise prefix-cache aware | **8,730** | **0.54s** | **0.30s** | **0.1** |
| Approximate prefix-cache | 6,944 | 31.1s | 13.3s | 8.1 |
| Load-only (no cache) | 4,429 | 94.9s | 47.0s | 28.9 |

**Precise scheduling is 57× faster TTFT than approximate scheduling** — on identical hardware, serving identical traffic.

The reason for the gap: cache-blind schedulers **cache-thrash** — they constantly duplicate and evict the same prefixes across different pods. Precise scheduling avoids this by routing cache-hit requests consistently, resulting in virtually no queues and stable throughput.

### The vLLM Router benchmark

In a head-to-head comparison with 8 prefill + 8 decode pods:
- vLLM Router vs. Kubernetes-native load balancer: **100% higher throughput**, similar TTFT
- vLLM Router vs. llm-d: **25% higher throughput**, **1,200ms faster TTFT**

For DeepSeek V3 (1 prefill pod TP8 + 1 decode pod TP8):
- vLLM Router vs. K8s-native: **100% higher throughput**, **2,000ms faster TTFT**

---

## 8. Production Implementations

Three open-source systems implement prefix-aware instance routing at production scale.

### SGLang Model Gateway

Written in Rust for minimal overhead. Ships three policies: `RoundRobin`, `PowerOfTwo` (least-load with random sampling), `CacheAware` (prefix-cache-aware with a per-worker radix tree). The `CacheAware` policy implements the two-guard strategy described in §6.

The SGLang CLI exposes all three policies directly:
```bash
python -m sglang.launch_server \
  --router-policy cache_aware \
  --router-cache-threshold 0.5 \
  --router-balance-abs-threshold 32
```

### vLLM Router

Forked from the SGLang model gateway, modified for vLLM backends. Adds:
- **Consistent hashing**: requests with the same routing key are "sticky" — always routed to the same worker, maximising KV cache reuse without explicit prefix tracking.
- **Prefill/Decode disaggregation**: routes prefill-phase requests to dedicated prefill workers and decode-phase to decode workers — a different architectural mode that separates the two phases entirely.
- **Kubernetes service discovery**: automatic discovery of vLLM pods via label selectors.

| vLLM Router policy | SGLang equivalent | Concept |
|---|---|---|
| `PowerOfTwoPolicy` | `policies/power_of_two.rs` | Least-load with d=2 sampling |
| Round-robin | `policies/round_robin.rs` | Cyclic distribution |
| `ConsistentHashingPolicy` | — | Hash-ring sticky routing |

> "The vLLM Router is derived from a fork of the SGLang model gateway, modified and simplified to work with vLLM." — vLLM team

### llm-d (Kubernetes-native)

llm-d maintains a **global view of the cluster's KV cache** via `KVEvents` — a live feed of physical cache changes emitted by each vLLM pod. A precise prefix-cache scorer computes a "cache affinity score" for each pod based on the longest matching token sequence in the global index. Unlike SGLang's router-side radix tree (updated from request history), llm-d's index is updated by the engine in real time via live events.

---

## 9. The Production Maturity Ladder

A minimal instance router implements the core ideas. Production systems add layers of hardening on top:

| Feature | Minimal router | SGLang Gateway | vLLM Router | llm-d |
|---------|---------------|----------------|-------------|-------|
| Round-robin | ✔ | ✔ | ✔ | ✔ |
| Least-load | ✔ | ✔ | ✔ | ✔ |
| Prefix-cache-aware | ✔ (router-side trie) | ✔ (Rust, radix tree) | ✔ (consistent hash) | ✔ (live KV events) |
| Health checks | ✔ (basic) | ✔ (circuit breaker) | ✔ (circuit breaker) | ✔ |
| Implementation language | Python | Rust | Rust | Go/Rust |
| Service discovery | Static config | — | Kubernetes labels | Kubernetes GAIE |
| Metrics | Logging | Prometheus `/metrics` | Prometheus `/metrics` | Prometheus |
| gRPC | ✘ | ✔ | ✔ | ✔ |
| PD disaggregation | ✘ | ✘ | ✔ | ✔ |
| Live KV cache index | ✘ | ✘ | ✘ | ✔ (KVEvents) |

**The progression:**

```
Minimal Python router (learning / prototyping)
  → SGLang sgl-model-gateway (Rust, circuit breakers, Prometheus)
    → vLLM Router (+ consistent hashing, PD disaggregation)
      → llm-d (+ global KV cache index, Kubernetes-native)
        → Portkey / LiteLLM (+ multi-provider, guardrails, budgets)
```

Each step adds production hardening. The core ideas — policy abstraction, health checking, prefix indexing, the two-guard strategy — are present at every level. Everything above the minimal router is optimisation and operational complexity.

---

## 10. Key Quotes

> "An AI gateway is a **control plane for LLM usage**. Every request, regardless of model provider, team, or use case, flows through the gateway." — Portkey

> "This isn't a rare event in production — it's the **default behavior** of any distributed deployment with a stateless load balancer." — llm-d team

> "The KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost." — Manus (Context Engineering for AI Agents), quoted in llm-d blog

> "Prefix-aware routing is an optimization for high-prefix-overlap workloads, not a universal replacement for traditional load balancing. The best production systems treat it as a routing preference that can be overridden when load balance requires it." — OptiVerse

> "The vLLM Router is derived from a fork of the SGLang model gateway, modified and simplified to work with vLLM." — vLLM team

---

## Appendix: What Is Left Out and Why

### Left out: Multi-provider routing details (LiteLLM / Portkey / OpenRouter)

The source articles cover Portkey's semantic caching, guardrails, virtual keys, and enterprise features in depth. These are omitted because they address **model routing** (routing across different LLM providers), not **instance routing** (distributing load across replicas of the same model). The feature comparison table in §3 captures the relevant positioning; the rest is a different problem domain.

### Left out: llm-d KVEvents engine internals

The llm-d articles describe the `KVEvents` stream, `Pool`, `kvcache.Index`, and `PrefixStore` components. These describe the **engine side** — how vLLM pods emit cache state to the router. This document focuses on the **router side**: how routing decisions are made given prefix state. The engine-side implementation is relevant when building the feedback loop between inference engine and router.

### Left out: Prefill/Decode disaggregation

The vLLM Router introduces PD disaggregation — routing prefill-phase requests to dedicated prefill workers and decode-phase requests to decode workers. This is a distinct architectural mode that requires separate worker pools and state transfer between phases. It is a separate topic from the three-policy routing strategies covered here.

### Left out: Consistent hashing details

The vLLM Router's `ConsistentHashingPolicy` maps routing keys (session ID, user ID) to workers via a hash ring, providing sticky routing without explicit prefix tracking. It achieves similar per-session cache locality to prefix-aware routing but misses cross-session reuse (e.g., different users with the same system prompt). Relevant as an alternative to explicit prefix indexing for session-oriented workloads.

### Left out: Kubernetes configuration and GAIE details

The Red Hat/llm-d article includes YAML configuration for the `cache-aware-router` plugin and vLLM pod flags (`--enable-prefix-caching`, `--block-size`, etc.). These are deployment specifics for Kubernetes-native environments and are separate from the routing algorithm itself.
