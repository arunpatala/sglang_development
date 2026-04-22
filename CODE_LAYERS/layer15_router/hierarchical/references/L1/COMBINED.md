# LLM Routing — From Proxy to Prefix-Aware Scheduling

**Level:** L1 + L2 — Concept, mechanism, and production evidence. No source code; no paper math.
**What this file is:** A single coherent blog synthesising all L1 and L2 source material into a progressive narrative. Sections build from "why do we need a router at all?" to "what does prefix-aware routing actually look like in production?" Sections above L2 (implementation internals, research papers, Kubernetes YAML, PD disaggregation) are deliberately left out.

**Sources synthesised:**
- L1/01 — Portkey: LLM Proxy vs. AI Gateway
- L1/02 — PkgPulse: Portkey vs LiteLLM vs OpenRouter
- L1/03 — Portkey AI Gateway Introduction (10B tokens/day)
- L2/01 — OptiVerse: Prefix-Aware Routing
- L2/02 — llm-d: KV Cache Wins You Can See (57× TTFT gap)
- L2/03 — vLLM Router Release
- L2/04 — Red Hat: Master KV Cache Aware Routing with llm-d

**Omitted (above L2):** PD disaggregation, KVEvents engine internals, consistent hashing hash-ring details, GAIE/EPP Kubernetes YAML, vLLM pod flags, Preble paper formal analysis, RadixTrie source code.

---

## Section Plan

| § | Title | Sources | Reading time |
|---|-------|---------|------|
| 1 | [The Scale Problem: Why One Instance Isn't Enough](#1-the-scale-problem-why-one-instance-isnt-enough) | L2/02, L1/03 | 2 min |
| 2 | [Two Patterns: LLM Proxy vs. AI Gateway](#2-two-patterns-llm-proxy-vs-ai-gateway) | L1/01, L1/03 | 3 min |
| 3 | [The Gateway Ecosystem: Real Products](#3-the-gateway-ecosystem-real-products) | L1/02, L1/03 | 3 min |
| 4 | [Why Standard Load Balancers Fail for LLM Inference](#4-why-standard-load-balancers-fail-for-llm-inference) | L2/01, L2/02, L2/04 | 4 min |
| 5 | [The KV Cache: LLM's Short-Term Memory](#5-the-kv-cache-llms-short-term-memory) | L2/02, L2/01 | 3 min |
| 6 | [The Routing Strategy Ladder](#6-the-routing-strategy-ladder) | L2/01, L2/02, L2/04 | 4 min |
| 7 | [Prefix-Aware Routing: How It Works](#7-prefix-aware-routing-how-it-works) | L2/01, L2/04 | 4 min |
| 8 | [The Two-Guard Strategy: Cache vs. Load](#8-the-two-guard-strategy-cache-vs-load) | L2/01, L2/04 | 3 min |
| 9 | [The Numbers: What Prefix-Aware Routing Delivers](#9-the-numbers-what-prefix-aware-routing-delivers) | L2/02, L2/04, L2/01 | 3 min |
| 10 | [Production Implementations](#10-production-implementations) | L2/01, L2/03, L2/04 | 4 min |
| 11 | [When Prefix-Aware Routing Is Not Worth It](#11-when-prefix-aware-routing-is-not-worth-it) | L2/01, L1/02 | 2 min |
| 12 | [The Stack: Where an Instance Router Fits](#12-the-stack-where-an-instance-router-fits) | L1/01, L1/02, L1/03 | 2 min |

**Total reading time:** ~37 minutes

---

## 1. The Scale Problem: Why One Instance Isn't Enough

A single LLM instance has limits. A small team can prototype with one GPU, one inference engine, and a direct API call — and it works fine. Production doesn't.

Production AI services handle hundreds to thousands of concurrent users, each with their own session context, each expecting a response within a second or two. No single GPU handles that load. The solution is horizontal scaling: run multiple instances of the same model and distribute requests across them.

At this point, a question that seemed trivial becomes architectural: **how do you decide which instance handles which request?**

The Portkey AI Gateway — one of the most widely deployed open-source gateways — processes over **10 billion tokens per day** in production. At that scale, the routing decision runs for every single request, hundreds of times per second. A bad routing policy doesn't just waste a little compute — it systematically destructs the most expensive optimisation modern inference engines perform: the KV cache.

This document explains why, and what the right policy looks like.

---

## 2. Two Patterns: LLM Proxy vs. AI Gateway

Before understanding routers, it helps to understand the spectrum of infrastructure they live in. Two patterns have emerged to solve slightly different versions of the same routing problem.

### The LLM Proxy

An LLM proxy is a lightweight middleware layer that sits between your application and one or more LLM backends. It does one job: forward and shape requests. Common extras — caching, logging, token counting — are useful but secondary.

> "An LLM proxy is fundamentally built for simple, fast routing." — Portkey

**Use when:** prototyping, single-provider setups, or when governance is not a concern.

### The AI Gateway

An AI gateway is a production-grade infrastructure layer designed to manage, govern, and optimise all LLM traffic across an organisation. Every request — regardless of model provider, team, or use case — flows through the gateway, which applies rules, logs activity, and ensures safe, cost-efficient usage.

> "An AI gateway is a **control plane for LLM usage**." — Portkey

| Feature | LLM Proxy | AI Gateway |
|---|---|---|
| Request routing | ✔ | ✔ |
| Basic caching | ✔ | ✔ (advanced, configurable) |
| Logging | ✔ (limited) | ✔ (structured, queryable) |
| Access control | ✘ | ✔ (role-based, multi-tenant) |
| Guardrails & moderation | ✘ | ✔ |
| Budget controls | ✘ | ✔ (per team/app/model) |
| Rate limiting | ✘ | ✔ |
| Multi-provider abstraction | ✔ | ✔ |
| Observability | ✘ | ✔ (latency, error, usage analytics) |
| Audit logs | ✘ | ✔ |
| Prompt management | ✘ | ✔ (version-controlled) |

The key framing: a gateway is not just a reverse proxy — it is the **policy enforcement point** for your entire LLM traffic.

### Where a Simple Instance Router Fits

A minimal instance router — one that distributes requests across replicas of the same model — sits at the **proxy end of this spectrum**: it routes based on a configurable policy, performs health checks, and does basic logging. It has no guardrails, audit logs, or multi-provider abstraction.

The SGLang Model Gateway and vLLM Router are production instance routers that add circuit breakers, Prometheus metrics, and Kubernetes service discovery — moving along the spectrum toward a full gateway, without crossing into the multi-provider abstraction territory that tools like Portkey and LiteLLM cover.

---

## 3. The Gateway Ecosystem: Real Products

Three products dominate open-source LLM gateway deployments in 2026:

| | LiteLLM | Portkey | OpenRouter |
|---|---|---|---|
| **What it is** | Open-source Python proxy + server | Enterprise AI gateway (managed/self-hosted) | SaaS model marketplace |
| **Provider support** | 100+ | 25+ | 200+ models |
| **Open source** | Yes (MIT) | Partial | No |
| **Proxy latency** | ~10–20ms | ~5ms | ~50–100ms |
| **Routing strategies** | round-robin, least-busy, latency-based, cost-based | weighted, conditional, fallback | fixed |
| **Semantic caching** | No | Yes (up to 40% cost reduction) | No |
| **Self-hosted** | Yes | Yes / Cloud | Cloud only |
| **Best for** | Infrastructure-control teams | Enterprise production | Zero-setup multi-model access |

### LiteLLM

LiteLLM translates any LLM call to OpenAI format. It is the most-starred LLM gateway on GitHub (15k+ stars). You self-host the proxy and get unified routing, load balancing, cost tracking, and fallbacks — with full data ownership.

### Portkey

Portkey adds semantic caching (caches similar, not just identical prompts), guardrails for content filtering and PII detection, virtual keys (one Portkey key per team, underlying provider keys centrally managed), and advanced routing (weighted load balancing, conditional routing, automatic failover). Processing 10B+ tokens daily, with sub-1ms latency, 122kb footprint.

### OpenRouter

A SaaS model marketplace — one API key, 200+ models, pay per token. No infrastructure to manage. Not self-hostable.

### The Critical Distinction in This Space

These products route across **different model providers** (OpenAI vs Anthropic vs Gemini). An instance router — like the SGLang Model Gateway or vLLM Router — routes across **different instances of the same model** (engine-0 vs engine-1, both serving Llama-3 70B).

These are **orthogonal routing problems**. They can be composed: a model-provider gateway (LiteLLM/Portkey) sits upstream of an instance router (SGLang/vLLM). The problems are different and solved by different software.

---

## 4. Why Standard Load Balancers Fail for LLM Inference

Three strategies dominate traditional web service load balancing:

- **Round-robin**: Each request goes to the next replica in a fixed cyclic order. Simple, predictable, stateless.
- **Least-loaded**: Routes to the replica with the shortest queue. Adapts to uneven processing times.
- **Power-of-two-choices**: Picks two replicas at random, routes to the less busy one. Near-optimal load distribution with minimal overhead (based on Mitzenmacher's proof that d=2 choices gives exponentially better balance than d=1).

All three share a fundamental assumption: **requests are stateless**. Any replica can handle any request equally well. The work done on request N tells you nothing useful about request N+1.

LLM inference violates this assumption.

### The Hidden Cost of Processing a Prompt

Every LLM request begins with a **prefill phase**: the model reads the entire input prompt in a single forward pass, computing attention key and value vectors for every input token. This computation is both expensive and **stateful** — its result is stored in GPU memory as the KV cache, and it can be reused by future requests.

For a ~10,000-token prompt, prefill takes approximately **4.3 seconds** on a single vLLM instance. If the same prompt arrives again — and the KV cache hasn't been evicted — TTFT drops to **0.6 seconds**. Same hardware, same model: **7× faster** on the second request.

This single-instance optimisation is powerful. The problem starts when you scale out.

### The Scale-Out Problem: Cache Scatter

When you add replicas, each instance maintains its **own isolated KV cache**. A stateless load balancer scatters related requests across replicas — destroying the cache locality that made the single-instance optimisation work.

**Concrete example — multi-turn conversation with round-robin:**

A user sends five messages in a conversation. Each message includes the full conversation history. With round-robin across five replicas:

| Turn | Replica | Tokens computed | Why |
|---|---|---|---|
| Turn 1 | Replica A | 600 (full) | Cold cache |
| Turn 2 | Replica B | 900 (full) | Turn 1's 600 already cached on A — wasted |
| Turn 3 | Replica C | 1,100 (full) | Previous turns cached on A and B — wasted |
| Turn 4 | Replica D | 1,400 (full) | All previous turns wasted |
| Turn 5 | Replica E | 1,700 (full) | Everything computed from scratch |

**Total prefill with round-robin: 5,700 tokens**

If every turn had gone to the same replica:
- Turn 2 → only 300 new tokens (not 900)
- Turn 5 → only 300 new tokens (not 1,700)

**Total prefill with affinity: 1,700 tokens — a 67% reduction.**

At a prefill rate of ~10,000 tokens/sec, Turn 5 alone saves ~140ms TTFT.

> "This isn't a rare event in production — it's the **default behavior** of any distributed deployment with a stateless load balancer." — llm-d team

### The Cascade of Failures

When a cache-blind load balancer routes a request to a replica that doesn't hold the relevant KV cache:

1. **Cache miss**: The warm cache on the correct replica is unused.
2. **Duplicated work**: The same prefill computation runs twice — once on the original replica, again on the new one.
3. **Increased TTFT**: Users wait longer for every response.
4. **Wasted GPU resources**: Hardware is tied up re-doing work it has already done, instead of serving new requests.

The economics are stark: **cached tokens cost roughly 10× less than uncached tokens** ($0.30/M vs $3.00/M for most production inference workloads). An 87.4% cache hit rate — achievable with prefix-aware routing — translates directly into a near-8× reduction in compute cost for the prefill phase.

---

## 5. The KV Cache: LLM's Short-Term Memory

Understanding the KV cache is necessary before understanding why prefix-aware routing works.

### What Is the KV Cache?

Self-attention is the core computation in every transformer layer. For each token in the input, the model computes three vectors: **Query (Q)**, **Key (K)**, and **Value (V)**. During the prefill phase, the model computes K and V for every input token and stores them — this is the **KV cache**.

During the decode phase (generating output tokens one by one), the model uses the cached K and V vectors from the input instead of recomputing them. The KV cache is the model's short-term memory for the current conversation.

> "The KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost." — Manus, Context Engineering for AI Agents

### What Gets Cached and How Long It Stays

KV cache entries are associated with specific token sequences. If two requests share the same prefix — the first 500 tokens are identical — those 500 tokens' K and V vectors can be reused without recomputation.

In a single instance with vLLM's **Automatic Prefix Caching (APC)**, this happens via hash-based block matching: each 16-token block of the KV cache is hashed. A new request triggers a hash lookup; if a matching block exists, it is reused. The result: TTFT drops from 4.3 seconds to 0.6 seconds on a 10,000-token prompt (the second time the same prompt is seen).

### Why This Breaks in Distributed Deployments

In a single-instance setup, all requests share the same cache. In a distributed cluster, each replica has its own isolated cache. A round-robin balancer doesn't know which replica holds a request's relevant cache. It routes to the next available replica — which almost certainly has a cold cache — and throws away all the investment in the warm replica.

The KV cache doesn't transfer automatically between replicas. The warm cache on Replica A is invisible to Replica B. Standard load balancers treat this as a non-problem. **It is the primary efficiency problem in scaled LLM inference.**

---

## 6. The Routing Strategy Ladder

There is a natural progression of routing strategies, each building on the limitations of the previous. Think of it as a ladder: each rung fixes a failure mode of the rung below.

### Rung 1: Round-Robin

Simple, predictable, stateless. Requests cycle through replicas in order.

**Works well for:** Batch inference over diverse, unrelated queries where prefix overlap is negligible.

**Failure mode:** Destroys KV cache locality for any workload with shared prefixes (multi-turn conversations, shared system prompts). All cache investment is wasted.

**Production name:** `random` / `RoundRobin` — the baseline in every router. Use it to measure how bad cache-blind routing actually is.

---

### Rung 2: Least-Load (Power-of-Two Choices)

Routes to the replica with the smallest queue. The power-of-two-choices variant picks two replicas at random and routes to the less busy one — a probabilistic algorithm that achieves near-optimal load distribution with O(1) overhead.

**Improvement over round-robin:** Prevents hotspots when some requests take much longer than others. Reduces queue buildup under uneven load.

**Failure mode:** Still stateless with respect to the KV cache. Does not improve cache hit rates. Better load balance, but still cache-blind.

**Production name:** `PowerOfTwo` / `least-busy` — present in every production router as the safe default for cache-insensitive workloads.

---

### Rung 3: Session Affinity (Sticky Sessions)

Routes all requests from the same user or session to the same replica. The simplest form of cache locality — if the same user always lands on the same replica, their multi-turn conversation always hits the same cache.

**Improvement:** Captures multi-turn cache reuse for individual users.

**Failure mode:** Misses **cross-user cache reuse** (different users with the same system prompt still scatter). Creates load hotspots when some sessions are much heavier than others. No load rebalancing.

---

### Rung 4: Prefix-Cache-Aware Routing

The router maintains an index of which token sequences are cached on which replicas. When a new request arrives, the router:
1. Extracts the request's token prefix.
2. Queries the index for the replica with the longest matching cached prefix.
3. Routes to that replica (maximise cache hit probability).
4. Overrides with load-balance if a replica is significantly more loaded than peers.

**Improvement:** Captures both multi-turn reuse (same user, same replica) and cross-user shared-prefix reuse (different users with the same system prompt, same replica). Balances cache affinity against load balance via configurable thresholds.

**Production names:** `CacheAware` (SGLang), `PrefixCacheAffinityRouter` (Ray Serve), `precise-scheduling` (llm-d).

---

### The Strategy Hierarchy at a Glance

| Strategy | Cache locality | Load balance | Cross-user reuse |
|---|---|---|---|
| Round-robin | ✘ | ✔ (simple) | ✘ |
| Least-load | ✘ | ✔ (adaptive) | ✘ |
| Session affinity | ✔ (per-user) | ✘ | ✘ |
| Prefix-cache-aware | ✔ (per-prefix) | ✔ (guarded) | ✔ |

---

## 7. Prefix-Aware Routing: How It Works

### What Prefixes Are Shared

Two patterns dominate in production workloads and both benefit from prefix routing:

**Pattern 1 — Multi-turn conversations:** Each turn includes the full conversation history as its prefix. The prefix grows with every turn. Chatbots, customer support copilots, and coding assistants all exhibit this pattern. After 5 turns, the prefix is the entire conversation history — often 1,000–3,000 tokens — and it is identical to what was processed in Turn 4.

**Pattern 2 — Shared system prompts:** Many users of the same application share an identical system prompt. An enterprise copilot might have a 6,000-token system prompt (instructions, persona, tool definitions, policy constraints) shared across thousands of user sessions. In production B2B workloads, shared system prompt overlap routinely reaches **40–80% of all input tokens**.

Both patterns create an opportunity: route requests that share a prefix to the replica that has already cached that prefix.

> "Think of it as sorting mail by ZIP code. Traditional routing hands each letter to the next available mail carrier, regardless of destination. Prefix-aware routing gives the Kyoto letters to the carrier who already knows the Kyoto neighborhoods. The map is the KV cache. The ZIP code is the prefix." — OptiVerse

### The Router's Prefix Index

The router maintains a **per-replica prefix index** — a data structure that tracks which token sequences each replica has cached. When a request arrives, the router computes a prefix match score for each replica and routes to the best match.

**SGLang uses a radix tree:** A compact prefix tree where each edge represents a multi-token sequence. The router walks the tree from the root, matching the request's token prefix against existing edges. If the first 1,400 tokens match an existing entry, the tree walk identifies those 1,400 tokens and stops at the divergence point. Only the unmatched suffix needs prefill.

**vLLM uses hash-based block matching:** Each KV cache block (16 tokens) is associated with a hash of its token sequence. New requests trigger a hash lookup to identify reusable blocks.

**llm-d uses a global live index:** A `kvcache.Index` fed by live `KVEvents` emitted by each vLLM pod. The index maps token sequences directly to the pods that hold them, with block-level precision.

All three approaches serve the same function: identify which replica has the most relevant cached context so the router can route to it.

### The Routing Decision (Step by Step)

For an incoming request with prompt `P`:

```
1. Tokenise P → prefix token sequence
2. For each replica, compute: match_ratio = len(matched_prefix) / len(P)
3. Find max_queue_len and min_queue_len across all replicas

4. If (max_queue_len - min_queue_len) > balance_threshold:
   → Route to the least-loaded replica
   (load is too unequal; cache affinity overridden)

5. Else if best match_ratio >= cache_threshold:
   → Route to the replica with the highest match_ratio
   (cache affinity wins)

6. Else:
   → Route to the replica with the smallest prefix tree
   (no match; route to replica with most free cache capacity)
```

---

## 8. The Two-Guard Strategy: Cache vs. Load

Pure prefix-cache-aware routing has a predictable failure mode: if all requests with a popular system prompt route to the same replica, that replica becomes a **hotspot** while others sit idle.

Production systems address this with a **two-guard strategy** — used independently by SGLang's `CacheAware` policy, Ray Serve's `PrefixCacheAffinityRouter`, and llm-d's precise scheduler:

### Guard 1 — Load Balance Check

Compare queue depths across replicas. If the most-loaded replica's queue exceeds the least-loaded replica's queue by more than a configured threshold, **override prefix affinity and route to the least-loaded replica**. This prevents hot-spots from forming.

Parameter names across systems:
- SGLang: `balance_abs_threshold`
- Ray Serve: `imbalanced_threshold`
- llm-d: load-awareness gate

### Guard 2 — Prefix Match Check

Among load-balanced replicas, find the one with the highest prefix match ratio. If the match ratio meets or exceeds a minimum threshold, **route to that replica (cache affinity wins)**. If no replica has a meaningful match, route to the replica with the smallest prefix tree — the one with the most available cache capacity for new entries.

Parameter names:
- SGLang: `cache_threshold` (default: 0.5 — at least 50% of the request must match an existing cache entry to prefer that replica)
- Ray Serve: `match_rate_threshold` (default: 0.1 — more permissive)

### The Two-Guard Algorithm

```
for each incoming request:
    if max_queue - min_queue > balance_abs_threshold:
        → route to least-loaded replica        ← load guard wins
    else:
        if best_match_ratio >= cache_threshold:
            → route to best-prefix replica     ← cache affinity wins
        else:
            → route to smallest-trie replica   ← cache capacity wins
```

**Why this is the right design:** The two-guard strategy is not a compromise — it is the correct formulation of the problem. Both constraints are real: you want cache locality, and you want load balance. The guards enforce priorities: load balance is a hard constraint (you cannot let replicas sit idle while one is overwhelmed), and cache affinity is an optimisation within that constraint.

---

## 9. The Numbers: What Prefix-Aware Routing Delivers

The improvements from prefix-aware routing are not marginal. They are order-of-magnitude.

### The 57× TTFT Gap (llm-d Benchmark)

**Setup:** 8 vLLM pods (16 H100 GPUs total) serving a B2B SaaS workload:
- 150 enterprise customers, each with 6,000-token system prompts
- 5 concurrent users per customer, 1,200-token queries
- Load ramping from 3 QPS to 60 QPS

| Routing strategy | Output tokens/sec | TTFT p90 | TTFT mean | Avg queue depth |
|---|---|---|---|---|
| Precise prefix-cache aware | **8,730** | **0.54s** | **0.30s** | **0.1** |
| Approximate prefix-cache | 6,944 | 31.1s | 13.3s | 8.1 |
| Load-only (cache-blind) | 4,429 | 94.9s | 47.0s | 28.9 |

**Precise prefix-cache routing is 57× faster TTFT than approximate scheduling — on identical hardware serving identical traffic.**

**Why such a large gap?** Cache-blind schedulers constantly duplicate and evict the same prefixes across different pods — "cache thrashing." Every time the 6,000-token system prompt lands on a cold pod, it triggers a 4+ second prefill. With 150 customers × 5 users each, this happens constantly. Precise scheduling routes cache-hit requests consistently, resulting in virtually no queue depth and stable throughput.

### Cache Hit Rate and Cost (Red Hat / llm-d)

**Setup:** 4,776 total queries; 87.4% cache hit rate with llm-d prefix-cache routing.

| Scenario | Without cache routing | With KV cache routing | Improvement |
|---|---|---|---|
| Cold inference | 2,850ms TTFT | 2,850ms TTFT | Baseline |
| Warm cache hit | 2,850ms TTFT | ~285ms TTFT | **10× faster** |
| Multi-turn conversation | 2,200ms avg TTFT | 310ms avg TTFT | **7× faster** |

**87.4% cache hit rate → 87.4% of requests run at 10× lower cost and 10× lower TTFT.**

Cost economics:
- Uncached token: ~$3.00 per million tokens
- Cached token: ~$0.30 per million tokens
- At 87.4% hit rate: effective average cost ~$0.67/M tokens — a **4.5× cost reduction** over cache-blind routing.

### Single-Instance Prefill Savings (OptiVerse)

Within a single replica with prefix caching enabled, SGLang's RadixAttention achieves:
- **5× throughput improvement** on typical workloads
- **6.4× on workloads with high prefix variation** (many different prefixes being cached and reused)

Separately, the multi-turn conversation example:
- Round-robin: 5,700 total tokens computed across 5 turns
- With affinity: 1,700 total tokens — **67% reduction**
- Saving per turn at 10,000 tokens/sec prefill rate: **up to 140ms TTFT per turn**

### vLLM Router vs. Kubernetes Native Load Balancer

| Model | Metric | vs K8s-native | vs llm-d |
|---|---|---|---|
| Llama 3.1 8B (8P+8D pods) | Throughput | **+100%** | +25% |
| Llama 3.1 8B (8P+8D pods) | TTFT | Similar | 1,200ms faster |
| DeepSeek V3 (1P+1D TP8 pods) | Throughput | **+100%** | Similar |
| DeepSeek V3 (1P+1D TP8 pods) | TTFT | **2,000ms faster** | 2,000ms faster |

---

## 10. Production Implementations

Three open-source systems implement prefix-aware instance routing at production scale. All three share the same core concept — prefix index + two-guard strategy — but differ in implementation language, update mechanism, and operational integration.

### SGLang Model Gateway

The most direct reference implementation. Written in **Rust** for minimal proxy overhead. Ships three policies:

- `RoundRobin`: Cyclic distribution, no state.
- `PowerOfTwo`: Two-random-choice least-load, no cache state.
- `CacheAware`: Full prefix-cache-aware routing with a per-worker radix tree and the two-guard strategy.

The `CacheAware` policy:
- Maintains a radix tree per worker, updated from request history (each routed request updates the router's index of what that worker now has cached).
- On each new request, walks all trees to find the best prefix match.
- Applies the two guards: load balance check first, prefix match second.

SGLang CLI to enable:
```bash
python -m sglang.launch_server \
  --router-policy cache_aware \
  --router-cache-threshold 0.5 \
  --router-balance-abs-threshold 32
```

### vLLM Router

Forked from the SGLang Model Gateway, modified for vLLM backends. Also written in Rust.

> "The vLLM Router is derived from a fork of the SGLang model gateway, modified and simplified to work with vLLM." — vLLM team

Policies:
- `PowerOfTwoPolicy`: Least-load with d=2 random sampling.
- `RoundRobinPolicy`: Cyclic distribution.
- `ConsistentHashingPolicy`: Hash-ring sticky routing — requests with the same routing key (session ID, user ID) always go to the same worker. Achieves per-session cache locality without an explicit prefix index, but misses cross-user prefix reuse.

Adds production hardening over the basic gateway:
- Kubernetes service discovery via label selectors
- Configurable retry logic with exponential backoff
- Circuit breakers (not in the SGLang gateway)
- Prometheus `/metrics` endpoint with request volume, latency, error rates, and worker health

Benchmark headline: **100% higher throughput and 2,000ms faster TTFT** than Kubernetes-native load balancing on DeepSeek V3.

### llm-d (Kubernetes-Native)

The most production-complete implementation. Rather than the router-side prefix tree (updated from routing history), llm-d uses a **live global KV cache index** fed by `KVEvents` — a real-time stream of cache state changes emitted by each vLLM pod.

| Component | Layer 15 analog | llm-d equivalent |
|---|---|---|
| Prefix index | Per-worker `RadixTrie` (updated from request history) | `kvcache.Index` (live KVEvents feed) |
| Routing policy | `PrefixCacheAwarePolicy._pick_worker` | Precise Prefix-Cache Scorer |
| Load guard | `balance_abs_threshold` | Load-awareness gate |

**87.4% cache hit rate** on a 4,776-query production workload.

**The critical difference:** Router-side prefix trees (SGLang/vLLM Router) can fall out of sync with actual cache state — a cached prefix may be evicted between the routing decision and request arrival. llm-d's live index is always accurate because it tracks actual evictions in real time.

**Trade-off:** llm-d requires a Kubernetes deployment with vLLM pods emitting KVEvents. It is a full platform, not a standalone router binary.

---

## 11. When Prefix-Aware Routing Is Not Worth It

Prefix-aware routing is an optimisation for specific workload characteristics. It is not a universal replacement for traditional load balancing.

### When it helps most

- **Long shared prefixes**: B2B SaaS workloads with enterprise-specific system prompts (2,000–10,000 tokens), RAG pipelines where retrieved context repeats across requests, coding assistants with large shared codebases as context.
- **Multi-turn conversations**: Each turn appends to an existing prefix. The longer the conversation, the more tokens are saved on each turn.
- **High request volume**: More requests → higher probability of cache hits → larger savings from prefix affinity.

### When it is not worth the complexity

**Low or no prefix overlap**: If your workload sends diverse, unrelated prompts (batch translation, document summarisation over unique documents, diverse creative writing), there is no shared prefix to exploit. Round-robin or least-load is simpler and equally effective.

**Short prompts, short outputs**: When the median prompt is under 512 tokens and generation is under 100 tokens, the absolute prefill saving is small (a few milliseconds). The overhead of maintaining a prefix index and making more complex routing decisions may exceed the gain.

**Small deployments**: Below ~4 replicas, the probability of any cache hit from prefix routing is low simply because there are few replicas to match against. The two-guard strategy adds complexity without meaningful benefit.

**Cache eviction pressure**: If GPU memory is tight and the KV cache eviction rate is high (many diverse prefixes competing for limited space), the router's prefix index may routinely point to already-evicted entries — turning cache-miss into a routing misdirection that also increases load on one replica.

> "Prefix-aware routing is an optimization for high-prefix-overlap workloads, not a universal replacement for traditional load balancing. The best production systems treat it as a routing preference that can be overridden when load balance requires it." — OptiVerse

### The Diagnostic Question

Before adopting prefix-aware routing, measure your actual workload:

1. **What is your system prompt length?** Longer = more benefit.
2. **What fraction of input tokens are shared across requests?** Above 40% → strong candidate for prefix routing.
3. **What is your multi-turn rate?** High multi-turn → session affinity at minimum; prefix-aware routing for cross-user savings.
4. **How many replicas do you have?** Below 4 → minimal benefit; above 8 → strong candidate.

---

## 12. The Stack: Where an Instance Router Fits

The full LLM infrastructure stack has multiple routing layers. Understanding where each fits prevents both over-engineering and under-engineering.

```
Client Application
       ↓
[Model-Provider Gateway]     LiteLLM / Portkey / OpenRouter
  Routes across: OpenAI vs Anthropic vs Gemini
  Policy: cost, latency, fallback, rate limits
       ↓
[Instance Router]            SGLang Gateway / vLLM Router / llm-d
  Routes across: engine-0, engine-1, engine-2 (same model)
  Policy: round-robin, least-load, prefix-cache-aware
       ↓
[Inference Engine]           SGLang / vLLM / TensorRT-LLM
  Serves: model weights, KV cache, output tokens
```

These layers are **orthogonal** — each solves a different routing problem. A model-provider gateway can sit upstream of an instance router; they compose cleanly.

### Maturity Ladder for Instance Routers

| Feature | Minimal router | SGLang Gateway | vLLM Router | llm-d |
|---|---|---|---|---|
| Round-robin | ✔ | ✔ | ✔ | ✔ |
| Least-load | ✔ | ✔ | ✔ | ✔ |
| Prefix-cache-aware | ✔ (router-side trie) | ✔ (Rust, radix tree) | ✔ (consistent hash) | ✔ (live KVEvents) |
| Health checks | ✔ (basic removal) | ✔ (circuit breaker) | ✔ (circuit breaker) | ✔ |
| Implementation | Python | Rust | Rust | Go/Rust |
| Service discovery | Static config | — | Kubernetes labels | Kubernetes GAIE |
| Observability | Logging | Prometheus | Prometheus | Prometheus |
| Live cache index | ✘ | ✘ | ✘ | ✔ (KVEvents) |

**The progression:**

```
Minimal Python router (learning / prototyping)
  → SGLang sgl-model-gateway (Rust, circuit breakers, Prometheus)
    → vLLM Router (+ consistent hashing, Kubernetes discovery)
      → llm-d (+ global live KV cache index, Kubernetes-native GAIE)
        → Portkey / LiteLLM (+ multi-provider, guardrails, budgets)
```

Each step adds production hardening. The core ideas — policy abstraction, health checking, prefix indexing, the two-guard strategy — are present at every level. Everything above the minimal router is operational complexity and performance hardening, not conceptually new routing logic.

---

## Key Quotes

> "An AI gateway is a **control plane for LLM usage**. Every request, regardless of model provider, team, or use case, flows through the gateway." — Portkey

> "The KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost." — Manus (Context Engineering for AI Agents), cited by llm-d

> "This isn't a rare event in production — it's the **default behavior** of any distributed deployment with a stateless load balancer." — llm-d team

> "Think of it as sorting mail by ZIP code. Traditional routing hands each letter to the next available mail carrier, regardless of destination. Prefix-aware routing gives the Kyoto letters to the carrier who already knows the Kyoto neighborhoods. The map is the KV cache. The ZIP code is the prefix." — OptiVerse

> "Prefix-aware routing is an optimization for high-prefix-overlap workloads, not a universal replacement for traditional load balancing. The best production systems treat it as a routing preference that can be overridden when load balance requires it." — OptiVerse

> "The vLLM Router is derived from a fork of the SGLang model gateway, modified and simplified to work with vLLM." — vLLM team

---

## What Is Left Out and Why

### Left out: PD disaggregation

The vLLM Router introduces prefill/decode disaggregation — routing prefill-phase requests to dedicated prefill workers and decode-phase requests to decode workers. This is architecturally distinct from the three-policy routing strategies covered here and requires its own layer to understand correctly. It is a Layer 16 concept.

### Left out: llm-d KVEvents engine internals

llm-d's `KVEvents` stream, `Pool`, `kvcache.Index`, and `PrefixStore` describe the **engine-side** of cache tracking — how vLLM pods emit cache state to the router. This document focuses on the **router-side**: how routing decisions are made given prefix state. The engine-side implementation is relevant when building the feedback loop between inference engine and router (a Layer 16+ concern).

### Left out: Consistent hashing hash-ring details

The vLLM Router's `ConsistentHashingPolicy` maps routing keys to workers via a hash ring. It achieves per-session cache locality without a prefix index, but misses cross-session reuse. The trade-off between consistent hashing and explicit prefix indexing is an implementation question (which to choose) rather than a conceptual question (what does cache affinity mean). Relevant when extending a basic router.

### Left out: Kubernetes YAML and GAIE configuration

The Red Hat/llm-d article includes YAML for the `cache-aware-router` plugin and vLLM pod flags. These are deployment specifics for a Kubernetes-native deployment model different from a standalone router binary. Relevant when deploying prefix-aware routing into a real cluster.

### Left out: Portkey semantic caching, guardrails, virtual keys

These are features of the multi-provider governance layer, not of instance routing. They are mentioned in the ecosystem overview (§3) but not expanded, because they address different problems from instance routing.
