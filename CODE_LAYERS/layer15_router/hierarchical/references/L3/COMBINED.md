# LLM Router / Gateway — L3+L4 Combined Reference

**What this file is:** A deep-mechanism and production-systems synthesis of all L3 and L4 orientation articles on LLM instance routing. This document is a **strict superset** of `L1/COMBINED.md`: every concept covered there appears here, with additional algorithmic detail, parameter references, academic grounding, and production architecture. The reading order moves from algorithmic foundations → parameter-level configuration → the theoretical paper (Preble) → the production system (SGLang Gateway) → what comes after single-region routing.

**Prerequisite:** Read `L1/COMBINED.md` first. That document establishes the problem, the proxy-vs-gateway distinction, why stateless load balancers fail, the strategy ladder, and the basic two-guard algorithm. This document picks up from those foundations.

**Sources synthesized:**
- L3/01 — Ray Serve Docs (2.54.0): `PrefixCacheAffinityRouter` three-tier strategy; full parameter reference; eviction configuration
- L3/02 — SGLang Router CLI Reference: All policy names, parameter defaults, deployment modes, Kubernetes/Prometheus integration
- L3/03 — IBM Research (arXiv:2408.13510): Empirical baseline study; round-robin vs. join-shortest-queue vs. RL-based routing; prefill/decode mixing effects
- L3/preble_explained — Preble ICLR 2025 (deep dive): E2 algorithm; load cost function; global radix tree; post-assignment rebalancing and autoscaling; prefill/decode balancing; local scheduler fairness; two-level architecture; evaluation
- L4/01 — SGLang Model Gateway Docs: Full production architecture; circuit breaker; retries; all CLI parameters; observability (40+ metrics); deployment modes
- L4/02 — Preble ICLR 2025 (paper summary): Abstract, E2 formal framing, architecture, evaluation highlights
- L4/03 — SkyWalker EUROSYS 2026: Cross-region extension; KV cache transfer; benchmark set
- L4/04 — Survey of LLM Inference Systems (arXiv 2025): Academic framing of the three tiers; Mitzenmacher reference; disaggregated runtimes

---

## 1. The Problem — Brief Restatement

*(Full treatment in `L1/COMBINED.md` §1–4. Summarised here for self-containment.)*

LLM inference is not stateless. Every GPU engine maintains a **KV cache** — computed attention key-value tensors for previously processed tokens. Reusing a cached prefix is 7× faster than recomputing it. The problem: a naïve load balancer (round-robin, least-load) scatters related requests across replicas, destroying cache locality. The result is redundant prefill computation, wasted GPU cycles, and latency that degrades with scale.

The 57× TTFT gap between precise prefix-cache-aware scheduling and approximate scheduling (measured by llm-d at 16 H100 GPUs) is not a marginal difference — it is the gap between a working distributed system and a broken one using identical hardware.

---

## 2. The Academic Framework: Three Tiers of Load Balancing

The 2025 Survey of LLM Inference Systems (arXiv:2506.21901) establishes the canonical three-tier classification for load balancing in distributed LLM inference. Every router design falls into one of these tiers:

### Tier 1: Stateless (Round-Robin)

- No worker state tracked at the router.
- Each request goes to the next worker in fixed cyclic rotation.
- Works for: symmetric workloads with negligible prefix overlap (batch inference over diverse, unrelated queries).
- **Failure mode**: Ignores KV cache state, output length variance, and GPU load. Scatters related requests, destroys cache locality.

The IBM Intelligent Router study (arXiv:2408.13510) validates round-robin as the correct **baseline to beat** — not a production choice. The SkyWalker paper (EUROSYS 2026) confirms: under round-robin, two replicas can have memory usage differences up to **2.64×**.

### Tier 2: Load-Aware (Power-of-Two Choices)

- Samples two workers at random, routes to the less loaded one.
- Why two and not all? **Mitzenmacher (2001)** proved d=2 gives exponential improvement over d=1 (random), while d=3 gives only constant improvement over d=2. Two choices is the sweet spot.
- Used in TetriInfer for decode-side load balancing (cited in the 2025 survey).
- Works for: heterogeneous workloads where request durations vary significantly.
- **Failure mode**: Still stateless with respect to the KV cache. Does not improve cache hit rates.

### Tier 3: Cache-Aware (Preble E2 / SGLang Router / llm-d)

- Maintains a per-worker prefix index (radix tree or hash table).
- Routes to the worker with the best prefix match, subject to a load balance guard.
- Captures both multi-turn conversation reuse and cross-user shared-prefix reuse.
- **The design space** for Tier 3 is what most of this document covers.

---

## 3. Empirical Baseline: What IBM Measured

The IBM Intelligent Router study (arXiv:2408.13510) is the most systematic empirical comparison of routing algorithms for LLM workloads on real infrastructure (A100s with Llama 3.1 8B). Key findings:

### Why prefill/decode mixing matters

LLM requests have two distinct phases:
- **Prefill**: Processes the full input prompt. Compute-bound. Cost scales with input length.
- **Decode**: Auto-regressively generates output tokens. Memory-bandwidth-bound. Duration scales with output length.

Routing a long-decode request to an instance already full of long-decode requests creates a bottleneck: short-prefill requests queue behind decode phases. This **mixing effect** is a primary source of latency variance that stateless routing ignores.

### Algorithm comparison results

| Routing Algorithm | Avg E2E Latency (s) | Improvement over Round Robin |
|---|---|---|
| Round Robin | 248.41 | — (baseline) |
| Join-Shortest-Queue | ~240 | ~3–5% |
| Baseline RL router | 240.58 | 3.15% |
| Workload-Aware RL | ~232 | ~7% |
| Workload-Guided RL (best) | ~220 | ~11% |

**Join-Shortest-Queue** — route to the instance with the fewest outstanding prompt and decode tokens — is the practical middle ground: no training required, meaningful improvement over round-robin for heterogeneous workloads. It is what production systems call `power_of_two` or `least-busy`.

**Key insight from IBM**: Even with a perfect oracle for output length (the "Decode Balancer" configuration), the improvement over round-robin is bounded. The real gains come from prefix cache locality — which this study does not measure but motivates.

---

## 4. The Routing Strategy Ladder — Detailed

Building on the three-tier framework, here is the full strategy ladder with mechanism-level detail:

### Rung 1: Round-Robin

Simple cyclic distribution. Request i → worker (i mod N).

**Production CLI:**
```bash
# SGLang
python -m sglang_router.launch_router --policy round_robin

# Ray Serve: default LLM deployment uses round-robin unless overridden
```

**When to use:** Batch jobs over diverse, unrelated queries. Never in production for chat/agent workloads.

---

### Rung 2: Power-of-Two Choices (Least-Load)

Pick two workers uniformly at random. Route to the one with the smaller queue depth.

**Why d=2:** Mitzenmacher 2001 proof — d=2 reduces the maximum load from O(log N / log log N) (pure random) to O(log log N). d=3 gives only O(log log N / log 3) — constant improvement over d=2, not exponential. Two choices is the algorithmic sweet spot.

**Load signal options:**
- `in_flight` count (simple: count of outstanding requests regardless of length)
- Token-weighted load (IBM Join-Shortest-Queue: count outstanding tokens)
- Windowed computation time (Preble: sum of estimated prefill + decode time in past H minutes)

Most production systems use `in_flight` count as the load signal — simple and sufficient.

**Production CLI:**
```bash
# SGLang
python -m sglang_router.launch_router --policy power_of_two

# vLLM Router equivalent: PowerOfTwoPolicy
```

---

### Rung 3: Session Affinity (Sticky Sessions)

Route all requests with the same user/session ID to the same worker. Ensures multi-turn conversations reuse cached context.

**Implementation:** Hash the session ID, map to a worker. Or: consistent hash ring (used by vLLM Router's `ConsistentHashingPolicy`).

**When to use:** Single-user workloads where per-session affinity is enough. Misses cross-user shared prefix reuse (shared system prompts).

**Failure mode:** Creates hotspots when some sessions are much heavier than others. No load balancing.

---

### Rung 4: Prefix-Cache-Aware

The router maintains a per-worker prefix index and routes to the worker with the longest matching cached prefix — subject to a load balance guard.

**Two sub-variants:**
1. **Router-side prefix tracking** (SGLang Router, Ray Serve): Router maintains its own radix tree or hash table, updated from request history. Approximate — the router infers cache state from what it has routed, not from live GPU cache events.
2. **Engine-side cache events** (llm-d via KVEvents): The inference engine emits real-time events when cache entries are created and evicted. The router's index is exact.

Both use the same routing decision logic (§5). The difference is in index accuracy.

---

## 5. Prefix-Cache-Aware Routing: The Algorithm in Detail

### 5.1 The three-tier routing strategy (Ray Serve formulation)

Ray Serve's `PrefixCacheAffinityRouter` implements the canonical three-tier strategy:

```
for each incoming request R with prompt P:
    
    TIER 1 — Load balance check:
    if max_queue_depth - min_queue_depth > imbalanced_threshold:
        → route to least-loaded worker  (load guard wins)
        (Power-of-Two Choices fallback)
    
    else — Load is balanced, proceed to cache routing:
        
        TIER 2 — Prefix match check:
        query prefix tree for each worker
        best_worker = worker with highest match_rate for P
        
        if best_match_rate >= match_rate_threshold:
            → route to best_worker  (cache affinity wins)
        
        else — No strong cache match anywhere:
        
            TIER 3 — Fallback:
            → route to worker with lowest prefix tree utilization
            (most free cache space for new entries)
```

**Why three tiers and not two?**

The third tier (route to the worker with the most free cache) handles a case that pure exploitation misses: a new workload arriving for the first time, with no cache entries anywhere. Routing to the worker with the most available cache gives this new workload the best chance of building a useful cache entry that future requests can hit.

### 5.2 The prefix index: radix tree vs. hash table

**SGLang Router / Ray Serve use a radix tree (RadixTrie):**
- Each edge represents a multi-token sequence.
- A request's prompt is walked from the root, matching tokens left to right.
- The walk stops at the first divergence. Everything before the divergence is already cached.
- Compact: a 1,800-token system prompt shared by 10,000 requests is stored as a single path in the tree.

**vLLM uses hash-based block matching:**
- Each KV cache block (16 tokens) is hashed.
- New requests trigger a block-by-block hash lookup.
- Less space-efficient than a radix tree but simpler to maintain across distributed workers.

**llm-d uses a global KV event stream:**
- Engine pods emit `KVCacheUpdatedEvent` whenever a new prefix is computed.
- Engine pods emit `KVCacheEvictedEvent` whenever a prefix is evicted.
- The router's `kvcache.Index` stays synchronized with actual GPU cache state.
- This is exact (vs. the approximation of router-side trees).

### 5.3 Key parameter reference

All three production implementations converge on the same two tunable thresholds:

| Concept | Ray Serve | SGLang Router | Preble paper |
|---|---|---|---|
| Load imbalance threshold | `imbalanced_threshold` | `--balance-abs-threshold` | `Thbal` |
| Min match ratio for cache routing | `match_rate_threshold` | `--cache-threshold` | E2 exploit condition |

**Defaults across systems:**

| Parameter | Ray Serve default | SGLang default | Recommended start |
|---|---|---|---|
| Load threshold | `infinity` (no load guard) | `64` requests | `32` |
| Min match ratio | `0.1` (10%) | `0.3` (30%) | `0.5` |

**Tuning guidance:**
- Lower `imbalanced_threshold` → router prioritises load balance over cache hits. Use when workload is highly heterogeneous (large variance in request sizes).
- Lower `match_rate_threshold` → router is more aggressive about exploiting cache even with weak matches. Use when cache warmup is slow and any reuse is valuable.
- Higher `match_rate_threshold` → router only exploits cache when there is a strong match. Use when you want to avoid building up prefix debt on a single worker.

### 5.4 Prefix tree eviction

Router-side prefix trees grow unboundedly without eviction. Production systems manage this:

**Ray Serve eviction parameters:**
```python
request_router_kwargs={
    "do_eviction": True,
    "eviction_threshold_chars": 500_000,  # Trigger eviction at 500K chars stored
    "eviction_target_chars": 400_000,     # Evict down to 400K chars
    "eviction_interval_secs": 30,         # Check every 30 seconds
}
```

**SGLang Router eviction parameters:**
```bash
--eviction-interval-secs 120 \
--max-tree-size 67108864   # 64M nodes maximum
```

Without eviction, the prefix tree accumulates entries for rare prompts indefinitely, consuming memory and slowing prefix lookups.

### 5.5 Full configuration examples

**Ray Serve (Python API):**
```python
from ray import serve
from ray.serve.llm import LLMConfig, build_openai_app
from ray.serve.llm.request_router import PrefixCacheAffinityRouter

llm_config = LLMConfig(
    model_loading_config={
        "model_id": "llama-3-70b",
        "model_source": "meta-llama/Meta-Llama-3-70B-Instruct",
    },
    deployment_config={
        "autoscaling_config": {"min_replicas": 4, "max_replicas": 4},
        "request_router_config": {
            "request_router_class": PrefixCacheAffinityRouter,
            "request_router_kwargs": {
                "imbalanced_threshold": 5,
                "match_rate_threshold": 0.15,
                "do_eviction": True,
                "eviction_threshold_chars": 500_000,
                "eviction_target_chars": 400_000,
                "eviction_interval_secs": 30,
            },
        },
    },
    runtime_env={"env_vars": {"VLLM_USE_V1": "1"}},
)
app = build_openai_app({"llm_configs": [llm_config]})
serve.run(app, blocking=True)
```

**SGLang Router (CLI):**
```bash
# Separate launch mode: start workers first, then router
python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8000

python -m sglang.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --port 8001

python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware \
  --cache-threshold 0.5 \
  --balance-abs-threshold 32 \
  --balance-rel-threshold 1.5 \
  --eviction-interval-secs 120 \
  --max-tree-size 67108864 \
  --host 0.0.0.0 \
  --port 30000 \
  --prometheus-port 29000

# Co-launch mode (simpler): router and workers start together
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 \
  --port 30000
```

---

## 6. Preble: The Theoretical Foundation

*(Srivatsa et al., ICLR 2025, arXiv:2407.00023)*

Preble is the academic paper that puts the two-guard heuristic on a rigorous theoretical and empirical footing. It introduces the **E2 algorithm** — a principled approach to the exploitation/exploration tradeoff in distributed KV cache scheduling — and demonstrates 1.5×–14.5× latency improvement over state-of-the-art distributed serving systems.

### 6.1 Why modern prompts are expensive to scatter

Preble studied five real-world workloads and found that prefix sharing is not an edge case — it is the default:

| Workload | Avg prompt length | Avg output length | Prompt/output ratio | Shared prefix % |
|---|---|---|---|---|
| Tool use (Toolbench) | 1,835 tokens | 43 tokens | **43×** | **85%** |
| Embodied agent | 2,285 tokens | 16 tokens | **143×** | **97%** |
| Program generation | 3,871 tokens | 190 tokens | **20×** | **97%** |
| Video QA | 9,865 tokens | 4 tokens | **2,466×** | **88%** |
| Long document QA | 23,474 tokens | 16 tokens | **1,467×** | **91%** |

Three observations:
1. **Prompts massively outnumber outputs.** The prefill phase — processing the input — dominates total request time for modern LLM use cases.
2. **Most prompt tokens are shared across requests.** 85–97% of tokens in a typical prompt are shared with at least one other request.
3. **One key prefix is shared by many requests.** Each request has a "key portion" — the deepest node in its prefix tree holding more tokens than all predecessors combined. On average, 8.6 to 126 requests share this key portion.

**Implication:** The performance bottleneck for modern LLM serving is not generating output — it is recomputing the same enormous input over and over on different GPUs. Fix that, and the gains are order-of-magnitude.

### 6.2 The E2 algorithm: decision rule

E2 resolves the exploitation/exploration tradeoff per-request:

```
function schedule(request R with prompt P):
    
    match P against the global radix tree
    
    cached_len = number of tokens P shares with tree entries (already cached somewhere)
    missed_len = len(P) - cached_len
    
    if cached_len > missed_len:
        # Cache savings > new computation cost → EXPLOIT
        candidates = all GPUs that hold the longest matching node
        return argmin over candidates: load_cost(GPU, R)
    
    else:
        # New computation ≥ cache savings → EXPLORE
        if any GPU has decode_ratio >> others:
            return that GPU  # prefill/decode balancing opportunity (see §6.5)
        return argmin over all GPUs: load_cost(GPU, R)
```

**The key insight:** Only exploit (favour cache affinity) when the saved recomputation outweighs the cost of concentrating load on an already-busy GPU. If a request has 2,000 shared tokens and 50 new tokens — exploit. If it has 1,000 shared and 2,000 new — explore.

The `cache_threshold` parameter in SGLang's production implementation is a simplified binary version of this ratio: `cached_len / len(P) >= cache_threshold`.

### 6.3 The load cost function

When E2 needs to compare GPUs, it uses a unified three-part cost function that estimates total GPU compute time if this request is assigned to GPU_i:

```
load_cost(GPU_i, request R) = L_i + M_i + P_i
```

**L_i — existing load on GPU_i (history window H, default: 3 minutes):**

For each request that has run on GPU_i in the past H minutes, estimate its compute cost:

```
L_i = Σ over requests r in window H:
          prefillTime(missed_tokens_of_r) + decodeTime(avg_output_length)
```

Prefill time and decode time are both approximately linear in token count (verified empirically in the paper). The router uses only token counts — no GPU communication or profiling overhead at routing time. The regression coefficients are measured offline once per GPU type.

**M_i — eviction cost (what we'd lose by running R on GPU_i):**

GPUs are always at full memory capacity. Running R on GPU_i requires evicting some cached KV entries. Evicting frequently-used entries is expensive — future requests must recompute them.

```
M_i = Σ over entries j that would be evicted:
          prefillTime(length_of_j) × hit_rate_of_j

where hit_rate_of_j = (requests using j in window H) / (total requests on GPU_i in window H)
```

A cache entry used by 100 requests has high hit rate — evicting it is very expensive. An entry used once — cheap to evict.

**P_i — cost of running the new request R on GPU_i:**

```
P_i = prefillTime(missed_len_of_R)
```

This is the prefill cost for the tokens of R not already cached on GPU_i. Decode time is the same regardless of which GPU handles R (same model, same output length).

**Total:**
```
load_cost(GPU_i, R) = L_i + M_i + P_i
```

Pick the GPU with the lowest total. This balances: current load, value of what would be evicted, and cost of running the new request — all in the same units (GPU compute time).

**Why this matters:** Simpler production implementations (SGLang Router, Ray Serve) use queue depth as the load signal — a point-in-time count rather than a windowed time estimate. This loses information about long-running requests, but is much simpler and still achieves most of Preble's benefit.

### 6.4 The global radix tree

The global scheduler maintains a radix tree tracking every prompt sequence processed across the entire cluster.

**Structure:** A radix tree compresses sequences by merging common prefixes. Two requests that share the first 1,800 tokens share the same path down to the same node, then diverge into separate child nodes. For each node, the scheduler records:
1. Number of tokens in this node's sequence
2. Which GPUs have this node's KV entries cached
3. Per-GPU request count in the past H minutes

**Insertion:** When a new request arrives:
1. Walk tree from root, matching tokens left to right.
2. If tokens match an existing path → full prefix hit (cache is warm somewhere).
3. If tokens match partway → split existing node at divergence point, insert new child.
4. If no tokens match → create new root node (new prefix family).

**Eviction notification:** When a GPU evicts a cached KV entry, it notifies the global scheduler. The scheduler removes that GPU from the node's caching-GPU set. If no GPU caches a node and no recent requests used it, the node is pruned from the tree.

**Why this works:** The global radix tree gives the scheduler a complete, up-to-date view of what is cached where — without querying any GPU directly. The tree is the index.

**Implementation:** The tree is accessed by many concurrent request handlers. Preble makes it **lock-free**: most operations (prefix matching, reading GPU assignments) are read-only — no locks needed. Only two write operations ("assign a GPU to a node" and "increment request count") are expressed as atomic instructions. This allows the global scheduler to handle hundreds of requests per second without becoming a bottleneck.

### 6.5 Post-assignment: handling load changes

E2's per-request decision is greedy — optimal at the moment of routing, but popular prefixes can surge. Preble adds two post-assignment mechanisms:

**Mechanism 1: Load rebalancing (shift load)**

The scheduler continuously monitors per-GPU load. If the most-loaded GPU's load is more than `Thbal` × the least-loaded GPU's load:

```
if max_GPU_load > Thbal × min_GPU_load:
    redirect future requests (that would exploit the heavy GPU)
    → send to the light GPU instead
    continue until load difference < Thbal
```

This is done without moving cached data — the light GPU simply handles the request at the cost of a cache miss. It accepts higher recomputation in exchange for load balance.

Production systems simplify this: if `max_queue - min_queue > balance_abs_threshold`, route to the least-loaded worker instead of the best-prefix worker.

**Mechanism 2: Prefix autoscaling (replicate the cache)**

Sometimes load rebalancing is not enough. If a prefix becomes so popular that its request load exceeds what a single GPU can handle — even after redirecting requests — Preble **replicates** the cached KV entries onto additional GPUs:

```
if avg_queue_time for prefix_subtree doubles over window H:
    select second GPU to hold this prefix
    copy KV entries → second GPU
    split future requests between both GPUs (now both can serve with full cache hits)
```

If demand grows further, a third GPU can be added. **This is not in any production router today.** It requires the router to coordinate KV cache state transfer between engines — a capability that currently lives only in research systems. The production fallback (SGLang Router, Ray Serve) accepts recomputation when a prefix is hot, rather than replicating the cache.

### 6.6 Prefill/Decode balancing

This is Preble's less obvious but impactful contribution.

Prefill and decode have fundamentally different GPU utilization profiles:
- **Prefill**: Compute-bound. GPU arithmetic units are maxed out processing all input tokens in one parallel forward pass.
- **Decode**: Memory-bandwidth-bound. GPU generates one token at a time, reading model weights from memory on each step. Arithmetic units are underused.

When exploring (no strong cache preference), Preble checks each GPU's `decode_ratio` (fraction of requests currently in decode phase):

```
if GPU_i has high decode_ratio:
    → route R to GPU_i
    # R will trigger prefill (compute-bound)
    # This fills GPU_i's idle arithmetic units during decode
    # Both phases run simultaneously, better overall utilization
```

A cache-miss request (needs full prefill) is treated as a **prefill-phase compute unit** and deliberately sent to a decode-heavy GPU. A full cache hit (only decode needed) is a **decode-phase unit** and should go to a compute-heavy GPU.

**Ablation result:** Prefill/decode balancing contributes meaningful improvement to both average and tail latency, on top of E2 routing and load rebalancing. It is orthogonal to cache-aware routing — one helps where requests go, the other helps what happens after they arrive.

### 6.7 The local scheduler: fairness inside a GPU

Each GPU also runs a **local scheduler** that orders requests within its wait queue. This is separate from the global routing decision.

**The starvation problem:** A naïve local scheduler processes requests with the highest cache hit ratio first (they are faster to process, improving throughput). But a request with zero cache hits — a new topic, a new user, a unique prompt — waits forever in a busy system.

**Preble's priority-based wait queue:**

Each request is assigned a priority group based on its cache hit ratio:
```
hit_ratio = cached_tokens / total_prompt_tokens
priority_group = floor(hit_ratio × P)   # P groups total, e.g. P=10

# group 10 = 90–100% cache hit (fastest)
# group 5  = 50–60% cache hit
# group 0  = 0–10% cache hit (slowest, needs full prefill)
```

When forming the next batch:
```
# Selecting 55 requests with P=10 groups
group 10 → 10 requests
group 9  →  9 requests
group 8  →  8 requests
...
group 1  →  1 request
group 0  →  1 request
```

Higher-hit groups get more requests per batch (they are faster and contribute more to throughput). Every group gets at least some representation per batch — no starvation.

**Effect:** Requests with full cache hits are processed quickly (improving average latency). Requests with zero cache hits are never starved (controlling tail latency and ensuring fairness in multi-tenant environments).

### 6.8 Two-level architecture

Preble's complete architecture decouples global and local scheduling:

```
Incoming request
       ↓
  [Tokenizer pool]  ← parallel tokenization (separate from scheduling)
       ↓
  [Global Scheduler]  ← one scheduler node per cluster
     │
     │  Maintains:
     │  - Global radix tree (which prefixes are cached, on which GPUs)
     │  - Per-GPU load estimates (updated on each assignment and eviction)
     │  - Background threads: load rebalancing, autoscaling, eviction bookkeeping
     │
     ↓             ↓             ↓
  [GPU-1]       [GPU-2]       [GPU-N]
  Local          Local          Local
  Scheduler      Scheduler      Scheduler
     │
     │  Each maintains:
     │  - Local radix tree (actual cached KV entries on this GPU)
     │  - Priority wait queue (fairness-aware ordering)
     │  - Chunked prefill (break long prompts into smaller pieces)
     │  - LRU eviction (remove least-recently-used entries when memory fills)
     │  - Reports evictions → Global Scheduler (background, not on critical path)
```

**Why two levels?**
- The global scheduler needs a cluster-wide view to make routing decisions. One centralized view is necessary.
- The local scheduler needs sub-millisecond response times to manage individual GPU iterations. A remote global scheduler would be too slow for iteration-level decisions.
- Decoupling them lets each optimize for its timescale: global for request placement (seconds), local for batch formation (milliseconds).

**Scalability:** Preble's global scheduler handles 245–2,931 requests/second depending on radix tree complexity, sustaining 70–391 concurrent A100 GPUs on a single scheduler instance.

### 6.9 Chunked prefill (local scheduler detail)

Long prompts processed all at once cause the GPU to be busy with prefill for hundreds of milliseconds, blocking all decode requests in the queue. The local scheduler **chunks** long non-shared prompts: splits them into smaller pieces and interleaves them with decode requests in each batch.

A request needing 5,000 tokens of prefill might be split into 5 chunks of 1,000 tokens, with decode work filling the gaps between chunks. This keeps the GPU busy on both prefill and decode concurrently, reducing head-of-line blocking.

### 6.10 Evaluation results

**Setup:**
- Models: Mistral 7B, Llama-3 70B
- Hardware: 2× and 4× NVIDIA A6000 GPUs; 8× NVIDIA H100 GPUs
- Baseline: SGLang and vLLM with round-robin distribution (the standard distributed serving approach)
- Workloads: Toolbench (tool use), ALFWorld (embodied agents), APPS (programming), NExT-QA (video QA), LooGLE (document QA)

**End-to-end results** (Preble vs. SGLang/vLLM with round-robin):

| Metric | Improvement range |
|---|---|
| Average latency | **1.5× to 14.5×** |
| p99 (tail) latency | **2× to 10×** |

Gains are largest on workloads with the highest prompt-to-output ratios (Video QA at 2,466×, tool use at 43×). Even on programming workloads with longer outputs, Preble achieves 1.56×–1.8× average latency improvement.

**Ablation — which component contributes what (Toolbench workload):**

| Configuration added incrementally | Average latency | p99 latency |
|---|---|---|
| Round-robin (baseline) | 100% | 100% |
| + E2 per-request routing | ↓ significantly | ↓ significantly |
| + Load rebalancing & autoscaling | ↓ further | ↓ further (especially p99) |
| + Prefill/decode awareness | ↓ further | ↓ further |
| + Priority wait queue (local) | same average | ↓ further p99 only |

Key takeaways:
- **E2 routing** gives the single largest improvement — smart routing beats all else.
- **Rebalancing + autoscaling** primarily reduces tail latency — prevents hotspot accumulation.
- **Prefill/decode balancing** improves both — better GPU utilization through phase mixing.
- **Priority wait queue** is pure fairness — reduces tail latency without hurting average.

**Azure trace (real bursty traffic):**

| Metric | Preble vs. SGLang (bursty trace) |
|---|---|
| Average TTFT | **4.61× improvement** |
| TPOT (time per output token) | **5.60× improvement** |
| Average latency | **6.02× improvement** |
| p99 latency | **2.42× improvement** |

Real request patterns are bursty. Round-robin degrades faster under bursts than Preble because requests pile up on one GPU while others are idle.

---

## 7. SGLang Model Gateway: The Production Reference

The SGLang Model Gateway (`sglang-router`, written in Rust) is the production implementation of all three routing policies. Every parameter in the SGLang CLI reference corresponds to a design decision made by Preble or the general distributed systems literature.

### 7.1 Architecture overview

**Control Plane:**
- **Worker Manager**: Discovers worker capabilities (`/server_info`, `/get_model_info`), tracks load, registers and removes workers.
- **Job Queue**: Serializes add/remove requests, exposes status via `/workers/{worker_id}`.
- **Load Monitor**: Feeds cache-aware and power-of-two policies with live worker load statistics.
- **Health Checker**: Continuously probes workers, updates readiness, circuit breaker state, and metrics.

**Data Plane:**
- **HTTP routers** (regular and PD-disaggregation): Implement `/generate`, `/v1/chat/completions`, `/v1/completions`, `/v1/responses`, `/v1/embeddings`, `/v1/rerank`, `/v1/classify`, and admin endpoints.
- **gRPC router**: Streams tokenized requests to SRT gRPC workers — tokenizer, reasoning parser, and tool parser in-process.
- **OpenAI router**: Proxies OpenAI-compatible endpoints while keeping chat history and MCP sessions local.

### 7.2 All five routing policies

| SGLang `--policy` | Description |
|---|---|
| `random` | Uniform random selection — not typically used in production |
| `round_robin` | Cycles through workers in order — baseline/comparison |
| `power_of_two` | Samples two workers, routes to the lighter one — default for latency-sensitive, non-prefix workloads |
| `cache_aware` | Prefix-cache-aware with two-guard strategy — **production default** |
| `bucket` | Divides workers into load buckets with dynamic boundaries — for highly heterogeneous workloads |

### 7.3 Full parameter reference

**Core routing:**
```bash
--host 127.0.0.1
--port 30000
--worker-urls http://w1:8000 http://w2:8001
--policy cache_aware
--max-concurrent-requests 64
--request-timeout-secs 600
--max-payload-size 256MB  # default 256 MB
```

**Cache-aware tuning:**
```bash
--cache-threshold 0.3           # Min prefix match ratio (SGLang default: 0.3, Layer15: 0.5)
--balance-abs-threshold 64      # Absolute queue difference (SGLang default: 64)
--balance-rel-threshold 1.5     # Relative queue ratio before rebalancing
--eviction-interval-secs 120    # Router tree eviction cadence
--max-tree-size 67108864        # Max nodes in cache tree (64M)
```

**Retries:**
```bash
--retry-max-retries 5
--retry-initial-backoff-ms 50
--retry-max-backoff-ms 30000
--retry-backoff-multiplier 1.5
--retry-jitter-factor 0.2
```
Retryable HTTP status codes: 408, 429, 500, 502, 503, 504.

**Circuit breaker (per-worker):**
```bash
--cb-failure-threshold 5         # Failures before opening circuit
--cb-success-threshold 2         # Successes before closing circuit
--cb-timeout-duration-secs 30    # Time in Open state before trying HalfOpen
--cb-window-duration-secs 60     # Sliding window for failure counting
```

States: `Closed` (normal) → `Open` (too many failures) → `HalfOpen` (testing recovery) → `Closed` (recovered).

A simple health check loop (periodic HTTP ping, remove worker if unhealthy) is the minimal version of this. The circuit breaker adds the three-state machine: rather than immediately removing a failed worker, it opens the circuit and periodically tests recovery — preventing thrashing on flapping workers.

**Kubernetes integration:**
```bash
--service-discovery                           # Enable K8s pod discovery
--service-discovery-namespace default         # Watch this namespace
--bootstrap-port-annotation sglang.port       # Annotation for worker ports
```

**Observability:**
```bash
--prometheus-port 29000     # Prometheus metrics endpoint
--prometheus-host 127.0.0.1
--log-dir ./router_logs
--log-level info
```

**Key Prometheus metrics (40+ total):**
- `sgl_request_total` — total requests by policy
- `sgl_request_duration_seconds` — latency histogram
- `sgl_worker_load` — per-worker queue depth
- `sgl_cache_hit_rate` — prefix cache hit ratio
- `sgl_worker_health` — worker health state
- `sgl_circuit_breaker_state` — per-worker circuit breaker state

### 7.4 Deployment modes

**Mode 1: Separate launch (most flexible)**
```bash
# Start workers
python -m sglang.launch_server --model ... --port 8000
python -m sglang.launch_server --model ... --port 8001

# Start router pointing at workers
python -m sglang_router.launch_router \
  --worker-urls http://worker1:8000 http://worker2:8001 \
  --policy cache_aware
```

**Mode 2: Co-launch (simpler, single-node)**
```bash
python -m sglang_router.launch_server \
  --model meta-llama/Meta-Llama-3.1-8B-Instruct \
  --dp-size 4 \
  --host 0.0.0.0 --port 30000
```

**Mode 3: Prefill/Decode disaggregation**
```bash
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://prefill1:30001 9001 \
  --decode http://decode1:30011 \
  --prefill-policy cache_aware \
  --decode-policy power_of_two
```
Prefill and decode workers are separate pools. The router orchestrates the two-phase handoff. This is a distinct architectural mode: prefill workers process prompts and pass KV state to decode workers, which complete token generation. The `--prefill-policy cache_aware` ensures that the expensive prefill computation is cache-aware; `--decode-policy power_of_two` load-balances the memory-bound decode phase.

**Mode 4: Kubernetes service discovery**
```bash
python -m sglang_router.launch_router \
  --service-discovery \
  --service-discovery-namespace production \
  --bootstrap-port-annotation sglang.ai/port
```
Workers are discovered automatically via K8s label selectors — no static `--worker-urls` needed.

### 7.5 Worker management API

```bash
# List all workers with health and load
curl http://localhost:30000/workers

# Response
{
  "workers": [{
    "id": "2f3a0c3e-...",
    "url": "http://0.0.0.0:31378",
    "model_id": "meta-llama/Llama-3.1-8B",
    "is_healthy": true,
    "load": 12,
    "worker_type": "regular"
  }]
}

# Add a new worker at runtime
curl -X POST http://localhost:30000/add_worker \
  -H "Content-Type: application/json" \
  -d '{"url": "http://new-worker:8002"}'

# Remove a worker
curl -X POST http://localhost:30000/remove_worker \
  -H "Content-Type: application/json" \
  -d '{"url": "http://worker1:8000"}'
```

### 7.6 Source code mapping (SGLang Rust → concepts)

| SGLang gateway file | Implements |
|---|---|
| `src/policies/round_robin.rs` | Round-robin cyclic distribution |
| `src/policies/power_of_two.rs` | Power-of-two choices (least-load) |
| `src/policies/cache_aware.rs` | Two-guard cache-aware routing with radix tree |
| `src/core/worker.rs` | Worker state (URL, health, in-flight count) |
| `src/routers/http/router.rs` | HTTP proxy loop, request dispatching |
| `src/health/circuit_breaker.rs` | Three-state circuit breaker per worker |
| `src/metrics/prometheus.rs` | 40+ Prometheus metrics |
| `src/k8s/service_discovery.rs` | Kubernetes label-selector pod discovery |

---

## 8. Cross-Region Extension: SkyWalker

*(Xia et al., EUROSYS 2026, arXiv:2505.24095)*

SkyWalker extends single-region prefix-aware routing to multi-region deployments. It is useful both as a "what comes next" system and as a benchmark context: its evaluation explicitly uses SGLang Router as the single-region baseline, confirming that the three-policy hierarchy (round-robin < least-load < cache-aware) is the recognised production benchmark set.

### 8.1 What SkyWalker adds beyond single-region routing

**Problem it solves:** LLM traffic has diurnal (time-of-day) variation across regions. A system with fixed regional capacity wastes GPUs in low-traffic periods and becomes overloaded in peaks. Cross-region routing can redistribute excess traffic to underloaded regions.

**But naive cross-region routing destroys cache locality:** routing a request from Region A to Region B because Region B has spare capacity means Region B has no KV cache for the request's prefix — full recomputation.

**SkyWalker's solution: cache-aware cross-region traffic handler**

Routes requests to the region with the most matching prefix cache, analogous to single-region `PrefixCacheAffinityRouter` — but applied across geographic regions rather than replicas within one region.

**Selective pushing-based load balancing:**

When load is imbalanced, pushes excess requests from overloaded replicas to underloaded ones. More aggressive than the single-region load guard — instead of simply routing away from hotspots, it actively migrates in-flight request capacity.

**KV cache transfer (the new capability):**

When routing to a non-cache-holding replica is necessary (due to load balance), SkyWalker can transfer the KV cache blocks from the cache-holding replica to the target replica. The target replica receives the KV state without needing to recompute it.

This eliminates the recomputation cost even when cache affinity is overridden by load balance — the key limitation of the single-region two-guard strategy.

### 8.2 Benchmark results

SkyWalker's evaluation baseline set matches the three-policy hierarchy exactly:

| System | Policy | Single-region analog |
|---|---|---|
| GKE Gateway | Network load balancer (round-robin per connection) | Round-robin |
| Round Robin (RR) | Stateless cyclic | Round-robin |
| Least Load (LL) | Outstanding request count | Power-of-two / least-load |
| Consistent Hashing (CH) | Ring hash on session ID | Session affinity |
| SGLang Router (SGL) | **Prefix-aware, two-guard** | Cache-aware (production baseline) |
| SkyWalker | Cross-region cache-aware + KV transfer | — (the new system) |

**Results vs. baselines:**

| Baseline | Throughput vs. SkyWalker | Latency vs. SkyWalker |
|---|---|---|
| Round Robin | 1.12–2.06× worse | 1.74–6.30× higher |
| Least Load | Still worse | Still higher |
| SGLang Router | Worse in cross-region | Higher in cross-region |

Total serving cost reduction: **25%** (by moving traffic to lower-cost regions during off-peak hours without sacrificing cache hit rates).

### 8.3 Round-robin memory imbalance confirmation

> "We find two replicas under round-robin can have memory usage difference up to **2.64×**." — SkyWalker §2.3

This quantifies what the single-region analysis shows qualitatively: round-robin is not just cache-blind, it creates severe resource imbalance because LLM output lengths are non-deterministic and request processing times vary by orders of magnitude.

---

## 9. What Workloads Benefit Most — and Least

Based on Preble's workload analysis and the broader benchmark literature:

### High benefit (prefix-cache-aware routing wins decisively)

| Workload type | Why prefix routing wins |
|---|---|
| Multi-turn chat / conversational agents | Each turn contains the full conversation history; turn N shares 80–95% of tokens with turn N-1 |
| Tool-use / function-calling agents | System prompt + tool definitions (1,800+ tokens) shared across all requests using the same tools |
| Long document Q&A | Same 20,000-token document sent with different questions; 97% prefix overlap |
| Video / multi-modal QA | Tokenized video/image (thousands of tokens) shared across all questions about that content |
| Enterprise chatbots | Large system prompts encoding persona, policies, and product knowledge shared by all users |
| Few-shot learning workloads | Fixed few-shot examples sent with every request |

**Signal to look for:** prefix overlap routinely 40–80% of input tokens in production workloads (measured independently by llm-d: 87.4% cache hit rate on B2B SaaS workload with 6,000-token system prompts).

### Moderate benefit

| Workload type | Notes |
|---|---|
| Code completion | Some shared context (project files, imports) but high variability in per-request context |
| Retrieval-augmented generation (RAG) | Retrieved chunks vary per query; only the system prompt is consistently shared |
| Summarisation | Long documents shared within a session, but different documents across sessions |

### Low benefit (round-robin or least-load is sufficient)

| Workload type | Why prefix routing doesn't help |
|---|---|
| Diverse single-turn queries | No prefix sharing across requests; each query is unique |
| Batch embedding generation | Each document is unique; no KV cache reuse |
| Classification workloads | Short, diverse prompts; low prefix overlap |

---

## 10. The Full Production Architecture: From Minimal to Full-Scale

Building on `L1/COMBINED.md` §9, here is the complete production maturity ladder:

```
Minimal Python router (prototype / learning)
  Policies: RoundRobin, LeastLoad (in-flight count), PrefixCacheAware (router-side radix tree)
  Health: Simple HTTP ping, binary healthy/unhealthy
  Config: Static YAML (worker URLs)
  Missing: Circuit breaker, eviction, retries, observability, Kubernetes
  ↓
SGLang sgl-model-gateway (Rust)
  + Sub-millisecond routing overhead
  + Three-state circuit breaker per worker
  + Exponential backoff retries
  + 40+ Prometheus metrics + OpenTelemetry
  + Tree eviction (--eviction-interval-secs, --max-tree-size)
  + Kubernetes service discovery
  + gRPC routing
  + `bucket` policy for heterogeneous workers
  ↓
vLLM Router (Rust, vLLM backends)
  + Consistent hashing (session-sticky without prefix tracking)
  + Prefill/Decode disaggregation (separate prefill and decode worker pools)
  + DeepSeek-specific optimisations (TP8 prefill + TP8 decode)
  ↓
llm-d (Kubernetes-native, Go/Rust)
  + Live KV event stream (exact cache state, not router approximation)
  + Global kvcache.Index (precise prefix scorer)
  + Kubernetes Gateway API Integration Engine (GAIE)
  + Inference Scheduling Extension (ISE) interface
  ↓
Preble (research system, single-region)
  + E2 algorithm (full load cost function: L_i + M_i + P_i)
  + Lock-free global radix tree (cluster-wide prefix index)
  + Load rebalancing (redirect requests from hot to cold GPUs)
  + Prefix autoscaling (replicate KV entries to a second GPU)
  + Prefill/decode balancing (fill decode GPUs' idle arithmetic units)
  + Priority wait queue local scheduler (fairness without starvation)
  ↓
SkyWalker (EUROSYS 2026, cross-region)
  + Cross-region cache-aware traffic routing
  + Selective pushing-based load balancing
  + KV cache block transfer between replicas (eliminates recomputation on redirect)
  + Regional traffic aggregation (diurnal load shifting)
  ↓
Portkey / LiteLLM (model-provider layer, above instance routing)
  + Multi-provider routing (OpenAI vs Anthropic vs Gemini vs self-hosted)
  + Semantic caching (deduplicate semantically equivalent requests)
  + Guardrails and moderation
  + Budget controls per team/app/model
  + Audit logging and compliance
```

**Where each system focuses:**
- Minimal router → teaches the core algorithm
- SGLang Gateway → production hardening of the core algorithm
- vLLM Router → vLLM-specific optimisations and PD disaggregation
- llm-d → exact cache state instead of approximation
- Preble → theoretical optimum for single-region serving
- SkyWalker → cross-region extension with active KV transfer
- Portkey/LiteLLM → model-provider layer (orthogonal routing problem)

---

## 11. Key Quotes

> "Current distributed LLM serving systems are not prompt-cache-aware; they attempt to distribute LLM computation load equally across GPUs to achieve high cluster-level GPU utilization. Yet, this distribution could result in requests with shared prefixes being sent to different GPUs, causing KV computation at all these GPUs that could otherwise be avoided." — Preble (ICLR 2025)

> "E2 allows requests to exploit computed prompt prefixes on the same GPU but also gives chances for requests with shared prefixes to explore other GPUs. E2 chooses exploitation when the amount of recomputation saved is larger than that of new computation." — Preble (ICLR 2025)

> "A request with its entire prompt shared and cached would only perform the decoding phase. Thus, it can be treated as a decoding-phase computing unit. Meanwhile, a request with a long prompt not cached and a short output length can be treated as a prefill-phase computing unit." — Preble (ICLR 2025)

> "Better load balancing across LLM instances can improve end-to-end latency more than optimizing the instance-level scheduler." — IBM Intelligent Router (arXiv:2408.13510)

> "We find two replicas under round-robin can have memory usage difference up to 2.64×." — SkyWalker (EUROSYS 2026)

> "TetriInfer addresses this issue by adopting power-of-two load balancing [Mitzenmacher 2001]." — Survey of LLM Inference Systems (arXiv:2506.21901)

> "This isn't a rare event in production — it's the default behavior of any distributed deployment with a stateless load balancer." — llm-d team

> "The KV-cache hit rate is the single most important metric for a production-stage AI agent. It directly affects both latency and cost." — Manus (Context Engineering for AI Agents), quoted in llm-d blog

---

## Appendix: What This Document Leaves Out

### Left out: Preble's cross-GPU KV transfer mechanism

Preble's prefix autoscaling (§6.5) replicates KV entries between GPUs when a prefix becomes a hotspot, but the actual transfer protocol is not deeply covered in the paper. SkyWalker (§8) extends this to selective cross-region KV block migration — a separate system described at a higher level. The mechanism for efficient, low-latency KV state transfer between GPU processes is an open engineering problem addressed in different ways by DistServe (NVLink), Mooncake (network-attached KV cache), and SkyWalker (selective block push).

### Left out: Disaggregated runtimes (DistServe, TetriInfer, SplitWise, Mooncake)

The 2025 survey (§2's source L4/04) covers a family of systems that disaggregate prefill and decode computation at the cluster level — separate hardware pools for each phase, with KV state streamed between them. These are distinct from the instance routing problem: they change the architecture of the serving system, not just the routing layer above it. SkyWalker and Preble touch on this, but the full disaggregated runtime design is a separate topic.

### Left out: RL-based workload-aware routing

The IBM study (§3) shows that a trained RL router achieves 11% lower latency than round-robin by predicting output length and modelling mixing effects. This improvement is smaller than prefix-cache-aware routing gains (57× TTFT gap from llm-d), and requires training on workload-specific data. The practical deployment tradeoff (training cost vs. routing quality gain) is not covered here.

### Left out: Semantic caching

Portkey and LiteLLM implement semantic caching — deduplicating semantically equivalent requests that are textually different. This operates above the routing layer and is a model-provider gateway concern (not instance routing). It can interact with prefix caching (semantically equivalent requests share a prefix), but the mechanism is different: vector similarity search over request embeddings, not token-level prefix matching.

### Left out: Heterogeneous GPU clusters

All evaluation in Preble and SkyWalker uses same-generation GPUs within a cluster. Mixed GPU types (A100 + H100) would require updated cost regression functions per GPU type for E2's load cost formula. The routing policy structure (three tiers, two guards) would remain the same; only the cost coefficients would change.

### Left out: Prefill/Decode disaggregation at the routing level

SGLang Router's `--pd-disaggregation` mode (§7.4) separates the inference pipeline into two separate worker pools — one for prefill, one for decode — and the router orchestrates the handoff. This is a distinct architectural mode from the three-policy routing problem: it changes what a "worker" is and introduces state transfer between the two pools. It is the natural extension of single-phase instance routing but requires separate treatment.
