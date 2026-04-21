# Preble: Efficient Distributed Prompt Scheduling for LLM Serving

**Source paper:** Srivatsa et al., ICLR 2025 — https://arxiv.org/abs/2407.00023  
**Full paper:** `references/SURVEY/03_preble_iclr2025_full.md`  
**Level:** L3 — Mechanism level (pseudocode, invariants, design decisions)  
**Why here:** Preble's E2 algorithm is the direct theoretical parent of Layer 15's `PrefixCacheAwarePolicy`. Every parameter in `config.yml` (`cache_threshold`, `balance_abs_threshold`) traces back to a decision made in this paper.

---

## The Short Version

SGLang and vLLM already cache repeated prompt prefixes on a single GPU — if you send the same 10,000-token document twice, the second request reuses the stored computation and responds 7× faster. This works beautifully on one GPU.

The moment you add a second GPU, it breaks.

A standard load balancer scatters requests across both GPUs. The second GPU has never seen your document. The cache hit on GPU-1 is worthless — GPU-2 redoes the full 10,000-token computation from scratch. The more GPUs you add, the worse it gets.

**Preble fixes this.** It makes the router aware of which GPU has which prefix cached, and sends requests to the GPU that already holds the relevant context — while making sure no single GPU gets overwhelmed.

The result: **1.5× to 14.5× lower latency** on real workloads, on the same hardware, with no model changes.

---

## 1. Why Prompts Are No Longer Simple Questions

When LLMs were first deployed, prompts were simple: "What is the capital of France?" The answer is short and the question is unique.

Modern LLM usage looks nothing like this. Preble studied five real-world workloads and found:

| Workload | Avg prompt length | Avg output length | Prompt/output ratio | Shared prefix % |
|---|---|---|---|---|
| Tool use (Toolbench) | 1,835 tokens | 43 tokens | **43×** | **85%** |
| Embodied agent | 2,285 tokens | 16 tokens | **143×** | **97%** |
| Program generation | 3,871 tokens | 190 tokens | **20×** | **97%** |
| Video QA | 9,865 tokens | 4 tokens | **2,466×** | **88%** |
| Long document QA (LooGLE) | 23,474 tokens | 16 tokens | **1,467×** | **91%** |

Three things stand out:

**Prompts massively outnumber outputs.** In video QA, the model tokenizes the entire video clip (9,865 tokens) to answer a 4-token multiple-choice question. In document QA, a 23,000-token document is sent with a one-sentence question. The prefill phase — processing the input — dominates total request time.

**Most prompt tokens are shared across requests.** In tool-use workloads, every query that uses the same tool shares the same system prompt and tool instructions (1,800+ tokens). In document QA, multiple users asking questions about the same document share the entire document prefix. 85–97% of tokens in a typical prompt are shared with at least one other request.

**One key prefix is shared by many requests.** Each request has a "key portion" — the deepest node in its prefix tree that has more tokens than all its predecessors combined. This key portion is shared by 8.6 to 126 requests on average. It is the most valuable thing to cache.

**Implication:** The performance bottleneck for modern LLM serving is not generating output — it is recomputing the same enormous input over and over. Fix that, and you win.

---

## 2. How Single-GPU Prefix Caching Works (and Why It Breaks at Scale)

### Inside a single GPU: RadixAttention

SGLang implements **RadixAttention**: every time the GPU finishes processing a request, it stores the computed Key-Value (KV) tensors in a radix tree indexed by the token sequence. When the next request arrives:

1. Walk the radix tree, matching the new request's tokens from the beginning.
2. If the first N tokens match an existing node, reuse that node's KV tensors.
3. Only compute from token N+1 onward (the "missed" suffix).

**Result:** A 10,000-token document that was processed before costs only the length of the new question (e.g., 50 tokens) the second time. TTFT drops from 4.3 seconds to 0.6 seconds. 7× faster, same GPU, same model, no changes.

### The scale-out problem

Add a second GPU. Use round-robin load balancing. Now:

- Request 1 ("question A about document X") → GPU-1. GPU-1 computes and caches 10,000 tokens.
- Request 2 ("question B about document X") → GPU-2. GPU-2 has never seen document X. **Full 10,000-token recompute.**

The cache on GPU-1 is completely wasted. Every GPU independently re-does the same expensive prefill computation. Worse, as traffic grows, GPU-1's cache fills up. The eviction policy removes document X from GPU-1 to make room for new content. Now if question C about document X arrives on GPU-1, GPU-1 also has to recompute.

This is called **cache thrashing**: the system constantly evicts and recomputes the same content across different GPUs.

Preble's solution: make the router remember which GPU holds which prefix, and use that information when routing.

---

## 3. The E2 Algorithm: Exploitation + Exploration

E2 is the heart of Preble. The name captures the fundamental tension:

- **Exploit:** send the request to the GPU that already has its prefix cached. Maximum cache reuse, minimum recomputation.
- **Explore:** send the request to any GPU that needs work. Maximum load balance, potential cache miss.

Doing pure exploitation causes hotspots — one GPU handles all requests for a popular tool, while others sit idle. Doing pure exploration destroys cache locality — every request recomputes from scratch.

E2 chooses between them per request, per GPU, per moment.

### The decision rule

```
function schedule(request R):
    match R's prompt against the global radix tree
    
    cached_len = number of tokens matched (already cached somewhere)
    missed_len = total prompt length - cached_len
    
    if cached_len > missed_len:
        # Cache savings > new computation cost → EXPLOIT
        candidates = all GPUs that hold the longest matching node
        return candidate with lowest load_cost(GPU, R)
    else:
        # New computation > cache savings → EXPLORE
        if any GPU has decode_ratio >> others:
            return that GPU   # prefill-decode balancing (see §6)
        return GPU with lowest load_cost(GPU, R)
```

The key insight: **only exploit when the saved recomputation outweighs the cost of concentrating load**. If a request has 1,000 shared tokens and 2,000 new tokens, exploiting the cache saves 1,000 tokens of prefill but forces 2,000 tokens of new computation onto a potentially already-loaded GPU. Better to spread the load and eat the recomputation cost.

If a request has 2,000 shared tokens and 50 new tokens (like a follow-up question about a long document), the cache savings are enormous relative to the new computation. Exploit.

### The load cost function

When E2 needs to compare GPUs (whether exploiting or exploring), it uses a unified three-part cost function that measures "total GPU computation time if we assign this request here":

```
load_cost(GPU_i, request_R) = L_i + M_i + P_i
```

**L_i — existing load on GPU_i (history window H, default: 3 minutes):**

For each request that has run on GPU_i in the past H minutes:
- `prefill_time(request)` = computed from how many tokens *weren't* cached (proportional to non-shared token count)
- `decode_time(request)` = estimated from average output length observed on this GPU in window H

```
L_i = sum over all requests r in window H:
          prefillTime(missed_tokens_of_r) + decodeTime(avg_output_length)
```

Note: the paper profiles that prefill time and decode time are both approximately linear in token count. This means the router can estimate them using just token counts — no actual GPU profiling needed at routing time.

**M_i — eviction cost (what we'd lose by running R here):**

GPUs are always at full memory capacity. To run a new request, the GPU must evict some cached KV entries. Evicting popular cache entries is expensive — many future requests will have to recompute them.

For each cache entry that would be evicted:

```
M_i = sum over all entries j that would be evicted:
          prefillTime(length_of_j) × hit_rate_of_j
```

where `hit_rate_of_j` = (requests that used entry j in window H) / (total requests on GPU_i in window H).

A cache entry shared by 100 requests has a high hit rate — evicting it is very expensive. A cache entry used by only 1 request has a low hit rate — cheap to evict.

**P_i — cost of running the new request itself:**

```
P_i = prefillTime(missed_len_of_R)
```

This is just the prefill time for the tokens of R that aren't already cached on GPU_i. Decode time is not included because it's the same regardless of which GPU handles the request (same model, same output length).

**Total:**

```
load_cost(GPU_i, R) = L_i + M_i + P_i
```

Pick the GPU with the lowest total. This balances: current load, the value of what we'd evict, and the cost of running the new request — all measured in the same units (GPU compute time).

---

## 4. The Global Radix Tree

The global scheduler maintains a **global radix tree** — a prefix tree tracking every prompt sequence that has been processed across the entire cluster.

### Structure

A radix tree compresses sequences by merging common prefixes. Instead of storing each token sequence separately, each tree edge represents a sequence of tokens. Two requests that share the first 1,800 tokens share the same path down to the same node, then diverge into separate child nodes from that point.

For each tree node, the scheduler records:
1. **Number of tokens** in this node's sequence
2. **Which GPUs** have this node's KV entries cached
3. **Per-GPU request count** — how many requests shared this node in the past H minutes

### Insertion

When a new request arrives:
1. Walk the tree from root, matching tokens left to right.
2. If all tokens match an existing path → this request is a full prefix hit (rare but wonderful).
3. If tokens match partway and then diverge → split the existing node at the divergence point. Insert new child.
4. If no tokens match → create a new root node (a completely new prefix family).

### Eviction

When a GPU evicts a cached KV entry (because it needs memory for a new request), it notifies the global scheduler. The scheduler removes that GPU from the node's "caching GPU" set. If no GPU caches a node and no requests in window H have used it, the node is removed from the tree.

### Why this works

The global radix tree gives the scheduler a complete, up-to-date view of what is cached where. When a new request arrives, the scheduler can instantly identify which GPU holds the longest matching prefix — without querying any GPU directly. The tree is the index.

---

## 5. Post-Assignment: Handling Load Changes Over Time

E2's per-request decision is greedy — it makes the best choice given current information. But popular prefixes evolve: a tool that was used by 5 requests/second might suddenly surge to 50 requests/second. Preble handles this with two mechanisms after assignment.

### Mechanism 1: Load rebalancing (shift load)

The scheduler continuously monitors per-GPU load. If the most-loaded GPU's load is more than `Thbal` times higher than the least-loaded GPU:

```
if max_GPU_load > Thbal × min_GPU_load:
    redirect future requests (that would exploit heavy_GPU)
    → send them to light_GPU instead
    continue until load difference < Thbal
```

This is done without moving any cached data — the light GPU simply handles the request without the cache benefit. It accepts higher recomputation cost in exchange for load balance.

In Layer 15, `balance_abs_threshold` is the simplified version of `Thbal`: if the queue difference between the most-loaded and least-loaded worker exceeds this threshold, stop exploiting and route to the least-loaded worker.

### Mechanism 2: Prefix autoscaling (replicate the cache)

Sometimes load rebalancing isn't enough. If a prefix becomes so popular that its request load exceeds what a single GPU can handle — even after redirecting requests — Preble **replicates** the cached KV entries onto additional GPUs.

```
if avg_queue_time for prefix_subtree doubles over window H:
    select second GPU to hold this prefix
    copy KV entries → second GPU
    split future requests between both GPUs
```

Now two GPUs can handle the hot prefix with full cache hits. If demand grows further, a third GPU can be added.

**Layer 15 does not implement this.** When a prefix is hot, Layer 15's `balance_abs_threshold` guard simply routes some requests to less-loaded workers without cache — accepting recomputation. Autoscaling requires the router to coordinate KV cache state transfer between engines, which is a production-only capability.

---

## 6. Prefill/Decode Balancing

This is one of Preble's less obvious but important contributions.

**The problem:** Prefill and decode have fundamentally different GPU utilization profiles:
- Prefill is **compute-bound**: the GPU processes all input tokens in one parallel forward pass. Arithmetic units are maxed out.
- Decode is **memory-bandwidth-bound**: the GPU generates one token at a time, reading model weights from memory on each step. Arithmetic units are underused.

When one GPU is full of decode-phase requests (memory-bound) and another has spare capacity, Preble exploits this:

```
when exploring (no strong cache preference for this request):
    check decode_ratio for each GPU
    (decode_ratio = fraction of requests currently in decode phase)
    
    if GPU_i has high decode_ratio:
        → route R to GPU_i
        # R will trigger prefill (compute-bound)
        # This fills GPU_i's idle arithmetic units during decode
        # Both phases run simultaneously, better utilization
```

A request that needs full recomputation (a missed cache → prefill phase) is actually useful sent to a GPU that is decode-heavy. The prefill computation fills the arithmetic units that decode leaves idle.

Preble's insight: **a cache-miss request can be treated as a prefill-phase unit and deliberately mixed with decode-heavy GPUs to improve overall resource utilization.** Conversely, a full cache hit (only decode needed) is a decode-phase unit and should go to a compute-heavy GPU.

This is in addition to — not instead of — the load cost calculation. Preble prioritizes decode/prefill balancing, then falls back to load cost comparison if all GPUs have relatively balanced decode-prefill ratios.

---

## 7. The Local Scheduler: Fairness Inside a GPU

Each GPU also runs a **local scheduler** that decides the order of requests within that GPU's wait queue. This is separate from the global scheduler's routing decision.

### The fairness problem with prefix-first scheduling

A naive approach: always process the request with the highest cache hit ratio first. High cache hits = faster processing = more throughput.

Problem: a request that shares no prefix with anything already cached will wait forever. In a busy system with continuously arriving cached requests, the uncached request gets starved. This violates basic service fairness.

### Priority-based wait queue

Preble creates P priority groups (P is configurable). Each request is assigned to a priority group based on its cache hit ratio:

```
hit_ratio = (cached tokens) / (total prompt tokens)

priority_group = floor(hit_ratio × P)
# group 10 = 90-100% cache hit
# group 5 = 50-60% cache hit
# group 0 = 0-10% cache hit (mostly missed)
```

When forming the next batch, the scheduler proportionally selects from each group:

```
# Example: selecting 55 requests for a batch with P=10
group 10 → 10 requests
group 9  → 9 requests
group 8  → 8 requests
...
group 1  → 1 request
group 0  → 1 request (or fewer, weighted)
```

Higher-priority (higher-hit-ratio) groups get more requests per batch — they are faster to process and contribute more to throughput. But every group gets at least some representation per batch — no starvation.

**Effect:** Requests with full cache hits are processed quickly (improving average latency). Requests with zero cache hits are not starved (controlling tail latency and fairness in multi-tenant environments).

---

## 8. The Two-Level Architecture

Preble's complete architecture has two separate scheduling loops:

```
Incoming request
       ↓
  [Tokenizer pool]  ← parallel tokenization
       ↓
  [Global Scheduler]  ← runs on separate server (multi-server) or same server (single-server)
     │
     │  Maintains:
     │  - Global radix tree (which prefixes are cached, on which GPUs)
     │  - Per-GPU load counts (updated on each assignment and eviction)
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
     │  - Chunked prefill (break long prompts into chunks for smoother batching)
     │  - LRU eviction (evicts least recently used cache entries when memory is tight)
     │  - Reports evictions → Global Scheduler (background)
```

**Why two levels?**

- The global scheduler needs a cluster-wide view to make routing decisions. One centralized scheduler can see everything.
- The local scheduler needs sub-millisecond response times to manage individual iterations. A remote global scheduler would be too slow for iteration-level decisions.
- Decoupling them lets each optimize for its own timescale: global scheduler for request placement (seconds), local scheduler for batch formation (milliseconds).

**Scalability:** Preble's global scheduler processes 245–2,931 requests/second depending on radix tree complexity. The paper demonstrates it can sustain 70–391 concurrent A100 GPUs on one scheduler instance.

---

## 9. Implementation Details

### Lock-free global radix tree

The global radix tree is accessed by many concurrent request handlers. Preble makes this lock-free:
- Most operations (prefix matching, reading GPU assignments) are **read-only** — no locks needed.
- Only two operations are write operations: "assign a GPU to a tree node" and "increment request count". Both are expressed as **atomic instructions**.

This is what allows the global scheduler to handle hundreds of requests per second without becoming a bottleneck.

### Token counting as a proxy for compute time

E2's load cost formula requires prefill time and decode time estimates. Profiling actual GPU computation at routing time would be too expensive. Instead:

- Prefill time scales **linearly** with number of non-cached tokens (linear layer computation dominates)
- Decode time scales **linearly** with context length

The paper profiles this relationship offline once per GPU type, producing regression functions. At routing time, the scheduler uses only token counts — no GPU communication, no profiling overhead.

### Chunked prefill (local scheduler)

Long prompts processed all at once cause the GPU to be busy with prefill for hundreds of milliseconds, blocking all decode requests in the queue. The local scheduler **chunks** long non-shared prompts: splits them into smaller pieces and interleaves them with decode requests in each batch. A request needing 5,000 tokens of prefill might be split into 5 chunks of 1,000, with decode work filling the gaps between chunks.

---

## 10. Evaluation Results

### Setup

- **Models:** Mistral 7B, Llama-3 70B
- **Hardware:** 2× and 4× NVIDIA A6000 GPUs; 8× NVIDIA H100 GPUs
- **Baseline:** SGLang and vLLM with round-robin distribution (the standard way to scale out)
- **Workloads:** Toolbench (tool use), ALFWorld (embodied agents), APPS (programming), NExT-QA (video QA), LooGLE (document QA)

### End-to-end results

Preble vs. SGLang with round-robin, across all workloads and hardware configurations:

- **Average latency:** 1.5× to 14.5× improvement
- **p99 (tail) latency:** 2× to 10× improvement

The gains are largest on workloads with the highest prompt-to-output ratios (Video QA, tool use, embodied agents). The programming workload has longer outputs, so decode time is more significant — less room for Preble to improve. Even there, Preble achieves 1.56× to 1.8× average latency improvement.

### Ablation: which component contributes what?

The paper adds Preble features incrementally on the Toolbench workload:

| Configuration | Average latency | p99 latency |
|---|---|---|
| Round-robin (baseline) | 100% | 100% |
| + E2 per-request routing | ↓ significantly | ↓ significantly |
| + load rebalancing & autoscaling | ↓ further | ↓ further (especially p99) |
| + prefill/decode awareness | ↓ further | ↓ further |
| + priority wait queue (local) | same avg | ↓ further p99 only |

Key takeaways:
- **E2 routing** gives the biggest single improvement (smart routing beats all else)
- **Rebalancing + autoscaling** primarily improves tail latency (prevents hotspot accumulation)
- **Prefill/decode balancing** improves both (better GPU utilization through mixing)
- **Priority wait queue** is pure fairness — improves tail latency without hurting average

### Azure trace (real-world arrival patterns)

Real request arrival patterns are bursty — not the smooth Poisson distributions used in most ML systems papers. Preble was tested with the Azure LLM Inference Trace (mix of tool use + video QA):

- Average TTFT: **4.61× improvement** over SGLang
- TPOT (time per output token): **5.60× improvement**
- Average latency: **6.02× improvement**
- p99 latency: **2.42× improvement**

The improvements are even larger under real bursty traffic because round-robin load balancing degrades faster under bursts (requests pile up on one GPU while others are idle).

---

## 11. Connection to Layer 15

Layer 15's `PrefixCacheAwarePolicy` is a simplified, single-machine, teaching-scale implementation of Preble's E2 algorithm. Here is the mapping:

| Preble concept | Layer 15 implementation | What's simplified |
|---|---|---|
| Global radix tree | `RadixTrie` per `Worker` | Per-worker (not global); updated from request history, not live KV events |
| E2 exploit decision (`cached_len > missed_len`) | `cache_threshold` parameter | Binary threshold instead of dynamic ratio |
| E2 explore trigger (load imbalance) | `balance_abs_threshold` | Absolute queue difference instead of a multiplier |
| Load cost L_i (history window) | `worker.num_requests` (current queue depth) | Point-in-time count, not windowed computation time |
| Load cost M_i (eviction cost) | Not implemented | Trie size used as a proxy for "how much is cached here" |
| Load cost P_i (new request cost) | Implicit in prefix match ratio | Match ratio ≈ fraction of request already cached |
| Post-assignment rebalancing | `balance_abs_threshold` guard | Stateless per-request check only |
| Prefix autoscaling | Not implemented | Would require cross-engine KV transfer |
| Prefill/decode balancing | Not implemented | No decode-phase awareness in Layer 15 |
| Priority wait queue | Not implemented | FCFS within each engine |
| Local scheduler (per GPU) | Handled by SGLang/vLLM engine | Layer 15 only implements the global routing layer |

Despite these simplifications, Layer 15 captures the core insight: **match requests to the engine that already holds their prefix, but override with load balance when necessary.** Everything else in Preble is a production hardening of this core idea.

---

## 12. Key Quotes from the Paper

> "Current distributed LLM serving systems are not prompt-cache-aware; they attempt to distribute LLM computation load equally across GPUs to achieve high cluster-level GPU utilization. Yet, this distribution could result in requests with shared prefixes being sent to different GPUs, causing KV computation at all these GPUs that could otherwise be avoided."

> "A naive solution that always sends requests with shared prefixes to the same GPU would result in imbalanced loads and low overall GPU utilization because the GPU that initially serves a request with a popular prefix will accumulate a huge load of new requests all trying to reuse the calculated prefix KV."

> "E2 allows requests to exploit computed prompt prefixes on the same GPU but also gives chances for requests with shared prefixes to explore other GPUs. E2 chooses exploitation when the amount of recomputation saved is larger than that of new computation."

> "A request with its entire prompt shared and cached would only perform the decoding phase. Thus, it can be treated as a decoding-phase computing unit. Meanwhile, a request with a long prompt not cached and a short output length can be treated as a prefill-phase computing unit."

> "Optimizing prefill computation can largely improve overall application performance, and imbalanced prefill and decoding computation features should be considered in LLM serving."

---

## 13. What Preble Does Not Cover

**Cross-GPU KV transfer:** Preble's autoscaling replicates KV entries between GPUs, but the actual transfer mechanism is not deeply covered. SkyWalker (EUROSYS 2026) extends this to selective KV transfer across regions.

**Model quantization:** Preble assumes standard precision models. The interaction between prefix caching and quantization (e.g., FP8 KV cache) is not addressed.

**Heterogeneous GPU clusters:** All evaluation uses same-generation GPUs within a cluster. Mixed GPU types (e.g., A100 + H100) would require updated cost regression functions per GPU type.

**Network-attached KV caches:** Preble stores KV caches in GPU HBM. Newer systems (Mooncake, MemServe) explore disaggregated KV caches stored in CPU memory or across network — a different architecture that Preble's design does not address.

**Multi-modal inputs:** Video QA is studied, but the KV cache for tokenized video is treated the same as text. Multi-modal inputs with more complex prefix structures (e.g., mixed image-text) are future work.
