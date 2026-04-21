# 01 — The Eviction Problem: Why Discarding KV Tensors Is Wasteful

## What This Section Covers

Layer 12 built `RadixCache` — a prefix-aware GPU memory manager that reuses KV tensors across requests that share a prompt prefix. This is a significant optimisation for workloads where many requests share common text (a system prompt, a document, a few-shot example). But `RadixCache` has a hard limit: GPU VRAM. When the pool fills up, it must evict the least-recently-used nodes to make room. When it evicts a node, the KV tensors are freed — permanently.

This section explains why that permanent discard is the core problem HiCache solves, and what the actual cost looks like in production.

---

## Recap: What RadixCache Stores and Why It Runs Out

Each time a request is processed, the attention mechanism produces K (Key) and V (Value) tensors for every token in the prompt across every layer of the transformer. These are the KV cache. Reusing them avoids re-running the expensive transformer attention computation over those tokens for the next request.

`RadixCache` (Layer 12) organises these tensors as a prefix tree: each path from root to leaf corresponds to a sequence of tokens, and the nodes along the path hold the GPU memory indices where those tokens' KV tensors live. When a new request arrives, SGLang walks the tree, finds the longest matching prefix, and only runs the model over the unmatched tail — the prefix tokens already have their KV tensors in VRAM.

The pool has a finite size. For a 70B-parameter model on an 8×A100 setup with 80 GB each, the KV cache might consume 20–40 GB of VRAM — leaving limited headroom as concurrent requests accumulate. When the pool is full, `RadixCache.evict()` frees the memory held by the least-recently-used leaves.

```python
# Layer 12 RadixCache: eviction frees GPU memory permanently
def evict(self, num_tokens):
    # find LRU leaves, free their token indices from the GPU pool
    # the node is deleted from the tree
    # next request with the same prefix: full prefill required
```

---

## The Cost of Re-Prefill

"Full prefill" means running the entire transformer forward pass over the evicted tokens to rebuild their KV tensors. For a long context, this is not cheap:

| Context length | Typical prefill time (A100) |
|---|---|
| 1 k tokens | ~0.1–0.3 s |
| 4 k tokens | ~0.4–1.2 s |
| 16 k tokens | ~2–6 s |
| 64 k tokens | ~10–25 s |

These are rough figures that depend on model size, batch state, and GPU generation — but the scaling is clear: prefill time grows roughly quadratically with context length (due to attention), and linearly with the number of layers.

For a user in a multi-turn conversation with a 10 k-token history, an eviction means their next turn takes 4–10 seconds longer than it should. That latency is invisible in single-request benchmarks but dominates user-facing TTFT (time to first token) in production.

---

## The Access Pattern That Makes This Worse

The workloads where `RadixCache` gives the biggest benefit are precisely the workloads where eviction hurts the most:

**Multi-turn chat**: The same conversation history grows with every turn. After 20 turns, the shared prefix is 20 k+ tokens. If that prefix is evicted between turns, every subsequent turn pays full prefill cost.

**RAG (Retrieval-Augmented Generation)**: A system prompt plus a 10 k-token retrieved document is prepended to every user query. If 100 users hit the same document concurrently, the first miss evicts the document's KV cache from memory; subsequent users re-prefill the same 10 k tokens again.

**Batch document processing**: The same instruction prompt is prepended to hundreds of documents processed sequentially. A single eviction wastes all the saved prefill work accumulated so far.

The common thread is **temporal locality** — the same prefix will almost certainly be needed again within seconds or minutes. Discarding it permanently is the wrong policy for these workloads.

---

## What RadixCache Does Not Know

`RadixCache` tracks only GPU-resident pages. A node is either in the GPU pool (warm) or deleted (gone). There is no intermediate state. The tree has no concept of "evicted but potentially recoverable."

```
RadixCache node states (Layer 12):

  ┌─────────────────────┐
  │  WARM (GPU-resident) │  ← match_prefix() returns this node
  └─────────────────────┘
             │ evict()
             ▼
          (deleted)          ← node removed from tree, KV tensors freed
```

This means:
- The scheduler cannot distinguish "this prefix was recently computed and just evicted" from "this prefix has never been computed."
- Both cases result in a full prefill.
- There is no way to say "load this prefix back from somewhere cheaper before dispatching."

---

## The HiCache Answer: Write Before Evicting, Load Before Computing

HiCache changes the eviction contract. Instead of deleting an evicted node, it writes its KV pages to a lower-cost storage tier first, then marks the node as "cold" — still present in the tree, but with no GPU-resident pages. A future prefix match against a cold node triggers a load operation rather than a full prefill.

```
HiRadixCache node states (Layer 17):

  ┌─────────────────────┐
  │  GPU (warm)          │  ← match_prefix() returns immediately
  └─────────────────────┘
             │ evict_host() — KV pages written to CPU RAM
             ▼
  ┌─────────────────────┐
  │  CPU (cold)          │  ← match_prefix() triggers load_back()
  └─────────────────────┘
             │ evict_host() from CPU — pages written to storage backend
             ▼
  ┌─────────────────────┐
  │  STORAGE (frozen)    │  ← match_prefix() triggers storage.read()
  └─────────────────────┘
             │ evict from storage (capacity exceeded, or TTL)
             ▼
          (deleted)          ← only now is the data truly gone
```

The load cost is much lower than re-prefill:
- CPU → GPU load: ~10–100 ms (PCIe bandwidth bound, not compute bound)
- Storage → CPU → GPU: ~50–500 ms (NVMe or RDMA bound)
- Re-prefill: 1–25 seconds (compute bound, scales with context length)

For workloads with temporal locality, even a 200 ms storage load is an order-of-magnitude improvement over a 5-second re-prefill.

---

## What This Section Does Not Cover

- **How the three tiers are structured** — that is Section 02.
- **How the CPU pool is implemented** (pinned memory, DMA transfers) — Section 03.
- **What storage backends are available** — Section 04.
- **How to measure whether HiCache is helping** — Section 05.
- **How to enable HiCache** — Section 06.

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/mem_cache/radix_cache.py` | Layer 12 `RadixCache.evict()` — the baseline being extended |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:835` | `HiRadixCache.evict()` — how eviction becomes a write operation |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:905` | `HiRadixCache.evict_host()` — CPU pool eviction to storage |
| `REPOS/sglang/python/sglang/srt/mem_cache/hiradix_cache.py:940` | `HiRadixCache.load_back()` — the load path triggered on a cold hit |
