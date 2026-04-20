# Layer 12 — Summary

Layer 12 adds a `RadixCache` that stores finished requests' K/V page indices in a compressed radix tree keyed on token sequences, so that subsequent requests sharing a page-aligned prefix can skip recomputing those tokens entirely. The extend kernel, the decode kernel, `ForwardBatch`, and all model files are unchanged; the only caller-visible difference is that `kv_committed_len` is set to `prefix_len` before `prefill_batch` runs, and `cache_finished_req` replaces the direct `kv_pool.free` call on request completion.

---

## From Layer 11 to Layer 12

In Layer 11, every request computed its own K/V from `input_ids[0]`, regardless of how much it shared with prior requests:

```python
# Layer 11 — PrefillAdder.build(), new request path (simplified)
req.fill_ids         = req.input_ids          # always the full prompt
req.extend_input_len = req.prompt_len
req.kv_committed_len = 0                      # always start from zero
```

In Layer 12, `PrefillAdder._apply_prefix_match` runs before `fill_ids` is set:

```python
# Layer 12 — PrefillAdder._apply_prefix_match
page_indices, matched_len, last_node = self.radix_cache.match_prefix(req.input_ids)

req.prefix_page_indices = page_indices
req.prefix_len          = matched_len
req.last_node           = last_node
req.kv_committed_len    = matched_len           # skip cached tokens
self.radix_cache.inc_lock_ref(last_node)        # protect from eviction
```

If `matched_len = 512`, the 512-token prefix's K/V pages are already in the pool, owned by the tree. `fill_ids` is set to `req.input_ids[512:]` and the extend kernel processes only the unique suffix. Position IDs for the suffix start at 512, not at zero, so rotary embeddings remain correct.

---

## The Radix Tree Structure

`RadixCache` wraps a compressed radix tree whose nodes hold page-aligned token sequences and their corresponding pool page indices:

```python
class TreeNode:
    __slots__ = ("children", "parent", "key", "value", "lock_ref", "last_access_time")

    def __init__(self) -> None:
        self.children:         Dict[Tuple[int, ...], "TreeNode"] = {}
        self.parent:           Optional["TreeNode"]               = None
        self.key:              Tuple[int, ...]                    = ()
        self.value:            List[int]                          = []
        self.lock_ref:         int                               = 0
        self.last_access_time: float                             = time.monotonic()

    @property
    def evictable(self) -> bool:
        return self.lock_ref == 0 and not self.children
```

`key` holds the full token sequence for one tree edge, always a multiple of `page_size` tokens long. `value` holds the corresponding pool page indices. Children are indexed by the first-page token tuple of their edge — the `_child_key` method takes the first `page_size` tokens of a key and uses that as the dict lookup. This gives `O(P)` space per cached page rather than one node per token, and allows partial matches to be resolved by comparing full-page chunks rather than individual tokens.

The root node is always present and holds no key or value. All real prefix data lives in its descendants.

---

## match_prefix and Lock Reference Counting

`match_prefix` traverses the tree from root, following `_child_key` lookups and comparing full pages at each edge:

```python
def match_prefix(self, token_ids: List[int]) -> Tuple[List[int], int, TreeNode]:
    P = self.page_size
    # Never match the last token — we need at least one non-cached token
    # so that the forward pass produces a logit to sample from.
    max_match = ((len(token_ids) - 1) // P) * P
    if max_match <= 0:
        return [], 0, self.root

    key = tuple(token_ids[:max_match])
    page_indices, last_node = self._match_helper(self.root, key)
    matched_len = len(page_indices) * P
    return page_indices, matched_len, last_node
```

The `(len - 1) // P * P` cap ensures that at least one token is always left unmatched. This is required because the extend kernel must produce at least one output logit to sample the first token from. A prompt whose length is an exact multiple of `page_size` therefore has its last page withheld from the match, even if the tree contains it.

Immediately after `match_prefix`, `inc_lock_ref(last_node)` walks from `last_node` to the root and increments `lock_ref` at each ancestor. As long as the request is active, every node along that path has `lock_ref > 0` and therefore `evictable` is false. When the request finishes (or is evicted from the scheduler), `dec_lock_ref(req.last_node)` reverses this, potentially making those nodes eviction-eligible for the first time.

---

## insert and _split_node

When a request finishes, `cache_finished_req` inserts its full token sequence and page list into the tree:

```python
def insert(self, token_ids: List[int], page_indices: List[int]) -> int:
    P           = self.page_size
    aligned_len = (len(token_ids) // P) * P
    key         = tuple(token_ids[:aligned_len])
    value       = list(page_indices[:aligned_len // P])

    if not key:
        return 0
    return self._insert_helper(self.root, key, value)
```

`insert` returns `n_overlap_pages` — the number of pages the tree already held for this prefix before the insert. When two concurrent requests have identical prompts, both compute the same K/V independently. The second request to finish discovers `n_overlap > n_prefix_pages` and frees the redundant pages:

```python
if n_overlap > n_prefix_pages:
    kv_pool.free(all_pages[n_prefix_pages:n_overlap])
```

When a new prefix partially matches an existing edge, `_split_node` divides the edge into two:

```python
def _split_node(self, child: TreeNode, n_pages: int) -> TreeNode:
    # Creates new_node holding child.key[:n_pages*P] and child.value[:n_pages]
    # child becomes child.key[n_pages*P:] and child.value[n_pages:]
    # new_node inherits child's lock_ref and last_access_time
    new_node.lock_ref         = child.lock_ref
    new_node.last_access_time = child.last_access_time
```

Inheriting `lock_ref` is critical: if `child` was locked by an active request, the new prefix node must also be locked so it cannot be evicted while that request's decode step is running.

---

## evict and LRU Ordering

When `prefill_batch` detects that the pool is close to full, it calls `radix_cache.evict(n_pages_needed)`:

```python
def evict(self, n_pages_needed: int) -> int:
    leaves = [n for n in self._iter_nodes() if n.evictable]
    heapq.heapify(leaves)   # min-heap by last_access_time (oldest first)

    freed = 0
    while freed < n_pages_needed and leaves:
        node = heapq.heappop(leaves)
        if not node.evictable:
            continue
        self.kv_pool.free(node.value)
        freed += len(node.value)
        parent = node.parent
        self._delete_leaf(node)
        if parent is not self.root and parent is not None and parent.evictable:
            heapq.heappush(leaves, parent)

    return freed
```

`TreeNode.__lt__` compares `last_access_time`, so the heap orders nodes from oldest to most recently accessed. After deleting a leaf, its parent may become a leaf itself (no remaining children, `lock_ref == 0`). Pushing the parent back into the heap allows cascading eviction of entire stale subtrees without a second pass. The `not node.evictable` check after popping handles the case where a node's state changed between `heapify` and the pop — for example, if a concurrent `inc_lock_ref` incremented its `lock_ref` before the eviction loop ran.

---

## Integration with prefill_batch

Layer 12 modifies `model_runner.prefill_batch` in two places. First, before `compute_write_info`, prefix pages are injected into `req_to_token_pool` for any request with a cache hit:

```python
for req in reqs:
    if req.prefix_page_indices:
        n_pfx = len(req.prefix_page_indices)
        self.req_to_token_pool.req_to_token[
            req.req_pool_idx, :n_pfx
        ] = torch.tensor(req.prefix_page_indices, dtype=torch.int32, device=DEVICE)
```

This makes the Triton `kv_indices` kernel read the correct page indices for the cached prefix portion from GPU memory — no CPU-side intermediate is needed. `compute_write_info` then sees `kv_committed_len = prefix_len` and begins allocating new pages starting from the suffix position, correctly handling the partial-page continuation case if `prefix_len % page_size != 0`.

Second, immediately after writing prefix pages, the eviction guard runs:

```python
pages_needed  = sum(math.ceil(req.extend_input_len / P) for req in reqs)
available     = self.kv_pool.free_pages_count()
if available < pages_needed and self.radix_cache is not None:
    freed = self.radix_cache.evict(pages_needed - available)
```

Eviction happens before page allocation so the extend kernel never stalls waiting for free pages.

---

## The Full Loop

Consider two requests: R1 with a 512-token system prompt followed by a 64-token user query, and R2 arriving shortly after with the same 512-token system prompt followed by a different 32-token query.

R1 arrives with an empty tree. `match_prefix` returns `([], 0, root)`. `kv_committed_len = 0`. `fill_ids = input_ids` (all 576 tokens). `prefill_batch` runs the full extend. After R1 finishes generating, `cache_finished_req` calls `insert(R1.input_ids + R1.output_ids, R1.slot_indices)`. The 512-token system prefix (32 pages at page_size=16) is now cached in the tree. The unaligned tail page (partial last page) is freed. `dec_lock_ref(R1.last_node)` is called — since R1 is the only user, all nodes now have `lock_ref = 0`.

R2 arrives. `match_prefix(R2.input_ids)` traverses the tree, matches all 512 system-prompt tokens, and returns the 32 cached pages with `matched_len = 512`. `inc_lock_ref(last_node)` protects those 32 nodes. `kv_committed_len = 512`. `fill_ids = input_ids[512:]` (32 tokens). `prefill_batch` injects the 32 prefix pages into `req_to_token`, then allocates only 2 new pages (32 tokens at page_size=16) for the suffix. The extend kernel processes 32 tokens instead of 544. The cached K/V is read directly from the pool via `kv_indices` — no recomputation, no copy. After R2 finishes, `cache_finished_req` inserts its new suffix pages and releases its lock.

---

## What Comes Next

Layer 12 eliminates redundant KV computation but leaves model weight memory unchanged. For Qwen3-1.7B, the 28-layer model with hidden_dim=2048, FFN intermediate 11008, and 16 attention heads stores roughly 3.4 GB in bfloat16 — the weights dwarf the KV pool at typical batch sizes. Layer 13 replaces every `nn.Linear` in the attention and FFN layers with `GPTQLinear`: a 4-bit quantized weight store (`qweight` packed 8 values per int32, `scales` and `qzeros` for group-wise dequantization) driven by the `gptq_gemm` fused CUDA kernel. Weights shrink to approximately 0.85 GB — a 4× reduction — with inference quality measured by downstream perplexity rather than throughput. The KV pool, `RadixCache`, `scheduler.py`, and `forward_batch.py` are unchanged; only `model_runner.py` gains a `use_gptq` flag and the `model_gptq/` directory is added alongside `model/`.
