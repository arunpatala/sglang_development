# 02 — The Radix Tree Structure

The `RadixCache` is the CPU-side index that maps token sequences to KV pool pages. It is a compressed radix tree: instead of one node per token, each edge represents one or more full pages (a multiple of `page_size` tokens). This compression is mandatory because the pool operates in page-sized units — a partial page cannot be cached independently, and the tree's granularity must match the pool's.

---

## TreeNode

```python
# radix_cache.py — TreeNode
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

    def __lt__(self, other: "TreeNode") -> bool:
        return self.last_access_time < other.last_access_time
```

`key` holds the token sequence for this tree edge as a tuple of integers. Its length is always a multiple of `page_size`: `len(key) % page_size == 0`. The sequence is the complete token run from the start of this edge to its end — not just the first token of the edge. This means an edge labeled `(t0, t1, ..., t15)` (16 tokens, one page) covers exactly one pool page. An edge labeled `(t0, ..., t31)` covers two pages.

`value` holds the list of pool page indices for this edge, in order. `len(value) == len(key) // page_size`. The invariant is that `value[i]` is the page index for `key[i*P : (i+1)*P]`.

`children` is a dict keyed by the first-page tuple of each child's edge. `_child_key(key)` extracts `key[:page_size]` and is used for both lookup and insertion. This means two edges can share the same first page only if their first `page_size` tokens are identical — which, for token sequences representing real prompts, means the edges diverge within the first page rather than at a page boundary.

`lock_ref` is a reference count. A node with `lock_ref > 0` is being used by one or more active requests. Its pages cannot be evicted. `lock_ref` counts are incremented on every ancestor from the matching node to the root (via `inc_lock_ref`), so the entire path from root to the cached prefix is protected while any request attending over those pages is alive.

`last_access_time` is updated whenever a node is visited by `_match_helper` or `_insert_helper`. This timestamp drives the LRU eviction order: older nodes (smaller `last_access_time`) are evicted first.

`__lt__` makes `TreeNode` orderable by access time, enabling Python's `heapq` to build a min-heap of evictable nodes with the least-recently-used node at the top.

---

## The Page Alignment Invariant

All keys and values in the tree are page-aligned: `len(key) % page_size == 0` and `len(value) == len(key) // page_size`. This invariant is enforced at every insertion point.

`match_prefix` caps the match at `((len(token_ids) - 1) // P) * P` tokens, discarding the last partial page even if it would match. This ensures at least one non-cached token remains in the request — the forward pass needs to produce at least one output logit to sample from.

`insert` truncates the token sequence to `(len(token_ids) // P) * P` before inserting. An incomplete last page is never cached.

`_split_node` always splits at a page boundary: `n_pages` is the number of full pages that match, and the split point is `n_pages * page_size` tokens into the edge key.

This alignment means the pool's page-level operations (alloc/free in page-sized units) and the tree's page-level operations (insert/evict in page-valued node.value lists) are always consistent. There is never a case where the tree holds a reference to a fractional page.

---

## Edge Key and Child Key

`_child_key(key)` returns `key[:page_size]` — the first-page tuple. Children are stored in `node.children` with this as the dict key. This design means:

A lookup for a new token sequence `tok_ids` at node `n` computes `child_key = tuple(tok_ids[:page_size])` and does `n.children.get(child_key)`. If a child exists, the full edge key is compared page by page via `_count_match_pages` to find the longest matching prefix.

A newly inserted sequence creates a child with `_child_key(new_key)` as the dict key. If a child with the same first-page key already exists, the existing edge must be compared — it might match fully (extend deeper), partially (split required), or not at all (impossible since the first-page key matched).

This "first-page dict, full-edge compare" approach keeps lookup O(depth × page_size) in the common case and handles partial matches cleanly via `_split_node`.

---

## The Root Node

The root node is a permanent sentinel: `self.root = TreeNode()`. It is never evicted, never inserted into any free list, and holds no key or value. All real prefix data lives in its descendants. `inc_lock_ref` and `dec_lock_ref` both stop at the root (checked via `node is not self.root`). `evict` and `_iter_nodes` skip the root.

The root's `children` dict is the entry point for all lookups. A server starting fresh has `root.children = {}` — every request's first `match_prefix` call returns `([], 0, root)` (no match). The first request to finish will call `insert` and populate `root.children` with the first cached prefix.

Section 03 covers the lookup path (`match_prefix`) and the lock mechanics (`inc_lock_ref`, `dec_lock_ref`) in detail.
