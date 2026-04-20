# 05 — evict and LRU Ordering

The `RadixCache` can grow without bound if pages are only ever added and never removed. A long-running server accumulates cached prefixes for every unique conversation history it has ever processed. Eventually the pool runs out of free pages — new requests cannot be prefilled. `evict` reclaims pages from the tree by removing the least-recently-used evictable leaves until enough pages are free.

---

## The evict Call Site

```python
# model_runner.py — prefill_batch, before compute_write_info
pages_needed  = sum(math.ceil(req.extend_input_len / P) for req in reqs)
available     = self.kv_pool.free_pages_count()
if available < pages_needed and self.radix_cache is not None:
    freed = self.radix_cache.evict(pages_needed - available)
    if freed < pages_needed - available:
        raise RuntimeError("OOM: cannot free enough pages even after eviction")
```

The eviction guard runs before page allocation, so the pool always has enough space when `compute_write_info` calls `kv_pool.alloc`. If eviction cannot free enough pages — all remaining tree nodes are locked by active requests — a hard OOM is raised. In a correctly configured server (where `kv_memory_fraction` leaves enough headroom for the peak working set), this should not occur in practice.

---

## evict: Heap-Based LRU

```python
# radix_cache.py — RadixCache.evict
def evict(self, n_pages_needed: int) -> int:
    leaves = [n for n in self._iter_nodes() if n.evictable]
    heapq.heapify(leaves)   # min-heap: oldest last_access_time first

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

`_iter_nodes` collects all nodes in the tree (excluding the root). Only those satisfying `evictable = lock_ref == 0 and not self.children` are candidates. The `heapq` min-heap uses `TreeNode.__lt__` — which compares `last_access_time` — to put the oldest node at the top. Each `heappop` yields the node least recently accessed.

After evicting a leaf, its parent may become a leaf: if the parent now has no remaining children and `lock_ref == 0`, it satisfies `evictable`. Pushing the parent back onto the heap enables cascading eviction of entire stale subtrees. A chat history that has not been accessed in hours might be evicted root-to-leaf in a single `evict` call if all its descendant leaves were already evicted first.

The `if not node.evictable` check after popping is defensive: the heap is built from a snapshot of all evictable nodes at the moment `evict` is called. If a node's state changed between the heapify and the pop — for instance, if `inc_lock_ref` was called for a new request while eviction was in progress — the stale heap entry must be discarded. In the current single-threaded scheduler model this cannot happen; the guard is future-proofing.

---

## _delete_leaf

```python
# radix_cache.py — _delete_leaf (simplified)
def _delete_leaf(self, node: TreeNode) -> None:
    parent = node.parent
    del parent.children[self._child_key(node.key)]
    node.parent = None
```

Deletion removes the node from its parent's `children` dict. After this, the node is unreachable from the tree and will be garbage-collected when no external reference to it remains. `kv_pool.free(node.value)` has already returned the pool pages; the node itself holds no pool resources after the free.

---

## Why LRU Is the Right Policy

LRU maximizes cache hit rate under the assumption that recently accessed prefixes are more likely to be accessed again in the near future. This holds for conversational workloads: users continue conversations, so a just-used conversation history is very likely to be needed again in the next request. A system prompt used 100 times in the last minute is much more valuable to keep than one used once three hours ago.

Alternative policies — MRU (most recently used, evict newest first), FIFO, or random — would perform poorly on real workloads. MRU would evict active conversations; FIFO would evict system prompts that haven't changed but are accessed constantly.

The `last_access_time` is updated during `_match_helper` and `_insert_helper` on every traversal, not only on exact matches. This means even a partial match (cache hit for 3 of 4 pages in an edge) updates the access time for the matching portion, correctly reflecting that the matching prefix was recently useful.

---

## Eviction and the Lock

`evict` never frees a locked node. A node is locked when an active request has called `inc_lock_ref` along the path from root to that node. The `evictable` property checks `lock_ref == 0` before `not self.children`, so any node with active references is skipped entirely during the initial `leaves` collection. This is the guarantee that makes prefix caching safe: a request's cached pages cannot be reclaimed while the request is alive, regardless of pool pressure.

Section 06 covers how `prefill_batch` integrates the prefix pages into the extend kernel's page index arrays.
