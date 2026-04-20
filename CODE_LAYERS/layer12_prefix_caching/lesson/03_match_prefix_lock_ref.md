# 03 — match_prefix and Lock Reference Counting

`match_prefix` is called once per new request, before its `fill_ids` is set. It traverses the tree to find the longest cached prefix of the request's token sequence, and returns the corresponding page indices alongside a reference to the deepest matching node. The caller — `PrefillAdder._apply_prefix_match` — immediately calls `inc_lock_ref` to protect those pages from eviction while the request is alive.

---

## match_prefix

```python
# radix_cache.py — RadixCache.match_prefix
def match_prefix(
    self, token_ids: List[int]
) -> Tuple[List[int], int, TreeNode]:
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

The `(len - 1) // P * P` cap is the single guard that prevents the tree from returning a match covering the entire prompt. A prompt of 512 tokens (exactly 32 pages at `page_size=16`) would have `max_match = (512-1)//16*16 = 31*16 = 496`. Even if the tree has all 512 tokens cached, only 496 tokens (31 pages) are returned. The 33rd token — `token_ids[496]` — is left uncached, guaranteeing at least one non-cached token for the forward pass.

`_match_helper` does the recursive tree traversal:

```python
# radix_cache.py — _match_helper (simplified)
def _match_helper(self, node: TreeNode, key: Tuple[int, ...]) -> Tuple[List[int], TreeNode]:
    if not key:
        return [], node

    child_key = self._child_key(key)   # key[:page_size]
    child = node.children.get(child_key)
    if child is None:
        return [], node

    n_match = self._count_match_pages(child.key, key)
    node.last_access_time = time.monotonic()   # update LRU
    child.last_access_time = time.monotonic()

    if n_match < len(child.key) // P:
        # Partial edge match — return what matched
        return list(child.value[:n_match]), child

    # Full edge match — continue deeper
    deeper_pages, deeper_node = self._match_helper(child, key[len(child.key):])
    return list(child.value) + deeper_pages, deeper_node
```

`last_access_time` is updated on every visited node during the traversal — both the parent and the matching child. This prevents recently-accessed cache entries from being evicted even if they haven't been inserted recently.

The return value `(page_indices, last_node)` has a crucial asymmetry: `page_indices` is a flat list of all pool page indices along the matched path, while `last_node` is the deepest node reached. For a partial edge match (matching 2 of 4 pages in an edge), `last_node` points to the child node even though only 2 of its value's pages are returned. This is used by `_split_node` during insert if the partial match triggers a split.

---

## inc_lock_ref

```python
# radix_cache.py — RadixCache.inc_lock_ref
def inc_lock_ref(self, node: TreeNode) -> None:
    while node is not self.root and node is not None:
        node.lock_ref += 1
        node = node.parent
```

After `match_prefix` returns, `_apply_prefix_match` immediately calls `inc_lock_ref(last_node)`. This walks from `last_node` up to (but not including) the root, incrementing `lock_ref` at each ancestor. The result is that every node whose pages are part of the matched prefix has `lock_ref > 0`, making `evictable` false for all of them.

Why must the entire ancestor chain be locked, not just `last_node`? Because `evict` prunes from leaves upward: after evicting a leaf, it checks if the parent became a leaf and may evict the parent too. If only the deepest node were locked, a parent node (whose value covers pages the request is attending over) could be evicted by a cascading eviction that started with a sibling leaf.

The lock is transitive along the path from root to `last_node`. Every node in that path holds pages that the request is attending over (its `value` is part of `page_indices`). If any of those pages were evicted, the request's KV data would be lost mid-decode.

---

## dec_lock_ref

```python
# radix_cache.py — RadixCache.dec_lock_ref
def dec_lock_ref(self, node: TreeNode) -> None:
    while node is not self.root and node is not None:
        if node.lock_ref > 0:
            node.lock_ref -= 1
        node = node.parent
```

`dec_lock_ref` is called in `cache_finished_req` after a request completes (or is cancelled). It mirrors `inc_lock_ref` exactly. After decrementing, nodes with `lock_ref == 0` and no children become eviction candidates immediately — no deferred cleanup, no GC cycle, no memory fence. The next `evict` call will see them as available.

The `if node.lock_ref > 0` guard prevents underflow in edge cases where a node was already decremented by a concurrent request sharing the same path. In the current single-threaded scheduler model this cannot happen, but the guard is defensive.

---

## Why lock_ref on the Request

The request object carries `req.last_node` — a direct reference to the deepest matching tree node. This is what `inc_lock_ref` receives at prefill time and what `dec_lock_ref` receives at completion time. Storing the node reference directly avoids re-traversing the tree on completion.

`_apply_prefix_match` guards against double-matching by checking `req.last_node is not None`:

```python
# scheduler.py — PrefillAdder._apply_prefix_match
if self.radix_cache is None or req.prefix_len > 0 or req.last_node is not None:
    return   # already matched or no cache
```

For a chunked request, `_apply_prefix_match` is called only on the first chunk — when `req.prefix_len == 0` and `req.last_node is None`. Subsequent chunks skip the match because `req.last_node is not None` after the first chunk. The lock established at the first chunk persists until `cache_finished_req` is called at the very end.

Section 04 explains the insert path and how `cache_finished_req` transitions pages from the request to the tree while releasing the lock.
