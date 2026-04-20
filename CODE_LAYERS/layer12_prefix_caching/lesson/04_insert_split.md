# 04 — insert and _split_node

When a request finishes, its accumulated K/V pages should be made available to future requests that share its token sequence. `cache_finished_req` orchestrates this: it inserts the finished request's sequence into the tree, handles any overlap with already-cached pages, releases the lock acquired at prefill, and returns the pool rows to their respective free lists.

---

## cache_finished_req

```python
# radix_cache.py — cache_finished_req
def cache_finished_req(self, req, req_to_token_pool, kv_pool) -> None:
    P              = self.page_size
    token_ids      = req.input_ids + req.output_ids
    n_prefix_pages = len(req.prefix_page_indices)
    all_pages      = list(req.slot_indices)

    aligned_len        = (len(token_ids) // P) * P
    aligned_page_count = aligned_len // P

    # Insert into tree
    if aligned_page_count > 0:
        n_overlap = self.insert(token_ids, all_pages)
    else:
        n_overlap = 0

    # Free pages that were redundantly computed
    if n_overlap > n_prefix_pages:
        kv_pool.free(all_pages[n_prefix_pages:n_overlap])

    # Free the unaligned tail page (partial last page)
    if aligned_page_count < len(all_pages):
        kv_pool.free(all_pages[aligned_page_count:])

    # Release lock on prefix nodes
    if req.last_node is not None:
        self.dec_lock_ref(req.last_node)

    # Return row in ReqToTokenPool
    if req.req_pool_idx is not None:
        req_to_token_pool.free(req.req_pool_idx)
```

The full token sequence is `input_ids + output_ids` — caching both the prompt and the generated response allows future requests that share the same conversation history to skip recomputing both. `aligned_len` rounds down to a page boundary; the last partial page (if any) cannot be cached because it may not be full.

`n_overlap` is the number of pages the tree already held for this prefix before the `insert` call. Pages `[0 : n_prefix_pages]` were lent from the tree (prefix match) — the request did not compute them. Pages `[n_prefix_pages : n_overlap]` were computed by this request but were also already in the tree — redundant writes by a concurrent request with the same sequence. These redundant pages must be freed. Pages `[n_overlap : aligned_page_count]` are new — they are now owned by the tree after `insert`.

---

## insert

```python
# radix_cache.py — RadixCache.insert
def insert(self, token_ids: List[int], page_indices: List[int]) -> int:
    P           = self.page_size
    aligned_len = (len(token_ids) // P) * P
    key         = tuple(token_ids[:aligned_len])
    value       = list(page_indices[:aligned_len // P])

    if not key:
        return 0
    return self._insert_helper(self.root, key, value)
```

`_insert_helper` recurses from the root, following existing edges as far as they match. There are three cases:

**Case 1 — No child with this first-page key:** Create a new leaf node with the full key and value, attach it to the current node, return 0 overlap.

**Case 2 — Full edge match:** The existing edge's key is a prefix of the new key. Recurse into the matching child with the suffix `key[len(child.key):]` and `value[n_child_pages:]`. The overlapping pages are those already in the child's value: `n_overlap = n_child_pages + recursive_result`.

**Case 3 — Partial edge match:** The new key shares `n_match` pages with the existing edge but diverges before the edge ends. Call `_split_node(child, n_match)` to divide the edge, then insert the new suffix under the split node.

---

## _split_node

```python
# radix_cache.py — _split_node
def _split_node(self, child: TreeNode, n_pages: int) -> TreeNode:
    P = self.page_size
    new_node = TreeNode()
    new_node.parent           = child.parent
    new_node.key              = child.key[:n_pages * P]
    new_node.value            = child.value[:n_pages]
    new_node.lock_ref         = child.lock_ref
    new_node.last_access_time = child.last_access_time

    # Reattach: parent → new_node → child
    child_key_in_parent = self._child_key(child.key)
    child.parent.children[child_key_in_parent] = new_node
    new_node.children[self._child_key(child.key[n_pages * P:])] = child

    # Trim child to the suffix
    child.key    = child.key[n_pages * P:]
    child.value  = child.value[n_pages:]
    child.parent = new_node

    return new_node
```

`_split_node` inserts a new intermediate node between `child.parent` and `child`. The new node takes the first `n_pages` pages of the edge (the matching prefix); `child` retains the remaining suffix. The new node's `lock_ref` and `last_access_time` are copied from `child`.

Why copy `lock_ref`? If `child` was locked (an active request is attending over its pages), the new prefix node holds those same pages in its value. Failing to copy `lock_ref` would make the prefix node appear evictable, and a concurrent eviction could free the very pages the locked request is attending over.

After `_split_node`, the caller inserts the new sequence as a sibling of the trimmed `child` under the new prefix node. The tree structure correctly represents the shared prefix and the two diverging suffixes.

---

## The Overlap Case in Practice

Two concurrent requests R1 and R2 have identical prompts of 512 tokens. Both arrive before either has been prefilled.

R1 is prefilled first. On R1's first `match_prefix` call, the tree is empty — no match. R1 is prefilled with all 512 tokens. When R1 finishes, `cache_finished_req(R1)` calls `insert(R1.token_ids, R1.slot_indices)`. The tree has no entry yet, so `n_overlap = 0`. The 32 pages are now owned by the tree.

R2 arrives and calls `match_prefix`. It finds the 32-page entry (for 512 tokens, minus the last-token guard which leaves 31 pages = 496 tokens cached). `kv_committed_len = 496`, `fill_ids = input_ids[496:]`. Only the last 16 tokens plus the unique suffix need to be prefilled.

But what if R2 were prefilled concurrently before R1 finished? Then R2's `match_prefix` returns empty (tree was empty at the time), both are prefilled independently, both call `insert`. The second `insert` call finds all `aligned_page_count` pages already in the tree: `n_overlap = aligned_page_count`. `all_pages[n_prefix_pages:n_overlap]` covers all of R2's freshly computed pages — they are freed immediately, returning the pool pages from R2's redundant computation.

The overlap mechanism ensures that even without perfect scheduling, the pool's page count converges to the unique content rather than accumulating duplicates.

Section 05 explains how `evict` handles the case where the pool is under pressure and some cached pages must be sacrificed.
