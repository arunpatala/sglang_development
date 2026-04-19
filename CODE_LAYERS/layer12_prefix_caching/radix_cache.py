"""
Layer 11 — RadixCache: prefix caching via a compressed radix tree.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Design overview
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  The tree maps token-ID sequences → page indices in the KV pool.

  TreeNode fields:
    key:   tuple of token IDs labelling the edge from parent to this node.
           Always a multiple of page_size tokens long.
    value: list of page indices, len == len(key) // page_size.
    lock_ref:  reference count; > 0 means an active request is using
               this node's KV — it must NOT be evicted.
    last_access_time: for LRU eviction.

  Child dict key: tuple of first page_size token IDs.
    With page_size=16, child_key = (tok0, tok1, ..., tok15).
    This ensures two sequences are differentiated at the first page
    that differs, even if they share tok0.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Page-granularity constraint
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Matching and insertion work in multiples of page_size.

  match_prefix([t0..t99], P=16):
    aligned_len = 96  (floor(100/16)*16)
    Walks tree, returns page indices for the longest matching prefix
    of t0..t95.  Returned matched_len is always a multiple of P.

  insert([t0..t99], pages):
    Inserts only the page-aligned prefix (t0..t95, 6 pages).
    Partial last page (t96..t99) is discarded — its K/V may be
    written to the KV pool but is never shared.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Page ownership rules
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • Pages in a TreeNode.value  → owned by the tree.  Never free.
  • Prefix pages lent to a Req → still owned by tree; protected by lock_ref.
  • Duplicate pages (insert returned overlap > matched from cache)
    → this request computed them redundantly; free immediately.
  • Unaligned tail page        → not inserted; free immediately.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comparison with SGLang RadixCache
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  SGLang:  stores per-token KV slot indices; supports page_size ≥ 1.
  Layer11: stores per-page indices; inherently page-granular.
           Simpler because our KVPool is already page-granular.
"""

from __future__ import annotations

import heapq
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

if TYPE_CHECKING:
    from kv_cache import KVPool


# ─────────────────────────────────────────────────────────────────────────────
# TreeNode
# ─────────────────────────────────────────────────────────────────────────────

class TreeNode:
    __slots__ = ("children", "parent", "key", "value", "lock_ref", "last_access_time")

    def __init__(self) -> None:
        # Dict key: tuple of first page_size token IDs (unique per page-page match)
        self.children:         Dict[Tuple[int, ...], "TreeNode"] = {}
        self.parent:           Optional["TreeNode"]              = None
        self.key:              Tuple[int, ...]                   = ()
        self.value:            List[int]                         = []   # page indices
        self.lock_ref:         int                               = 0
        self.last_access_time: float                             = time.monotonic()

    @property
    def evictable(self) -> bool:
        """A leaf with no active references can be LRU-evicted."""
        return self.lock_ref == 0 and not self.children

    def __lt__(self, other: "TreeNode") -> bool:
        return self.last_access_time < other.last_access_time


# ─────────────────────────────────────────────────────────────────────────────
# RadixCache
# ─────────────────────────────────────────────────────────────────────────────

class RadixCache:
    """Prefix cache backed by a compressed radix tree of KV page indices."""

    def __init__(self, kv_pool: "KVPool", page_size: int) -> None:
        self.kv_pool   = kv_pool
        self.page_size = page_size
        self.root      = TreeNode()

    # ── Public API ────────────────────────────────────────────────────────────

    def match_prefix(
        self, token_ids: List[int]
    ) -> Tuple[List[int], int, TreeNode]:
        """
        Find the longest cached prefix of token_ids.

        To guarantee at least one token goes through the model (so we can
        sample), we never match beyond floor((len-1)/P)*P tokens.

        Returns:
          page_indices: page indices for the matched prefix
          matched_len:  number of matched tokens (always a multiple of page_size)
          last_node:    deepest matching node (caller must call inc_lock_ref)
        """
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

    def insert(
        self,
        token_ids:    List[int],
        page_indices: List[int],
    ) -> int:
        """
        Insert the page-aligned prefix of the sequence into the tree.

        Returns n_overlap_pages: how many pages were already in the tree
        before this call (duplicate pages; caller should free them).
        """
        P           = self.page_size
        aligned_len = (len(token_ids) // P) * P
        key         = tuple(token_ids[:aligned_len])
        value       = list(page_indices[:aligned_len // P])

        if not key:
            return 0
        return self._insert_helper(self.root, key, value)

    def inc_lock_ref(self, node: TreeNode) -> None:
        """Walk from node to root, incrementing lock_ref at each ancestor."""
        while node is not self.root and node is not None:
            node.lock_ref += 1
            node = node.parent

    def dec_lock_ref(self, node: TreeNode) -> None:
        """Walk from node to root, decrementing lock_ref at each ancestor."""
        while node is not self.root and node is not None:
            if node.lock_ref > 0:
                node.lock_ref -= 1
            node = node.parent

    def evict(self, n_pages_needed: int) -> int:
        """
        Evict LRU-evictable leaves until >= n_pages_needed pages are freed.
        Pages are returned to kv_pool.  Returns actual pages freed.
        """
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
            # Parent may have become a leaf after we removed this child.
            if parent is not self.root and parent is not None and parent.evictable:
                heapq.heappush(leaves, parent)

        return freed

    def cache_finished_req(self, req: Any, req_to_token_pool: Any, kv_pool: "KVPool") -> None:
        """
        Insert a finished request's KV into the tree and release resources.

        Ownership after this call:
          prefix pages              → tree (unchanged; lock just released)
          newly-inserted pages      → tree (now cached for future requests)
          duplicate pages           → freed (computed redundantly)
          unaligned tail page       → freed
          req_pool_idx              → freed back to ReqToTokenPool
        """
        P              = self.page_size
        token_ids      = req.input_ids + req.output_ids
        n_prefix_pages = len(req.prefix_page_indices)
        all_pages      = list(req.slot_indices)

        aligned_len        = (len(token_ids) // P) * P
        aligned_page_count = aligned_len // P

        # Insert into tree and learn how much was already there.
        if aligned_page_count > 0 and aligned_len > 0:
            n_overlap = self.insert(token_ids, all_pages)
        else:
            n_overlap = 0

        # Free pages this request computed that overlap with already-cached data.
        if n_overlap > n_prefix_pages:
            kv_pool.free(all_pages[n_prefix_pages:n_overlap])

        # Free unaligned tail page(s) (partial last page, not inserted into tree).
        if aligned_page_count < len(all_pages):
            kv_pool.free(all_pages[aligned_page_count:])

        # Release lock so the prefix nodes become eligible for LRU eviction.
        if req.last_node is not None:
            self.dec_lock_ref(req.last_node)

        # Return row in ReqToTokenPool.
        if req.req_pool_idx is not None:
            req_to_token_pool.free(req.req_pool_idx)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _child_key(self, key: Tuple[int, ...]) -> Tuple[int, ...]:
        """First-page tuple used as dict key in node.children."""
        return key[:self.page_size]

    def _count_match_pages(
        self, key_a: Tuple[int, ...], key_b: Tuple[int, ...]
    ) -> int:
        """Return how many full pages match between key_a and key_b."""
        P       = self.page_size
        min_len = min(len(key_a), len(key_b))
        i       = 0
        while i + P <= min_len:
            if key_a[i:i + P] != key_b[i:i + P]:
                break
            i += P
        return i // P

    def _split_node(self, child: TreeNode, n_pages: int) -> TreeNode:
        """
        Split `child` after `n_pages` full pages.
        Returns the new prefix node (which becomes the parent of the old child).
        """
        P         = self.page_size
        split_tok = n_pages * P
        parent    = child.parent

        # The key used in parent.children to reach child (before split).
        old_child_key = self._child_key(child.key)

        # New node takes the prefix portion of child.
        new_node                  = TreeNode()
        new_node.parent           = parent
        new_node.key              = child.key[:split_tok]
        new_node.value            = child.value[:n_pages]
        new_node.lock_ref         = child.lock_ref
        new_node.last_access_time = child.last_access_time

        # child keeps the suffix portion.
        suffix_ck                         = self._child_key(child.key[split_tok:])
        new_node.children[suffix_ck]      = child
        child.parent                      = new_node
        child.key                         = child.key[split_tok:]
        child.value                       = child.value[n_pages:]

        # Repoint parent's reference from old child to new_node.
        parent.children[old_child_key] = new_node
        return new_node

    def _match_helper(
        self, node: TreeNode, key: Tuple[int, ...]
    ) -> Tuple[List[int], TreeNode]:
        P            = self.page_size
        t            = time.monotonic()
        node.last_access_time = t
        page_indices: List[int] = []

        while len(key) >= P:
            ck = self._child_key(key)
            if ck not in node.children:
                break
            child                  = node.children[ck]
            child.last_access_time = t
            n_pages_match          = self._count_match_pages(child.key, key)

            if n_pages_match == 0:
                break

            if n_pages_match < len(child.value):
                # Partial match → split the node at the matching boundary.
                new_node = self._split_node(child, n_pages_match)
                page_indices.extend(new_node.value)
                node = new_node
                break
            else:
                # Full match of this node — continue down.
                page_indices.extend(child.value)
                node = child
                key  = key[len(child.key):]

        return page_indices, node

    def _insert_helper(
        self, node: TreeNode, key: Tuple[int, ...], value: List[int]
    ) -> int:
        """Insert key/value starting at node. Returns n_overlap_pages."""
        P       = self.page_size
        t       = time.monotonic()
        node.last_access_time = t
        overlap = 0

        while len(key) >= P:
            ck = self._child_key(key)

            if ck not in node.children:
                # Insert as a new leaf.
                leaf                  = TreeNode()
                leaf.parent           = node
                leaf.key              = key
                leaf.value            = list(value)
                leaf.last_access_time = t
                node.children[ck]     = leaf
                return overlap

            child                  = node.children[ck]
            child.last_access_time = t
            n_pages_match          = self._count_match_pages(child.key, key)

            if n_pages_match < len(child.value):
                # Partial match → split, then insert remainder after the split.
                new_node  = self._split_node(child, n_pages_match)
                overlap  += n_pages_match
                key       = key[n_pages_match * P:]
                value     = value[n_pages_match:]
                node      = new_node
            else:
                # Full match → descend and continue inserting the remainder.
                overlap += len(child.value)
                key      = key[len(child.key):]
                value    = value[len(child.value):]
                node     = child

        return overlap

    def _delete_leaf(self, node: TreeNode) -> None:
        if node.parent is not None:
            ck = self._child_key(node.key)
            node.parent.children.pop(ck, None)
            node.parent = None

    def _iter_nodes(self):
        """Depth-first iteration over all non-root nodes."""
        stack = list(self.root.children.values())
        while stack:
            n = stack.pop()
            yield n
            stack.extend(n.children.values())

    # ── Debug helpers ─────────────────────────────────────────────────────────

    def total_cached_pages(self) -> int:
        return sum(len(n.value) for n in self._iter_nodes())

    def __repr__(self) -> str:
        return (
            f"RadixCache(page_size={self.page_size}, "
            f"nodes={sum(1 for _ in self._iter_nodes())}, "
            f"cached_pages={self.total_cached_pages()})"
        )
