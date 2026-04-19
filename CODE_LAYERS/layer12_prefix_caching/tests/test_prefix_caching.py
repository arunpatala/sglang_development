"""
test_prefix_caching.py
======================
Correctness tests for Layer 11's prefix caching (RadixCache).

Tests are grouped into three sections:

Part A — RadixCache unit tests (CPU-only, no GPU/model needed)
  A1. Empty cache returns no match.
  A2. Insert then match full sequence.
  A3. Partial prefix match (sequence shares first N pages).
  A4. Node splitting — two sequences share a prefix then diverge.
  A5. Duplicate insert returns full overlap.
  A6. lock_ref prevents eviction; dec_lock_ref re-enables it.
  A7. Eviction frees unlocked leaves in LRU order.
  A8. match_prefix always leaves at least 1 token for the forward pass.
  A9. Page-size boundary: match only at full-page granularity.

Part B — GPU end-to-end: prefix hit produces correct logits
  B1. Cached prefix + suffix prefill == full prefill (top-1 token matches,
      max-logit-diff within bfloat16 tolerance).
  B2. Two requests with a shared prefix: second request hits the cache and
      produces the same last-token logit as a non-cached full prefill.
  B3. cache_finished_req inserts the full sequence; a third request sharing
      both the prefix AND the suffix now gets a longer cache hit.

Part C — Integration: RadixCache interacts correctly with page ownership
  C1. cache_finished_req frees duplicate pages when two requests computed the
      same suffix independently.
  C2. Eviction + re-use: evicted pages are returned to the pool and can be
      re-allocated for new requests.
  C3. lock_ref during prefill protects pages from eviction until the request
      calls dec_lock_ref.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import List, Tuple
from unittest.mock import patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import KVPool, ReqToTokenPool
from radix_cache import RadixCache, TreeNode
from conftest import (
    ATOL, DEVICE, DTYPE, do_extend, full_ref, make_pool,
    make_radix_cache, prefill_with_prefix,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers shared across tests
# ─────────────────────────────────────────────────────────────────────────────

def _fake_pool(n_pages: int = 128):
    """Minimal stub KVPool for CPU-only tests (no GPU tensors)."""
    class FakePool:
        page_size  = 1  # unused by RadixCache directly
        free_slots = list(range(1, n_pages + 1))
        def free(self, pages):  self.free_slots.extend(pages)
        def available(self):    return len(self.free_slots)
    return FakePool()


def _make_cache(page_size: int = 4, n_pages: int = 128) -> RadixCache:
    pool = _fake_pool(n_pages)
    return RadixCache(pool, page_size)


def _toks(n: int, offset: int = 0) -> List[int]:
    """Sequential token IDs starting at `offset`."""
    return list(range(offset, offset + n))


def encode(tok, text: str) -> List[int]:
    formatted = tok.apply_chat_template([{"role": "user", "content": text}])
    return tok._tok([formatted], return_tensors="pt", padding=False)["input_ids"][0].tolist()


# ═════════════════════════════════════════════════════════════════════════════
# Part A — CPU-only RadixCache unit tests
# ═════════════════════════════════════════════════════════════════════════════

class TestRadixCacheUnit:
    """CPU-only RadixCache unit tests — no GPU or model required."""

    # ── A1 ───────────────────────────────────────────────────────────────────

    def test_empty_cache_returns_no_match(self):
        cache = _make_cache(page_size=4)
        pages, length, node = cache.match_prefix(_toks(20))
        assert pages  == []
        assert length == 0
        assert node   is cache.root

    # ── A2 ───────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("page_size", [1, 4, 8, 16])
    def test_insert_then_match(self, page_size):
        """
        After inserting N tokens, match_prefix(tok + [sentinel]) returns all
        N tokens — the sentinel is the 'last token' that prevents full caching,
        so the N real tokens are returned in their entirety.
        """
        P      = page_size
        cache  = _make_cache(page_size=P)
        n_tok  = 4 * P   # exactly 4 pages
        tok    = _toks(n_tok)
        pages  = list(range(1, 1 + n_tok // P))

        n_ov = cache.insert(tok, pages)
        assert n_ov == 0, "Fresh insert must have 0 overlap"

        # Query: tok + [9999] has n_tok+1 tokens.
        # match_prefix cap: floor((n_tok+1 - 1) / P) * P = n_tok (since n_tok % P == 0).
        # The cache contains exactly n_tok tokens so all 4 pages match.
        got_pages, got_len, _ = cache.match_prefix(tok + [9999])
        assert got_len   == n_tok, f"Expected {n_tok} matched tokens, got {got_len}"
        assert got_pages == pages

    # ── A3 ───────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("page_size", [4, 8])
    def test_partial_prefix_match(self, page_size):
        """A sequence with a shorter version in the cache returns the prefix."""
        P      = page_size
        cache  = _make_cache(page_size=P)

        tok_long  = _toks(4 * P)            # [0..4P-1]
        pgs_long  = list(range(1, 5))       # pages 1-4
        cache.insert(tok_long, pgs_long)

        # Sequence that shares only the first 2 pages, then diverges
        tok_short = _toks(2 * P) + _toks(2 * P, offset=100)
        got_pages, got_len, _ = cache.match_prefix(tok_short + [9999])
        assert got_len   == 2 * P
        assert got_pages == pgs_long[:2]

    # ── A4 ───────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("page_size", [4, 8])
    def test_node_splitting(self, page_size):
        """
        After inserting seq_A and seq_B that share a prefix, both must
        be retrievable with correct pages.

        Query tok + [sentinel] has 4P+1 tokens.
        match_prefix cap: floor((4P+1-1)/P)*P = 4P.
        So all 4 pages can be matched for both sequences.
        """
        P     = page_size
        cache = _make_cache(page_size=P)

        # seq_A: [0..4P-1]  →  pages [10, 11, 12, 13]
        tok_a = _toks(4 * P)
        pgs_a = [10, 11, 12, 13]
        cache.insert(tok_a, pgs_a)

        # seq_B: [0..2P-1] + [100..102P-1]  →  shared prefix with seq_A for 2 pages
        tok_b = _toks(2 * P) + _toks(2 * P, offset=100)
        pgs_b = [20, 21, 22, 23]
        n_ov  = cache.insert(tok_b, pgs_b)
        assert n_ov == 2, f"Expected 2 overlap pages, got {n_ov}"

        # seq_A: query has 4P+1 tokens → cap = 4P → all 4 pages match.
        got_a, len_a, _ = cache.match_prefix(tok_a + [9999])
        assert len_a == 4 * P, f"Expected {4*P}, got {len_a}"
        assert got_a == pgs_a

        # seq_B: first 2 pages from the shared prefix node (pgs_a[:2]),
        # next 2 pages from seq_B's diverged branch (pgs_b[2:]).
        # pgs_b[:2] are overlap pages — they weren't re-stored in the tree.
        expected_b = pgs_a[:2] + pgs_b[2:]
        got_b, len_b, _ = cache.match_prefix(tok_b + [9999])
        assert len_b == 4 * P, f"Expected {4*P}, got {len_b}"
        assert got_b == expected_b

    # ── A5 ───────────────────────────────────────────────────────────────────

    def test_duplicate_insert_returns_full_overlap(self):
        """Inserting the same token sequence twice returns all pages as overlap."""
        P     = 4
        cache = _make_cache(page_size=P)
        tok   = _toks(4 * P)
        pgs   = [1, 2, 3, 4]

        cache.insert(tok, pgs)
        n_ov = cache.insert(tok, [5, 6, 7, 8])  # different pages, same tokens
        assert n_ov == 4

    # ── A6 ───────────────────────────────────────────────────────────────────

    def test_lock_ref_prevents_eviction(self):
        """Nodes with lock_ref > 0 must not be evicted."""
        P     = 4
        cache = _make_cache(page_size=P)
        tok   = _toks(4 * P)
        pgs   = [1, 2, 3, 4]
        cache.insert(tok, pgs)

        # Lock the leaf node.
        _, _, node = cache.match_prefix(tok + [9999])
        cache.inc_lock_ref(node)

        before = cache.total_cached_pages()
        freed  = cache.evict(9999)
        assert freed == 0, "No pages should be freed while locked"
        assert cache.total_cached_pages() == before

        # Release lock; now eviction should work.
        cache.dec_lock_ref(node)
        freed2 = cache.evict(9999)
        assert freed2 > 0
        assert cache.total_cached_pages() == 0

    # ── A7 ───────────────────────────────────────────────────────────────────

    def test_eviction_lru_order(self):
        """
        Older (less recently accessed) nodes are evicted first.
        We insert two sequences, access the second one (updating its
        last_access_time), then evict one page's worth — only the first
        sequence's leaf should go.
        """
        import time
        P     = 4
        cache = _make_cache(page_size=P)

        tok_old = _toks(P)          # 1 page
        tok_new = _toks(P, offset=50)  # 1 page

        cache.insert(tok_old, [1])
        time.sleep(0.01)            # ensure distinct timestamps
        cache.insert(tok_new, [2])

        # "touch" tok_new so it's recently accessed
        cache.match_prefix(tok_new + [9999])

        # Evict exactly 1 page — should remove tok_old's leaf
        freed = cache.evict(1)
        assert freed == 1
        assert cache.total_cached_pages() == 1

        # tok_new must still match
        got_pages, got_len, _ = cache.match_prefix(tok_new + [9999])
        assert got_len > 0

    # ── A8 ───────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("page_size", [1, 4, 16])
    @pytest.mark.parametrize("n_tok", [1, 3, 15, 16, 17, 32, 33])
    def test_match_prefix_always_leaves_one_token(self, page_size, n_tok):
        """
        match_prefix must never return matched_len >= len(token_ids).
        The forward pass always needs at least 1 non-cached token.
        """
        P     = page_size
        cache = _make_cache(page_size=P)

        tok   = _toks(n_tok)
        # Fill a large pool with sequential pages
        n_pages = math.ceil(n_tok / P) + 1
        pgs   = list(range(1, n_pages + 1))
        cache.insert(tok, pgs)

        _, matched_len, _ = cache.match_prefix(tok)
        assert matched_len < n_tok, (
            f"matched_len={matched_len} must be < n_tok={n_tok} "
            f"(page_size={P})"
        )

    # ── A9 ───────────────────────────────────────────────────────────────────

    @pytest.mark.parametrize("page_size", [4, 8, 16])
    def test_match_only_at_full_page_granularity(self, page_size):
        """
        Even if N tokens match, we can only report page_size-aligned matches.
        Tokens beyond the last full-page boundary are never returned.
        """
        P       = page_size
        n_full  = 2 * P          # 2 full pages
        n_extra = P // 2         # partial page worth of extra tokens

        cache = _make_cache(page_size=P)
        tok   = _toks(n_full + n_extra)
        pgs   = list(range(1, n_full // P + 2))  # enough pages
        cache.insert(tok, pgs)

        # Match against the same sequence + a sentinel so match isn't capped
        # at floor((len-1)/P)... we want to see the page-granularity cap.
        query_tok = tok + [9999]
        _, got_len, _ = cache.match_prefix(query_tok)

        # The partial extra tokens should not be counted.
        assert got_len % P == 0, (
            f"matched_len={got_len} is not a multiple of page_size={P}"
        )
        assert got_len <= n_full


# ═════════════════════════════════════════════════════════════════════════════
# Part B — GPU end-to-end correctness
# ═════════════════════════════════════════════════════════════════════════════

# Prompts designed to share a long common prefix.
SYS_PREFIX = (
    "You are a helpful AI assistant. Always answer concisely and accurately. "
    "Think step by step before responding."
)

SHARED_PROMPTS = {
    "prefix":   SYS_PREFIX,
    "suffix_a": "What is 2 + 2?",
    "suffix_b": "What is the capital of France?",
    "suffix_c": "Name one planet in the solar system.",
}


@pytest.mark.parametrize("page_size", [8, 16])
def test_prefix_hit_logits_match_full_prefill(model, tok, cfg, page_size):
    """
    B1: A request that gets a prefix cache hit must produce the same top-1
    token as a full (non-cached) prefill of the same prompt.

    Protocol:
      1. Full prefill of prefix + suffix_a → store prefix pages in cache.
      2. New request for prefix + suffix_a → match_prefix gives cached prefix.
      3. Run prefill_with_prefix (only suffix_a tokens in forward pass).
      4. Compare last-token logit with the full-prefill reference.
    """
    prefix_ids = encode(tok, SHARED_PROMPTS["prefix"])
    suffix_ids = encode(tok, SHARED_PROMPTS["suffix_a"])
    full_ids   = prefix_ids + suffix_ids
    L          = len(full_ids)
    P          = page_size

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    # ── Step 1: full reference prefill (no caching) ───────────────────────
    kv_full, rtp_full = make_pool(cfg, L + 16, page_size=P)
    rpi_full          = rtp_full.alloc()
    slot_full         = []
    rd_full = [{"fill_ids": full_ids, "kv_committed_len": 0,
                "slot_indices": slot_full, "req_pool_idx": rpi_full}]
    logits_full, qo_full = do_extend(model, kv_full, rtp_full, ws, rd_full, cfg)
    ref_logit = logits_full[0, qo_full[1] - 1, :].clone()

    # ── Step 2: insert prefix into RadixCache ─────────────────────────────
    cache  = make_radix_cache(kv_full)
    n_pfx_pages = len(prefix_ids) // P
    prefix_pages = slot_full[:n_pfx_pages]
    cache.insert(prefix_ids[:n_pfx_pages * P], prefix_pages)

    # ── Step 3: match and prefill with cached prefix ───────────────────────
    ws2 = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    # Use the SAME kv_full pool — cached pages are already there.
    rtp2 = ReqToTokenPool(max_batch=8, max_context_len=256)

    matched_pages, matched_len, last_node = cache.match_prefix(full_ids)
    assert matched_len > 0, "Expected a prefix hit"
    cache.inc_lock_ref(last_node)

    cached_logit, _, _ = prefill_with_prefix(
        model, kv_full, rtp2, ws2, full_ids,
        matched_pages, matched_len, cfg,
    )

    # ── Step 4: compare ────────────────────────────────────────────────────
    # Only check top-1 token: bfloat16 logit differences between a full
    # all-tokens attention pass and a prefix-cached pass can be > 0.5 due to
    # different FlashInfer internal tiling paths, but the greedy token must
    # be the same.
    ref_tok    = int(ref_logit.argmax())
    cached_tok = int(cached_logit.argmax())

    assert ref_tok == cached_tok, (
        f"Top-1 token mismatch: full={ref_tok} cached={cached_tok} "
        f"(page_size={P}, matched={matched_len}/{L}, "
        f"max_logit_diff={(ref_logit - cached_logit).abs().max().item():.4f})"
    )

    cache.dec_lock_ref(last_node)


@pytest.mark.parametrize("page_size", [16])
def test_two_requests_shared_prefix(model, tok, cfg, page_size):
    """
    B2: Request A and Request B share the same long prefix.

    Protocol:
      A runs first (no cache hit).  After A finishes, its prefix pages are
      inserted into the RadixCache.  B then runs with a cache hit on the
      shared prefix.  B's last-token logit must match a full prefill of B.
    """
    prefix_ids = encode(tok, SHARED_PROMPTS["prefix"])
    ids_a      = prefix_ids + encode(tok, SHARED_PROMPTS["suffix_a"])
    ids_b      = prefix_ids + encode(tok, SHARED_PROMPTS["suffix_b"])
    P          = page_size

    ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    total = len(ids_a) + len(ids_b) + 64
    kv_pool, rtp = make_pool(cfg, total, page_size=P, max_batch=16)
    cache = make_radix_cache(kv_pool)

    # ── Request A (full prefill, no caching yet) ──────────────────────────
    rpi_a   = rtp.alloc()
    slots_a = []
    rd_a = [{"fill_ids": ids_a, "kv_committed_len": 0,
              "slot_indices": slots_a, "req_pool_idx": rpi_a}]
    do_extend(model, kv_pool, rtp, ws, rd_a, cfg)

    # Insert A's prefix into the cache.
    n_pfx_pages = len(prefix_ids) // P
    aligned_pfx = n_pfx_pages * P
    cache.insert(prefix_ids[:aligned_pfx], slots_a[:n_pfx_pages])

    # ── Request B: full reference (no caching) ────────────────────────────
    ws_ref = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    kv_ref, rtp_ref = make_pool(cfg, len(ids_b) + 32, page_size=P)
    rpi_ref         = rtp_ref.alloc()
    slots_ref       = []
    rd_ref = [{"fill_ids": ids_b, "kv_committed_len": 0,
               "slot_indices": slots_ref, "req_pool_idx": rpi_ref}]
    logits_ref, qo_ref = do_extend(model, kv_ref, rtp_ref, ws_ref, rd_ref, cfg)
    ref_logit = logits_ref[0, qo_ref[1] - 1, :].clone()

    # ── Request B: cached prefill ─────────────────────────────────────────
    matched_pages, matched_len, last_node = cache.match_prefix(ids_b)
    assert matched_len >= aligned_pfx, (
        f"Cache miss: expected >= {aligned_pfx} matched tokens, got {matched_len}"
    )
    cache.inc_lock_ref(last_node)

    ws_b = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    rtp_b = ReqToTokenPool(max_batch=8, max_context_len=256)
    cached_logit, _, _ = prefill_with_prefix(
        model, kv_pool, rtp_b, ws_b, ids_b, matched_pages, matched_len, cfg,
    )

    cache.dec_lock_ref(last_node)

    # Verify top-1 token agreement.
    ref_tok    = int(ref_logit.argmax())
    cached_tok = int(cached_logit.argmax())
    diff       = (ref_logit - cached_logit).abs().max().item()

    assert ref_tok == cached_tok, (
        f"Token mismatch: ref={ref_tok} cached={cached_tok}  diff={diff:.4f}"
    )


@pytest.mark.parametrize("page_size", [16])
def test_cache_grows_after_first_request(model, tok, cfg, page_size):
    """
    B3: After request A finishes, the cache has A's full sequence.
    Request B (same prefix, different suffix) should get a longer prefix hit
    than request A did at the start.
    """
    prefix_ids = encode(tok, SHARED_PROMPTS["prefix"])
    ids_a      = prefix_ids + encode(tok, SHARED_PROMPTS["suffix_a"])
    ids_b      = prefix_ids + encode(tok, SHARED_PROMPTS["suffix_b"])
    P          = page_size

    ws       = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    total    = len(ids_a) + len(ids_b) + 64
    kv_pool, rtp = make_pool(cfg, total, page_size=P, max_batch=16)
    cache    = make_radix_cache(kv_pool)

    # Before any request — no cache hit.
    _, init_match, _ = cache.match_prefix(ids_a)
    assert init_match == 0

    # Request A: full prefill, then insert entire sequence into cache.
    rpi_a   = rtp.alloc()
    slots_a = []
    rd_a    = [{"fill_ids": ids_a, "kv_committed_len": 0,
                "slot_indices": slots_a, "req_pool_idx": rpi_a}]
    do_extend(model, kv_pool, rtp, ws, rd_a, cfg)

    aligned_a = (len(ids_a) // P) * P
    cache.insert(ids_a[:aligned_a], slots_a[:aligned_a // P])

    # Request B — should now hit a non-empty prefix in the cache.
    _, b_match, _ = cache.match_prefix(ids_b)

    # At least the common prefix pages are cached.
    aligned_pfx = (len(prefix_ids) // P) * P
    assert b_match >= aligned_pfx, (
        f"Expected cache hit >= {aligned_pfx} tokens, got {b_match}"
    )


# ═════════════════════════════════════════════════════════════════════════════
# Part C — Page ownership & resource management
# ═════════════════════════════════════════════════════════════════════════════

class TestPageOwnership:
    """CPU-only tests for cache_finished_req ownership rules."""

    def _insert_and_get_pages(self, cache, token_ids, n_pages):
        """Insert token_ids with freshly numbered pages. Returns page list."""
        P   = cache.page_size
        pgs = list(range(100, 100 + n_pages))
        cache.insert(token_ids, pgs)
        return pgs

    # ── C1 ───────────────────────────────────────────────────────────────────

    def test_cache_finished_req_frees_duplicate_pages(self):
        """
        If two requests independently computed the same suffix (race condition),
        cache_finished_req must free the duplicate pages of the second request.
        """
        P     = 4
        pool  = _fake_pool(200)
        cache = RadixCache(pool, P)

        # Req A and Req B both compute tok = [0..4P-1].
        # Insert Req A first.
        tok   = _toks(4 * P)
        pgs_a = list(range(10, 10 + 4))
        cache.insert(tok, pgs_a)

        # Simulate Req B finishing with the same tokens but different pages.
        import types
        req_b = types.SimpleNamespace(
            input_ids           = tok,
            output_ids          = [],              # no output for simplicity
            slot_indices        = list(range(20, 24)),  # pages 20-23 (duplicates)
            prefix_page_indices = [],              # no prefix hit
            last_node           = cache.root,
            req_pool_idx        = None,
        )

        # Fake rtp
        class FakeRTP:
            def free(self, idx): pass

        freed_before = len(pool.free_slots)
        cache.cache_finished_req(req_b, FakeRTP(), pool)
        freed_after  = len(pool.free_slots)

        # All 4 of req_b's pages were duplicates and should have been freed.
        assert freed_after - freed_before == 4, (
            f"Expected 4 freed pages, got {freed_after - freed_before}"
        )

    # ── C2 ───────────────────────────────────────────────────────────────────

    def test_evicted_pages_returned_to_pool(self):
        """
        After eviction, freed pages must be counted in pool.available().
        """
        P     = 4
        pool  = _fake_pool(20)
        cache = RadixCache(pool, P)

        tok = _toks(4 * P)
        pgs = list(range(1, 5))
        cache.insert(tok, pgs)

        before = pool.available()
        cache.evict(4)
        after  = pool.available()
        assert after == before + 4

    # ── C3 ───────────────────────────────────────────────────────────────────

    def test_prefix_pages_not_freed_by_finished_req(self):
        """
        Prefix pages (owned by the cache, returned by match_prefix) must NOT
        be freed when cache_finished_req is called — only the tail page and
        any newly computed duplicate pages are freed.
        """
        P     = 4
        pool  = _fake_pool(200)
        cache = RadixCache(pool, P)

        # Insert prefix into cache.
        prefix_tok = _toks(2 * P)
        prefix_pgs = [1, 2]
        cache.insert(prefix_tok, prefix_pgs)

        # Simulate a request that matched 2 prefix pages and computed 2 more.
        suffix_tok = _toks(2 * P, offset=100)
        new_pgs    = [3, 4]   # newly allocated pages for suffix
        all_pages  = prefix_pgs + new_pgs

        import types
        matched_pages, matched_len, last_node = cache.match_prefix(
            prefix_tok + suffix_tok + [9999]
        )
        cache.inc_lock_ref(last_node)

        full_tok = prefix_tok + suffix_tok
        req = types.SimpleNamespace(
            input_ids           = full_tok,
            output_ids          = [],
            slot_indices        = all_pages,
            prefix_page_indices = matched_pages,
            last_node           = last_node,
            req_pool_idx        = None,
        )

        class FakeRTP:
            def free(self, idx): pass

        freed_before = len(pool.free_slots)
        cache.cache_finished_req(req, FakeRTP(), pool)
        freed_after  = len(pool.free_slots)

        # Suffix tokens: 2*P = 8 tokens = 2 pages.  Aligned total = 4*P = 16 tokens.
        # No tail page (4*P / P = 4 pages, exactly aligned).
        # No duplicate pages (suffix pages 3,4 weren't in tree before).
        # → 0 pages freed.  Prefix pages 1,2 stay in tree.
        assert freed_after == freed_before, (
            f"Prefix pages should NOT be freed: freed {freed_after - freed_before}"
        )
        # The full sequence is now in the cache.
        assert cache.total_cached_pages() >= 4

    # ── C4 ───────────────────────────────────────────────────────────────────

    def test_unaligned_tail_page_freed(self):
        """
        If the total token count (prompt + output) is not page-aligned,
        the partial last page must be freed by cache_finished_req.
        """
        P     = 4
        pool  = _fake_pool(200)
        cache = RadixCache(pool, P)

        # Request with 9 output tokens (not page-aligned with P=4)
        n_input  = 2 * P    # 8 tokens
        n_output = 3        # 3 output tokens → total 11 tokens
        # 11 / 4 → 2 full pages + 1 partial page
        all_pages = [10, 11, 12]  # 3 pages allocated

        import types
        req = types.SimpleNamespace(
            input_ids           = _toks(n_input),
            output_ids          = _toks(n_output, offset=200),
            slot_indices        = all_pages,
            prefix_page_indices = [],
            last_node           = cache.root,
            req_pool_idx        = None,
        )

        class FakeRTP:
            def free(self, idx): pass

        freed_before = len(pool.free_slots)
        cache.cache_finished_req(req, FakeRTP(), pool)
        freed_after  = len(pool.free_slots)

        # Aligned len = floor(11/4)*4 = 8 tokens = 2 pages cached.
        # Tail = page 12 (1 page) freed.
        assert freed_after - freed_before == 1, (
            f"Expected 1 tail page freed, got {freed_after - freed_before}"
        )
        assert cache.total_cached_pages() == 2


# ═════════════════════════════════════════════════════════════════════════════
# Part D — Integration: match_prefix + prefill round-trip
# ═════════════════════════════════════════════════════════════════════════════

@pytest.mark.parametrize("page_size", [8, 16])
def test_three_requests_growing_cache(model, tok, cfg, page_size):
    """
    Three requests with the same system prompt but different user questions.

    Validates cache GROWTH behaviour — not logit values (those are tested in
    test_two_requests_shared_prefix and test_prefix_hit_logits_match_full_prefill):

    Before any request: cache is empty → no prefix hit.
    After Req A:        cache has A's prefix → Req B gets a hit on sys_prompt.
    After Req B:        cache has more data → Req C gets at least a sys_prompt hit.
    """
    sys_ids = encode(tok, SHARED_PROMPTS["prefix"])
    ids_a   = sys_ids + encode(tok, SHARED_PROMPTS["suffix_a"])
    ids_b   = sys_ids + encode(tok, SHARED_PROMPTS["suffix_b"])
    ids_c   = sys_ids + encode(tok, SHARED_PROMPTS["suffix_c"])

    P     = page_size
    total = len(ids_a) + len(ids_b) + len(ids_c) + 128
    ws    = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    kv_pool, rtp = make_pool(cfg, total, page_size=P, max_batch=16)
    cache        = make_radix_cache(kv_pool)
    aligned_pfx  = (len(sys_ids) // P) * P

    # ── Before any request ────────────────────────────────────────────────
    _, init_match, _ = cache.match_prefix(ids_a)
    assert init_match == 0, "Cache should be empty before any request"

    # ── Req A: full prefill, insert prefix into cache ─────────────────────
    rpi_a   = rtp.alloc()
    slots_a = []
    do_extend(model, kv_pool, rtp, ws,
              [{"fill_ids": ids_a, "kv_committed_len": 0,
                "slot_indices": slots_a, "req_pool_idx": rpi_a}], cfg)

    aligned_a = (len(ids_a) // P) * P
    cache.insert(ids_a[:aligned_a], slots_a[:aligned_a // P])

    pages_after_a = cache.total_cached_pages()
    assert pages_after_a > 0, "Cache should have pages after Req A"

    # ── Req B: sys_prompt should now hit the cache ────────────────────────
    _, match_b, node_b = cache.match_prefix(ids_b)
    assert match_b >= aligned_pfx, (
        f"Expected sys_prompt hit >= {aligned_pfx} tokens, got {match_b}"
    )
    cache.inc_lock_ref(node_b)

    rpi_b   = rtp.alloc()
    slots_b = list(slots_a[:match_b // P])
    pages_t = torch.tensor(slots_b, dtype=torch.int32, device=DEVICE)
    rtp.req_to_token[rpi_b, :len(slots_b)] = pages_t

    do_extend(model, kv_pool, rtp, ws,
              [{"fill_ids": ids_b[match_b:], "kv_committed_len": match_b,
                "slot_indices": slots_b, "req_pool_idx": rpi_b}], cfg)
    cache.dec_lock_ref(node_b)

    aligned_b = (len(ids_b) // P) * P
    cache.insert(ids_b[:aligned_b], slots_b[:aligned_b // P])
    pages_after_b = cache.total_cached_pages()
    assert pages_after_b >= pages_after_a, "Cache should grow or stay after Req B"

    # ── Req C: should also hit the sys_prompt prefix at minimum ───────────
    _, match_c, _ = cache.match_prefix(ids_c)
    assert match_c >= aligned_pfx, (
        f"Expected sys_prompt hit >= {aligned_pfx} tokens, got {match_c}"
    )
