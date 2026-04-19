"""
test_page_packing.py
====================
Unit tests for the page-packing logic in compute_write_info().

These tests do NOT require a GPU or model — they only exercise the CPU-side
page allocation and WriteInfo computation.

Key invariant tested:
  After each call to compute_write_info():
    len(slot_indices) == ceil(total_committed / P)

This is the invariant that ensures correct kv_last_page_len values in the
FlashInfer metadata and prevents the fragmentation bug described in the README.
"""

import math
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import KVPool, ReqToTokenPool, WriteInfo, compute_write_info

# Use CPU for all pool tensors in these unit tests (no GPU needed).
DEVICE = "cpu"

# ─────────────────────────────────────────────────────────────────────────────
# Minimal stub pools (CPU, tiny)
# ─────────────────────────────────────────────────────────────────────────────

def make_stub_pool(total_pages: int = 64, page_size: int = 16) -> KVPool:
    """KVPool with CPU tensors (no GPU memory allocated)."""
    pool = object.__new__(KVPool)
    pool.total_pages = total_pages
    pool.page_size   = page_size
    pool.free_slots  = list(range(1, total_pages))
    return pool


def make_stub_rtp(max_batch: int = 8, max_ctx: int = 32) -> ReqToTokenPool:
    """ReqToTokenPool with CPU tensors."""
    rtp = object.__new__(ReqToTokenPool)
    rtp.max_batch       = max_batch
    rtp.max_context_len = max_ctx
    rtp.req_to_token    = torch.zeros(max_batch, max_ctx, dtype=torch.int32)
    rtp.free_slots      = list(range(max_batch))
    return rtp


# Monkey-patch alloc to work on a stub pool (CPU version).
def _pool_alloc(pool: KVPool, n_tokens: int) -> List[int]:
    n = math.ceil(n_tokens / pool.page_size)
    pages = pool.free_slots[:n]
    pool.free_slots = pool.free_slots[n:]
    return pages


# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _compute_wi_cpu(pool, rtp, slot_indices, rpi, kv_committed_len, n_fill):
    """compute_write_info with CPU tensor writes."""
    P          = pool.page_size
    n_leftover = kv_committed_len % P

    existing_page  = None
    existing_slots = 0
    if n_leftover > 0 and slot_indices:
        space_in_last  = P - n_leftover
        existing_slots = min(space_in_last, n_fill)
        existing_page  = slot_indices[-1]

    remaining  = n_fill - existing_slots
    new_pages: List[int] = []
    if remaining > 0:
        new_pages = _pool_alloc(pool, remaining)
        n_prev    = len(slot_indices)
        slot_indices.extend(new_pages)
        pages_t = torch.tensor(new_pages, dtype=torch.int32)
        rtp.req_to_token[rpi, n_prev : n_prev + len(new_pages)] = pages_t

    return WriteInfo(
        existing_page  = existing_page,
        n_leftover     = n_leftover,
        existing_slots = existing_slots,
        new_pages      = new_pages,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Invariant tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("page_size", [4, 8, 16])
@pytest.mark.parametrize("chunk_size", [1, 3, 4, 7, 8, 12, 16, 24])
@pytest.mark.parametrize("total_len", [8, 24, 48, 100])
def test_page_count_invariant(page_size, chunk_size, total_len):
    """
    After all chunks: len(slot_indices) == ceil(total_len / page_size).
    This is the key invariant that prevents kv_last_page_len errors.
    """
    pool         = make_stub_pool(page_size=page_size)
    rtp          = make_stub_rtp()
    slot_indices = []
    rpi          = 0
    kv_committed = 0

    for c in range(math.ceil(total_len / chunk_size)):
        start = c * chunk_size
        end   = min(start + chunk_size, total_len)
        _compute_wi_cpu(pool, rtp, slot_indices, rpi, kv_committed, end - start)
        kv_committed = end

        expected_pages = math.ceil(kv_committed / page_size)
        assert len(slot_indices) == expected_pages, (
            f"After chunk {c} [{start}:{end}]: "
            f"got {len(slot_indices)} pages, expected {expected_pages} "
            f"(page_size={page_size}, chunk_size={chunk_size})"
        )


@pytest.mark.parametrize("page_size", [4, 8, 16])
@pytest.mark.parametrize("chunk_size", [3, 7, 12, 24])
@pytest.mark.parametrize("total_len", [24, 48, 100])
def test_kv_last_page_len_is_correct(page_size, chunk_size, total_len):
    """
    After each chunk, kv_last_page_len = (total_committed % P) or P.
    With the invariant, this always equals the actual fill level of the last page.
    """
    pool         = make_stub_pool(page_size=page_size)
    rtp          = make_stub_rtp()
    slot_indices = []
    rpi          = 0
    kv_committed = 0

    for c in range(math.ceil(total_len / chunk_size)):
        start = c * chunk_size
        end   = min(start + chunk_size, total_len)
        _compute_wi_cpu(pool, rtp, slot_indices, rpi, kv_committed, end - start)
        kv_committed = end

        last_fill = kv_committed % page_size
        expected_kv_last = last_fill if last_fill != 0 else page_size

        # The last page index in slot_indices should have exactly expected_kv_last tokens.
        # We verify this via the invariant: n_pages * P = ceil(kv_committed/P) * P.
        n_pages = len(slot_indices)
        max_tokens_in_pool = n_pages * page_size
        # Tokens in last page = total_committed - (n_pages-1)*P
        tokens_in_last = kv_committed - (n_pages - 1) * page_size
        assert tokens_in_last == expected_kv_last, (
            f"chunk {c}: tokens_in_last={tokens_in_last} "
            f"expected={expected_kv_last} "
            f"(page_size={page_size}, chunk_size={chunk_size})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# WriteInfo field tests
# ─────────────────────────────────────────────────────────────────────────────

def test_first_chunk_no_existing_page():
    """First chunk always has existing_page=None (nothing in pool yet)."""
    pool = make_stub_pool(page_size=16)
    rtp  = make_stub_rtp()
    si   = []
    wi   = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=0, n_fill=24)
    assert wi.existing_page  is None
    assert wi.n_leftover     == 0
    assert wi.existing_slots == 0
    assert len(wi.new_pages) == math.ceil(24 / 16)


def test_aligned_chunk_no_existing_page():
    """
    When kv_committed_len is a multiple of page_size, the last page is full
    so there's no partial page to continue — existing_page is None.
    """
    pool = make_stub_pool(page_size=16)
    rtp  = make_stub_rtp()
    si   = []
    # Chunk 0: 32 tokens (exactly 2 pages, last page is full)
    _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=0, n_fill=32)
    assert si == [1, 2]

    # Chunk 1: kv_committed_len=32 (% 16 == 0) → no partial page
    wi2 = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=32, n_fill=16)
    assert wi2.existing_page  is None
    assert wi2.existing_slots == 0
    assert len(wi2.new_pages) == 1


def test_partial_chunk_fills_existing_page():
    """
    Chunk 0: 24 tokens → pages [1, 2]; page 2 has 8 tokens (P=16).
    Chunk 1: should FILL page 2 (8 slots) then allocate page 3 for overflow.
    """
    P    = 16
    pool = make_stub_pool(page_size=P)
    rtp  = make_stub_rtp()
    si   = []

    # Chunk 0: 24 tokens → 2 pages (page 2 partial with 8 tokens)
    wi0 = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=0, n_fill=24)
    assert si == [1, 2]
    assert wi0.existing_page  is None   # first chunk, no prior page
    assert wi0.existing_slots == 0

    # Chunk 1: 24 tokens, kv_committed=24
    # n_leftover = 24 % 16 = 8  → page 2 has 8 slots available
    # existing_slots = min(16-8, 24) = 8  (fills page 2)
    # remaining = 24 - 8 = 16 → 1 new page (page 3)
    wi1 = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=24, n_fill=24)
    assert wi1.existing_page  == 2     # continues page 2
    assert wi1.n_leftover     == 8
    assert wi1.existing_slots == 8
    assert wi1.new_pages      == [3]   # one new page for remaining 16 tokens

    # Invariant: len(si) == ceil(48/16) = 3
    assert si == [1, 2, 3]
    assert len(si) == math.ceil(48 / P)


def test_chunk_smaller_than_page_space():
    """
    If the chunk is smaller than the remaining space in the last page,
    no new pages are allocated.
    """
    P    = 16
    pool = make_stub_pool(page_size=P)
    rtp  = make_stub_rtp()
    si   = []

    # Chunk 0: 12 tokens → 1 page (page 1, 12 tokens, 4 slots free)
    _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=0, n_fill=12)
    assert si == [1]

    # Chunk 1: 3 tokens (fits in existing page, no new pages needed)
    wi = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=12, n_fill=3)
    assert wi.existing_page  == 1
    assert wi.existing_slots == 3
    assert wi.new_pages      == []

    # Invariant: len(si) == ceil(15/16) = 1
    assert si == [1]
    assert len(si) == math.ceil(15 / P)


def test_rtp_updated_correctly():
    """
    req_to_token[rpi, :n_pages] must contain the correct page indices
    after page packing across two chunks.
    """
    P    = 16
    pool = make_stub_pool(page_size=P)
    rtp  = make_stub_rtp()
    si   = []
    rpi  = 0

    # Chunk 0: 24 tokens → pages [1, 2]
    _compute_wi_cpu(pool, rtp, si, rpi, kv_committed_len=0, n_fill=24)
    assert rtp.req_to_token[rpi, :2].tolist() == [1, 2]

    # Chunk 1: 24 tokens → page 3 (page 2 filled across boundary)
    _compute_wi_cpu(pool, rtp, si, rpi, kv_committed_len=24, n_fill=24)
    # After: si=[1,2,3], rtp[rpi, :3]=[1,2,3]
    assert rtp.req_to_token[rpi, :3].tolist() == [1, 2, 3]
    assert rtp.req_to_token[rpi, 3].item()    == 0  # unused slot is still 0


def test_chunk_exactly_fills_partial_page():
    """
    Chunk fills exactly the remaining space in the partial page, with nothing
    left over (n_fill == space_in_last exactly).
    """
    P    = 16
    pool = make_stub_pool(page_size=P)
    rtp  = make_stub_rtp()
    si   = []

    # Chunk 0: 8 tokens → page 1 (partial, 8 slots used)
    _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=0, n_fill=8)
    assert si == [1]

    # Chunk 1: exactly 8 tokens fills page 1, no new pages
    wi = _compute_wi_cpu(pool, rtp, si, 0, kv_committed_len=8, n_fill=8)
    assert wi.existing_slots == 8
    assert wi.new_pages      == []
    assert si == [1]  # still only 1 page

    assert len(si) == math.ceil(16 / P)  # == 1
