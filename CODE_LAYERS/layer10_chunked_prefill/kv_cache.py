"""
Layer 10 — KV Cache: unified ExtendKVCtx for batched + chunked prefill.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changes from Layer 9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 9 (PrefillKVCtx):            Layer 10 (ExtendKVCtx):
  ─────────────────────────────────   ─────────────────────────────────────
  B=1 prefill only                    B=N batched prefill
  F.sdpa with causal mask             BatchPrefillWithPagedKVCacheWrapper
  No chunked-prefill support          Full chunked-prefill support
  kv_committed_len always 0           kv_committed_len tracks prior chunks
  PrefillKVCtx type                   ExtendKVCtx type (one type for all)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Page packing across chunk boundaries
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

When a chunk ends in the MIDDLE of a page (kv_committed_len % P != 0), the
next chunk CONTINUES filling that partial page before allocating new ones.

Example: P=16, chunk_size=24
  Chunk 0 (24 tokens → 2 pages):
    Page 1: positions 0..15 (full, 16 tokens)
    Page 2: positions 16..23 (partial, 8 tokens; entries 8..15 are zero-pad)
    kv_last_page_len = 8   (page 2 has 8 valid tokens)

  Chunk 1 (24 tokens → CONTINUE page 2, then 1 new page):
    Page 2: positions 16..23 from chunk 0 + positions 24..31 from chunk 1
            → now full (16 tokens)
    Page 3: positions 32..47 (full, 16 tokens)
    Total slot_indices after chunk 1 = [1, 2, 3] (3 pages, not 4!)
    kv_last_page_len = 48 % 16 = 0 → P = 16 (page 3 is full)

Without page packing (naïve approach):
  Chunk 1 would allocate 2 NEW pages (3, 4), giving 4 pages for 48 tokens.
  Page 4 would have 8 valid tokens but kv_last_page_len=16 → WRONG!

Page packing guarantees:
  len(slot_indices) == ceil(total_committed / P)
  kv_last_page_len == total_committed % P  (or P if exact multiple)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ExtendKVCtx.write_info per request
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  existing_page  — last page from prior chunks (None if aligned or no prior)
  n_leftover     — tokens already in existing_page before this chunk
  existing_slots — tokens from this chunk going into existing_page
  new_pages      — newly allocated pages for the overflow beyond existing_page

store() writes:
  1. fill_ids[0:existing_slots] → existing_page[n_leftover : n_leftover+existing_slots]
  2. fill_ids[existing_slots:]  → new_pages (padded to page boundaries)
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
import flashinfer

logger = logging.getLogger(__name__)

DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# ReqToTokenPool (unchanged from Layer 9)
# ─────────────────────────────────────────────────────────────────────────────

class ReqToTokenPool:
    """2D GPU int32 table: req_to_token[req_pool_idx, page_pos] = page_idx."""

    def __init__(self, max_batch: int, max_context_len: int) -> None:
        self.max_batch       = max_batch
        self.max_context_len = max_context_len
        self.req_to_token = torch.zeros(
            (max_batch, max_context_len), dtype=torch.int32, device=DEVICE
        )
        self.free_slots: List[int] = list(range(max_batch))
        mem_mb = (max_batch * max_context_len * 4) / (1024 ** 2)
        logger.info(
            f"ReqToTokenPool: [{max_batch}, {max_context_len}] int32 "
            f"= {mem_mb:.1f} MB on GPU"
        )

    def available(self) -> int:
        return len(self.free_slots)

    def alloc(self) -> int:
        if not self.free_slots:
            raise RuntimeError("ReqToTokenPool exhausted")
        return self.free_slots.pop()

    def free(self, idx: int) -> None:
        self.free_slots.append(idx)


# ─────────────────────────────────────────────────────────────────────────────
# KVPool (unchanged from Layer 9)
# ─────────────────────────────────────────────────────────────────────────────

class KVPool:
    """Pre-allocated paged GPU KV store. Shape: [total_pages, page_size, n_kv, D]."""

    def __init__(
        self,
        total_pages: int,
        page_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.total_pages = total_pages
        self.page_size   = page_size
        self.n_layers    = n_layers
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = head_dim
        self.dtype       = dtype

        self.k_pool: List[torch.Tensor] = [
            torch.zeros(total_pages, page_size, n_kv_heads, head_dim,
                        dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]
        self.v_pool: List[torch.Tensor] = [
            torch.zeros(total_pages, page_size, n_kv_heads, head_dim,
                        dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]
        # Page 0 is never used (reserved as sentinel).
        self.free_slots: List[int] = list(range(1, total_pages))

        mem_gb = (
            n_layers * 2 * total_pages * page_size * n_kv_heads * head_dim
            * torch.finfo(dtype).bits // 8
        ) / (1024 ** 3)
        logger.info(
            f"KVPool: {total_pages} pages × page_size={page_size} × {n_layers} layers "
            f"= {mem_gb:.2f} GB"
        )

    def available(self) -> int:
        return len(self.free_slots)

    def n_pages_for(self, n_tokens: int) -> int:
        return math.ceil(n_tokens / self.page_size)

    def alloc(self, n_tokens: int) -> List[int]:
        n_pages = self.n_pages_for(n_tokens)
        if n_pages > len(self.free_slots):
            raise RuntimeError(
                f"KVPool OOM: need {n_pages} pages for {n_tokens} tokens, "
                f"only {len(self.free_slots)} free"
            )
        pages = self.free_slots[:n_pages]
        self.free_slots = self.free_slots[n_pages:]
        return pages

    def free(self, page_indices: List[int]) -> None:
        self.free_slots.extend(page_indices)


# ─────────────────────────────────────────────────────────────────────────────
# WriteInfo — describes how to write K/V for one request in an extend round
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class WriteInfo:
    """
    Tells ExtendKVCtx.store() exactly where each request's new tokens go.

    existing_page:  the last page from prior chunks (None if no partial page)
    n_leftover:     tokens already written in existing_page (0 if no partial)
    existing_slots: tokens from this chunk that complete existing_page
    new_pages:      newly allocated pages for overflow tokens
    """
    existing_page:  Optional[int]
    n_leftover:     int
    existing_slots: int
    new_pages:      List[int]


# ─────────────────────────────────────────────────────────────────────────────
# ExtendKVCtx — unified context for ALL prefill / extend operations
# ─────────────────────────────────────────────────────────────────────────────

class ExtendKVCtx:
    """
    Context for batched extend (prefill) — replaces PrefillKVCtx from Layer 9.

    Detected in attention.py by hasattr(kv_cache, 'extend_wrapper').

    Supports:
      • Full prefill, B=1  (single request, kv_committed_len=0)
      • Batched prefill B=N (multiple packed requests, kv_committed_len=0 each)
      • Continuation chunk  (kv_committed_len > 0, with page packing)

    Key invariants maintained by page packing:
      len(slot_indices) == ceil(total_committed / P) for every request.
      kv_last_page_len  == total_committed % P  (or P if exact multiple).
    """

    def __init__(
        self,
        wrapper: flashinfer.BatchPrefillWithPagedKVCacheWrapper,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        qo_indptr: List[int],
        write_infos: List[WriteInfo],
        page_size: int,
    ) -> None:
        self.extend_wrapper = wrapper          # duck-typing marker for attention.py
        self.k_pool         = k_pool
        self.v_pool         = v_pool
        self.qo_indptr      = qo_indptr        # [B+1] token offsets into packed seq
        self.write_infos    = write_infos
        self.page_size      = page_size

    def get_seq_length(self) -> int:
        # position_ids are always provided explicitly; this is never used.
        return 0

    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Write each request's new-chunk K/V into the pool using page packing.

        k shape from attention: [1, n_kv_heads, total_tokens, head_dim]
          where total_tokens = sum of all requests' extend_input_len.

        For each request i with WriteInfo wi:
          1. Fill existing partial page (if wi.existing_slots > 0):
               pool[layer][wi.existing_page, wi.n_leftover : n_leftover+es]
               ← k_i[0 : wi.existing_slots]
          2. Write overflow tokens into new pages:
               pool[layer][wi.new_pages] ← k_i[wi.existing_slots:] (padded)
        """
        P    = self.page_size
        n_kv = k.shape[1]
        D    = k.shape[3]

        for i, wi in enumerate(self.write_infos):
            s, e = self.qo_indptr[i], self.qo_indptr[i + 1]

            # [1, n_kv, L, D] → [L, n_kv, D]
            k_i = k[:, :, s:e, :].squeeze(0).permute(1, 0, 2).contiguous()
            v_i = v[:, :, s:e, :].squeeze(0).permute(1, 0, 2).contiguous()

            offset = 0

            # Step 1: fill existing partial page
            if wi.existing_slots > 0:
                es = wi.existing_slots
                lo = wi.n_leftover
                pg = wi.existing_page
                self.k_pool[layer_idx][pg, lo : lo + es] = k_i[:es]
                self.v_pool[layer_idx][pg, lo : lo + es] = v_i[:es]
                offset = es

            # Step 2: write overflow into new pages
            if wi.new_pages:
                k_rem = k_i[offset:].contiguous()
                v_rem = v_i[offset:].contiguous()
                n_pg  = len(wi.new_pages)
                pad   = n_pg * P - k_rem.shape[0]
                if pad > 0:
                    k_rem = F.pad(k_rem, (0, 0, 0, 0, 0, pad))
                    v_rem = F.pad(v_rem, (0, 0, 0, 0, 0, pad))
                page_t = torch.tensor(wi.new_pages, dtype=torch.int64, device=DEVICE)
                self.k_pool[layer_idx][page_t] = k_rem.view(n_pg, P, n_kv, D)
                self.v_pool[layer_idx][page_t] = v_rem.view(n_pg, P, n_kv, D)


# ─────────────────────────────────────────────────────────────────────────────
# DecodeKVCtx (unchanged from Layer 9)
# ─────────────────────────────────────────────────────────────────────────────

class DecodeKVCtx:
    """
    Context for batched decode.  Detected by hasattr(kv_cache, 'wrapper').

    last_page_indices: [B] int64 — page where each request's new token lands
    token_offsets:     [B] int64 — seq_len % page_size for each request
    """

    def __init__(
        self,
        wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        last_page_indices: torch.Tensor,
        token_offsets: torch.Tensor,
    ) -> None:
        self.wrapper           = wrapper
        self.k_pool            = k_pool
        self.v_pool            = v_pool
        self.last_page_indices = last_page_indices
        self.token_offsets     = token_offsets

    def get_seq_length(self) -> int:
        return 0

    def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
        """k_fi: [B, n_kv_heads, head_dim]"""
        self.k_pool[layer_idx][self.last_page_indices, self.token_offsets] = k_fi
        self.v_pool[layer_idx][self.last_page_indices, self.token_offsets] = v_fi


# ─────────────────────────────────────────────────────────────────────────────
# Helper: compute WriteInfo for a single request in an extend round
# ─────────────────────────────────────────────────────────────────────────────

def compute_write_info(
    kv_pool: KVPool,
    rtp: ReqToTokenPool,
    slot_indices: List[int],
    req_pool_idx: int,
    kv_committed_len: int,
    n_fill: int,
) -> WriteInfo:
    """
    Compute the WriteInfo for one request and update slot_indices / rtp in-place.

    Args:
      kv_pool:          the global KVPool (used for alloc)
      rtp:              the ReqToTokenPool (used for updating req_to_token)
      slot_indices:     the request's current list of page indices (mutated in-place)
      req_pool_idx:     the row index in rtp
      kv_committed_len: tokens already in pool (determines partial-page leftover)
      n_fill:           tokens being added this extend round

    Returns WriteInfo with page packing applied.
    Also updates slot_indices (extends) and rtp.req_to_token in-place.
    """
    P          = kv_pool.page_size
    n_leftover = kv_committed_len % P   # tokens already in last page

    existing_page:  Optional[int] = None
    existing_slots: int = 0

    # If the last page from prior chunks is partial, fill it first.
    if n_leftover > 0 and slot_indices:
        space_in_last = P - n_leftover
        existing_slots = min(space_in_last, n_fill)
        existing_page  = slot_indices[-1]

    # Remaining tokens for NEW pages.
    remaining = n_fill - existing_slots
    new_pages: List[int] = []
    if remaining > 0:
        new_pages = kv_pool.alloc(remaining)
        n_prev = len(slot_indices)
        slot_indices.extend(new_pages)
        pages_t = torch.tensor(new_pages, dtype=torch.int32, device=DEVICE)
        rtp.req_to_token[req_pool_idx, n_prev : n_prev + len(new_pages)] = pages_t

    return WriteInfo(
        existing_page  = existing_page,
        n_leftover     = n_leftover,
        existing_slots = existing_slots,
        new_pages      = new_pages,
    )
