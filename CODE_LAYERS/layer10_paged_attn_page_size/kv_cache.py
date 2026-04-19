"""
Layer 9 — KV Cache: variable page_size support.

Extends Layer 8 by generalising the KVPool from page_size=1 (one slot per
token) to page_size=P (P tokens packed into each pool entry).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changes from Layer 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 8 (page_size=1)               Layer 9 (page_size=P)
  ─────────────────────────────────   ─────────────────────────────────────
  KVPool shape: [slots, n_kv, D]      KVPool shape: [pages, P, n_kv, D]
  alloc(n_tokens) → n_tokens slots    alloc(n_tokens) → ceil(n/P) pages
  kv_indices = one int per token      kv_indices = one int per page
  kv_last_page_lens = all ones        kv_last_page_lens = 1..P (variable)
  New slot per decode step always     New page only when page fills up
  pool[layer][slot_t] = k_token       pool[layer][page, offset] = k_token

  req_to_token col axis = token pos   req_to_token col axis = page pos
  req_to_token[idx, pos] = slot       req_to_token[idx, page] = page_idx

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReqToTokenPool (unchanged API, different semantics)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
With page_size=P the second axis stores *page* indices (not token indices):
    req_to_token[req_pool_idx, page_pos] = page_idx

max_context_len now means max pages, not max tokens.
  max_pages = ceil(max_token_context / page_size)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KVPool
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    k_pool[layer]:  [total_pages, page_size, n_kv_heads, head_dim]
    free_slots:     list of page indices  (page 0 = padding, real pages start at 1)

  alloc(n_tokens) → ceil(n_tokens / page_size) page indices
  free(page_indices) → returns those indices to the free list

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PrefillKVCtx.store()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  k shape from attention: [1, n_kv, prompt_len, head_dim]
  1. Permute → [prompt_len, n_kv, head_dim]
  2. Pad to multiple of page_size (zeros in padding slots — never read
     because kv_last_page_lens tells FlashInfer the true fill level)
  3. View → [num_pages, page_size, n_kv, head_dim]
  4. Scatter-write: k_pool[layer][page_indices] = k_paged

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DecodeKVCtx.store()
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  k_fi shape: [B, n_kv, head_dim]  (one new token per request)
  Write: k_pool[layer][last_page_indices, token_offsets] = k_fi
    last_page_indices: [B] — page where the new token lands
    token_offsets:     [B] — position within that page (seq_len % page_size)
"""

from __future__ import annotations

import logging
import math
from typing import List

import torch
import torch.nn.functional as F
import flashinfer

logger = logging.getLogger(__name__)

DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# ReqToTokenPool — GPU-resident request → page lookup table
# ─────────────────────────────────────────────────────────────────────────────

class ReqToTokenPool:
    """
    2D GPU int32 table storing page indices per request:
        req_to_token[req_pool_idx, page_pos] = page_idx

    With page_size=1 this is identical to Layer 8 (page_pos == token_pos,
    page_idx == slot_idx).  With page_size=P, each column is a page index
    and there are ceil(seq_len / P) columns per request.

    max_context_len here means max *pages* per request.
    """

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
            raise RuntimeError("ReqToTokenPool exhausted — too many concurrent requests")
        return self.free_slots.pop()

    def free(self, idx: int) -> None:
        self.free_slots.append(idx)


# ─────────────────────────────────────────────────────────────────────────────
# KVPool — pre-allocated paged token store
# ─────────────────────────────────────────────────────────────────────────────

class KVPool:
    """
    Pre-allocated GPU KV store with configurable page_size.

    Shape per layer:  [total_pages, page_size, n_kv_heads, head_dim]
    Page 0 is reserved as a zero-filled padding page (FlashInfer convention).
    Real pages start at index 1.

    alloc(n_tokens) pops ceil(n_tokens / page_size) pages from free_slots.
    free(page_indices) pushes them back.
    """

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

        # Shape: [total_pages, page_size, n_kv_heads, head_dim]
        # Page 0 left as zeros — used as dummy padding by FlashInfer.
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

        # Page 0 reserved; real pages start at 1.
        self.free_slots: List[int] = list(range(1, total_pages))

        mem_gb = (
            n_layers * 2 * total_pages * page_size * n_kv_heads * head_dim
            * torch.finfo(dtype).bits // 8
        ) / (1024 ** 3)
        logger.info(
            f"KVPool: {total_pages} pages × page_size={page_size} × {n_layers} layers "
            f"= {mem_gb:.2f} GB  (n_kv={n_kv_heads}, dim={head_dim})"
        )

    def available(self) -> int:
        return len(self.free_slots)

    def n_pages_for(self, n_tokens: int) -> int:
        """Number of pages needed to hold n_tokens."""
        return math.ceil(n_tokens / self.page_size)

    def alloc(self, n_tokens: int) -> List[int]:
        """
        Allocate ceil(n_tokens / page_size) pages.
        Returns a list of page indices.
        """
        n_pages = self.n_pages_for(n_tokens)
        if n_pages > len(self.free_slots):
            raise RuntimeError(
                f"KVPool OOM: need {n_pages} pages for {n_tokens} tokens, "
                f"only {len(self.free_slots)} pages free"
            )
        pages = self.free_slots[:n_pages]
        self.free_slots = self.free_slots[n_pages:]
        return pages

    def free(self, page_indices: List[int]) -> None:
        """Return page indices to the free list."""
        self.free_slots.extend(page_indices)


# ─────────────────────────────────────────────────────────────────────────────
# PrefillKVCtx — passed to model() during B=1 prefill
# ─────────────────────────────────────────────────────────────────────────────

class PrefillKVCtx:
    """
    Context for prefill.  Detected by hasattr(kv_cache, 'prefill_slots').

    page_indices: list of page indices allocated for this request's prompt.
    store() writes the prompt K/V into the paged pool.
    Attention itself uses F.sdpa over the fresh K/V — pool write is a side-effect.
    """

    def __init__(
        self,
        page_indices: List[int],
        kv_pool: KVPool,
    ) -> None:
        self.prefill_slots = page_indices   # duck-typing marker
        self._kv_pool      = kv_pool
        self._page_t       = torch.tensor(page_indices, dtype=torch.int64, device=DEVICE)
        self._n_pages      = len(page_indices)

    def get_seq_length(self) -> int:
        return 0

    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Write prompt K/V into the paged pool.

        k shape from attention: [1, n_kv_heads, prompt_len, head_dim]

        Steps:
          1. [1, n_kv, L, D] → [L, n_kv, D]
          2. Pad L to n_pages * page_size (zero-fill unused slots in last page)
          3. View → [n_pages, page_size, n_kv, D]
          4. Scatter into pool: k_pool[layer][page_indices] = k_paged
        """
        P   = self._kv_pool.page_size
        L   = k.shape[2]

        k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()   # [L, n_kv, D]
        v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()

        pad = self._n_pages * P - L
        if pad > 0:
            k_nhd = F.pad(k_nhd, (0, 0, 0, 0, 0, pad))
            v_nhd = F.pad(v_nhd, (0, 0, 0, 0, 0, pad))

        n_kv = self._kv_pool.n_kv_heads
        D    = self._kv_pool.head_dim
        k_paged = k_nhd.view(self._n_pages, P, n_kv, D)
        v_paged = v_nhd.view(self._n_pages, P, n_kv, D)

        self._kv_pool.k_pool[layer_idx][self._page_t] = k_paged
        self._kv_pool.v_pool[layer_idx][self._page_t] = v_paged


# ─────────────────────────────────────────────────────────────────────────────
# DecodeKVCtx — passed to model() during batched decode
# ─────────────────────────────────────────────────────────────────────────────

class DecodeKVCtx:
    """
    Context for batched decode.  Detected by hasattr(kv_cache, 'wrapper').

    last_page_indices: [B] int64 — the page where each request's new token lands
                         (either a freshly allocated page or the existing last page)
    token_offsets:     [B] int64 — seq_len % page_size for each request
                         (position within that page)

    store() writes k_fi/v_fi into pool[layer][last_page_indices, token_offsets].
    wrapper.forward() then reads the full KV history via kv_indices.
    """

    def __init__(
        self,
        wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        last_page_indices: torch.Tensor,   # [B] int64
        token_offsets: torch.Tensor,       # [B] int64
    ) -> None:
        self.wrapper           = wrapper   # duck-typing marker
        self.k_pool            = k_pool
        self.v_pool            = v_pool
        self.last_page_indices = last_page_indices
        self.token_offsets     = token_offsets

    def get_seq_length(self) -> int:
        return 0

    def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
        """
        Write the new decode token's K/V into the pool for each request.

        k_fi shape: [B, n_kv_heads, head_dim]

        Advanced indexing: k_pool[layer][last_page_indices[i], token_offsets[i]]
        selects the correct (page, within-page offset) for each request i.
        """
        self.k_pool[layer_idx][self.last_page_indices, self.token_offsets] = k_fi
        self.v_pool[layer_idx][self.last_page_indices, self.token_offsets] = v_fi
