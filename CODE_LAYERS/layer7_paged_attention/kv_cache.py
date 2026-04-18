"""
Layer 7 — KV Cache: pre-allocated global pool + per-step context objects.

Replaces Layer 6's PerReqKVCache + PackedKVCache entirely.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KVPool  (one global instance, lives for the server's lifetime)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pre-allocates two flat tensors per layer — the K pool and V pool:

    k_pool[layer]:  [total_slots, n_kv_heads, head_dim]   dtype=bfloat16
    v_pool[layer]:  [total_slots, n_kv_heads, head_dim]

total_slots = how many tokens the server can hold simultaneously across all
requests. Calculated from available GPU memory at startup (see model_runner).

Each row (slot) belongs to exactly one token. slot 0 is intentionally left
as padding (FlashInfer convention).

The free-slot list tracks which rows are available:
    free_slots: List[int]   # all rows initially

When a request prefills prompt_len tokens: alloc(prompt_len) → pop that many
When a decode step generates 1 new token per request: alloc(1) each
When a request finishes: free(req.slot_indices) → push all rows back

No copy, no gather — FlashInfer reads k_pool/v_pool directly via kv_indices.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PrefillKVCtx  (built once per prefill call in model_runner.prefill)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Passed as kv_cache= to model(). Detected in attention.py by hasattr 'prefill_slots'.

During the model forward pass, each attention layer calls:
    ctx.store(layer_idx, k, v)   →  writes K/V to allocated pool slots
                                     (k/v are [1, n_kv, prompt_len, head_dim])

Attention itself is still F.sdpa over the freshly computed K/V — the pool
write is a side-effect (storing for future decode steps).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DecodeKVCtx  (built once per decode step in model_runner.decode_step)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Passed as kv_cache= to model(). Detected in attention.py by hasattr 'wrapper'.

The FlashInfer BatchDecodeWithPagedKVCacheWrapper is planned once with:
    kv_indptr        [B+1]         — cumsum of (kv_len_i + 1)
    kv_indices       [total_kvs]   — flat list of all pool slots per request
                                     (historical slots + new slot at end)
    kv_last_page_lens [B]           — always 1 for page_size=1

During the model forward pass, each attention layer calls:
    ctx.store(layer_idx, k_fi, v_fi)  →  writes new token K/V to new_slots
    wrapper.forward(q, (k_pool[layer].unsqueeze(1), v_pool[layer].unsqueeze(1)))
                                      →  FlashInfer reads from pool via kv_indices

No gather, no copy of float KV data. FlashInfer indexes directly into the
pre-allocated pool tensors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Comparison to Layer 6
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 6 (PackedKVCache):           Layer 7 (KVPool):
  ─────────────────────────          ──────────────────
  PerReqKVCache grows each step      Fixed-size pool, slots assigned
  Gather all into packed buffer      kv_indices only (integers, no KV copy)
  Copy cost: O(total_kv_tokens)      Copy cost: O(0) — pool never moved
  FlashInfer ragged prefill API      FlashInfer paged decode API
"""

from __future__ import annotations

import logging
from itertools import accumulate
from typing import Dict, List, Optional, Tuple

import torch
import flashinfer

logger = logging.getLogger(__name__)

DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# KVPool — the global pre-allocated token store
# ─────────────────────────────────────────────────────────────────────────────

class KVPool:
    """
    Pre-allocated flat GPU token store.  Slot 0 is reserved as a dummy
    padding slot (FlashInfer convention); real tokens start at slot 1.
    """

    def __init__(
        self,
        total_slots: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.total_slots = total_slots
        self.n_layers    = n_layers
        self.n_kv_heads  = n_kv_heads
        self.head_dim    = head_dim
        self.dtype       = dtype

        # One flat tensor per layer for K and V.
        # Shape: [total_slots, n_kv_heads, head_dim]
        # Slot 0 is left as zeros — used as dummy by FlashInfer padding.
        self.k_pool: List[torch.Tensor] = [
            torch.zeros(total_slots, n_kv_heads, head_dim, dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]
        self.v_pool: List[torch.Tensor] = [
            torch.zeros(total_slots, n_kv_heads, head_dim, dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]

        # Slot 0 reserved; real slots start at 1.
        self.free_slots: List[int] = list(range(1, total_slots))

        mem_gb = (
            n_layers * 2 * total_slots * n_kv_heads * head_dim
            * torch.finfo(dtype).bits // 8
        ) / (1024 ** 3)
        logger.info(
            f"KVPool: {total_slots} slots × {n_layers} layers "
            f"= {mem_gb:.2f} GB  (n_kv={n_kv_heads}, dim={head_dim})"
        )

    def available(self) -> int:
        return len(self.free_slots)

    def alloc(self, n: int) -> List[int]:
        """Pop n slots from the free list.  Raises if pool is exhausted."""
        if n > len(self.free_slots):
            raise RuntimeError(
                f"KVPool OOM: need {n} slots, only {len(self.free_slots)} free"
            )
        slots = self.free_slots[:n]
        self.free_slots = self.free_slots[n:]
        return slots

    def free(self, slots: List[int]) -> None:
        """Return slots to the free list."""
        self.free_slots.extend(slots)


# ─────────────────────────────────────────────────────────────────────────────
# PrefillKVCtx — passed to model() during B=1 prefill
# ─────────────────────────────────────────────────────────────────────────────

class PrefillKVCtx:
    """
    Context object for prefill.  Detected by hasattr(kv_cache, 'prefill_slots').

    Carries the pool slot indices for this request's prompt tokens.
    attention.py calls store() after computing K/V to persist them to the pool.
    Attention itself (F.sdpa) runs over the freshly computed K/V, not the pool.
    """

    def __init__(self, slot_indices: List[int], kv_pool: KVPool) -> None:
        # Duck-typing marker recognised by attention.py
        self.prefill_slots = slot_indices

        self._kv_pool = kv_pool
        # Tensor version for fancy-indexed pool writes (avoid repeated conversion)
        self._slot_t  = torch.tensor(slot_indices, dtype=torch.int64, device=DEVICE)

    def get_seq_length(self) -> int:
        """No past KV during prefill; position_ids are passed explicitly."""
        return 0

    def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
        """
        Write prompt K/V into the global pool.

        k shape: [1, n_kv_heads, prompt_len, head_dim]  (from attention.py)
        Pool row: [n_kv_heads, head_dim]  per slot

        Convert k → [prompt_len, n_kv_heads, head_dim] then scatter into slots.
        """
        # [1, n_kv, L, D] → [L, n_kv, D]
        k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()
        v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()
        self._kv_pool.k_pool[layer_idx][self._slot_t] = k_nhd
        self._kv_pool.v_pool[layer_idx][self._slot_t] = v_nhd


# ─────────────────────────────────────────────────────────────────────────────
# DecodeKVCtx — passed to model() during batched decode
# ─────────────────────────────────────────────────────────────────────────────

class DecodeKVCtx:
    """
    Context object for batched decode.  Detected by hasattr(kv_cache, 'wrapper').

    Carries:
      wrapper    — FlashInfer BatchDecodeWithPagedKVCacheWrapper, already planned
      k_pool     — reference to the global K pool (FlashInfer reads it directly)
      v_pool     — reference to the global V pool
      new_slots  — [B] tensor of fresh slot indices (one per request)
                   each attention layer writes the new decode token here

    attention.py calls store() to write the new token, then calls
    wrapper.forward() which reads the entire pool slice for each request
    via kv_indices (set during begin_forward in model_runner).
    """

    def __init__(
        self,
        wrapper: flashinfer.BatchDecodeWithPagedKVCacheWrapper,
        k_pool: List[torch.Tensor],
        v_pool: List[torch.Tensor],
        new_slots: torch.Tensor,       # [B] int64
    ) -> None:
        # Duck-typing marker recognised by attention.py
        self.wrapper   = wrapper
        self.k_pool    = k_pool
        self.v_pool    = v_pool
        self.new_slots = new_slots     # [B]

    def get_seq_length(self) -> int:
        """position_ids are passed explicitly; past_len not needed."""
        return 0

    def store(self, layer_idx: int, k_fi: torch.Tensor, v_fi: torch.Tensor) -> None:
        """
        Write the new decode token's K/V into the pool for each request.

        k_fi shape: [B, n_kv_heads, head_dim]  (squeezed q_len=1 dim)
        """
        self.k_pool[layer_idx][self.new_slots] = k_fi
        self.v_pool[layer_idx][self.new_slots] = v_fi
