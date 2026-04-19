"""
Layer 8 — KV Cache: paged pool + req_to_token GPU table.

Builds on Layer 7 (KVPool + per-step context objects) by adding:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
ReqToTokenPool  (one global instance, lives for the server's lifetime)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
A 2D GPU int32 tensor  [max_batch, max_context_len]  that mirrors SGLang's
own ReqToTokenPool (mem_cache/memory_pool.py).

    req_to_token[req_pool_idx, token_pos] = physical_slot

This table is the source of truth for the Triton kernel.  Every slot index
that exists in req.slot_indices is also written here at the same time, so
the kernel can read them on-device without any Python-loop iteration.

The pool manages row indices with a free list (same pattern as KVPool):
    free_slots: List[int]  ← which rows are available
    alloc() → int           ← returns a row index, called once at prefill
    free(idx)               ← returns the row, called on request finish

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
KVPool  (unchanged from Layer 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    k_pool[layer]:  [total_slots, n_kv_heads, head_dim]   dtype=bfloat16
    v_pool[layer]:  [total_slots, n_kv_heads, head_dim]

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PrefillKVCtx  (unchanged from Layer 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Writes prompt K/V to the KVPool.  model_runner.prefill() separately writes
the slot indices to req_to_token[req_pool_idx, 0:L] before calling forward.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
DecodeKVCtx  (unchanged from Layer 7)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Carries the FlashInfer wrapper + pool references + new_slots.
model_runner.decode_step() writes new slots to req_to_token then calls
the Triton kernel to build kv_indices on-GPU before calling begin_forward.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changed from Layer 7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  Layer 7                             Layer 8
  ──────────────────────              ────────────────────────────────────
  kv_indices built in Python          Triton kernel builds kv_indices on GPU
    for req in reqs:                    one threadblock per request, parallel
      list.extend(req.slot_indices)     reads req_to_token[row, :] on-device
  torch.tensor(list) — CPU→GPU copy  NO copy of slot data CPU→GPU each step
  kv_indptr via itertools.accumulate  kv_indptr via torch.cumsum on GPU
  No req_to_token table               req_to_token pre-allocated on GPU
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import flashinfer

logger = logging.getLogger(__name__)

DEVICE = "cuda"


# ─────────────────────────────────────────────────────────────────────────────
# ReqToTokenPool — GPU-resident request → slot lookup table
# ─────────────────────────────────────────────────────────────────────────────

class ReqToTokenPool:
    """
    2D GPU int32 table: req_to_token[req_pool_idx, token_pos] = slot_idx.

    Mirrors SGLang's ReqToTokenPool (mem_cache/memory_pool.py lines 140-155).

    Allocated once at startup.  Each active request owns one row (req_pool_idx).
    The Triton kernel reads rows directly on-device to build kv_indices without
    any Python iteration or CPU→GPU copy of slot data.

    Row ownership:
      alloc()  → pop a row index (called once per request at prefill)
      free(i)  → return the row index (called when request finishes)

    Writes are done externally (in model_runner) via direct tensor indexing:
      At prefill:    req_to_token[idx, 0:L]         = slot_indices_tensor
      At each decode: req_to_token[indices, seq_lens] = new_slots_tensor   (batch)
    """

    def __init__(self, max_batch: int, max_context_len: int) -> None:
        self.max_batch       = max_batch
        self.max_context_len = max_context_len

        # Shape: [max_batch, max_context_len]  int32  on GPU
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
        """Return a free row index.  Raises if all rows are in use."""
        if not self.free_slots:
            raise RuntimeError("ReqToTokenPool exhausted — too many concurrent requests")
        return self.free_slots.pop()

    def free(self, idx: int) -> None:
        """Return row `idx` to the free list."""
        self.free_slots.append(idx)


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
