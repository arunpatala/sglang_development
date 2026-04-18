"""
Layer 6 — KV Cache: PerReqKVCache (unchanged) + PackedKVCache (new).

PerReqKVCache
─────────────
Stores the accumulated K and V tensors for a single request across all
decoder layers.  Identical to layer5.

  Shape per layer: [1, n_kv_heads, seq_len, head_dim]

  Lives for the entire lifetime of a request: created at prefill,
  updated one token at a time during decode.  Discarded when the
  request finishes.

PackedKVCache                                         ← NEW (replaces BatchedKVCache)
─────────────
A temporary object built at the start of each decode step.

  Problem: Layer 5 solution (BatchedKVCache) left-pads all per-request
           KV caches to max_kv_len and processes a rectangular
           [B, heads, max_kv_len, dim] tensor.  This wastes compute on
           padding positions inside the attention kernel.

  Solution (Layer 6 — ragged packing):
    1. Concatenate all per-request historical KVs into a single
       contiguous tensor (no padding) with an indptr array that marks
       the start/end of each request's slice.
    2. At each attention layer, append the new decode token's K/V for
       each request (one token total, interleaved with its history).
    3. Call FlashInfer's BatchPrefillWithRaggedKVCacheWrapper with
       q_len=1 per request — FlashInfer only computes attention over
       real tokens, skipping the "empty space" that padding would have
       filled.
    4. write_back() appends the new token's K/V to each PerReqKVCache.

  Trade-off still present in Layer 6:
    Copy cost — building pack_k/pack_v at each decode step requires
    gathering all per-request K/V tensors into a new contiguous buffer.
    This O(total_kv_tokens) copy happens every decode step even though
    only 1 new token per request was added.

    Layer 7 (paged KV cache) eliminates this copy by keeping K/V in a
    pre-allocated block table that FlashInfer can read directly without
    any gather step.

  Diagram for B=3, kv_lens=[10, 6, 4]:

    Layer 5 (padded):         Layer 6 (packed):
    ┌────────────────┐         ┌──────────────┐
    │ req0 [pad][10] │         │ req0 [10+1]  │ ← kv_indptr[0:2] = 0, 11
    │ req1 [pad] [6] │         │ req1 [ 6+1]  │ ← kv_indptr[1:3] = 11, 18
    │ req2 [pad]  [4]│         │ req2 [ 4+1]  │ ← kv_indptr[2:4] = 18, 23
    └────────────────┘         └──────────────┘
    shape: [3, kv, 10, d]      shape: [23, kv, d]  (no wasted columns)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from itertools import accumulate
from typing import Dict, List, Tuple

import flashinfer

DEVICE = "cuda"

# ─────────────────────────────────────────────────────────────────────────────
# PerReqKVCache — unchanged from Layer 5
# ─────────────────────────────────────────────────────────────────────────────

class PerReqKVCache:
    """Per-request KV cache.  Grows by one token per decode step."""

    def __init__(self) -> None:
        self._k: Dict[int, torch.Tensor] = {}   # layer_idx → [1, n_kv, seq, dim]
        self._v: Dict[int, torch.Tensor] = {}

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,    # [1, n_kv, q_len, head_dim]
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Append new_k/new_v and return the full accumulated cache."""
        if layer_idx in self._k:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        else:
            self._k[layer_idx] = new_k
            self._v[layer_idx] = new_v
        return self._k[layer_idx], self._v[layer_idx]

    def get_seq_length(self) -> int:
        if not self._k:
            return 0
        return next(iter(self._k.values())).shape[2]

    def memory_bytes(self) -> int:
        return sum(t.nbytes for d in (self._k, self._v) for t in d.values())


# ─────────────────────────────────────────────────────────────────────────────
# PackedKVCache — replaces BatchedKVCache
# ─────────────────────────────────────────────────────────────────────────────

class PackedKVCache:
    """
    Temporary KV cache for one batched decode step using ragged packing.

    Constructed from a list of Req objects (each with a populated
    PerReqKVCache).  Packs historical K/V into contiguous ragged tensors
    so FlashInfer can attend over real tokens only, with no padding waste.

    Lifecycle (called once per decode step):
        1. __init__: compute indptr, create FlashInfer wrapper.
        2. plan(): call begin_forward once with shape metadata.
        3. forward_attn(): called by each attention layer (28 times).
           Packs historical KVs + new decode token → calls FlashInfer.
        4. write_back(): appends new token to each PerReqKVCache.
    """

    def __init__(self, reqs: list, workspace: torch.Tensor) -> None:
        self._reqs = reqs
        B = len(reqs)

        kv_lens = [r.kv_cache.get_seq_length() for r in reqs]

        # qo_indptr: each request contributes exactly 1 query token (decode).
        # Shape [B+1]: [0, 1, 2, ..., B]
        self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEVICE)

        # kv_indptr: cumulative sum of (historical_kv_len_i + 1).
        # The +1 accounts for the new decode token that we append for each req
        # before calling FlashInfer, so req i's attention spans
        # kv_indptr[i] .. kv_indptr[i+1] (inclusive of the new token).
        kv_full_lens = [l + 1 for l in kv_lens]   # each req: history + new tok
        kv_cumsum = [0] + list(accumulate(kv_full_lens))
        self.kv_indptr = torch.tensor(kv_cumsum, dtype=torch.int32, device=DEVICE)

        # FlashInfer wrapper — one workspace reused across all layers.
        self._wrapper = flashinfer.BatchPrefillWithRaggedKVCacheWrapper(
            workspace, kv_layout="NHD"
        )

        # Storage for new K/V tokens (saved in forward_attn, consumed in write_back).
        self._new_k: Dict[int, torch.Tensor] = {}  # layer_idx → [B, n_kv, head_dim]
        self._new_v: Dict[int, torch.Tensor] = {}

    def plan(
        self,
        num_q_heads: int,
        num_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        """
        Call begin_forward once with shape metadata.

        FlashInfer uses this to select the right kernel and allocate
        any internal temp buffers inside the workspace.  The same plan
        is reused for all 28 attention layers in this decode step
        (indptrs + shapes are identical across layers).
        """
        self._wrapper.begin_forward(
            self.qo_indptr,
            self.kv_indptr,
            num_q_heads,
            num_kv_heads,
            head_dim,
            causal=False,           # q_len=1 per req, no future tokens to mask
            q_data_type=dtype,
        )

    @property
    def wrapper(self):
        """
        Expose the FlashInfer wrapper so attention.py can call wrapper.forward()
        directly.  The attention layer owns the attention computation; this cache
        object only owns the data layout.
        """
        return self._wrapper

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,   # [B, n_kv_heads, head_dim] — new decode token, RoPE'd
        new_v: torch.Tensor,   # [B, n_kv_heads, head_dim]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack historical KV + new decode token for each request into a single
        contiguous ragged tensor.  Returns (k_packed, v_packed) for the caller
        (attention.py) to pass directly to the FlashInfer wrapper.

        Shape returned: [total_kv_tokens, n_kv_heads, head_dim]  (NHD layout)

        Packing layout for req i:
            [ hist_k_i[0..L_i-1] | new_k_i ]
        All requests concatenated back-to-back.
        kv_indptr (set at __init__) tells FlashInfer which slice is which.

        Also saves new_k/v per layer for write_back() to append to each
        request's PerReqKVCache after the full 28-layer forward pass.
        """
        segs_k: List[torch.Tensor] = []
        segs_v: List[torch.Tensor] = []

        for i, req in enumerate(self._reqs):
            # Historical KV for this layer: [1, n_kv, L_i, head_dim]
            hist_k = req.kv_cache._k[layer_idx]
            hist_v = req.kv_cache._v[layer_idx]

            # [1, n_kv, L_i, dim] → [L_i, n_kv, dim]  (NHD layout)
            hist_k_nhd = hist_k.squeeze(0).permute(1, 0, 2).contiguous()
            hist_v_nhd = hist_v.squeeze(0).permute(1, 0, 2).contiguous()

            segs_k.append(hist_k_nhd)
            segs_k.append(new_k[i].unsqueeze(0))   # [1, n_kv, dim]
            segs_v.append(hist_v_nhd)
            segs_v.append(new_v[i].unsqueeze(0))

        # Save new tokens for write_back (consumed after the forward pass).
        self._new_k[layer_idx] = new_k   # [B, n_kv, dim]
        self._new_v[layer_idx] = new_v

        return torch.cat(segs_k, dim=0), torch.cat(segs_v, dim=0)   # [total_kv, n_kv, dim]

    def get_seq_length(self) -> int:
        """
        Returns 0: PackedKVCache is decode-only and passes explicit position_ids,
        so the model's past_len offset is not used for RoPE or mask construction.
        """
        return 0

    def write_back(self) -> None:
        """
        After the forward pass: append the new decode token's K/V to each
        request's PerReqKVCache so the next decode step sees the updated history.

        new_k stored in forward_attn: [B, n_kv_heads, head_dim]
        We convert back to [1, n_kv_heads, 1, head_dim] to match PerReqKVCache format.
        """
        for layer_idx, new_k in self._new_k.items():
            new_v = self._new_v[layer_idx]
            for i, req in enumerate(self._reqs):
                # [n_kv, dim] → [1, n_kv, 1, dim]
                k_tok = new_k[i].unsqueeze(0).unsqueeze(2)
                v_tok = new_v[i].unsqueeze(0).unsqueeze(2)
                cache = req.kv_cache
                if layer_idx in cache._k:
                    cache._k[layer_idx] = torch.cat([cache._k[layer_idx], k_tok], dim=2)
                    cache._v[layer_idx] = torch.cat([cache._v[layer_idx], v_tok], dim=2)
                else:
                    cache._k[layer_idx] = k_tok
                    cache._v[layer_idx] = v_tok

    def end_forward(self) -> None:
        """Call after the decode step to release FlashInfer's internal state."""
        self._wrapper.end_forward()
