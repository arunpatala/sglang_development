"""
Layer 5 — KV Cache: two classes for two different purposes.

PerReqKVCache
─────────────
Stores the accumulated K and V tensors for a single request across all
decoder layers.  Identical to layer4's KVCache but renamed for clarity.

  Shape per layer: [1, n_kv_heads, seq_len, head_dim]

  Lives for the entire lifetime of a request: created at prefill,
  updated one token at a time during decode.  Discarded when the
  request finishes.

BatchedKVCache
──────────────
A temporary view built at the start of each decode step.

  Problem: different running requests have different KV lengths.
           F.sdpa requires a rectangular [B, heads, kv_len, dim] tensor.

  Solution (Option 1 — pad to max KV length):
    1. Find max_kv_len across all active requests.
    2. Left-pad each request's per-layer K/V to max_kv_len with zeros.
    3. Stack them into [B, heads, max_kv_len, dim].
    4. Build attention_mask [B, max_kv_len+1] with 0s on the left for
       padding positions — _build_additive_mask will mask those to -inf.
    5. After the forward pass, extract the new token's K/V (last position)
       from the stacked tensor and append it to each PerReqKVCache.

  Initialisation is lazy per layer (in update()) to avoid iterating all
  28 layers upfront — the model calls update() in order during the forward
  pass anyway.

  The trade-off: O(max_kv_len) compute wasted on padding for short
  requests.  This is acceptable for Layer 5 and is eliminated in
  Layer 6 with paged KV cache + FlashInfer ragged kernels.
"""

import torch
import torch.nn.functional as F
from typing import Dict, List, Set, Tuple


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


class BatchedKVCache:
    """
    Temporary KV cache for one batched decode step.

    Constructed from a list of PerReqKVCache objects, pads each to
    max_kv_len so F.sdpa can process them in one rectangular batch.
    After the forward pass, write_back() appends just the new token's
    K/V to each request's PerReqKVCache.
    """

    def __init__(self, reqs: list, max_kv_len: int) -> None:
        self._reqs       = reqs
        self._max_kv_len = max_kv_len
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}
        self._initialized: Set[int] = set()

    def _init_layer(self, layer_idx: int) -> None:
        """
        Pad each request's historical K/V to max_kv_len and stack into [B, ...].
        Called lazily on the first update() for each layer.
        """
        ks, vs = [], []
        for req in self._reqs:
            rk = req.kv_cache._k[layer_idx]   # [1, n_kv, kv_len_i, dim]
            rv = req.kv_cache._v[layer_idx]
            pad = self._max_kv_len - rk.shape[2]
            if pad > 0:
                # Left-pad the sequence dimension.
                # F.pad args are reverse-ordered: (..., dim-2_left, dim-2_right, dim-1_left, dim-1_right)
                # We want to pad dim=-2 (seq) on the left → (0, 0, pad, 0)
                rk = F.pad(rk, (0, 0, pad, 0))
                rv = F.pad(rv, (0, 0, pad, 0))
            ks.append(rk)
            vs.append(rv)
        # [B, n_kv, max_kv_len, dim]
        self._k[layer_idx] = torch.cat(ks, dim=0)
        self._v[layer_idx] = torch.cat(vs, dim=0)
        self._initialized.add(layer_idx)

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,    # [B, n_kv, 1, head_dim]
        new_v: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Called by each attention layer during the forward pass.
        Returns [B, n_kv, max_kv_len+1, head_dim] — historical + new token.
        """
        if layer_idx not in self._initialized:
            self._init_layer(layer_idx)

        self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
        self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        return self._k[layer_idx], self._v[layer_idx]

    def get_seq_length(self) -> int:
        """Returns max_kv_len — used by the model to determine past_len."""
        return self._max_kv_len

    def write_back(self) -> None:
        """
        After the forward pass: extract the new token's K/V from the last
        position of each stacked tensor and append to per-request caches.

        Before: req.kv_cache has [1, n_kv, kv_len_i, dim]
        After:  req.kv_cache has [1, n_kv, kv_len_i+1, dim]
        """
        for layer_idx in self._initialized:
            full_k = self._k[layer_idx]   # [B, n_kv, max_kv_len+1, dim]
            full_v = self._v[layer_idx]
            for i, req in enumerate(self._reqs):
                new_k = full_k[i : i + 1, :, -1:, :]   # [1, n_kv, 1, dim]
                new_v = full_v[i : i + 1, :, -1:, :]
                cache = req.kv_cache
                if layer_idx in cache._k:
                    cache._k[layer_idx] = torch.cat([cache._k[layer_idx], new_k], dim=2)
                    cache._v[layer_idx] = torch.cat([cache._v[layer_idx], new_v], dim=2)
                else:
                    cache._k[layer_idx] = new_k
                    cache._v[layer_idx] = new_v
