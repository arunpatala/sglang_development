"""
KVCache — clean implementation now that we own the model.

Since we control the forward pass end-to-end, we no longer need to mirror
HuggingFace's DynamicCache interface (no get_mask_sizes, no is_sliding).
The contract is simple:

    full_k, full_v = kv_cache.update(layer_idx, new_k, new_v)

Each layer calls update() with its NEW token's K and V; receives back the
full accumulated [B, n_kv_heads, seq_len, head_dim] tensors to attend over.

The KVCache object is created in model_runner.py, passed into the model's
forward(), and mutated in-place by each attention layer.  No return value
needed — the caller holds a reference to the same object throughout the
entire generation sequence.

Future extensions (same interface, different internals):
    • Pre-allocated static buffer (no per-step torch.cat)
    • Paged / block-sparse storage (non-contiguous memory)
    • Prefix sharing across requests
"""

from __future__ import annotations

import torch


class KVCache:
    def __init__(self) -> None:
        # Dict so we don't need to know num_hidden_layers upfront.
        self._k: dict[int, torch.Tensor] = {}
        self._v: dict[int, torch.Tensor] = {}

    def update(
        self,
        layer_idx: int,
        new_k: torch.Tensor,    # [B, n_kv_heads, new_tokens, head_dim]
        new_v: torch.Tensor,    # [B, n_kv_heads, new_tokens, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_k / new_v to the cache for layer_idx and return the
        full accumulated tensors.
        """
        if layer_idx not in self._k:
            self._k[layer_idx] = new_k
            self._v[layer_idx] = new_v
        else:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=-2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=-2)
        return self._k[layer_idx], self._v[layer_idx]

    def get_seq_length(self) -> int:
        """Current number of cached token positions (same for all layers)."""
        if not self._k:
            return 0
        return next(iter(self._k.values())).shape[-2]

    def memory_bytes(self) -> int:
        return (
            sum(t.nbytes for t in self._k.values())
            + sum(t.nbytes for t in self._v.values())
        )

    def __repr__(self) -> str:
        return (
            f"KVCache(layers={len(self._k)}, "
            f"seq_len={self.get_seq_length()}, "
            f"mem={self.memory_bytes() / 1024**2:.1f} MB)"
        )
