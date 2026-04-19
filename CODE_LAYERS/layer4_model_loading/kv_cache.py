"""
Layer 2 — KVCache: our own KV cache, replacing HuggingFace's DynamicCache.

Mirrors DynamicCache's interface so HuggingFace attention layers accept it
without any changes to the model internals.

The interface contract (from modeling_qwen3.py line 271):
    key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)

Each layer's attention calls .update() with the NEW token's K and V, and
expects back the FULL accumulated K and V to attend over.

Structure:
    KVCache
    └── self._layers: list[LayerCache]   one per transformer layer, lazy-init
        └── LayerCache
            ├── keys:   Tensor [batch, n_kv_heads, seq_len, head_dim]
            └── values: Tensor [batch, n_kv_heads, seq_len, head_dim]

This is intentionally simple — identical behaviour to DynamicCache/DynamicLayer.
Later layers will change this class alone to add:
    - Layer 3: pre-allocated fixed buffer (no per-step torch.cat)
    - Layer 4: paged blocks (non-contiguous memory)
    - Layer 5: prefix sharing across requests
"""

import torch


class LayerCache:
    """
    Holds the accumulated K and V tensors for a single transformer layer.
    Grows by one position each decode step via torch.cat.

    Implements the interface HuggingFace attention expects on each layer:
        update(new_k, new_v)          → (full_k, full_v)
        get_seq_length()              → int
        get_mask_sizes(query_length)  → (kv_length, kv_offset)
        is_sliding                    → False
    """

    is_sliding: bool = False   # tells HF masking: full attention, no sliding window

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None

    def update(
        self,
        new_keys: torch.Tensor,    # [batch, n_kv_heads, new_tokens, head_dim]
        new_values: torch.Tensor,  # [batch, n_kv_heads, new_tokens, head_dim]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Append new_keys/new_values to the cache along the sequence dimension
        and return the full accumulated tensors.
        """
        if self.keys is None:
            self.keys = new_keys
            self.values = new_values
        else:
            self.keys   = torch.cat([self.keys,   new_keys],   dim=-2)
            self.values = torch.cat([self.values, new_values], dim=-2)
        return self.keys, self.values

    def get_seq_length(self) -> int:
        """Current number of cached token positions."""
        return 0 if self.keys is None else self.keys.shape[-2]

    def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
        """
        Called by HuggingFace masking_utils to build the causal attention mask.
        Returns (kv_length, kv_offset):
            kv_length = total positions attention will span (cached + new query)
            kv_offset = 0 for full attention (non-sliding)
        """
        kv_length = self.get_seq_length() + query_length
        kv_offset = 0
        return kv_length, kv_offset

    @property
    def seq_len(self) -> int:
        return self.get_seq_length()

    def __repr__(self) -> str:
        if self.keys is None:
            return "LayerCache(empty)"
        return (
            f"LayerCache(seq_len={self.seq_len}, "
            f"shape={tuple(self.keys.shape)}, "
            f"dtype={self.keys.dtype}, "
            f"device={self.keys.device})"
        )


class KVCache:
    """
    Full KV cache for all transformer layers.

    Drop-in replacement for HuggingFace's DynamicCache.
    HuggingFace attention calls:
        full_k, full_v = past_key_values.update(new_k, new_v, layer_idx)

    We implement that exact method. Layers are initialised lazily on first use
    so we don't need to know the model's layer count upfront.
    """

    def __init__(self):
        self._layers: list[LayerCache] = []

    def update(
        self,
        key_states: torch.Tensor,    # new K for this layer
        value_states: torch.Tensor,  # new V for this layer
        layer_idx: int,
        *args,                        # absorb any extra positional args HF might pass
        **kwargs,                     # absorb any extra keyword args HF might pass
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Grow the list lazily as layers call in for the first time.
        while len(self._layers) <= layer_idx:
            self._layers.append(LayerCache())

        return self._layers[layer_idx].update(key_states, value_states)

    # ------------------------------------------------------------------
    # HuggingFace Cache interface — called by masking_utils and model internals
    # ------------------------------------------------------------------

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Current cached sequence length. HF calls this with a layer_idx."""
        if not self._layers or layer_idx >= len(self._layers):
            return 0
        return self._layers[layer_idx].get_seq_length()

    def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
        """Delegate mask size calculation to the appropriate layer."""
        while len(self._layers) <= layer_idx:
            self._layers.append(LayerCache())
        return self._layers[layer_idx].get_mask_sizes(query_length)

    @property
    def is_sliding(self) -> list[bool]:
        """HF uses this to decide mask type per layer."""
        return [layer.is_sliding for layer in self._layers]

    def memory_bytes(self) -> int:
        """Total GPU bytes occupied by all cached K and V tensors."""
        total = 0
        for layer in self._layers:
            if layer.keys is not None:
                total += layer.keys.nbytes + layer.values.nbytes
        return total

    def __len__(self) -> int:
        return len(self._layers)

    def __repr__(self) -> str:
        seq = self.get_seq_length()
        mem_mb = self.memory_bytes() / 1024**2
        return (
            f"KVCache(layers={len(self._layers)}, "
            f"seq_len={seq}, "
            f"memory={mem_mb:.1f} MB)"
        )
