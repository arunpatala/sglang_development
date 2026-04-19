"""
Qwen3Config — plain dataclass parsed from config.json.

Why not inherit PretrainedConfig?
  HuggingFace's PretrainedConfig adds ~1000 lines of serialisation, hub
  downloading, and legacy-compat machinery.  We only need the numeric
  hyperparameters; a dataclass is enough.

Extensibility hook (SGLang-style):
  SGLang's model files do `Qwen3Config = None` and import it from
  transformers.  Our version owns the dataclass so we can add fields
  (e.g. quantisation config, tensor-parallel degree) without touching HF.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path


class AttnBackend(Enum):
    SDPA       = "sdpa"        # F.scaled_dot_product_attention (prefill + decode)
    FLASHINFER = "flashinfer"  # FlashInfer ragged kernel for decode, F.sdpa for prefill


@dataclass
class Qwen3Config:
    # Vocabulary / embedding
    vocab_size: int = 151_936

    # Transformer dimensions
    hidden_size: int = 1_024
    num_hidden_layers: int = 28
    intermediate_size: int = 3_072

    # Attention
    num_attention_heads: int = 16   # Q heads
    num_key_value_heads: int = 8    # KV heads (GQA: each KV head shared by 2 Q heads)
    head_dim: int = 128             # per-head dimension

    # Norms
    rms_norm_eps: float = 1e-6

    # Positional encoding
    rope_theta: float = 1_000_000.0

    # Misc
    attention_bias: bool = False
    tie_word_embeddings: bool = True
    hidden_act: str = "silu"

    # Attention backend — set by ModelRunner, not read from config.json.
    # SDPA: F.scaled_dot_product_attention for all paths (default, no deps).
    # FLASHINFER: ragged packed attention for decode; F.sdpa for prefill.
    attn_backend: AttnBackend = AttnBackend.SDPA

    # ------------------------------------------------------------------ #
    # Derived properties
    # ------------------------------------------------------------------ #

    @property
    def num_kv_groups(self) -> int:
        """How many Q heads share each KV head."""
        return self.num_attention_heads // self.num_key_value_heads

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #

    @classmethod
    def from_json(cls, path: str | Path) -> "Qwen3Config":
        """Read config.json and populate the dataclass."""
        with open(path) as f:
            d = json.load(f)

        return cls(
            vocab_size=d.get("vocab_size", 151_936),
            hidden_size=d.get("hidden_size", 1_024),
            num_hidden_layers=d.get("num_hidden_layers", 28),
            intermediate_size=d.get("intermediate_size", 3_072),
            num_attention_heads=d.get("num_attention_heads", 16),
            num_key_value_heads=d.get("num_key_value_heads", 8),
            head_dim=d.get("head_dim", 128),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 1_000_000.0),
            attention_bias=d.get("attention_bias", False),
            tie_word_embeddings=d.get("tie_word_embeddings", True),
            hidden_act=d.get("hidden_act", "silu"),
        )
