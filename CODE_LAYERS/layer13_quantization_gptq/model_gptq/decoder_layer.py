"""
Qwen3DecoderLayer (GPTQ) — one transformer block with quantised projections.

Pre-norm architecture (normalize BEFORE the sublayer, add residual AFTER):
    x = x + self_attn(input_layernorm(x))
    x = x + mlp(post_attention_layernorm(x))

Layer 13: passes bits and group_size down to Qwen3Attention and Qwen3MLP
so they construct GPTQLinear instead of nn.Linear.

Layer 13 changes vs Layer 12:
  • (attention_mask, kv_cache) replaced by forward_batch: ForwardBatch.
    The attention backend selects the right kernel from forward_batch.mode;
    the kv_cache is read from forward_batch.kv_cache.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch  # noqa: E402

from .attention import Qwen3Attention
from .config import Qwen3Config
from .mlp import Qwen3MLP
from .norm import RMSNorm


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.self_attn                = Qwen3Attention(config, layer_idx, bits, group_size)
        self.mlp                      = Qwen3MLP(config, bits, group_size)
        self.input_layernorm          = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, q_len, hidden]
        cos:           torch.Tensor,   # [B, q_len, head_dim]
        sin:           torch.Tensor,   # [B, q_len, head_dim]
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # ── Self-attention sublayer ───────────────────────────────────────
        residual      = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, forward_batch)
        hidden_states = residual + hidden_states

        # ── MLP sublayer ──────────────────────────────────────────────────
        residual      = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
