"""
Qwen3DecoderLayer — one transformer block.

Pre-norm architecture (normalize BEFORE the sublayer, add residual AFTER):
    x = x + self_attn(input_layernorm(x))
    x = x + mlp(post_attention_layernorm(x))

Mirrors HuggingFace Qwen3DecoderLayer (modeling_qwen3.py L294-334).

Extensibility:
    SGLang's LayerCommunicator wraps the pre/post-norm + residual-add
    steps to support pipeline parallelism and fused residual-add kernels.
    Our forward() is the simpler equivalent that makes that role explicit.
"""

import torch
import torch.nn as nn

from .attention import Qwen3Attention
from .config import Qwen3Config
from .mlp import Qwen3MLP
from .norm import RMSNorm


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.self_attn             = Qwen3Attention(config, layer_idx)
        self.mlp                   = Qwen3MLP(config)
        self.input_layernorm       = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, q_len, hidden]
        cos:           torch.Tensor,   # [B, q_len, head_dim]
        sin:           torch.Tensor,   # [B, q_len, head_dim]
        forward_batch,                 # ForwardBatch
    ) -> torch.Tensor:
        # ── Self-attention sublayer ───────────────────────────────────────
        residual     = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, forward_batch)
        hidden_states = residual + hidden_states

        # ── MLP sublayer ──────────────────────────────────────────────────
        residual     = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states
