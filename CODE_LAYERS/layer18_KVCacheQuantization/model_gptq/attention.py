"""
Qwen3Attention (GPTQ) — multi-head self-attention with int4 projections.

  ┌──────────────────────────────────────────────────────────────────────┐
  │ ForwardMode │ kv_cache       │ Kernel                               │
  │─────────────┼────────────────┼──────────────────────────────────────│
  │ EXTEND      │ ExtendKVCtx    │ FlashInfer BatchPrefillWithPagedKV   │
  │ DECODE      │ DecodeKVCtx    │ FlashInfer BatchDecodeWithPagedKV    │
  │ NOCACHE     │ None           │ F.sdpa only (reference / verify)     │
  └──────────────────────────────────────────────────────────────────────┘

Layer 13 changes vs Layer 12:
  • q_proj / k_proj / v_proj / o_proj are GPTQLinear (int4) instead of
    nn.Linear (bf16).  The attention compute and dispatch are unchanged.
  • q_norm / k_norm remain RMSNorm in bf16 (norm weights are not quantised).

All dispatch logic lives in backend.py.  This file calls one method:
    self.backend.forward(q, k, v, self.layer_idx, forward_batch)
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch  # noqa: E402

from .backend import PagedExtendBackend
from .config import Qwen3Config
from .gptq_linear import GPTQLinear
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_idx: int,
        bits: int = 4,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.layer_idx    = layer_idx
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads      # 8
        self.head_dim     = config.head_dim                 # 128

        q_dim  = self.num_heads    * self.head_dim   # 2048
        kv_dim = self.num_kv_heads * self.head_dim   # 1024

        # ── GPTQ quantised projections ────────────────────────────────────
        # bias=False: Qwen3 has attention_bias=False; GPTQLinear doesn't support bias.
        self.q_proj = GPTQLinear(config.hidden_size, q_dim,  bits, group_size)
        self.k_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
        self.v_proj = GPTQLinear(config.hidden_size, kv_dim, bits, group_size)
        self.o_proj = GPTQLinear(q_dim, config.hidden_size,  bits, group_size)

        # ── Per-head QK norms stay in bf16 (norm weights are not quantised) ──
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Backend object owns all kernel dispatch; set once at init.
        self.backend = PagedExtendBackend(config)

    def forward(
        self,
        hidden_states: torch.Tensor,   # [B, q_len, hidden]  B=1 packed for EXTEND
        cos:           torch.Tensor,   # [B, q_len, head_dim]
        sin:           torch.Tensor,   # [B, q_len, head_dim]
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape

        # ── 1. Project Q / K / V ──────────────────────────────────────────
        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        # ── 2. Per-head QK RMSNorm (Qwen3-specific) ───────────────────────
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Transpose to [B, n_heads, seq, head_dim] for attention.
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # ── 3. Rotary Position Embedding ───────────────────────────────────
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 4. Backend dispatch ────────────────────────────────────────────
        attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
        # [B, n_heads, q_len, head_dim]

        # ── 5. Merge heads and output projection ──────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)
