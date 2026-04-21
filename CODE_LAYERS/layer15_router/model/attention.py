"""
Qwen3Attention — multi-head self-attention with:
  • Grouped Query Attention (GQA): 16 Q heads share 8 KV heads
  • Per-head QK RMSNorm (Qwen3-specific, not in Llama/Qwen2)
  • Rotary Position Embedding applied after QK norm
  • Backend-object dispatch (SGLang style):

    ┌──────────────────────────────────────────────────────────────────────┐
    │ ForwardMode │ kv_cache       │ Kernel                               │
    │─────────────┼────────────────┼──────────────────────────────────────│
    │ EXTEND      │ ExtendKVCtx    │ FlashInfer BatchPrefillWithPagedKV   │
    │ DECODE      │ DecodeKVCtx    │ FlashInfer BatchDecodeWithPagedKV    │
    │ NOCACHE     │ None           │ F.sdpa only (reference / verify)     │
    └──────────────────────────────────────────────────────────────────────┘

  All dispatch logic lives in backend.py.  This file calls one method:
      self.backend.forward(q, k, v, self.layer_idx, forward_batch)
  Adding a new backend means writing a new class in backend.py — this file
  never changes again.

Layer 11 changes vs Layer 10:
  • Inline hasattr()-based dispatch removed from this file.
  • repeat_kv helper moved to backend.py (only needed on NOCACHE path).
  • ForwardBatch replaces the separate (attention_mask, kv_cache) parameters.
  • PagedExtendBackend stored on self.backend; handles all three modes.

Mirror of HuggingFace Qwen3Attention (modeling_qwen3.py L222-291) but:
  - layer_idx stored on the object (SGLang style) for KV cache dispatch
  - ForwardBatch passed instead of (attention_mask, kv_cache)
  - Backend object owns kernel selection and mask construction
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch  # noqa: E402

from .backend import PagedExtendBackend
from .config import Qwen3Config
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx    = layer_idx
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads      # 8
        self.head_dim     = config.head_dim                 # 128

        q_dim  = self.num_heads    * self.head_dim   # 2048
        kv_dim = self.num_kv_heads * self.head_dim   # 1024

        self.q_proj = nn.Linear(config.hidden_size, q_dim,  bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size,  bias=config.attention_bias)

        # Per-head QK normalisation — Qwen3-specific stabilisation.
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
        # All kernel logic (FlashInfer paged extend/decode, F.sdpa, pool write)
        # lives in backend.py.  This call is the only line that varies.
        attn_out = self.backend.forward(q, k, v, self.layer_idx, forward_batch)
        # [B, n_heads, q_len, head_dim]

        # ── 5. Merge heads and output projection ──────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)
