"""
Qwen3Attention — multi-head self-attention with three backends:

  ┌─────────────────────────────────────────────────────────────────────┐
  │ kv_cache type    │ Backend                  │ When used             │
  │──────────────────┼──────────────────────────┼───────────────────────│
  │ None             │ F.sdpa                   │ no-cache (verify.py)  │
  │ ExtendKVCtx      │ BatchPrefillWithPaged...  │ prefill / extend      │
  │ DecodeKVCtx      │ BatchDecodeWithPaged...   │ decode                │
  └─────────────────────────────────────────────────────────────────────┘

Layer 10 changes vs Layer 9:
  • PrefillKVCtx (F.sdpa B=1) is gone.
  • ExtendKVCtx replaces it for ALL prefill/extend operations (B=1 or B=N,
    first chunk or continuation chunk).

Extend path in detail:
  Input hidden_states: [1, total_tokens, hidden_dim]  (packed batch, B=1)
  1. Project Q/K/V, apply QK-norm and RoPE (unchanged).
     Q: [1, n_heads, total_tokens, head_dim]
     K: [1, n_kv,   total_tokens, head_dim]
  2. ctx.store(layer_idx, k, v)
     → writes K/V slices for each request into their new pool pages.
  3. wrapper.forward(q_packed, (k_pool[layer], v_pool[layer]))
     → FlashInfer paged prefill: each request's Q attends causally over
        all its pages (cached from prior chunks + new pages from step 2).
     → Output: [total_tokens, n_heads, head_dim]
  4. Reshape back to [1, n_heads, total_tokens, head_dim] for merge-heads.

Decode path: unchanged from Layer 9.

FlashInfer NHD pool layout (page_size=P):
  k_pool[layer]: [total_pages, P, n_kv_heads, head_dim]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


class Qwen3Attention(nn.Module):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx    = layer_idx
        self.hidden_size  = config.hidden_size
        self.num_heads    = config.num_attention_heads      # 16
        self.num_kv_heads = config.num_key_value_heads      # 8
        self.num_kv_groups = config.num_kv_groups           # 2
        self.head_dim     = config.head_dim                 # 128
        self.scale        = self.head_dim ** -0.5

        q_dim  = self.num_heads    * self.head_dim
        kv_dim = self.num_kv_heads * self.head_dim

        self.q_proj = nn.Linear(config.hidden_size, q_dim,  bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size,  bias=config.attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, q_len, hidden]  B=1 for extend
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
        kv_cache=None,                        # ExtendKVCtx | DecodeKVCtx | None
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape

        # ── 1. Project Q / K / V ─────────────────────────────────────────
        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        # ── 2. Per-head QK RMSNorm ────────────────────────────────────────
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)   # [B, n_heads,    q_len, head_dim]
        k = k.transpose(1, 2)   # [B, n_kv_heads, q_len, head_dim]
        v = v.transpose(1, 2)

        # ── 3. Rotary Position Embedding ───────────────────────────────────
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 4. Backend dispatch ────────────────────────────────────────────

        if kv_cache is not None and hasattr(kv_cache, "extend_wrapper"):
            # ── ExtendKVCtx: batched paged prefill ────────────────────────
            # k/v shape here: [1, n_kv_heads, total_tokens, head_dim]
            #   where total_tokens = sum of all requests' extend_input_len.

            # Write new K/V to pool pages first (FlashInfer reads from pool).
            kv_cache.store(self.layer_idx, k, v)

            # Reshape Q to FlashInfer NHD format: [total_tokens, n_heads, D]
            q_fi = q.squeeze(0).permute(1, 0, 2)   # [total_tokens, n_heads, D]

            # Pool tensors: [total_pages, page_size, n_kv, head_dim]
            k_paged = kv_cache.k_pool[self.layer_idx]
            v_paged = kv_cache.v_pool[self.layer_idx]

            # Paged causal prefill: Q attends over all pages (cached + new).
            # causal=True must be passed to both begin_forward AND forward().
            # Output: [total_tokens, n_heads, head_dim]
            attn_out = kv_cache.extend_wrapper.forward(q_fi, (k_paged, v_paged), causal=True)

            # Reshape back: [total_tokens, n_heads, D] → [1, n_heads, total_tokens, D]
            attn_out = attn_out.unsqueeze(0).permute(0, 2, 1, 3)

        elif kv_cache is not None and hasattr(kv_cache, "wrapper"):
            # ── DecodeKVCtx: paged decode (unchanged from Layer 9) ────────
            q_fi = q.squeeze(2)    # [B, n_heads,    head_dim]
            k_fi = k.squeeze(2)    # [B, n_kv_heads, head_dim]
            v_fi = v.squeeze(2)

            kv_cache.store(self.layer_idx, k_fi, v_fi)

            k_paged = kv_cache.k_pool[self.layer_idx]
            v_paged = kv_cache.v_pool[self.layer_idx]

            attn_out = kv_cache.wrapper.forward(q_fi, (k_paged, v_paged))
            attn_out = attn_out.unsqueeze(2)   # [B, n_heads, 1, D]

        else:
            # ── No cache: plain F.sdpa (used by verify_batch.py baseline) ─
            k_rep = repeat_kv(k, self.num_kv_groups)
            v_rep = repeat_kv(v, self.num_kv_groups)
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attention_mask,
                scale=self.scale,
            )

        # ── 5. Merge heads → output projection ────────────────────────────
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, q_len, -1)
        return self.o_proj(attn_out)
