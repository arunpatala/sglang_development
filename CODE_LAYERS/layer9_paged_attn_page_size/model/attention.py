"""
Qwen3Attention — multi-head self-attention with:
  • Grouped Query Attention (GQA): 16 Q heads share 8 KV heads
  • Per-head QK RMSNorm (Qwen3-specific)
  • Rotary Position Embedding applied after QK norm
  • Three KV backends selected by kv_cache type:

    ┌─────────────────────────────────────────────────────────────────────┐
    │ kv_cache type    │ Backend       │ When used                        │
    │──────────────────┼───────────────┼──────────────────────────────────│
    │ None             │ F.sdpa        │ no-cache (not used in server)    │
    │ PrefillKVCtx     │ F.sdpa        │ B=1 prefill — writes to KVPool   │
    │ DecodeKVCtx      │ FlashInfer    │ B=N decode — reads from KVPool   │
    └─────────────────────────────────────────────────────────────────────┘

Separation of concerns (same principle as Layer 6):
  kv_cache.store(layer, k, v)  — data only: write new K/V into the pool
  attention.py                 — kernel only: F.sdpa or FlashInfer decode

Prefill path in detail:
  1. Compute Q, K, V from projections + QK norm + RoPE (unchanged).
  2. ctx.store(layer_idx, k, v)  →  write compact [1,n_kv,L,D] K/V to pool slots.
  3. F.sdpa(q, k_rep, v_rep, causal_mask)  →  self-attention over prompt.
  Pool write is a side-effect; attention uses the fresh tensors, not the pool.

Decode path in detail:
  1. Compute Q, K, V for the single new token (q_len=1 per request).
  2. ctx.store(layer_idx, k_fi, v_fi)  →  write [B,n_kv,D] to new_slots in pool.
  3. wrapper.forward(q_fi, (k_pool[layer].unsqueeze(1), v_pool[layer].unsqueeze(1)))
     →  FlashInfer reads entire KV history from pool via kv_indices (no copy).

GQA note:
  F.sdpa path still uses repeat_kv to expand 8→16 heads.
  FlashInfer handles GQA natively (num_qo_heads=16, num_kv_heads=8).

FlashInfer pool tensor convention (page_size=P, NHD layout):
  k_pool[layer]:  [total_pages, page_size, n_kv_heads, head_dim]
  Passed as-is:   (k_pool[layer], v_pool[layer])
  FlashInfer reads the (page_idx, within-page offset) from kv_indices +
  kv_last_page_lens.  No unsqueeze needed; the page_size dim is already there.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import Qwen3Config
from .norm import RMSNorm
from .rope import apply_rotary_pos_emb


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads for GQA: [B, n_kv, S, D] → [B, n_kv*n_rep, S, D].
    Only used in the F.sdpa path; FlashInfer handles GQA natively.
    """
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

        q_dim  = self.num_heads    * self.head_dim    # 2048
        kv_dim = self.num_kv_heads * self.head_dim    # 1024

        self.q_proj = nn.Linear(config.hidden_size, q_dim,  bias=config.attention_bias)
        self.k_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(config.hidden_size, kv_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(q_dim, config.hidden_size,  bias=config.attention_bias)

        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,          # [B, q_len, hidden]
        cos: torch.Tensor,                    # [B, q_len, head_dim]
        sin: torch.Tensor,                    # [B, q_len, head_dim]
        attention_mask: torch.Tensor | None,  # additive [B,1,q_len,kv_len] — prefill only
        kv_cache=None,                        # PrefillKVCtx | DecodeKVCtx | None
    ) -> torch.Tensor:
        B, q_len, _ = hidden_states.shape

        # ── 1. Project Q / K / V ─────────────────────────────────────────
        q = self.q_proj(hidden_states).view(B, q_len, self.num_heads,    self.head_dim)
        k = self.k_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(B, q_len, self.num_kv_heads, self.head_dim)

        # ── 2. Per-head QK RMSNorm (Qwen3-specific) ───────────────────────
        q = self.q_norm(q)
        k = self.k_norm(k)

        q = q.transpose(1, 2)   # [B, n_heads,    q_len, head_dim]
        k = k.transpose(1, 2)   # [B, n_kv_heads, q_len, head_dim]
        v = v.transpose(1, 2)

        # ── 3. Rotary Position Embedding ───────────────────────────────────
        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # ── 4. Backend dispatch ────────────────────────────────────────────

        if kv_cache is not None and hasattr(kv_cache, "prefill_slots"):
            # ── PrefillKVCtx: F.sdpa + write K/V to pool ──────────────────
            # Store compact K/V into pool BEFORE repeat_kv (pool uses n_kv format).
            # k shape here: [1, n_kv_heads, prompt_len, head_dim]
            kv_cache.store(self.layer_idx, k, v)

            # GQA expand for F.sdpa
            k_rep = repeat_kv(k, self.num_kv_groups)
            v_rep = repeat_kv(v, self.num_kv_groups)

            # Causal self-attention over the prompt (attention_mask has causal triu)
            attn_out = F.scaled_dot_product_attention(
                q, k_rep, v_rep,
                attn_mask=attention_mask,
                scale=self.scale,
            )  # [1, n_heads, prompt_len, head_dim]

        elif kv_cache is not None and hasattr(kv_cache, "wrapper"):
            # ── DecodeKVCtx: write new token to pool, FlashInfer reads pool ─
            # q_len == 1 per request; squeeze that dim for FlashInfer NHD.
            q_fi  = q.squeeze(2)    # [B, n_q_heads,  head_dim]
            k_fi  = k.squeeze(2)    # [B, n_kv_heads, head_dim]
            v_fi  = v.squeeze(2)    # [B, n_kv_heads, head_dim]

            # Write new decode token to pool FIRST (FlashInfer reads it below).
            kv_cache.store(self.layer_idx, k_fi, v_fi)

            # FlashInfer reads the full KV history from the pool via kv_indices.
            # Pool tensors: [total_pages, page_size, n_kv, head_dim]
            # FlashInfer NHD paged layout expects exactly this shape.
            k_paged = kv_cache.k_pool[self.layer_idx]
            v_paged = kv_cache.v_pool[self.layer_idx]

            # Output: [B, n_q_heads, head_dim]
            attn_out = kv_cache.wrapper.forward(q_fi, (k_paged, v_paged))

            # Re-add seq dim so the merge-heads step below is unchanged.
            attn_out = attn_out.unsqueeze(2)   # [B, n_q_heads, 1, head_dim]

        else:
            # ── No cache: plain F.sdpa (causal, used for standalone testing) ─
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
