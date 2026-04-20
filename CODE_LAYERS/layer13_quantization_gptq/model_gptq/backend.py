"""
PagedExtendBackend — the single attention backend for Layer 11.

Mirrors SGLang's AttentionBackend pattern (srt/layers/attention/):
  One backend class implements forward(). Qwen3Attention stores one instance
  and calls backend.forward() — no if/else dispatch in the attention layer.

Layer 11 introduces three sub-paths:

  EXTEND  (ExtendKVCtx)
    kv_cache.store(layer_idx, k, v)
      → writes K/V slices into the new pool pages allocated by prefill_batch.
    FlashInfer BatchPrefillWithPagedKVCacheWrapper.forward(q_fi, (k_pool, v_pool))
      → Q attends causally over all pages: committed (prior chunks) + new.
      → causal=True passed to both begin_forward() and forward().
    Input layout:  [1, n_heads, total_tokens, head_dim]  (B=1 packed batch)
    Output layout: [1, n_heads, total_tokens, head_dim]

  DECODE  (DecodeKVCtx)
    Same as Layer 9:
    kv_cache.store(layer_idx, k_fi, v_fi)
      → writes the single new token into the current page slot.
    FlashInfer BatchDecodeWithPagedKVCacheWrapper.forward(q_fi, (k_pool, v_pool))
      → GQA handled natively (num_qo_heads=16, num_kv_heads=8).
    Input layout:  [B, n_heads, 1, head_dim]  → squeeze(2) for FlashInfer NHD
    Output layout: [B, n_heads, 1, head_dim]

  NOCACHE (None)
    Plain F.scaled_dot_product_attention for the no-cache verify baseline.
    GQA expanded via repeat_kv.
    attention_mask from ForwardBatch (prebuilt additive [B, 1, q, kv]).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch, ForwardMode

from .config import Qwen3Config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads for GQA: [B, n_kv_heads, S, D] → [B, n_kv_heads*n_rep, S, D].
    Uses expand + reshape (no data copy). Only needed on the F.sdpa NOCACHE path;
    FlashInfer handles GQA natively on EXTEND and DECODE paths.
    """
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


# ─────────────────────────────────────────────────────────────────────────────
# PagedExtendBackend
# ─────────────────────────────────────────────────────────────────────────────

class PagedExtendBackend:
    """
    The single attention backend for Layer 11.

    EXTEND:  FlashInfer paged prefill (full prompt or any chunk).
    DECODE:  FlashInfer paged decode (one token per request).
    NOCACHE: F.sdpa only, no KV pool interaction.
    """

    def __init__(self, config: Qwen3Config) -> None:
        self.num_kv_groups = config.num_kv_groups
        self.scale         = config.head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,           # [B, n_q_heads,  q_len, head_dim]
        k: torch.Tensor,           # [B, n_kv_heads, q_len, head_dim]
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:             # [B, n_q_heads, q_len, head_dim]

        if forward_batch.mode == ForwardMode.EXTEND:
            return self._extend_forward(q, k, v, layer_idx, forward_batch)
        elif forward_batch.mode == ForwardMode.DECODE:
            return self._decode_forward(q, k, v, layer_idx, forward_batch)
        else:
            return self._nocache_forward(q, k, v, forward_batch)

    def _extend_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        kv = forward_batch.kv_cache

        # Write new K/V to pool pages first — FlashInfer reads from pool below.
        kv.store(layer_idx, k, v)

        # Reshape Q from [1, n_heads, total_tokens, D] to FlashInfer NHD:
        # [total_tokens, n_heads, D]
        q_fi = q.squeeze(0).permute(1, 0, 2)

        # Pool tensors: [total_pages, page_size, n_kv_heads, head_dim]
        k_paged = kv.k_pool[layer_idx]
        v_paged = kv.v_pool[layer_idx]

        # Paged causal prefill: Q attends over all pages (committed + new).
        # causal=True must match the causal=True passed to begin_forward().
        # Output: [total_tokens, n_heads, head_dim]
        attn_out = kv.extend_wrapper.forward(q_fi, (k_paged, v_paged), causal=True)

        # Reshape back: [total_tokens, n_heads, D] → [1, n_heads, total_tokens, D]
        return attn_out.unsqueeze(0).permute(0, 2, 1, 3)

    def _decode_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        kv = forward_batch.kv_cache

        # q_len == 1 per request; squeeze that dim for FlashInfer NHD layout.
        q_fi = q.squeeze(2)    # [B, n_q_heads,  head_dim]
        k_fi = k.squeeze(2)    # [B, n_kv_heads, head_dim]
        v_fi = v.squeeze(2)

        # Write new decode token to pool FIRST — FlashInfer reads it below.
        kv.store(layer_idx, k_fi, v_fi)

        # Pool tensors: [total_pages, page_size, n_kv_heads, head_dim]
        k_paged = kv.k_pool[layer_idx]
        v_paged = kv.v_pool[layer_idx]

        # FlashInfer reads full KV history via kv_indices set in begin_forward.
        # GQA (16Q / 8KV) is handled natively — no repeat_kv needed.
        attn_out = kv.wrapper.forward(q_fi, (k_paged, v_paged))
        # [B, n_q_heads, head_dim] → [B, n_q_heads, 1, head_dim]
        return attn_out.unsqueeze(2)

    def _nocache_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # GQA expansion for F.sdpa (FlashInfer handles this natively above).
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        return F.scaled_dot_product_attention(
            q, k_rep, v_rep,
            attn_mask=forward_batch.attention_mask,
            scale=self.scale,
        )  # [B, n_q_heads, q_len, head_dim]
