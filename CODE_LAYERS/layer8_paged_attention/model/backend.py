"""
PagedBackend — the single attention backend for Layer 8.

Mirrors SGLang's AttentionBackend pattern (srt/layers/attention/):
  One backend class implements forward(). Qwen3Attention stores one instance
  and calls backend.forward() — no if/else dispatch in the attention layer.

Layer 8 has no SDPA fallback for decode: the KVPool design requires FlashInfer
to index directly into the pre-allocated pool. The two sub-paths are:

  PREFILL (ForwardMode.PREFILL)
    kv_cache is PrefillKVCtx | None
    → kv_cache.store(layer_idx, k, v) writes K/V into pool slots (side-effect)
    → repeat_kv + build_additive_mask + F.sdpa runs over the fresh tensors

  DECODE (ForwardMode.DECODE)
    kv_cache is DecodeKVCtx
    → kv_cache.store(layer_idx, k_fi, v_fi) writes the new token to new_slots
    → wrapper.forward(q_fi, (k_paged, v_paged)) reads full history from pool
      via kv_indices set in begin_forward — no copy, no gather

GQA:
  F.sdpa path: repeat_kv expands 8 KV heads → 16 Q heads before the call.
  FlashInfer decode path: GQA handled natively (num_qo_heads=16, num_kv_heads=8).
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
    Uses expand + reshape (no data copy). Only needed on the F.sdpa prefill path;
    FlashInfer handles GQA natively on the decode path.
    """
    if n_rep == 1:
        return x
    B, H, S, D = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, H, n_rep, S, D)
        .reshape(B, H * n_rep, S, D)
    )


def build_additive_mask(
    attention_mask: torch.Tensor | None,
    q_len: int,
    kv_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor | None:
    """
    Build a [B, 1, q_len, kv_len] additive mask:
      0     → position is attended to
      -inf  → position is masked (future token or padding)

    For q_len == 1 (decode step): the causal part is all-zero because a
    single query token can attend to every key in the cache.
    """
    NEG_INF = torch.finfo(dtype).min

    causal = torch.zeros(q_len, kv_len, dtype=dtype, device=device)
    if q_len > 1:
        mask_upper = torch.ones(q_len, kv_len, dtype=torch.bool, device=device)
        mask_upper = torch.triu(mask_upper, diagonal=kv_len - q_len + 1)
        causal = causal.masked_fill(mask_upper, NEG_INF)

    causal = causal.unsqueeze(0).unsqueeze(0)  # [1, 1, q_len, kv_len]

    if attention_mask is None:
        return causal

    pad = attention_mask.to(dtype)       # 1.0 or 0.0
    pad = (1.0 - pad) * NEG_INF         # 0.0 or -inf
    pad = pad[:, None, None, :]          # [B, 1, 1, kv_len]

    return causal + pad                  # [B, 1, q_len, kv_len]


# ─────────────────────────────────────────────────────────────────────────────
# PagedBackend
# ─────────────────────────────────────────────────────────────────────────────

class PagedBackend:
    """
    The only attention backend in Layer 8.

    PREFILL: F.sdpa over the fresh prompt K/V; pool write is a side-effect.
    DECODE:  FlashInfer BatchDecodeWithPagedKVCacheWrapper reads KVPool directly.
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

        if forward_batch.mode == ForwardMode.PREFILL:
            return self._prefill_forward(q, k, v, layer_idx, forward_batch)
        else:
            return self._decode_forward(q, k, v, layer_idx, forward_batch)

    def _prefill_forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        kv = forward_batch.kv_cache

        # Write K/V into pool slots (side-effect); attention runs over the
        # fresh tensors, not the pool.
        if kv is not None:
            kv.store(layer_idx, k, v)

        # GQA expansion for F.sdpa (FlashInfer handles this natively on decode).
        k_rep = repeat_kv(k, self.num_kv_groups)
        v_rep = repeat_kv(v, self.num_kv_groups)

        B, _, q_len, _ = q.shape
        kv_len = k.shape[2]
        additive_mask = build_additive_mask(
            forward_batch.attention_mask, q_len, kv_len, q.dtype, q.device
        )

        return F.scaled_dot_product_attention(
            q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale
        )  # [B, n_q_heads, q_len, head_dim]

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
        v_fi = v.squeeze(2)    # [B, n_kv_heads, head_dim]

        # Write new decode token to pool FIRST — FlashInfer reads it below.
        kv.store(layer_idx, k_fi, v_fi)

        # Pool tensors in NHD layout; unsqueeze(1) inserts the page_size=1 dim
        # that BatchDecodeWithPagedKVCacheWrapper expects:
        #   [total_slots, 1, n_kv_heads, head_dim]
        k_paged = kv.k_pool[layer_idx].unsqueeze(1)
        v_paged = kv.v_pool[layer_idx].unsqueeze(1)

        # FlashInfer reads the full KV history via kv_indices set in begin_forward.
        # GQA (16Q / 8KV) is handled natively — no repeat_kv needed.
        attn_out = kv.wrapper.forward(q_fi, (k_paged, v_paged))
        # [B, n_q_heads, head_dim] → [B, n_q_heads, 1, head_dim]
        return attn_out.unsqueeze(2)
