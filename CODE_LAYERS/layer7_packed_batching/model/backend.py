"""
Attention backends — own all kernel dispatch logic so attention.py stays clean.

Mirrors SGLang's AttentionBackend pattern (srt/layers/attention/):
  Each backend implements a single forward() method.
  Qwen3Attention stores one backend object and calls backend.forward() —
  zero if/else in the attention layer itself.

Backends
────────
SDPABackend
  Uses F.scaled_dot_product_attention for every call (prefill + decode).
  Handles padding via an additive mask built from ForwardBatch.attention_mask.
  GQA: expands KV heads with repeat_kv before the SDPA call.

FlashInferBackend
  Prefill (ForwardMode.PREFILL): delegates to _sdpa_forward (same as SDPABackend).
  Decode  (ForwardMode.DECODE):  calls PackedKVCache.update() to get the ragged
    packed tensor, then wrapper.forward() — FlashInfer ragged attention.
    GQA handled natively by FlashInfer; no repeat_kv needed.

Factory
───────
make_backend(config) → SDPABackend | FlashInferBackend
  Called once in Qwen3Attention.__init__ based on config.attn_backend.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

# forward_batch lives one level up from model/
sys.path.insert(0, str(Path(__file__).parent.parent))
from forward_batch import ForwardBatch, ForwardMode

from .config import AttnBackend, Qwen3Config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (previously in attention.py and qwen3.py)
# ─────────────────────────────────────────────────────────────────────────────

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Expand KV heads to match Q head count for GQA.
    [B, n_kv_heads, S, D] → [B, n_kv_heads * n_rep, S, D]

    Uses expand + reshape (no data copy). Only needed on the F.sdpa path;
    FlashInfer handles GQA natively.
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
# Shared F.sdpa kernel (used by both backends on the prefill path)
# ─────────────────────────────────────────────────────────────────────────────

def _sdpa_forward(
    q: torch.Tensor,        # [B, n_q_heads, q_len, head_dim]
    k: torch.Tensor,        # [B, n_kv_heads, q_len, head_dim]  (pre-RoPE)
    v: torch.Tensor,
    layer_idx: int,
    forward_batch: ForwardBatch,
    num_kv_groups: int,
    scale: float,
) -> torch.Tensor:
    """F.sdpa path shared by SDPABackend and FlashInferBackend (prefill)."""
    kv = forward_batch.kv_cache
    if kv is not None:
        k, v = kv.update(layer_idx, k, v)

    k = repeat_kv(k, num_kv_groups)   # [B, n_q_heads, kv_len, head_dim]
    v = repeat_kv(v, num_kv_groups)

    B, _, q_len, _ = q.shape
    kv_len = k.shape[2]
    additive_mask = build_additive_mask(
        forward_batch.attention_mask, q_len, kv_len, q.dtype, q.device
    )

    return F.scaled_dot_product_attention(
        q, k, v, attn_mask=additive_mask, scale=scale
    )  # [B, n_q_heads, q_len, head_dim]


# ─────────────────────────────────────────────────────────────────────────────
# SDPABackend
# ─────────────────────────────────────────────────────────────────────────────

class SDPABackend:
    """F.sdpa for all paths. No FlashInfer dependency."""

    def __init__(self, config: Qwen3Config) -> None:
        self.num_kv_groups = config.num_kv_groups
        self.scale         = config.head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        return _sdpa_forward(
            q, k, v, layer_idx, forward_batch, self.num_kv_groups, self.scale
        )


# ─────────────────────────────────────────────────────────────────────────────
# FlashInferBackend
# ─────────────────────────────────────────────────────────────────────────────

class FlashInferBackend:
    """
    Prefill → F.sdpa (B=1, PerReqKVCache, rectangular KV).
    Decode  → FlashInfer ragged attention (B=N, PackedKVCache, no padding).
    """

    def __init__(self, config: Qwen3Config) -> None:
        self.num_kv_groups = config.num_kv_groups
        self.scale         = config.head_dim ** -0.5

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer_idx: int,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        if forward_batch.mode == ForwardMode.PREFILL:
            return _sdpa_forward(
                q, k, v, layer_idx, forward_batch, self.num_kv_groups, self.scale
            )

        # ── Decode: FlashInfer ragged path ────────────────────────────────
        # q/k/v are [B, heads, 1, head_dim] — squeeze the q_len=1 dim for NHD.
        q_fi = q.squeeze(2)   # [B, n_q_heads,  head_dim]
        k_fi = k.squeeze(2)   # [B, n_kv_heads, head_dim]
        v_fi = v.squeeze(2)

        # PackedKVCache.update() gathers historical KV + new token into a
        # ragged tensor [total_kv_tokens, n_kv_heads, head_dim] and saves the
        # new token for write_back() after the full forward pass.
        kv = forward_batch.kv_cache
        k_packed, v_packed = kv.update(layer_idx, k_fi, v_fi)

        # FlashInfer handles GQA natively — no repeat_kv needed.
        # wrapper.forward() uses the plan from begin_forward (called once
        # via PackedKVCache.plan() before the 28-layer forward pass).
        attn_out = kv.wrapper.forward(q_fi, k_packed, v_packed)
        # [B, n_q_heads, head_dim] → [B, n_q_heads, 1, head_dim]
        return attn_out.unsqueeze(2)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def make_backend(config: Qwen3Config) -> SDPABackend | FlashInferBackend:
    if config.attn_backend == AttnBackend.FLASHINFER:
        return FlashInferBackend(config)
    return SDPABackend(config)
