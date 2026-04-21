"""
ForwardBatch — per-call metadata passed through the model's forward stack.

Mirrors SGLang's ForwardBatch (srt/model_executor/forward_batch_info.py):
  ForwardMode  — which kernel path to take in Qwen3Attention
  ForwardBatch — carries the mode + per-step state so the attention backend
                 can select the right FlashInfer wrapper without inspecting
                 the KV-cache object type directly.

Layer 11 has three modes (vs. Layer 8's two):

  EXTEND  (ForwardMode.EXTEND)
    kv_cache is ExtendKVCtx
    → ctx.store(layer_idx, k, v) writes K/V into new pool pages.
    → FlashInfer BatchPrefillWithPagedKVCacheWrapper attends over all pages
      (already-committed chunks + new pages from store()).
    → Handles both first-chunk and continuation-chunk prefill uniformly.
    → B=1 packed batch: all request tokens concatenated in one [1, T, H] tensor.

  DECODE  (ForwardMode.DECODE)
    kv_cache is DecodeKVCtx
    → ctx.store(layer_idx, k_fi, v_fi) writes the single new token per request.
    → FlashInfer BatchDecodeWithPagedKVCacheWrapper reads full KV history from pool.
    → B=N; one token per request per step.

  NOCACHE (ForwardMode.NOCACHE)
    kv_cache is None
    → Plain F.scaled_dot_product_attention over the full prompt.
    → Used by verify_batch.py baseline (no KV pool, no paging).
    → attention_mask carries the prebuilt additive [B, 1, q_len, kv_len] mask.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class ForwardMode(Enum):
    EXTEND  = auto()   # paged prefill (full prompt or chunk); uses ExtendKVCtx
    DECODE  = auto()   # paged decode one token per request;   uses DecodeKVCtx
    NOCACHE = auto()   # plain F.sdpa, no KV pool (verify baseline)


@dataclass
class ForwardBatch:
    mode:           ForwardMode
    kv_cache:       object                  # ExtendKVCtx | DecodeKVCtx | None
    attention_mask: Optional[torch.Tensor]  # additive [B, 1, q_len, kv_len]
                                            # populated only for NOCACHE mode;
                                            # FlashInfer paths set this to None.
