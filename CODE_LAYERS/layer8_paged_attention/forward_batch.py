"""
ForwardBatch — per-call metadata passed through the model's forward stack.

Mirrors SGLang's ForwardBatch (srt/model_executor/forward_batch_info.py):
  ForwardMode  — whether this is a prefill or a decode step
  ForwardBatch — carries the mode + per-step state so the attention backend
                 can select the right kernel without inspecting the cache type.

Layer 8 has exactly one backend (PagedBackend):
  PREFILL → F.sdpa + PrefillKVCtx.store() side-effect (pool write)
  DECODE  → DecodeKVCtx.store() + FlashInfer BatchDecodeWithPagedKVCacheWrapper

There is no SDPA fallback for decode: the KVPool architecture requires
FlashInfer to read KV history without copying it. Removing the SDPA decode
path is intentional — it was only present in Layer 7 for the ragged packed
case where SDPA was a meaningful alternative.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class ForwardMode(Enum):
    PREFILL = auto()   # B=1 per request; writes K/V to KVPool via PrefillKVCtx
    DECODE  = auto()   # B=N; reads K/V from KVPool via DecodeKVCtx + FlashInfer


@dataclass
class ForwardBatch:
    mode:           ForwardMode
    kv_cache:       object                  # PrefillKVCtx | DecodeKVCtx | None
    attention_mask: Optional[torch.Tensor]  # binary [B, kv_len] (1=real, 0=pad)
                                            # None on the FlashInfer decode path
