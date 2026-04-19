"""
ForwardBatch — per-call metadata passed through the model's forward stack.

Mirrors SGLang's ForwardBatch (srt/model_executor/forward_batch_info.py):
  ForwardMode  — whether this is a prefill or a decode step
  ForwardBatch — carries the mode + per-step state so attention layers
                 can choose the right kernel without inspecting the cache type.

Why ForwardBatch instead of passing (attention_mask, kv_cache) separately?
  Separating the "what data" (kv_cache) from "what mode" (ForwardMode) makes
  the attention backend's dispatch explicit and extensible:
    • Adding a new backend (paged, tensor-parallel) means writing a new
      backend class, not adding another `hasattr` branch in attention.py.
    • The mode is declared by the caller (model_runner) rather than inferred
      from which cache type happens to be present.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch


class ForwardMode(Enum):
    PREFILL = auto()   # B=1 per request; populates kv_cache from full prompt
    DECODE  = auto()   # B=N; one new token per request


@dataclass
class ForwardBatch:
    mode:           ForwardMode
    kv_cache:       object                  # PerReqKVCache | PackedKVCache | None
    attention_mask: Optional[torch.Tensor]  # binary [B, kv_len] (1=real, 0=pad)
                                            # None on the FlashInfer decode path
