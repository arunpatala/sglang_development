"""
Layer 5 — Batch: a group of requests sharing one forward pass.

Mirrors SGLang's ForwardMode + ScheduleBatch concept.

ForwardMode:
  PREFILL — one or more requests being prefilled (B=1 each, sequential)
  DECODE  — all running requests decoded together in a single batched step

The model_runner inspects forward_mode to decide which codepath to use:
  PREFILL → individual B=1 forward, populate req.kv_cache
  DECODE  → batched B=N forward with padded+stacked KV caches
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import List

from request import Req


class ForwardMode(Enum):
    PREFILL = auto()
    DECODE  = auto()


@dataclass
class Batch:
    reqs:         List[Req]
    forward_mode: ForwardMode

    def __len__(self) -> int:
        return len(self.reqs)

    def __repr__(self) -> str:
        return (
            f"Batch(mode={self.forward_mode.name}, "
            f"size={len(self.reqs)}, "
            f"rids={[r.rid[:6] for r in self.reqs]})"
        )
