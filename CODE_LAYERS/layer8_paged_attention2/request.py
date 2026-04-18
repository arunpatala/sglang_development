"""
Layer 8 — Request: per-request state tracked by the scheduler.

Mirrors SGLang's Req class (managers/schedule_batch.py) but stripped to
the essentials needed for continuous batching with paged KV cache:

  input_ids     — tokenized prompt
  output_ids    — tokens generated so far
  slot_indices  — physical KV pool slot for each token (prompt + decode).
                  Used only for KVPool.free() on finish.
  req_pool_idx  — row index in ReqToTokenPool.req_to_token [max_batch, max_ctx].
                  The Triton kernel reads this row to build kv_indices on-GPU
                  each decode step — no Python loop, no CPU→GPU copy of slot data.
  status        — WAITING → RUNNING → FINISHED
  future        — asyncio.Future resolved by the scheduler when done

The scheduler never touches HTTP; the server never touches GPU.
They communicate only through Req.future.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class ReqStatus(Enum):
    WAITING  = auto()   # in waiting_queue, not yet prefilled
    RUNNING  = auto()   # prefilled, in the active decode batch
    FINISHED = auto()   # EOS emitted or max_new_tokens reached


@dataclass
class Req:
    # ── Identity ──────────────────────────────────────────────────────
    rid: str

    # ── Input ─────────────────────────────────────────────────────────
    input_ids: List[int]
    max_new_tokens: int
    temperature: float

    # ── Async bridge ──────────────────────────────────────────────────
    # Created in the asyncio event loop thread; resolved by the scheduler
    # thread via loop.call_soon_threadsafe(future.set_result, result).
    future: asyncio.Future

    # ── Mutable state ─────────────────────────────────────────────────
    status: ReqStatus = ReqStatus.WAITING
    output_ids: List[int] = field(default_factory=list)

    # Physical KV pool slot for each token (prompt + generated).
    # Freed via KVPool.free(req.slot_indices) when the request finishes.
    slot_indices: List[int] = field(default_factory=list)

    # Row index in ReqToTokenPool.req_to_token for this request.
    # Assigned at prefill; freed at finish.
    req_pool_idx: Optional[int] = None

    # ── Timing ────────────────────────────────────────────────────────
    t_arrive:      float = field(default_factory=time.perf_counter)
    t_first_token: float = 0.0
    t_finish:      float = 0.0

    @property
    def prompt_len(self) -> int:
        return len(self.input_ids)

    @property
    def output_len(self) -> int:
        return len(self.output_ids)

    @property
    def ttft_ms(self) -> float:
        return (self.t_first_token - self.t_arrive) * 1000

    @property
    def latency_ms(self) -> float:
        return (self.t_finish - self.t_arrive) * 1000
