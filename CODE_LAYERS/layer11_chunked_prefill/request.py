"""
Layer 10 — Request: adds chunked-prefill state.

New fields vs Layer 9:

  fill_ids          — token slice being processed this extend round.
                      Set by the scheduler before each prefill_batch call:
                        • Full prefill:   fill_ids = input_ids
                        • Chunk k:        fill_ids = input_ids[start:end]
                      NOT the same as input_ids for chunked requests.

  kv_committed_len  — number of prompt tokens whose K/V have already been
                      written to the KV pool (from previous extend rounds).
                      0 on first extend; incremented after each round.

  extend_input_len  — len(fill_ids) for this round. Set by scheduler.

  ReqStatus.PREFILLING — new status for chunked requests mid-flight.
                      WAITING → PREFILLING (chunk 0..N-2) → RUNNING → FINISHED
                      Full-prompt prefill skips PREFILLING: WAITING → RUNNING.
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional


class ReqStatus(Enum):
    WAITING   = auto()   # in waiting_queue, not yet prefilled
    PREFILLING = auto()  # mid-chunked-prefill: more chunks to process
    RUNNING   = auto()   # prefill complete, active decode batch
    FINISHED  = auto()   # EOS emitted or max_new_tokens reached


@dataclass
class Req:
    # ── Identity ──────────────────────────────────────────────────────
    rid: str

    # ── Input ─────────────────────────────────────────────────────────
    input_ids: List[int]
    max_new_tokens: int
    temperature: float

    # ── Async bridge ──────────────────────────────────────────────────
    future: asyncio.Future

    # ── Mutable status ────────────────────────────────────────────────
    status: ReqStatus = ReqStatus.WAITING
    output_ids: List[int] = field(default_factory=list)

    # ── KV pool state ─────────────────────────────────────────────────
    # Page indices allocated for this request (grows with each extend round
    # and each decode step that crosses a page boundary).
    slot_indices: List[int] = field(default_factory=list)

    # Row index in ReqToTokenPool.req_to_token (assigned at first extend).
    req_pool_idx: Optional[int] = None

    # ── Chunked-prefill state  (set by scheduler before each prefill_batch)
    # ──────────────────────────────────────────────────────────────────
    # Token slice fed to the model this extend round.
    fill_ids: List[int] = field(default_factory=list)

    # Tokens already committed to KV pool from previous extend rounds.
    # Defines where in req_to_token the new pages start.
    kv_committed_len: int = 0

    # len(fill_ids) for the current round; set by scheduler.
    extend_input_len: int = 0

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

    @property
    def is_last_chunk(self) -> bool:
        """True when this extend round completes the full prompt."""
        return self.kv_committed_len + self.extend_input_len >= self.prompt_len
