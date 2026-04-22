"""
Layer 11 — Request: adds prefix-caching state on top of Layer 10.

New fields vs Layer 10:

  prefix_len          — number of prompt tokens whose K/V were found in the
                        RadixCache (always a multiple of page_size).
                        0 when no prefix was cached.

  prefix_page_indices — page indices returned by RadixCache.match_prefix().
                        These pages are owned by the tree; we must NOT free them.
                        pre-populates slot_indices before prefill_batch.

  last_node           — deepest TreeNode matched by match_prefix().
                        We call inc_lock_ref(last_node) at match time and
                        dec_lock_ref(last_node) when the request finishes.

Layer 10 fields (unchanged):

  fill_ids          — token slice being processed this extend round.
                      With prefix caching: fill_ids = input_ids[prefix_len:]
                      for the first extend round (skipping cached tokens).

  kv_committed_len  — tokens already in KV pool from prior rounds.
                      With prefix: starts at prefix_len (not 0).

  extend_input_len  — len(fill_ids) for the current round.

  ReqStatus.PREFILLING — mid-chunked-prefill status (unchanged).
"""

import asyncio
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


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

    # ── Prefix-caching state  (set by PrefillAdder before first prefill_batch)
    # ──────────────────────────────────────────────────────────────────
    # Number of prompt tokens whose K/V are already in the RadixCache.
    # Always a multiple of page_size.  0 when no prefix was cached.
    prefix_len: int = 0

    # Page indices returned by RadixCache.match_prefix().
    # Owned by the tree; must NOT be freed by this request's cleanup.
    # Pre-populates slot_indices so prefill_batch can build kv_indptr correctly.
    prefix_page_indices: List[int] = field(default_factory=list)

    # Deepest TreeNode matched by match_prefix().
    # inc_lock_ref called at match time; dec_lock_ref called at finish.
    last_node: Optional[Any] = None

    # ── Chunked-prefill state  (set by scheduler before each prefill_batch)
    # ──────────────────────────────────────────────────────────────────
    # Token slice fed to the model this extend round.
    fill_ids: List[int] = field(default_factory=list)

    # Tokens already committed to KV pool from previous extend rounds.
    # With prefix caching: starts at prefix_len (not 0) after match.
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
