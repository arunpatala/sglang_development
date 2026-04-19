"""
Layer 11 — Scheduler: prefix caching integration.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changed from Layer 10
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  PrefillAdder.build() now accepts an optional radix_cache.  When set,
  before deciding how many tokens to allocate for a new request, it calls:

      page_indices, matched_len, last_node = radix_cache.match_prefix(req.input_ids)

  Then on the request:
      req.prefix_page_indices = page_indices
      req.prefix_len          = matched_len
      req.last_node           = last_node
      req.slot_indices        = list(page_indices)   # pre-populate
      req.kv_committed_len    = matched_len           # skip cached tokens
      req.fill_ids            = req.input_ids[matched_len:]
      req.extend_input_len    = len(req.fill_ids)
      radix_cache.inc_lock_ref(last_node)             # protect from eviction

  The effective token cost charged to the budget is extend_input_len
  (only new tokens), not the full prompt length.  This means a request
  with a long shared prefix is "cheap" to schedule.

  Chunked_req continuation is unchanged — the second and later chunks of
  a chunked request have kv_committed_len > 0 from prior extend rounds
  and never re-run match_prefix (prefix was already set on chunk 0).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PrefillAdder (simplified vs SGLang)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  • max_prefill_tokens  — token budget for this extend round
  • chunked_prefill_size — max tokens per request per round (0 = unlimited)
  • FIFO scheduling (no priority / preemption)
  • One chunked request at a time

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chunked_req state machine (unchanged from Layer 10)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  self.chunked_req = None means no request is mid-prefill.

  When a request needs chunking:
    chunk 0:    chunked_req = req,  req.status = PREFILLING
    chunk k:    chunked_req = req (same),  req.status = PREFILLING
    last chunk: req.status = RUNNING or FINISHED,  chunked_req = None
"""

import asyncio
import logging
import queue
import sys
import time
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).parent))

from model_runner import ModelRunner
from radix_cache import RadixCache
from request import Req, ReqStatus

logger = logging.getLogger(__name__)

IDLE_SLEEP_S = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# PrefillAdder — decides which requests to extend this round
# ─────────────────────────────────────────────────────────────────────────────

class PrefillAdder:
    """
    Simplified SGLang PrefillAdder — extended with prefix-cache matching.

    Given the waiting queue and current state, builds the list of requests to
    include in the next EXTEND forward pass and sets fill_ids / extend_input_len
    on each selected request.

    With prefix caching:
      • match_prefix() is called once per new request when it's dequeued.
      • req.prefix_len / prefix_page_indices / last_node are set.
      • The token budget is charged based on extend_input_len (non-cached tokens).

    Args:
      waiting:              thread-safe queue of Req
      running_count:        number of requests currently in decode batch
      max_running_reqs:     hard limit on decode batch size
      max_prefill_tokens:   token budget for this extend round (across all reqs)
      chunked_prefill_size: max tokens per request per round (0 = unlimited)
      chunked_req:          in-flight chunked request (None if none)
      radix_cache:          optional RadixCache for prefix matching
    """

    def __init__(
        self,
        waiting: queue.Queue,
        running_count: int,
        max_running_reqs: int,
        max_prefill_tokens: int,
        chunked_prefill_size: int,
        chunked_req: Optional[Req],
        radix_cache: Optional[RadixCache] = None,
    ) -> None:
        self.waiting              = waiting
        self.running_count        = running_count
        self.max_running_reqs     = max_running_reqs
        self.max_prefill_tokens   = max_prefill_tokens
        self.chunked_prefill_size = chunked_prefill_size
        self.chunked_req          = chunked_req
        self.radix_cache          = radix_cache

        # Set by build():
        self.new_chunked_req: Optional[Req] = None   # req that became chunked

    def _apply_prefix_match(self, req: Req) -> None:
        """
        Call match_prefix for a brand-new request (prefix_len == 0).
        Sets req.prefix_*, slot_indices, kv_committed_len.
        Does nothing if radix_cache is None or req was already matched.
        """
        if self.radix_cache is None or req.prefix_len > 0 or req.last_node is not None:
            return
        page_indices, matched_len, last_node = self.radix_cache.match_prefix(
            req.input_ids
        )
        req.prefix_page_indices = page_indices
        req.prefix_len          = matched_len
        req.last_node           = last_node
        req.slot_indices        = list(page_indices)   # pre-populate
        req.kv_committed_len    = matched_len           # skip cached tokens
        self.radix_cache.inc_lock_ref(last_node)
        if matched_len > 0:
            logger.info(
                f"  prefix hit rid={req.rid[:8]}  "
                f"matched={matched_len}/{req.prompt_len} tokens  "
                f"({len(page_indices)} pages)"
            )

    def build(self) -> List[Req]:
        """
        Return the list of requests for the next prefill_batch call.
        Sets fill_ids and extend_input_len on each returned request.
        Also sets self.new_chunked_req if a new chunk was started.
        """
        # ── Case 1: continue an in-flight chunked request ─────────────────
        if self.chunked_req is not None:
            req   = self.chunked_req
            start = req.kv_committed_len
            size  = self.chunked_prefill_size or (req.prompt_len - start)
            end   = min(start + size, req.prompt_len)
            req.fill_ids         = req.input_ids[start:end]
            req.extend_input_len = end - start
            logger.debug(
                f"  chunked_req rid={req.rid[:8]} "
                f"chunk [{start}:{end}] / {req.prompt_len}"
            )
            return [req]

        # ── Case 2: pick new requests from waiting_queue ───────────────────
        batch: List[Req] = []
        rem_tokens = self.max_prefill_tokens

        while True:
            # Stop if decode batch is full.
            if self.running_count + len(batch) >= self.max_running_reqs:
                break
            # Stop if token budget exhausted.
            if rem_tokens <= 0 and batch:
                break
            if self.waiting.empty():
                break

            req = self.waiting.queue[0]   # peek

            # Match prefix for any brand-new request before deciding cost.
            # We peek first, match, then decide whether to dequeue.
            self._apply_prefix_match(req)
            effective_len = req.prompt_len - req.prefix_len   # tokens to compute

            if self.chunked_prefill_size and effective_len > self.chunked_prefill_size:
                # This request needs to be chunked (even after prefix reduction).
                req = self.waiting.get_nowait()
                chunk_end = req.prefix_len + min(
                    self.chunked_prefill_size, effective_len
                )
                req.fill_ids         = req.input_ids[req.prefix_len:chunk_end]
                req.extend_input_len = len(req.fill_ids)
                self.new_chunked_req = req
                batch.append(req)
                break   # only one chunked request per round

            if effective_len > rem_tokens and batch:
                # Doesn't fit in remaining budget; defer to next round.
                # Note: prefix match is already set on this req; it will be
                # reused when the req is dequeued next round (last_node != None
                # guards against double-matching in _apply_prefix_match).
                break

            req = self.waiting.get_nowait()
            req.fill_ids         = req.input_ids[req.prefix_len:]
            req.extend_input_len = len(req.fill_ids)
            batch.append(req)
            rem_tokens -= effective_len

        return batch


# ─────────────────────────────────────────────────────────────────────────────
# Scheduler
# ─────────────────────────────────────────────────────────────────────────────

class Scheduler:

    def __init__(
        self,
        model_runner: ModelRunner,
        max_running_reqs: int = 16,
        chunked_prefill_size: int = 0,      # 0 = no chunking
        max_prefill_tokens: int = 4096,     # token budget per extend round
    ) -> None:
        self.model_runner         = model_runner
        self.max_running_reqs     = max_running_reqs
        self.chunked_prefill_size = chunked_prefill_size
        self.max_prefill_tokens   = max_prefill_tokens
        # Prefix cache lives on ModelRunner; scheduler just holds a reference.
        self.radix_cache: Optional[RadixCache] = model_runner.radix_cache

        self._waiting: queue.Queue[Req] = queue.Queue()
        self._running: List[Req] = []
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # In-flight chunked request (None if no request is mid-prefill).
        self._chunked_req: Optional[Req] = None

        self._n_prefilled = 0
        self._n_finished  = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_request(self, req: Req) -> None:
        self._waiting.put(req)
        logger.debug(f"queued rid={req.rid[:8]} prompt_len={req.prompt_len}")

    # ------------------------------------------------------------------
    # Main scheduler loop
    # ------------------------------------------------------------------

    def run(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        logger.info(
            f"Scheduler started  max_running={self.max_running_reqs}  "
            f"chunked_prefill_size={self.chunked_prefill_size}  "
            f"max_prefill_tokens={self.max_prefill_tokens}  "
            f"prefix_caching={'on' if self.radix_cache else 'off'}"
        )

        while True:
            did_work = False

            # ── Step 1: Build and run the extend (prefill) batch ──────────
            adder = PrefillAdder(
                waiting              = self._waiting,
                running_count        = len(self._running),
                max_running_reqs     = self.max_running_reqs,
                max_prefill_tokens   = self.max_prefill_tokens,
                chunked_prefill_size = self.chunked_prefill_size,
                chunked_req          = self._chunked_req,
                radix_cache          = self.radix_cache,
            )
            prefill_batch = adder.build()

            if prefill_batch:
                n_toks = sum(r.extend_input_len for r in prefill_batch)
                logger.info(
                    f"extend B={len(prefill_batch)} tokens={n_toks}  "
                    f"chunked={'yes' if self._chunked_req else 'no'}  "
                    f"running={len(self._running)}"
                )
                self.model_runner.prefill_batch(prefill_batch)
                self._n_prefilled += len(prefill_batch)
                did_work = True

                # Update chunked_req.
                # If a new chunked request was started, register it.
                if adder.new_chunked_req is not None:
                    self._chunked_req = adder.new_chunked_req

                # Route each request based on status after prefill_batch.
                for req in prefill_batch:
                    if req.status == ReqStatus.FINISHED:
                        self._resolve(req)
                        self._n_finished += 1
                        if req is self._chunked_req:
                            self._chunked_req = None
                    elif req.status == ReqStatus.RUNNING:
                        self._running.append(req)
                        if req is self._chunked_req:
                            self._chunked_req = None
                    elif req.status == ReqStatus.PREFILLING:
                        # chunked_req stays set; req will be continued next round
                        pass

            # ── Step 2: Decode step ────────────────────────────────────────
            if self._running:
                newly_finished = self.model_runner.decode_step(self._running)
                did_work = True

                for req in newly_finished:
                    self._resolve(req)
                    self._n_finished += 1
                    logger.info(
                        f"finished rid={req.rid[:8]}  "
                        f"out_len={req.output_len}  "
                        f"latency={req.latency_ms:.0f}ms"
                    )

                self._running = [
                    r for r in self._running if r.status == ReqStatus.RUNNING
                ]

            # ── Step 3: Idle ───────────────────────────────────────────────
            if not did_work:
                time.sleep(IDLE_SLEEP_S)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve(self, req: Req) -> None:
        text = self.model_runner.decode_output(req)
        result = {
            "text":              text,
            "prompt_tokens":     req.prompt_len,
            "completion_tokens": req.output_len,
            "ttft_ms":           round(req.ttft_ms, 1),
            "latency_ms":        round(req.latency_ms, 1),
        }
        self._loop.call_soon_threadsafe(req.future.set_result, result)
