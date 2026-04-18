"""
Layer 10 — Scheduler: batched prefill + chunked prefill.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changed from Layer 9
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Layer 9 scheduler loop:
  while True:
    for each waiting req:
      model_runner.prefill(req)   ← B=1, one at a time
    if running:
      model_runner.decode_step(running)

Layer 10 scheduler loop:
  while True:
    batch = PrefillAdder.build()  ← picks N requests to extend this round
    if batch:
      model_runner.prefill_batch(batch)   ← one B=N EXTEND forward
      route each req to running / chunked_req / finished
    if running:
      model_runner.decode_step(running)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
PrefillAdder (simplified vs SGLang)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Mirrors SGLang's PrefillAdder (schedule_policy.py) but stripped to essentials:

  • max_prefill_tokens  — total token budget per extend round (across all reqs)
  • chunked_prefill_size — max tokens from one request per round (0 = unlimited)
  • FIFO scheduling (no priority; SGLang adds priority, preemption etc.)
  • One chunked request at a time (SGLang supports N chunked)

Decision logic:
  1. If self.chunked_req is not None:
       Continue it: set fill_ids to the next chunk slice.
       Only this one request is prefilled this round.
  2. Else: drain waiting_queue, up to max_prefill_tokens total.
       For each candidate request:
         If its prompt fits in the remaining budget → take it fully.
         If it's too long and chunked_prefill_size is set → take first chunk,
           mark as PREFILLING, store in chunked_req, stop filling this round.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
chunked_req state machine
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  self.chunked_req = None means no request is mid-prefill.

  When a request needs chunking:
    chunk 0:   chunked_req = req,  req.status = PREFILLING
    chunk k:   chunked_req = req (same),  req.status = PREFILLING
    last chunk: req.status = RUNNING or FINISHED,  chunked_req = None

  During chunks 0..N-1, the request is NOT in self._running.
  It only joins the decode batch after the last chunk completes.

  Each round while chunked_req is set:
    • decode_step() still runs for all existing _running requests.
    • prefill_batch([chunked_req]) runs the next chunk.
  This interleaves chunked prefill with ongoing decode.
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
from request import Req, ReqStatus

logger = logging.getLogger(__name__)

IDLE_SLEEP_S = 0.001


# ─────────────────────────────────────────────────────────────────────────────
# PrefillAdder — decides which requests to extend this round
# ─────────────────────────────────────────────────────────────────────────────

class PrefillAdder:
    """
    Simplified SGLang PrefillAdder.

    Given the waiting queue and current state, builds the list of requests to
    include in the next EXTEND forward pass and sets fill_ids / extend_input_len
    on each selected request.

    Args:
      waiting:              thread-safe queue of Req
      running_count:        number of requests currently in decode batch
      max_running_reqs:     hard limit on decode batch size
      max_prefill_tokens:   token budget for this extend round (across all reqs)
      chunked_prefill_size: max tokens per request per round (0 = unlimited)
      chunked_req:          in-flight chunked request (None if none)
    """

    def __init__(
        self,
        waiting: queue.Queue,
        running_count: int,
        max_running_reqs: int,
        max_prefill_tokens: int,
        chunked_prefill_size: int,
        chunked_req: Optional[Req],
    ) -> None:
        self.waiting              = waiting
        self.running_count        = running_count
        self.max_running_reqs     = max_running_reqs
        self.max_prefill_tokens   = max_prefill_tokens
        self.chunked_prefill_size = chunked_prefill_size
        self.chunked_req          = chunked_req

        # Set by build():
        self.new_chunked_req: Optional[Req] = None   # req that became chunked

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

            if self.chunked_prefill_size and req.prompt_len > self.chunked_prefill_size:
                # This request needs to be chunked.
                req = self.waiting.get_nowait()
                chunk_end = min(self.chunked_prefill_size, req.prompt_len)
                req.fill_ids         = req.input_ids[:chunk_end]
                req.extend_input_len = chunk_end
                self.new_chunked_req = req
                batch.append(req)
                break   # only one chunked request per round

            if req.prompt_len > rem_tokens and batch:
                # Doesn't fit in remaining budget; defer to next round.
                break

            req = self.waiting.get_nowait()
            req.fill_ids         = req.input_ids
            req.extend_input_len = req.prompt_len
            batch.append(req)
            rem_tokens -= req.prompt_len

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
            f"max_prefill_tokens={self.max_prefill_tokens}"
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
