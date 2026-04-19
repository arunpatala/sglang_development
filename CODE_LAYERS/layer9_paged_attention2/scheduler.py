"""
Layer 5 — Scheduler: the heart of continuous batching.

Mirrors SGLang's Scheduler (managers/scheduler.py) at a conceptual level:

  waiting_queue  — requests that have arrived but not yet been prefilled
  running_reqs   — requests currently in the decode batch

Event loop (runs in a background thread):

  while True:
    1. Drain waiting_queue → prefill each new request individually (B=1)
       and move it to running_reqs.
    2. If running_reqs is non-empty → one batched decode step (B=N).
    3. Remove newly finished requests, resolve their asyncio Futures.
    4. If both queues are empty → sleep briefly to avoid spinning.

Key design decisions vs layer3 static batching:

  • No head-of-line blocking: a long request doesn't stall short ones.
    Short requests finish and free slots while long ones are still decoding.

  • Dynamic batch size: running_reqs grows as new requests arrive and
    shrinks as requests finish, without waiting for a full batch.

  • Per-request KV cache: each Req owns its own PerReqKVCache so that
    requests can join and leave the batch independently.

Thread safety:
  • The scheduler loop runs on a dedicated thread.
  • add_request() is called from the FastAPI (asyncio) thread.
  • queue.Queue is thread-safe; no explicit locking needed.
  • Future resolution uses loop.call_soon_threadsafe() to safely
    schedule the callback on the asyncio event loop.
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

IDLE_SLEEP_S = 0.001   # 1 ms spin-sleep when nothing to do


class Scheduler:

    def __init__(
        self,
        model_runner: ModelRunner,
        max_running_reqs: int = 16,
    ) -> None:
        self.model_runner     = model_runner
        self.max_running_reqs = max_running_reqs

        # Thread-safe queue for incoming requests.
        # The asyncio thread puts; the scheduler thread gets.
        self._waiting: queue.Queue[Req] = queue.Queue()

        # Active decode batch — only touched by the scheduler thread.
        self._running: List[Req] = []

        # Set by Server before starting the scheduler thread so we can
        # call loop.call_soon_threadsafe() to resolve futures.
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self._n_prefilled  = 0
        self._n_finished   = 0

    # ------------------------------------------------------------------
    # Public API (called from the asyncio thread)
    # ------------------------------------------------------------------

    def add_request(self, req: Req) -> None:
        """Enqueue a request.  Thread-safe."""
        self._waiting.put(req)
        logger.debug(f"queued rid={req.rid[:8]} prompt_len={req.prompt_len}")

    # ------------------------------------------------------------------
    # Scheduler event loop (runs on its own background thread)
    # ------------------------------------------------------------------

    def run(self, loop: asyncio.AbstractEventLoop) -> None:
        """
        Main scheduler loop.  Call this in a daemon thread:
            threading.Thread(target=scheduler.run, args=(loop,), daemon=True).start()
        """
        self._loop = loop
        logger.info(
            f"Scheduler started  max_running={self.max_running_reqs}"
        )

        while True:
            did_work = False

            # ── Step 1: Prefill new requests ──────────────────────────
            # Drain as many as will fit in the running batch.
            while (
                not self._waiting.empty()
                and len(self._running) < self.max_running_reqs
            ):
                req = self._waiting.get_nowait()
                logger.info(
                    f"prefill rid={req.rid[:8]}  "
                    f"waiting={self._waiting.qsize()}  "
                    f"running={len(self._running)}"
                )
                self.model_runner.prefill(req)
                self._n_prefilled += 1
                did_work = True

                if req.status == ReqStatus.FINISHED:
                    # EOS on first token — resolve immediately, don't add to running
                    self._resolve(req)
                    self._n_finished += 1
                else:
                    self._running.append(req)

            # ── Step 2: Decode step for all running requests ──────────
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

                # Remove finished requests from the running batch.
                self._running = [
                    r for r in self._running if r.status == ReqStatus.RUNNING
                ]

            # ── Step 3: Idle ──────────────────────────────────────────
            if not did_work:
                time.sleep(IDLE_SLEEP_S)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _resolve(self, req: Req) -> None:
        """Resolve the request's asyncio Future from the scheduler thread."""
        text = self.model_runner.decode_output(req)
        result = {
            "text":              text,
            "prompt_tokens":     req.prompt_len,
            "completion_tokens": req.output_len,
            "ttft_ms":           round(req.ttft_ms, 1),
            "latency_ms":        round(req.latency_ms, 1),
        }
        # call_soon_threadsafe schedules the callback on the asyncio thread.
        self._loop.call_soon_threadsafe(req.future.set_result, result)
