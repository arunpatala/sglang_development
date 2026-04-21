"""
Layer 13 — SpecRunner: standalone speculative decoding (greedy / temperature=0).

┌─────────────────────────────────────────────────────────────────────────┐
│ Overview                                                                │
└─────────────────────────────────────────────────────────────────────────┘

Speculative decoding accelerates inference by having a fast *draft* model
propose N candidate tokens in serial, then verifying all N+1 positions
(the last confirmed token + N proposals) with the large *target* model in
a *single* parallel extend pass.

  Draft model:  Qwen/Qwen3-0.6B   — small, fast, ~70 % accuracy vs target
  Target model: Qwen/Qwen3-1.7B   — correct by definition

For greedy (temperature=0) decoding the accept/reject rule is trivial:
a draft token d_k is accepted iff target.argmax(pos_k) == d_k.

┌─────────────────────────────────────────────────────────────────────────┐
│ One spec-decode iteration (per-request view)                            │
└─────────────────────────────────────────────────────────────────────────┘

  State before:
    confirmed_tokens = [..., t_last]   (everything target has accepted so far)
    Both model KV caches cover the same history.

  ── DRAFT PHASE ────────────────────────────────────────────────────────
  Run N autoregressive decode steps with the 0.6B model:

      d1 = draft.argmax( forward(t_last)  )   # step 1
      d2 = draft.argmax( forward(d1)      )   # step 2
      ...
      dN = draft.argmax( forward(d_{N-1}) )   # step N

  Draft KV grows by N new positions.

  ── VERIFY PHASE ────────────────────────────────────────────────────────
  One EXTEND pass of the 1.7B model over [t_last, d1, d2, ..., dN]:

      v1, v2, ..., v_{N+1} = target.forward([t_last, d1, ..., dN])

  All N+1 logits computed in parallel via causal attention.

  ── ACCEPT / REJECT ─────────────────────────────────────────────────────
  Find the longest accepted prefix (greedy comparison):

      s_i = argmax(v_i)
      accept while s_i == d_i,  stop at first mismatch.

  Let k = number of accepted tokens  (0 ≤ k ≤ N).
  Emit k+1 new tokens:  [d1, ..., dk, s_{k+1}]
  (s_{k+1} is the *bonus* token: target's correction at the rejection site)

  ── KV REWIND ───────────────────────────────────────────────────────────
  Target: extended N+1 positions → keep only k+1 → free N-k extra pages.
  Draft:  ran N steps → keep only k → free N-k newest pages.

┌─────────────────────────────────────────────────────────────────────────┐
│ Memory layout                                                           │
└─────────────────────────────────────────────────────────────────────────┘

  Target has its own KVPool + ReqToTokenPool (managed by target ModelRunner).
  Draft  has its own KVPool + ReqToTokenPool (managed by draft ModelRunner).
  The two pools are completely independent.

  Each logical request r has:
    r            — the "canonical" Req (tracks confirmed output_ids, etc.)
    _draft_reqs[r] — a mirror Req used internally by the draft ModelRunner

┌─────────────────────────────────────────────────────────────────────────┐
│ Position convention note                                                │
└─────────────────────────────────────────────────────────────────────────┘

  prefill_batch positions:  0 .. T-1  (prompt, kv_committed_len = T)
  verify extend  positions: T .. T+N  (kv_committed_len + 0 .. +N)
  draft decode   positions: based on seq_len = len(input_ids)+len(output_ids)

  Both models use the same position convention internally, so cross-model
  comparisons (accept/reject on token ids) remain valid.
"""

from __future__ import annotations

import logging
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import flashinfer

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import (
    DecodeKVCtx,
    ExtendKVCtx,
    KVPool,
    ReqToTokenPool,
    WriteInfo,
    compute_write_info,
)
from model_runner import ModelRunner
from request import Req, ReqStatus
from triton_utils import create_flashinfer_kv_indices_triton

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16


# ─────────────────────────────────────────────────────────────────────────────
# SpecRunner
# ─────────────────────────────────────────────────────────────────────────────

class SpecRunner:
    """
    Orchestrates standalone speculative decoding.

    Usage::

        runner = SpecRunner(
            target_path = "Qwen/Qwen3-1.7B",
            draft_path  = "Qwen/Qwen3-0.6B",
            N = 5,
        )

        # Prefill — both models process the prompt; target produces the first token.
        runner.prefill([req])

        # Spec-decode — iterate until all requests finish.
        while running_reqs:
            finished = runner.spec_decode_step(running_reqs)
            for r in finished:
                running_reqs.remove(r)
    """

    def __init__(
        self,
        target_path: str,
        draft_path:  str,
        N: int   = 5,
        page_size: int = 16,
        enable_prefix_caching: bool = False,
        target_kv_fraction: float = 0.35,
        draft_kv_fraction:  float = 0.45,
    ) -> None:
        self.N         = N
        self.page_size = page_size

        # Load target first, allocating only a fraction of free VRAM for its KV
        # pool so that the draft model and its KV pool can load afterwards.
        logger.info("SpecRunner: loading target model …")
        self.target = ModelRunner(
            target_path,
            page_size              = page_size,
            enable_prefix_caching  = enable_prefix_caching,
            kv_memory_fraction     = target_kv_fraction,
        )

        # Draft model loads into the VRAM left after target model + target KV.
        logger.info("SpecRunner: loading draft model …")
        self.draft = ModelRunner(
            draft_path,
            page_size              = page_size,
            enable_prefix_caching  = enable_prefix_caching,
            kv_memory_fraction     = draft_kv_fraction,
        )

        self.tokenizer = self.target.tokenizer
        self.eos_id    = self.target.eos_id

        # Mirror Req objects for the draft model (keyed by id(target_req))
        self._draft_reqs: Dict[int, Req] = {}

        # Telemetry
        self.total_draft_tokens:    int = 0
        self.total_accepted_tokens: int = 0
        self.total_spec_steps:      int = 0

    # ------------------------------------------------------------------
    # Public: prefill both models
    # ------------------------------------------------------------------

    def prefill(self, reqs: List[Req]) -> None:
        """
        Prefill both target and draft models with the prompts in `reqs`.

        After this call:
          • req.output_ids = [first_token]  (target's prediction)
          • req.kv_committed_len = len(req.input_ids)  (target KV committed)
          • Draft model has processed the same prompts independently.
          • req.status is RUNNING (or FINISHED if first token is EOS).
        """
        import asyncio

        # ── 1. Target prefill ─────────────────────────────────────────────
        for req in reqs:
            req.fill_ids         = list(req.input_ids)
            req.extend_input_len = len(req.input_ids)
            req.kv_committed_len = 0

        self.target.prefill_batch(reqs)
        # req.output_ids = [first_token], req.kv_committed_len = T

        # ── 2. Draft prefill ──────────────────────────────────────────────
        # Create a mirror Req for each request and prefill the draft model.
        d_reqs = []
        for req in reqs:
            d_req = Req(
                rid            = f"draft_{req.rid}",
                input_ids      = list(req.input_ids),
                max_new_tokens = 99_999,   # never auto-finish
                temperature    = 0.0,
                future         = req.future,
            )
            d_req.fill_ids         = list(req.input_ids)
            d_req.extend_input_len = len(req.input_ids)
            d_req.kv_committed_len = 0
            d_reqs.append(d_req)

        self.draft.prefill_batch(d_reqs)
        # d_req.kv_committed_len = T; d_req.output_ids = [draft_first_token]

        # ── 3. Link draft reqs; sync output_ids with target ───────────────
        for req, d_req in zip(reqs, d_reqs):
            # Discard draft's own first-token prediction.
            # The confirmed sequence is determined entirely by the target.
            d_req.output_ids = list(req.output_ids)   # [t0]
            self._draft_reqs[id(req)] = d_req

        logger.info(
            f"Prefilled {len(reqs)} request(s).  "
            f"Target KV pages={self.target.kv_pool.total_pages - self.target.kv_pool.available()}  "
            f"Draft KV pages={self.draft.kv_pool.total_pages - self.draft.kv_pool.available()}"
        )

    # ------------------------------------------------------------------
    # Public: one speculative decode iteration
    # ------------------------------------------------------------------

    def spec_decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        Run one speculative decoding iteration for all active requests.

        Returns the list of newly finished requests (status=FINISHED).
        Finished requests are cleaned up (KV freed, draft req removed).
        """
        newly_finished: List[Req] = []

        for req in reqs:
            if req.status != ReqStatus.RUNNING:
                continue

            d_req = self._draft_reqs[id(req)]

            # ── sync confirmed state into draft req ───────────────────────
            # d_req.output_ids must end with the same token as req, so that
            # draft.decode_step uses req.output_ids[-1] as the next input.
            d_req.output_ids = list(req.output_ids)

            # ── DRAFT PHASE ───────────────────────────────────────────────
            draft_tokens, new_pages_per_step = self._draft_phase(d_req)
            self.total_draft_tokens += self.N

            # ── VERIFY PHASE ──────────────────────────────────────────────
            # Extend target with [last_confirmed, d1, ..., dN]
            last_confirmed  = req.output_ids[-1]
            verify_tokens   = [last_confirmed] + draft_tokens   # N+1 tokens

            verify_logits = self._verify_extend(req, verify_tokens)
            # verify_logits: [N+1, vocab]

            # ── ACCEPT / REJECT ───────────────────────────────────────────
            accept_len, bonus_token = self._accept_reject(
                draft_tokens  = draft_tokens,
                verify_logits = verify_logits,
            )
            # accept_len ∈ [0, N]; bonus_token is always emitted
            self.total_accepted_tokens += accept_len
            self.total_spec_steps      += 1

            logger.debug(
                f"req={req.rid}  draft={draft_tokens}  "
                f"accept={accept_len}/{self.N}  bonus={bonus_token}"
            )

            # ── EMIT CONFIRMED TOKENS ─────────────────────────────────────
            for tok in draft_tokens[:accept_len]:
                req.output_ids.append(tok)
            req.output_ids.append(bonus_token)

            # ── KV REWIND ─────────────────────────────────────────────────
            # Target: verify extended N+1 positions; keep only accept_len+1.
            self._rewind_target_kv(req, accept_len)
            # After rewind: req.slot_indices covers exactly the kept positions.

            # Draft: ran N decode steps; keep only accept_len positions.
            self._rewind_draft_kv(d_req, new_pages_per_step, accept_len)

            # ── UPDATE kv_committed_len ───────────────────────────────────
            # The target's committed KV grew by (accept_len + 1) positions.
            req.kv_committed_len += accept_len + 1

            # ── EOS / MAX-TOKENS CHECK ────────────────────────────────────
            hit_eos  = (bonus_token == self.eos_id) or any(
                t == self.eos_id for t in draft_tokens[:accept_len]
            )
            hit_max  = len(req.output_ids) >= req.max_new_tokens

            if hit_eos or hit_max:
                req.status   = ReqStatus.FINISHED
                req.t_finish = time.perf_counter()
                self._cleanup_req(req, d_req)
                newly_finished.append(req)

        return newly_finished

    # ------------------------------------------------------------------
    # Decode output
    # ------------------------------------------------------------------

    def decode_output(self, req: Req) -> str:
        return self.target.decode_output(req)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    @property
    def acceptance_rate(self) -> float:
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens

    @property
    def tokens_per_step(self) -> float:
        if self.total_spec_steps == 0:
            return 0.0
        return (self.total_accepted_tokens + self.total_spec_steps) / self.total_spec_steps

    def stats_str(self) -> str:
        return (
            f"acceptance_rate={self.acceptance_rate:.2%}  "
            f"tokens_per_step={self.tokens_per_step:.2f}  "
            f"total_spec_steps={self.total_spec_steps}  "
            f"total_accepted={self.total_accepted_tokens}  "
            f"total_draft={self.total_draft_tokens}"
        )

    # ------------------------------------------------------------------
    # Internal: draft phase
    # ------------------------------------------------------------------

    def _draft_phase(
        self,
        d_req: Req,
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Run N autoregressive decode steps with the draft model.

        Returns:
          draft_tokens:        [d1, d2, ..., dN]
          new_pages_per_step:  list of N lists; each list = pages newly
                               allocated by the draft KV pool at that step.
                               Used by _rewind_draft_kv to undo rejected steps.
        """
        draft_tokens       : List[int]       = []
        new_pages_per_step : List[List[int]] = []

        for _ in range(self.N):
            len_before = len(d_req.slot_indices)
            self.draft.decode_step([d_req])
            new_pages = list(d_req.slot_indices[len_before:])
            new_pages_per_step.append(new_pages)
            draft_tokens.append(d_req.output_ids[-1])

        return draft_tokens, new_pages_per_step

    # ------------------------------------------------------------------
    # Internal: verify extend (target model)
    # ------------------------------------------------------------------

    def _verify_extend(
        self,
        req:           Req,
        verify_tokens: List[int],
    ) -> torch.Tensor:
        """
        Run one extend forward of the TARGET model with `verify_tokens`.

        `verify_tokens` = [last_confirmed, d1, ..., dN]  (N+1 tokens).
        Positions:         kv_committed_len .. kv_committed_len + N

        The KV for all N+1 positions is written into the target pool so that
        subsequent iterations can attend to the accepted tokens.

        Returns:
          logits: Tensor [N+1, vocab]
        """
        P   = self.page_size
        cfg = self.target.model.model.config
        n   = len(verify_tokens)   # N+1

        # ── Allocate KV pages for the new tokens ──────────────────────────
        write_info = compute_write_info(
            kv_pool          = self.target.kv_pool,
            rtp              = self.target.req_to_token_pool,
            slot_indices     = req.slot_indices,
            req_pool_idx     = req.req_pool_idx,
            kv_committed_len = req.kv_committed_len,
            n_fill           = n,
        )

        # ── Build input tensors ───────────────────────────────────────────
        ids_t = torch.tensor(
            [verify_tokens], dtype=torch.long, device=DEVICE
        )  # [1, N+1]

        pos_list = list(range(req.kv_committed_len, req.kv_committed_len + n))
        pos_t    = torch.tensor([pos_list], dtype=torch.long, device=DEVICE)

        # ── FlashInfer extend metadata ────────────────────────────────────
        total_committed = req.kv_committed_len + n
        n_pages         = len(req.slot_indices)
        last_fill       = total_committed % P
        kv_last_pg_len  = last_fill if last_fill != 0 else P

        qo_indptr       = torch.tensor([0, n],       dtype=torch.int32, device=DEVICE)
        kv_indptr       = torch.tensor([0, n_pages], dtype=torch.int32, device=DEVICE)
        kv_last_pg_lens = torch.tensor([kv_last_pg_len], dtype=torch.int32, device=DEVICE)

        req_pool_idx_t = torch.tensor(
            [req.req_pool_idx], dtype=torch.int32, device=DEVICE
        )
        num_pages_t = torch.tensor([n_pages], dtype=torch.int32, device=DEVICE)
        kv_indices  = torch.empty(n_pages, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(1,)](
            self.target.req_to_token_pool.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr,
            None,
            kv_indices,
            self.target.req_to_token_pool.req_to_token.shape[1],
        )

        # ── FlashInfer begin_forward ──────────────────────────────────────
        self.target._extend_wrapper.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_pg_lens,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            P,
            causal       = True,
            q_data_type  = DTYPE,
        )

        ctx = ExtendKVCtx(
            wrapper     = self.target._extend_wrapper,
            k_pool      = self.target.kv_pool.k_pool,
            v_pool      = self.target.kv_pool.v_pool,
            qo_indptr   = [0, n],
            write_infos = [write_info],
            page_size   = P,
        )

        # ── Forward pass (no sampling) ────────────────────────────────────
        with torch.no_grad():
            logits_3d = self.target.model(
                ids_t,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_t,
            )  # [1, N+1, vocab]

        self.target._extend_wrapper.end_forward()

        return logits_3d[0]   # [N+1, vocab]

    # ------------------------------------------------------------------
    # Internal: greedy accept/reject
    # ------------------------------------------------------------------

    @staticmethod
    def _accept_reject(
        draft_tokens:  List[int],
        verify_logits: torch.Tensor,   # [N+1, vocab]
    ) -> Tuple[int, int]:
        """
        Greedy accept/reject rule.

        For each position i ∈ [0, N-1]:
          target_token[i] = argmax(verify_logits[i])
          if target_token[i] == draft_tokens[i]: accept
          else: reject (stop here)

        The bonus token is argmax(verify_logits[accept_len]).

        Returns: (accept_len, bonus_token)
          accept_len  ∈ [0, N]
          bonus_token — always emitted regardless of accept_len
        """
        N = len(draft_tokens)

        # Target's greedy predictions at positions 0..N (N+1 total)
        target_tokens = verify_logits.argmax(dim=-1).tolist()   # [N+1]

        accept_len = 0
        for i in range(N):
            if target_tokens[i] == draft_tokens[i]:
                accept_len += 1
            else:
                break

        bonus_token = target_tokens[accept_len]   # target's correction / next
        return accept_len, bonus_token

    # ------------------------------------------------------------------
    # Internal: KV rewind
    # ------------------------------------------------------------------

    def _rewind_target_kv(self, req: Req, accept_len: int) -> None:
        """
        After the verify extend wrote KV for N+1 positions (indices
        kv_committed_len .. kv_committed_len+N), retain only the first
        accept_len+1 of them and free the rest.

        The verify extend is done via compute_write_info (paged allocation),
        so the KV length maps cleanly to page count via ceil division.
        """
        kept_kv_len = req.kv_committed_len + accept_len + 1
        pages_needed = math.ceil(kept_kv_len / self.page_size)

        if pages_needed < len(req.slot_indices):
            pages_to_free = req.slot_indices[pages_needed:]
            self.target.kv_pool.free(pages_to_free)
            req.slot_indices = req.slot_indices[:pages_needed]

    def _rewind_draft_kv(
        self,
        d_req:             Req,
        new_pages_per_step: List[List[int]],
        accept_len:         int,
    ) -> None:
        """
        After N draft decode steps allocated pages in new_pages_per_step[0..N-1],
        free the pages from steps accept_len..N-1 (the rejected steps).

        slot_indices is updated to exclude the freed pages.
        """
        pages_to_free: List[int] = []
        for step_pages in new_pages_per_step[accept_len:]:
            pages_to_free.extend(step_pages)

        if pages_to_free:
            for page in pages_to_free:
                d_req.slot_indices.remove(page)
            self.draft.kv_pool.free(pages_to_free)

    # ------------------------------------------------------------------
    # Internal: cleanup
    # ------------------------------------------------------------------

    def _cleanup_req(self, req: Req, d_req: Req) -> None:
        """Free all KV resources for a finished request."""
        self.target.kv_pool.free(req.slot_indices)
        self.target.req_to_token_pool.free(req.req_pool_idx)

        self.draft.kv_pool.free(d_req.slot_indices)
        self.draft.req_to_token_pool.free(d_req.req_pool_idx)

        del self._draft_reqs[id(req)]
        req.slot_indices   = []
        req.req_pool_idx   = None
        d_req.slot_indices = []
        d_req.req_pool_idx = None
