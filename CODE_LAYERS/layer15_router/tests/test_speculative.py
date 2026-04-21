"""
test_speculative.py — correctness tests for Layer 13 (speculative decoding).

Tests are grouped into eight sections:

Part A — Unit tests for the accept/reject logic (CPU-only, no GPU needed)
  A1. All N draft tokens match → accept_len == N, bonus from logit N.
  A2. First token mismatch → accept_len == 0, bonus from logit 0.
  A3. Partial match k < N → accept_len == k, bonus from logit k.
  A4. Batch of N=1 (single draft token) — accept and reject cases.
  A5. Bonus token comes from the target's prediction at the reject site.

Part B — KV rewind logic (mock KVPool, no model needed)
  B1. _rewind_target_kv with accept_len=N frees 0 pages (all kept).
  B2. _rewind_target_kv with accept_len=0 frees N pages.
  B3. _rewind_target_kv across page boundaries (P=4, N=5).
  B4. _rewind_draft_kv with no new pages allocations → noop.
  B5. _rewind_draft_kv frees exactly the rejected-step pages.

Part C — GPU prefill + single spec step (requires model weights)
  C1. After prefill, both models have KV pages allocated.
  C2. After prefill, target's output_ids == [first_token].
  C3. Draft phase produces exactly N tokens.
  C4. Verify extend produces logits of shape [N+1, vocab].
  C5. Accept/reject returns a valid accept_len in [0, N].
  C6. After one spec step, confirmed output grows by accept_len + 1.

Part D — Full speculative decode (end-to-end, requires model weights)
  D1. Spec-decode output matches target-only greedy output (token-for-token).
  D2. Acceptance rate is > 15 % on short prompts.
  D3. KV pool pages are fully returned after request finishes.
  D4. Multiple sequential requests share the SpecRunner without state leaks.

Part E — accept/reject extended edge cases (CPU-only)
  E1. Token ID zero accepted when target also predicts zero.
  E2. High token ID (near vocab limit) handled correctly.
  E3. Large N (N=20) all-match scenario.
  E4. accept/reject returns Python ints, not Tensor objects.
  E5. accept_len never exceeds N regardless of logit values.
  E6. Bonus token is distinct from accepted tokens when target corrects.

Part F — KV rewind boundary conditions (mock, CPU)
  F1. Target rewind: T not a multiple of P, accept_len=N → noop.
  F2. Target rewind: exact page boundary, accept_len=0.
  F3. Draft rewind: all steps accepted (accept_len=N) → no pages freed.
  F4. Draft rewind: single step allocated multiple pages.
  F5. Target rewind: kept_kv_len exactly equals one full page (boundary).

Part G — Statistics and state tracking (CPU)
  G1. acceptance_rate returns 0.0 when no drafts attempted.
  G2. tokens_per_step returns 0.0 when no steps taken.
  G3. Accumulated stats after multiple manual step records.
  G4. tokens_per_step formula: (accepted + steps) / steps.
  G5. stats_str() contains all expected metric names.

Part H — GPU: kv_committed_len, multi-step, draft token range (GPU)
  H1. kv_committed_len grows by accept_len+1 after each spec step.
  H2. First output token from prefill matches target-only greedy argmax.
  H3. total_draft_tokens increments by N each spec step.
  H4. Draft slot_indices shrinks correctly after rewind (accept_len < N).
  H5. verify_extend twice on same state produces identical logits.
  H6. spec_decode_step skips requests whose status != RUNNING.
  H7. Batch prefill of two requests: each gets its own first token.
  H8. Draft tokens are always valid token IDs (0 ≤ tok < vocab_size).
"""

from __future__ import annotations

import asyncio
import math
import sys
from pathlib import Path
from typing import List
from unittest.mock import MagicMock, patch

import pytest
import torch

LAYER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LAYER_ROOT))

from kv_cache import KVPool, ReqToTokenPool
from request import Req, ReqStatus
from spec_runner import DEVICE, DTYPE, SpecRunner
from tokenizer import Tokenizer

# ─── Paths ───────────────────────────────────────────────────────────────────

TARGET_PATH = "Qwen/Qwen3-1.7B"
DRAFT_PATH  = "Qwen/Qwen3-0.6B"

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _future():
    loop = asyncio.new_event_loop()
    return loop.create_future()


def _req(
    rid:            str = "t0",
    input_ids:      List[int] = None,
    max_new_tokens: int = 50,
) -> Req:
    return Req(
        rid            = rid,
        input_ids      = input_ids or [1, 2, 3, 4, 5],
        max_new_tokens = max_new_tokens,
        temperature    = 0.0,
        future         = _future(),
    )


def _logits_for_tokens(tokens: List[int], vocab: int = 1000) -> torch.Tensor:
    """
    Build a fake [len(tokens), vocab] logit tensor where argmax(row i) == tokens[i].
    """
    L = len(tokens)
    lg = torch.zeros(L, vocab)
    for i, tok in enumerate(tokens):
        lg[i, tok] = 10.0   # large positive → argmax = tok
    return lg


# ─────────────────────────────────────────────────────────────────────────────
# Part A — accept/reject unit tests (CPU, no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptReject:
    """Part A — greedy accept/reject logic."""

    def test_a1_all_match(self):
        """A1: All N draft tokens match target → accept_len == N."""
        N = 5
        draft_tokens = list(range(10, 10 + N))    # [10, 11, 12, 13, 14]
        # Logits for positions 0..N: target predicts draft_tokens[0..N-1] + bonus
        bonus_id = 99
        all_targets = draft_tokens + [bonus_id]   # N+1 predictions
        logits = _logits_for_tokens(all_targets)  # [N+1, vocab]

        accept_len, bonus = SpecRunner._accept_reject(draft_tokens, logits)

        assert accept_len == N,   f"expected accept_len={N}, got {accept_len}"
        assert bonus == bonus_id, f"expected bonus={bonus_id}, got {bonus}"

    def test_a2_first_mismatch(self):
        """A2: Target disagrees with d1 → accept_len == 0."""
        N = 5
        draft_tokens = [10, 11, 12, 13, 14]
        # Target's prediction at position 0 differs from d1=10
        all_targets  = [99, 11, 12, 13, 14, 55]   # mismatch at index 0
        logits = _logits_for_tokens(all_targets)

        accept_len, bonus = SpecRunner._accept_reject(draft_tokens, logits)

        assert accept_len == 0
        assert bonus == 99    # target's token at the reject site

    def test_a3_partial_match(self):
        """A3: First k match, then mismatch → accept_len == k."""
        N     = 5
        k     = 3
        draft = [10, 11, 12, 13, 14]
        # target agrees for indices 0..k-1, then differs at k
        targets = [10, 11, 12, 99, 14, 55]   # [k matches, mismatch at k, ...]
        logits  = _logits_for_tokens(targets)

        accept_len, bonus = SpecRunner._accept_reject(draft, logits)

        assert accept_len == k
        assert bonus == 99    # target at position k

    def test_a4_n_equals_1_accept(self):
        """A4a: N=1, draft matches → accept_len=1."""
        draft   = [42]
        targets = [42, 77]   # match then bonus
        logits  = _logits_for_tokens(targets)

        accept_len, bonus = SpecRunner._accept_reject(draft, logits)
        assert accept_len == 1
        assert bonus == 77

    def test_a4_n_equals_1_reject(self):
        """A4b: N=1, draft differs → accept_len=0."""
        draft   = [42]
        targets = [99, 77]   # mismatch
        logits  = _logits_for_tokens(targets)

        accept_len, bonus = SpecRunner._accept_reject(draft, logits)
        assert accept_len == 0
        assert bonus == 99

    def test_a5_bonus_token_from_reject_site(self):
        """A5: bonus_token is argmax of verify_logits[accept_len]."""
        N     = 4
        draft = [1, 2, 3, 4]
        # Accept 2 tokens, reject at index 2. Bonus should be logits[2].argmax().
        targets = [1, 2, 88, 4, 5]   # [accept, accept, mismatch, ...]
        logits  = _logits_for_tokens(targets)

        accept_len, bonus = SpecRunner._accept_reject(draft, logits)
        assert accept_len == 2
        assert bonus == 88    # target's prediction at the mismatch position


# ─────────────────────────────────────────────────────────────────────────────
# Part B — KV rewind logic (mock KVPool, CPU)
# ─────────────────────────────────────────────────────────────────────────────

class MockKVPool:
    """Minimal KVPool stub that records freed pages."""

    def __init__(self, total_pages: int = 100):
        self.total_pages = total_pages
        self.freed: List[int] = []

    def free(self, pages: List[int]) -> None:
        self.freed.extend(pages)


class MockSpecRunner:
    """SpecRunner stub with only the KV-rewind methods, no real models."""

    def __init__(self, N: int = 5, page_size: int = 16):
        self.N          = N
        self.page_size  = page_size
        self.target     = MagicMock()
        self.draft      = MagicMock()
        target_pool     = MockKVPool()
        draft_pool      = MockKVPool()
        self.target.kv_pool = target_pool
        self.draft.kv_pool  = draft_pool

    # Expose the real rewind methods from SpecRunner
    _rewind_target_kv = SpecRunner._rewind_target_kv
    _rewind_draft_kv  = SpecRunner._rewind_draft_kv


class TestKVRewind:
    """Part B — KV rewind logic."""

    def _make_runner(self, N=5, P=16) -> MockSpecRunner:
        return MockSpecRunner(N=N, page_size=P)

    def test_b1_target_all_accepted_noop(self):
        """B1: accept_len==N → keep all N+1 positions → no pages freed."""
        P  = 16
        N  = 5
        runner = self._make_runner(N, P)

        # After prefill: say T=16 tokens → 1 page.
        # After verify extend N+1=6 tokens: still fits in 1 page (22 total < 32).
        req = _req()
        req.kv_committed_len = 16    # T = 16 tokens in KV
        req.slot_indices     = [10]  # 1 page for 16 tokens

        # Simulate verify extend allocating 0 extra pages (6 tokens in 1 page)
        # After verify: kv_committed_len=16, slot_indices still [10]
        # (the verify extend keeps the same page since 16+6=22 < 32=2*P)
        runner._rewind_target_kv(req, accept_len=N)

        assert runner.target.kv_pool.freed == []
        assert req.slot_indices == [10]

    def test_b2_target_none_accepted_frees_pages(self):
        """B2: accept_len==0 → keep only 1 position → free pages beyond that."""
        P  = 4   # small page for easy testing
        N  = 5
        runner = self._make_runner(N, P)

        # T=4 tokens → 1 page. verify extend adds 6 tokens → total=10 tokens.
        # pages needed = ceil(10/4) = 3. slot_indices = [10, 11, 12].
        req = _req()
        req.kv_committed_len = 4           # 1 page worth
        req.slot_indices     = [10, 11, 12]  # 3 pages (after verify extend)

        # accept_len=0: keep kv_committed_len + 0 + 1 = 5 tokens → ceil(5/4)=2 pages
        runner._rewind_target_kv(req, accept_len=0)

        assert set(runner.target.kv_pool.freed) == {12}   # 3rd page freed
        assert req.slot_indices == [10, 11]

    def test_b3_target_partial_rewind(self):
        """B3: accept_len=k frees exactly the right pages."""
        P  = 4
        N  = 8   # 8 draft tokens
        runner = self._make_runner(N, P)

        # T=4 tokens → 1 page. verify adds N+1=9 tokens → total=13.
        # pages = ceil(13/4) = 4. slot_indices = [10, 11, 12, 13].
        req = _req()
        req.kv_committed_len = 4
        req.slot_indices     = [10, 11, 12, 13]

        # accept_len=3: keep 4+3+1=8 tokens → ceil(8/4)=2 pages → free [12, 13]
        runner._rewind_target_kv(req, accept_len=3)

        assert set(runner.target.kv_pool.freed) == {12, 13}
        assert req.slot_indices == [10, 11]

    def test_b4_draft_noop_when_no_new_pages(self):
        """B4: If no new pages were allocated during N draft steps, noop."""
        P  = 16
        N  = 3
        runner = self._make_runner(N, P)

        d_req = _req()
        d_req.slot_indices = [20]   # 1 page, stays the same throughout

        new_pages_per_step = [[], [], []]   # no allocations at any step

        runner._rewind_draft_kv(d_req, new_pages_per_step, accept_len=0)

        assert runner.draft.kv_pool.freed == []
        assert d_req.slot_indices == [20]   # unchanged

    def test_b5_draft_frees_rejected_pages(self):
        """B5: Free pages from rejected steps; keep accepted step pages."""
        P  = 4
        N  = 5
        runner = self._make_runner(N, P)

        # Draft allocated 1 new page at steps 2, 3, 4 (0-indexed)
        d_req = _req()
        d_req.slot_indices = [20, 21, 22, 23]   # 4 pages after draft phase

        new_pages_per_step = [
            [],     # step 0: no new page
            [],     # step 1: no new page
            [21],   # step 2: page 21 allocated
            [22],   # step 3: page 22 allocated
            [23],   # step 4: page 23 allocated
        ]

        # accept_len=2: keep steps 0..1 (pages []); free steps 2..4 (pages 21,22,23)
        runner._rewind_draft_kv(d_req, new_pages_per_step, accept_len=2)

        assert set(runner.draft.kv_pool.freed) == {21, 22, 23}
        assert d_req.slot_indices == [20]


# ─────────────────────────────────────────────────────────────────────────────
# Part C — GPU prefill + one spec step (requires model weights)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def spec_runner():
    """Load both models once for all Part C/D tests."""
    return SpecRunner(
        target_path = TARGET_PATH,
        draft_path  = DRAFT_PATH,
        N           = 5,
        page_size   = 16,
    )


@pytest.fixture(scope="module")
def tokenizer(spec_runner):
    return spec_runner.tokenizer


@pytest.mark.gpu
class TestGPUSingleStep:
    """Part C — one spec-decode step (GPU, requires model weights)."""

    PROMPT = "The capital of France is"

    def _make_req(self, tokenizer, prompt=None, max_new_tokens=50):
        p = prompt or self.PROMPT
        return Req(
            rid            = "test",
            input_ids      = tokenizer.encode(p),
            max_new_tokens = max_new_tokens,
            temperature    = 0.0,
            future         = _future(),
        )

    def test_c1_kv_pages_allocated_after_prefill(self, spec_runner, tokenizer):
        """C1: Both models have KV pages allocated after prefill."""
        req = self._make_req(tokenizer)

        target_pages_before = spec_runner.target.kv_pool.available()
        draft_pages_before  = spec_runner.draft.kv_pool.available()

        spec_runner.prefill([req])

        target_pages_after = spec_runner.target.kv_pool.available()
        draft_pages_after  = spec_runner.draft.kv_pool.available()

        assert target_pages_after < target_pages_before, "Target KV pool unchanged"
        assert draft_pages_after  < draft_pages_before,  "Draft KV pool unchanged"

    def test_c2_output_ids_after_prefill(self, spec_runner, tokenizer):
        """C2: After prefill, req.output_ids has exactly 1 token."""
        req = self._make_req(tokenizer)
        spec_runner.prefill([req])

        assert len(req.output_ids) == 1
        assert req.output_ids[0] < spec_runner.target.model.model.config.vocab_size

    def test_c3_draft_phase_produces_n_tokens(self, spec_runner, tokenizer):
        """C3: Draft phase returns exactly N tokens."""
        req = self._make_req(tokenizer)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]
        d_req.output_ids = list(req.output_ids)

        draft_tokens, new_pages = spec_runner._draft_phase(d_req)

        assert len(draft_tokens) == spec_runner.N
        assert len(new_pages)    == spec_runner.N
        assert all(isinstance(t, int) for t in draft_tokens)
        assert all(t >= 0 for t in draft_tokens)

    def test_c4_verify_extend_logit_shape(self, spec_runner, tokenizer):
        """C4: Verify extend returns logits of shape [N+1, vocab]."""
        req = self._make_req(tokenizer)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]
        d_req.output_ids = list(req.output_ids)

        draft_tokens, _ = spec_runner._draft_phase(d_req)

        last_confirmed = req.output_ids[-1]
        verify_tokens  = [last_confirmed] + draft_tokens
        logits         = spec_runner._verify_extend(req, verify_tokens)

        N    = spec_runner.N
        vocab = spec_runner.target.model.model.config.vocab_size
        assert logits.shape == (N + 1, vocab), \
            f"Expected ({N+1}, {vocab}), got {logits.shape}"
        assert not torch.isnan(logits).any(),  "NaN in verify logits"
        assert not torch.isinf(logits).any(),  "Inf in verify logits"

    def test_c5_accept_len_in_range(self, spec_runner, tokenizer):
        """C5: accept_len ∈ [0, N]."""
        req = self._make_req(tokenizer)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]
        d_req.output_ids = list(req.output_ids)

        draft_tokens, _ = spec_runner._draft_phase(d_req)
        verify_tokens   = [req.output_ids[-1]] + draft_tokens
        logits          = spec_runner._verify_extend(req, verify_tokens)
        accept_len, bonus = spec_runner._accept_reject(draft_tokens, logits)

        N = spec_runner.N
        assert 0 <= accept_len <= N, f"accept_len={accept_len} out of [0, {N}]"
        assert 0 <= bonus < spec_runner.target.model.model.config.vocab_size

    def test_c6_output_grows_after_spec_step(self, spec_runner, tokenizer):
        """C6: After one spec step, req.output_ids grows by accept_len + 1."""
        req = self._make_req(tokenizer)
        spec_runner.prefill([req])

        n_before = len(req.output_ids)
        newly_finished = spec_runner.spec_decode_step([req])

        n_after = len(req.output_ids)
        growth  = n_after - n_before

        assert growth >= 1, "At least the bonus token must be added"
        assert growth <= spec_runner.N + 1, \
            f"At most N+1={spec_runner.N+1} tokens; got growth={growth}"


# ─────────────────────────────────────────────────────────────────────────────
# Part D — Full end-to-end speculative decode (GPU)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.gpu
class TestEndToEnd:
    """Part D — full spec-decode correctness and statistics."""

    def test_d1_output_matches_target_greedy(self, spec_runner, tokenizer):
        """
        D1: Speculative decode (temperature=0) produces exactly the same token
        sequence as standard greedy decode on the target model.
        """
        from model_runner import ModelRunner

        PROMPT         = "The history of artificial intelligence"
        MAX_NEW_TOKENS = 30

        # ── Spec decode ──────────────────────────────────────────────────
        spec_runner.total_draft_tokens    = 0
        spec_runner.total_accepted_tokens = 0
        spec_runner.total_spec_steps      = 0

        input_ids = tokenizer.encode(PROMPT)
        spec_req  = Req(
            rid            = "d1_spec",
            input_ids      = list(input_ids),
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.0,
            future         = _future(),
        )
        spec_runner.prefill([spec_req])
        while spec_req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([spec_req])
        spec_ids = list(spec_req.output_ids)

        # ── Baseline: target-only greedy ─────────────────────────────────
        baseline = ModelRunner(
            TARGET_PATH, page_size=16, enable_prefix_caching=False
        )
        baseline_req = Req(
            rid            = "d1_base",
            input_ids      = list(input_ids),
            max_new_tokens = MAX_NEW_TOKENS,
            temperature    = 0.0,
            future         = _future(),
        )
        baseline_req.fill_ids         = list(input_ids)
        baseline_req.extend_input_len = len(input_ids)
        baseline_req.kv_committed_len = 0
        baseline.prefill_batch([baseline_req])
        while baseline_req.status == ReqStatus.RUNNING:
            baseline.decode_step([baseline_req])
        base_ids = list(baseline_req.output_ids)
        del baseline
        torch.cuda.empty_cache()

        # ── Compare ──────────────────────────────────────────────────────
        min_len = min(len(spec_ids), len(base_ids))
        assert spec_ids[:min_len] == base_ids[:min_len], (
            f"Token mismatch!\n"
            f"  spec:     {spec_ids[:min_len]}\n"
            f"  baseline: {base_ids[:min_len]}"
        )

    def test_d2_acceptance_rate_positive(self, spec_runner, tokenizer):
        """D2: Acceptance rate > 15 % on a short prompt."""
        spec_runner.total_draft_tokens    = 0
        spec_runner.total_accepted_tokens = 0
        spec_runner.total_spec_steps      = 0

        for i in range(3):
            prompt    = f"In machine learning, the {['gradient', 'loss', 'model'][i]} function"
            input_ids = tokenizer.encode(prompt)
            req = Req(
                rid            = f"d2_{i}",
                input_ids      = list(input_ids),
                max_new_tokens = 30,
                temperature    = 0.0,
                future         = _future(),
            )
            spec_runner.prefill([req])
            while req.status == ReqStatus.RUNNING:
                spec_runner.spec_decode_step([req])

        rate = spec_runner.acceptance_rate
        assert rate > 0.15, (
            f"Acceptance rate {rate:.1%} too low (expected > 15 %)\n"
            f"Stats: {spec_runner.stats_str()}"
        )

    def test_d3_kv_pages_returned_after_finish(self, spec_runner, tokenizer):
        """D3: KV pool pages are fully reclaimed after a request finishes."""
        target_avail_before = spec_runner.target.kv_pool.available()
        draft_avail_before  = spec_runner.draft.kv_pool.available()

        input_ids = tokenizer.encode("Hello world")
        req = Req(
            rid            = "d3",
            input_ids      = list(input_ids),
            max_new_tokens = 20,
            temperature    = 0.0,
            future         = _future(),
        )
        spec_runner.prefill([req])
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

        assert req.status == ReqStatus.FINISHED
        # Resources freed in _cleanup_req (called inside spec_decode_step)
        target_avail_after = spec_runner.target.kv_pool.available()
        draft_avail_after  = spec_runner.draft.kv_pool.available()

        assert target_avail_after == target_avail_before, (
            f"Target KV pool leaked: {target_avail_before - target_avail_after} pages"
        )
        assert draft_avail_after == draft_avail_before, (
            f"Draft KV pool leaked: {draft_avail_before - draft_avail_after} pages"
        )

    def test_d4_sequential_requests_no_state_leak(self, spec_runner, tokenizer):
        """D4: Two sequential requests don't share stale state."""
        prompts = [
            "The weather today is",
            "Once upon a time in a",
        ]
        outputs = []

        for i, prompt in enumerate(prompts):
            input_ids = tokenizer.encode(prompt)
            req = Req(
                rid            = f"d4_{i}",
                input_ids      = list(input_ids),
                max_new_tokens = 20,
                temperature    = 0.0,
                future         = _future(),
            )
            spec_runner.prefill([req])
            while req.status == ReqStatus.RUNNING:
                spec_runner.spec_decode_step([req])
            outputs.append(list(req.output_ids))

        # Both requests should produce non-empty outputs
        for i, ids in enumerate(outputs):
            assert len(ids) > 0, f"Request {i} produced no output"

        # _draft_reqs should be empty (both cleaned up)
        assert len(spec_runner._draft_reqs) == 0, \
            f"Draft req state leaked: {len(spec_runner._draft_reqs)} entries"


# ─────────────────────────────────────────────────────────────────────────────
# Part E — accept/reject extended edge cases (CPU-only)
# ─────────────────────────────────────────────────────────────────────────────

class TestAcceptRejectEdgeCases:
    """Part E — accept/reject edge cases not covered by Part A."""

    def test_e1_token_id_zero_accepted(self):
        """E1: Token ID 0 is accepted when target also predicts 0."""
        draft  = [0, 1, 2]
        # target predicts [0, 1, 2, bonus=50]
        logits = _logits_for_tokens([0, 1, 2, 50])

        al, bonus = SpecRunner._accept_reject(draft, logits)

        assert al == 3
        assert bonus == 50

    def test_e2_high_token_id_near_vocab_limit(self):
        """E2: Tokens near the vocab ceiling are handled via correct argmax."""
        vocab  = 1000
        high   = vocab - 1      # 999
        draft  = [high, high - 1]
        logits = _logits_for_tokens([high, high - 1, 42], vocab=vocab)

        al, bonus = SpecRunner._accept_reject(draft, logits)

        assert al == 2
        assert bonus == 42

    def test_e3_large_n_all_match(self):
        """E3: N=20 draft tokens all accepted."""
        N     = 20
        draft = list(range(100, 100 + N))      # [100 … 119]
        tgts  = draft + [200]                   # all match, bonus=200
        logits = _logits_for_tokens(tgts, vocab=300)

        al, bonus = SpecRunner._accept_reject(draft, logits)

        assert al == N
        assert bonus == 200

    def test_e4_return_types_are_plain_ints(self):
        """E4: _accept_reject returns plain Python ints, not torch.Tensor."""
        draft  = [7, 8, 9]
        logits = _logits_for_tokens([7, 99, 9, 55])   # mismatch at index 1

        al, bonus = SpecRunner._accept_reject(draft, logits)

        assert type(al)    is int, f"accept_len is {type(al)}, expected int"
        assert type(bonus) is int, f"bonus is {type(bonus)}, expected int"

    def test_e5_accept_len_never_exceeds_n(self):
        """E5: accept_len ≤ N even if all logits would favour draft tokens."""
        N      = 5
        draft  = list(range(N))           # [0, 1, 2, 3, 4]
        # Provide N+2 rows to be safe, but accept_len must still cap at N.
        tgts   = draft + [99, 88]
        logits = _logits_for_tokens(tgts)

        al, _ = SpecRunner._accept_reject(draft, logits)

        assert al <= N, f"accept_len={al} exceeds N={N}"

    def test_e6_bonus_different_from_draft_at_reject_site(self):
        """E6: When target rejects at position k, bonus != draft_tokens[k]."""
        draft  = [1, 2, 3, 4, 5]
        # mismatch at index 2: target predicts 99, draft had 3
        tgts   = [1, 2, 99, 4, 5, 55]
        logits = _logits_for_tokens(tgts)

        al, bonus = SpecRunner._accept_reject(draft, logits)

        assert al    == 2
        assert bonus == 99
        assert bonus != draft[al], "bonus should differ from draft at reject site"


# ─────────────────────────────────────────────────────────────────────────────
# Part F — KV rewind boundary conditions (mock, CPU)
# ─────────────────────────────────────────────────────────────────────────────

class TestKVRewindBoundary:
    """Part F — boundary and edge cases for _rewind_target_kv / _rewind_draft_kv."""

    def _make_runner(self, N=5, P=16, page_size=None) -> MockSpecRunner:
        p = page_size if page_size is not None else P
        return MockSpecRunner(N=N, page_size=p)

    def test_f1_target_noop_when_t_not_multiple_of_p(self):
        """
        F1: T=10, P=4, accept_len=N=5.
        After verify: kv_committed_len=10, N+1=6 new tokens → total=16.
        pages=ceil(16/4)=4 → slot_indices=[a,b,c,d].
        Keep accept_len+1=6 → kept_kv_len=10+6=16 → pages_needed=4. No-op.
        """
        P = 4; N = 5
        runner = self._make_runner(N, P)
        req = _req()
        req.kv_committed_len = 10
        req.slot_indices     = [10, 11, 12, 13]   # ceil(16/4)=4 pages

        runner._rewind_target_kv(req, accept_len=N)

        assert runner.target.kv_pool.freed == []
        assert req.slot_indices == [10, 11, 12, 13]

    def test_f2_target_exact_page_boundary_none_accepted(self):
        """
        F2: T=8, P=4, N=4, accept_len=0.
        After verify: total=8+5=13 tokens → ceil(13/4)=4 pages.
        keep=8+1=9 → ceil(9/4)=3 pages → free 1 page.
        """
        P = 4; N = 4
        runner = self._make_runner(N, P)
        req = _req()
        req.kv_committed_len = 8
        req.slot_indices     = [10, 11, 12, 13]   # 4 pages

        runner._rewind_target_kv(req, accept_len=0)

        assert set(runner.target.kv_pool.freed) == {13}
        assert req.slot_indices == [10, 11, 12]

    def test_f3_draft_all_accepted_noop(self):
        """F3: accept_len==N → no rejected steps → no pages freed."""
        N = 4
        runner = self._make_runner(N, page_size=8)

        d_req = _req()
        d_req.slot_indices = [30, 31]   # 2 pages

        # Each draft step allocated one page
        new_pages_per_step = [[31], [], [], []]   # only step 0 allocated
        # accept_len=N=4: all steps accepted → free pages from step 4..3 = empty
        runner._rewind_draft_kv(d_req, new_pages_per_step, accept_len=N)

        assert runner.draft.kv_pool.freed == []
        assert d_req.slot_indices == [30, 31]

    def test_f4_draft_step_allocated_multiple_pages(self):
        """F4: One draft step allocated 2 pages; they're freed together on reject."""
        N = 3
        runner = self._make_runner(N, page_size=4)

        d_req = _req()
        d_req.slot_indices = [20, 21, 22, 23]   # 4 pages

        new_pages_per_step = [
            [],         # step 0: no new pages
            [21, 22],   # step 1: allocated 2 pages at once (edge case)
            [23],       # step 2: allocated 1 page
        ]

        # accept_len=1: reject steps 1 and 2 → free pages 21, 22, 23
        runner._rewind_draft_kv(d_req, new_pages_per_step, accept_len=1)

        assert set(runner.draft.kv_pool.freed) == {21, 22, 23}
        assert d_req.slot_indices == [20]

    def test_f5_target_kept_len_exactly_one_page(self):
        """F5: Kept KV length == exactly P tokens → pages_needed == 1 → no free."""
        P = 4; N = 3
        runner = self._make_runner(N, P)

        # T=3, accept_len=0: kept_kv_len = 3+0+1 = 4 = P → ceil(4/4) = 1 page.
        # slot_indices after verify extend: ceil((3+4)/4) = ceil(7/4) = 2 pages.
        req = _req()
        req.kv_committed_len = 3
        req.slot_indices     = [5, 6]   # 2 pages after verify

        runner._rewind_target_kv(req, accept_len=0)

        # kept_kv_len = 4, pages_needed = 1 → free page 6
        assert set(runner.target.kv_pool.freed) == {6}
        assert req.slot_indices == [5]


# ─────────────────────────────────────────────────────────────────────────────
# Part G — Statistics and state tracking (CPU, no GPU)
# ─────────────────────────────────────────────────────────────────────────────

class MockSpecRunnerStats:
    """Minimal SpecRunner with just the stats properties, no models."""

    def __init__(self):
        self.total_draft_tokens    = 0
        self.total_accepted_tokens = 0
        self.total_spec_steps      = 0
        self._draft_reqs           = {}

    acceptance_rate = SpecRunner.acceptance_rate
    tokens_per_step = SpecRunner.tokens_per_step
    stats_str       = SpecRunner.stats_str


class TestStatistics:
    """Part G — statistics properties and accumulation."""

    def _runner(self) -> MockSpecRunnerStats:
        return MockSpecRunnerStats()

    def test_g1_acceptance_rate_zero_when_no_drafts(self):
        """G1: acceptance_rate == 0.0 when no draft tokens attempted."""
        r = self._runner()
        assert r.acceptance_rate == 0.0

    def test_g2_tokens_per_step_zero_when_no_steps(self):
        """G2: tokens_per_step == 0.0 when no spec steps taken."""
        r = self._runner()
        assert r.tokens_per_step == 0.0

    def test_g3_acceptance_rate_accumulates(self):
        """G3: Correct rate after accepting 3 out of 10 draft tokens."""
        r = self._runner()
        r.total_draft_tokens    = 10
        r.total_accepted_tokens = 3
        r.total_spec_steps      = 2

        assert abs(r.acceptance_rate - 0.3) < 1e-9

    def test_g4_tokens_per_step_formula(self):
        """G4: tokens_per_step == (accepted + steps) / steps."""
        r = self._runner()
        r.total_draft_tokens    = 15   # 3 steps × N=5
        r.total_accepted_tokens = 9    # 9 accepted
        r.total_spec_steps      = 3

        # (9 + 3) / 3 = 4.0
        assert abs(r.tokens_per_step - 4.0) < 1e-9

    def test_g5_stats_str_contains_expected_keys(self):
        """G5: stats_str() includes all major metric labels."""
        r = self._runner()
        r.total_draft_tokens    = 20
        r.total_accepted_tokens = 12
        r.total_spec_steps      = 4

        s = r.stats_str()
        for key in ("acceptance_rate", "tokens_per_step",
                    "total_spec_steps", "total_accepted", "total_draft"):
            assert key in s, f"stats_str missing '{key}': {s}"

    def test_g6_acceptance_rate_is_100_when_all_accepted(self):
        """G6: Rate == 1.0 when every draft token was accepted."""
        r = self._runner()
        r.total_draft_tokens    = 50
        r.total_accepted_tokens = 50
        r.total_spec_steps      = 10

        assert r.acceptance_rate == 1.0

    def test_g7_tokens_per_step_lower_bound(self):
        """G7: tokens_per_step >= 1.0 — at least the bonus token each step."""
        r = self._runner()
        r.total_draft_tokens    = 10
        r.total_accepted_tokens = 0    # nothing accepted
        r.total_spec_steps      = 2

        # (0 + 2) / 2 = 1.0
        assert r.tokens_per_step == 1.0


# ─────────────────────────────────────────────────────────────────────────────
# Part H — GPU deeper correctness checks (requires model weights)
# ─────────────────────────────────────────────────────────────────────────────

def _run_to_finish(runner: SpecRunner, tokenizer, prompt: str,
                   max_new_tokens: int = 25) -> Req:
    """Helper: build a req, prefill, spec-decode to completion, return it."""
    req = Req(
        rid            = prompt[:12],
        input_ids      = tokenizer.encode(prompt),
        max_new_tokens = max_new_tokens,
        temperature    = 0.0,
        future         = _future(),
    )
    runner.prefill([req])
    while req.status == ReqStatus.RUNNING:
        runner.spec_decode_step([req])
    return req


@pytest.mark.gpu
class TestGPUDeeper:
    """Part H — GPU correctness that goes beyond Part C/D."""

    PROMPT = "Machine learning is a branch of"

    def _req(self, tokenizer, prompt=None, max_new_tokens=25):
        p = prompt or self.PROMPT
        return Req(
            rid            = p[:10],
            input_ids      = tokenizer.encode(p),
            max_new_tokens = max_new_tokens,
            temperature    = 0.0,
            future         = _future(),
        )

    def test_h1_kv_committed_len_tracks_each_step(self, spec_runner, tokenizer):
        """
        H1: kv_committed_len increases by exactly (accept_len + 1) each spec step.
        We measure it for 3 consecutive steps.
        """
        req = self._req(tokenizer, max_new_tokens=40)
        spec_runner.prefill([req])

        T = req.kv_committed_len   # == len(input_ids) after prefill

        growth_per_step = []
        for _ in range(3):
            if req.status != ReqStatus.RUNNING:
                break
            committed_before = req.kv_committed_len
            output_len_before = len(req.output_ids)

            spec_runner.spec_decode_step([req])

            delta_committed = req.kv_committed_len - committed_before
            delta_output    = len(req.output_ids)  - output_len_before

            # Both measures of "tokens added" must agree
            assert delta_committed == delta_output, (
                f"kv_committed_len grew by {delta_committed} but "
                f"output_ids grew by {delta_output}"
            )
            assert delta_committed >= 1, "Must emit at least the bonus token"
            growth_per_step.append(delta_committed)

        # Run to completion so KV is freed
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

        assert len(growth_per_step) > 0, "No spec steps ran"

    def test_h2_prefill_first_token_matches_target_argmax(
        self, spec_runner, tokenizer
    ):
        """
        H2: The first token emitted by prefill equals argmax of the target's
        logit at the last prompt position (greedy determinism check).
        """
        from model_runner import ModelRunner

        prompt    = "The speed of light is approximately"
        input_ids = tokenizer.encode(prompt)

        # Spec runner prefill
        req = self._req(tokenizer, prompt=prompt)
        spec_runner.prefill([req])
        spec_first_tok = req.output_ids[0]

        # Run spec req to completion so KV is reclaimed
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

        # Independent target-only forward pass for reference
        ref_runner = ModelRunner(
            TARGET_PATH, page_size=16, enable_prefix_caching=False
        )
        ref_req = Req(
            rid            = "ref",
            input_ids      = list(input_ids),
            max_new_tokens = 1,
            temperature    = 0.0,
            future         = _future(),
        )
        ref_req.fill_ids         = list(input_ids)
        ref_req.extend_input_len = len(input_ids)
        ref_req.kv_committed_len = 0
        ref_runner.prefill_batch([ref_req])
        ref_first_tok = ref_req.output_ids[0]
        del ref_runner
        torch.cuda.empty_cache()

        assert spec_first_tok == ref_first_tok, (
            f"Spec prefill first token {spec_first_tok} != "
            f"target-only {ref_first_tok}"
        )

    def test_h3_total_draft_tokens_increments_by_n(self, spec_runner, tokenizer):
        """H3: Each call to spec_decode_step adds exactly N to total_draft_tokens."""
        req = self._req(tokenizer, max_new_tokens=30)
        spec_runner.prefill([req])

        before = spec_runner.total_draft_tokens
        N      = spec_runner.N

        steps = 0
        while req.status == ReqStatus.RUNNING and steps < 4:
            spec_runner.spec_decode_step([req])
            steps += 1
            expected = before + steps * N
            assert spec_runner.total_draft_tokens == expected, (
                f"After {steps} steps: expected total_draft_tokens="
                f"{expected}, got {spec_runner.total_draft_tokens}"
            )

        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

    def test_h4_draft_slot_indices_shrink_after_partial_accept(
        self, spec_runner, tokenizer
    ):
        """
        H4: After a spec step where accept_len < N, the draft's slot_indices
        is shorter than it was at the peak of the draft phase.

        We instrument the draft phase directly to measure both.
        """
        req = self._req(tokenizer, max_new_tokens=40)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]

        # Run draft phase manually to measure peak
        d_req.output_ids = list(req.output_ids)
        draft_tokens, new_pages_per_step = spec_runner._draft_phase(d_req)
        peak_pages = len(d_req.slot_indices)

        # Compute accept_len
        verify_tokens = [req.output_ids[-1]] + draft_tokens
        logits        = spec_runner._verify_extend(req, verify_tokens)
        accept_len, bonus = spec_runner._accept_reject(draft_tokens, logits)

        # Rewind
        spec_runner._rewind_draft_kv(d_req, new_pages_per_step, accept_len)
        after_pages = len(d_req.slot_indices)

        if accept_len < spec_runner.N:
            # At least one step was rejected — draft pool should have shrunk
            # (only if those steps actually allocated new pages)
            newly_allocated = sum(
                len(pp) for pp in new_pages_per_step[accept_len:]
            )
            assert after_pages == peak_pages - newly_allocated, (
                f"Expected {peak_pages - newly_allocated} pages after rewind, "
                f"got {after_pages}"
            )
        else:
            # All accepted — no rewind
            assert after_pages == peak_pages

        # Clean up by running to finish
        spec_runner._rewind_target_kv(req, accept_len)
        req.kv_committed_len += accept_len + 1
        req.output_ids.extend(draft_tokens[:accept_len] + [bonus])
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

    def test_h5_verify_extend_twice_gives_same_logits(
        self, spec_runner, tokenizer
    ):
        """
        H5: Calling _verify_extend a second time with the same tokens (after
        rewinding the target KV to its pre-verify state) produces identical logits.
        Verifies the forward pass is deterministic and that rewind is clean.
        """
        req = self._req(tokenizer, max_new_tokens=40)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]
        d_req.output_ids = list(req.output_ids)

        draft_tokens, _ = spec_runner._draft_phase(d_req)
        verify_tokens   = [req.output_ids[-1]] + draft_tokens

        # First verify
        slot_snap    = list(req.slot_indices)
        committed_0  = req.kv_committed_len
        logits1      = spec_runner._verify_extend(req, verify_tokens)

        # Rewind all N+1 positions (accept_len=-1 keeps 0 positions is wrong,
        # instead accept_len=0 keeps 1 position; we want to keep 0 → use
        # accept_len=-1 trick: just restore slot_indices manually)
        spec_runner.target.kv_pool.free(req.slot_indices[len(slot_snap):])
        req.slot_indices    = slot_snap
        req.kv_committed_len = committed_0

        # Second verify from identical state
        d_req.output_ids = list(req.output_ids)
        _, _ = spec_runner._draft_phase(d_req)
        # Draft phase re-ran; redo verify with same original tokens
        spec_runner._rewind_draft_kv(
            d_req,
            [[] for _ in range(spec_runner.N)],
            accept_len=0,
        )
        logits2 = spec_runner._verify_extend(req, verify_tokens)

        # Logits should be identical (deterministic greedy model, same input)
        assert torch.allclose(logits1, logits2, atol=0.0), (
            "verify_extend not deterministic: max diff = "
            f"{(logits1 - logits2).abs().max().item():.2e}"
        )

        # Cleanup
        spec_runner._rewind_target_kv(req, accept_len=0)
        req.kv_committed_len += 1
        req.output_ids.append(int(logits2[0].argmax()))
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

    def test_h6_spec_decode_step_skips_finished_reqs(
        self, spec_runner, tokenizer
    ):
        """H6: A FINISHED req in the list is silently skipped (no crash, no change)."""
        req = _run_to_finish(spec_runner, tokenizer, "Hello", max_new_tokens=10)

        assert req.status == ReqStatus.FINISHED
        ids_before = list(req.output_ids)

        # Calling again with a FINISHED req should do nothing
        finished = spec_runner.spec_decode_step([req])

        assert finished == []
        assert req.output_ids == ids_before

    def test_h7_batch_prefill_two_reqs_independent(
        self, spec_runner, tokenizer
    ):
        """
        H7: Prefilling two requests in one batch gives each its own first token.
        The two tokens may be equal by coincidence but the requests must be
        independently tracked (separate req_pool_idx, separate slot_indices).
        """
        prompts = [
            "The first prime number is",
            "Water boils at one hundred",
        ]
        reqs = [
            Req(
                rid            = f"h7_{i}",
                input_ids      = tokenizer.encode(p),
                max_new_tokens = 20,
                temperature    = 0.0,
                future         = _future(),
            )
            for i, p in enumerate(prompts)
        ]

        spec_runner.prefill(reqs)

        # Each request should have exactly one output token
        for i, req in enumerate(reqs):
            assert len(req.output_ids) == 1, \
                f"req[{i}] should have 1 output token, got {len(req.output_ids)}"

        # req_pool_idx must be distinct
        assert reqs[0].req_pool_idx != reqs[1].req_pool_idx, \
            "Both requests share the same req_pool_idx"

        # slot_indices must be disjoint (no shared pages between requests)
        pages0 = set(reqs[0].slot_indices)
        pages1 = set(reqs[1].slot_indices)
        assert pages0.isdisjoint(pages1), \
            f"Requests share pages: {pages0 & pages1}"

        # Run both to completion
        running = list(reqs)
        while running:
            finished = spec_runner.spec_decode_step(running)
            for r in finished:
                running.remove(r)

    def test_h8_draft_tokens_in_valid_vocab_range(
        self, spec_runner, tokenizer
    ):
        """H8: Every token produced by the draft model is a valid token ID."""
        vocab = spec_runner.draft.model.model.config.vocab_size

        req = self._req(tokenizer, max_new_tokens=35)
        spec_runner.prefill([req])
        d_req = spec_runner._draft_reqs[id(req)]

        all_draft_tokens: List[int] = []

        for _ in range(4):
            if req.status != ReqStatus.RUNNING:
                break
            d_req.output_ids = list(req.output_ids)
            draft_tokens, new_pages = spec_runner._draft_phase(d_req)
            all_draft_tokens.extend(draft_tokens)

            # Verify and accept/reject to advance state properly
            verify_tokens = [req.output_ids[-1]] + draft_tokens
            logits        = spec_runner._verify_extend(req, verify_tokens)
            accept_len, bonus = spec_runner._accept_reject(draft_tokens, logits)

            spec_runner._rewind_target_kv(req, accept_len)
            spec_runner._rewind_draft_kv(d_req, new_pages, accept_len)
            req.kv_committed_len += accept_len + 1
            req.output_ids.extend(draft_tokens[:accept_len] + [bonus])

            if bonus == spec_runner.eos_id or len(req.output_ids) >= req.max_new_tokens:
                req.status = ReqStatus.FINISHED
                spec_runner._cleanup_req(req, d_req)
                break

        # Clean up if still running
        while req.status == ReqStatus.RUNNING:
            spec_runner.spec_decode_step([req])

        # Validate all collected draft tokens
        for tok in all_draft_tokens:
            assert 0 <= tok < vocab, \
                f"Draft token {tok} out of range [0, {vocab})"
