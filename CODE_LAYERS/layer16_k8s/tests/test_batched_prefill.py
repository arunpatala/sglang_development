"""
test_batched_prefill.py
========================
Tests for batched prefill: multiple requests processed together in one
EXTEND forward pass.

Tests:
  1. B=N batched prefill matches per-request F.sdpa (correctness)
  2. Batched prefill == N sequential single-request prefills (consistency)
  3. Mixed batch sizes: B=1, 2, 4
  4. Requests with different prompt lengths (ragged batch)
"""

import math
import sys
from pathlib import Path
from typing import List

import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import KVPool, ReqToTokenPool
from conftest import ATOL, DEVICE, DTYPE, do_extend, full_ref, make_pool


# ─────────────────────────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────────────────────────

PROMPT_PAIRS = [
    # (name, prompts)
    ("two_short", [
        "What is 2+2?",
        "What is the capital of France?",
    ]),
    ("two_medium", [
        "Explain what a neural network is.",
        "What is the difference between a compiled and interpreted language?",
    ]),
    ("four_mixed", [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain what a neural network is.",
        "What is the difference between a compiled and interpreted language?",
    ]),
    ("ragged", [
        "Hi!",
        "Explain in detail how backpropagation works in a deep neural network.",
        "What is 2+2?",
        "What is quantum entanglement and why is it important for quantum computing?",
    ]),
]


def encode(tok, text: str) -> List[int]:
    formatted = tok.apply_chat_template([{"role": "user", "content": text}])
    return tok._tok([formatted], return_tensors="pt", padding=False)["input_ids"][0].tolist()


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: batched vs F.sdpa reference
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair_name,prompts", PROMPT_PAIRS)
@pytest.mark.parametrize("page_size", [8, 16])
def test_batched_prefill_matches_full_recompute(
    model, tok, cfg, pair_name, prompts, page_size
):
    """
    Each request's last-token logit from a batched EXTEND pass must match
    the F.sdpa full-recompute reference for the same prompt.
    """
    all_ids = [encode(tok, p) for p in prompts]
    total_tokens = sum(len(ids) for ids in all_ids)

    kv_pool, rtp = make_pool(cfg, total_tokens + 16, page_size=page_size, max_batch=16)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    reqs = [
        {"fill_ids": ids, "kv_committed_len": 0,
         "slot_indices": [], "req_pool_idx": rtp.alloc()}
        for ids in all_ids
    ]
    logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, reqs, cfg)

    for i, ids in enumerate(all_ids):
        ref   = full_ref(model, ids)
        batch = logits[0, qo_indptr[i + 1] - 1, :]
        diff  = (ref - batch).abs().max().item()
        assert diff < ATOL, (
            f"[{pair_name}] request {i} '{prompts[i][:30]}': "
            f"max_diff={diff:.4f} >= ATOL={ATOL} (page_size={page_size})"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: batched == sequential single-request prefills
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair_name,prompts", PROMPT_PAIRS)
@pytest.mark.parametrize("page_size", [16])
def test_batched_matches_sequential_single_prefills(
    model, tok, cfg, pair_name, prompts, page_size
):
    """
    Batched prefill (all requests in one EXTEND call) must produce identical
    last-token logits to running each request individually.

    This ensures the packing and qo_indptr slicing are correct regardless of
    which requests share a batch.
    """
    all_ids      = [encode(tok, p) for p in prompts]
    total_tokens = sum(len(ids) for ids in all_ids)

    # Sequential: one request at a time
    seq_logits = []
    for ids in all_ids:
        kv_pool_s, rtp_s = make_pool(cfg, len(ids) + 8, page_size=page_size)
        ws = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
        rd = [{"fill_ids": ids, "kv_committed_len": 0,
               "slot_indices": [], "req_pool_idx": rtp_s.alloc()}]
        lg, qo = do_extend(model, kv_pool_s, rtp_s, ws, rd, cfg)
        seq_logits.append(lg[0, qo[1] - 1, :].clone())

    # Batched: all at once
    kv_pool_b, rtp_b = make_pool(cfg, total_tokens + 16, page_size=page_size, max_batch=16)
    ws_b = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    reqs = [
        {"fill_ids": ids, "kv_committed_len": 0,
         "slot_indices": [], "req_pool_idx": rtp_b.alloc()}
        for ids in all_ids
    ]
    lg_b, qo_b = do_extend(model, kv_pool_b, rtp_b, ws_b, reqs, cfg)

    for i, ids in enumerate(all_ids):
        lg_batch = lg_b[0, qo_b[i + 1] - 1, :]
        diff     = (seq_logits[i] - lg_batch).abs().max().item()
        assert diff < ATOL, (
            f"[{pair_name}] request {i}: sequential vs batched "
            f"max_diff={diff:.4f} >= ATOL={ATOL}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: top-1 token agreement (greedy decoding matches between paths)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("pair_name,prompts", PROMPT_PAIRS)
def test_batched_greedy_token_matches_reference(model, tok, cfg, pair_name, prompts):
    """
    The greedy (argmax) token from batched prefill must match the greedy
    token from F.sdpa full recompute for every request in the batch.
    """
    all_ids      = [encode(tok, p) for p in prompts]
    total_tokens = sum(len(ids) for ids in all_ids)
    page_size    = 16

    kv_pool, rtp = make_pool(cfg, total_tokens + 16, page_size=page_size, max_batch=16)
    workspace    = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    reqs = [
        {"fill_ids": ids, "kv_committed_len": 0,
         "slot_indices": [], "req_pool_idx": rtp.alloc()}
        for ids in all_ids
    ]
    logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, reqs, cfg)

    for i, ids in enumerate(all_ids):
        ref_tok   = int(full_ref(model, ids).argmax())
        batch_tok = int(logits[0, qo_indptr[i + 1] - 1, :].argmax())

        # Allow the token to differ only if the logit difference is tiny
        # (near-tie scenario — both are valid).
        if ref_tok != batch_tok:
            ref_lg   = full_ref(model, ids)
            batch_lg = logits[0, qo_indptr[i + 1] - 1, :]
            diff     = (ref_lg - batch_lg).abs().max().item()
            assert diff < ATOL, (
                f"[{pair_name}] req {i}: greedy mismatch "
                f"ref={ref_tok} batch={batch_tok} max_diff={diff:.4f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: B=1 batched == F.sdpa (degenerate case, batch of one)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("page_size", [8, 16])
def test_batch_of_one_matches_reference(model, tok, cfg, page_size):
    """
    When B=1, the batched path must agree with F.sdpa — this is the simplest
    correctness check and catches any broken batching overhead.
    """
    ids      = encode(tok, "What is the meaning of life?")
    kv_pool, rtp = make_pool(cfg, len(ids) + 8, page_size=page_size)
    workspace    = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    rd = [{"fill_ids": ids, "kv_committed_len": 0,
           "slot_indices": [], "req_pool_idx": rtp.alloc()}]
    logits, qo = do_extend(model, kv_pool, rtp, workspace, rd, cfg)

    ref  = full_ref(model, ids)
    diff = (ref - logits[0, qo[1] - 1, :]).abs().max().item()
    assert diff < ATOL, f"B=1 max_diff={diff:.4f} >= ATOL={ATOL} (page_size={page_size})"
