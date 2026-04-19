"""
test_chunked_vs_full_prefill.py
================================
Core correctness tests for Layer 10's chunked prefill.

These tests answer the question:
  "Does chunked prefill produce the same result as a single full prefill?"

The comparison is done at THREE levels:

  Level 1 — Last-token logits match at every chunk boundary
    After each chunk, the last-position logit must match F.sdpa over the same
    prefix length.  This validates that the paged KV cache correctly stores
    tokens from prior chunks.

  Level 2 — Chunked == full paged prefill (same engine, different chunk size)
    A full paged prefill (1 chunk = whole prompt) and a chunked paged prefill
    (N chunks) must produce identical last-token logits.  This isolates the
    chunking logic from the model itself.

  Level 3 — Decode tokens after chunked prefill match decode after full prefill
    The first K decode tokens following a chunked prefill must match those from
    a full prefill.  This validates that the KV pool state is identical after
    either path (since the decode kernel reads from the pool).

Test parameterisation:
  chunk_sizes  — various sizes, including aligned to page_size (16) and not
  page_sizes   — 8 and 16 (exercises page packing in both cases)
  prompts      — short (fits in one page), medium (multi-page), long (many chunks)
"""

import math
import sys
from pathlib import Path
from typing import List

import pytest
import torch
import flashinfer

sys.path.insert(0, str(Path(__file__).parent.parent))

from kv_cache import DecodeKVCtx, KVPool, ReqToTokenPool, compute_write_info
from triton_utils import create_flashinfer_kv_indices_triton
from conftest import ATOL, DEVICE, DTYPE, do_extend, full_ref, make_pool


# ─────────────────────────────────────────────────────────────────────────────
# Test data
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = {
    "short":  "What is 2+2?",
    "medium": "Explain what a transformer neural network is and how attention works.",
    "long": (
        "The history of artificial intelligence is long and fascinating. "
        "It began in the 1950s with pioneers like Alan Turing, who proposed "
        "the famous Turing Test as a measure of machine intelligence. "
        "Since then, the field has gone through multiple winters and springs, "
        "culminating in the deep learning revolution of the 2010s. "
        "Today, large language models trained on vast corpora of text are able "
        "to converse, reason, and write code with remarkable fluency."
    ),
}

CHUNK_SIZES = [8, 16, 24, 48]
PAGE_SIZES  = [8, 16]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def encode(tok, text: str) -> List[int]:
    """Tokenise a chat-formatted prompt."""
    formatted = tok.apply_chat_template([{"role": "user", "content": text}])
    return tok._tok([formatted], return_tensors="pt", padding=False)["input_ids"][0].tolist()


def full_paged_prefill(model, kv_pool, rtp, workspace, ids, cfg):
    """
    Run a single (non-chunked) paged prefill over `ids` and return:
      (logit_last [vocab], slot_indices, req_pool_idx)
    """
    slot_indices = []
    rpi = rtp.alloc()
    rd = [{"fill_ids": ids, "kv_committed_len": 0,
           "slot_indices": slot_indices, "req_pool_idx": rpi}]
    logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, rd, cfg)
    return logits[0, qo_indptr[1] - 1, :], slot_indices, rpi


def chunked_prefill(model, kv_pool, rtp, workspace, ids, chunk_size, cfg):
    """
    Run a chunked paged prefill over `ids` in chunks of ≤ chunk_size tokens.
    Returns:
      (logit_last [vocab], slot_indices, req_pool_idx, all_chunk_logits)

    all_chunk_logits[k] = logit at last token of chunk k (for boundary checks).
    """
    L         = len(ids)
    n_chunks  = math.ceil(L / chunk_size)
    slot_indices  = []
    rpi           = rtp.alloc()
    kv_committed  = 0
    all_chunk_logits = []

    for c in range(n_chunks):
        start = c * chunk_size
        end   = min(start + chunk_size, L)
        rd = [{"fill_ids": ids[start:end], "kv_committed_len": kv_committed,
               "slot_indices": slot_indices, "req_pool_idx": rpi}]
        logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, rd, cfg)
        kv_committed = end
        all_chunk_logits.append(logits[0, qo_indptr[1] - 1, :].clone())

    return all_chunk_logits[-1], slot_indices, rpi, all_chunk_logits


def decode_step_manual(model, kv_pool, rtp, slot_indices, rpi, next_tok, seq_len, cfg, workspace_d):
    """Run one decode step and return the output logit [vocab]."""
    P            = kv_pool.page_size
    token_offset = seq_len % P
    num_pages    = len(slot_indices)

    if token_offset == 0:
        new_page = kv_pool.alloc(1)[0]
        slot_indices.append(new_page)
        rtp.req_to_token[rpi, num_pages] = new_page
        last_pg   = new_page
        num_pages += 1
    else:
        last_pg = slot_indices[-1]

    seq_t   = torch.tensor([seq_len],   dtype=torch.int32, device=DEVICE)
    toff_t  = torch.tensor([token_offset], dtype=torch.int32, device=DEVICE)
    npg_t   = torch.tensor([num_pages], dtype=torch.int32, device=DEVICE)
    rpi_t   = torch.tensor([rpi],       dtype=torch.int32, device=DEVICE)
    lpg_t   = torch.tensor([last_pg],   dtype=torch.int64, device=DEVICE)
    toff_i64 = toff_t.to(torch.int64)
    kv_last  = toff_t + 1

    kv_iptr  = torch.zeros(2, dtype=torch.int32, device=DEVICE)
    kv_iptr[0] = 0; kv_iptr[1] = num_pages
    kv_idx   = torch.empty(num_pages, dtype=torch.int32, device=DEVICE)
    create_flashinfer_kv_indices_triton[(1,)](
        rtp.req_to_token, rpi_t, npg_t, kv_iptr,
        None, kv_idx, rtp.req_to_token.shape[1],
    )

    dec = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace_d, "NHD", use_tensor_cores=False)
    dec.begin_forward(
        kv_iptr, kv_idx, kv_last,
        cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, P,
        data_type=DTYPE, q_data_type=DTYPE,
    )
    ctx = DecodeKVCtx(
        wrapper=dec, k_pool=kv_pool.k_pool, v_pool=kv_pool.v_pool,
        last_page_indices=lpg_t, token_offsets=toff_i64,
    )
    tok_t = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
    pos_t = torch.tensor([[seq_len]],  dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        lg = model(tok_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
    dec.end_forward()
    return lg[0, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# Level 1: each chunk boundary logit matches F.sdpa over the same prefix
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt_key", ["short", "medium", "long"])
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("page_size",  PAGE_SIZES)
def test_chunk_boundary_matches_full_recompute(
    model, tok, cfg, prompt_key, chunk_size, page_size
):
    """
    After each chunk k, the last-token logit must agree with F.sdpa over
    ids[:end_of_chunk_k].  Tests that prior-chunk KV is read correctly.
    """
    ids = encode(tok, PROMPTS[prompt_key])
    L   = len(ids)
    if L <= chunk_size:
        pytest.skip(f"Prompt too short for chunk_size={chunk_size}: L={L}")

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    kv_pool, rtp = make_pool(cfg, L + 4, page_size=page_size)

    _, _, _, all_chunk_logits = chunked_prefill(
        model, kv_pool, rtp, workspace, ids, chunk_size, cfg
    )

    n_chunks = math.ceil(L / chunk_size)
    for c in range(n_chunks):
        end = min((c + 1) * chunk_size, L)
        ref = full_ref(model, ids[:end])

        chunked_lg = all_chunk_logits[c]
        diff       = (ref - chunked_lg).abs().max().item()

        assert diff < ATOL, (
            f"[{prompt_key}] chunk {c} [{c*chunk_size}:{end}] "
            f"(page_size={page_size}, chunk_size={chunk_size}): "
            f"max_diff={diff:.4f} >= ATOL={ATOL}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Level 2: chunked paged == full paged (same engine, different chunk sizes)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt_key", ["short", "medium", "long"])
@pytest.mark.parametrize("chunk_size", CHUNK_SIZES)
@pytest.mark.parametrize("page_size",  PAGE_SIZES)
def test_chunked_paged_matches_full_paged_prefill(
    model, tok, cfg, prompt_key, chunk_size, page_size
):
    """
    Full paged prefill (1 chunk = full prompt) and chunked paged prefill
    must produce identical last-token logits.

    This is the strongest correctness check: same kernel, same memory layout,
    same FlashInfer wrapper — only the number of forward passes differs.
    """
    ids = encode(tok, PROMPTS[prompt_key])
    L   = len(ids)
    if L <= chunk_size:
        pytest.skip(f"Prompt too short for chunk_size={chunk_size}: L={L}")

    # Size the pool for two independent requests (full + chunked).
    total_tokens = L * 2 + 32
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    kv_pool_full, rtp_full = make_pool(cfg, total_tokens // 2 + 16, page_size=page_size)
    kv_pool_chk,  rtp_chk  = make_pool(cfg, total_tokens // 2 + 16, page_size=page_size)

    ws_full = workspace
    ws_chk  = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    full_last_logit, _, _ = full_paged_prefill(model, kv_pool_full, rtp_full, ws_full, ids, cfg)
    chk_last_logit, _, _, _ = chunked_prefill(model, kv_pool_chk, rtp_chk, ws_chk, ids, chunk_size, cfg)

    diff = (full_last_logit - chk_last_logit).abs().max().item()
    assert diff < ATOL, (
        f"[{prompt_key}] full paged vs chunked (page_size={page_size}, chunk_size={chunk_size}): "
        f"max_diff={diff:.4f} >= ATOL={ATOL}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Level 3: decode tokens after chunked prefill match after full prefill
# ─────────────────────────────────────────────────────────────────────────────

N_DECODE_STEPS = 5

@pytest.mark.parametrize("prompt_key", ["short", "medium"])
@pytest.mark.parametrize("chunk_size", [16, 24])
@pytest.mark.parametrize("page_size",  [16])
def test_decode_after_chunked_matches_full_prefill(
    model, tok, cfg, prompt_key, chunk_size, page_size
):
    """
    After chunked prefill is complete, decode steps must produce the same
    logits as decode after a single full prefill.

    Validates that the KV pool state is bit-for-bit equivalent after both
    prefill paths — and that the decode kernel reads it correctly.
    """
    ids = encode(tok, PROMPTS[prompt_key])
    L   = len(ids)
    if L <= chunk_size:
        pytest.skip(f"Prompt too short for chunk_size={chunk_size}: L={L}")

    total_tokens = L + N_DECODE_STEPS + 16
    ws_full = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    ws_chk  = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    ws_dec  = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    kv_pool_full, rtp_full = make_pool(cfg, total_tokens, page_size=page_size)
    kv_pool_chk,  rtp_chk  = make_pool(cfg, total_tokens, page_size=page_size)

    # Run both prefill paths.
    full_logit, full_slots, full_rpi = full_paged_prefill(
        model, kv_pool_full, rtp_full, ws_full, ids, cfg
    )
    chk_logit, chk_slots, chk_rpi, _ = chunked_prefill(
        model, kv_pool_chk, rtp_chk, ws_chk, ids, chunk_size, cfg
    )

    # Both last-token logits must agree (sanity check before decode).
    prefill_diff = (full_logit - chk_logit).abs().max().item()
    assert prefill_diff < ATOL, (
        f"[{prompt_key}] prefill last-logit diff={prefill_diff:.4f}: "
        f"full vs chunked disagree before decode"
    )

    # Use the greedy token from the full reference as the shared next-token.
    next_tok_full = int(full_logit.argmax())
    next_tok_chk  = int(chk_logit.argmax())

    # N decode steps, comparing logits at each step.
    seq_len_full = L
    seq_len_chk  = L
    ntok_full = next_tok_full
    ntok_chk  = next_tok_chk

    for step in range(N_DECODE_STEPS):
        lg_full = decode_step_manual(
            model, kv_pool_full, rtp_full, full_slots, full_rpi,
            ntok_full, seq_len_full, cfg, ws_dec,
        )
        lg_chk = decode_step_manual(
            model, kv_pool_chk, rtp_chk, chk_slots, chk_rpi,
            ntok_chk, seq_len_chk, cfg, ws_dec,
        )

        diff = (lg_full - lg_chk).abs().max().item()
        assert diff < ATOL, (
            f"[{prompt_key}] decode step {step}: "
            f"full vs chunked diff={diff:.4f} >= ATOL={ATOL} "
            f"(page_size={page_size}, chunk_size={chunk_size})"
        )

        ntok_full    = int(lg_full.argmax())
        ntok_chk     = int(lg_chk.argmax())
        seq_len_full += 1
        seq_len_chk  += 1


# ─────────────────────────────────────────────────────────────────────────────
# Edge case: chunk_size >= prompt_len  (degenerates to full prefill)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("page_size", PAGE_SIZES)
def test_chunk_size_larger_than_prompt_is_single_chunk(model, tok, cfg, page_size):
    """
    When chunk_size >= prompt_len, chunked prefill must behave identically
    to full paged prefill (just one extend call).
    """
    ids   = encode(tok, PROMPTS["short"])
    L     = len(ids)
    chunk = L + 10  # definitely larger than the prompt

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    kv_pool_full, rtp_full = make_pool(cfg, L + 16, page_size=page_size)
    kv_pool_chk,  rtp_chk  = make_pool(cfg, L + 16, page_size=page_size)

    ws_full = workspace
    ws_chk  = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    full_lg, _, _ = full_paged_prefill(model, kv_pool_full, rtp_full, ws_full, ids, cfg)
    chk_lg, _, _, chunk_logits = chunked_prefill(
        model, kv_pool_chk, rtp_chk, ws_chk, ids, chunk, cfg
    )

    assert len(chunk_logits) == 1, "Should be exactly one chunk"
    diff = (full_lg - chk_lg).abs().max().item()
    assert diff < ATOL, f"Single-chunk diff={diff:.4f} >= ATOL={ATOL}"


# ─────────────────────────────────────────────────────────────────────────────
# Edge case: chunk_size == 1 (extreme chunking, one token per chunk)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("prompt_key", ["short"])
@pytest.mark.parametrize("page_size", [8, 16])
def test_chunk_size_one(model, tok, cfg, prompt_key, page_size):
    """
    chunk_size=1: each token is a separate extend call.
    The last-token logit must still match F.sdpa.
    """
    ids = encode(tok, PROMPTS[prompt_key])
    L   = len(ids)

    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    kv_pool, rtp = make_pool(cfg, L + 8, page_size=page_size)

    _, _, _, all_chunk_logits = chunked_prefill(
        model, kv_pool, rtp, workspace, ids, chunk_size=1, cfg=cfg
    )

    assert len(all_chunk_logits) == L, "Should have one chunk per token"

    ref  = full_ref(model, ids)
    diff = (ref - all_chunk_logits[-1]).abs().max().item()
    assert diff < ATOL, (
        f"chunk_size=1 last-token diff={diff:.4f} >= ATOL={ATOL} "
        f"(page_size={page_size})"
    )
