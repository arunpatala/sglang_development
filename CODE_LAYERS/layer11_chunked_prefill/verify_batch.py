"""
verify_batch.py — Verify Layer 10: batched prefill + chunked prefill.

Three tests, all compared against F.sdpa full-recompute (kv_cache=None):

  Test 1 — Batched prefill (B=4 in one EXTEND pass):
    Pack all 4 prompts into one prefill_batch call using ExtendKVCtx.
    Check each request's last-token logit vs. F.sdpa.

  Test 2 — Chunked prefill (B=1, chunk_size < prompt_len):
    One long prompt split into ceil(L/chunk_size) chunks.
    After each chunk, compare the last-position logit vs. F.sdpa baseline
    running over the full prefix.  Validates that cached KV from prior
    chunks is correctly read by the continuation-chunk extend path.

  Test 3 — Chunked prefill → decode (B=1):
    Same as Test 2 for the full prompt, then run N batched-decode steps.
    Each decode logit is compared against F.sdpa full recompute.

Usage:
    python verify_batch.py
    python verify_batch.py --page-size 16 --chunk-size 16 --n-compare 8 --atol 0.75
"""

import argparse
import math
import sys
from pathlib import Path
from typing import Dict, List

import torch
import flashinfer

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import DecodeKVCtx, ExtendKVCtx, KVPool, ReqToTokenPool, compute_write_info
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

ATOL = 0.75

parser = argparse.ArgumentParser()
parser.add_argument("--model",      default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare",  type=int,   default=8)
parser.add_argument("--page-size",  type=int,   default=16)
parser.add_argument("--chunk-size", type=int,   default=16)
parser.add_argument("--atol",       type=float, default=ATOL)
args = parser.parse_args()
ATOL       = args.atol
PAGE_SIZE  = args.page_size
CHUNK_SIZE = args.chunk_size

DEVICE = "cuda"
DTYPE  = torch.bfloat16

PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the difference between a compiled and interpreted language?",
]

LONG_PROMPT = (
    "The history of artificial intelligence is long and fascinating. "
    "It began in the 1950s with pioneers like Alan Turing, who proposed "
    "the famous Turing Test as a measure of machine intelligence. "
    "Since then, the field has gone through multiple winters and springs, "
    "culminating in the deep learning revolution of the 2010s. "
    "Today, large language models trained on vast corpora of text are able "
    "to converse, reason, and write code with remarkable fluency."
)


def bar(ok: bool) -> str:
    return "✓" if ok else "✗"


# ─────────────────────────────────────────────────────────────────────────────
# Core helper: run one EXTEND forward pass for a list of requests
# ─────────────────────────────────────────────────────────────────────────────

def do_extend(
    model,
    kv_pool: KVPool,
    rtp: ReqToTokenPool,
    workspace: torch.Tensor,
    reqs: List[Dict],          # each: {fill_ids, kv_committed_len, slot_indices, req_pool_idx}
    cfg,
) -> tuple:
    """
    Run one batched extend forward pass with page packing.

    Uses compute_write_info() to correctly handle chunk-boundary page filling.
    Mutates req["slot_indices"] in-place.

    Returns: (logits: [1, total_tokens, vocab], qo_indptr: List[int])
    """
    P = kv_pool.page_size
    B = len(reqs)

    write_infos:    List = []
    qo_indptr_list: List[int] = [0]
    num_pages_list: List[int] = []
    kv_last_pg_list: List[int] = []

    for rd in reqs:
        fill_ids         = rd["fill_ids"]
        kv_committed_len = rd["kv_committed_len"]
        slot_indices     = rd["slot_indices"]
        req_pool_idx     = rd["req_pool_idx"]

        wi = compute_write_info(
            kv_pool          = kv_pool,
            rtp              = rtp,
            slot_indices     = slot_indices,
            req_pool_idx     = req_pool_idx,
            kv_committed_len = kv_committed_len,
            n_fill           = len(fill_ids),
        )
        write_infos.append(wi)

        n_pages = len(slot_indices)   # updated in-place by compute_write_info
        num_pages_list.append(n_pages)

        total_committed = kv_committed_len + len(fill_ids)
        last_fill = total_committed % P
        kv_last_pg_list.append(last_fill if last_fill != 0 else P)
        qo_indptr_list.append(qo_indptr_list[-1] + len(fill_ids))

    qo_indptr_t      = torch.tensor(qo_indptr_list,  dtype=torch.int32, device=DEVICE)
    num_pages_t      = torch.tensor(num_pages_list,   dtype=torch.int32, device=DEVICE)
    kv_last_pg_t     = torch.tensor(kv_last_pg_list,  dtype=torch.int32, device=DEVICE)
    req_pool_idx_t   = torch.tensor(
        [rd["req_pool_idx"] for rd in reqs], dtype=torch.int32, device=DEVICE
    )

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    torch.cumsum(num_pages_t, dim=0, out=kv_indptr[1:])

    total_pg = int(num_pages_t.sum().item())
    kv_indices = torch.empty(total_pg, dtype=torch.int32, device=DEVICE)
    create_flashinfer_kv_indices_triton[(B,)](
        rtp.req_to_token, req_pool_idx_t, num_pages_t, kv_indptr,
        None, kv_indices, rtp.req_to_token.shape[1],
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.begin_forward(
        qo_indptr_t, kv_indptr, kv_indices, kv_last_pg_t,
        cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.head_dim, P, causal=True, q_data_type=DTYPE,
    )

    all_ids: List[int] = []
    pos_ids: List[int] = []
    for rd in reqs:
        all_ids.extend(rd["fill_ids"])
        for j in range(len(rd["fill_ids"])):
            pos_ids.append(rd["kv_committed_len"] + j)

    ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    pos_t = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    ctx = ExtendKVCtx(
        wrapper     = wrapper,
        k_pool      = kv_pool.k_pool,
        v_pool      = kv_pool.v_pool,
        qo_indptr   = qo_indptr_list,
        write_infos = write_infos,
        page_size   = P,
    )

    with torch.no_grad():
        logits = model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)

    wrapper.end_forward()
    return logits, qo_indptr_list


def ref_logit_last(model, token_ids: List[int]) -> torch.Tensor:
    """Full F.sdpa recompute over token_ids; returns logit at last position."""
    ids_t = torch.tensor([token_ids], device=DEVICE)
    pos_t = torch.arange(len(token_ids), device=DEVICE).unsqueeze(0)
    mask  = torch.ones(1, len(token_ids), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        lg = model(ids_t, attention_mask=mask, kv_cache=None, position_ids=pos_t)
    return lg[0, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Batched prefill — all prompts in one EXTEND pass
# ─────────────────────────────────────────────────────────────────────────────

def test_batched_prefill(model, tok, cfg) -> bool:
    P = PAGE_SIZE
    print("─" * 60)
    print(f"Test 1: Batched prefill (B={len(PROMPTS)}, one EXTEND pass)")
    print("─" * 60)

    prompt_ids: List[List[int]] = []
    for p in PROMPTS:
        text = tok.apply_chat_template([{"role": "user", "content": p}])
        ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()
        prompt_ids.append(ids)

    B = len(prompt_ids)
    max_ctx    = max(math.ceil(len(ids) / P) for ids in prompt_ids) + 4
    total_pgs  = sum(math.ceil(len(ids) / P) for ids in prompt_ids) + 10
    kv_pool    = KVPool(total_pgs, P, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, DTYPE)
    rtp        = ReqToTokenPool(B + 2, max_ctx)
    workspace  = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    reqs = [
        {"fill_ids": ids, "kv_committed_len": 0, "slot_indices": [], "req_pool_idx": rtp.alloc()}
        for ids in prompt_ids
    ]
    logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, reqs, cfg)

    all_ok = True
    for i, ids in enumerate(prompt_ids):
        ref   = ref_logit_last(model, ids)
        paged = logits[0, qo_indptr[i + 1] - 1, :]
        diff  = (ref - paged).abs().max().item()
        ok    = diff < ATOL
        all_ok = all_ok and ok
        print(f"  [{i}] '{PROMPTS[i][:42]}'")
        print(f"       max_diff={diff:.4f}  {bar(ok)}")

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Chunked prefill — one prompt in ceil(L/chunk_size) EXTEND passes
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_prefill(model, tok, cfg) -> bool:
    P, C = PAGE_SIZE, CHUNK_SIZE
    print("─" * 60)
    print(f"Test 2: Chunked prefill (B=1, chunk_size={C})")
    print("─" * 60)

    text = tok.apply_chat_template([{"role": "user", "content": LONG_PROMPT}])
    ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()
    L    = len(ids)
    n_chunks = math.ceil(L / C)
    print(f"  Prompt length: {L} tokens  → {n_chunks} chunks of ≤{C}")

    max_ctx   = math.ceil(L / P) + 4
    total_pgs = max_ctx + 4
    kv_pool   = KVPool(total_pgs, P, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, DTYPE)
    rtp       = ReqToTokenPool(4, max_ctx)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    slot_indices: List[int] = []
    req_pool_idx = rtp.alloc()
    kv_committed = 0
    all_ok = True

    for c in range(n_chunks):
        start, end = c * C, min((c + 1) * C, L)
        rd = [{
            "fill_ids":         ids[start:end],
            "kv_committed_len": kv_committed,
            "slot_indices":     slot_indices,
            "req_pool_idx":     req_pool_idx,
        }]
        logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, rd, cfg)
        kv_committed = end

        ref   = ref_logit_last(model, ids[:end])
        paged = logits[0, qo_indptr[1] - 1, :]
        diff  = (ref - paged).abs().max().item()
        ok    = diff < ATOL
        all_ok = all_ok and ok
        print(f"  chunk {c:2d} [{start:4d}:{end:4d}]  diff={diff:.4f}  {bar(ok)}")

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Chunked prefill followed by decode steps
# ─────────────────────────────────────────────────────────────────────────────

def test_chunked_then_decode(model, tok, cfg, n_decode: int) -> bool:
    P, C = PAGE_SIZE, CHUNK_SIZE
    print("─" * 60)
    print(f"Test 3: Chunked prefill → decode ({n_decode} steps, chunk_size={C})")
    print("─" * 60)

    text = tok.apply_chat_template([{"role": "user", "content": PROMPTS[0]}])
    ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()
    L    = len(ids)

    max_ctx   = math.ceil((L + n_decode + 4) / P) + 4
    total_pgs = max_ctx + 10
    kv_pool   = KVPool(total_pgs, P, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, DTYPE)
    rtp       = ReqToTokenPool(4, max_ctx)
    ws_extend = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    ws_decode = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    dec_wrap  = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws_decode, "NHD", use_tensor_cores=False)
    kv_iptr   = torch.zeros(2, dtype=torch.int32, device=DEVICE)

    slot_indices: List[int] = []
    rpi = rtp.alloc()
    kv_committed = 0

    # ── Chunked prefill ────────────────────────────────────────────────────
    for c in range(math.ceil(L / C)):
        start, end = c * C, min((c + 1) * C, L)
        rd = [{"fill_ids": ids[start:end], "kv_committed_len": kv_committed,
               "slot_indices": slot_indices, "req_pool_idx": rpi}]
        logits, qo_indptr = do_extend(model, kv_pool, rtp, ws_extend, rd, cfg)
        kv_committed = end

    # ── Sample first decode token from last chunk's last-position logit ───
    next_tok = int(logits[0, qo_indptr[1] - 1, :].argmax())
    full_seq  = list(ids) + [next_tok]

    print(f"  Prompt length: {L} tokens  chunks: {math.ceil(L/C)}")

    # ── Decode steps ───────────────────────────────────────────────────────
    all_ok = True
    for step in range(n_decode):
        seq_len      = len(full_seq) - 1       # tokens in pool
        token_offset = seq_len % P
        num_pages    = len(slot_indices)

        if token_offset == 0:
            new_page = kv_pool.alloc(1)[0]
            slot_indices.append(new_page)
            rtp.req_to_token[rpi, num_pages] = new_page
            last_pg = new_page
            num_pages += 1
        else:
            last_pg = slot_indices[-1]

        seq_t    = torch.tensor([seq_len],   dtype=torch.int32, device=DEVICE)
        tok_off  = torch.tensor([token_offset], dtype=torch.int32, device=DEVICE)
        npg_t    = torch.tensor([num_pages], dtype=torch.int32, device=DEVICE)
        rpi_t    = torch.tensor([rpi],       dtype=torch.int32, device=DEVICE)
        lpg_t    = torch.tensor([last_pg],   dtype=torch.int64, device=DEVICE)
        toff_i64 = tok_off.to(torch.int64)
        kv_last  = tok_off + 1

        kv_iptr[0] = 0; kv_iptr[1] = num_pages
        kv_idx = torch.empty(num_pages, dtype=torch.int32, device=DEVICE)
        create_flashinfer_kv_indices_triton[(1,)](
            rtp.req_to_token, rpi_t, npg_t, kv_iptr,
            None, kv_idx, rtp.req_to_token.shape[1],
        )

        dec_wrap.begin_forward(
            kv_iptr, kv_idx, kv_last,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, P,
            data_type=DTYPE, q_data_type=DTYPE,
        )
        ctx = DecodeKVCtx(
            wrapper=dec_wrap, k_pool=kv_pool.k_pool, v_pool=kv_pool.v_pool,
            last_page_indices=lpg_t, token_offsets=toff_i64,
        )
        cur_tok_t = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        pos_t     = torch.tensor([[seq_len]], dtype=torch.long, device=DEVICE)
        with torch.no_grad():
            lg = model(cur_tok_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
        dec_wrap.end_forward()

        ref  = ref_logit_last(model, full_seq)
        diff = (ref - lg[0, -1, :]).abs().max().item()
        ok   = diff < ATOL
        all_ok = all_ok and ok
        print(f"  decode step {step:2d}  diff={diff:.4f}  {bar(ok)}")

        next_tok = int(ref.argmax())
        full_seq.append(next_tok)

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print(f" verify_batch.py — Layer 10: Batched + Chunked Prefill")
    print(f" page_size={PAGE_SIZE}  chunk_size={CHUNK_SIZE}  atol={ATOL}")
    print(f" Model: {args.model}")
    print("=" * 64)
    print()

    print("Loading tokenizer + model …")
    tok = Tokenizer(args.model)
    mdl = Qwen3ForCausalLM.from_pretrained(args.model, dtype=DTYPE)
    cfg = mdl.model.config
    print()

    r1 = test_batched_prefill(mdl, tok, cfg)
    r2 = test_chunked_prefill(mdl, tok, cfg)
    r3 = test_chunked_then_decode(mdl, tok, cfg, args.n_compare)

    overall = r1 and r2 and r3
    print("=" * 64)
    print(f" OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    print("=" * 64)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
