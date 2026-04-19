"""
verify_batch.py — Verify Layer 9's page_size > 1 paged KV cache
                  against a full-recompute F.sdpa baseline.

What is being tested (layer9-specific):
  KVPool shape: [total_pages, page_size, n_kv, head_dim]
  Prefill:  PrefillKVCtx.store() pads and reshapes K/V to page granularity.
  Decode:   Conditional page alloc (new page only when token_offset == 0).
            kv_last_page_lens = token_offset + 1  (variable, 1..page_size).
            kv_indptr based on num_pages, not num_tokens.
            Triton kernel unchanged — reads page indices from req_to_token.

Reference: model(full_seq, kv_cache=None) — F.sdpa full recompute.

Usage:
    python verify_batch.py
    python verify_batch.py --page-size 16 --n-compare 20 --atol 0.75
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx, ReqToTokenPool
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

import flashinfer

ATOL = 0.75

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int,   default=20)
parser.add_argument("--page-size", type=int,   default=16)
parser.add_argument("--atol",      type=float, default=ATOL)
args = parser.parse_args()
ATOL      = args.atol
PAGE_SIZE = args.page_size

DEVICE = "cuda"
DTYPE  = torch.bfloat16

PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the difference between a compiled and interpreted language?",
]


def bar(ok: bool) -> str:
    return "✓" if ok else "✗"


def run_lockstep(model, tok, n_steps: int, page_size: int):
    B   = len(PROMPTS)
    P   = page_size
    cfg = model.model.config

    formatted = [
        tok.apply_chat_template([{"role": "user", "content": p}])
        for p in PROMPTS
    ]
    prompt_ids_list = []
    for text in formatted:
        enc = tok._tok([text], return_tensors="pt", padding=False)
        prompt_ids_list.append(enc["input_ids"][0].tolist())

    prompt_lens = [len(ids) for ids in prompt_ids_list]
    print(f"  Prompt lengths : {prompt_lens}  page_size={P}")

    max_token_ctx = max(prompt_lens) + n_steps + 8
    max_pages_per_req = math.ceil(max_token_ctx / P)

    # ── Allocate KVPool + ReqToTokenPool ──────────────────────────────────
    # Total pages needed = sum(ceil(L/P)) + B*(steps+4 pages)
    total_pages_needed = (
        sum(math.ceil(L / P) for L in prompt_lens)
        + B * math.ceil((n_steps + 4) / P)
        + 10
    )
    kv_pool = KVPool(
        total_pages = total_pages_needed,
        page_size   = P,
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )
    rtp = ReqToTokenPool(max_batch=B + 2, max_context_len=max_pages_per_req)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=False
    )
    kv_indptr_buf = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)

    # ── Prefill — PrefillKVCtx (B=1 per request) ──────────────────────────
    page_indices_per_req = []   # List[List[int]]  — page indices per request
    req_pool_indices     = []   # List[int]
    prefill_logits       = []

    for i in range(B):
        ids      = prompt_ids_list[i]
        L        = len(ids)
        n_pages  = math.ceil(L / P)
        prompt_t = torch.tensor([ids], device=DEVICE)
        mask     = torch.ones(1, L, dtype=torch.long, device=DEVICE)
        pos      = torch.arange(L, device=DEVICE).unsqueeze(0)

        pages = kv_pool.alloc(L)
        ctx   = PrefillKVCtx(pages, kv_pool)

        rpi     = rtp.alloc()
        pages_t = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
        rtp.req_to_token[rpi, :n_pages] = pages_t

        with torch.no_grad():
            logits = model(prompt_t, attention_mask=mask, kv_cache=ctx, position_ids=pos)

        page_indices_per_req.append(pages)
        req_pool_indices.append(rpi)
        prefill_logits.append(logits[0, -1, :])

    next_toks_buf = [int(lg.argmax()) for lg in prefill_logits]
    full_seqs     = [list(ids) for ids in prompt_ids_list]
    results       = [[] for _ in range(B)]

    # ── Decode steps (lockstep) ────────────────────────────────────────────
    for step in range(n_steps):
        for i in range(B):
            full_seqs[i].append(next_toks_buf[i])

        # ── Reference: full recompute kv_cache=None ───────────────────────
        ref_logits = []
        for i in range(B):
            ids  = full_seqs[i]
            t    = torch.tensor([ids], device=DEVICE)
            mask = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
            pos  = torch.arange(len(ids), device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                lg = model(t, attention_mask=mask, kv_cache=None, position_ids=pos)
            ref_logits.append(lg[0, -1, :])

        # ── Layer-9 paged decode ───────────────────────────────────────────
        # seq_len = prompt_len + decode tokens so far (= len(full_seqs[i]) - 1 + 1)
        # Actually full_seqs[i] was just appended with next_toks_buf[i], so
        # seq_len (tokens in pool BEFORE this step) = len(full_seqs[i]) - 1.
        seq_lens_list      = [len(full_seqs[i]) - 1 for i in range(B)]
        token_offsets_list = [sl % P               for sl in seq_lens_list]
        num_pages_list     = [len(page_indices_per_req[i]) for i in range(B)]

        last_page_idx_list = []
        for i in range(B):
            if token_offsets_list[i] == 0:
                new_page = kv_pool.alloc(1)[0]
                page_indices_per_req[i].append(new_page)
                rtp.req_to_token[req_pool_indices[i], num_pages_list[i]] = new_page
                last_page_idx_list.append(new_page)
                num_pages_list[i] += 1
            else:
                last_page_idx_list.append(page_indices_per_req[i][-1])

        seq_lens_t       = torch.tensor(seq_lens_list,      dtype=torch.int32, device=DEVICE)
        token_offsets_t  = torch.tensor(token_offsets_list, dtype=torch.int32, device=DEVICE)
        num_pages_t      = torch.tensor(num_pages_list,     dtype=torch.int32, device=DEVICE)
        req_pool_idx_t   = torch.tensor(req_pool_indices,   dtype=torch.int32, device=DEVICE)
        last_page_idx_t  = torch.tensor(last_page_idx_list, dtype=torch.int64, device=DEVICE)
        token_offsets_i64 = token_offsets_t.to(torch.int64)

        kv_last_page_lens = token_offsets_t + 1

        kv_indptr_buf[0] = 0
        torch.cumsum(num_pages_t, dim=0, out=kv_indptr_buf[1:])
        kv_indptr = kv_indptr_buf[:B + 1]

        total_pages_batch = sum(num_pages_list)
        kv_indices = torch.empty(total_pages_batch, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            rtp.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr,
            None,
            kv_indices,
            rtp.req_to_token.shape[1],
        )

        pos_ids  = seq_lens_t.unsqueeze(1).to(torch.long)
        cur_toks = torch.tensor(
            [[t] for t in next_toks_buf], dtype=torch.long, device=DEVICE
        )

        decode_wrapper.begin_forward(
            kv_indptr, kv_indices, kv_last_page_lens,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            P,
            data_type   = DTYPE,
            q_data_type = DTYPE,
        )

        ctx = DecodeKVCtx(
            wrapper           = decode_wrapper,
            k_pool            = kv_pool.k_pool,
            v_pool            = kv_pool.v_pool,
            last_page_indices = last_page_idx_t,
            token_offsets     = token_offsets_i64,
        )

        with torch.no_grad():
            bat_lg = model(
                cur_toks,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_ids,
            )

        decode_wrapper.end_forward()

        # ── Compare logits ─────────────────────────────────────────────────
        paged_logits = bat_lg[:, -1, :]
        for i in range(B):
            rl   = ref_logits[i]
            pl   = paged_logits[i]
            diff = (rl - pl).abs().max().item()
            results[i].append((
                int(rl.argmax()) == int(pl.argmax()),
                diff,
                int(rl.argmax()),
                int(pl.argmax()),
            ))

        next_toks_buf = [int(lg.argmax()) for lg in ref_logits]

    return results


def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — Layer 9: page_size={PAGE_SIZE}")
    print(f" Reference  : full recompute F.sdpa (kv_cache=None)")
    print(f" Paged      : ReqToTokenPool + KVPool[pages,P,n_kv,D] (B={len(PROMPTS)})")
    print(f" FlashInfer : BatchDecodeWithPagedKVCacheWrapper")
    print(f" page_size  : {PAGE_SIZE}")
    print(f" Steps  : {n}")
    print(f" Model  : {args.model}")
    print(f" Atol   : {ATOL}")
    print("=" * 64)
    print()

    print("Loading tokenizer + model …")
    tok = Tokenizer(args.model)
    mdl = Qwen3ForCausalLM.from_pretrained(args.model, dtype=DTYPE)
    print()

    results = run_lockstep(mdl, tok, n_steps=n, page_size=PAGE_SIZE)
    print()

    overall_pass = True
    for i, prompt in enumerate(PROMPTS):
        steps    = results[i]
        n_match  = sum(r[0] for r in steps)
        avg_diff = sum(r[1] for r in steps) / len(steps)
        max_diff = max(r[1] for r in steps)
        logits_ok = all(r[1] < ATOL for r in steps)

        print(f"  [{i}] '{prompt}'")
        print(f"       Tokens   : {n_match}/{len(steps)} exact  {bar(n_match == len(steps))}")
        print(f"       Avg diff : {avg_diff:.4f}   Max diff : {max_diff:.4f}  {bar(logits_ok)}")
        print(f"       Result   : {'PASS ✓' if logits_ok else 'FAIL ✗'}")

        if not logits_ok:
            overall_pass = False
            bad = [(s, f"{r[1]:.4f}") for s, r in enumerate(steps) if r[1] >= ATOL]
            print(f"       Steps > atol : {bad}")

        mismatches = [(s, r) for s, r in enumerate(steps) if not r[0]]
        for s, r in mismatches:
            ref_t   = repr(tok._tok.decode([r[2]]))
            paged_t = repr(tok._tok.decode([r[3]]))
            label   = "(near-tie)" if r[1] < ATOL else "(BUG)"
            print(f"       step {s:2d}: ref→{ref_t:12s}  paged→{paged_t:12s}  "
                  f"diff={r[1]:.4f}  {label}")
        print()

    print("=" * 64)
    print(f" RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 64)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
