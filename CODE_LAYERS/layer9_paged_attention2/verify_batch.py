"""
verify_batch.py — Verify Layer 9's ReqToTokenPool + Triton kv_indices kernel
                  + variable page_size against a full-recompute F.sdpa baseline.

What is being tested:
  Prefill: PrefillKVCtx pads prompt K/V to multiples of PAGE_SIZE, scatter-
           writes ceil(L/PAGE_SIZE) pages into KVPool.
           Attention uses F.sdpa (side-effect = pool write).

  Decode:  Conditional page allocation (new page only when last page fills).
           kv_indptr   built via torch.cumsum on GPU (page counts).
           kv_indices  built by create_flashinfer_kv_indices_triton (Triton, GPU).
           kv_last_page_lens = token_offset_within_last_page + 1 (variable).
           BatchDecodeWithPagedKVCacheWrapper reads K/V from pool via kv_indices.

Reference (ground truth):
  At every decode step, the full token sequence is fed through the model with
  kv_cache=None — plain F.sdpa, full recompute, always correct.

Test method — lockstep over N decode steps:
  1. Prefill B prompts individually into KVPool + ReqToTokenPool.
  2. At each step:
       Reference: model(full_seq_i, ForwardBatch(PREFILL, kv_cache=None)) per request
       Paged:     Triton kernel → kv_indices → DecodeKVCtx → ForwardBatch(DECODE)
       Both use the reference greedy token as input (kept in sync).
  3. Compare logits; assert max absolute diff < ATOL.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16 --atol 0.75 --page-size 16
"""

import argparse
import math
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from forward_batch import ForwardBatch, ForwardMode
from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx, ReqToTokenPool
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

import flashinfer

ATOL = 0.75   # bfloat16 ~3 ULPs at logit scale ~40

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int,   default=16)
parser.add_argument("--atol",      type=float, default=ATOL)
parser.add_argument("--page-size", type=int,   default=16)
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


def run_lockstep(model, tok, n_steps: int):
    B   = len(PROMPTS)
    P   = PAGE_SIZE
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
    print(f"  Prompt lengths : {prompt_lens}")
    print(f"  Page size      : {P}")

    # ── Allocate KVPool + ReqToTokenPool ─────────────────────────────────────
    # KVPool needs enough pages for all prompt pages + decode pages.
    max_prompt_pages = sum(math.ceil(pl / P) for pl in prompt_lens)
    max_decode_pages = math.ceil((n_steps + 2) / P) * B
    total_pages_needed = max_prompt_pages + max_decode_pages + 4

    kv_pool = KVPool(
        total_pages = total_pages_needed,
        page_size   = P,
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )

    max_pages_per_req = math.ceil((max(prompt_lens) + n_steps + 4) / P)
    rtp = ReqToTokenPool(max_batch=B + 2, max_context_len=max_pages_per_req)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=False
    )

    kv_indptr_buf = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)

    # ── Prefill — PrefillKVCtx (B=1 per request) ─────────────────────────────
    page_indices_per_req = []   # List[List[int]] — physical page indices
    req_pool_indices     = []   # List[int]       — which rtp row each request owns
    num_tokens_in_cache  = []   # List[int]       — tokens in pool after prefill
    prefill_logits       = []

    for i in range(B):
        ids      = prompt_ids_list[i]
        L        = len(ids)
        n_pages  = math.ceil(L / P)

        prompt_t = torch.tensor([ids], device=DEVICE)
        mask     = torch.ones(1, L, dtype=torch.long, device=DEVICE)
        pos      = torch.arange(L, device=DEVICE).unsqueeze(0)

        pages = kv_pool.alloc(L)   # returns n_pages page indices
        ctx   = PrefillKVCtx(pages, kv_pool)

        rpi = rtp.alloc()
        pages_t = torch.tensor(pages, dtype=torch.int32, device=DEVICE)
        rtp.req_to_token[rpi, :n_pages] = pages_t

        fb = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
        with torch.no_grad():
            logits = model(prompt_t, forward_batch=fb, position_ids=pos)

        page_indices_per_req.append(list(pages))
        req_pool_indices.append(rpi)
        num_tokens_in_cache.append(L)   # prompt tokens now in pool
        prefill_logits.append(logits[0, -1, :])

    # First greedy tokens seeded from prefill logits.
    next_toks_buf = [int(lg.argmax()) for lg in prefill_logits]

    # full_seqs for reference recompute: prompt + all generated tokens so far.
    full_seqs = [list(ids) for ids in prompt_ids_list]

    results = [[] for _ in range(B)]

    # ── Decode steps (lockstep) ───────────────────────────────────────────────
    for step in range(n_steps):
        # Append the token we're about to process to the reference sequence.
        for i in range(B):
            full_seqs[i].append(next_toks_buf[i])

        # ── Reference: full recompute kv_cache=None ───────────────────────
        ref_logits = []
        for i in range(B):
            ids  = full_seqs[i]
            t    = torch.tensor([ids], device=DEVICE)
            mask = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
            pos  = torch.arange(len(ids), device=DEVICE).unsqueeze(0)
            fb_ref = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=None, attention_mask=mask)
            with torch.no_grad():
                lg = model(t, forward_batch=fb_ref, position_ids=pos)
            ref_logits.append(lg[0, -1, :])

        # ── Layer-9 paged decode ──────────────────────────────────────────
        # seq_len[i] = tokens already in KV cache = position of current token.
        seq_lens_list     = list(num_tokens_in_cache)
        token_offsets_list = [sl % P for sl in seq_lens_list]
        num_pages_list     = [len(page_indices_per_req[i]) for i in range(B)]

        # Conditional page allocation: new page only when last page is full.
        last_page_idx_list = []
        for i in range(B):
            if token_offsets_list[i] == 0:
                new_page = kv_pool.alloc(1)[0]
                page_indices_per_req[i].append(new_page)
                # Scalar write to req_to_token on GPU.
                rtp.req_to_token[req_pool_indices[i], num_pages_list[i]] = new_page
                last_page_idx_list.append(new_page)
                num_pages_list[i] += 1
            else:
                last_page_idx_list.append(page_indices_per_req[i][-1])

        seq_lens_t         = torch.tensor(seq_lens_list,        dtype=torch.int32, device=DEVICE)
        token_offsets_t    = torch.tensor(token_offsets_list,   dtype=torch.int32, device=DEVICE)
        num_pages_t        = torch.tensor(num_pages_list,        dtype=torch.int32, device=DEVICE)
        req_pool_idx_t     = torch.tensor(req_pool_indices,      dtype=torch.int32, device=DEVICE)
        last_page_idx_t    = torch.tensor(last_page_idx_list,    dtype=torch.int64, device=DEVICE)
        token_offsets_i64  = token_offsets_t.to(torch.int64)

        # kv_last_page_lens: valid tokens in last page after this write.
        kv_last_page_lens = token_offsets_t + 1   # range 1..P

        # kv_indptr on GPU via cumsum (page counts, not token counts).
        kv_indptr_buf[0] = 0
        torch.cumsum(num_pages_t, dim=0, out=kv_indptr_buf[1 : B + 1])
        kv_indptr = kv_indptr_buf[: B + 1]

        # kv_indices on GPU via Triton kernel.
        total_pages_in_batch = int(num_pages_t.sum().item())
        kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)
        create_flashinfer_kv_indices_triton[(B,)](
            rtp.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr,
            None,
            kv_indices,
            rtp.req_to_token.shape[1],
        )

        pos_ids  = seq_lens_t.unsqueeze(1).to(torch.long)   # [B, 1]
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
        fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)

        with torch.no_grad():
            bat_lg = model(cur_toks, forward_batch=fb, position_ids=pos_ids)

        decode_wrapper.end_forward()

        # Advance token-in-cache count for each request.
        for i in range(B):
            num_tokens_in_cache[i] += 1

        # ── Compare logits ────────────────────────────────────────────────
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

        # Next step's input = greedy from this step's reference.
        next_toks_buf = [int(lg.argmax()) for lg in ref_logits]

    return results


def main():
    n = args.n_compare
    print("=" * 68)
    print(f" verify_batch.py — Layer 9: ReqToTokenPool + Triton kv_indices")
    print(f"                            + variable page_size")
    print(f" Reference  : full recompute F.sdpa (kv_cache=None) per request")
    print(f" Paged      : ReqToTokenPool + KVPool + Triton kernel (B={len(PROMPTS)})")
    print(f" FlashInfer : BatchDecodeWithPagedKVCacheWrapper (page_size={PAGE_SIZE})")
    print(f" Steps  : {n}")
    print(f" Model  : {args.model}")
    print(f" Atol   : {ATOL}")
    print("=" * 68)
    print()

    print("Loading tokenizer + model …")
    tok = Tokenizer(args.model)
    mdl = Qwen3ForCausalLM.from_pretrained(args.model, dtype=DTYPE)
    print()

    results = run_lockstep(mdl, tok, n_steps=n)
    print()

    overall_pass = True
    for i, prompt in enumerate(PROMPTS):
        steps     = results[i]
        n_match   = sum(r[0] for r in steps)
        avg_diff  = sum(r[1] for r in steps) / len(steps)
        max_diff  = max(r[1] for r in steps)
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

    print("=" * 68)
    print(f" RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 68)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
