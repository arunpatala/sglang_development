"""
verify_batch.py — Verify Layer 8's req_to_token table + Triton kv_indices kernel
                  against a full-recompute F.sdpa baseline.

What is being tested (layer8-specific):
  Prefill: PrefillKVCtx writes K/V to KVPool slots.
           model_runner writes slot indices to req_to_token[req_pool_idx, 0:L].
           Attention uses F.sdpa (side-effect = pool write).

  Decode:  req_to_token[req_pool_indices, seq_lens] = new_slots   (GPU scatter)
           kv_indptr built via torch.cumsum on GPU.
           kv_indices built by create_flashinfer_kv_indices_triton (Triton, GPU).
           BatchDecodeWithPagedKVCacheWrapper reads K/V from pool via kv_indices.

Reference (ground truth):
  At every step, the full token sequence is fed through the model with
  kv_cache=None — plain F.sdpa, full recompute, always correct.

Test method — lockstep over N decode steps:
  1. Prefill B prompts individually into KVPool + ReqToTokenPool.
  2. At each step:
       Reference: model(full_seq_i, kv_cache=None)   per request
       Paged:     Triton kernel → kv_indices → DecodeKVCtx → forward
       Both use the reference greedy token as input (kept in sync).
  3. Compare logits; assert max absolute diff < ATOL.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16 --atol 0.75
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

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
args = parser.parse_args()
ATOL = args.atol

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

    max_context = max(prompt_lens) + n_steps + 8

    # ── Allocate KVPool + ReqToTokenPool ──────────────────────────────────────
    max_tokens_needed = sum(prompt_lens) + B * (n_steps + 4) + 10
    kv_pool = KVPool(
        total_slots = max_tokens_needed,
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )
    rtp = ReqToTokenPool(max_batch=B + 2, max_context_len=max_context)

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
        workspace, "NHD", use_tensor_cores=False
    )

    # Pre-alloc kv_indptr buffer [B+1] and reuse each step (as model_runner does).
    kv_indptr_buf = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)

    # ── Prefill — PrefillKVCtx (B=1 per request) ──────────────────────────────
    slot_indices_per_req = []   # List[List[int]]
    req_pool_indices     = []   # List[int]  — which rtp row each request owns
    prefill_logits       = []

    for i in range(B):
        ids      = prompt_ids_list[i]
        prompt_t = torch.tensor([ids], device=DEVICE)
        mask     = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
        pos      = torch.arange(len(ids), device=DEVICE).unsqueeze(0)

        slots = kv_pool.alloc(len(ids))
        ctx   = PrefillKVCtx(slots, kv_pool)

        # Write slots to req_to_token (same as model_runner.prefill).
        rpi = rtp.alloc()
        slots_t = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        rtp.req_to_token[rpi, :len(ids)] = slots_t

        with torch.no_grad():
            logits = model(prompt_t, attention_mask=mask, kv_cache=ctx, position_ids=pos)

        slot_indices_per_req.append(slots)
        req_pool_indices.append(rpi)
        prefill_logits.append(logits[0, -1, :])

    # First greedy tokens seeded from prefill logits.
    next_toks_buf = [int(lg.argmax()) for lg in prefill_logits]

    # full_seqs for reference recompute: prompt + all generated tokens so far.
    full_seqs = [list(ids) for ids in prompt_ids_list]

    results = [[] for _ in range(B)]

    # ── Decode steps (lockstep) ────────────────────────────────────────────────
    for step in range(n_steps):
        # Append the token we're about to process to the reference sequence.
        for i in range(B):
            full_seqs[i].append(next_toks_buf[i])

        # ── Reference: full recompute kv_cache=None ───────────────────────────
        ref_logits = []
        for i in range(B):
            ids  = full_seqs[i]
            t    = torch.tensor([ids], device=DEVICE)
            mask = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
            pos  = torch.arange(len(ids), device=DEVICE).unsqueeze(0)
            with torch.no_grad():
                lg = model(t, attention_mask=mask, kv_cache=None, position_ids=pos)
            ref_logits.append(lg[0, -1, :])

        # ── Layer-8 paged decode ───────────────────────────────────────────────
        new_slots = [kv_pool.alloc(1)[0] for _ in range(B)]

        # Small CPU metadata → GPU  (O(B) ints, not O(Σ kv_lens)).
        seq_lens_list     = [len(slot_indices_per_req[i]) for i in range(B)]
        seq_lens_t        = torch.tensor(seq_lens_list,    dtype=torch.int32, device=DEVICE)
        req_pool_idx_t    = torch.tensor(req_pool_indices, dtype=torch.int32, device=DEVICE)
        new_slots_t_i32   = torch.tensor(new_slots,        dtype=torch.int32, device=DEVICE)

        # Write new slots into req_to_token on GPU (vectorised scatter).
        rtp.req_to_token[req_pool_idx_t, seq_lens_t] = new_slots_t_i32

        # Build kv_indptr on GPU via cumsum.
        seq_lens_with_new = seq_lens_t + 1
        kv_indptr_buf[0]  = 0
        torch.cumsum(seq_lens_with_new, dim=0, out=kv_indptr_buf[1:])
        kv_indptr = kv_indptr_buf[:B + 1]

        # Build kv_indices on GPU via Triton kernel.
        total_kv   = sum(s + 1 for s in seq_lens_list)
        kv_indices = torch.empty(total_kv, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            rtp.req_to_token,
            req_pool_idx_t,
            seq_lens_with_new,
            kv_indptr,
            None,
            kv_indices,
            rtp.req_to_token.shape[1],
        )

        kv_last_page_len = torch.ones(B, dtype=torch.int32, device=DEVICE)
        pos_ids = seq_lens_t.unsqueeze(1).to(torch.long)

        cur_toks = torch.tensor(
            [[t] for t in next_toks_buf], dtype=torch.long, device=DEVICE
        )

        decode_wrapper.begin_forward(
            kv_indptr, kv_indices, kv_last_page_len,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            1,
            data_type   = DTYPE,
            q_data_type = DTYPE,
        )

        new_slots_t_i64 = new_slots_t_i32.to(torch.int64)
        ctx = DecodeKVCtx(
            wrapper   = decode_wrapper,
            k_pool    = kv_pool.k_pool,
            v_pool    = kv_pool.v_pool,
            new_slots = new_slots_t_i64,
        )

        with torch.no_grad():
            bat_lg = model(
                cur_toks,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_ids,
            )

        decode_wrapper.end_forward()

        for i in range(B):
            slot_indices_per_req[i].append(new_slots[i])

        # ── Compare logits ─────────────────────────────────────────────────────
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

        # Advance: next step's input = greedy from this step's reference.
        next_toks_buf = [int(lg.argmax()) for lg in ref_logits]

    return results


def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — Layer 8: req_to_token + Triton kv_indices")
    print(f" Reference  : full recompute F.sdpa (kv_cache=None) per request")
    print(f" Paged      : ReqToTokenPool + KVPool + Triton kernel (B={len(PROMPTS)})")
    print(f" FlashInfer : BatchDecodeWithPagedKVCacheWrapper (page_size=1)")
    print(f" Steps  : {n}")
    print(f" Model  : {args.model}")
    print(f" Atol   : {ATOL}")
    print("=" * 64)
    print()

    print("Loading tokenizer + model …")
    tok = Tokenizer(args.model)
    mdl = Qwen3ForCausalLM.from_pretrained(args.model, dtype=DTYPE)
    print()

    results = run_lockstep(mdl, tok, n_steps=n)
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
