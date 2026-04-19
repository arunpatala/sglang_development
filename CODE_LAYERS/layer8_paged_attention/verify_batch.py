"""
verify_batch.py — Verify that Layer 7's paged KV cache + FlashInfer decode
                  produces logits consistent with a full-recompute F.sdpa baseline.

What is being tested (layer7-specific):
  Prefill: PrefillKVCtx writes K/V from all 28 layers into KVPool slots.
           Attention uses F.sdpa over the fresh tensors (side-effect = pool write).

  Decode:  DecodeKVCtx writes the new token's K/V to a new pool slot per request.
           FlashInfer reads the full K/V history via kv_indices (no copy).
           We batch B=4 requests in a single forward pass.

Reference (ground truth):
  At every step, each request's full token sequence (prompt + generated so far)
  is fed through the model with kv_cache=None — plain F.sdpa, no cache, full
  recompute. This is always correct regardless of KV layout or attention kernel.

Test method — lockstep comparison over N decode steps:
  1. Prefill all B prompts into the shared KVPool (PrefillKVCtx, B=1 each).
  2. At each decode step:
       Reference: model(full_seq_i, mask, kv_cache=None)  per request → logit_i_ref
       Paged:     DecodeKVCtx + BatchPrefillWithPagedKVCacheWrapper → logit_i_paged
       Use greedy argmax of the reference to select the next token (both paths stay in sync).
  3. Compare logits, greedy tokens, and max absolute difference.

Pass condition: max logit diff < ATOL at every step for every prompt.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16 --atol 0.75
"""

import argparse
import sys
from itertools import accumulate
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from forward_batch import ForwardBatch, ForwardMode
from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer

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

# Deliberately different prompt lengths to stress-test slot boundary handling.
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

    # Tokenise each prompt — no padding
    prompt_ids_list = []
    for text in formatted:
        enc = tok._tok([text], return_tensors="pt", padding=False)
        prompt_ids_list.append(enc["input_ids"][0].tolist())

    prompt_lens = [len(ids) for ids in prompt_ids_list]
    print(f"  Prompt lengths : {prompt_lens}")

    # ── Allocate a KVPool sized for the test (prompt_len + n_steps per req) ──
    max_tokens_needed = sum(prompt_lens) + B * (n_steps + 4)
    kv_pool = KVPool(
        total_slots = max_tokens_needed + 10,   # small headroom
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )

    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    # ── Prefill — PrefillKVCtx (B=1 per request) ─────────────────────────────
    # Writes K/V to pool slots and returns logit at last prompt position.
    # We save the prefill logit to seed the first decode step.
    slot_indices_per_req = []   # List[List[int]], one entry per request
    prefill_logits       = []

    for i in range(B):
        ids      = prompt_ids_list[i]
        prompt_t = torch.tensor([ids], device=DEVICE)
        mask     = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
        pos      = torch.arange(len(ids), device=DEVICE).unsqueeze(0)

        slots = kv_pool.alloc(len(ids))
        ctx   = PrefillKVCtx(slots, kv_pool)
        fb    = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)

        with torch.no_grad():
            logits = model(prompt_t, forward_batch=fb, position_ids=pos)

        slot_indices_per_req.append(slots)
        prefill_logits.append(logits[0, -1, :])   # logit for first generated token

    # First greedy tokens from prefill — the first INPUT to both decode paths.
    next_toks_buf = [int(lg.argmax()) for lg in prefill_logits]

    # full_seqs tracks [prompt + all generated tokens] for the reference recompute.
    # Starts as the prompt; we append each greedy token before the reference runs.
    full_seqs = [list(ids) for ids in prompt_ids_list]

    results = [[] for _ in range(B)]

    # ── Decode steps (lockstep) ───────────────────────────────────────────────
    #
    # Alignment:  next_toks_buf  holds the token that BOTH paths take as INPUT
    #             at this step.  Both produce the logit for the FOLLOWING token.
    #
    # Step 0:  next_toks_buf = first generated tokens (from prefill logit)
    #   paged:     decode([first_tok], pos=prompt_len, kv=[prompt..., new_slot])
    #              → logit for token at position prompt_len+1
    #   reference: model([prompt + first_tok])
    #              → logit at last position (= prompt_len)  ← same position
    #   compare those two logits  ✓
    #
    for step in range(n_steps):
        # Extend reference sequence with the token we're about to process.
        for i in range(B):
            full_seqs[i].append(next_toks_buf[i])

        # ── Reference: full recompute with kv_cache=None (F.sdpa ground truth) ─
        ref_logits = []
        for i in range(B):
            ids  = full_seqs[i]           # prompt + all tokens up to & including next_toks_buf[i]
            t    = torch.tensor([ids], device=DEVICE)
            mask = torch.ones(1, len(ids), dtype=torch.long, device=DEVICE)
            pos  = torch.arange(len(ids), device=DEVICE).unsqueeze(0)
            fb_ref = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=None, attention_mask=mask)
            with torch.no_grad():
                lg = model(t, forward_batch=fb_ref, position_ids=pos)
            ref_logits.append(lg[0, -1, :])   # logit at last position

        # ── Paged decode: B=N DecodeKVCtx + FlashInfer ────────────────────────
        # Input: next_toks_buf (same tokens fed to reference above).
        new_slots = [kv_pool.alloc(1)[0] for _ in range(B)]

        kv_lens_plus1   = [len(slot_indices_per_req[i]) + 1 for i in range(B)]
        kv_indptr_list  = [0] + list(accumulate(kv_lens_plus1))
        kv_indices_list = []
        for i in range(B):
            kv_indices_list.extend(slot_indices_per_req[i])   # historical
            kv_indices_list.append(new_slots[i])               # new token

        kv_indptr        = torch.tensor(kv_indptr_list,  dtype=torch.int32, device=DEVICE)
        kv_indices       = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)
        kv_last_page_len = torch.ones(B, dtype=torch.int32, device=DEVICE)

        # Position = len(slot_indices_per_req[i]) = prompt_len + decode_steps_so_far
        pos_ids = torch.tensor(
            [[len(slot_indices_per_req[i])] for i in range(B)],
            dtype=torch.long, device=DEVICE,
        )

        cur_toks = torch.tensor(
            [[t] for t in next_toks_buf], dtype=torch.long, device=DEVICE
        )

        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(workspace, "NHD", use_tensor_cores=False)
        decode_wrapper.begin_forward(
            kv_indptr, kv_indices, kv_last_page_len,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim,
            1,              # page_size
            data_type   = DTYPE,
            q_data_type = DTYPE,
        )

        new_slots_t = torch.tensor(new_slots, dtype=torch.int64, device=DEVICE)
        ctx = DecodeKVCtx(
            wrapper   = decode_wrapper,
            k_pool    = kv_pool.k_pool,
            v_pool    = kv_pool.v_pool,
            new_slots = new_slots_t,
        )
        fb_dec = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)

        with torch.no_grad():
            bat_lg = model(cur_toks, forward_batch=fb_dec, position_ids=pos_ids)
        # [B, 1, vocab]

        decode_wrapper.end_forward()

        for i in range(B):
            slot_indices_per_req[i].append(new_slots[i])

        # ── Compare logits ─────────────────────────────────────────────────────
        paged_logits = bat_lg[:, -1, :]   # [B, vocab]
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

        # Advance buffer: next step's input = greedy from this step's reference logit.
        next_toks_buf = [int(lg.argmax()) for lg in ref_logits]

    return results


def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — Layer 7 Paged KV Cache")
    print(f" Reference: full recompute F.sdpa (kv_cache=None) per request")
    print(f" Paged:     KVPool + PrefillKVCtx + DecodeKVCtx (B={len(PROMPTS)})")
    print(f" FlashInfer: BatchDecodeWithPagedKVCacheWrapper (page_size=1, use_tensor_cores=False)")
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
