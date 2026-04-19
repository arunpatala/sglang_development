"""
verify_batch.py — Verify that Layer 6's PackedKVCache decode produces
                  logits consistent with individual PerReqKVCache decode.

What is being tested (layer6-specific):
  In layer6, prefill is identical to layer5 (B=1 per request, PerReqKVCache).
  At each decode step the scheduler batches them via PackedKVCache, which:
    1. Concatenates each request's historical KV (no padding) into one
       ragged tensor with kv_indptr marking request boundaries.
    2. Appends the new decode token's K/V for each request.
    3. Calls FlashInfer's BatchPrefillWithRaggedKVCacheWrapper (q_len=1).
    4. write_back() appends only the new token's K/V to each PerReqKVCache.

  This script checks that the packed/ragged batched logits match the
  logits from running each request individually with its own PerReqKVCache
  — confirming the ragged packing + FlashInfer + write_back round-trip
  is numerically correct.

Test method — lockstep comparison over N decode steps:
  1. Prefill all B prompts individually (B=1) → each gets a PerReqKVCache.
  2. Deep-copy KV caches so both paths start from identical state.
  3. At each step:
       Individual path: model(tok[i], mask[i], kv=ind_kv[i], pos=[kv_len_i])
       Batch path:      PackedKVCache(bat_reqs) → model(mask=None) → write_back
       Both paths are driven with the same greedy token to stay in sync.
  4. Compare logits per prompt per step.

Pass condition: max logit diff < ATOL at every step for every prompt.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16 --atol 0.75
"""

import argparse
import copy
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).parent))

from forward_batch import ForwardBatch, ForwardMode
from kv_cache import PackedKVCache, PerReqKVCache
from model import Qwen3ForCausalLM
from model.config import AttnBackend
from tokenizer import Tokenizer

ATOL = 0.75   # bfloat16 ~3 ULPs at logit scale 40

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int,   default=16)
parser.add_argument("--atol",      type=float, default=ATOL)
args = parser.parse_args()
ATOL = args.atol

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# Deliberately different lengths to stress-test the ragged indptr boundaries.
PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the difference between a compiled and interpreted language?",
]


def bar(ok: bool) -> str:
    return "✓" if ok else "✗"


def load_model_and_tok(model_path: str):
    tok = Tokenizer(model_path)
    mdl = Qwen3ForCausalLM.from_pretrained(
        model_path, dtype=DTYPE, attn_backend=AttnBackend.FLASHINFER
    )
    return mdl, tok


def run_lockstep(model, tok, n_steps: int):
    B = len(PROMPTS)

    formatted = [
        tok.apply_chat_template([{"role": "user", "content": p}])
        for p in PROMPTS
    ]

    # Tokenise individually — no padding (exactly what model_runner.prefill does)
    ind_ids_list, ind_mask_list = [], []
    for text in formatted:
        enc = tok._tok([text], return_tensors="pt", padding=False)
        ind_ids_list.append(enc["input_ids"].to(DEVICE))
        ind_mask_list.append(enc["attention_mask"].to(DEVICE))

    prompt_lens = [m.shape[1] for m in ind_mask_list]
    max_len     = max(prompt_lens)

    print(f"  Prompt lengths : {prompt_lens}  (max={max_len})")

    # ── Prefill: B=1 per request (identical for both paths) ───────────────
    ind_kvs    = []
    ind_masks  = []
    ind_logits = []

    for i in range(B):
        kv  = PerReqKVCache()
        pos = torch.arange(prompt_lens[i], device=DEVICE).unsqueeze(0)
        fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=kv,
                           attention_mask=ind_mask_list[i])
        with torch.no_grad():
            logits = model(ind_ids_list[i], forward_batch=fb, position_ids=pos)
        ind_kvs.append(kv)
        ind_masks.append(ind_mask_list[i].clone())
        ind_logits.append(logits[0, -1, :])

    # Deep-copy KV caches so both paths start from exactly the same state.
    bat_kvs  = [copy.deepcopy(kv) for kv in ind_kvs]
    bat_reqs = [SimpleNamespace(kv_cache=kv) for kv in bat_kvs]

    # Pre-allocate workspace once (reused every decode step, same as ModelRunner).
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    cfg       = model.model.config

    results = [[] for _ in range(B)]

    # ── Decode steps (lockstep) ────────────────────────────────────────────
    for step in range(n_steps):
        # Greedy token from the individual path drives both paths identically.
        next_toks = [int(ind_logits[i].argmax()) for i in range(B)]

        # ── Individual B=1 forward ─────────────────────────────────────────
        ind_step_logits = []
        for i in range(B):
            cur   = torch.tensor([[next_toks[i]]], device=DEVICE)
            ind_masks[i] = torch.cat(
                [ind_masks[i], torch.ones(1, 1, dtype=torch.long, device=DEVICE)], dim=1
            )
            kv_len = ind_kvs[i].get_seq_length()
            pos    = torch.tensor([[kv_len]], dtype=torch.long, device=DEVICE)
            fb_ind = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ind_kvs[i],
                                  attention_mask=ind_masks[i])
            with torch.no_grad():
                lg = model(cur, forward_batch=fb_ind, position_ids=pos)
            ind_step_logits.append(lg[0, -1, :])

        # ── Batch decode via PackedKVCache + FlashInfer ────────────────────
        kv_lens = [req.kv_cache.get_seq_length() for req in bat_reqs]
        bat_cur = torch.tensor([[t] for t in next_toks], dtype=torch.long, device=DEVICE)  # [B,1]

        # Per-request position IDs: each request is at its own next position.
        pos_ids = torch.tensor(
            [[kv_len] for kv_len in kv_lens], dtype=torch.long, device=DEVICE
        )

        # PackedKVCache: no padding mask needed — FlashInfer uses kv_indptr.
        pack_kv = PackedKVCache(bat_reqs, workspace)
        pack_kv.plan(cfg.num_attention_heads, cfg.num_key_value_heads,
                     cfg.head_dim, DTYPE)

        fb_bat = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=pack_kv,
                              attention_mask=None)
        with torch.no_grad():
            bat_lg = model(bat_cur, forward_batch=fb_bat, position_ids=pos_ids)

        pack_kv.write_back()
        pack_kv.end_forward()

        bat_step_logits = bat_lg[:, -1, :]   # [B, vocab]

        # ── Compare ────────────────────────────────────────────────────────
        for i in range(B):
            il   = ind_step_logits[i]
            bl   = bat_step_logits[i]
            diff = (il - bl).abs().max().item()
            results[i].append((int(il.argmax()) == int(bl.argmax()), diff,
                               int(il.argmax()), int(bl.argmax())))

        ind_logits = ind_step_logits   # advance individual path

    return results


def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — Layer 6 PackedKVCache decode")
    print(f" B=4 (PackedKVCache+FlashInfer) vs 4×B=1 (PerReqKVCache+F.sdpa)")
    print(f" Steps  : {n}")
    print(f" Model  : {args.model}")
    print(f" Atol   : {ATOL}  (bfloat16 ~3 ULPs at logit scale 40)")
    print("=" * 64)
    print()

    print("Loading tokenizer + model …")
    model, tok = load_model_and_tok(args.model)
    print()

    results = run_lockstep(model, tok, n_steps=n)
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
            ind_t = repr(tok._tok.decode([r[2]]))
            bat_t = repr(tok._tok.decode([r[3]]))
            label = "(near-tie)" if r[1] < ATOL else "(BUG)"
            print(f"       step {s:2d}: ind→{ind_t:12s}  bat→{bat_t:12s}  "
                  f"diff={r[1]:.4f}  {label}")
        print()

    print("=" * 64)
    print(f" RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 64)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
