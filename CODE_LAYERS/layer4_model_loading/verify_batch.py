"""
verify_batch.py — Check that our Qwen3ForCausalLM produces consistent
                  logits for B=4 batched inference vs 4×B=1 individual runs,
                  with different-length prompts (left-padding stress test).

Pass condition: max logit diff < ATOL at every step for every prompt.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16 --atol 0.75
"""

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache import KVCache
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer

ATOL = 0.75   # bfloat16 ~3 ULPs at logit scale 40

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int, default=16)
parser.add_argument("--atol",      type=float, default=ATOL)
args = parser.parse_args()
ATOL = args.atol

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# Deliberately different token lengths to stress-test left-padding + position_ids.
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
    mdl = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)
    return mdl, tok


def prefill_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """Per-example position_ids for a left-padded batch."""
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def run_lockstep(model, tok, n_steps: int):
    B = len(PROMPTS)

    formatted = [
        tok.apply_chat_template([{"role": "user", "content": p}])
        for p in PROMPTS
    ]

    # Tokenise individually (B=1, no padding)
    ind_ids_list, ind_mask_list = [], []
    for text in formatted:
        enc = tok._tok([text], return_tensors="pt", padding=False)
        ind_ids_list.append(enc["input_ids"].to(DEVICE))
        ind_mask_list.append(enc["attention_mask"].to(DEVICE))

    prompt_lens = torch.tensor(
        [m.sum().item() for m in ind_mask_list], dtype=torch.long, device=DEVICE
    )

    # Tokenise as batch (B=4, left-padded)
    enc = tok._tok(formatted, return_tensors="pt", padding=True)
    bat_ids  = enc["input_ids"].to(DEVICE)
    bat_mask = enc["attention_mask"].to(DEVICE)
    max_len  = bat_ids.shape[1]

    print(f"  Prompt lengths : {prompt_lens.tolist()}  (max={max_len})")
    print(f"  Padding added  : {(max_len - prompt_lens).tolist()}")

    # ── Prefill: individual B=1 runs ──────────────────────────────────
    ind_kvs, ind_masks, ind_logits = [], [], []
    for i in range(B):
        kv = KVCache()
        with torch.no_grad():
            logits, _ = model(ind_ids_list[i],
                              attention_mask=ind_mask_list[i],
                              past_key_values=kv)
        ind_kvs.append(kv)
        ind_masks.append(ind_mask_list[i])
        ind_logits.append(logits[0, -1, :])

    # ── Prefill: batch B=4 with per-example position_ids ─────────────
    bat_pos = prefill_position_ids(bat_mask)
    bat_kv  = KVCache()
    with torch.no_grad():
        bat_logits_full, _ = model(
            bat_ids,
            attention_mask=bat_mask,
            past_key_values=bat_kv,
            position_ids=bat_pos,
        )
    bat_logits = bat_logits_full[:, -1, :]   # [B, vocab]

    results = [[] for _ in range(B)]

    # Step 0: prefill logit comparison
    for i in range(B):
        il, bl = ind_logits[i], bat_logits[i]
        diff = (il - bl).abs().max().item()
        results[i].append((int(il.argmax()) == int(bl.argmax()), diff,
                           int(il.argmax()), int(bl.argmax())))

    # ── Decode: lockstep steps 1..n_steps-1 ──────────────────────────
    for step in range(1, n_steps):
        # Drive both models with the greedy token from the B=1 run
        next_toks = [int(ind_logits[i].argmax()) for i in range(B)]

        # Individual B=1 forwards
        for i in range(B):
            cur = torch.tensor([[next_toks[i]]], device=DEVICE)
            ind_masks[i] = torch.cat(
                [ind_masks[i], torch.ones(1, 1, dtype=torch.long, device=DEVICE)], dim=1
            )
            with torch.no_grad():
                lg, _ = model(cur,
                              attention_mask=ind_masks[i],
                              past_key_values=ind_kvs[i])
            ind_logits[i] = lg[0, -1, :]

        # Batch B=4 forward
        bat_cur = torch.tensor([[t] for t in next_toks], device=DEVICE)   # [B, 1]
        bat_mask = torch.cat(
            [bat_mask, torch.ones(B, 1, dtype=torch.long, device=DEVICE)], dim=1
        )
        bat_pos_dec = (prompt_lens + step - 1).unsqueeze(1)   # [B, 1]
        with torch.no_grad():
            bat_lg, _ = model(
                bat_cur,
                attention_mask=bat_mask,
                past_key_values=bat_kv,
                position_ids=bat_pos_dec,
            )
        bat_logits = bat_lg[:, -1, :]

        for i in range(B):
            il, bl = ind_logits[i], bat_logits[i]
            diff = (il - bl).abs().max().item()
            results[i].append((int(il.argmax()) == int(bl.argmax()), diff,
                               int(il.argmax()), int(bl.argmax())))

    return results


def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — Layer 4A model loading")
    print(f" B=4 (different lengths) vs 4×B=1  ({n} steps)")
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
            print(f"       step {s:2d}: B=1→{ind_t:12s}  B=4→{bat_t:12s}  "
                  f"diff={r[1]:.4f}  {label}")
        print()

    print("=" * 64)
    print(f" RESULT: {'PASS ✓' if overall_pass else 'FAIL ✗'}")
    print("=" * 64)
    sys.exit(0 if overall_pass else 1)


if __name__ == "__main__":
    main()
