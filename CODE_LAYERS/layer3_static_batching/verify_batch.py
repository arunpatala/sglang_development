"""
verify_batch.py — Check that B=4 batch logits match 4 × B=1 logits,
                  including with DIFFERENT prompt lengths.

Why different-length prompts are hard:
  Left-padding a batch creates padding tokens on the left of shorter prompts.
  If no position_ids are passed, HuggingFace assigns sequential positions
  0..max_len-1 to ALL batch elements — so a short prompt (length L) padded
  to max_len M gets its real tokens at positions M-L..M-1 instead of 0..L-1.
  This changes the RoPE encoding and breaks token match vs individual run.

The fix — compute per-example position_ids from attention_mask:
  prefill:
    position_ids = attention_mask.cumsum(-1) - 1     # 0-indexed per real token
    position_ids = position_ids.clamp(min=0)         # padding → 0 (masked anyway)
  decode step k:
    position_ids = (prompt_lens + k).unsqueeze(1)    # [B, 1], per-example

  With these position_ids, RoPE encoding is identical to the individual B=1
  runs → logits should match within bfloat16 tolerance.

Pass condition:  max logit diff < ATOL at every step for every prompt.

Usage:
    python verify_batch.py
    python verify_batch.py --n-compare 16
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── imports from this layer ────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))
from kv_cache import KVCache

# ── CLI ────────────────────────────────────────────────────────────────────
ATOL = 0.75   # bfloat16 ~3 ULPs at logit scale 40

parser = argparse.ArgumentParser()
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int, default=16)
parser.add_argument("--atol",      type=float, default=ATOL)
args = parser.parse_args()
ATOL = args.atol

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# Deliberately different token lengths to exercise left-padding.
PROMPTS = [
    "What is 2+2?",
    "What is the capital of France?",
    "Explain what a neural network is.",
    "What is the difference between a compiled and interpreted language?",
]

def bar(ok: bool) -> str:
    return "✓" if ok else "✗"

# ── Helpers ────────────────────────────────────────────────────────────────

def load_model_and_tok(model_path: str):
    tok = AutoTokenizer.from_pretrained(model_path)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    mdl = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE)
        .to(DEVICE).eval()
    )
    return mdl, tok


def format_prompts(tok, prompts: list[str]) -> list[str]:
    return [
        tok.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        for p in prompts
    ]


def tokenize_single(tok, text: str):
    enc = tok([text], return_tensors="pt", padding=False)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def tokenize_batch(tok, texts: list[str]):
    enc = tok(texts, return_tensors="pt", padding=True)
    return enc["input_ids"].to(DEVICE), enc["attention_mask"].to(DEVICE)


def prefill_position_ids(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Per-example position_ids for a left-padded batch.
    Real tokens are numbered 0, 1, ... within each sequence.
    Padding positions get 0 (they are masked in attention, so value doesn't matter).

    Example for [PAD, PAD, t0, t1, t2]:
        attention_mask = [0, 0, 1, 1, 1]
        cumsum         = [0, 0, 1, 2, 3]
        cumsum - 1     = [-1,-1, 0, 1, 2]
        clamp(min=0)   = [ 0, 0, 0, 1, 2]   ← same as individual run
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

# ── Lockstep comparison ────────────────────────────────────────────────────

def run_lockstep(model, tok, n_steps: int):
    """
    For each of the B=len(PROMPTS) prompts:
      - Individual B=1 run: no padding, sequential position_ids
      - Batch B=4 run: left-padded, per-example position_ids (fixed)

    Both receive the SAME token at every decode step (B=1 greedy choice).
    Returns per-prompt list of (tok_match, logit_diff, ind_tok, bat_tok).
    """
    B = len(PROMPTS)
    formatted = format_prompts(tok, PROMPTS)

    # Tokenise individually (B=1, no padding)
    ind_ids_list, ind_mask_list = [], []
    for text in formatted:
        ids, mask = tokenize_single(tok, text)
        ind_ids_list.append(ids)
        ind_mask_list.append(mask)

    prompt_lens = [m.sum().item() for m in ind_mask_list]

    # Tokenise as batch (B=4, left-padded)
    bat_ids, bat_mask = tokenize_batch(tok, formatted)
    max_len = bat_ids.shape[1]

    print(f"  Prompt lengths : {prompt_lens}  (max={max_len})")
    print(f"  Padding added  : {[max_len - l for l in prompt_lens]}")

    # ── Prefill ───────────────────────────────────────────────────────
    ind_pasts, ind_masks_cur, ind_last_logits = [], [], []

    for i in range(B):
        kv = KVCache()
        with torch.no_grad():
            out = model(
                ind_ids_list[i],
                attention_mask=ind_mask_list[i],
                past_key_values=kv,
                use_cache=True,
            )
        ind_pasts.append(out.past_key_values)
        ind_masks_cur.append(ind_mask_list[i])
        ind_last_logits.append(out.logits[0, -1, :])

    # Batch prefill — with per-example position_ids
    bat_pos = prefill_position_ids(bat_mask)  # [B, max_len]
    bat_kv  = KVCache()
    with torch.no_grad():
        bat_out = model(
            bat_ids,
            attention_mask=bat_mask,
            position_ids=bat_pos,
            past_key_values=bat_kv,
            use_cache=True,
        )
    bat_past        = bat_out.past_key_values
    bat_mask_cur    = bat_mask
    bat_last_logits = bat_out.logits[:, -1, :]  # [B, vocab]

    results = [[] for _ in range(B)]

    # Step 0: prefill logit comparison
    for i in range(B):
        il = ind_last_logits[i]
        bl = bat_last_logits[i]
        diff = (il - bl).abs().max().item()
        results[i].append((int(il.argmax()) == int(bl.argmax()), diff,
                           int(il.argmax()), int(bl.argmax())))

    # ── Decode steps 1..n_steps-1 ─────────────────────────────────────
    for step in range(1, n_steps):
        # B=1 greedy token for each prompt (drives both models)
        next_toks = [int(ind_last_logits[i].argmax()) for i in range(B)]

        # Individual B=1 forwards
        for i in range(B):
            cur = torch.tensor([[next_toks[i]]], device=DEVICE)
            ind_masks_cur[i] = torch.cat(
                [ind_masks_cur[i], torch.ones(1, 1, dtype=torch.long, device=DEVICE)],
                dim=1,
            )
            with torch.no_grad():
                out = model(
                    cur,
                    attention_mask=ind_masks_cur[i],
                    past_key_values=ind_pasts[i],
                    use_cache=True,
                )
            ind_pasts[i]        = out.past_key_values
            ind_last_logits[i]  = out.logits[0, -1, :]

        # Batch B=4 forward — per-example decode position_ids
        bat_cur = torch.tensor([[t] for t in next_toks], device=DEVICE)  # [B, 1]
        bat_mask_cur = torch.cat(
            [bat_mask_cur, torch.ones(B, 1, dtype=torch.long, device=DEVICE)],
            dim=1,
        )
        # Each example i is at position: prompt_len_i + (step - 1)
        # step-1 because step 0 was the prefill's last token
        bat_pos_dec = torch.tensor(
            [[l + step - 1] for l in prompt_lens],
            dtype=torch.long, device=DEVICE,
        )  # [B, 1]
        with torch.no_grad():
            bat_out = model(
                bat_cur,
                attention_mask=bat_mask_cur,
                position_ids=bat_pos_dec,
                past_key_values=bat_past,
                use_cache=True,
            )
        bat_past        = bat_out.past_key_values
        bat_last_logits = bat_out.logits[:, -1, :]

        for i in range(B):
            il   = ind_last_logits[i]
            bl   = bat_last_logits[i]
            diff = (il - bl).abs().max().item()
            results[i].append((int(il.argmax()) == int(bl.argmax()), diff,
                               int(il.argmax()), int(bl.argmax())))

    return results

# ── Main ───────────────────────────────────────────────────────────────────

def main():
    n = args.n_compare
    print("=" * 64)
    print(f" verify_batch.py — B=4 (different lengths) vs 4×B=1  ({n} steps)")
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
        steps       = results[i]
        n_tok_match = sum(r[0] for r in steps)
        avg_diff    = sum(r[1] for r in steps) / len(steps)
        max_diff    = max(r[1] for r in steps)
        logits_ok   = all(r[1] < ATOL for r in steps)
        pass_i      = logits_ok

        print(f"  [{i}] '{prompt}'")
        print(f"       Tokens   : {n_tok_match}/{len(steps)} exact  {bar(n_tok_match == len(steps))}")
        print(f"       Avg diff : {avg_diff:.4f}   Max diff : {max_diff:.4f}  {bar(logits_ok)}")
        print(f"       Result   : {'PASS ✓' if pass_i else 'FAIL ✗'}")

        if not logits_ok:
            overall_pass = False
            bad = [(s, f"{r[1]:.4f}") for s, r in enumerate(steps) if r[1] >= ATOL]
            print(f"       Steps > atol : {bad}")

        mismatches = [(s, r) for s, r in enumerate(steps) if not r[0]]
        for s, r in mismatches:
            ind_t = repr(tok.decode([r[2]]))
            bat_t = repr(tok.decode([r[3]]))
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
