"""
verify.py — Correctness check for our Qwen3ForCausalLM implementation.

Runs both models in lockstep for N_COMPARE decode steps:
  - Both receive the SAME token as input at every step (HF's greedy choice).
  - Compares logits and decoded token at each step.

Pass conditions (both must hold):
  1. All N_COMPARE tokens match exactly.
  2. Max logit diff < ATOL at every step.

Why 16 tokens?
  16 steps = very little accumulated bfloat16 error, so we can require
  exact token match with a tight tolerance (~0.5).  At 84+ steps near-ties
  start causing argmax differences that look like bugs but aren't.

Usage:
    python verify.py
    python verify.py --model Qwen/Qwen3-0.6B
    python verify.py --prompt "Explain Newton's laws"
    python verify.py --n-compare 32
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent))
from kv_cache import KVCache
from model import Qwen3ForCausalLM

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_COMPARE = 16      # tokens to compare
ATOL      = 0.5    # max acceptable logit diff (bfloat16: ~2 ULPs at logit scale 40)

parser = argparse.ArgumentParser(description="Layer 4 correctness verification")
parser.add_argument("--model",     default="Qwen/Qwen3-0.6B")
parser.add_argument("--n-compare", type=int, default=N_COMPARE)
parser.add_argument(
    "--prompt",
    default="Explain the difference between a process and a thread in one paragraph.",
)
args = parser.parse_args()

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def format_prompt(tokenizer, prompt: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

def bar(ok: bool) -> str:
    return "✓" if ok else "✗"

# ---------------------------------------------------------------------------
# Lockstep comparison
# ---------------------------------------------------------------------------

def run_lockstep(model_path: str, input_ids: torch.Tensor, n_steps: int):
    """
    Load both models, run prefill on the same input_ids, then decode
    n_steps tokens in lockstep.

    At each step:
      - Feed HF's greedy token to BOTH models (ensures KV caches stay in sync).
      - Record max-abs logit diff and whether top-1 tokens match.

    Returns: list of (step, hf_tok, our_tok, logit_diff, tokens_match)
    """
    print("Loading HuggingFace reference model …")
    hf_model = (
        AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=DTYPE)
        .to(DEVICE)
        .eval()
    )
    print("Loading our Qwen3ForCausalLM …")
    our_model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)

    B, prompt_len = input_ids.shape
    attn_mask = torch.ones(B, prompt_len, dtype=torch.long, device=DEVICE)
    kv = KVCache()

    # ── Prefill ───────────────────────────────────────────────────────
    with torch.no_grad():
        hf_out     = hf_model(input_ids, use_cache=True)
        our_logits = our_model(input_ids, attention_mask=attn_mask, kv_cache=kv)

    hf_past = hf_out.past_key_values

    results = []

    # Step 0: logits from the prefill pass (predict first new token)
    hf_logits_step = hf_out.logits[0, -1, :]      # [vocab]
    our_logits_step = our_logits[0, -1, :]         # [vocab]

    hf_tok  = int(hf_logits_step.argmax())
    our_tok = int(our_logits_step.argmax())
    diff    = (hf_logits_step - our_logits_step).abs().max().item()
    results.append((0, hf_tok, our_tok, diff, hf_tok == our_tok))

    next_tok = hf_tok  # always advance with HF's token to keep caches in sync

    # ── Decode steps 1..n_steps-1 ─────────────────────────────────────
    for step in range(1, n_steps):
        current  = torch.tensor([[next_tok]], device=DEVICE)
        attn_mask = torch.cat(
            [attn_mask, torch.ones(B, 1, dtype=torch.long, device=DEVICE)],
            dim=1,
        )
        with torch.no_grad():
            hf_out     = hf_model(current, past_key_values=hf_past, use_cache=True)
            our_logits = our_model(current, attention_mask=attn_mask, kv_cache=kv)

        hf_past         = hf_out.past_key_values
        hf_logits_step  = hf_out.logits[0, -1, :]
        our_logits_step = our_logits[0, -1, :]

        hf_tok  = int(hf_logits_step.argmax())
        our_tok = int(our_logits_step.argmax())
        diff    = (hf_logits_step - our_logits_step).abs().max().item()
        results.append((step, hf_tok, our_tok, diff, hf_tok == our_tok))

        next_tok = hf_tok

        if next_tok == 151645:  # Qwen3 eos
            break

    del hf_model, our_model
    torch.cuda.empty_cache()
    return results

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    n_compare = args.n_compare

    print("=" * 64)
    print(f" Layer 4 — Correctness Verification  ({n_compare} tokens)")
    print(f" Model  : {args.model}")
    print(f" Prompt : {args.prompt[:58]}…")
    print(f" Atol   : {ATOL}  (bfloat16 ~2 ULPs at logit scale 40)")
    print("=" * 64)

    print("\nLoading tokenizer …")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    formatted = format_prompt(tokenizer, args.prompt)
    input_ids = tokenizer(formatted, return_tensors="pt")["input_ids"].to(DEVICE)
    print(f"Prompt tokens: {input_ids.shape[1]}")
    print()

    results = run_lockstep(args.model, input_ids, n_steps=n_compare)

    # ── Per-step table ────────────────────────────────────────────────
    print()
    print(f"  {'Step':>4}  {'HF tok':>8}  {'Our tok':>8}  {'Diff':>8}  {'Match':>5}  Token text")
    print(f"  {'─'*4}  {'─'*8}  {'─'*8}  {'─'*8}  {'─'*5}  {'─'*20}")

    for step, hf_tok, our_tok, diff, match in results:
        token_text = repr(tokenizer.decode([hf_tok]))
        diff_flag  = f"{diff:.4f}" + ("  !" if diff >= ATOL else "   ")
        print(
            f"  {step:>4}  {hf_tok:>8}  {our_tok:>8}  {diff_flag}  "
            f"  {bar(match)}    {token_text}"
        )

    # ── Summary ───────────────────────────────────────────────────────
    n_steps       = len(results)
    tokens_ok     = all(r[4] for r in results)
    logits_ok     = all(r[3] < ATOL for r in results)
    avg_diff      = sum(r[3] for r in results) / n_steps
    max_diff      = max(r[3] for r in results)
    n_tok_match   = sum(r[4] for r in results)

    print()
    print(f"  Steps     : {n_steps}")
    print(f"  Tokens    : {n_tok_match}/{n_steps} exact match  {bar(tokens_ok)}")
    print(f"  Avg diff  : {avg_diff:.4f}")
    print(f"  Max diff  : {max_diff:.4f}  (atol={ATOL})  {bar(logits_ok)}")

    overall = tokens_ok and logits_ok
    print()
    print("=" * 64)
    print(f" RESULT: {'PASS ✓' if overall else 'FAIL ✗'}")
    print("=" * 64)

    if not tokens_ok:
        first_bad = next(s for s, _, _, _, ok in results if not ok)
        print(f"\n First token mismatch at step {first_bad} — this is a bug.")
    if not logits_ok:
        bad = [(s, d) for s, _, _, d, _ in results if d >= ATOL]
        print(f"\n Steps exceeding atol={ATOL}: {bad}")

    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
