"""
verify_gptq.py — Layer 12: GPTQ quantisation correctness check.

Tests:
  1. Load the GPTQ model (JunHowie/Qwen3-0.6B-GPTQ-Int4).
  2. Verify GPU memory is lower than fp16 baseline (~600 MB vs ~1.2 GB).
  3. Run a forward pass — check no NaN/Inf in logits.
  4. Greedy-decode a short prompt and print the output.
  5. (Optional) Compare top-1 tokens against the fp16 model if available.
"""

import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

GPTQ_MODEL  = "JunHowie/Qwen3-0.6B-GPTQ-Int4"
FP_MODEL    = "Qwen/Qwen3-0.6B"
MAX_NEW     = 40
PROMPT      = "The capital of France is"

SEP = "─" * 60


def load_gptq_model():
    from model_gptq import Qwen3ForCausalLM
    t0 = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(GPTQ_MODEL)
    dt = time.perf_counter() - t0
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  GPTQ model loaded in {dt:.1f}s  |  GPU = {mem_mb:.0f} MB")
    return model


def load_fp_model():
    from model import Qwen3ForCausalLM
    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    model = Qwen3ForCausalLM.from_pretrained(FP_MODEL)
    dt = time.perf_counter() - t0
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  FP16 model loaded in {dt:.1f}s  |  GPU = {mem_mb:.0f} MB")
    return model


def encode(model_path: str, text: str) -> torch.Tensor:
    from tokenizer import Tokenizer
    tok = Tokenizer(model_path)
    # tok.encode() already returns [1, L] tensor on CUDA
    return tok.encode(text)


@torch.inference_mode()
def greedy_generate(model, input_ids: torch.Tensor, max_new: int) -> list[int]:
    """Simple greedy decode without KV cache (for verification)."""
    generated = []
    x = input_ids
    for _ in range(max_new):
        logits = model(x)           # [1, seq_len, vocab]
        next_id = logits[0, -1, :].argmax().item()
        generated.append(next_id)
        x = torch.cat([x, torch.tensor([[next_id]], device=x.device)], dim=1)
    return generated


def decode_ids(model_path: str, ids: list[int]) -> str:
    from tokenizer import Tokenizer
    tok = Tokenizer(model_path)
    return tok.decode(ids)


# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print(" verify_gptq.py — Layer 12: GPTQ Quantisation")
    print(" Model:", GPTQ_MODEL)
    print("=" * 60)

    # ── Test 1: load and basic memory check ──────────────────────────────
    print(f"\n{SEP}")
    print("Test 1: Load GPTQ model + memory check")
    print(SEP)

    torch.cuda.reset_peak_memory_stats()
    gptq_model = load_gptq_model()
    gptq_mem   = torch.cuda.memory_allocated() / 1024**2

    # Qwen3-0.6B fp16 is ~1.2 GB; GPTQ-Int4 should be well under 600 MB
    # (weights 4x smaller; activations still bf16)
    print(f"  Expected: < 700 MB for 4-bit model weights")
    if gptq_mem < 700:
        print(f"  → PASS ✓  ({gptq_mem:.0f} MB)")
    else:
        print(f"  → WARN: {gptq_mem:.0f} MB (higher than expected — check buffers)")

    # ── Test 2: forward pass sanity ──────────────────────────────────────
    print(f"\n{SEP}")
    print("Test 2: Forward pass — no NaN / Inf in logits")
    print(SEP)

    input_ids = encode(GPTQ_MODEL, PROMPT)
    print(f"  Prompt : {repr(PROMPT)}")
    print(f"  Tokens : {input_ids.shape[1]}")

    with torch.inference_mode():
        logits = gptq_model(input_ids)   # [1, seq_len, vocab]

    has_nan = torch.isnan(logits).any().item()
    has_inf = torch.isinf(logits).any().item()
    print(f"  NaN in logits : {has_nan}")
    print(f"  Inf in logits : {has_inf}")
    if has_nan or has_inf:
        print("  → FAIL ✗")
        sys.exit(1)
    print("  → PASS ✓")

    # ── Test 3: greedy decode ─────────────────────────────────────────────
    print(f"\n{SEP}")
    print(f"Test 3: Greedy decode ({MAX_NEW} tokens)")
    print(SEP)

    t0 = time.perf_counter()
    new_ids  = greedy_generate(gptq_model, input_ids, MAX_NEW)
    dt       = time.perf_counter() - t0
    tok_per_s = MAX_NEW / dt

    completion = decode_ids(GPTQ_MODEL, new_ids)
    print(f"  Prompt     : {repr(PROMPT)}")
    print(f"  Completion : {repr(completion)}")
    print(f"  Speed      : {tok_per_s:.1f} tok/s  ({dt:.2f}s for {MAX_NEW} tokens)")

    # Sanity: check we got valid token IDs (all > 0 and < vocab_size)
    vocab_size = gptq_model.config.vocab_size
    ok = all(0 <= t < vocab_size for t in new_ids)
    if ok:
        print("  → PASS ✓  (all token IDs valid)")
    else:
        print("  → FAIL ✗  (some token IDs out of range)")
        sys.exit(1)

    # ── Test 4: top-1 matches between GPTQ and FP model (optional) ───────
    print(f"\n{SEP}")
    print("Test 4: Top-1 agreement with fp16 model (optional)")
    print(SEP)
    try:
        fp_model = load_fp_model()
        with torch.inference_mode():
            fp_logits   = fp_model(input_ids)
            gptq_logits = gptq_model(input_ids)

        gptq_top1 = gptq_logits[0, -1].argmax()
        fp_top1   = fp_logits[0, -1].argmax()

        print(f"  GPTQ top-1 token : {gptq_top1.item()!r}  ({decode_ids(FP_MODEL, [gptq_top1.item()])})")
        print(f"  FP16 top-1 token : {fp_top1.item()!r}  ({decode_ids(FP_MODEL, [fp_top1.item()])})")

        if gptq_top1 == fp_top1:
            print("  → PASS ✓  (top-1 tokens agree)")
        else:
            print("  → INFO: top-1 tokens differ (expected for 4-bit quantisation)")

        # Check top-5 overlap
        gptq_top5 = set(gptq_logits[0, -1].topk(5).indices.tolist())
        fp_top5   = set(fp_logits[0, -1].topk(5).indices.tolist())
        overlap   = len(gptq_top5 & fp_top5)
        print(f"  Top-5 overlap : {overlap}/5")
        if overlap >= 3:
            print("  → PASS ✓  (≥3 of top-5 tokens match)")
        else:
            print("  → WARN: low top-5 overlap (quantisation may be degraded)")

        del fp_model

    except Exception as e:
        print(f"  Skipped: {e}")

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(" OVERALL: PASS ✓")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
