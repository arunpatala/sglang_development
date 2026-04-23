"""
verify_lora.py — Verify the Layer 20 single-adapter LoRA implementation.

Four tests, ordered from prerequisite to full correctness:

  Test 0 — Base model match (prerequisite)
    Our custom Qwen3ForCausalLM (NOCACHE path) must match HuggingFace
    AutoModelForCausalLM on the same prompt before we trust any LoRA comparison.

  Test 1 — Adapter changes output
    Run the same prompt with lora_mask=1.0 vs lora_mask=0.0.
    PASS if the logits differ (adapter is actually applied).

  Test 2 — Our LoRA matches PEFT (core correctness)
    Compare our NOCACHE+LoRA output against HuggingFace PeftModel.
    This is the ground-truth check: same math, same weights → same numbers.

  Test 3 — Mixed-batch mask separation (CUDA + flashinfer required)
    Run a batched EXTEND with [lora, base] requests.
    PASS if the lora token matches the LoRA reference and the base token
    matches the base reference — proves the mask correctly gates the delta.

  Test 4 — Paged LoRA prefill + decode (CUDA + flashinfer required)
    Run the full paged path with LoRA. Each step compared to NOCACHE+LoRA
    reference (same as verify_batch.py does for the base model).

Usage:
    python verify_lora.py
    python verify_lora.py --model Qwen/Qwen3-0.6B --lora phh/Qwen3-0.6B-TLDR-Lora
    python verify_lora.py --atol 0.5
    python verify_lora.py --device cpu    # force CPU (slow, for debugging)
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

# ── path setup ────────────────────────────────────────────────────────────────
LAYER_ROOT = Path(__file__).parent
sys.path.insert(0, str(LAYER_ROOT))

# ── args ──────────────────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="verify_lora.py — LoRA correctness tests")
parser.add_argument("--model",    default="Qwen/Qwen3-0.6B",            help="Base model path or HF repo")
parser.add_argument("--lora",     default="phh/Qwen3-0.6B-TLDR-Lora",  help="LoRA adapter path or HF repo")
parser.add_argument("--atol",     type=float, default=1.0,              help="Max-abs-diff tolerance (default 1.0)")
parser.add_argument("--n-decode", type=int,   default=4,                help="Decode steps for paged test")
parser.add_argument("--page-size",type=int,   default=16,               help="KV pool page size")
parser.add_argument("--device",   default=None,                         help="Force device: cuda/mps/cpu")
args = parser.parse_args()

ATOL      = args.atol
PAGE_SIZE = args.page_size

# ── device detection ──────────────────────────────────────────────────────────
if args.device:
    DEVICE = args.device
elif torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

DTYPE = torch.bfloat16

# ── flashinfer availability ───────────────────────────────────────────────────
try:
    import flashinfer
    HAS_FLASHINFER = True
except ImportError:
    HAS_FLASHINFER = False

PAGED_AVAILABLE = HAS_FLASHINFER and DEVICE == "cuda"

# ── our imports ───────────────────────────────────────────────────────────────
from lora import LoRAAdapter
from model.config import Qwen3Config
from model.qwen3 import Qwen3ForCausalLM
from tokenizer import Tokenizer

if PAGED_AVAILABLE:
    from kv_cache import (
        DecodeKVCtx, ExtendKVCtx, KVPool, ReqToTokenPool,
        WriteInfo, compute_write_info,
    )
    from triton_utils import create_flashinfer_kv_indices_triton


# ─────────────────────────────────────────────────────────────────────────────
# Model loading helpers (device-flexible; avoids from_pretrained's .to("cuda"))
# ─────────────────────────────────────────────────────────────────────────────

def _resolve_model_path(model_path: str) -> Path:
    """Return a local Path to a model directory, downloading from HF if needed."""
    p = Path(model_path)
    if p.is_dir() and (p / "config.json").exists():
        return p
    offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    from huggingface_hub import snapshot_download
    return Path(snapshot_download(model_path, local_files_only=offline))


def load_our_model(model_path: str, device: str, dtype: torch.dtype) -> Qwen3ForCausalLM:
    """
    Load our custom model onto `device` without relying on from_pretrained()'s
    hardcoded `.to("cuda")`.
    """
    from safetensors import safe_open

    model_dir = _resolve_model_path(model_path)
    config    = Qwen3Config.from_json(model_dir / "config.json")
    model     = Qwen3ForCausalLM(config).to(dtype)

    weights_path = model_dir / "model.safetensors"
    def _iter():
        with safe_open(str(weights_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                yield key, f.get_tensor(key).to(dtype)

    model.load_weights(_iter())
    return model.to(device).eval()


def load_hf_base_model(model_path: str, device: str, dtype: torch.dtype):
    """Load HuggingFace AutoModelForCausalLM (base, no LoRA)."""
    from transformers import AutoModelForCausalLM
    model_dir = str(_resolve_model_path(model_path))
    m = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dtype)
    return m.to(device).eval()


def load_peft_model(model_path: str, lora_path: str, device: str, dtype: torch.dtype):
    """Load HuggingFace base model wrapped with PEFT LoRA."""
    from peft import PeftModel
    base = load_hf_base_model(model_path, device, dtype)
    lora_dir = str(_resolve_model_path(lora_path))
    peft_model = PeftModel.from_pretrained(base, lora_dir, torch_dtype=dtype)
    return peft_model.to(device).eval()


# ─────────────────────────────────────────────────────────────────────────────
# Reference helpers
# ─────────────────────────────────────────────────────────────────────────────

def our_nocache_logit(
    model: Qwen3ForCausalLM,
    token_ids: List[int],
    adapter: Optional[LoRAAdapter] = None,
) -> torch.Tensor:
    """
    Run our model in NOCACHE mode (F.sdpa, no flashinfer).
    Returns logit vector at the last token position [vocab].
    """
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    pos_t = torch.arange(len(token_ids), device=DEVICE).unsqueeze(0)
    mask  = torch.ones(1, len(token_ids), dtype=torch.long, device=DEVICE)

    lora_mask = None
    if adapter is not None:
        lora_mask = torch.ones(1, len(token_ids), 1, dtype=DTYPE, device=DEVICE)

    with torch.no_grad():
        lg = model(
            ids_t,
            attention_mask = mask,
            kv_cache       = None,
            position_ids   = pos_t,
            lora_mask      = lora_mask,
            lora_adapter   = adapter,
        )
    return lg[0, -1, :]


def peft_logit(peft_model, token_ids: List[int]) -> torch.Tensor:
    """Run PEFT model (HF + adapter) and return last-token logit [vocab]."""
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = peft_model(input_ids=ids_t)
    return out.logits[0, -1, :]


def hf_base_logit(hf_model, token_ids: List[int]) -> torch.Tensor:
    """Run HuggingFace base model and return last-token logit [vocab]."""
    ids_t = torch.tensor([token_ids], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        out = hf_model(input_ids=ids_t)
    return out.logits[0, -1, :]


# ─────────────────────────────────────────────────────────────────────────────
# Paged helpers (CUDA + flashinfer only)
# ─────────────────────────────────────────────────────────────────────────────

def do_extend_lora(
    model: Qwen3ForCausalLM,
    kv_pool: "KVPool",
    rtp: "ReqToTokenPool",
    workspace: torch.Tensor,
    reqs: List[Dict],
    cfg,
    adapter: Optional[LoRAAdapter],
) -> Tuple[torch.Tensor, List[int]]:
    """
    One batched EXTEND pass with optional per-request LoRA mask.

    Each req dict:
      fill_ids         List[int]
      kv_committed_len int
      slot_indices     List[int]  mutated in-place
      req_pool_idx     int
      use_lora         bool       True → mask=1.0 for this request's tokens
    """
    P = PAGE_SIZE
    B = len(reqs)

    write_infos:     List[WriteInfo] = []
    qo_indptr_list:  List[int] = [0]
    num_pages_list:  List[int] = []
    kv_last_pg_list: List[int] = []

    for rd in reqs:
        wi = compute_write_info(
            kv_pool          = kv_pool,
            rtp              = rtp,
            slot_indices     = rd["slot_indices"],
            req_pool_idx     = rd["req_pool_idx"],
            kv_committed_len = rd["kv_committed_len"],
            n_fill           = len(rd["fill_ids"]),
        )
        write_infos.append(wi)
        num_pages_list.append(len(rd["slot_indices"]))
        total = rd["kv_committed_len"] + len(rd["fill_ids"])
        last  = total % P
        kv_last_pg_list.append(last if last else P)
        qo_indptr_list.append(qo_indptr_list[-1] + len(rd["fill_ids"]))

    qo_t  = torch.tensor(qo_indptr_list, dtype=torch.int32, device=DEVICE)
    npg_t = torch.tensor(num_pages_list,  dtype=torch.int32, device=DEVICE)
    klp_t = torch.tensor(kv_last_pg_list, dtype=torch.int32, device=DEVICE)
    rpi_t = torch.tensor([rd["req_pool_idx"] for rd in reqs], dtype=torch.int32, device=DEVICE)

    kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
    torch.cumsum(npg_t, dim=0, out=kv_indptr[1:])
    kv_indices = torch.empty(int(npg_t.sum()), dtype=torch.int32, device=DEVICE)
    create_flashinfer_kv_indices_triton[(B,)](
        rtp.req_to_token, rpi_t, npg_t, kv_indptr,
        None, kv_indices, rtp.req_to_token.shape[1],
    )

    wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(workspace, "NHD")
    wrapper.begin_forward(
        qo_t, kv_indptr, kv_indices, klp_t,
        cfg.num_attention_heads, cfg.num_key_value_heads,
        cfg.head_dim, P, causal=True, q_data_type=DTYPE,
    )

    all_ids: List[int] = []
    pos_ids: List[int] = []
    mask_vals: List[float] = []
    for rd in reqs:
        all_ids.extend(rd["fill_ids"])
        for j in range(len(rd["fill_ids"])):
            pos_ids.append(rd["kv_committed_len"] + j)
        lora_val = 1.0 if rd.get("use_lora", False) else 0.0
        mask_vals.extend([lora_val] * len(rd["fill_ids"]))

    ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    pos_t = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    total_tokens = qo_indptr_list[-1]
    lora_mask    = torch.tensor(mask_vals, dtype=DTYPE, device=DEVICE).view(1, total_tokens, 1)

    ctx = ExtendKVCtx(
        wrapper     = wrapper,
        k_pool      = kv_pool.k_pool,
        v_pool      = kv_pool.v_pool,
        qo_indptr   = qo_indptr_list,
        write_infos = write_infos,
        page_size   = P,
    )
    with torch.no_grad():
        logits = model(
            ids_t,
            attention_mask = None,
            kv_cache       = ctx,
            position_ids   = pos_t,
            lora_mask      = lora_mask if adapter else None,
            lora_adapter   = adapter,
        )
    wrapper.end_forward()
    return logits, qo_indptr_list


def bar(ok: bool) -> str:
    return "✓" if ok else "✗"


# ─────────────────────────────────────────────────────────────────────────────
# Test 0 — Base model match (prerequisite)
# ─────────────────────────────────────────────────────────────────────────────

def test_base_match(our_model, hf_model, tok, prompts: List[str]) -> bool:
    print("─" * 64)
    print("Test 0: Base model match (our NOCACHE vs HuggingFace)")
    print("─" * 64)

    all_ok = True
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}])
        ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()

        our_lg = our_nocache_logit(our_model, ids, adapter=None)
        hf_lg  = hf_base_logit(hf_model, ids)

        diff = (our_lg - hf_lg).abs().max().item()
        ok   = diff < ATOL
        all_ok = all_ok and ok
        print(f"  '{p[:50]}'")
        print(f"    max_diff={diff:.4f}  {bar(ok)}")

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    if not all_ok:
        print("  ⚠  Base models don't match — LoRA comparison results are unreliable.")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 1 — Adapter changes output
# ─────────────────────────────────────────────────────────────────────────────

def test_adapter_changes_output(our_model, adapter, tok, prompt: str) -> bool:
    print("─" * 64)
    print("Test 1: Adapter changes output (LoRA ≠ base)")
    print("─" * 64)

    text = tok.apply_chat_template([{"role": "user", "content": prompt}])
    ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()

    base_lg = our_nocache_logit(our_model, ids, adapter=None)
    lora_lg = our_nocache_logit(our_model, ids, adapter=adapter)

    diff = (lora_lg - base_lg).abs().max().item()
    # Adapter should change output meaningfully (> 1e-3)
    ok   = diff > 1e-3
    print(f"  prompt: '{prompt[:50]}'")
    print(f"  max_diff (lora vs base) = {diff:.6f}  {bar(ok)}")
    if not ok:
        print("  ⚠  LoRA delta is near-zero — adapter weights may be zero or not applied.")

    result = "PASS ✓" if ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 2 — Our LoRA matches PEFT (core correctness)
# ─────────────────────────────────────────────────────────────────────────────

def test_lora_matches_peft(
    our_model, adapter, hf_model, peft_model, tok, prompts: List[str]
) -> bool:
    print("─" * 64)
    print("Test 2: Our LoRA NOCACHE vs PEFT (ground-truth correctness)")
    print("  Two checks per prompt:")
    print("  (a) absolute: our_lora vs peft_lora   [full logit comparison]")
    print("  (b) delta:    (our_lora-our_base) vs (peft_lora-hf_base)  [LoRA math only]")
    print("─" * 64)

    all_ok = True
    for p in prompts:
        text = tok.apply_chat_template([{"role": "user", "content": p}])
        ids  = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()

        our_base_lg = our_nocache_logit(our_model, ids, adapter=None)
        our_lora_lg = our_nocache_logit(our_model, ids, adapter=adapter)
        hf_base_lg  = hf_base_logit(hf_model, ids)
        peft_lg     = peft_logit(peft_model, ids)

        # (a) Absolute comparison — may be affected by base model numerical diffs
        diff_abs = (our_lora_lg - peft_lg).abs().max().item()

        # (b) Delta comparison — isolates our LoRA math from base model differences
        our_delta  = our_lora_lg  - our_base_lg   # our LoRA contribution
        peft_delta = peft_lg      - hf_base_lg    # PEFT LoRA contribution
        diff_delta = (our_delta - peft_delta).abs().max().item()

        ok_delta = diff_delta < ATOL
        ok_abs   = diff_abs   < ATOL
        # Primary pass/fail is on the delta (isolates LoRA correctness)
        ok = ok_delta
        all_ok = all_ok and ok

        print(f"  '{p[:50]}'")
        print(f"    abs  diff={diff_abs:.4f}  {bar(ok_abs)}")
        print(f"    delta diff={diff_delta:.4f}  {bar(ok_delta)}  ← primary check")

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 3 — Mixed-batch mask separation (CUDA + flashinfer)
# ─────────────────────────────────────────────────────────────────────────────

def test_mixed_batch_separation(our_model, adapter, tok, cfg) -> bool:
    print("─" * 64)
    print("Test 3: Mixed-batch separation (lora + base in one EXTEND pass)")
    print("─" * 64)

    prompt_lora = "Summarize this article briefly."
    prompt_base = "What is 2+2?"

    def encode(p):
        text = tok.apply_chat_template([{"role": "user", "content": p}])
        return tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()

    ids_lora = encode(prompt_lora)
    ids_base = encode(prompt_base)

    ref_lora = our_nocache_logit(our_model, ids_lora, adapter=adapter)
    ref_base = our_nocache_logit(our_model, ids_base, adapter=None)

    total_toks = len(ids_lora) + len(ids_base)
    total_pgs  = math.ceil(total_toks / PAGE_SIZE) + 8
    kv_pool    = KVPool(total_pgs, PAGE_SIZE, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, DTYPE)
    rtp        = ReqToTokenPool(4, total_pgs + 4)
    workspace  = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)

    reqs = [
        {
            "fill_ids":         ids_lora,
            "kv_committed_len": 0,
            "slot_indices":     [],
            "req_pool_idx":     rtp.alloc(),
            "use_lora":         True,
        },
        {
            "fill_ids":         ids_base,
            "kv_committed_len": 0,
            "slot_indices":     [],
            "req_pool_idx":     rtp.alloc(),
            "use_lora":         False,
        },
    ]

    logits, qo_indptr = do_extend_lora(our_model, kv_pool, rtp, workspace, reqs, cfg, adapter)

    paged_lora = logits[0, qo_indptr[1] - 1, :]
    paged_base = logits[0, qo_indptr[2] - 1, :]

    diff_lora = (paged_lora - ref_lora).abs().max().item()
    diff_base = (paged_base - ref_base).abs().max().item()

    ok_lora = diff_lora < ATOL
    ok_base = diff_base < ATOL
    all_ok  = ok_lora and ok_base

    print(f"  lora request:  max_diff vs ref_lora = {diff_lora:.4f}  {bar(ok_lora)}")
    print(f"  base request:  max_diff vs ref_base = {diff_base:.4f}  {bar(ok_base)}")

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Test 4 — Paged LoRA prefill + decode (CUDA + flashinfer)
# ─────────────────────────────────────────────────────────────────────────────

def test_paged_lora(our_model, adapter, tok, cfg, n_decode: int) -> bool:
    print("─" * 64)
    print(f"Test 4: Paged LoRA prefill + {n_decode} decode steps")
    print("─" * 64)

    prompt = "Summarize this article in one sentence."
    text   = tok.apply_chat_template([{"role": "user", "content": prompt}])
    ids    = tok._tok([text], return_tensors="pt", padding=False)["input_ids"][0].tolist()
    L      = len(ids)

    max_ctx   = math.ceil((L + n_decode + 4) / PAGE_SIZE) + 4
    total_pgs = max_ctx + 8
    kv_pool   = KVPool(total_pgs, PAGE_SIZE, cfg.num_hidden_layers, cfg.num_key_value_heads, cfg.head_dim, DTYPE)
    rtp       = ReqToTokenPool(4, max_ctx)
    ws_ext    = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    ws_dec    = torch.empty(256 * 1024 * 1024, dtype=torch.uint8, device=DEVICE)
    dec_wrap  = flashinfer.BatchDecodeWithPagedKVCacheWrapper(ws_dec, "NHD", use_tensor_cores=False)
    kv_iptr   = torch.zeros(2, dtype=torch.int32, device=DEVICE)

    rpi          = rtp.alloc()
    slot_indices: List[int] = []

    # ── Prefill ──────────────────────────────────────────────────────────────
    rd = [{"fill_ids": ids, "kv_committed_len": 0,
           "slot_indices": slot_indices, "req_pool_idx": rpi, "use_lora": True}]
    logits, qo_indptr = do_extend_lora(our_model, kv_pool, rtp, ws_ext, rd, cfg, adapter)

    ref_prefill = our_nocache_logit(our_model, ids, adapter=adapter)
    diff0       = (logits[0, qo_indptr[1] - 1, :] - ref_prefill).abs().max().item()
    ok0         = diff0 < ATOL
    print(f"  prefill:  max_diff={diff0:.4f}  {bar(ok0)}")

    # Sample first decode token
    next_tok = int(logits[0, qo_indptr[1] - 1, :].argmax())
    full_seq  = list(ids) + [next_tok]

    # ── Decode steps ─────────────────────────────────────────────────────────
    all_ok = ok0
    for step in range(n_decode):
        seq_len      = len(full_seq) - 1
        token_offset = seq_len % PAGE_SIZE
        num_pages    = len(slot_indices)

        if token_offset == 0:
            new_pg = kv_pool.alloc(1)[0]
            slot_indices.append(new_pg)
            rtp.req_to_token[rpi, num_pages] = new_pg
            last_pg    = new_pg
            num_pages += 1
        else:
            last_pg = slot_indices[-1]

        npg_t    = torch.tensor([num_pages],    dtype=torch.int32, device=DEVICE)
        rpi_t    = torch.tensor([rpi],          dtype=torch.int32, device=DEVICE)
        lpg_t    = torch.tensor([last_pg],      dtype=torch.int64, device=DEVICE)
        kv_last  = torch.tensor([token_offset + 1], dtype=torch.int32, device=DEVICE)

        kv_iptr[0] = 0; kv_iptr[1] = num_pages
        kv_idx = torch.empty(num_pages, dtype=torch.int32, device=DEVICE)
        create_flashinfer_kv_indices_triton[(1,)](
            rtp.req_to_token, rpi_t, npg_t, kv_iptr,
            None, kv_idx, rtp.req_to_token.shape[1],
        )

        dec_wrap.begin_forward(
            kv_iptr, kv_idx, kv_last,
            cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, PAGE_SIZE,
            data_type=DTYPE, q_data_type=DTYPE,
        )
        ctx = DecodeKVCtx(
            wrapper           = dec_wrap,
            k_pool            = kv_pool.k_pool,
            v_pool            = kv_pool.v_pool,
            last_page_indices = lpg_t,
            token_offsets     = torch.tensor([token_offset], dtype=torch.int64, device=DEVICE),
        )

        cur_tok_t = torch.tensor([[next_tok]], dtype=torch.long, device=DEVICE)
        pos_t     = torch.tensor([[seq_len]],  dtype=torch.long, device=DEVICE)
        dec_mask  = torch.ones(1, 1, 1, dtype=DTYPE, device=DEVICE)   # [B=1, q_len=1, 1]

        with torch.no_grad():
            lg = our_model(
                cur_tok_t,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_t,
                lora_mask      = dec_mask,
                lora_adapter   = adapter,
            )
        dec_wrap.end_forward()

        ref  = our_nocache_logit(our_model, full_seq, adapter=adapter)
        diff = (ref - lg[0, -1, :]).abs().max().item()
        ok   = diff < ATOL
        all_ok = all_ok and ok
        print(f"  decode step {step:2d}:  max_diff={diff:.4f}  {bar(ok)}")

        next_tok = int(ref.argmax())
        full_seq.append(next_tok)

    result = "PASS ✓" if all_ok else "FAIL ✗"
    print(f"  → {result}")
    print()
    return all_ok


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

PROMPTS = [
    "What is 2+2?",
    "Explain what a neural network is in one sentence.",
    "What is the capital of France?",
]

def main():
    print("=" * 64)
    print(" verify_lora.py — Layer 20: LoRA Correctness Tests")
    print(f" model={args.model}")
    print(f" lora={args.lora}")
    print(f" device={DEVICE}  dtype={DTYPE}  atol={ATOL}")
    print(f" flashinfer={HAS_FLASHINFER}  paged_tests={PAGED_AVAILABLE}")
    print("=" * 64)
    print()

    # ── Load tokenizer ────────────────────────────────────────────────────────
    print("Loading tokenizer …")
    tok = Tokenizer(args.model)
    print()

    # ── Load our custom model ─────────────────────────────────────────────────
    print("Loading our Qwen3ForCausalLM …")
    our_model = load_our_model(args.model, DEVICE, DTYPE)
    cfg       = our_model.model.config
    print()

    # ── Load HuggingFace base model ───────────────────────────────────────────
    print("Loading HuggingFace base model (for Test 0) …")
    hf_model = load_hf_base_model(args.model, DEVICE, DTYPE)
    print()

    # ── Load our LoRA adapter ─────────────────────────────────────────────────
    print("Loading LoRAAdapter …")
    adapter = LoRAAdapter(args.lora, dtype=DTYPE, device=DEVICE)
    print()

    # ── Load PEFT model ───────────────────────────────────────────────────────
    print("Loading PEFT model (HuggingFace + adapter, for Test 2) …")
    peft_model = load_peft_model(args.model, args.lora, DEVICE, DTYPE)
    print()

    # ── Run tests ─────────────────────────────────────────────────────────────
    r0 = test_base_match(our_model, hf_model, tok, PROMPTS)
    r1 = test_adapter_changes_output(our_model, adapter, tok, PROMPTS[0])
    r2 = test_lora_matches_peft(our_model, adapter, hf_model, peft_model, tok, PROMPTS)

    if PAGED_AVAILABLE:
        r3 = test_mixed_batch_separation(our_model, adapter, tok, cfg)
        r4 = test_paged_lora(our_model, adapter, tok, cfg, args.n_decode)
    else:
        print("─" * 64)
        print("Test 3: Mixed-batch separation  — SKIPPED (requires CUDA + flashinfer)")
        print("Test 4: Paged LoRA prefill+decode — SKIPPED (requires CUDA + flashinfer)")
        print("─" * 64)
        print()
        r3 = r4 = None

    # ── Summary ───────────────────────────────────────────────────────────────
    print("=" * 64)
    print(" SUMMARY")
    print("=" * 64)
    print(f"  Test 0 Base match:           {'PASS ✓' if r0 else 'FAIL ✗'}")
    print(f"  Test 1 Adapter changes out:  {'PASS ✓' if r1 else 'FAIL ✗'}")
    print(f"  Test 2 Matches PEFT:         {'PASS ✓' if r2 else 'FAIL ✗'}")
    if r3 is not None:
        print(f"  Test 3 Mixed-batch mask:     {'PASS ✓' if r3 else 'FAIL ✗'}")
        print(f"  Test 4 Paged LoRA:           {'PASS ✓' if r4 else 'FAIL ✗'}")
    else:
        print(f"  Test 3 Mixed-batch mask:     SKIPPED")
        print(f"  Test 4 Paged LoRA:           SKIPPED")

    core_results = [r for r in [r0, r1, r2, r3, r4] if r is not None]
    overall = all(core_results)
    print("=" * 64)
    print(f" OVERALL: {'PASS ✓' if overall else 'FAIL ✗'}")
    print("=" * 64)
    sys.exit(0 if overall else 1)


if __name__ == "__main__":
    main()
