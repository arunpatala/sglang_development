"""
conftest.py — shared pytest fixtures for Layer 11 tests.

Fixtures:
  model     — Qwen3ForCausalLM loaded once per session (module scope).
  tok       — Tokenizer for the same checkpoint.
  cfg       — model.model.config shortcut.

Helpers exported (importable from conftest):
  make_pool          — creates a fresh KVPool + ReqToTokenPool for each test.
  make_radix_cache   — creates a RadixCache backed by a KVPool.
  do_extend          — runs one batched extend forward pass and returns logits.
  prefill_with_prefix — do_extend for a request with a cached prefix already
                        written into rtp.
  full_ref           — F.sdpa full-recompute reference for a single token
                       sequence.

The model is only loaded once per pytest session to keep tests fast.
"""

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pytest
import torch
import flashinfer

# Add the layer root so imports resolve without installing the package.
LAYER_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LAYER_ROOT))

from kv_cache import (
    DecodeKVCtx, ExtendKVCtx, KVPool, ReqToTokenPool,
    WriteInfo, compute_write_info,
)
from model import Qwen3ForCausalLM
from radix_cache import RadixCache
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────

MODEL_PATH = (
    Path.home()
    / ".cache/huggingface/hub"
    / "models--Qwen--Qwen3-0.6B/snapshots"
    / "c1899de289a04d12100db370d81485cdf75e47ca"
)
DEVICE = "cuda"
DTYPE  = torch.bfloat16
ATOL   = 0.75   # max-abs-diff tolerance between chunked and full-recompute


# ─────────────────────────────────────────────────────────────────────────────
# Session-scoped model fixture
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def tok():
    return Tokenizer(str(MODEL_PATH))


@pytest.fixture(scope="session")
def model():
    return Qwen3ForCausalLM.from_pretrained(str(MODEL_PATH), dtype=DTYPE)


@pytest.fixture(scope="session")
def cfg(model):
    return model.model.config


# ─────────────────────────────────────────────────────────────────────────────
# Per-test pool factory
# ─────────────────────────────────────────────────────────────────────────────

def make_pool(
    cfg,
    total_tokens: int,
    page_size: int = 16,
    max_batch: int = 8,
) -> Tuple[KVPool, ReqToTokenPool]:
    """Return a fresh KVPool + ReqToTokenPool sized for `total_tokens`."""
    total_pages   = math.ceil(total_tokens / page_size) + 8
    max_pages_req = total_pages + 4
    kv_pool = KVPool(
        total_pages = total_pages,
        page_size   = page_size,
        n_layers    = cfg.num_hidden_layers,
        n_kv_heads  = cfg.num_key_value_heads,
        head_dim    = cfg.head_dim,
        dtype       = DTYPE,
    )
    rtp = ReqToTokenPool(max_batch=max_batch, max_context_len=max_pages_req)
    return kv_pool, rtp


def make_radix_cache(kv_pool: KVPool) -> RadixCache:
    """Return a fresh RadixCache backed by the given KVPool."""
    return RadixCache(kv_pool, kv_pool.page_size)


def prefill_with_prefix(
    model,
    kv_pool: KVPool,
    rtp: ReqToTokenPool,
    workspace: torch.Tensor,
    full_ids: List[int],
    prefix_page_indices: List[int],
    prefix_len: int,
    cfg,
) -> Tuple[torch.Tensor, List[int], int]:
    """
    Run an extend pass for only the non-cached portion of a prompt.

    Before calling do_extend, writes the prefix pages into the rtp row so
    FlashInfer can see the cached KV when the suffix tokens attend backwards.

    Args:
      full_ids:            complete prompt token IDs
      prefix_page_indices: page indices returned by RadixCache.match_prefix()
      prefix_len:          number of tokens matched (multiple of page_size)

    Returns:
      (logit_last [vocab], slot_indices, req_pool_idx)
    """
    P   = kv_pool.page_size
    rpi = rtp.alloc()

    # Pre-populate slot_indices and rtp with the cached prefix pages.
    slot_indices = list(prefix_page_indices)
    n_pfx = len(prefix_page_indices)
    if n_pfx > 0:
        pages_t = torch.tensor(prefix_page_indices, dtype=torch.int32, device=DEVICE)
        rtp.req_to_token[rpi, :n_pfx] = pages_t

    fill_ids = full_ids[prefix_len:]
    rd = [{
        "fill_ids":         fill_ids,
        "kv_committed_len": prefix_len,
        "slot_indices":     slot_indices,
        "req_pool_idx":     rpi,
    }]
    logits, qo_indptr = do_extend(model, kv_pool, rtp, workspace, rd, cfg)
    return logits[0, qo_indptr[1] - 1, :], slot_indices, rpi


# ─────────────────────────────────────────────────────────────────────────────
# Extend helper (mirrors verify_batch.py's do_extend)
# ─────────────────────────────────────────────────────────────────────────────

def do_extend(
    model,
    kv_pool: KVPool,
    rtp: ReqToTokenPool,
    workspace: torch.Tensor,
    reqs: List[Dict],
    cfg,
) -> Tuple[torch.Tensor, List[int]]:
    """
    Run one batched extend forward pass.

    Each element of `reqs` is a dict:
      fill_ids         List[int]   — token IDs for this chunk
      kv_committed_len int         — tokens already in pool
      slot_indices     List[int]   — mutated in-place (page indices)
      req_pool_idx     int         — row in rtp

    Returns (logits [1, total_tokens, vocab], qo_indptr [B+1]).
    """
    P = kv_pool.page_size
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

    qo_t   = torch.tensor(qo_indptr_list,  dtype=torch.int32, device=DEVICE)
    npg_t  = torch.tensor(num_pages_list,   dtype=torch.int32, device=DEVICE)
    klp_t  = torch.tensor(kv_last_pg_list,  dtype=torch.int32, device=DEVICE)
    rpi_t  = torch.tensor([rd["req_pool_idx"] for rd in reqs], dtype=torch.int32, device=DEVICE)

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
    for rd in reqs:
        all_ids.extend(rd["fill_ids"])
        for j in range(len(rd["fill_ids"])):
            pos_ids.append(rd["kv_committed_len"] + j)

    ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
    pos_t = torch.tensor(pos_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

    ctx = ExtendKVCtx(
        wrapper     = wrapper,
        k_pool      = kv_pool.k_pool,
        v_pool      = kv_pool.v_pool,
        qo_indptr   = qo_indptr_list,
        write_infos = write_infos,
        page_size   = P,
    )
    with torch.no_grad():
        logits = model(ids_t, attention_mask=None, kv_cache=ctx, position_ids=pos_t)
    wrapper.end_forward()
    return logits, qo_indptr_list


def full_ref(model, token_ids: List[int]) -> torch.Tensor:
    """F.sdpa full-recompute over token_ids; returns logit tensor [vocab]."""
    ids_t = torch.tensor([token_ids], device=DEVICE)
    pos_t = torch.arange(len(token_ids), device=DEVICE).unsqueeze(0)
    mask  = torch.ones(1, len(token_ids), dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        lg = model(ids_t, attention_mask=mask, kv_cache=None, position_ids=pos_t)
    return lg[0, -1, :]
