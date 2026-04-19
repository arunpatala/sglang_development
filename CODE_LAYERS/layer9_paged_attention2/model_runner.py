"""
Layer 9 — ModelRunner: ReqToTokenPool + Triton kv_indices + variable page_size.

Extends Layer 8 (paged KV, page_size=1) with two improvements introduced
together as a single step:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  ReqToTokenPool  (GPU-resident 2D lookup table)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    req_to_token[req_pool_idx, page_pos] = physical_page_idx

Allocated once at startup: [max_concurrent_reqs, max_pages_per_req] int32.

At prefill:   req_to_token[req_pool_idx, 0:n_pages] = page_indices_tensor
At decode:    when a new page is needed:
                req_to_token[req_pool_idx, num_pages] = new_page_idx (scalar)
              when writing within an existing last page: no table update.

Mirror of SGLang's ReqToTokenPool (srt/mem_cache/memory_pool.py).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.  create_flashinfer_kv_indices_triton  (Triton kernel)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads req_to_token rows and writes the flat kv_indices tensor on-GPU.
One threadblock per request; runs all B requests in parallel.

In Layer 8 (page_size=1), kv_indices was assembled in Python:
    for req in reqs: list.extend(req.slot_indices) + [new_slot]
    kv_indices = torch.tensor(list)   ← O(Σ kv_tokens) CPU→GPU copy each step

This Triton kernel eliminates that copy entirely. The slot/page data never
crosses the PCIe bus again after prefill writes it to req_to_token on-GPU.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
3.  Variable page_size (PAGE_SIZE = 16)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Layer 8 kept page_size=1: every token occupied its own pool slot.
Layer 9 groups PAGE_SIZE tokens into each pool entry.

Prefill:
  • Allocates ceil(prompt_len / PAGE_SIZE) pages (not prompt_len slots).
  • Writes page indices to req_to_token (one column per page, not per token).
  • req.slot_indices stores page indices.

Decode — conditional page allocation:
  seq_len      = prompt_len + len(output_ids)   ← total tokens in pool
  token_offset = seq_len % PAGE_SIZE            ← position within last page
  If token_offset == 0:  current last page is full → alloc 1 new page,
                         append to slot_indices, write to req_to_token.
  Else:                  write into the existing last page at token_offset.
  Reduces page alloc calls by factor PAGE_SIZE vs Layer 8.

kv_indptr counts pages (not tokens) → Triton writes PAGE_SIZE× fewer ints.
kv_last_page_lens = token_offset + 1  (range 1..PAGE_SIZE, not always 1).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Additional micro-optimisations vs Layer 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • BatchDecodeWithPagedKVCacheWrapper created once at __init__ (not per step).
  • kv_indptr built via torch.cumsum on GPU (not itertools.accumulate on CPU).
  • kv_indptr buffer pre-allocated [max_batch+1] and reused every step.
  • kv_last_page_lens buffer pre-allocated; filled from token_offsets each step.

What still happens on CPU each step (O(B), not O(Σ kv_tokens)):
  • Python loop over reqs to read len(input_ids)+len(output_ids) → seq_lens
  • torch.tensor(seq_lens) → small CPU→GPU transfer (B ints)
  • Page alloc only when needed (O(1) per request when page fills)
"""

import logging
import math
import sys
import time
from pathlib import Path
from typing import List

import torch
import flashinfer

sys.path.insert(0, str(Path(__file__).parent))

from forward_batch import ForwardBatch, ForwardMode
from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx, ReqToTokenPool
from model import Qwen3ForCausalLM
from request import Req, ReqStatus
from sampler import sample_token
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16

_KV_MEMORY_FRACTION  = 0.85
_WORKSPACE_MB        = 256
_MAX_CONCURRENT_REQS = 128
_MAX_TOKEN_CONTEXT   = 4096   # max tokens per request (used to size req_to_token)
PAGE_SIZE            = 16     # tokens per KV page


class ModelRunner:

    def __init__(
        self,
        model_path: str,
        page_size: int = PAGE_SIZE,
        kv_memory_fraction: float = _KV_MEMORY_FRACTION,
    ) -> None:
        logger.info(
            f"ModelRunner: loading model from {model_path}  "
            f"page_size={page_size}  kv_memory_fraction={kv_memory_fraction}"
        )
        t0 = time.perf_counter()

        self.page_size = page_size

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)
        cfg = self.model.model.config

        # ── KVPool ────────────────────────────────────────────────────────
        free_bytes, _ = torch.cuda.mem_get_info()
        bytes_per_token = (
            cfg.num_hidden_layers * 2
            * cfg.num_key_value_heads
            * cfg.head_dim
            * (torch.finfo(DTYPE).bits // 8)
        )
        # Size in pages: same total GPU memory, different granularity.
        max_pages = int(free_bytes * kv_memory_fraction / (page_size * bytes_per_token))

        self.kv_pool = KVPool(
            total_pages = max_pages,
            page_size   = page_size,
            n_layers    = cfg.num_hidden_layers,
            n_kv_heads  = cfg.num_key_value_heads,
            head_dim    = cfg.head_dim,
            dtype       = DTYPE,
        )

        # ── ReqToTokenPool ────────────────────────────────────────────────
        # Columns store page indices; fewer columns needed than with page_size=1.
        max_pages_per_req = math.ceil(_MAX_TOKEN_CONTEXT / page_size)
        self.req_to_token_pool = ReqToTokenPool(
            max_batch       = _MAX_CONCURRENT_REQS,
            max_context_len = max_pages_per_req,
        )
        self._max_pages_per_req = max_pages_per_req

        # ── Pre-allocated decode-step buffers ─────────────────────────────
        self._kv_indptr_buf = torch.zeros(
            _MAX_CONCURRENT_REQS + 1, dtype=torch.int32, device=DEVICE
        )
        self._kv_last_page_lens_buf = torch.ones(
            _MAX_CONCURRENT_REQS, dtype=torch.int32, device=DEVICE
        )

        # ── FlashInfer workspace + decode wrapper (created once) ──────────
        # use_tensor_cores=False: Qwen3-0.6B GQA group=2 is below the
        # threshold of 4 where tensor cores help.
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, "NHD", use_tensor_cores=False
        )

        logger.info(
            f"ModelRunner ready in {time.perf_counter()-t0:.1f}s  "
            f"GPU={torch.cuda.memory_allocated()/1024**2:.0f} MB  "
            f"KVPool pages={max_pages}  page_size={page_size}  "
            f"ReqToTokenPool [{_MAX_CONCURRENT_REQS}, {max_pages_per_req}]"
        )

    # ------------------------------------------------------------------
    # Prefill — B=1, allocates pages, writes K/V + req_to_token
    # ------------------------------------------------------------------

    def prefill(self, req: Req) -> None:
        """
        B=1 prefill.

        Allocates ceil(prompt_len / page_size) pages from KVPool.
        Writes page indices to req_to_token[req_pool_idx, 0:n_pages].
        req.slot_indices = page index list (one per page, not per token).
        """
        prompt_len = len(req.input_ids)
        P          = self.page_size
        n_pages    = math.ceil(prompt_len / P)

        page_indices     = self.kv_pool.alloc(prompt_len)   # returns n_pages entries
        req.slot_indices = page_indices

        # Allocate a ReqToTokenPool row; write all page indices at once.
        req.req_pool_idx = self.req_to_token_pool.alloc()
        pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t

        ids  = torch.tensor([req.input_ids], device=DEVICE)
        mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
        pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

        ctx = PrefillKVCtx(page_indices, self.kv_pool)
        fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
        with torch.no_grad():
            logits = self.model(ids, forward_batch=fb, position_ids=pos)

        req.t_first_token = time.perf_counter()

        next_tok = sample_token(logits[0, -1], req.temperature)
        req.output_ids.append(next_tok)

        if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
            req.status   = ReqStatus.FINISHED
            req.t_finish = time.perf_counter()
            self.kv_pool.free(req.slot_indices)
            self.req_to_token_pool.free(req.req_pool_idx)
        else:
            req.status = ReqStatus.RUNNING

        logger.debug(
            f"prefill rid={req.rid[:8]} prompt_len={prompt_len} "
            f"n_pages={n_pages} req_pool_idx={req.req_pool_idx}"
        )

    # ------------------------------------------------------------------
    # Decode step — conditional page alloc + Triton kv_indices + GPU cumsum
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One batched decode step.

        Key differences from Layer 8:
          1. req_to_token table on GPU: new page written with a scalar index
             op (only when page fills), no per-token slot writes each step.
          2. kv_indptr built via torch.cumsum on GPU (not itertools on CPU).
          3. kv_indices built by Triton kernel on GPU (no CPU→GPU copy of
             page/slot data).
          4. Conditional page allocation: only allocate a new page every
             PAGE_SIZE steps, not every step.
          5. kv_last_page_lens = token_offset + 1, not always ones.
          6. decode_wrapper reused from __init__ (not re-created each step).
        """
        if not reqs:
            return []

        B   = len(reqs)
        P   = self.page_size
        cfg = self.model.model.config

        # ── Per-request metadata (O(B) Python, unavoidable) ───────────
        # seq_len = tokens already in KV cache = position of current input token.
        # output_ids[-1] is the input token for this step (appended by the
        # previous decode or prefill), so its position = (prompt + outputs - 1).
        seq_lens_list      = [len(r.input_ids) + len(r.output_ids) - 1 for r in reqs]
        token_offsets_list = [sl % P                                for sl in seq_lens_list]
        num_pages_list     = [len(r.slot_indices)                   for r in reqs]
        req_pool_idx_list  = [r.req_pool_idx                        for r in reqs]

        # ── Conditional page allocation ────────────────────────────────
        # A new page is needed only when token_offset == 0 (last page full).
        last_page_idx_list = []
        for i, req in enumerate(reqs):
            if token_offsets_list[i] == 0:
                new_page = self.kv_pool.alloc(1)[0]
                req.slot_indices.append(new_page)
                # Scalar write: one int32 into the GPU table.
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, num_pages_list[i]
                ] = new_page
                last_page_idx_list.append(new_page)
                num_pages_list[i] += 1
            else:
                last_page_idx_list.append(req.slot_indices[-1])

        # ── Build GPU tensors (O(B) CPU→GPU) ──────────────────────────
        seq_lens_t        = torch.tensor(seq_lens_list,       dtype=torch.int32, device=DEVICE)
        token_offsets_t   = torch.tensor(token_offsets_list,  dtype=torch.int32, device=DEVICE)
        num_pages_t       = torch.tensor(num_pages_list,      dtype=torch.int32, device=DEVICE)
        req_pool_idx_t    = torch.tensor(req_pool_idx_list,   dtype=torch.int32, device=DEVICE)
        last_page_idx_t   = torch.tensor(last_page_idx_list,  dtype=torch.int64, device=DEVICE)
        token_offsets_i64 = token_offsets_t.to(torch.int64)

        # ── kv_last_page_lens: valid tokens in last page after writing ─
        kv_last_page_lens = token_offsets_t + 1   # [B], range 1..page_size

        # ── kv_indptr on GPU via cumsum of page counts ─────────────────
        self._kv_indptr_buf[0] = 0
        torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])
        kv_indptr = self._kv_indptr_buf[: B + 1]

        # ── kv_indices on GPU via Triton kernel ───────────────────────
        total_pages_in_batch = sum(num_pages_list)
        kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            self.req_to_token_pool.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token_pool.req_to_token.shape[1],
        )

        # ── Per-request position IDs ───────────────────────────────────
        pos_ids = seq_lens_t.unsqueeze(1).to(torch.long)   # [B, 1]

        # ── Input tokens ───────────────────────────────────────────────
        last_toks = torch.tensor(
            [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
        )

        # ── FlashInfer plan ────────────────────────────────────────────
        self._decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            P,               # page_size — now > 1
            data_type   = DTYPE,
            q_data_type = DTYPE,
        )

        ctx = DecodeKVCtx(
            wrapper           = self._decode_wrapper,
            k_pool            = self.kv_pool.k_pool,
            v_pool            = self.kv_pool.v_pool,
            last_page_indices = last_page_idx_t,
            token_offsets     = token_offsets_i64,
        )
        fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=ctx, attention_mask=None)

        # ── Batched forward ────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(
                last_toks,
                forward_batch = fb,
                position_ids  = pos_ids,
            )   # [B, 1, vocab]

        self._decode_wrapper.end_forward()

        # ── Sample + handle finished requests ─────────────────────────
        newly_finished: List[Req] = []
        for i, req in enumerate(reqs):
            next_tok = sample_token(logits[i, -1], req.temperature)
            req.output_ids.append(next_tok)

            if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
                req.status   = ReqStatus.FINISHED
                req.t_finish = time.perf_counter()
                newly_finished.append(req)
                self.kv_pool.free(req.slot_indices)
                self.req_to_token_pool.free(req.req_pool_idx)

        return newly_finished

    # ------------------------------------------------------------------
    # Token → text
    # ------------------------------------------------------------------

    def decode_output(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids)
