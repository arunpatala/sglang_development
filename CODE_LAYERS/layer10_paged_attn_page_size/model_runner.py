"""
Layer 9 — ModelRunner: variable page_size support.

Extends Layer 8 by making the KV pool use pages of page_size > 1 tokens.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
What changes from Layer 8
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Prefill:
  • Allocates ceil(prompt_len / page_size) pages (not prompt_len slots).
  • Writes page indices (not token slot indices) to req_to_token.
  • req.slot_indices stores page indices (one per page, not one per token).

Decode — the key difference:
  • Determines seq_len = prompt_len + len(output_ids) = total tokens so far.
  • token_offset = seq_len % page_size
    - If token_offset == 0: new page is needed → alloc 1 page, write to
      req_to_token at num_pages position, store page index on req.slot_indices.
    - If token_offset != 0: reuse the existing last page; no new allocation.
  • last_page_indices[i] = page where the new token lands (new or existing last).
  • token_offsets[i]     = position within that page.
  • kv_last_page_lens[i] = token_offset + 1  (1..page_size).
  • kv_indptr based on num_pages (not num_tokens).
  • Triton kernel still reads req_to_token[req_pool_idx, 0:num_pages] unchanged.

Memory:
  • total_pages = floor(free_bytes * fraction / (page_size * bytes_per_token))
  • Same GPU memory as page_size=1 (just different granularity).

Benefits of page_size > 1:
  • Fewer free-list operations per request (pages instead of tokens).
  • kv_indices is shorter by factor page_size.
  • Triton kernel writes fewer ints per step.
  • Allocation overhead O(Σ prompt_len / page_size) instead of O(Σ prompt_len).
  • (Diminishing returns for very short sequences.)
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

from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx, ReqToTokenPool
from model import Qwen3ForCausalLM
from request import Req, ReqStatus
from tokenizer import Tokenizer
from triton_utils import create_flashinfer_kv_indices_triton

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16

_KV_MEMORY_FRACTION  = 0.85
_WORKSPACE_MB        = 256
_MAX_CONCURRENT_REQS = 128
_MAX_TOKEN_CONTEXT   = 4096   # max tokens per request (used to size req_to_token)
PAGE_SIZE            = 16     # tokens per KV page — the new knob


class ModelRunner:

    def __init__(self, model_path: str, page_size: int = PAGE_SIZE) -> None:
        logger.info(f"ModelRunner: loading model from {model_path}  page_size={page_size}")
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
        # Total pages = token budget / page_size
        max_pages = int(free_bytes * _KV_MEMORY_FRACTION / (page_size * bytes_per_token))

        self.kv_pool = KVPool(
            total_pages = max_pages,
            page_size   = page_size,
            n_layers    = cfg.num_hidden_layers,
            n_kv_heads  = cfg.num_key_value_heads,
            head_dim    = cfg.head_dim,
            dtype       = DTYPE,
        )

        # ── ReqToTokenPool ────────────────────────────────────────────────
        # Columns now hold *page* indices; max_context = ceil(tokens / page_size).
        max_pages_per_req = math.ceil(_MAX_TOKEN_CONTEXT / page_size)
        self.req_to_token_pool = ReqToTokenPool(
            max_batch       = _MAX_CONCURRENT_REQS,
            max_context_len = max_pages_per_req,
        )
        self._max_pages_per_req = max_pages_per_req

        # ── Pre-allocated decode-step buffers ─────────────────────────────
        # kv_indptr: [max_batch+1] — cumsum of num_pages per request
        # kv_last_page_lens: [max_batch] — valid tokens in last page (1..P)
        self._kv_indptr_buf = torch.zeros(
            _MAX_CONCURRENT_REQS + 1, dtype=torch.int32, device=DEVICE
        )
        self._kv_last_page_lens_buf = torch.ones(
            _MAX_CONCURRENT_REQS, dtype=torch.int32, device=DEVICE
        )

        # ── FlashInfer workspace + decode wrapper ─────────────────────────
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
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        if temperature == 0.0:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    # ------------------------------------------------------------------
    # Prefill — B=1, allocates pages, writes K/V during forward
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

        req.req_pool_idx = self.req_to_token_pool.alloc()
        pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t

        ids  = torch.tensor([req.input_ids], device=DEVICE)
        mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
        pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

        ctx = PrefillKVCtx(page_indices, self.kv_pool)
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask, kv_cache=ctx, position_ids=pos)

        req.t_first_token = time.perf_counter()

        next_tok = self._sample(logits[0, -1], req.temperature)
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
    # Decode step — conditional page allocation + Triton kv_indices
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One batched decode step.

        Key difference from Layer 8:
          We only allocate a new KV page when the current last page is full
          (token_offset == 0).  Otherwise we write into the existing last page.
          This reduces allocator churn by a factor of page_size.

        seq_len   = prompt_len + output_len    ← total tokens in pool (before new)
        num_pages = len(req.slot_indices)       ← pages already allocated
        token_offset = seq_len % page_size      ← slot within the last page

        If token_offset == 0: alloc 1 new page, append to slot_indices,
                               write to req_to_token at column num_pages.
        Else:                  use existing last page (slot_indices[-1]).

        kv_last_page_lens[i] = token_offset + 1  (after writing new token)
        kv_indptr is built from num_pages_after (page count, not token count).
        """
        if not reqs:
            return []

        B   = len(reqs)
        P   = self.page_size
        cfg = self.model.model.config

        # ── Per-request metadata (Python, O(B)) ───────────────────────
        seq_lens_list      = [len(r.input_ids) + len(r.output_ids) for r in reqs]
        token_offsets_list = [sl % P                                for sl in seq_lens_list]
        num_pages_list     = [len(r.slot_indices)                   for r in reqs]
        req_pool_idx_list  = [r.req_pool_idx                        for r in reqs]

        # ── Conditional page allocation ────────────────────────────────
        # needs_new_page[i] is True when the new token starts a fresh page.
        last_page_idx_list = []
        new_page_allocs: List[tuple] = []   # (req_idx, page_idx) for new pages

        for i, req in enumerate(reqs):
            if token_offsets_list[i] == 0:
                # Current last page is full — need a fresh page.
                new_page = self.kv_pool.alloc(1)[0]
                req.slot_indices.append(new_page)
                # Write new page index into req_to_token at the new column.
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, num_pages_list[i]
                ] = new_page
                last_page_idx_list.append(new_page)
                new_page_allocs.append((i, new_page))
                num_pages_list[i] += 1       # update for kv_indptr below
            else:
                # Reuse the existing last page.
                last_page_idx_list.append(req.slot_indices[-1])

        # ── Build GPU tensors (O(B) CPU→GPU, same as Layer 8) ─────────
        seq_lens_t       = torch.tensor(seq_lens_list,      dtype=torch.int32, device=DEVICE)
        token_offsets_t  = torch.tensor(token_offsets_list, dtype=torch.int32, device=DEVICE)
        num_pages_t      = torch.tensor(num_pages_list,     dtype=torch.int32, device=DEVICE)
        req_pool_idx_t   = torch.tensor(req_pool_idx_list,  dtype=torch.int32, device=DEVICE)
        last_page_idx_t  = torch.tensor(last_page_idx_list, dtype=torch.int64, device=DEVICE)
        token_offsets_i64 = token_offsets_t.to(torch.int64)

        # ── kv_last_page_lens: valid tokens in last page after writing ─
        # = token_offset + 1  (range 1..page_size)
        kv_last_page_lens = token_offsets_t + 1   # [B]

        # ── kv_indptr on GPU via cumsum of num_pages ───────────────────
        self._kv_indptr_buf[0] = 0
        torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])
        kv_indptr = self._kv_indptr_buf[: B + 1]

        # ── kv_indices on GPU via Triton kernel ───────────────────────
        # The Triton kernel reads req_to_token[req_pool_idx, 0:num_pages]
        # and writes page indices to kv_indices — unchanged from Layer 8.
        total_pages_in_batch = sum(num_pages_list)
        kv_indices = torch.empty(total_pages_in_batch, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            self.req_to_token_pool.req_to_token,
            req_pool_idx_t,
            num_pages_t,                              # page count (not token count)
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
            P,              # page_size — now > 1
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

        # ── Batched forward ────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(
                last_toks,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_ids,
            )   # [B, 1, vocab]

        self._decode_wrapper.end_forward()

        # ── Sample + handle finished requests ─────────────────────────
        newly_finished: List[Req] = []
        for i, req in enumerate(reqs):
            next_tok = self._sample(logits[i, -1], req.temperature)
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
