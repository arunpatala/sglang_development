"""
Layer 10 — ModelRunner: batched prefill + chunked prefill support.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
New entry point: prefill_batch(reqs)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Replaces Layer 9's prefill(req) which was B=1 always.

The scheduler sets on each req before calling prefill_batch:
  req.fill_ids         — token slice for this extend round
  req.extend_input_len — len(fill_ids)
  req.kv_committed_len — tokens already in KV pool from prior chunks

prefill_batch() handles all cases uniformly:
  • Single full-prompt (B=1, kv_committed_len=0, fill_ids=input_ids)
  • Batched full prompts (B=N, kv_committed_len=0 for all)
  • Chunked continuation (B=1, kv_committed_len>0, fill_ids=chunk_slice)

Steps inside prefill_batch:
  1.  For each req with kv_committed_len==0: alloc req_pool_idx.
  2.  Allocate new KV pages for each req's chunk tokens.
  3.  Write new page indices into req_to_token at the right column offset.
  4.  Pack all fill_ids into one [1, total_tokens] input tensor.
  5.  Build position_ids: each req's tokens start at kv_committed_len.
  6.  Build qo_indptr (token boundaries), kv_indptr (page boundaries),
      kv_last_page_lens, and kv_indices (via Triton kernel from req_to_token).
  7.  begin_forward on the extend wrapper.
  8.  Run forward with ExtendKVCtx.
  9.  end_forward.
  10. Update req.kv_committed_len += extend_input_len.
  11. For last-chunk reqs: sample first token, set status RUNNING or FINISHED.
  12. For non-last-chunk reqs: set status PREFILLING (more chunks remain).

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
decode_step (unchanged from Layer 9)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
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

from kv_cache import DecodeKVCtx, ExtendKVCtx, KVPool, ReqToTokenPool, WriteInfo, compute_write_info
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
_MAX_TOKEN_CONTEXT   = 4096
PAGE_SIZE            = 16


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
        max_pages_per_req = math.ceil(_MAX_TOKEN_CONTEXT / page_size)
        self.req_to_token_pool = ReqToTokenPool(
            max_batch       = _MAX_CONCURRENT_REQS,
            max_context_len = max_pages_per_req,
        )
        self._max_pages_per_req = max_pages_per_req

        # ── Pre-allocated decode buffers ──────────────────────────────────
        self._kv_indptr_buf = torch.zeros(
            _MAX_CONCURRENT_REQS + 1, dtype=torch.int32, device=DEVICE
        )
        self._kv_last_page_lens_buf = torch.ones(
            _MAX_CONCURRENT_REQS, dtype=torch.int32, device=DEVICE
        )

        # ── FlashInfer workspaces ─────────────────────────────────────────
        # Two separate workspaces: one for extend (prefill), one for decode.
        # They don't overlap since extend and decode are separate forward passes.
        self._workspace_extend = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
        self._workspace_decode = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )

        # Decode wrapper created once at startup.
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace_decode, "NHD", use_tensor_cores=False
        )
        # Extend wrapper created fresh each prefill_batch call (shapes vary).
        # Uses a persistent workspace to avoid reallocation.
        self._extend_wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace_extend, "NHD"
        )

        logger.info(
            f"ModelRunner ready in {time.perf_counter()-t0:.1f}s  "
            f"GPU={torch.cuda.memory_allocated()/1024**2:.0f} MB  "
            f"KVPool pages={max_pages}  page_size={page_size}"
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
    # prefill_batch — the new unified extend entry point
    # ------------------------------------------------------------------

    def prefill_batch(self, reqs: List[Req]) -> None:
        """
        Run one EXTEND forward pass for all requests in `reqs`.

        Each req must have fill_ids, extend_input_len, kv_committed_len
        set by the scheduler before this call.

        After this call:
          • req.slot_indices includes all pages up to kv_committed_len + extend_input_len.
          • req.kv_committed_len is updated.
          • req.status is set to RUNNING, FINISHED, or PREFILLING.
          • For RUNNING/FINISHED requests: req.output_ids has the first token.
        """
        P   = self.page_size
        cfg = self.model.model.config

        # ── Step 1: alloc req_pool_idx for brand-new requests ─────────────
        for req in reqs:
            if req.req_pool_idx is None:
                req.req_pool_idx = self.req_to_token_pool.alloc()

        # ── Step 2: page packing — allocate pages with chunk-boundary fill ──
        # compute_write_info handles the partial-page continuation logic and
        # updates req.slot_indices + req_to_token in-place.
        write_infos: List[WriteInfo] = []
        for req in reqs:
            wi = compute_write_info(
                kv_pool          = self.kv_pool,
                rtp              = self.req_to_token_pool,
                slot_indices     = req.slot_indices,
                req_pool_idx     = req.req_pool_idx,
                kv_committed_len = req.kv_committed_len,
                n_fill           = req.extend_input_len,
            )
            write_infos.append(wi)

        # ── Step 3: pack all fill_ids into one input tensor ──────────────
        all_ids: List[int] = []
        for req in reqs:
            all_ids.extend(req.fill_ids)

        ids_t = torch.tensor(all_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # ── Step 4: position IDs — each req starts at kv_committed_len ──
        pos_list: List[int] = []
        for req in reqs:
            for j in range(req.extend_input_len):
                pos_list.append(req.kv_committed_len + j)
        pos_t = torch.tensor(pos_list, dtype=torch.long, device=DEVICE).unsqueeze(0)

        # ── Step 5: build qo_indptr, kv_indptr, kv_last_page_lens ───────
        # With page packing: len(slot_indices) == ceil(total_committed / P).
        B = len(reqs)
        qo_indptr_list  = [0]
        num_pages_list  = []
        kv_last_pg_list = []

        for req in reqs:
            qo_indptr_list.append(qo_indptr_list[-1] + req.extend_input_len)
            total_committed = req.kv_committed_len + req.extend_input_len
            n_pages = len(req.slot_indices)   # == ceil(total_committed / P) by invariant
            num_pages_list.append(n_pages)
            last_fill = total_committed % P
            kv_last_pg_list.append(last_fill if last_fill != 0 else P)

        qo_indptr_t       = torch.tensor(qo_indptr_list,  dtype=torch.int32, device=DEVICE)
        num_pages_t        = torch.tensor(num_pages_list,  dtype=torch.int32, device=DEVICE)
        kv_last_page_lens  = torch.tensor(kv_last_pg_list, dtype=torch.int32, device=DEVICE)
        req_pool_idx_t     = torch.tensor(
            [r.req_pool_idx for r in reqs], dtype=torch.int32, device=DEVICE
        )

        # kv_indptr via cumsum
        kv_indptr = torch.zeros(B + 1, dtype=torch.int32, device=DEVICE)
        torch.cumsum(num_pages_t, dim=0, out=kv_indptr[1:])

        # kv_indices via Triton kernel (reads req_to_token on-GPU)
        total_pages_batch = int(num_pages_t.sum().item())
        kv_indices = torch.empty(total_pages_batch, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            self.req_to_token_pool.req_to_token,
            req_pool_idx_t,
            num_pages_t,
            kv_indptr,
            None,
            kv_indices,
            self.req_to_token_pool.req_to_token.shape[1],
        )

        # ── Step 6: FlashInfer paged prefill plan ─────────────────────────
        self._extend_wrapper.begin_forward(
            qo_indptr_t,
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            P,
            causal=True,
            q_data_type = DTYPE,
        )

        ctx = ExtendKVCtx(
            wrapper     = self._extend_wrapper,
            k_pool      = self.kv_pool.k_pool,
            v_pool      = self.kv_pool.v_pool,
            qo_indptr   = qo_indptr_list,
            write_infos = write_infos,
            page_size   = P,
        )

        # ── Step 7: forward pass ─────────────────────────────────────────
        # attention_mask=None: FlashInfer handles causal internally.
        with torch.no_grad():
            logits = self.model(
                ids_t,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_t,
            )   # [1, total_tokens, vocab]

        self._extend_wrapper.end_forward()

        # ── Step 8: update state, sample first token for last-chunk reqs ─
        for i, req in enumerate(reqs):
            req.kv_committed_len += req.extend_input_len

            if not req.is_last_chunk:
                # More chunks to process — stay in PREFILLING state.
                req.status = ReqStatus.PREFILLING
                continue

            # Last chunk: sample the first output token.
            req.t_first_token = time.perf_counter()
            last_tok_pos = qo_indptr_list[i + 1] - 1   # last token's logit index
            next_tok = self._sample(logits[0, last_tok_pos], req.temperature)
            req.output_ids.append(next_tok)

            if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
                req.status   = ReqStatus.FINISHED
                req.t_finish = time.perf_counter()
                self.kv_pool.free(req.slot_indices)
                self.req_to_token_pool.free(req.req_pool_idx)
            else:
                req.status = ReqStatus.RUNNING

    # ------------------------------------------------------------------
    # decode_step (unchanged from Layer 9)
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """One batched decode step. Unchanged from Layer 9."""
        if not reqs:
            return []

        B   = len(reqs)
        P   = self.page_size
        cfg = self.model.model.config

        seq_lens_list      = [len(r.input_ids) + len(r.output_ids) for r in reqs]
        token_offsets_list = [sl % P                                for sl in seq_lens_list]
        num_pages_list     = [len(r.slot_indices)                   for r in reqs]
        req_pool_idx_list  = [r.req_pool_idx                        for r in reqs]

        last_page_idx_list = []
        for i, req in enumerate(reqs):
            if token_offsets_list[i] == 0:
                new_page = self.kv_pool.alloc(1)[0]
                req.slot_indices.append(new_page)
                self.req_to_token_pool.req_to_token[
                    req.req_pool_idx, num_pages_list[i]
                ] = new_page
                last_page_idx_list.append(new_page)
                num_pages_list[i] += 1
            else:
                last_page_idx_list.append(req.slot_indices[-1])

        seq_lens_t       = torch.tensor(seq_lens_list,      dtype=torch.int32, device=DEVICE)
        token_offsets_t  = torch.tensor(token_offsets_list, dtype=torch.int32, device=DEVICE)
        num_pages_t      = torch.tensor(num_pages_list,     dtype=torch.int32, device=DEVICE)
        req_pool_idx_t   = torch.tensor(req_pool_idx_list,  dtype=torch.int32, device=DEVICE)
        last_page_idx_t  = torch.tensor(last_page_idx_list, dtype=torch.int64, device=DEVICE)
        token_offsets_i64 = token_offsets_t.to(torch.int64)

        kv_last_page_lens = token_offsets_t + 1

        self._kv_indptr_buf[0] = 0
        torch.cumsum(num_pages_t, dim=0, out=self._kv_indptr_buf[1 : B + 1])
        kv_indptr = self._kv_indptr_buf[: B + 1]

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

        pos_ids  = seq_lens_t.unsqueeze(1).to(torch.long)
        last_toks = torch.tensor(
            [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
        )

        self._decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            cfg.num_attention_heads,
            cfg.num_key_value_heads,
            cfg.head_dim,
            P,
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

        with torch.no_grad():
            logits = self.model(
                last_toks,
                attention_mask = None,
                kv_cache       = ctx,
                position_ids   = pos_ids,
            )

        self._decode_wrapper.end_forward()

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
