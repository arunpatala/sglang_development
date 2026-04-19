"""
Layer 8 — ModelRunner: req_to_token table + Triton kv_indices kernel.

Builds on Layer 7 (paged KV pool) with two key changes:

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1.  ReqToTokenPool  (GPU-resident 2D lookup table)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    req_to_token[req_pool_idx, token_pos] = physical_slot_idx

Allocated once at startup: [max_concurrent_reqs, max_context_len] int32.

At prefill:   req_to_token[req_pool_idx, 0:L]  = slot_indices_tensor
At decode:    req_to_token[req_pool_indices, seq_lens] = new_slots_tensor
              (vectorised, single GPU kernel scatter — no Python loop over slots)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
2.  create_flashinfer_kv_indices_triton  (Triton kernel, runs entirely on GPU)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Reads req_to_token rows and writes the flat kv_indices tensor without any
CPU→GPU copy of slot data.  One threadblock per request, runs in parallel.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Additional micro-optimisations vs Layer 7
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  • BatchDecodeWithPagedKVCacheWrapper created once at startup (not every step).
  • kv_indptr built via torch.cumsum on GPU (not itertools.accumulate on CPU).
  • kv_indptr buffer pre-allocated [max_batch+1] and reused every step.
  • kv_last_page_lens pre-allocated all-ones and reused (never changes for
    page_size=1).
  • req_pool_indices and seq_lens small CPU→GPU copies (B ints) each step.
    The *slot data* (O(Σ kv_lens) ints) is never copied CPU→GPU again.

What still happens on CPU each step (unavoidable without further changes):
  • Python loop over reqs to read len(req.slot_indices) → seq_lens list
  • torch.tensor(seq_lens_list) → small CPU→GPU transfer (B ints)
  • torch.tensor(req_pool_indices_list) → small CPU→GPU transfer (B ints)
  • new_slots allocation (one pop per req from free_slots list)
  Layer 9 target: GPU-resident seq_lens tensor updated in-place.
"""

import logging
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

_KV_MEMORY_FRACTION  = 0.85    # fraction of post-model-load free VRAM for KVPool
_WORKSPACE_MB        = 256     # FlashInfer workspace
_MAX_CONCURRENT_REQS = 128     # rows in ReqToTokenPool


class ModelRunner:

    def __init__(self, model_path: str, max_context_len: int = 4096) -> None:
        logger.info(f"ModelRunner: loading model from {model_path}")
        t0 = time.perf_counter()

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
        max_tokens = int(free_bytes * _KV_MEMORY_FRACTION / bytes_per_token)

        self.kv_pool = KVPool(
            total_slots = max_tokens,
            n_layers    = cfg.num_hidden_layers,
            n_kv_heads  = cfg.num_key_value_heads,
            head_dim    = cfg.head_dim,
            dtype       = DTYPE,
        )

        # ── ReqToTokenPool ────────────────────────────────────────────────
        # 2D GPU table: [max_batch, max_context_len] int32.
        # The Triton kernel reads this on-device each decode step.
        self.req_to_token_pool = ReqToTokenPool(
            max_batch       = _MAX_CONCURRENT_REQS,
            max_context_len = max_context_len,
        )
        self._max_context_len = max_context_len

        # ── Pre-allocated decode-step buffers ─────────────────────────────
        # kv_indptr:        [max_batch+1] int32 — reused every step via in-place cumsum
        # kv_last_page_lens: [max_batch]  int32 — always 1 for page_size=1
        self._kv_indptr_buf = torch.zeros(
            _MAX_CONCURRENT_REQS + 1, dtype=torch.int32, device=DEVICE
        )
        self._kv_last_page_lens = torch.ones(
            _MAX_CONCURRENT_REQS, dtype=torch.int32, device=DEVICE
        )

        # ── FlashInfer workspace + decode wrapper ─────────────────────────
        # Created once at startup; reused every decode step.
        # use_tensor_cores=False: Qwen3-0.6B GQA group = 16Q/8KV = 2,
        # below the threshold of 4; tensor cores don't help and require JIT.
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )
        self._decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, "NHD", use_tensor_cores=False
        )

        logger.info(
            f"ModelRunner ready in {time.perf_counter()-t0:.1f}s  "
            f"GPU={torch.cuda.memory_allocated()/1024**2:.0f} MB  "
            f"KVPool slots={max_tokens}  "
            f"ReqToTokenPool [{_MAX_CONCURRENT_REQS}, {max_context_len}]"
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
    # Prefill — B=1, allocates pool slots + req_to_token row
    # ------------------------------------------------------------------

    def prefill(self, req: Req) -> None:
        """
        B=1 prefill.

        Allocates:
          • prompt_len KVPool slots → written to KVPool during forward
          • 1 ReqToTokenPool row    → req_to_token[row, 0:L] = slot_indices

        The req_to_token write happens here (before the forward) so the table
        is consistent for any decode step that follows.
        """
        prompt_len = len(req.input_ids)

        # Allocate physical pool slots for every prompt token.
        slots    = self.kv_pool.alloc(prompt_len)
        req.slot_indices = slots

        # Allocate a ReqToTokenPool row and write all slot indices at once.
        req.req_pool_idx = self.req_to_token_pool.alloc()
        slots_t = torch.tensor(slots, dtype=torch.int32, device=DEVICE)
        self.req_to_token_pool.req_to_token[req.req_pool_idx, :prompt_len] = slots_t

        ids  = torch.tensor([req.input_ids], device=DEVICE)
        mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
        pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)

        ctx = PrefillKVCtx(slots, self.kv_pool)
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
            f"req_pool_idx={req.req_pool_idx} ttft={req.ttft_ms:.1f}ms"
        )

    # ------------------------------------------------------------------
    # Decode step — GPU kv_indptr + Triton kv_indices
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One batched decode step for all running requests.

        Layer 8 changes vs Layer 7:
          1. req_to_token[req_pool_indices, seq_lens] = new_slots  (vectorised GPU write)
          2. kv_indptr built via torch.cumsum on GPU  (not itertools.accumulate on CPU)
          3. kv_indices built by Triton kernel on GPU (not Python list + CPU→GPU copy)
          4. decode_wrapper reused from __init__      (not re-created each step)
          5. kv_last_page_lens reused from __init__   (pre-allocated ones buffer)
        """
        if not reqs:
            return []

        B   = len(reqs)
        cfg = self.model.model.config

        # ── Allocate one new KVPool slot per request ───────────────────
        new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

        # ── Small CPU metadata → GPU  (unavoidable, but O(B) not O(Σ kv_lens)) ─
        # seq_lens = current history length (before new token)
        seq_lens_list    = [len(r.slot_indices)  for r in reqs]   # Python loop over B reqs
        req_pool_idx_list = [r.req_pool_idx      for r in reqs]

        seq_lens_t      = torch.tensor(seq_lens_list,    dtype=torch.int32, device=DEVICE)
        req_pool_idx_t  = torch.tensor(req_pool_idx_list, dtype=torch.int32, device=DEVICE)
        new_slots_t_i32 = torch.tensor(new_slots,         dtype=torch.int32, device=DEVICE)
        new_slots_t_i64 = new_slots_t_i32.to(torch.int64)

        # ── Write new slots into req_to_token on GPU ───────────────────
        # req_to_token[req_pool_idx_t[i], seq_lens_t[i]] = new_slots[i]
        # This is advanced (fancy) indexing: one scatter per request, in parallel.
        self.req_to_token_pool.req_to_token[req_pool_idx_t, seq_lens_t] = new_slots_t_i32

        # ── Build kv_indptr on GPU via cumsum ──────────────────────────
        # seq_lens_with_new[i] = seq_lens[i] + 1  (history + new token)
        seq_lens_with_new = seq_lens_t + 1    # GPU op
        self._kv_indptr_buf[0] = 0
        torch.cumsum(seq_lens_with_new, dim=0, out=self._kv_indptr_buf[1 : B + 1])
        kv_indptr = self._kv_indptr_buf[: B + 1]

        # ── Build kv_indices on GPU via Triton kernel ──────────────────
        # Total slots across all requests = sum of seq_lens_with_new.
        # We compute this sum on CPU since seq_lens_list is already there.
        total_kv = sum(s + 1 for s in seq_lens_list)
        kv_indices = torch.empty(total_kv, dtype=torch.int32, device=DEVICE)

        create_flashinfer_kv_indices_triton[(B,)](
            self.req_to_token_pool.req_to_token,    # [max_batch, max_ctx]
            req_pool_idx_t,                         # [B]
            seq_lens_with_new,                      # [B]  kv lengths per request
            kv_indptr,                              # [B+1]
            None,                                   # kv_start_idx (None = start at 0)
            kv_indices,                             # [total_kv] output
            self.req_to_token_pool.req_to_token.shape[1],  # stride (max_context_len)
        )

        kv_last_page_lens = self._kv_last_page_lens[:B]

        # ── Per-request position IDs ───────────────────────────────────
        pos_ids = seq_lens_t.unsqueeze(1).to(torch.long)   # [B, 1]

        # ── Input tokens (last generated token per request) ───────────
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
            1,                    # page_size
            data_type   = DTYPE,
            q_data_type = DTYPE,
        )

        ctx = DecodeKVCtx(
            wrapper   = self._decode_wrapper,
            k_pool    = self.kv_pool.k_pool,
            v_pool    = self.kv_pool.v_pool,
            new_slots = new_slots_t_i64,
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

        # ── Append new slots to request history ───────────────────────
        for i, req in enumerate(reqs):
            req.slot_indices.append(new_slots[i])

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
