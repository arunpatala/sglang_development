"""
Layer 7 — ModelRunner: GPU side of the scheduler.

Two entry points called by the Scheduler:

  prefill(req)                               ← allocates pool slots, writes K/V
  ────────────
  1. Allocate prompt_len slots from KVPool.
  2. Store slot indices on req.slot_indices.
  3. Run B=1 forward with PrefillKVCtx — each attention layer writes K/V
     into the assigned pool slots (F.sdpa for the attention itself).
  4. Sample first output token.

  decode_step(reqs) → List[Req]             ← zero-copy FlashInfer decode
  ─────────────────────────────
  1. Allocate 1 new slot per active request.
  2. Build kv_indptr + kv_indices from each req's slot_indices + new slot.
  3. Call FlashInfer begin_forward once with the index arrays.
  4. Run batched forward with DecodeKVCtx — each attention layer writes new
     token K/V to new_slots, then FlashInfer reads full history from pool.
  5. end_forward, append new slots to req.slot_indices.
  6. Sample next tokens; free slots for finished requests.

What changes vs Layer 6 (PackedKVCache):
  Layer 6:  each decode step gathers all PerReqKVCache tensors into a packed
            buffer — O(total_kv_tokens) float copy every step.
  Layer 7:  K/V stay in the pre-allocated pool forever. The only per-step
            work is building kv_indices (integer array, O(total_kv_tokens)
            but integer not float) and writing 1 new row per request.
"""

import logging
import sys
import time
from itertools import accumulate
from pathlib import Path
from typing import List

import torch
import flashinfer

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import DecodeKVCtx, KVPool, PrefillKVCtx
from model import Qwen3ForCausalLM
from request import Req, ReqStatus
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# Reserve this fraction of free GPU memory (after model load) for KVPool.
_KV_MEMORY_FRACTION = 0.85
# FlashInfer workspace size.
_WORKSPACE_MB = 256


class ModelRunner:

    def __init__(self, model_path: str) -> None:
        logger.info(f"ModelRunner: loading model from {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)

        # ── KVPool: allocate after weights are loaded so we know free memory ─
        cfg = self.model.model.config
        free_bytes, _ = torch.cuda.mem_get_info()
        bytes_per_token = (
            cfg.num_hidden_layers * 2      # K + V
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

        # FlashInfer workspace — reused every decode step.
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )

        logger.info(
            f"ModelRunner ready in {time.perf_counter()-t0:.1f}s  "
            f"GPU={torch.cuda.memory_allocated()/1024**2:.0f} MB  "
            f"KVPool slots={max_tokens}"
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
    # Prefill — B=1, allocates pool slots, writes K/V during forward
    # ------------------------------------------------------------------

    def prefill(self, req: Req) -> None:
        """
        Run a full B=1 prefill forward.

        Allocates prompt_len slots from KVPool, runs the model with
        PrefillKVCtx so each attention layer writes K/V to the pool.
        Samples the first output token.
        """
        prompt_len = len(req.input_ids)

        # Allocate physical pool slots for every prompt token.
        slots = self.kv_pool.alloc(prompt_len)
        req.slot_indices = slots   # store on req for decode steps to extend

        ids  = torch.tensor([req.input_ids], device=DEVICE)          # [1, L]
        mask = torch.ones(1, prompt_len, dtype=torch.long, device=DEVICE)
        pos  = torch.arange(prompt_len, device=DEVICE).unsqueeze(0)  # [1, L]

        ctx = PrefillKVCtx(slots, self.kv_pool)
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask, kv_cache=ctx, position_ids=pos)

        req.t_first_token = time.perf_counter()

        next_tok = self._sample(logits[0, -1], req.temperature)
        req.output_ids.append(next_tok)

        if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
            req.status   = ReqStatus.FINISHED
            req.t_finish = time.perf_counter()
            self.kv_pool.free(req.slot_indices)   # return slots immediately
        else:
            req.status = ReqStatus.RUNNING

        logger.debug(
            f"prefill rid={req.rid[:8]} prompt_len={prompt_len} "
            f"slots={slots[:3]}… ttft={req.ttft_ms:.1f}ms"
        )

    # ------------------------------------------------------------------
    # Decode step — B=N, zero-copy FlashInfer paged attention
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One batched decode step for all running requests.
        Returns newly finished requests.

        Steps:
          1. Allocate 1 new slot per request (for the new decode token).
          2. Build kv_indptr + kv_indices:
               For each request: [historical_slots..., new_slot]
               kv_indptr[i+1] = kv_indptr[i] + len(req.slot_indices) + 1
          3. begin_forward (once, covers all 28 layers).
          4. Forward pass — DecodeKVCtx.store() writes new K/V to pool,
             FlashInfer.forward() reads full history from pool via kv_indices.
          5. end_forward, append new_slot to req.slot_indices.
          6. Sample, free finished requests' slots.
        """
        if not reqs:
            return []

        B       = len(reqs)
        cfg     = self.model.model.config

        # ── Allocate one new slot per request ─────────────────────────
        new_slots = [self.kv_pool.alloc(1)[0] for _ in reqs]

        # ── Build kv_indptr and kv_indices ────────────────────────────
        # Each request's KV = historical_slots + new_slot (total = kv_len+1)
        kv_lens_plus1 = [len(r.slot_indices) + 1 for r in reqs]
        kv_indptr_list = [0] + list(accumulate(kv_lens_plus1))

        kv_indices_list: List[int] = []
        for i, req in enumerate(reqs):
            kv_indices_list.extend(req.slot_indices)   # historical
            kv_indices_list.append(new_slots[i])        # new token (will be written below)

        kv_indptr = torch.tensor(kv_indptr_list,  dtype=torch.int32, device=DEVICE)
        kv_indices = torch.tensor(kv_indices_list, dtype=torch.int32, device=DEVICE)
        # page_size=1 → last page always has exactly 1 token
        kv_last_page_lens = torch.ones(B, dtype=torch.int32, device=DEVICE)

        # ── Per-request position IDs ───────────────────────────────────
        # Each request is at position = its current kv length (before new token).
        pos_ids = torch.tensor(
            [[len(r.slot_indices)] for r in reqs], dtype=torch.long, device=DEVICE
        )

        # ── Input tokens ───────────────────────────────────────────────
        last_toks = torch.tensor(
            [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
        )

        # ── FlashInfer plan ────────────────────────────────────────────
        # BatchDecodeWithPagedKVCacheWrapper — the proper decode kernel.
        # use_tensor_cores=False: Qwen3-0.6B has GQA group = 16Q/8KV = 2,
        # below the threshold of 4 where tensor cores help for decode.
        # The pre-compiled AOT kernel (bf16, head_dim=128, no-swa, no-logits-cap)
        # is available in the flashinfer_jit_cache package and loads without JIT.
        decode_wrapper = flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            self._workspace, "NHD", use_tensor_cores=False
        )
        decode_wrapper.begin_forward(
            kv_indptr,
            kv_indices,
            kv_last_page_lens,
            cfg.num_attention_heads,    # num_qo_heads
            cfg.num_key_value_heads,    # num_kv_heads
            cfg.head_dim,
            1,                          # page_size
            data_type    = DTYPE,
            q_data_type  = DTYPE,
        )

        new_slots_t = torch.tensor(new_slots, dtype=torch.int64, device=DEVICE)
        ctx = DecodeKVCtx(
            wrapper   = decode_wrapper,
            k_pool    = self.kv_pool.k_pool,
            v_pool    = self.kv_pool.v_pool,
            new_slots = new_slots_t,
        )

        # ── Batched forward ────────────────────────────────────────────
        with torch.no_grad():
            logits = self.model(
                last_toks,
                attention_mask = None,   # FlashInfer uses kv_indices, not a mask
                kv_cache       = ctx,
                position_ids   = pos_ids,
            )   # [B, 1, vocab]

        decode_wrapper.end_forward()

        # ── Append new slots to request history ────────────────────────
        for i, req in enumerate(reqs):
            req.slot_indices.append(new_slots[i])

        # ── Sample + handle finished requests ──────────────────────────
        newly_finished: List[Req] = []
        for i, req in enumerate(reqs):
            next_tok = self._sample(logits[i, -1], req.temperature)
            req.output_ids.append(next_tok)

            if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
                req.status   = ReqStatus.FINISHED
                req.t_finish = time.perf_counter()
                newly_finished.append(req)
                # Return all pool slots to the free list.
                self.kv_pool.free(req.slot_indices)

        return newly_finished

    # ------------------------------------------------------------------
    # Token → text
    # ------------------------------------------------------------------

    def decode_output(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids)
