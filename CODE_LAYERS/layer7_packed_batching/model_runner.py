"""
Layer 6 — ModelRunner: GPU side of the scheduler.

Two entry points called by the Scheduler:

  prefill(req)                               ← unchanged from Layer 5
  ────────────
  Runs a single B=1 forward pass for a newly arrived request.
  Populates req.kv_cache (PerReqKVCache) with the full prompt's K/V.
  Samples the first output token and sets req.t_first_token.
  Uses F.sdpa via PerReqKVCache (same as Layer 5).

  decode_step(reqs) → List[Req]              ← NEW: uses PackedKVCache + FlashInfer
  ─────────────────────────────
  One decode step for ALL currently running requests simultaneously.

  Layer 5 approach (BatchedKVCache + F.sdpa):
    • Left-pads all KV caches to max_kv_len.
    • Builds [B, heads, max_kv_len, dim] rectangular tensor.
    • F.sdpa computes attention including the padded zero-columns.

  Layer 6 approach (PackedKVCache + FlashInfer):
    • Concatenates per-request KVs into a single ragged tensor
      (no padding, no zero-columns).
    • kv_indptr[i..i+1] tells FlashInfer which slice belongs to req i.
    • FlashInfer's BatchPrefillWithRaggedKVCacheWrapper processes only
      real tokens — no wasted compute.

  What stays the same vs Layer 5:
    • prefill() is identical (B=1, F.sdpa, PerReqKVCache).
    • Position IDs per request (kv_len_i, not shared max_kv_len + step).
    • write_back() appends the new token to each req's PerReqKVCache.
    • Sampling and result handling are identical.

  What changes vs Layer 5:
    • No attention_mask passed to model (FlashInfer handles masking internally).
    • PackedKVCache.plan() must be called once before model forward.
    • PackedKVCache.end_forward() called after the forward pass.
    • Workspace tensor (256 MB) allocated once at ModelRunner init and
      reused every decode step.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from forward_batch import ForwardBatch, ForwardMode
from kv_cache import PackedKVCache, PerReqKVCache
from model import Qwen3ForCausalLM
from model.config import AttnBackend
from request import Req, ReqStatus
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16

# 256 MB workspace shared across all decode steps.
# FlashInfer uses this for internal temp buffers during kernel planning.
_WORKSPACE_MB = 256


class ModelRunner:

    def __init__(
        self,
        model_path:   str,
        attn_backend: AttnBackend = AttnBackend.FLASHINFER,
    ) -> None:
        logger.info(f"ModelRunner: loading model {model_path}  backend={attn_backend.value}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path, dtype=DTYPE, attn_backend=attn_backend
        )

        # Pre-allocate FlashInfer workspace once; reused every decode step.
        self._workspace = torch.empty(
            _WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE
        )

        logger.info(
            f"Model ready in {time.perf_counter()-t0:.1f}s  "
            f"GPU={torch.cuda.memory_allocated()/1024**2:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample(self, logits: torch.Tensor, temperature: float) -> int:
        """Sample one token from logits [vocab]."""
        if temperature == 0.0:
            return int(logits.argmax())
        probs = torch.softmax(logits / temperature, dim=-1)
        return int(torch.multinomial(probs, num_samples=1))

    # ------------------------------------------------------------------
    # Prefill — B=1 per request (identical to Layer 5)
    # ------------------------------------------------------------------

    def prefill(self, req: Req) -> None:
        """
        Run a full prefill forward for one request.

        Populates req.kv_cache with the prompt's K/V cache.
        Appends the first generated token to req.output_ids.
        Sets req.t_first_token and marks FINISHED if EOS immediately.
        """
        ids  = torch.tensor([req.input_ids], device=DEVICE)        # [1, L]
        mask = torch.ones(1, len(req.input_ids), dtype=torch.long, device=DEVICE)
        pos  = torch.arange(len(req.input_ids), device=DEVICE).unsqueeze(0)  # [1, L]

        kv = PerReqKVCache()
        fb = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=kv, attention_mask=mask)
        with torch.no_grad():
            logits = self.model(ids, forward_batch=fb, position_ids=pos)

        req.kv_cache      = kv
        req.t_first_token = time.perf_counter()

        next_tok = self._sample(logits[0, -1], req.temperature)
        req.output_ids.append(next_tok)

        if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
            req.status   = ReqStatus.FINISHED
            req.t_finish = time.perf_counter()
        else:
            req.status = ReqStatus.RUNNING

        logger.debug(
            f"prefill rid={req.rid[:8]} prompt_len={req.prompt_len} "
            f"first_tok={next_tok} ttft={req.ttft_ms:.1f}ms"
        )

    # ------------------------------------------------------------------
    # Decode step — B=N with PackedKVCache + FlashInfer
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One decode step for all running requests simultaneously.
        Returns newly finished requests.

        Uses PackedKVCache to pack different-length KV caches into a
        single ragged tensor so FlashInfer can attend without padding.

        Steps:
          1. Build last_toks [B, 1] — each request's latest token.
          2. Build pos_ids   [B, 1] — per-request next position.
          3. Create PackedKVCache with the pre-allocated workspace.
          4. PackedKVCache.plan() — one call for all 28 layers.
          5. model forward with attention_mask=None (FlashInfer handles it).
          6. PackedKVCache.write_back() — append new token to per-req caches.
          7. PackedKVCache.end_forward() — release FlashInfer state.
          8. Sample, update output_ids, mark finished requests.
        """
        if not reqs:
            return []

        B       = len(reqs)
        kv_lens = [r.kv_cache.get_seq_length() for r in reqs]

        # ── [B, 1] input: last generated token ───────────────────────
        last_toks = torch.tensor(
            [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
        )

        # ── [B, 1] position IDs — per-request ────────────────────────
        # Each request is at its own next position (kv_len_i), NOT the
        # shared max_kv_len offset that would give wrong RoPE for shorter seqs.
        pos_ids = torch.tensor(
            [[kv_len] for kv_len in kv_lens], dtype=torch.long, device=DEVICE
        )

        # ── PackedKVCache + FlashInfer ────────────────────────────────
        pack_kv = PackedKVCache(reqs, self._workspace)

        # Plan once: tell FlashInfer the head counts + dtype for this step.
        cfg = self.model.model.config
        pack_kv.plan(
            num_q_heads  = cfg.num_attention_heads,
            num_kv_heads = cfg.num_key_value_heads,
            head_dim     = cfg.head_dim,
            dtype        = DTYPE,
        )

        # attention_mask=None — FlashInfer uses kv_indptr, not a mask tensor.
        fb = ForwardBatch(mode=ForwardMode.DECODE, kv_cache=pack_kv, attention_mask=None)
        with torch.no_grad():
            logits = self.model(last_toks, forward_batch=fb, position_ids=pos_ids)
        # [B, 1, vocab]

        # Write new K/V tokens back to per-request caches.
        pack_kv.write_back()
        pack_kv.end_forward()

        # ── Sample, update state ──────────────────────────────────────
        newly_finished: List[Req] = []
        for i, req in enumerate(reqs):
            next_tok = self._sample(logits[i, -1], req.temperature)
            req.output_ids.append(next_tok)

            if next_tok == self.eos_id or req.output_len >= req.max_new_tokens:
                req.status   = ReqStatus.FINISHED
                req.t_finish = time.perf_counter()
                newly_finished.append(req)

        return newly_finished

    # ------------------------------------------------------------------
    # Token → text helper (used by scheduler to build results)
    # ------------------------------------------------------------------

    def decode_output(self, req: Req) -> str:
        return self.tokenizer.decode(req.output_ids)
