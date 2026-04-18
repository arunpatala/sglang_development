"""
Layer 5 — ModelRunner: GPU side of the scheduler.

Two entry points called by the Scheduler:

  prefill(req)
  ────────────
  Runs a single B=1 forward pass for a newly arrived request.
  Populates req.kv_cache (PerReqKVCache) with the full prompt's K/V.
  Samples the first output token and sets req.t_first_token.

  decode_step(reqs) → List[Req]
  ─────────────────────────────
  One decode step for ALL currently running requests simultaneously.

  Key challenge: requests have different KV lengths because they
  arrived at different times.  We solve this with BatchedKVCache:

    1. Find max KV length across all active requests.
    2. Build attention_mask [B, max_kv_len+1]:
         left zeros for padding, ones for real tokens + new token.
    3. Build per-example position_ids [B, 1]:
         each request is at its own next position (kv_len_i).
    4. Create BatchedKVCache: lazily pads + stacks per-request K/V
         into [B, n_kv, max_kv_len, dim] for each layer.
    5. Run model forward → logits [B, 1, vocab].
    6. BatchedKVCache.write_back(): appends the new token's K/V to
         each request's PerReqKVCache.
    7. Sample next tokens, update output_ids, mark finished requests.

  Returns newly finished requests so the scheduler can resolve their
  futures and remove them from the running set.
"""

import logging
import sys
import time
from pathlib import Path
from typing import List

import torch

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import BatchedKVCache, PerReqKVCache
from model import Qwen3ForCausalLM
from request import Req, ReqStatus
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)

DEVICE = "cuda"
DTYPE  = torch.bfloat16


class ModelRunner:

    def __init__(self, model_path: str) -> None:
        logger.info(f"ModelRunner: loading model {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=DTYPE)

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
    # Prefill — B=1 per request
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
        with torch.no_grad():
            logits = self.model(ids, attention_mask=mask, kv_cache=kv, position_ids=pos)

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
    # Decode step — B=N for all running requests
    # ------------------------------------------------------------------

    def decode_step(self, reqs: List[Req]) -> List[Req]:
        """
        One decode step for all running requests simultaneously.
        Returns newly finished requests.

        Uses BatchedKVCache to pad different-length KV caches into a
        single rectangular [B, heads, max_kv_len, dim] tensor so that
        F.sdpa can process the whole batch in one kernel call.
        """
        if not reqs:
            return []

        B       = len(reqs)
        kv_lens = [r.kv_cache.get_seq_length() for r in reqs]
        max_kv  = max(kv_lens)

        # ── [B, 1] input: last generated token ───────────────────────
        last_toks = torch.tensor(
            [[r.output_ids[-1]] for r in reqs], dtype=torch.long, device=DEVICE
        )

        # ── [B, max_kv+1] attention mask ─────────────────────────────
        # KV is left-padded to max_kv; real tokens are on the right.
        # attn_mask[i, j] = 0 for left padding, 1 for real KV + new token.
        attn_mask = torch.zeros(B, max_kv + 1, dtype=torch.long, device=DEVICE)
        for i, kv_len in enumerate(kv_lens):
            attn_mask[i, max_kv - kv_len:] = 1   # real KV + new token slot

        # ── [B, 1] position IDs — per-request ────────────────────────
        # Each request is at its own next position (kv_len_i), NOT the
        # shared max_kv_len + step offset that would give wrong RoPE.
        pos_ids = torch.tensor(
            [[kv_len] for kv_len in kv_lens], dtype=torch.long, device=DEVICE
        )

        # ── Batched forward with padded KV cache ──────────────────────
        batch_kv = BatchedKVCache(reqs, max_kv)
        with torch.no_grad():
            logits = self.model(
                last_toks,
                attention_mask=attn_mask,
                kv_cache=batch_kv,
                position_ids=pos_ids,
            )   # [B, 1, vocab]

        # Write new K/V tokens back to per-request caches
        batch_kv.write_back()

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
