"""
Layer 4A — Model Runner: same generate_batch logic as Layer 3 but loading
the model via our own Qwen3ForCausalLM instead of AutoModelForCausalLM.

Diff from layer3/model_runner.py:
  1. Import:  Qwen3ForCausalLM  instead of AutoModelForCausalLM
  2. Init:    Qwen3ForCausalLM.from_pretrained(path, dtype=...)
             instead of AutoModelForCausalLM.from_pretrained(path, torch_dtype=...)
  3. KVCache (Layer 3's) is passed as past_key_values — HF's model calls
     kv.update() in-place, so there is no past_kv re-assignment after each
     forward call (unlike Layer 3's `past_kv = out.past_key_values`).

Everything else — tokenizer, left-padding, cumsum position IDs, pad injection,
mask extension, finished mask, sample_batch, server.py, benchmark.py — is
unchanged from Layer 3.

What Layer 4B adds on top of 4A:
  - Our own Qwen3Model (RMSNorm, RoPE, Attention, MLP, DecoderLayer)
  - KVCache updated with our own attention interface (layer_idx first, not last)
"""

import logging
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import KVCache
from model import Qwen3ForCausalLM
from sampling import sample_batch
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BatchedModel:

    def __init__(self, model_path: str):
        logger.info(f"Loading model: {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        # ── Key change from Layer 3: our own from_pretrained ─────────────
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
        )

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")

    # ------------------------------------------------------------------
    # Single request
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        return self.generate_batch(
            batch_messages=[messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )[0]

    # ------------------------------------------------------------------
    # Batched generation
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_messages: list[list[dict]],
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> list[dict]:
        B = len(batch_messages)
        t0 = time.perf_counter()

        # ── Tokenise ──────────────────────────────────────────────────
        input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(
            batch_messages
        )
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")

        # ── Prefill ───────────────────────────────────────────────────
        # cumsum gives per-token position IDs that correct for left-padding,
        # identical to the fix introduced in Layer 3.
        prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

        # KVCache implements HF's DynamicCache interface and is updated in-place
        # by HF's attention layers — no need to re-assign after the call.
        kv = KVCache()
        t_prefill = time.perf_counter()
        with torch.no_grad():
            logits, _ = self.model(
                input_ids,
                attention_mask=attention_mask,
                past_key_values=kv,
                position_ids=prefill_pos,
            )
        ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

        next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]

        # ── Decode loop ───────────────────────────────────────────────
        finished   = next_tokens == self.eos_id
        generated  = [[] for _ in range(B)]
        step_times: list[float] = []
        decode_step = 0

        for i, tok in enumerate(next_tokens.tolist()):
            if not finished[i]:
                generated[i].append(tok)

        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            t_step = time.perf_counter()

            current = next_tokens.unsqueeze(1)
            current = torch.where(
                finished.unsqueeze(1),
                torch.full_like(current, self.pad_id),
                current,
            )

            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
                dim=1,
            )

            decode_pos  = (prompt_lens + decode_step).unsqueeze(1)  # [B, 1]
            decode_step += 1

            with torch.no_grad():
                logits, _ = self.model(
                    current,
                    attention_mask=attention_mask,
                    past_key_values=kv,
                    position_ids=decode_pos,
                )

            next_tokens = sample_batch(logits[:, -1, :], temperature)
            step_times.append(time.perf_counter() - t_step)

            newly_finished = next_tokens == self.eos_id
            for i, tok in enumerate(next_tokens.tolist()):
                if not finished[i] and not newly_finished[i]:
                    generated[i].append(tok)
            finished = finished | newly_finished

        # ── Build results ─────────────────────────────────────────────
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        tpot_ms = (
            round((sum(step_times) / len(step_times)) * 1000, 1)
            if step_times else ttft_ms
        )
        texts = self.tokenizer.decode_batch(generated)

        results = []
        for i in range(B):
            results.append({
                "text": texts[i],
                "prompt_tokens": prompt_lens_list[i],
                "completion_tokens": len(generated[i]),
                "latency_ms": latency_ms,
                "ttft_ms": ttft_ms,
                "tpot_ms": tpot_ms,
            })
        return results
