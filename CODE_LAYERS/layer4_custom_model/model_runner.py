"""
Layer 4 — Model Runner: same generate_batch logic as Layer 3 but using our
own Qwen3ForCausalLM instead of AutoModelForCausalLM.

Key differences from layer3/model.py:
  1. Model is loaded via Qwen3ForCausalLM.from_pretrained()
     instead of AutoModelForCausalLM.from_pretrained().
  2. forward() returns logits directly (not a ModelOutput namedtuple).
  3. KVCache is our clean implementation — mutated in-place by attention
     layers, no need to do `past_kv = out.past_key_values` each step.
  4. No use_cache=True flag — our model always uses the kv_cache arg.

Everything else (generate_batch loop, left-padding, finished mask,
attention_mask extension) is identical to Layer 3.
"""

import logging
import sys
import time
from pathlib import Path

import torch

# Add layer directory to path so `from model import ...` resolves.
sys.path.insert(0, str(Path(__file__).parent))

from kv_cache import KVCache
from model import Qwen3ForCausalLM
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BatchedModel:

    def __init__(self, model_path: str):
        logger.info(f"Loading model: {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id    = self.tokenizer.eos_token_id
        self.pad_id    = self.tokenizer.pad_token_id

        # ── Key change: our own model init ───────────────────────────────
        self.model = Qwen3ForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
        )

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")
        logger.info(
            f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _sample_batch(
        self,
        logits: torch.Tensor,   # [B, vocab_size]
        temperature: float,
    ) -> torch.Tensor:           # [B]
        if temperature == 0.0:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    # ------------------------------------------------------------------
    # Single request (delegates to generate_batch with B=1)
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
        max_prompt_len = input_ids.shape[1]
        # Tensor version needed for per-example position_ids arithmetic.
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")
        logger.info(
            f"batch_size={B}  max_prompt_len={max_prompt_len}  "
            f"prompt_lens={prompt_lens_list}"
        )

        # ── Prefill ───────────────────────────────────────────────────
        # One forward pass populates the KV cache for all B prompts.
        # kv is mutated in-place by each attention layer — no need to
        # re-assign after the call (unlike HuggingFace's past_key_values).
        #
        # position_ids fix: without per-example position_ids, real tokens in
        # shorter (padded) prompts get wrong RoPE positions.  Compute from
        # attention_mask so each real token gets position 0..L_i-1.
        prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)  # [B, max_len]

        kv = KVCache()
        t_prefill = time.perf_counter()
        with torch.no_grad():
            logits = self.model(
                input_ids,
                attention_mask=attention_mask,
                kv_cache=kv,
                position_ids=prefill_pos,
            )
        ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
        logger.info(f"after prefill: {kv}  ttft={ttft_ms}ms")

        # Sample first token from the LAST position of each sequence.
        next_tokens = self._sample_batch(logits[:, -1, :], temperature)  # [B]

        # ── Decode loop ───────────────────────────────────────────────
        finished   = next_tokens == self.eos_id
        generated  = [[] for _ in range(B)]
        step_times: list[float] = []
        decode_step = 0   # tracks which decode step we're on for position_ids

        for i, tok in enumerate(next_tokens.tolist()):
            if not finished[i]:
                generated[i].append(tok)

        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            t_step = time.perf_counter()

            # Feed [B, 1].  Finished requests get pad_token so they don't
            # corrupt the cache but their output is discarded.
            current = next_tokens.unsqueeze(1)
            current = torch.where(
                finished.unsqueeze(1),
                torch.full_like(current, self.pad_id),
                current,
            )

            # Extend attention mask by one real position for every request.
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
                dim=1,
            )

            # Per-example decode position: prompt_len_i + decode_step.
            # Avoids using the shared KV cache length (max_prompt_len + step)
            # which would give wrong positions for shorter padded prompts.
            decode_pos = (prompt_lens + decode_step).unsqueeze(1)  # [B, 1]
            decode_step += 1

            with torch.no_grad():
                logits = self.model(
                    current,
                    attention_mask=attention_mask,
                    kv_cache=kv,
                    position_ids=decode_pos,
                )

            next_tokens = self._sample_batch(logits[:, -1, :], temperature)
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

        total_out = sum(len(g) for g in generated)
        logger.info(
            f"DONE batch_size={B} total_output_tokens={total_out} "
            f"latency={latency_ms}ms ttft={ttft_ms}ms tpot={tpot_ms}ms "
            f"tok/s={round(total_out / (latency_ms / 1000), 1)}"
        )
        return results
