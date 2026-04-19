"""
Layer 1 — NaiveModel: manual autoregressive decode loop, no KV cache.

This module owns everything model-related so that server.py stays clean.
The key difference from Layer 0 is that we write the decode loop ourselves
instead of calling model.generate(). Every step is visible.

The loop:
    for step in range(max_new_tokens):
        logits = model(full_sequence_so_far)   # full forward pass, every step
        next_token = sample(logits[-1])         # look only at last position
        append next_token to sequence
        if next_token == eos: break

This is identical in computation to Layer 0 (use_cache=False). Same O(seq²)
cost. But now we own the loop, so Layer 2 can add KV cache with a surgical
change inside this file alone — server.py stays untouched.
"""

import logging
import time
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger(__name__)


class NaiveModel:
    """
    Wraps a HuggingFace causal LM and exposes a single `generate` method
    that runs the autoregressive decode loop manually, one forward pass per
    token, with no KV cache.
    """

    _DTYPE_MAP = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }

    def __init__(self, model_path: str, dtype: str = "bfloat16", device: str = "cuda"):
        logger.info(f"Loading model: {model_path}")
        t0 = time.perf_counter()

        self.device = device
        weight_dtype = self._DTYPE_MAP.get(dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=weight_dtype,
        ).to(device)
        self.model.eval()

        self.eos_id = self.tokenizer.eos_token_id

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")
        logger.info(
            f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Sampling helper
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        logits: torch.Tensor,   # shape: [vocab_size]
        temperature: float,
    ) -> int:
        """
        Convert logits to a token id.

        temperature=1.0  → multinomial sampling (no scaling)
        temperature=0.0  → greedy (argmax)
        otherwise        → scale logits then sample
        """
        if temperature == 0.0:
            return int(logits.argmax(dim=-1).item())

        if temperature != 1.0:
            logits = logits / temperature

        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Public API — called by server.py
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],    # [{"role": "user", "content": "..."}]
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        """
        Run the manual autoregressive decode loop and return a result dict
        with the same keys as Layer 0's GenerateResponse.

        Returns:
            {
                "text": str,
                "prompt_tokens": int,
                "completion_tokens": int,
                "latency_ms": float,
            }
        """
        t0 = time.perf_counter()

        # --- Step 1: apply chat template and tokenize ---
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
        prompt_tokens = input_ids.shape[1]

        logger.info(f"prompt_tokens={prompt_tokens}")

        # --- Step 2: manual decode loop ---
        # `ids` grows by one token each iteration.
        # No past_key_values — every forward pass sees the full sequence from
        # the beginning. This is the defining cost of Layer 1 (same as L0).
        ids = input_ids
        generated_ids: list[int] = []
        step_times: list[float] = []   # wall time (seconds) for each decode step

        for step in range(max_new_tokens):
            t_step = time.perf_counter()

            with torch.no_grad():
                # Full forward pass over the entire current sequence.
                # use_cache=False prevents HuggingFace from secretly building
                # its own KV cache internally — we want the raw cost visible.
                out = self.model(input_ids=ids, use_cache=False)

            # out.logits shape: [batch=1, seq_len, vocab_size]
            # We only care about the last position's prediction.
            next_token_logits = out.logits[0, -1, :]   # [vocab_size]
            next_token_id = self._sample_next_token(next_token_logits, temperature)

            step_times.append(time.perf_counter() - t_step)

            if next_token_id == self.eos_id:
                break

            generated_ids.append(next_token_id)

            # Append the new token and go again.
            next_token_tensor = torch.tensor(
                [[next_token_id]], dtype=torch.long, device=self.device
            )
            ids = torch.cat([ids, next_token_tensor], dim=1)

        # --- Step 3: decode ---
        text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        completion_tokens = len(generated_ids)

        # TTFT: time until the first token was produced (step 0 = prefill + first sample).
        # In Layer 1 there is no separate prefill phase — the first forward pass runs
        # over the full prompt, so step_times[0] IS the prefill cost.
        ttft_ms = round(step_times[0] * 1000, 1) if step_times else 0.0

        # TPOT: average time per output token for steps 1..N (pure decode steps).
        # Step 0 is excluded because it includes prefill work.
        decode_steps = step_times[1:]
        tpot_ms = round(
            (sum(decode_steps) / len(decode_steps)) * 1000, 1
        ) if decode_steps else ttft_ms

        logger.info(
            f"DONE: completion_tokens={completion_tokens} "
            f"latency={latency_ms}ms "
            f"ttft={ttft_ms}ms "
            f"tpot={tpot_ms}ms "
            f"tok/s={round(completion_tokens / (latency_ms / 1000), 1)}"
        )

        return {
            "text": text,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "ttft_ms": ttft_ms,
            "tpot_ms": tpot_ms,
        }
