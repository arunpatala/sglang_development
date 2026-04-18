"""
Layer 2 — KVCacheModel: manual decode loop WITH KV cache.

The only file that changes vs Layer 1. server.py and benchmark.py are identical.

What changes in the loop (two lines):

    Layer 1:
        out = model(input_ids=ids, use_cache=False)
        # ids grows by cat every step → O(seq²) attention

    Layer 2:
        out = model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        # current_token is always shape [1,1] → O(1) attention per decode step

How past_key_values works:
    - After the prefill forward pass, out.past_key_values is a tuple of length
      n_layers. Each element is a (K, V) pair of tensors shaped:
          [batch, n_heads, seq_len, head_dim]
    - On each decode step we pass in ONE new token. The model appends its K/V
      to the cache internally and returns the updated cache.
    - The attention kernel attends the new token over ALL cached K/V — giving
      the same result as a full forward pass but only doing O(seq_len) work
      instead of O(seq_len²).

Cost comparison:
    Layer 1  prefill: O(L²)   decode step k: O((L+k)²)   total: O(L²·T + T³/6)
    Layer 2  prefill: O(L²)   decode step k: O(L+k)       total: O(L²   + L·T)

    For L=1000, T=128: Layer 1 ≈ 128M ops, Layer 2 ≈ 1.13M ops per step.
"""

import logging
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv_cache import KVCache

logger = logging.getLogger(__name__)


class KVCacheModel:
    """
    Wraps a HuggingFace causal LM and runs the autoregressive decode loop
    with past_key_values reuse. Only model.py changes vs Layer 1.
    """

    def __init__(self, model_path: str):
        logger.info(f"Loading model: {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model.eval()

        self.eos_id = self.tokenizer.eos_token_id

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")
        logger.info(
            f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Sampling helper  (identical to Layer 1)
    # ------------------------------------------------------------------

    def _sample_next_token(
        self,
        logits: torch.Tensor,   # shape: [vocab_size]
        temperature: float,
    ) -> int:
        if temperature == 0.0:
            return int(logits.argmax(dim=-1).item())
        if temperature != 1.0:
            logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        return int(torch.multinomial(probs, num_samples=1).item())

    # ------------------------------------------------------------------
    # Public API — called by server.py  (identical signature to Layer 1)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        t0 = time.perf_counter()

        # --- Step 1: apply chat template and tokenize  (same as Layer 1) ---
        formatted = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        input_ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
        prompt_tokens = input_ids.shape[1]
        logger.info(f"prompt_tokens={prompt_tokens}")

        # --- Step 2: prefill ---
        # Run the full prompt through the model once to build the KV cache.
        # We pass our own KVCache instance; HuggingFace's attention will call
        # past_kv.update(new_k, new_v, layer_idx) for each of the 28 layers.
        past_kv = KVCache()
        t_prefill = time.perf_counter()
        with torch.no_grad():
            out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

        past_kv = out.past_key_values          # same KVCache object, now populated
        logger.info(f"after prefill: {past_kv}")
        next_token_logits = out.logits[0, -1, :]
        next_token_id = self._sample_next_token(next_token_logits, temperature)
        ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

        generated_ids: list[int] = []
        step_times: list[float] = []

        if next_token_id == self.eos_id:
            text = ""
            tpot_ms = 0.0
        else:
            generated_ids.append(next_token_id)

            # --- Step 3: decode loop ---
            # Each iteration feeds ONE new token + the cached K/V.
            # The model extends the cache and returns the next logits.
            # current_token is always shape [batch=1, seq=1] — constant cost.
            for _ in range(max_new_tokens - 1):
                t_step = time.perf_counter()

                current_token = torch.tensor(
                    [[next_token_id]], dtype=torch.long, device="cuda"
                )

                with torch.no_grad():
                    # KEY CHANGE vs Layer 1:
                    #   input_ids = single new token (not growing sequence)
                    #   past_key_values = cached K/V from all previous steps
                    out = self.model(
                        input_ids=current_token,
                        past_key_values=past_kv,
                        use_cache=True,
                    )

                past_kv = out.past_key_values      # updated cache (one longer)
                next_token_logits = out.logits[0, -1, :]
                next_token_id = self._sample_next_token(next_token_logits, temperature)

                step_times.append(time.perf_counter() - t_step)

                if next_token_id == self.eos_id:
                    break

                generated_ids.append(next_token_id)

            tpot_ms = round(
                (sum(step_times) / len(step_times)) * 1000, 1
            ) if step_times else ttft_ms

            text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        completion_tokens = len(generated_ids)

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
