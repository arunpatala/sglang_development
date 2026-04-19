"""
Layer 3 — Static Batching: process multiple requests in one forward pass.

What changes vs Layer 2:
  - generate_batch() accepts B prompts, pads them to the same length,
    runs prefill as a single [B, max_prompt_len] forward pass, then
    decodes all B sequences simultaneously with [B, 1] inputs each step.

What stays the same:
  - kv_cache.py is unchanged — KVCache already stores [batch, heads, seq, dim]
  - generate() (single request) is unchanged — /generate endpoint still works

Key new concepts:
  - Left padding: all prompts padded from the left so the last real token
    is always at position -1. The attention_mask tells the model to ignore pads.
  - Finished mask: each request tracks whether it has emitted <eos>.
    Once finished, its next token is forced to pad_token_id and its output
    is excluded from results. The loop ends when ALL requests finish.
  - Padding waste: a batch of [10-token, 1000-token] prompts wastes 99% of
    prefill compute on the short request. This is the core flaw of static
    batching that continuous batching fixes.

GPU utilisation:
  - Layer 2: [1, 1] tensor each decode step → GPU ~5% utilised
  - Layer 3: [B, 1] tensor each decode step → B× more work per step,
    GPU utilisation climbs toward 100% as B increases.
"""

import logging
import time

import torch
from transformers import AutoModelForCausalLM

from kv_cache import KVCache
from sampling import sample_batch
from tokenizer import Tokenizer

logger = logging.getLogger(__name__)


class BatchedKVCacheModel:

    def __init__(self, model_path: str):
        logger.info(f"Loading model: {model_path}")
        t0 = time.perf_counter()

        self.tokenizer = Tokenizer(model_path)
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
        ).to("cuda")
        self.model.eval()

        logger.info(f"Model loaded in {time.perf_counter() - t0:.1f}s")
        logger.info(
            f"GPU memory after load: {torch.cuda.memory_allocated() / 1024**2:.0f} MB"
        )

    # ------------------------------------------------------------------
    # Single request (identical to Layer 2, for /generate endpoint)
    # ------------------------------------------------------------------

    def generate(
        self,
        messages: list[dict],
        max_new_tokens: int = 64,
        temperature: float = 1.0,
    ) -> dict:
        results = self.generate_batch(
            batch_messages=[messages],
            max_new_tokens=max_new_tokens,
            temperature=temperature,
        )
        return results[0]

    # ------------------------------------------------------------------
    # Batched generation — the new capability in Layer 3
    # ------------------------------------------------------------------

    def generate_batch(
        self,
        batch_messages: list[list[dict]],   # list of B conversations
        max_new_tokens: int = 128,
        temperature: float = 1.0,
    ) -> list[dict]:
        """
        Process B requests in one batched forward pass.

        Returns a list of B result dicts with the same keys as Layer 2's
        generate(), plus batch_size for logging.
        """
        B = len(batch_messages)
        t0 = time.perf_counter()

        # --- Step 1: format + tokenize all prompts ---
        input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(
            batch_messages
        )
        max_prompt_len = input_ids.shape[1]
        # Keep a tensor version for position_ids arithmetic; keep list for logging/results.
        prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")

        logger.info(
            f"batch_size={B} max_prompt_len={max_prompt_len} "
            f"prompt_lens={prompt_lens_list}"
        )

        # --- Step 3: batched prefill ---
        # One forward pass processes all B prompts simultaneously.
        # The KV cache stores [B, heads, max_prompt_len, head_dim] after this.
        #
        # position_ids fix: without explicit position_ids, HuggingFace assigns
        # sequential positions 0..max_len-1 globally, so a prompt of length L
        # padded to max_len M gets RoPE positions M-L..M-1 instead of 0..L-1.
        # We compute per-example position_ids from attention_mask so every real
        # token gets the same position it would have in the unpadded B=1 run.
        prefill_position_ids = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)

        past_kv = KVCache()
        t_prefill = time.perf_counter()
        with torch.no_grad():
            out = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=prefill_position_ids,
                past_key_values=past_kv,
                use_cache=True,
            )
        past_kv = out.past_key_values
        ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)

        logger.info(f"after prefill: {past_kv}  ttft={ttft_ms}ms")

        # Sample the first token for each request from the last position's logits.
        # out.logits shape: [B, max_prompt_len, vocab_size]
        # We take [:, -1, :] — the prediction at the last position of each row.
        next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]

        # --- Step 4: decode loop ---
        # finished[i] = True once request i has emitted <eos>
        finished   = next_tokens == self.eos_id          # [B] bool
        generated  = [[] for _ in range(B)]              # per-request token lists
        step_times = []
        decode_step = 0   # counts decode steps for position_ids computation

        for i, tok in enumerate(next_tokens.tolist()):
            if not finished[i]:
                generated[i].append(tok)

        for _ in range(max_new_tokens - 1):
            if finished.all():
                break

            t_step = time.perf_counter()

            # Feed [B, 1]: finished requests get pad_token so they don't pollute
            # the cache, but their output is discarded anyway.
            current_tokens = next_tokens.unsqueeze(1)   # [B, 1]
            current_tokens = torch.where(
                finished.unsqueeze(1),
                torch.full_like(current_tokens, self.pad_id),
                current_tokens,
            )

            # Attention mask for decode step: [B, cached_len + 1].
            # We extend the previous mask by one real position for every request.
            # Finished requests still get 1 so the cache shape stays consistent.
            attention_mask = torch.cat(
                [attention_mask, torch.ones(B, 1, dtype=torch.long, device="cuda")],
                dim=1,
            )

            # position_ids fix for decode: each example i is at position
            # prompt_lens[i] + decode_step, not the shared max_prompt_len + step.
            decode_position_ids = (prompt_lens + decode_step).unsqueeze(1).to("cuda")  # [B, 1]
            decode_step += 1

            with torch.no_grad():
                out = self.model(
                    input_ids=current_tokens,
                    attention_mask=attention_mask,
                    position_ids=decode_position_ids,
                    past_key_values=past_kv,
                    use_cache=True,
                )

            past_kv = out.past_key_values
            next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
            step_times.append(time.perf_counter() - t_step)

            newly_finished = next_tokens == self.eos_id
            for i, tok in enumerate(next_tokens.tolist()):
                if not finished[i] and not newly_finished[i]:
                    generated[i].append(tok)

            finished = finished | newly_finished

        # --- Step 5: decode tokens to text, build result dicts ---
        latency_ms = round((time.perf_counter() - t0) * 1000, 1)
        tpot_ms = round(
            (sum(step_times) / len(step_times)) * 1000, 1
        ) if step_times else ttft_ms

        texts = self.tokenizer.decode_batch(generated)

        results = []
        for i in range(B):
            text = texts[i]
            results.append({
                "text": text,
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
