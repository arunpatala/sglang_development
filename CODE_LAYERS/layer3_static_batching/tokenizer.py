"""
Layer 3 — Tokenizer: wraps HuggingFace tokenizer with inference-focused helpers.

Separating this from model.py means:
  - model.py only deals with tensors and forward passes
  - Later layers can swap tokenization strategies (e.g. chunked, streaming)
    without touching model logic
  - The tokenizer can be moved to a separate process (like SGLang's
    TokenizerManager) when we add async/multi-process serving
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class Tokenizer:
    """
    Thin wrapper around a HuggingFace tokenizer with helpers for:
      - Chat template formatting
      - Single-prompt tokenization
      - Batched tokenization with left-padding
      - Token decoding
    """

    def __init__(self, model_path: str):
        logger.info(f"Loading tokenizer from {model_path}")
        self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)

        # Left-pad so the last real token always aligns to position -1 in a batch.
        # This is required for decoder-only models (Qwen3, Llama, etc.) so that
        # all sequences in a batch share the same "generate from here" position.
        self._tok.padding_side = "left"

        # Use eos as pad if no dedicated pad token exists (common in LLMs).
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id

        logger.info(
            f"Tokenizer ready  vocab={self._tok.vocab_size}  "
            f"eos={self.eos_token_id}  pad={self.pad_token_id}"
        )

    # ------------------------------------------------------------------
    # Token IDs
    # ------------------------------------------------------------------

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    # ------------------------------------------------------------------
    # Formatting
    # ------------------------------------------------------------------

    def apply_chat_template(
        self,
        messages: list[dict],   # [{"role": "...", "content": "..."}]
        add_generation_prompt: bool = True,
    ) -> str:
        """
        Format a conversation into the model's expected prompt string.
        enable_thinking=False suppresses Qwen3's <think>...</think> mode.
        """
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    # ------------------------------------------------------------------
    # Encoding
    # ------------------------------------------------------------------

    def encode(self, text: str, device: str = "cuda"):
        """
        Tokenize a single already-formatted string.
        Returns input_ids tensor of shape [1, seq_len] on `device`.
        """
        enc = self._tok(text, return_tensors="pt")
        return enc["input_ids"].to(device)

    def encode_batch(
        self,
        texts: list[str],
        device: str = "cuda",
    ) -> tuple:
        """
        Tokenize a list of already-formatted strings with left-padding.

        Returns:
            input_ids:      [B, max_len]  — token ids, pad where needed
            attention_mask: [B, max_len]  — 1 for real tokens, 0 for pad
            prompt_lens:    list[int]     — real token count per request
        """
        enc = self._tok(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lens    = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, prompt_lens

    # ------------------------------------------------------------------
    # Decoding
    # ------------------------------------------------------------------

    def decode(self, token_ids: list[int]) -> str:
        """Decode a flat list of token ids to a string, skipping special tokens."""
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def decode_batch(self, batch_token_ids: list[list[int]]) -> list[str]:
        """Decode a list of token-id lists (one per request)."""
        return [self.decode(ids) for ids in batch_token_ids]

    # ------------------------------------------------------------------
    # Convenience: format + encode in one call
    # ------------------------------------------------------------------

    def prepare_single(
        self,
        messages: list[dict],
        device: str = "cuda",
    ):
        """Format messages and tokenize as a single prompt. Returns input_ids [1, L]."""
        formatted = self.apply_chat_template(messages)
        return self.encode(formatted, device=device)

    def prepare_batch(
        self,
        batch_messages: list[list[dict]],
        device: str = "cuda",
    ) -> tuple:
        """
        Format and tokenize a batch of conversations.

        Returns:
            input_ids, attention_mask, prompt_lens
        """
        formatted = [self.apply_chat_template(msgs) for msgs in batch_messages]
        return self.encode_batch(formatted, device=device)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(model={self._tok.name_or_path!r}, "
            f"vocab={self.vocab_size}, "
            f"padding_side={self._tok.padding_side!r})"
        )
