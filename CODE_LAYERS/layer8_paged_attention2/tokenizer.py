"""
Tokenizer — unchanged from Layer 3.

Wraps HuggingFace AutoTokenizer with inference-focused helpers.
Kept separate from model code so it can later move to its own process
(like SGLang's TokenizerManager).
"""

import logging

from transformers import AutoTokenizer, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self, model_path: str):
        logger.info(f"Loading tokenizer from {model_path}")
        self._tok: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(model_path)
        self._tok.padding_side = "left"
        if self._tok.pad_token_id is None:
            self._tok.pad_token_id = self._tok.eos_token_id
        logger.info(
            f"Tokenizer ready  vocab={self._tok.vocab_size}  "
            f"eos={self.eos_token_id}  pad={self.pad_token_id}"
        )

    @property
    def eos_token_id(self) -> int:
        return self._tok.eos_token_id

    @property
    def pad_token_id(self) -> int:
        return self._tok.pad_token_id

    @property
    def vocab_size(self) -> int:
        return self._tok.vocab_size

    def apply_chat_template(
        self,
        messages: list[dict],
        add_generation_prompt: bool = True,
    ) -> str:
        return self._tok.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            enable_thinking=False,
        )

    def encode(self, text: str, device: str = "cuda"):
        enc = self._tok(text, return_tensors="pt")
        return enc["input_ids"].to(device)

    def encode_batch(self, texts: list[str], device: str = "cuda") -> tuple:
        enc = self._tok(texts, return_tensors="pt", padding=True, truncation=False)
        input_ids      = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        prompt_lens    = attention_mask.sum(dim=1).tolist()
        return input_ids, attention_mask, prompt_lens

    def decode(self, token_ids: list[int]) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=True)

    def decode_batch(self, batch_token_ids: list[list[int]]) -> list[str]:
        return [self.decode(ids) for ids in batch_token_ids]

    def prepare_single(self, messages: list[dict], device: str = "cuda"):
        return self.encode(self.apply_chat_template(messages), device=device)

    def prepare_batch(self, batch_messages: list[list[dict]], device: str = "cuda") -> tuple:
        formatted = [self.apply_chat_template(msgs) for msgs in batch_messages]
        return self.encode_batch(formatted, device=device)

    def __repr__(self) -> str:
        return (
            f"Tokenizer(model={self._tok.name_or_path!r}, "
            f"vocab={self.vocab_size}, "
            f"padding_side={self._tok.padding_side!r})"
        )
