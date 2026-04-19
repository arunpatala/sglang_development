# 02 — The Tokenizer

## Why Tokenization Moved to Its Own File

In Layers 1 and 2, tokenization was inline inside `model.py`. A few lines applied the chat template, called the HuggingFace tokenizer, and moved the resulting tensor to the GPU. That worked for a single request. Layer 3 adds batched tokenization with left-padding — a meaningfully different operation — and the right place to put it is not alongside tensor arithmetic and forward calls. Keeping the tokenizer in its own class has a second reason: SGLang runs tokenization in a separate process (`TokenizerManager`) precisely because formatting and encoding can overlap with GPU computation. Having a clean boundary between the two makes that architectural move straightforward in a later layer.

The call that appears in `generate_batch()` is:

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
```

Everything the tokenizer does is in service of producing these three return values. The sections below walk through `tokenizer.py` top to bottom.

---

## `Tokenizer.__init__`

```python
def __init__(self, model_path: str):
    self._tok = AutoTokenizer.from_pretrained(model_path)
    self._tok.padding_side = "left"

    if self._tok.pad_token_id is None:
        self._tok.pad_token_id = self._tok.eos_token_id
```

`padding_side = "left"` is the most consequential line in the initialiser. When HuggingFace pads a batch of sequences to the same length, it appends padding tokens on whichever side this attribute specifies. Left-padding places all padding at the beginning of each row, so every sequence ends with its last real token at the final column. This alignment is required by decoder-only models: the logit at position `-1` is the prediction for what should come next, and that prediction must be at the same column index for every row in the batch. Right-padding would shift the last real token to different columns for different prompts, making the `[:, -1, :]` logit slice meaningless.

Most large language model tokenizers do not define a dedicated pad token. Qwen3 is one of them. Setting `pad_token_id = eos_token_id` gives the tokenizer a value to fill padding positions with. Using EOS as the pad is conventional — the model was trained on sequences ending with EOS, so it will produce near-zero attention weights for these tokens when the `attention_mask` correctly marks them as padding.

---

## `apply_chat_template`

```python
def apply_chat_template(self, messages: list[dict], add_generation_prompt: bool = True) -> str:
    return self._tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt,
        enable_thinking=False,
    )
```

This is the same chat template call used in Layers 1 and 2. It takes a list of messages in the standard `[{"role": "...", "content": "..."}]` format and formats them into the string the model was trained on — inserting special delimiter tokens like `<|im_start|>` and `<|im_end|>` in the positions Qwen3 expects. `tokenize=False` returns the formatted string rather than token IDs, because encoding happens separately in `encode_batch`. `enable_thinking=False` suppresses Qwen3's chain-of-thought mode.

---

## `encode_batch`

```python
def encode_batch(self, texts: list[str], device: str = "cuda") -> tuple:
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
```

Passing a list of strings to a HuggingFace tokenizer with `padding=True` and `return_tensors="pt"` produces a padded batch in one call. The tokenizer finds the longest string in the list, tokenizes all strings, and pads shorter ones with `pad_token_id` on the left (because `padding_side="left"` was set in `__init__`). The result is two tensors:

`input_ids` has shape `[B, max_len]`. Every row has the same width. Rows shorter than `max_len` start with padding tokens in the leftmost columns. `attention_mask` has the same shape, with 1 where a real token sits and 0 where a padding token sits.

`prompt_lens` is derived by summing each row of `attention_mask`. The sum of a row equals the number of 1s in that row, which is the number of real tokens — the actual prompt length before padding. This list is returned separately because `model.py` needs it to compute per-request position IDs in the decode loop, and it cannot be recovered from `input_ids` alone once padding has been mixed in.

---

## `prepare_batch`

```python
def prepare_batch(self, batch_messages: list[list[dict]], device: str = "cuda") -> tuple:
    formatted = [self.apply_chat_template(msgs) for msgs in batch_messages]
    return self.encode_batch(formatted, device=device)
```

`prepare_batch` is the method `generate_batch()` calls. It iterates over the B conversations, formats each one with `apply_chat_template`, then passes the resulting list of strings to `encode_batch`. The return value is passed directly to the prefill forward call:

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
```

The separation between formatting and encoding matters because formatting is CPU-bound string manipulation while encoding is a tokenizer lookup. If in a future layer the tokenizer runs in a separate process, these two steps can be distributed across processes without changing the interface.

---

## `decode_batch`

```python
def decode_batch(self, batch_token_ids: list[list[int]]) -> list[str]:
    return [self.decode(ids) for ids in batch_token_ids]
```

At the end of `generate_batch()`, `generated` is a list of B lists of integer token IDs. `decode_batch` converts each to a string by calling `tokenizer.decode(ids, skip_special_tokens=True)`. The `skip_special_tokens=True` flag removes any EOS or padding tokens that might have been recorded in `generated` — in practice `generated[i]` only contains tokens appended before EOS, so no special tokens should appear, but the flag is a safe default.
