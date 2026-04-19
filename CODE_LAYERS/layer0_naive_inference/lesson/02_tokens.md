# 02 — Tokens: The Unit of Language

## Why Not Characters or Words?

A language model does not read text the way a human does. Internally, it works with integers — token IDs — each of which maps to a fragment of text. How large or small those fragments are turns out to matter a great deal.

Character-level tokenization is tempting in its simplicity: every letter is a token, the vocabulary is tiny, and the representation is lossless. The problem is that even a short sentence becomes hundreds of tokens, and the model has to track relationships across far more positions than necessary.

Word-level tokenization goes too far in the other direction. Natural languages have enormous vocabularies — hundreds of thousands of unique word forms when you account for conjugations, compound words, proper nouns, technical jargon, and code identifiers. A word the model has never seen would have to be represented as `<unk>`, losing all information about what it actually says.

The practical solution is to split words into **subword pieces**. Common words like "the", "is", "model" are each a single token. A less common word like "tokenizer" might be split into "token" + "izer". A Python identifier like `AutoModelForCausalLM` might be split into several pieces. This way the vocabulary stays manageable — Qwen3 uses 151,936 tokens — while still being able to represent any string exactly, even ones that were never in the training data.

## The Tokenizer in Code

The tokenizer is a separate artifact from the model weights. It is loaded from the same directory and encapsulates the vocabulary and any special tokens (like `<|im_start|>`, `<|im_end|>`, `<|endoftext|>`). Loading it is a single call:

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
```

`AutoTokenizer.from_pretrained` reads `tokenizer.json` and `tokenizer_config.json` from the model directory. The call is fast compared to loading model weights because the tokenizer has no large tensor data.

To convert text into token IDs, call `encode`:

```python
ids = tokenizer.encode("Hello, world!")
print(ids)        # e.g. [9707, 11, 1879, 0]
print(len(ids))   # 4
```

Each integer in the resulting list is an index into the vocabulary. The mapping is deterministic and invertible: given the same text and the same tokenizer, you always get the same IDs.

To convert token IDs back to text, call `decode`:

```python
text = tokenizer.decode([9707, 11, 1879, 0])
print(text)   # "Hello, world!"
```

This round-trip — string → IDs → string — is exactly what happens at the boundary of every inference call. The prompt enters as text, becomes IDs for the model to process, and the model's output IDs are decoded back to text before being returned to the caller.

## Tokens Are Not Words

One detail worth internalising before moving on: a single token does not correspond to a single word, and the number of tokens in a string can surprise you. Whitespace is often included in a token (the token for " model" with a leading space is different from "model"). Punctuation may be its own token or fused with an adjacent word. Numbers are often split digit by digit. Code is split along syntax boundaries.

This matters in two practical ways. First, when the server returns `prompt_tokens` and `completion_tokens` in its response, those counts are in tokens, not words — and they are the correct unit for measuring compute cost. Second, `max_new_tokens` is a token count, not a word count, so "generate up to 64 tokens" produces somewhere between 40 and 80 English words depending on the content.

To see the token boundaries explicitly, you can use `convert_ids_to_tokens`:

```python
ids = tokenizer.encode("AutoModelForCausalLM")
pieces = tokenizer.convert_ids_to_tokens(ids)
print(pieces)   # e.g. ['Auto', 'Model', 'For', 'Causal', 'LM']
```

This shows the exact subword segmentation the model receives — a useful diagnostic when a model behaves unexpectedly on a particular string.

## Special Tokens

Alongside the regular vocabulary of text fragments, every tokenizer defines a set of **special tokens** — reserved IDs that carry structural meaning rather than representing natural language text. These are not words; they are signals to the model about the shape of its input or the state of generation.

The most important one for inference is the **end-of-sequence token**, commonly written as `<|endoftext|>` or `<eos>`. When the model generates this token, it is signalling that it considers its response complete. The generate loop watches for it and stops automatically when it appears, which is why you do not need to specify the exact number of tokens a reply should contain — the model decides on its own when it is done, and generation stops as soon as EOS is produced or `max_new_tokens` is reached, whichever comes first.

Qwen3 also uses `<|im_start|>` and `<|im_end|>` (short for "imaginary message") to mark the boundaries of each speaker's turn in a conversation. You will see these in the formatted prompt string produced by `apply_chat_template` in the next section. When you call `tokenizer.decode(..., skip_special_tokens=True)`, these structural tokens are stripped from the output so the caller receives clean text rather than raw markup.
