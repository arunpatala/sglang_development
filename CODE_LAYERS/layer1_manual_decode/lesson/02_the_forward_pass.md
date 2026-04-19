# 02 — The Forward Pass

## Calling the Model Directly

In Layer 0, the entire generation process was triggered by one call to `model.generate()`. That method is convenient, but it hides the mechanics: you hand it a prompt and get back a completed sequence, with no visibility into what happened in between.

In Layer 1 we replace that with a direct call to the model itself:

```python
with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)
```

This is a single forward pass. The model takes the current sequence of token IDs, runs them through all of its layers, and returns a result object. That result object is what this section is about — specifically, what is inside it and how to use it.

## What the Model Returns

The call above returns a `CausalLMOutputWithPast` object. For our purposes, only one field matters:

```python
out.logits   # shape: [batch_size, sequence_length, vocab_size]
```

This is a three-dimensional tensor. The first dimension is the batch size — since we process one request at a time, this is always 1. The second dimension is the length of the input sequence — every token we passed in. The third dimension is the vocabulary size — for Qwen3, that is 151,936.

So for a prompt of 20 tokens, `out.logits` has shape `[1, 20, 151936]`. Every one of those 151,936 numbers at a given position is a score the model assigns to the possibility of that vocabulary token appearing at that position in the sequence.

## Why One Score Vector Per Position?

The model produces logits for every position in the input, not just the last one. This is a consequence of how transformers are trained: during training, the model predicts the next token at every position simultaneously, which is what makes training efficient. Given the sequence "The cat sat", it predicts the token after "The", the token after "The cat", and the token after "The cat sat" all in a single forward pass.

At inference time, we only care about one of those position predictions: the last one. The logits at position `T-1` (zero-indexed, so the final position) tell us what token the model thinks should come after the entire sequence we have given it so far. Every other position's logits are byproducts of the forward pass that we discard.

## Slicing Out the Last Position

```python
next_token_logits = out.logits[0, -1, :]   # shape: [vocab_size]
```

This line does three things. The first index `0` selects the first (and only) element of the batch. The second index `-1` selects the last position in the sequence — Python's negative indexing convention, so this always refers to the final token regardless of sequence length. The third index `:` keeps all 151,936 vocabulary scores. The result is a one-dimensional tensor of shape `[vocab_size]`.

This single vector is everything the model has to say about what should come next. Picking a token from it is the subject of the sampling section. For now, the key thing to hold onto is the shape: one number per vocabulary token, higher meaning the model considers that token more likely to be a good continuation.

## What `use_cache=False` Does Here

```python
out = self.model(input_ids=ids, use_cache=False)
```

HuggingFace models have an internal optimisation where they save intermediate computations — specifically key and value tensors from each attention layer — so they do not have to recompute them on the next call. Setting `use_cache=False` disables this. We disable it deliberately here because we want the raw, unoptimised cost of the forward pass to be visible. Every step of the loop recomputes the full sequence from scratch, which is expensive but transparent. The next chapter will make this cost concrete by measuring step times individually.

## Putting It Together

Here is the complete sequence from receiving a request to having the logits for the next token:

```python
# 1. Format the conversation and tokenize it
formatted = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
# ids shape: [1, prompt_len]

# 2. Run one forward pass
with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)
# out.logits shape: [1, prompt_len, 151936]

# 3. Extract the last position's scores
next_token_logits = out.logits[0, -1, :]
# next_token_logits shape: [151936]
```

At this point `next_token_logits` is a vector of 151,936 numbers. The model has processed the entire conversation and is expressing, in the form of these scores, its beliefs about what the next token should be. The highest score corresponds to the token the model considers most likely. The job of the sampling step — covered in section 04 — is to turn those scores into an actual token ID we can append to the sequence.
