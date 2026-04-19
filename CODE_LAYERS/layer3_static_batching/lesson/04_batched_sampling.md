# 04 — Batched Sampling

## From a Scalar to a Tensor

In Layers 1 and 2, `sample_next_token` in `sampling.py` operated on a single logit vector and returned a single integer:

```python
def sample_next_token(logits, temperature):   # [vocab_size] → int
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())
```

`argmax` returned a scalar tensor; `.item()` converted it to a Python int. `torch.multinomial` returned shape `[1]`; again `.item()` converted it. One integer per call, one call per step.

Layer 3 must produce one token for each of the B requests in a single call. The input is now `[B, vocab_size]` and the return value is a `[B]` tensor:

```python
def sample_batch(logits, temperature):        # [B, vocab_size] → Tensor[B]
    if temperature == 0.0:
        return logits.argmax(dim=-1)
    if temperature != 1.0:
        logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)
```

The logic is identical. The shape handling is not.

---

## What Each Operation Does on the Batch Dimension

`logits.argmax(dim=-1)` finds the index of the maximum value along the last dimension. When `logits` has shape `[B, vocab_size]`, `dim=-1` is the vocabulary dimension — the same dimension as in the single-request case. Applying `argmax` along that dimension reduces it, leaving a `[B]` tensor where each element is the greedy token for the corresponding row. No `.item()` call: the result is kept as a GPU tensor for downstream use in `finished` comparisons and `torch.where` operations.

`torch.softmax(logits, dim=-1)` normalises each row independently. Each row sums to 1.0 after the operation. The temperature scaling `logits / temperature` also operates row-wise because scalar division broadcasts across all dimensions.

`torch.multinomial(probs, num_samples=1)` draws one sample from each row's probability distribution. When `probs` has shape `[B, vocab_size]`, the output has shape `[B, 1]` — one sample per row, wrapped in an extra dimension. The `.squeeze(-1)` removes that trailing dimension, returning `[B]`.

---

## Where It Is Called

`sample_batch` is called in two places in `generate_batch()`. The first call is after prefill:

```python
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
```

`out.logits` has shape `[B, max_prompt_len, vocab_size]` after the batched prefill. The slice `[:, -1, :]` selects the last-position logits for every row simultaneously, giving `[B, vocab_size]`. This is the prediction for what token should follow the last real token in each prompt.

The second call is inside the decode loop:

```python
next_tokens = sample_batch(out.logits[:, -1, :], temperature)  # [B]
```

After a decode step, `out.logits` has shape `[B, 1, vocab_size]` — one position per row, the single new token just processed. The `[:, -1, :]` slice reduces this to `[B, vocab_size]`. The result is the next token for each request.

Both calls are identical in code. The difference is the shape of `out.logits` going in: `[B, max_prompt_len, vocab_size]` after prefill, `[B, 1, vocab_size]` after each decode step. In both cases `[:, -1, :]` selects the last position correctly.

---

## Why Not Call `sample_next_token` B Times

Calling `sample_next_token` in a Python loop over B requests would produce the correct result but would be slower for two reasons. First, each call would issue a separate GPU kernel for the softmax and multinomial operations — B kernel launches instead of one. Second, the `temperature != 0.0` branch calls `torch.softmax` and `torch.multinomial`, both of which are GPU operations. On a GPU, a single kernel operating on a `[B, vocab_size]` matrix is substantially more efficient than B sequential kernels each operating on a `[vocab_size]` vector, because GPU hardware is designed to execute wide operations in parallel and kernel launch overhead is non-trivial relative to the computation for a small vocabulary slice.

`sample_batch` keeps the sampling on the GPU and batched. The returned `[B]` tensor is used directly in the next `torch.where` and comparison operations, also on the GPU, without any round-trip to CPU Python integers.

For now, every request in the batch shares a single `temperature` value. In continuous batching, each request arrives independently and carries its own sampling parameters — its own temperature, top-p, top-k, and so on. The sampling call will need to accept a per-request parameter vector rather than a scalar, so that different requests in the same forward pass can be sampled with different settings.
