# 05 — Logits, Softmax, and Sampling

## What a Logit Is

After the forward pass we have a vector of 151,936 numbers — one per vocabulary token — at the last sequence position. These numbers are called **logits**. They are the raw output of the model's final linear layer: unbounded real numbers, positive or negative, with no particular scale.

A logit is a score. A higher score means the model considers that token a more likely continuation of the sequence. But the scores cannot be interpreted as probabilities directly — they do not sum to 1, and there is no constraint on their range. A logit of 12.3 does not mean "12.3% probability". It only means that token scored higher than one with a logit of 9.1.

To turn logits into probabilities we need softmax.

## Softmax: From Scores to Probabilities

Softmax takes a vector of arbitrary real numbers and converts it into a probability distribution — a vector of numbers that are all between 0 and 1 and sum to exactly 1. It does this by exponentiating each value and then dividing by the sum of all the exponentiated values:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j) for all j)
```

The exponentiation does two useful things. First, it ensures all values are positive (since exp of anything is always positive). Second, it amplifies differences: if one token's logit is much higher than the others, its softmax probability is disproportionately higher — exponential growth means gaps widen significantly. A token with a logit 3 points higher than another does not just get a slightly higher probability; it gets roughly `exp(3) ≈ 20` times the probability.

In PyTorch:

```python
probs = torch.softmax(logits, dim=-1)
# probs.shape: [vocab_size]
# probs.sum() == 1.0
# probs[i] >= 0 for all i
```

Now we have a genuine probability distribution over the vocabulary. The question is how to pick a token from it.

## Greedy Decoding

The simplest strategy is to always pick the token with the highest probability — equivalently, the highest logit, since softmax preserves the ranking:

```python
next_token_id = int(logits.argmax(dim=-1).item())
```

`argmax` returns the index of the maximum value. No softmax is needed here because we only care about which logit is largest, not what the probabilities are. This is called **greedy decoding**: always take the most likely next token.

Greedy decoding is deterministic — the same prompt always produces the same output. It is also fast, since there is no sampling involved. The downside is that it can produce repetitive or locally optimal but globally suboptimal text: choosing the single most likely word at every step does not always produce the best overall sentence.

In `model.py`, greedy decoding is triggered when `temperature=0.0`:

```python
if temperature == 0.0:
    return int(logits.argmax(dim=-1).item())
```

## Temperature Sampling

Rather than always picking the top token, we can sample from the probability distribution — treating it as a random draw where higher-probability tokens are more likely to be chosen, but not guaranteed. This produces more varied and natural-sounding output.

**Temperature** is a parameter that controls how peaked or flat the distribution is before sampling. It works by dividing the logits before applying softmax:

```python
if temperature != 1.0:
    logits = logits / temperature
```

When temperature is **less than 1** (say 0.5), dividing makes the logit differences larger. Softmax then amplifies those larger differences further, producing a sharper distribution where the highest-probability token dominates even more. The output becomes more focused and predictable.

When temperature is **greater than 1** (say 2.0), dividing makes the logit differences smaller. Softmax compresses them, producing a flatter distribution where probability is spread more evenly across tokens. The output becomes more random and creative, but also more likely to produce incoherent text.

At temperature **1.0**, the logits are unchanged before softmax. This is the default — the raw model distribution, unmodified.

A helpful way to think about it: temperature controls confidence. Low temperature makes the model act more confident (commit harder to its top choice). High temperature makes it act less confident (spread its bets).

## Multinomial Sampling

Once the logits are scaled by temperature and converted to probabilities via softmax, we draw a sample using `torch.multinomial`:

```python
probs = torch.softmax(logits, dim=-1)
next_token_id = int(torch.multinomial(probs, num_samples=1).item())
```

`torch.multinomial` draws one index from the distribution defined by `probs`. A token with probability 0.6 will be chosen 60% of the time; a token with probability 0.01 will be chosen 1% of the time. This is not argmax — it is a genuine random draw from the distribution, so different runs of the same prompt can produce different outputs.

## The Complete `sample_next_token`

Putting it together, the full sampling function lives in `sampling.py` and handles all three cases:

```python
def sample_next_token(logits: torch.Tensor, temperature: float) -> int:
    if temperature == 0.0:
        return int(logits.argmax(dim=-1).item())   # greedy

    if temperature != 1.0:
        logits = logits / temperature              # scale before softmax

    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())  # sample
```

`temperature=0.0` → greedy argmax, no randomness.
`temperature=1.0` → multinomial sample from the raw model distribution.
Any other value → scale logits, then multinomial sample from the adjusted distribution.

This function is called once per decode step, receiving `out.logits[0, -1, :]` from the forward pass and returning a single integer: the ID of the next token to append to the sequence. `model.py` imports it with `from sampling import sample_next_token` and calls it directly — the sampling logic is completely separate from the loop.

## Other Sampling Parameters

Temperature is the most fundamental sampling control, but real serving systems expose several more. They all operate on the logits or the probability distribution before the final sample is drawn.

**Top-k** restricts sampling to only the `k` most probable tokens, setting all others to zero probability before sampling. If `k=50`, the model can only ever choose from the 50 highest-scoring tokens at each step, regardless of what temperature does to their relative probabilities. This prevents the model from sampling very unlikely tokens even when temperature is high.

**Top-p** (also called nucleus sampling) is similar but adapts to the shape of the distribution. Instead of a fixed count of tokens, it keeps the smallest set of tokens whose cumulative probability exceeds `p`. If `p=0.9`, it includes tokens until their combined probability reaches 90%, then discards the rest. When the model is very confident, the top-p set is small; when it is uncertain, the set grows to include more candidates.

**Repetition penalty** discourages the model from repeating tokens it has already generated. Logits for tokens that appear in the current sequence are divided by the penalty factor before sampling, making them less likely to be chosen again. This is a simple fix for a common failure mode where models get stuck repeating the same phrase.

In this layer we implement only temperature, which is enough to make the sampling behaviour visible and controllable. Top-k, top-p, and repetition penalty are additions to `sample_next_token` that layer on top of the same foundation — modify the logits or probabilities, then draw the sample.
