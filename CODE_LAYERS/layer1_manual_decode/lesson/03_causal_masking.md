# 03 — Causal Masking

## The Problem With Full Attention

In the previous section we saw that a single forward pass produces logits at every position in the sequence simultaneously. This is what makes training efficient — given the sequence "The cat sat on the mat", the model can predict the next token at every position in one shot. But it raises a critical question: when computing the representation for position 3 ("on"), is the model allowed to look at position 4 ("the") and position 5 ("mat")?

If it could, the model would trivially cheat during training. Predicting what comes after "The cat sat" is easy if you can already see the answer. The network would learn nothing useful — it would just learn to copy future tokens rather than understand language. To prevent this, every forward pass applies a **causal mask** that enforces a strict rule: each token may only attend to itself and to positions that came before it, never to positions that come after.

## What the Mask Actually Does

Attention scores are computed by comparing each token's query vector against every other token's key vector. The raw result is a matrix of scores — one row per query position, one column per key position. High scores mean high compatibility between that query and that key.

Before these scores are converted to weights via softmax, the causal mask is applied. Every entry in the score matrix that represents a future position — that is, every entry where the key position is after the query position — is replaced with negative infinity:

```
Sequence: [The, cat, sat, on]

Raw scores (each row is one query, each column one key):
            The    cat    sat    on
The     [  2.1,   0.8,   1.3,  -0.4 ]
cat     [  0.6,   3.2,   0.9,   1.1 ]
sat     [  1.4,  -0.2,   2.7,   0.5 ]
on      [ -0.1,   1.8,   0.3,   3.4 ]

After causal mask (future positions → -∞):
            The    cat    sat    on
The     [  2.1,   -∞,    -∞,    -∞  ]   ← The can only see itself
cat     [  0.6,   3.2,   -∞,    -∞  ]   ← cat sees The and cat
sat     [  1.4,  -0.2,   2.7,   -∞  ]   ← sat sees The, cat, sat
on      [ -0.1,   1.8,   0.3,   3.4 ]   ← on sees everything
```

When softmax is applied row by row, `exp(-∞) = 0`, so the masked positions contribute exactly zero weight to the final output. The attention output at each position is then a weighted sum of value vectors, with weights that only draw from positions at or before the current one. Token "The" produces its output using only its own value. Token "cat" blends its own value with "The"'s value. Token "on" can draw on all four.

This mask is built and applied entirely inside the model's attention layers — our decode loop code does not manage it. The model applies it on every forward pass automatically.

## Why the Mask Makes Training Efficient

The causal mask is what allows the model to learn from all positions simultaneously in a single forward pass. Without it, the model would need to be run separately for each position to avoid information leakage. With it, all positions are trained at once — position 0 predicts position 1, position 1 predicts position 2, and so on, all in parallel, with the mask ensuring that each prediction uses only the information a real autoregressive model would have access to.

This is sometimes called **teacher forcing**: during training, the ground truth sequence is given to the model, and the model is asked to predict each next token using only the tokens to its left. The mask enforces the "to its left" constraint.

At inference time — in our decode loop — the model runs forward passes over the current sequence and the mask still applies. During prefill, all prompt tokens are processed together and the mask ensures that position `i` in the prompt attends only to positions 0 through `i`. During each decode step, the same mask applies to the full sequence being processed.

## The Consequence for Key and Value Vectors

The causal mask has a property that is fundamental to understanding Layer 2: it makes the key and value vectors for any given token permanently fixed once that token has been processed.

To see why: the key and value vectors for token at position `i` are computed from that token's representation at each layer. That representation is the result of the attention operation at the layer below, which — because of the mask — can only draw on positions 0 through `i-1`. No token at position `i+1` or later can influence what happens at position `i`, because the mask blocks all such connections.

This means that when the decode loop appends a new token to the sequence and runs the next forward pass, the key and value vectors for all positions 0 through `i` are identical to what they were on the previous step. The computation that produced them is repeated, but the result is the same. Layer 2 exploits this by saving those vectors after each step and reusing them rather than recomputing them — which is exactly what `past_key_values` does.

Understanding the causal mask is therefore not just a detail about how attention works. It is the reason the KV cache is a valid optimisation at all.
