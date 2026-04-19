# 01 — Token-Level Generation

## From Text In, Text Out to Token by Token

In Layer 0 we treated the model as a black box: a conversation goes in, a string of text comes out. That framing was enough to build a working server, but it hides almost everything interesting about how a language model actually works.

The reality is that a language model does not produce a full response in one shot. It generates one token at a time. Each new token is chosen based on all the tokens that came before it — the prompt plus everything the model has generated so far. That token is then appended to the sequence, and the process repeats. The model keeps going until it produces the end-of-sequence token, at which point it is signalling that it considers the response complete.

Written out explicitly, the loop looks like this:

```python
ids = prompt_tokens
while True:
    next_token = model(ids)   # conditioned on everything so far
    if next_token == EOS:
        break
    ids = ids + [next_token]  # grow the sequence by one
```

That is the entirety of autoregressive generation. Everything else — batching, caching, sampling strategies, memory management — is engineering built on top of this loop. Understanding it at this level of detail is what makes all of those optimisations legible.

## Why This Simple Formulation Is So Powerful

The "predict the next token" objective sounds almost too simple to be useful. But consider what a model must actually learn to do it well.

To predict the next word in a sentence about history, it must know history. To predict the next line of code in a Python function, it must understand the function's intent and Python's syntax. To predict the next turn in a conversation, it must track who said what and what would be a coherent response. There is no shortcut: a model that cannot model the world cannot predict what text about the world looks like.

This also explains why so much of the internet is useful training data. A recipe blog, a legal document, a GitHub repository, a forum thread — all of it is structured text where each token is conditioned on the ones before it. The training signal is everywhere. The model never needs labelled examples or manually constructed tasks; it just needs text, and text is abundant.

The result is a single model, trained on a single objective, that develops a remarkably broad range of capabilities entirely as a consequence of learning to predict what comes next.

## What This Chapter Covers

Layer 0 called `model.generate()` and let HuggingFace manage the loop internally. In this layer we write that loop ourselves. The computation is identical — same model, same number of forward passes, same output — but every step is now visible in our code.

This chapter works through what that loop actually does, piece by piece. You will see what the model returns when you call it directly: a tensor of logits, one score per vocabulary token, for every position in the sequence. You will learn why we only care about the last position, and what we do with those scores to select the next token. You will see how greedy decoding and temperature sampling differ, and what `torch.softmax` and `torch.multinomial` are actually computing.

You will also see the two phases that every generation request passes through — the prefill, where the model processes the full prompt in a single forward pass, and the decode, where it generates one token at a time — and why timing them separately gives you better diagnostic information than a single end-to-end latency number.

By the end of this chapter the loop is no longer a black box. Layer 2 will then make one targeted change to it — adding a cache so the model stops recomputing work it has already done — and because you will understand the loop in detail, that change will make complete sense.
