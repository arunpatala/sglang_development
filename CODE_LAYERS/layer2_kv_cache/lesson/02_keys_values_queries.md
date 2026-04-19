# 02 — Keys, Values, and Queries in Attention

## What `past_key_values` Actually Holds

In section 01, every call in the decode loop passed `past_key_values=past_kv` to the model and received back `out.past_key_values` after each step:

```python
out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values
```

`past_kv` is not a tensor and not a dictionary. It is a cache object that stores the key and value tensors that every attention layer in the model computed during the prefill pass and each decode step since. To understand why saving those specific tensors is valid — why reading them on the next step gives the exact same result as recomputing them from scratch — you need to understand what keys and values are and how they are used inside each attention layer.

## The Three Projections

Every transformer layer applies an attention operation. The input to that layer is a sequence of token representations — one vector per token, the result of everything the network has done to that token up to this point. The attention layer begins by projecting each of those representations into three separate vectors using three learned weight matrices:

```
Q = token_representation @ W_Q    # query
K = token_representation @ W_K    # key
V = token_representation @ W_V    # value
```

These matrices are different for every layer and learned during training. The same token representation is multiplied by all three to produce the query, key, and value for that token at that layer. The three projections serve distinct roles in the computation that follows.

## What Each Vector Does

The **query** is what a token is asking for. It is a compact representation of what information this token needs from the rest of the sequence in order to produce a good output. When processing the word "bank" in the sentence "She deposited money at the bank", the query for "bank" encodes something like "I need to know whether this is a financial institution or a river bank — show me tokens that might disambiguate this."

The **key** is what a token is advertising. It is a representation of what information this token can provide to other tokens that query it. The key for "money" in the same sentence encodes something like "I am a financial concept and I am available to any token that is looking for that kind of information."

The **value** is what a token actually contributes when it is selected. After the attention scores determine how much attention to pay to each token, the output at any position is a weighted sum of value vectors — the keys determined who gets noticed, the values determine what gets contributed.

## The Attention Computation

With queries, keys, and values computed for every token in the sequence, the attention operation proceeds in three steps.

First, each token's query is compared against every other token's key via a dot product. A high dot product between a query and a key means those two tokens are highly compatible — the querying token wants what the key token is offering. This produces a matrix of raw attention scores, one score per (query position, key position) pair:

```
scores = Q @ K.transpose(-2, -1)   # shape: [seq_len, seq_len]
```

Second, the scores are scaled and passed through softmax to produce a probability distribution over positions. Scaling by the square root of the head dimension prevents the dot products from growing too large and pushing the softmax into regions where gradients vanish:

```
weights = softmax(scores / sqrt(d_head))   # each row sums to 1
```

Third, those weights are used to take a weighted sum of value vectors. The output at each position is a blend of every other token's value, with tokens that scored highly in attention getting more weight:

```
output = weights @ V   # shape: [seq_len, d_head]
```

This is the attention output for one layer. It is then passed through a feed-forward network and into the next layer, where the same process repeats.

## Why Causality Makes K and V Cacheable

In a decoder-only language model, the attention operation is causal: each token can only attend to tokens at earlier positions, never to tokens that come after it. This is enforced during training by masking out future positions in the attention score matrix — setting those scores to negative infinity before the softmax, so their weight becomes zero.

The causal mask has a critical consequence for the K and V vectors of any given token. Because a token at position `i` cannot be influenced by tokens at positions `i+1` onwards, its token representation — the input to the Q/K/V projection matrices — depends only on positions 0 through `i-1`. This means the key and value vectors for position `i` are fully determined by what came before position `i`, and they do not change when new tokens are appended to the sequence.

Concretely: after processing a 50-token prompt, the key and value vectors for token 0 through token 49 are fixed. When the model generates token 50 and runs a new forward pass, the keys and values for positions 0 through 49 are identical to what they were on the previous step. Computing them again is doing the same multiplication — same input representation, same weight matrices — and getting the same result.

## What the Query Does Not Share

The query vector is different. The query for the new token at position `k` is computed from that token's representation, which has not been seen before. But more importantly, the query at position `k` is the only query that matters for the current forward pass — we only need to know what token `k` is asking for. We do not need the queries for positions 0 through `k-1` again, because those positions already produced their output on previous steps.

This asymmetry is the foundation of the KV cache. Keys and values from past positions are reusable — same input, same output, no point recomputing. Queries from past positions are not needed — their job is already done. The new query attends over all past keys to produce all new attention scores, reads all past values to produce the attention output, and that is all that is required for the current step.

## In the Code

When the model runs a decode step with `past_key_values=past_kv`, the attention layer:

1. Computes fresh Q, K, V for the single new token.
2. Calls `past_kv.update(new_k, new_v, layer_idx)`, which appends the new key and value to the cache and returns the full accumulated K and V tensors covering all positions from 0 to the current one.
3. Computes attention scores between the new query and all cached keys: `scores = new_Q @ all_K.transpose(-2, -1)`.
4. Produces the attention output as a weighted sum of all cached values: `output = softmax(scores) @ all_V`.

The result is identical to a full forward pass over the complete sequence. From the attention layer's perspective, all keys and values are present. The only difference is that most of them came from the cache rather than being computed fresh.
