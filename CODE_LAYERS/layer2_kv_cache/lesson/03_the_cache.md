# 03 — What the KV Cache Is

## The Redundant Work in Layer 1

Layer 1 made the decode loop visible. Every decode step is now written in our code: call the model, get logits, sample a token, append it, repeat. The computation is honest. And because it is honest, a problem is also now visible.

Consider what happens on decode step 2. The model receives the full sequence — the original prompt plus the two tokens generated in steps 0 and 1. It runs all of that through every layer in the network. As part of that computation, for each token in the sequence, each attention layer computes a key vector and a value vector. These are the tensors that let each token attend to every other token in the sequence.

Now consider what happens on decode step 3. The model receives the sequence again — now one token longer — and runs all of it through every layer once more. The key and value vectors for the prompt tokens are computed again. The key and value vectors for the token from step 0 are computed again. The key and value vectors for the token from step 1 are computed again. All of that is identical to what was computed in step 2. The only genuinely new computation is for the single new token added to the end.

This is the redundant work. On every decode step, the model recomputes key and value vectors for every token it has already processed, even though those tokens — and therefore their key and value vectors — have not changed. For a prompt of length `L` generating `T` tokens, the total redundant computation is substantial: by step `T`, the model is doing `L+T-1` tokens' worth of attention work when only 1 token's worth is new.

## Why the Redundancy Can Be Eliminated

Section 02 covers the attention mechanism in detail — what query, key, and value vectors are and why each token's key and value vectors are fixed the moment they are computed. The short version relevant here is that the causality of the attention mask means no future token can change the key or value vectors of a past token. Once computed, they are immutable. Computing them again on any subsequent step produces an identical result.

This means the redundant work in Layer 1 is not just unnecessary — it is provably safe to skip without changing the model's output at all.

## The Cache as a Saved Scratchpad

The KV cache is the solution to this redundancy. Instead of discarding the key and value tensors at the end of each forward pass, we keep them. On the next decode step, rather than recomputing them from scratch, we pass them back to the model. The model then only needs to compute key and value tensors for the single new token, and it attends the new token's query against the full set of cached keys and values — which already contain all prior positions.

Conceptually the model does the same thing it always did: it attends every token to every other token. But mechanically it only computes the new token's contribution and reads the rest from the cache.

## What Changes in the Code

The change from Layer 1 to Layer 2 in `model.py` is two arguments and one assignment. In Layer 1, every forward pass looks like this:

```python
out = self.model(input_ids=ids, use_cache=False)
```

where `ids` is the full growing sequence. In Layer 2, the prefill pass looks like this:

```python
past_kv = KVCache()
out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values
```

And every subsequent decode step looks like this:

```python
current_token = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values
```

`input_ids` is now always `[1, 1]` — one token — instead of the growing sequence. `past_key_values` carries the accumulated key and value tensors for all prior positions. After each forward pass, `out.past_key_values` is the updated cache — the same object, now extended by one position — which we assign back to `past_kv` and pass in again on the next step.

That is the entirety of the change. `server.py` is unchanged. `benchmark.py` is unchanged. The API contract is unchanged. Everything that changes is in these few lines of `model.py`.

## Why This Matters for Performance

In Layer 1, each decode step processes a sequence of length `L + k` (where `L` is the prompt length and `k` is the number of tokens generated so far). The attention operation inside each layer is quadratic in sequence length — every position attends to every other position — so the cost of each decode step grows as `k` increases. By the time the model has generated 100 tokens into a 400-token prompt, each decode step is processing a 500-token sequence, most of which was already processed on all previous steps.

In Layer 2, each decode step processes exactly one token. The attention operation reads the cached keys and values for all prior positions, but there is only one query — the new token. The cost is linear in cache size (reading `L+k` cached entries), not quadratic. This is a much smaller number, and crucially it does not grow with the number of tokens already generated in the same way.

The practical effect is that TPOT — the time per output token — drops substantially and remains near-constant regardless of prompt length. TTFT — the time to the first token — stays roughly the same, because the prefill pass is unchanged: it still has to process the full prompt.
