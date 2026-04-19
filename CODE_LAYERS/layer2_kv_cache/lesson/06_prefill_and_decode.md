# 06 — Prefill and Decode in Layer 2

## Two Phases, Now Genuinely Different Code

In Layer 1, prefill and decode were conceptually distinct but looked identical in code — both were a call to `self.model(input_ids=ids, use_cache=False)` with a growing `ids` tensor. The distinction was in what `ids` contained on that particular step.

In Layer 2, the two phases are different in code. Prefill and decode now have different arguments, different input shapes, and different roles in building and consuming the cache. Understanding each phase separately is essential to understanding how the cache achieves its performance improvement.

## Prefill

The prefill phase is the first forward pass. It processes the full prompt — every token in the formatted conversation — in one call:

```python
past_kv = KVCache()
with torch.no_grad():
    out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)

past_kv = out.past_key_values
next_token_logits = out.logits[0, -1, :]
next_token_id = sample_next_token(next_token_logits, temperature)
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

`input_ids` has shape `[1, prompt_len]` — the full prompt, just as in Layer 1. The difference is that we now pass `past_key_values=past_kv` (an empty `KVCache` instance) and `use_cache=True`. During this forward pass, as each attention layer processes each token, it calls `past_kv.update(new_k, new_v, layer_idx)` to store the key and value vectors it computed. By the time the model returns, `past_kv` contains the key and value tensors for every token in the prompt, for every attention layer in the network.

The cost of prefill is unchanged from Layer 1. The model still has to process the full prompt, still has to run attention across all positions, and still produces logits for every position in the sequence. The prefill cost is `O(L²)` in attention work, where `L` is the prompt length. Nothing about the cache changes this — the cache is being built during prefill, not consumed.

After prefill, three things are available: the first generated token (sampled from the logits at the last position), the timing measurement for TTFT, and the populated `past_kv` cache ready to be used on every subsequent decode step.

## Decode

Each decode step processes exactly one token — the most recently generated one — and reads the cached keys and values for all prior positions from `past_kv`:

```python
current_token = torch.tensor([[next_token_id]], dtype=torch.long, device=self.device)
with torch.no_grad():
    out = self.model(
        input_ids=current_token,
        past_key_values=past_kv,
        use_cache=True,
    )

past_kv = out.past_key_values
next_token_logits = out.logits[0, -1, :]
next_token_id = sample_next_token(next_token_logits, temperature)
```

`current_token` has shape `[1, 1]` — always. It never grows. The model computes key and value vectors for this single new token, appends them to the appropriate `LayerCache` via `past_kv.update`, and then attends this token's query against the full set of accumulated keys and values in the cache. The result is identical to what a full forward pass would compute — the same logits, the same next token — but without reprocessing any of the prior tokens.

`past_kv = out.past_key_values` updates the reference to the cache after each step. The cache object itself has been extended by one position inside the forward pass; this assignment makes sure `past_kv` still points to it for the next iteration.

## The Cost Difference

In Layer 1, decode step `k` processes a sequence of length `L + k`, and the attention computation inside each layer is proportional to `(L + k)²`. Each step is more expensive than the previous one. Over `T` decode steps, the total decode work grows as the sum of `(L+1)² + (L+2)² + ... + (L+T)²`.

In Layer 2, decode step `k` processes one token as the query and attends it over `L + k` cached entries. The attention computation is proportional to `L + k` — linear, not quadratic. Each step is slightly more expensive than the previous one (the cache is one entry longer), but the growth is linear in `k` rather than quadratic. For a 400-token prompt generating 100 tokens, the difference between Layer 1 and Layer 2 for the final decode step is attending 1 query against 500 cached entries versus attending 500 queries against 500 entries — roughly a 500× reduction in attention work for that step alone.

In practice, for small models like Qwen3-0.6B, the linear layers (the feed-forward network, the projection matrices) dominate the total compute time per step, and those costs are constant per token regardless of sequence length. So the observed TPOT improvement is significant but not quite the theoretical attention ratio — the improvement is most visible for long prompts and long outputs.

## The Memory Trade-Off

The cache is not free. Storing the key and value tensors for every token, at every layer, costs GPU memory. For Qwen3-0.6B with its 28 attention layers, 8 KV heads, head dimension 128, and bfloat16 precision, the cache costs approximately:

```
28 layers × 2 (K and V) × 8 heads × 128 head_dim × 2 bytes (bfloat16)
= 28 × 2 × 8 × 128 × 2 = 114,688 bytes per token ≈ 112 KB per token
```

For a sequence of 512 tokens, that is roughly 56 MB. For 2048 tokens, around 224 MB. This memory is held on the GPU for the duration of the request. In a system that handles many concurrent requests — which later layers address — the total cache memory across all active sequences becomes a significant constraint that shapes how many requests can be in flight simultaneously.

For now, with sequential request handling, this is not a bottleneck. But it is worth knowing that the TPOT improvement of the KV cache comes with a memory cost that scales with sequence length.
