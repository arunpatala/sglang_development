# Layer 2 — Summary

Layer 2 makes inference faster. The model is the same, the API is the same, and the generated output is identical — but each decode step is now dramatically cheaper than it was in Layer 1. The improvement comes from a single targeted change in `model.py`: instead of discarding the intermediate attention tensors at the end of each forward pass, we save them and hand them back to the model on the next step. `server.py` and `benchmark.py` need no changes at all.

---

## What Changed in the Decode Loop

In Layer 1, the decode loop passed the full growing sequence to the model on every step:

```python
# Layer 1 — ids grows by one column each step
for step in range(max_new_tokens):
    out = self.model(input_ids=ids, use_cache=False)
    next_token_id = sample_next_token(out.logits[0, -1, :], temperature)
    ids = torch.cat([ids, torch.tensor([[next_token_id]], device=self.device)], dim=1)
```

In Layer 2, the loop is split in two. Prefill happens once before the loop, processing the full prompt and building a cache. The decode loop then feeds only a single new token per step:

```python
# Layer 2 — prefill once, then one token per decode step
past_kv = KVCache()
out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values   # cache is now populated with all prompt K/V
next_token_id = sample_next_token(out.logits[0, -1, :], temperature)

for _ in range(max_new_tokens - 1):
    current_token = torch.tensor([[next_token_id]], device=self.device)  # always [1, 1]
    out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
    past_kv = out.past_key_values   # cache extended by one position
    next_token_id = sample_next_token(out.logits[0, -1, :], temperature)
```

`input_ids` is now always `[1, 1]` — a single token — regardless of how long the context has grown. The model reads everything it needs from `past_kv` rather than reprocessing the full sequence. The entire change in `model.py` is two new arguments (`past_key_values`, `use_cache=True`) and one assignment (`past_kv = out.past_key_values`).

---

## What Keys, Values, and Queries Are

To understand why saving these tensors works — and why the result is identical to a full forward pass — it helps to know what happens inside an attention layer.

Every transformer layer applies an attention operation. For each token in the input sequence, the layer computes three vectors by multiplying the token's current representation by three separate learned weight matrices: a **query** vector `Q`, a **key** vector `K`, and a **value** vector `V`.

The query is what a token is "asking for": a compact representation of what information this token wants to gather from the rest of the sequence. The key is what a token is "advertising": a representation of what information it can provide. The value is what a token actually contributes when it is selected: the content that gets mixed into the output.

The attention computation compares each token's query against the keys of every other token — specifically by computing the dot product `Q · Kᵀ` — to produce a set of attention scores. These scores determine how much of each token's value to include in the output at the current position. The final output at each position is a weighted sum of value vectors, with the weights coming from the attention scores after a softmax:

```
attention scores = softmax(Q · Kᵀ / √d)
output = scores · V
```

This is computed for every position in the sequence simultaneously during prefill. During a Layer 1 decode step, it is computed for all positions again — including every token already processed on prior steps — because the model receives the full growing sequence each time.

The critical observation is that the keys and values for a token depend only on what came before it, not on what comes after. Attention in a causal language model is one-directional: token `i` attends to tokens 0 through `i`, never to tokens `i+1` onwards. This means the key and value vectors for a token are fully determined the moment that token is processed and will never change on any future step. Recomputing them is redundant.

---

## The Cache as Saved K and V

The KV cache stores the key and value tensors after each forward pass. On the next decode step, instead of recomputing K and V for all prior tokens, the model reads them from the cache. It only computes fresh K and V for the single new token, then attends that token's query against the full set of cached keys and values — covering the entire context from the start of the sequence.

The attention output is exactly what a full forward pass would produce. From the model's perspective, every prior token's K and V are present; it is attending over the full context. The difference is purely in what was computed freshly versus what was read from memory. The query for the new token is compared against all cached keys, producing attention scores across the full context, and the output is a weighted sum of all cached values. Nothing about the result changes; only the amount of computation does.

---

## The `KVCache` Class

We supply our own cache implementation rather than relying on HuggingFace's built-in `DynamicCache`. The two behave identically in this layer, but having our own code means we can inspect it, log it, and replace its internals in Layer 3 without touching `model.py`.

`KVCache` holds one `LayerCache` per transformer layer — 28 for Qwen3-0.6B. Each `LayerCache` stores the accumulated key and value tensors for that layer and grows them by one position on each decode step:

```python
class LayerCache:
    def update(self, new_keys, new_values):
        if self.keys is None:
            self.keys, self.values = new_keys, new_values   # first call: prefill
        else:
            self.keys   = torch.cat([self.keys,   new_keys],   dim=-2)
            self.values = torch.cat([self.values, new_values], dim=-2)
        return self.keys, self.values   # full accumulated K and V
```

HuggingFace's attention layers call `past_kv.update(new_k, new_v, layer_idx)` and receive back the full accumulated tensors to attend over. The cache is transparent to the model internals. After prefill on a 47-token prompt, each `LayerCache` holds tensors of shape `[1, 8, 47, 128]` (batch × KV heads × sequence length × head dimension). Each decode step appends one position, growing that last dimension by 1. The server logs the cache state after prefill:

```
KVCache(layers=28, seq_len=47, memory=6.5 MB)
```

Each additional token costs approximately 112 KB across all 28 layers. A request with a 400-token prompt generating 128 tokens ends with 528 positions in the cache, occupying roughly 57 MB. For sequential request handling this is not a concern. In a concurrent system, the total cache across all active sequences becomes the primary constraint on how many requests can be in flight at once.

---

## Prefill and Decode — Now Different Code

In Layer 1, prefill and decode used the same code path — both were `self.model(input_ids=ids, use_cache=False)` with a growing `ids`. In Layer 2 they are structurally separate.

Prefill processes the full prompt in one forward pass. Its cost is unchanged: the attention computation is `O(L²)` where `L` is the prompt length, because every token attends to every other token. As a side-effect of this forward pass, every attention layer calls `past_kv.update` for every token it processes, populating the cache. By the time prefill returns, the cache holds key and value tensors for the entire prompt across all 28 layers, and the first generated token is already available.

Each decode step processes exactly one token. The input is always `[1, 1]`. Inside the forward pass, the attention layer computes fresh K and V for this new token, appends them to the cache, and then attends the new token's single query against the full accumulated keys — a linear scan over `L + k` cached entries rather than a quadratic computation over all `L + k` tokens. This is the cost reduction: one query versus `L + k` queries. The total attention work per decode step drops from `O((L + k)²)` to `O(L + k)`.

---

## TTFT and TPOT

TTFT is measured as the duration of the prefill call. Because prefill is now a separate step outside the loop, the measurement is structurally clean — there is no need to treat step 0 specially as in Layer 1.

TPOT is the average of `step_times`, which contains only decode step timings. In Layer 1, the loop's `step_times[0]` was the prefill step and had to be excluded when computing TPOT. In Layer 2, prefill never enters `step_times`, so the average is unambiguously a decode average.

The benchmark shows TPOT near-constant across requests regardless of prompt length. In Layer 1, TPOT grew with prompt length because each decode step reprocessed the full growing context. In Layer 2, each step processes one token and reads the cached context linearly. For long prompts the improvement is large — the quadratic versus linear difference becomes significant at a few hundred tokens. For short prompts the improvement is smaller, because the feed-forward and projection layers have constant cost per token regardless of cache size, and for a small context those layers dominate the per-step time.

TTFT does not improve. The prefill pass still processes the full prompt and still pays `O(L²)` attention cost. Populating the cache during prefill adds no overhead — the cache writes are a side-effect of the attention computation that was already happening.

---

## The Full Loop

Now that all the parts have been explained, it is worth tracing a single `generate()` call from end to end to see how they connect.

The call arrives with a list of messages. The tokenizer applies the chat template and encodes the result into `input_ids` of shape `[1, prompt_len]`, which is moved to the GPU.

Prefill runs first. An empty `KVCache()` is created and passed to the model alongside the full prompt. As each of the 28 attention layers processes the prompt tokens, it calls `past_kv.update(new_k, new_v, layer_idx)`, storing the key and value tensors for every prompt token at every layer. By the time the model returns, the cache holds `[1, 8, prompt_len, 128]` tensors for all 28 layers. `out.logits[0, -1, :]` extracts the last-position logit vector, `sample_next_token` draws the first generated token, and TTFT is recorded. The first token is available before the decode loop even starts.

The decode loop then runs one step at a time. Each step wraps the most recently sampled token into a `[1, 1]` tensor and calls the model with `past_key_values=past_kv`. Inside the forward pass, each attention layer computes fresh K and V for this single new token, appends them to the cache via `past_kv.update`, and attends the new token's query against the full accumulated K and V — covering all prompt tokens plus every previously generated token. The cache grows by one position per layer per step. `out.logits[0, -1, :]` gives the next-position logit vector; `sample_next_token` draws the next token; the EOS check decides whether to continue. `step_times` records the wall time of every decode step, and its average becomes TPOT.

When the loop exits — either by EOS or by reaching `max_new_tokens` — `tokenizer.decode(generated_ids, skip_special_tokens=True)` converts the accumulated token list to a string. The result dict is assembled with `text`, `prompt_tokens`, `completion_tokens`, `latency_ms`, `ttft_ms`, and `tpot_ms`, and returned to `server.py` which wraps it into a `GenerateResponse`.

---

## What Comes Next

The `torch.cat` inside `LayerCache.update` is the remaining inefficiency. Every decode step allocates a new tensor large enough to hold all previously stored entries plus one new row, copies the old data in, writes the new row, and frees the old tensor. For 128 decode steps across 28 layers, that is 3,584 allocation-and-copy operations. The old data does not change; it is just being moved.

Layer 3 eliminates this by pre-allocating a fixed-size buffer at the start of the request and writing new entries into the next available slot in-place — no allocation, no copy of existing data. The `update` interface stays identical, so `model.py` is untouched. The pattern repeats: one file changes, the benchmark shows exactly what changed, and the rest of the system is unaffected.
