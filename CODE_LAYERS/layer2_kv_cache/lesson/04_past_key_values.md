# 04 — `past_key_values` in HuggingFace

## What the Assignment Actually Does

The decode loop in section 01 contains a line that appears after every model call:

```python
past_kv = out.past_key_values
```

This line is easy to read past, but it is where the cache contract between our code and HuggingFace's model internals is expressed. Understanding what `past_key_values` is — and why this assignment is the right way to thread the cache through the loop — requires understanding how HuggingFace exposes the cache interface.

When you call a HuggingFace causal language model with `use_cache=True`, the model accepts an optional `past_key_values` argument and returns one in the output. `past_key_values` is not a tensor — it is a cache object. The model does not know or care exactly how the cache is implemented internally; it only interacts with the cache through a specific interface: a method called `update`. Each attention layer, for each forward pass, calls this method to store the key and value vectors it just computed and to retrieve the full accumulated set.

## The Interface Contract

To understand what the HuggingFace model expects from a `past_key_values` object, the clearest source is the model's attention implementation itself. In `modeling_qwen3.py`, each attention layer calls:

```python
key_states, value_states = past_key_values.update(key_states, value_states, layer_idx)
```

This single line is the entire interface contract. The attention layer passes in the new key and value tensors it just computed for the current input tokens, along with its layer index. The cache appends those new tensors to whatever it has stored so far for that layer, and returns the full accumulated key and value tensors — covering all tokens from the start of the sequence to the current position.

The attention layer then uses the returned `key_states` and `value_states` — now covering the full context — to compute the attention scores and output for the current query. From the attention layer's perspective, it is attending over the entire context as usual. The cache is transparent to it.

## How the Cache Object Flows Through the Loop

The cache object is created once before the prefill pass and passed into every subsequent call. After each forward pass, `out.past_key_values` holds a reference to the same cache object — now populated with one more token's worth of key and value tensors per layer.

```python
# Before prefill — empty cache
past_kv = KVCache()

# Prefill — cache gets filled with all prompt token K/V vectors
out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values   # same object, now has prompt_len entries per layer

# Decode step 1 — cache grows by one entry per layer
out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values   # same object, now has prompt_len + 1 entries per layer

# Decode step 2 — cache grows again
out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)
past_kv = out.past_key_values   # prompt_len + 2 entries per layer
```

The assignment `past_kv = out.past_key_values` is important, but it does not create a copy. It is just assigning the Python variable `past_kv` to refer to the same object that `out.past_key_values` refers to. The cache object itself is mutated in-place inside each forward pass by the `update` calls within each attention layer. The assignment after each step simply ensures the variable name `past_kv` continues to refer to the right object for the next call.

## HuggingFace's `DynamicCache` vs Our `KVCache`

HuggingFace ships its own default cache implementation called `DynamicCache`. If you set `use_cache=True` without passing a `past_key_values` argument, the model will instantiate a `DynamicCache` internally and return it in `out.past_key_values`. In behaviour, `DynamicCache` is very similar to our `KVCache` — it also grows by appending new key and value tensors via `torch.cat` on each step.

We supply our own `KVCache` instead of relying on `DynamicCache` for two reasons. The first is visibility: our `KVCache` is code we wrote and can read. When you call `logger.info(f"after prefill: {past_kv}")` in `model.py`, the `__repr__` of our `KVCache` prints the number of layers, the current sequence length, and the total memory it occupies — information that `DynamicCache` does not expose in a useful way. The second reason is forward compatibility: Layer 3 will improve performance by pre-allocating a fixed-size buffer instead of doing `torch.cat` on every step. That change happens entirely inside `kv_cache.py`. Because `model.py` interacts with the cache only through the `update` interface, the model code does not need to change at all when the storage strategy changes.

## Why `layer_idx` Matters

A transformer model has many attention layers — Qwen3-0.6B has 28. Each layer has its own set of weight matrices and therefore computes its own distinct key and value tensors for each token. The cache must store these separately: the key tensors from layer 0 are unrelated to the key tensors from layer 1 and must not be mixed.

The `layer_idx` argument to `update` is how the cache knows which layer's stored tensors to retrieve and update. Our `KVCache` uses a list of `LayerCache` objects indexed by `layer_idx`, initialised lazily on first access:

```python
def update(self, key_states, value_states, layer_idx, *args, **kwargs):
    while len(self._layers) <= layer_idx:
        self._layers.append(LayerCache())
    return self._layers[layer_idx].update(key_states, value_states)
```

When the model runs its first forward pass, layer 0 calls `update` first, then layer 1, and so on up to layer 27. Each call either creates a new `LayerCache` (during prefill, on the first pass) or appends to the existing one (on every subsequent step). By the end of prefill, `self._layers` has 28 entries, one per attention layer in the model.
