# 05 — The `KVCache` Class

## The Complete Implementation

The entire KV cache for Layer 2 lives in `kv_cache.py`. It is two classes: `LayerCache`, which manages the stored key and value tensors for a single transformer layer, and `KVCache`, which holds one `LayerCache` per layer and implements the interface HuggingFace's attention code expects. Here is the full structure before we walk through each part:

```
KVCache
└── self._layers: list[LayerCache]   one per transformer layer, lazy-init
    └── LayerCache
        ├── keys:   Tensor [batch, n_kv_heads, seq_len, head_dim]
        └── values: Tensor [batch, n_kv_heads, seq_len, head_dim]
```

## `LayerCache` — Storing One Layer's K and V

`LayerCache` is the inner class. It stores the accumulated key and value tensors for one attention layer and grows them by one position each decode step.

```python
class LayerCache:
    is_sliding: bool = False

    def __init__(self):
        self.keys: torch.Tensor | None = None
        self.values: torch.Tensor | None = None
```

Both `keys` and `values` start as `None`. They are not allocated on initialisation because the cache does not know the tensor shapes until the first forward pass — and more importantly, the shapes differ between models and configurations. The `is_sliding = False` class attribute tells HuggingFace's masking utilities that this layer uses full causal attention rather than a sliding window variant.

### `LayerCache.update`

```python
def update(
    self,
    new_keys: torch.Tensor,    # [batch, n_kv_heads, new_tokens, head_dim]
    new_values: torch.Tensor,  # [batch, n_kv_heads, new_tokens, head_dim]
) -> tuple[torch.Tensor, torch.Tensor]:
    if self.keys is None:
        self.keys = new_keys
        self.values = new_values
    else:
        self.keys   = torch.cat([self.keys,   new_keys],   dim=-2)
        self.values = torch.cat([self.values, new_values], dim=-2)
    return self.keys, self.values
```

On the first call (during prefill), `self.keys` is `None`, so the new tensors are stored directly — no copying needed. On every subsequent call (each decode step), the new key and value tensors are appended to the existing ones along dimension `-2`, which is the sequence length dimension. `torch.cat` allocates a new tensor each time and copies both the old data and the new entry into it.

The method returns the full accumulated tensors. This is what HuggingFace's attention layer receives back from the `update` call and uses to compute the attention output for the current query. During prefill, the returned tensors are the prompt keys and values. During each decode step, they are the prompt keys and values plus all previously generated token keys and values.

The shape comment is important: `[batch, n_kv_heads, seq_len, head_dim]`. Qwen3-0.6B uses grouped-query attention with 8 KV heads and a head dimension of 128. After a 20-token prefill, each layer's `LayerCache` holds tensors of shape `[1, 8, 20, 128]` for both keys and values. After 10 decode steps, the shape is `[1, 8, 30, 128]`.

### Mask Helper Methods

HuggingFace's attention code also calls two other methods on each layer cache to build the causal attention mask:

```python
def get_seq_length(self) -> int:
    return 0 if self.keys is None else self.keys.shape[-2]

def get_mask_sizes(self, query_length: int) -> tuple[int, int]:
    kv_length = self.get_seq_length() + query_length
    kv_offset = 0
    return kv_length, kv_offset
```

`get_seq_length` returns the number of token positions currently stored in the cache. `get_mask_sizes` returns the total span of positions the attention mask must cover and the offset into that span where the query positions begin. For a full-attention (non-sliding) layer, the offset is always 0 and the total span is the cached length plus the new query length. These values tell the masking code how large to make the attention mask so that it correctly prevents future positions from attending to positions that come after them.

## `KVCache` — The Full Cache for All Layers

`KVCache` is the outer class. It holds a list of `LayerCache` objects and implements the interface that `model.py` and HuggingFace's model code interact with.

```python
class KVCache:
    def __init__(self):
        self._layers: list[LayerCache] = []
```

The list starts empty. `LayerCache` instances are created lazily inside `update` as each attention layer calls in for the first time, which means we do not need to know the model's layer count upfront.

### `KVCache.update` — The HuggingFace Interface

```python
def update(
    self,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    layer_idx: int,
    *args,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    while len(self._layers) <= layer_idx:
        self._layers.append(LayerCache())
    return self._layers[layer_idx].update(key_states, value_states)
```

This is the method HuggingFace's attention code calls. `*args` and `**kwargs` absorb any extra arguments that different HuggingFace model versions might pass, keeping our implementation forward-compatible. The `while` loop grows `self._layers` until it has an entry at `layer_idx`, creating `LayerCache` instances as needed. Then it delegates to the appropriate `LayerCache`.

During a full prefill pass of a 28-layer model, this method is called 28 times — once per attention layer, with `layer_idx` from 0 to 27. Each call populates the corresponding `LayerCache` with the prompt's key and value tensors for that layer. During each decode step, the same 28 calls happen again, this time appending one new position to each layer's stored tensors.

### Mask Delegation and the `is_sliding` Property

```python
def get_seq_length(self, layer_idx: int = 0) -> int:
    if not self._layers or layer_idx >= len(self._layers):
        return 0
    return self._layers[layer_idx].get_seq_length()

def get_mask_sizes(self, query_length: int, layer_idx: int) -> tuple[int, int]:
    while len(self._layers) <= layer_idx:
        self._layers.append(LayerCache())
    return self._layers[layer_idx].get_mask_sizes(query_length)

@property
def is_sliding(self) -> list[bool]:
    return [layer.is_sliding for layer in self._layers]
```

These are called by HuggingFace's masking utilities to build the attention mask. `get_seq_length` is used to determine how much context the model currently has cached. `is_sliding` tells the masking code whether each layer uses sliding-window attention (none of our layers do, so this always returns a list of `False` values). Together they allow the model's internal mask-building logic to operate correctly with our custom cache.

## Memory Accounting

```python
def memory_bytes(self) -> int:
    total = 0
    for layer in self._layers:
        if layer.keys is not None:
            total += layer.keys.nbytes + layer.values.nbytes
    return total
```

`memory_bytes` iterates over all layer caches and sums the byte sizes of the stored tensors. This is what appears in the server log after prefill:

```
KVCache(layers=28, seq_len=47, memory=6.5 MB)
```

The memory grows linearly with sequence length. After a 47-token prefill, with 28 layers, 8 KV heads, 128 head dimension, and bfloat16 (2 bytes), the expected size is:

```
28 layers × 2 (K and V) × 8 heads × 128 head_dim × 47 tokens × 2 bytes = 6,766,592 bytes ≈ 6.5 MB
```

Each additional decode step adds one token, growing the cache by:

```
28 × 2 × 8 × 128 × 1 × 2 = 114,688 bytes ≈ 112 KB per token
```

For a request that generates 100 tokens, the cache grows from its post-prefill size by about 11 MB. This is manageable for a single sequential request. When Layer 5 introduces batching — serving many requests concurrently — the total cache across all active sequences becomes the primary GPU memory constraint, and managing it efficiently becomes the central engineering challenge.

## What Layer 3 Will Change Here

The `torch.cat` inside `LayerCache.update` is the remaining performance cost of the current implementation. Every decode step allocates a new tensor of size `[1, 8, seq_len+1, 128]`, copies all `seq_len` previously stored positions into it, and then discards the old tensor. This allocation-and-copy pattern is wasteful: the old data does not change, but it has to be moved in memory every step.

Layer 3 eliminates this by pre-allocating a fixed-size buffer at the start of the request and writing new key and value vectors into the next available slot in-place, with no copying of existing data. The interface — `update(new_k, new_v)` returning the full accumulated tensors — stays identical. Only the internals of `LayerCache` change, and `model.py` is untouched.
