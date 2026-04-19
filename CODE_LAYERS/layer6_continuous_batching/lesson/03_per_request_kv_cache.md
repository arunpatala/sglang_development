# 03 — Per-Request KV Cache

The decode step (section 05) needs a single rectangular `[B, heads, kv_len, dim]` tensor for `F.sdpa`, but every request in `_running` has accumulated a different amount of KV history because they arrived and were prefilled at different times. Layer 6 solves this with two complementary classes: `PerReqKVCache` stores each request's history independently, and `BatchedKVCache` temporarily pads and stacks those histories into the rectangular shape the model requires for one forward pass.

Layer 6 introduces two KV cache classes: `PerReqKVCache` and `BatchedKVCache`. They solve complementary problems. `PerReqKVCache` gives every request its own independently-growing store of accumulated key and value tensors. `BatchedKVCache` temporarily packs those stores into a rectangular tensor so that a single batched forward pass can process all active requests at once. Understanding why both are needed requires understanding why the Layer 5 single-class design breaks down when requests have different lengths.

---

## Why Layer 5's `KVCache` Is Not Enough

In Layer 5, a single `KVCache` was created per `generate_batch` call and shared by all B requests:

```python
# Layer 5 — one shared cache for all B requests
kv = KVCache()
logits = self.model(input_ids, kv_cache=kv, ...)
```

The cache accumulated K and V tensors of shape `[B, n_kv_heads, seq_len, head_dim]` across all layers. This worked because all B requests had the same KV length at every step: they were prefilled together and decoded in lockstep.

In Layer 6, requests arrive and are prefilled independently. By the time a second request joins the decode batch, the first may already have 150 tokens of KV history. A third arriving later may have 0. The KV lengths are ragged — different for every request. A single shared rectangular tensor cannot represent this; `F.scaled_dot_product_attention` requires all batch rows to have the same sequence length.

---

## `PerReqKVCache`

```python
class PerReqKVCache:

    def __init__(self) -> None:
        self._k: Dict[int, torch.Tensor] = {}   # layer_idx → [1, n_kv, seq_len, head_dim]
        self._v: Dict[int, torch.Tensor] = {}

    def update(self, layer_idx: int, new_k: torch.Tensor, new_v: torch.Tensor):
        if layer_idx in self._k:
            self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
            self._v[layer_idx] = torch.cat([self._v[layer_idx], new_v], dim=2)
        else:
            self._k[layer_idx] = new_k
            self._v[layer_idx] = new_v
        return self._k[layer_idx], self._v[layer_idx]
```

`PerReqKVCache` is indexed by layer. Each entry has shape `[1, n_kv_heads, seq_len, head_dim]` — the batch dimension is always 1, because this cache belongs to exactly one request. During prefill, the attention layer for layer `i` calls `update(i, k, v)` with `k` of shape `[1, n_kv_heads, prompt_len, head_dim]`. During each decode step, `update` is called with `k` of shape `[1, n_kv_heads, 1, head_dim]` and the cache grows by one position via `torch.cat` on `dim=2`.

This design deliberately favors simplicity over efficiency. `torch.cat` allocates a new tensor on every decode step, copying the old content each time. For a 500-token output across 28 layers in `bfloat16`, this adds up to a non-trivial volume of allocations. Production systems preallocate a fixed-size buffer and write new tokens in-place — Layer 7 addresses this with paged KV cache. For now, the growth-by-concatenation pattern is correct and straightforward.

`get_seq_length()` returns the current sequence length by reading the `dim=2` size of any stored tensor. `BatchedKVCache` calls this to compute the padding needed for each request during a decode step.

---

## `BatchedKVCache`

```python
class BatchedKVCache:

    def __init__(self, reqs: list, max_kv_len: int) -> None:
        self._reqs       = reqs
        self._max_kv_len = max_kv_len
        self._k: Dict[int, torch.Tensor] = {}
        self._v: Dict[int, torch.Tensor] = {}
        self._initialized: Set[int] = set()
```

`BatchedKVCache` is constructed fresh at the start of every decode step, from the current list of running requests and the maximum KV length among them. It does not hold state between steps. Its job is to present a rectangular `[B, n_kv_heads, max_kv_len, head_dim]` view of ragged per-request caches to the model's attention layers, then dissolve.

**Lazy initialisation.** Rather than iterating all 28 layers upfront when the object is constructed, `BatchedKVCache` initialises each layer on first contact. The model's forward pass calls `update(layer_idx, new_k, new_v)` in layer order. The first call for layer `i` triggers `_init_layer(i)`:

```python
def _init_layer(self, layer_idx: int) -> None:
    ks, vs = [], []
    for req in self._reqs:
        rk = req.kv_cache._k[layer_idx]   # [1, n_kv, kv_len_i, head_dim]
        rv = req.kv_cache._v[layer_idx]
        pad = self._max_kv_len - rk.shape[2]
        if pad > 0:
            rk = F.pad(rk, (0, 0, pad, 0))   # left-pad the sequence dimension
            rv = F.pad(rv, (0, 0, pad, 0))
        ks.append(rk)
        vs.append(rv)
    self._k[layer_idx] = torch.cat(ks, dim=0)   # [B, n_kv, max_kv_len, head_dim]
    self._v[layer_idx] = torch.cat(vs, dim=0)
```

The padding is applied on the **left** of the sequence dimension (`dim=-2`). `F.pad` takes padding amounts in reverse dimension order: `(0, 0, pad, 0)` means "pad dim=-2 with `pad` zeros on the left and 0 on the right; pad dim=-1 with 0 on both sides." Short-history requests get zeros prepended to match `max_kv_len`. The attention mask built in `decode_step` marks those zero positions as padding so attention ignores them — section 05 shows the mask construction.

After `_init_layer` runs, `update` appends the new token's K/V to the batched tensor:

```python
self._k[layer_idx] = torch.cat([self._k[layer_idx], new_k], dim=2)
# shape: [B, n_kv, max_kv_len + 1, head_dim]
```

The attention layer receives this `[B, n_kv, max_kv_len + 1, head_dim]` tensor and runs `F.sdpa` over the full batch in one kernel call.

---

## `write_back`

After the model's forward pass completes, `BatchedKVCache.write_back()` updates each request's `PerReqKVCache` with the new token's K/V:

```python
def write_back(self) -> None:
    for layer_idx in self._initialized:
        full_k = self._k[layer_idx]   # [B, n_kv, max_kv_len+1, head_dim]
        full_v = self._v[layer_idx]
        for i, req in enumerate(self._reqs):
            new_k = full_k[i : i + 1, :, -1:, :]   # [1, n_kv, 1, head_dim]
            new_v = full_v[i : i + 1, :, -1:, :]
            cache = req.kv_cache
            cache._k[layer_idx] = torch.cat([cache._k[layer_idx], new_k], dim=2)
            cache._v[layer_idx] = torch.cat([cache._v[layer_idx], new_v], dim=2)
```

`full_k[i : i+1, :, -1:, :]` slices the last position along `dim=2` — the new token's key, not the padded historical keys — and inserts it into `req.kv_cache`. Only the one new position is appended; the rest of `full_k` (including the padding) is discarded with the `BatchedKVCache` object itself.

This completes the data flow: `PerReqKVCache` → pad and stack → `BatchedKVCache` → forward pass → slice last position → `write_back` → `PerReqKVCache`. Each decode step extends every active request's private cache by exactly one position, regardless of how different the cache lengths are across requests.
