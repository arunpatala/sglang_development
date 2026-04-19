# 03 — Paged KVPool and Prefill

`KVPool` is the GPU memory store that holds every K and V vector for every active token, across all attention layers. Layer 8's pool stored one token per slot. Layer 9 extends each slot to hold a full page of `page_size` consecutive tokens. This single shape change propagates to `PrefillKVCtx.store` — which must now pack prompt K/V into pages — and to `DecodeKVCtx.store` — which must write into a specific (page, offset) cell rather than a flat row.

---

## The Pool Shape

```python
class KVPool:
    def __init__(
        self,
        total_pages: int,
        page_size: int,
        n_layers: int,
        n_kv_heads: int,
        head_dim: int,
        dtype: torch.dtype,
    ) -> None:
        self.k_pool: List[torch.Tensor] = [
            torch.zeros(total_pages, page_size, n_kv_heads, head_dim,
                        dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]
        self.v_pool: List[torch.Tensor] = [
            torch.zeros(total_pages, page_size, n_kv_heads, head_dim,
                        dtype=dtype, device=DEVICE)
            for _ in range(n_layers)
        ]
        self.free_slots: List[int] = list(range(1, total_pages))
```

Each `k_pool[layer]` tensor is `[total_pages, page_size, n_kv_heads, head_dim]`. Layer 8's shape was `[total_slots, n_kv_heads, head_dim]` — equivalently, it was a pool of `total_slots` single-token pages. Layer 9 makes `page_size` an explicit dimension rather than an implicit 1. FlashInfer's `BatchDecodeWithPagedKVCacheWrapper` expects this exact four-dimensional layout and no longer requires the `unsqueeze(1)` that Layer 8 used to manufacture the missing page dimension.

`total_pages` is sized from available GPU memory after model load:

```python
free_bytes, _ = torch.cuda.mem_get_info()
bytes_per_token = (
    cfg.num_hidden_layers * 2
    * cfg.num_key_value_heads
    * cfg.head_dim
    * (torch.finfo(DTYPE).bits // 8)
)
max_pages = int(free_bytes * kv_memory_fraction / (page_size * bytes_per_token))
```

The formula divides the available memory by `page_size × bytes_per_token`. For a given GPU, the total token capacity `max_pages × page_size` is the same as Layer 8's `max_tokens` — the same physical memory is used. The number of logical pool entries shrinks by a factor of `page_size`. With `page_size=16` and 6 GB free after weight load, `max_pages ≈ 3500` pages, each holding 16 tokens across 28 layers.

Page 0 is reserved as all-zeros. FlashInfer's kernel uses page index 0 as a sentinel: any slot in `kv_indices` that points to page 0 reads zeros, which contribute nothing to the softmax output. No real token is ever allocated to page 0; `free_slots` starts at 1.

---

## `alloc`: Pages, Not Tokens

```python
def alloc(self, n_tokens: int) -> List[int]:
    n_pages = self.n_pages_for(n_tokens)   # math.ceil(n_tokens / page_size)
    if n_pages > len(self.free_slots):
        raise RuntimeError(...)
    pages = self.free_slots[:n_pages]
    self.free_slots = self.free_slots[n_pages:]
    return pages
```

`alloc(n_tokens)` returns `ceil(n_tokens / page_size)` page indices. For a 50-token prompt with `page_size=16`, it returns `ceil(50/16) = 4` entries. The caller holds a list of page indices, not token indices. `free(page_indices)` extends `free_slots` with all of them in one call when the request finishes.

---

## Prefill: From Prompt K/V to Pages

The prefill entry point allocates pages and writes them to `ReqToTokenPool` before the forward pass:

```python
def prefill(self, req: Req) -> None:
    prompt_len = len(req.input_ids)
    n_pages    = math.ceil(prompt_len / P)

    page_indices     = self.kv_pool.alloc(prompt_len)
    req.slot_indices = page_indices

    req.req_pool_idx = self.req_to_token_pool.alloc()
    pages_t = torch.tensor(page_indices, dtype=torch.int32, device=DEVICE)
    self.req_to_token_pool.req_to_token[req.req_pool_idx, :n_pages] = pages_t

    ctx = PrefillKVCtx(page_indices, self.kv_pool)
    fb  = ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=ctx, attention_mask=mask)
    logits = self.model(ids, forward_batch=fb, position_ids=pos)
```

The sequence of operations before the model call has gained two lines versus Layer 8: `req_to_token_pool.alloc()` and the slice assignment. After the slice assignment, the GPU table already holds the definitive page index list for this request. Everything the Triton kernel needs is on the GPU before the first decode step.

`PrefillKVCtx` receives the page index list and the pool reference:

```python
class PrefillKVCtx:
    def __init__(self, page_indices: List[int], kv_pool: KVPool) -> None:
        self._kv_pool = kv_pool
        self._page_t  = torch.tensor(page_indices, dtype=torch.int64, device=DEVICE)
        self._n_pages = len(page_indices)
```

`_page_t` is the int64 tensor of page indices, built once in `__init__` and reused by all 28 `store` calls during the forward pass. Building it once avoids 27 redundant CPU-to-GPU transfers.

---

## `PrefillKVCtx.store`: Packing Prompt K/V into Pages

```python
def store(self, layer_idx: int, k: torch.Tensor, v: torch.Tensor) -> None:
    P   = self._kv_pool.page_size
    L   = k.shape[2]

    k_nhd = k.squeeze(0).permute(1, 0, 2).contiguous()   # [L, n_kv, D]
    v_nhd = v.squeeze(0).permute(1, 0, 2).contiguous()

    pad = self._n_pages * P - L
    if pad > 0:
        k_nhd = F.pad(k_nhd, (0, 0, 0, 0, 0, pad))
        v_nhd = F.pad(v_nhd, (0, 0, 0, 0, 0, pad))

    k_paged = k_nhd.view(self._n_pages, P, n_kv, D)
    v_paged = v_nhd.view(self._n_pages, P, n_kv, D)

    self._kv_pool.k_pool[layer_idx][self._page_t] = k_paged
    self._kv_pool.v_pool[layer_idx][self._page_t] = v_paged
```

The `k` tensor arrives from `Qwen3Attention` in shape `[1, n_kv_heads, prompt_len, head_dim]`. The pool rows expect `[n_pages, page_size, n_kv_heads, head_dim]`. The transform has four steps.

`squeeze(0).permute(1, 0, 2)` gives `[prompt_len, n_kv_heads, head_dim]` — the NHD token order. `contiguous()` ensures the layout is materialized before padding or reshape; without it, the subsequent `view` might fail on a non-contiguous tensor.

`F.pad` adds `pad = n_pages × P - prompt_len` zero-filled rows at the end. For a 50-token prompt with `page_size=16`, `n_pages=4`, `pad = 64 - 50 = 14`. The last page has 14 zero rows that will never be read: `kv_last_page_lens` will tell FlashInfer that only 2 tokens (`50 mod 16 = 2`) in that last page are valid. The zeros are harmless placeholders, not a correctness concern.

`view(n_pages, P, n_kv, D)` reshapes the padded `[n_pages × P, n_kv, D]` into the final page-structured layout without any data copy — `view` is zero-copy when the tensor is contiguous.

`k_pool[layer_idx][self._page_t] = k_paged` is PyTorch's advanced index assignment. `_page_t` is a `[n_pages]` int64 tensor of row indices; this scatters all `n_pages` pages into the pool in a single GPU operation. After `store` returns, the prompt's K/V is permanently in the pool, addressable by page index. No write-back is needed.

---

## Why Prefill Still Uses `F.sdpa`

Inside `PagedBackend._prefill_forward`, the pool write in `kv.store(layer_idx, k, v)` is a side-effect that runs before `F.sdpa`:

```python
if kv is not None:
    kv.store(layer_idx, k, v)   # pool write

k_rep = repeat_kv(k, self.num_kv_groups)
v_rep = repeat_kv(v, self.num_kv_groups)
return F.scaled_dot_product_attention(q, k_rep, v_rep, attn_mask=additive_mask, scale=self.scale)
```

The attention computation runs over the fresh `k` and `v` tensors from the linear projection, not over the pool. The pool write and the attention computation use the same data independently. `BatchDecodeWithPagedKVCacheWrapper` is designed for the case where every query attends over K/V that already exists in the pool. During prefill, no prior KV exists — the prompt tokens are attending over each other — so `F.sdpa` with a causal mask over the prompt K/V is the correct and efficient kernel. The pool write is a deposit into the cache that future decode steps will draw on.
