# Layer 7 — Lesson Outline

## What This Lesson Covers

Layer 6 eliminated head-of-line blocking with a continuous-batching scheduler. But every decode step, `BatchedKVCache` left-padded every active request's KV history to `max_kv_len` and called `F.scaled_dot_product_attention` on the resulting rectangular `[B, heads, max_kv_len, dim]` tensor. When one request has 500 tokens of history and fifteen others have 3, roughly 97% of the attention compute is wasted on padding zeros that the mask immediately discards.

Layer 7 replaces `BatchedKVCache` with `PackedKVCache`. Instead of padding to a common length, it concatenates all request KV histories back-to-back into a single ragged tensor and passes an `indptr` array that tells FlashInfer where each request's slice begins and ends. FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` then attends only over real tokens — no zero columns, no wasted compute. The attention layer in `model/attention.py` grows a backend-dispatch table: `PerReqKVCache` → `F.sdpa`, `PackedKVCache` → FlashInfer.

The change touches two files: `kv_cache.py` (new `PackedKVCache` class) and `model/attention.py` (FlashInfer dispatch path). `model_runner.py` also changes slightly: it pre-allocates a 256 MB workspace tensor at init and calls `pack_kv.plan()` / `pack_kv.end_forward()` around the forward pass. `scheduler.py`, `request.py`, `server.py`, `tokenizer.py`, and `prefill()` are entirely unchanged.

The sections follow the decode path top to bottom: the structural shift from padded to packed, the `PackedKVCache` data layout and indptr arithmetic, the attention backend dispatch, the full `decode_step` call sequence, and the remaining copy-cost problem that Layer 8 solves.

---

## Sections

### 01 — From Padded to Packed (`01_from_padded_to_packed.md`)
- Layer 6's `decode_step` with `BatchedKVCache`: left-pad all KV caches to `max_kv_len`, call `F.sdpa` on `[B, heads, max_kv_len, dim]`; the compute and memory cost of the padding waste
- Layer 7's `decode_step` with `PackedKVCache`: no `attn_mask` passed to model, `attention_mask=None` because FlashInfer uses `indptr` for masking; `plan()` called once before the 28-layer forward; `end_forward()` called after
- The one structural addition: `self._workspace` pre-allocated at `ModelRunner.__init__` — a 256 MB `torch.uint8` tensor reused every decode step so FlashInfer does not allocate temp buffers mid-step
- What is unchanged: `prefill()` is identical (B=1, `F.sdpa`, `PerReqKVCache`); position IDs per request are still `kv_len_i`; sampling, `output_ids`, and finished-request handling are identical

### 02 — The Packed KV Cache (`02_packed_kv_cache.md`)
- The ragged packing layout: B=3 requests with `kv_lens=[10, 6, 4]` — Layer 6 allocates `[3, kv, 10, d]`; Layer 7 allocates `[21, kv, d]` with no wasted columns (diagram from `kv_cache.py` docstring)
- `qo_indptr [B+1]`: each request contributes exactly 1 query token during decode; `torch.arange(B+1)` gives `[0, 1, 2, ..., B]`
- `kv_indptr [B+1]`: cumulative sum of `(kv_len_i + 1)` — the `+1` accounts for the new decode token appended inside `update()` before FlashInfer sees the tensor; `kv_indptr[i..i+1]` is the slice for request `i`
- `PackedKVCache.__init__`: computes both indptrs and creates `flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")`
- `PackedKVCache.plan()`: calls `wrapper.begin_forward(qo_indptr, kv_indptr, num_q_heads, num_kv_heads, head_dim, causal=False, q_data_type=dtype)` — called once, reused by all 28 attention layers in the same decode step
- `PackedKVCache.get_seq_length()` returns 0 — the model's `past_len` offset is not used for RoPE or mask construction in the FlashInfer path; `pos_ids` is passed explicitly per request

### 03 — Packing and Attending (`03_packing_and_attending.md`)
- `PackedKVCache.update(layer_idx, new_k, new_v)`: called by each attention layer during the forward pass; for each request, reshapes historical K from `[1, n_kv, L_i, dim]` to `[L_i, n_kv, dim]` (NHD layout), appends the new decode token `new_k[i].unsqueeze(0)` — shape `[1, n_kv, dim]`; concatenates all segments into `[total_kv_tokens, n_kv, dim]`
- The `+1` in `kv_indptr` revisited: the concatenation in `update()` places the new token at the end of each request's slice, so `kv_indptr[i+1] - kv_indptr[i] = kv_len_i + 1` — FlashInfer attends over `history + new token` for each request
- `PackedKVCache.write_back()`: after the full forward pass, `_new_k[layer_idx]` holds `[B, n_kv, dim]` — one new token per request per layer; reshaped to `[1, n_kv, 1, dim]` and appended to each `req.kv_cache` (same `torch.cat` on `dim=2` as Layer 6)
- `PackedKVCache.end_forward()`: calls `wrapper.end_forward()` to release FlashInfer's internal state and free any workspace allocations made during `begin_forward`

### 04 — The Attention Dispatch (`04_the_attention_dispatch.md`)
- `Qwen3Attention.forward` now dispatches on `hasattr(kv_cache, "wrapper")`: `PackedKVCache` exposes a `wrapper` property; `PerReqKVCache` does not — this is the duck-type check that selects the kernel
- The FlashInfer path: reshape `q/k/v` from `[B, heads, 1, dim]` to `[B, heads, dim]` (squeeze the seq dim, NHD layout); call `kv_cache.update(layer_idx, k_fi, v_fi)` to get `(k_packed, v_packed)` of shape `[total_kv_tokens, n_kv_heads, dim]`; call `kv_cache.wrapper.forward(q_fi, k_packed, v_packed)` — output `[B, n_q_heads, dim]`; unsqueeze back to `[B, n_q_heads, 1, dim]`
- GQA natively: FlashInfer handles the 16 Q heads / 8 KV heads ratio internally — no `repeat_kv` expansion needed, unlike the `F.sdpa` path
- `attention_mask` ignored on the FlashInfer path: `causal=False` with `q_len=1` per request means there are no future tokens to mask; `kv_indptr` handles the ragged boundaries
- The F.sdpa path is unchanged: `PerReqKVCache.update()` appends and returns rectangular KV; `repeat_kv` expands heads; `F.sdpa` runs with an additive mask as in Layers 5 and 6

### 05 — The Full Loop (`05_the_full_loop.md`)
- End-to-end trace of one decode step with three concurrently running requests, naming every component in execution order
- Step 1 — Request arrival and prefill: server enqueues `Req`, scheduler prefills with `PerReqKVCache` + `F.sdpa` (unchanged from Layer 6); `req.kv_cache` populated; first token sampled; request enters `_running`
- Step 2 — `decode_step` entry: `kv_lens` collected; `last_toks [B, 1]` and `pos_ids [B, 1]` built per request; `PackedKVCache(reqs, workspace)` constructed; `indptr` arithmetic computed
- Step 3 — `plan()` and forward: `plan()` calls `begin_forward` once; 28 attention layers each call `attention.forward` → `hasattr(wrapper)` → `update()` → `wrapper.forward()` — FlashInfer attends over ragged real tokens only
- Step 4 — `write_back()` and `end_forward()`: new token K/V appended to each `PerReqKVCache`; FlashInfer state released; sampling per request; newly finished requests returned to scheduler; scheduler resolves their `asyncio.Future`s

### 06 — What Comes Next (`06_whats_next.md`)
- The remaining cost: `PackedKVCache.update()` gathers all per-request historical K/V tensors every decode step — a `torch.cat` of all segments into a new contiguous buffer; this is an `O(total_kv_tokens)` copy that grows with the number of active tokens across all requests
- Memory fragmentation persists: `PerReqKVCache` still grows by `torch.cat` per decode step, causing repeated allocations and fragmentation of GPU memory
- Layer 8 (paged KV cache): physical KV memory is divided into fixed-size pages; each request is assigned pages from a global pool rather than owning a contiguous tensor; FlashInfer reads directly from the page table without any gather copy
- What changes: `kv_cache.py` becomes a block-table-based allocator; `model_runner.py` manages page allocation and a `BlockManager`; `PackedKVCache` is replaced by a paged variant; `attention.py` dispatch table gains a third entry — `PagedKVCache` → FlashInfer paged kernel

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 7 concepts to SGLang source: `PackedKVCache` → `ModelRunner.forward_batch` with `flashinfer.BatchPrefillWithRaggedKVCacheWrapper`; indptr arrays → `req_to_token_pool`; workspace → SGLang's preallocated FlashInfer buffer; attention dispatch → `Attention.forward` in `srt/layers/attention/`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Workspace pre-allocation | `model_runner.py` line 81: `self._workspace = torch.empty(_WORKSPACE_MB * 1024 * 1024, dtype=torch.uint8, device=DEVICE)` |
| `PackedKVCache` class | `kv_cache.py` line 107: `class PackedKVCache:` |
| `qo_indptr` construction | `kv_cache.py` line 131: `self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEVICE)` |
| `kv_indptr` construction | `kv_cache.py` line 139: `self.kv_indptr = torch.tensor(kv_cumsum, dtype=torch.int32, device=DEVICE)` |
| FlashInfer wrapper init | `kv_cache.py` line 142: `flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")` |
| `plan()` / `begin_forward` | `kv_cache.py` line 165: `self._wrapper.begin_forward(self.qo_indptr, self.kv_indptr, ...)` |
| `update()` — ragged pack | `kv_cache.py` line 184: `def update(self, layer_idx, new_k, new_v)` |
| NHD reshape | `kv_cache.py` line 214: `hist_k.squeeze(0).permute(1, 0, 2).contiguous()` |
| `write_back()` | `kv_cache.py` line 235: `def write_back(self)` |
| `end_forward()` | `kv_cache.py` line 257: `self._wrapper.end_forward()` |
| `decode_step` — `plan()` call | `model_runner.py` line 183: `pack_kv.plan(num_q_heads=cfg.num_attention_heads, ...)` |
| Forward with `attention_mask=None` | `model_runner.py` line 190: `# Forward pass — attention_mask=None because FlashInfer uses indptr` |
| Backend dispatch check | `model/attention.py` line 134: `if kv_cache is not None and hasattr(kv_cache, "wrapper"):` |
| FlashInfer forward call | `model/attention.py` line 150: `attn_out = kv_cache.wrapper.forward(q_fi, k_packed, v_packed)` |
