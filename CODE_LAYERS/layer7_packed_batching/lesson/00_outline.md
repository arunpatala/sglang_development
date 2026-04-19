# Layer 7 — Lesson Outline

## What This Lesson Covers

Layer 6 eliminated head-of-line blocking with a continuous-batching scheduler. But every decode step, `BatchedKVCache` left-padded every active request's KV history to `max_kv_len` and called `F.scaled_dot_product_attention` on the resulting rectangular `[B, heads, max_kv_len, dim]` tensor. When one request has 500 tokens of history and fifteen others have 3, roughly 97% of the attention compute is wasted on padding zeros that the mask immediately discards.

Layer 7 removes that waste in two coordinated moves. First, `BatchedKVCache` is replaced by `PackedKVCache`, which concatenates all request KV histories back-to-back into a single ragged tensor and passes `indptr` arrays that tell FlashInfer where each request's slice begins and ends. FlashInfer's `BatchPrefillWithRaggedKVCacheWrapper` then attends only over real tokens — no zero columns, no wasted compute.

Second, the attention backend selection is lifted out of `attention.py` and made explicit. Layer 6 dispatched by duck-typing the cache object (`hasattr(kv_cache, "wrapper")`). Layer 7 introduces three new structures that mirror SGLang's design: an `AttnBackend` enum in `model/config.py` that names the backend, a `ForwardBatch` dataclass in `forward_batch.py` that carries per-call state (mode, kv_cache, mask) through the model stack, and backend objects (`SDPABackend`, `FlashInferBackend`) in `model/backend.py` that implement a single `forward()` method. `Qwen3Attention` stores one backend object at `__init__` and calls `self.backend.forward(q, k, v, layer_idx, forward_batch)` — no `if/else` in the attention layer itself. Adding a new backend (paged, tensor-parallel) means writing a new class in `backend.py`; `attention.py` never changes again.

The change touches five files: `kv_cache.py` (new `PackedKVCache`), `forward_batch.py` (new), `model/config.py` (`AttnBackend` enum), `model/backend.py` (new), and `model/attention.py` (now a thin wrapper). `model_runner.py` changes slightly — it pre-allocates a 256 MB workspace tensor, constructs `ForwardBatch` objects, and accepts an `attn_backend` argument. `server.py` reads `config.yml` and passes `attn_backend` through to `ModelRunner`. `scheduler.py`, `request.py`, `tokenizer.py`, and `prefill()` are entirely unchanged.

The sections follow the decode path top to bottom: the structural shift from padded to packed, the `PackedKVCache` data layout and indptr arithmetic, the update and write-back mechanics, the new backend-object dispatch, the full decode step call sequence, and the remaining copy-cost problem that Layer 8 solves.

---

## Sections

### 01 — From Padded to Packed (`01_from_padded_to_packed.md`)
- Layer 6's `decode_step` with `BatchedKVCache`: left-pad all KV caches to `max_kv_len`, build an attention mask, call `F.sdpa` on `[B, heads, max_kv_len, dim]`; the compute and memory cost of the padding waste
- Layer 7's `decode_step` with `PackedKVCache`: constructs `ForwardBatch(mode=DECODE, kv_cache=pack_kv, attention_mask=None)` — `attention_mask=None` because FlashInfer uses `indptr`, not a mask tensor; `plan()` called once before the 28-layer forward; `end_forward()` called after
- The one structural addition: `self._workspace` pre-allocated at `ModelRunner.__init__` — a 256 MB `torch.uint8` tensor reused every decode step; FlashInfer places its temp buffers inside it during `begin_forward`, avoiding mid-step allocations
- What is unchanged: `prefill()` is identical (B=1, `F.sdpa`, `PerReqKVCache`); position IDs per request are still `kv_len_i`; sampling, `output_ids`, and finished-request handling are identical

### 02 — The Packed KV Cache (`02_packed_kv_cache.md`)
- The ragged packing layout: B=3 requests with `kv_lens=[10, 6, 4]` — Layer 6 allocates `[3, kv, 10, d]`; Layer 7 allocates `[21, kv, d]` with no wasted columns (diagram from `kv_cache.py` docstring)
- `qo_indptr [B+1]`: each request contributes exactly 1 query token during decode; `torch.arange(B+1)` gives `[0, 1, 2, ..., B]`
- `kv_indptr [B+1]`: cumulative sum of `(kv_len_i + 1)` — the `+1` accounts for the new decode token appended inside `update()` before FlashInfer sees the tensor; `kv_indptr[i..i+1]` is the slice for request `i`
- `PackedKVCache.__init__`: computes both indptrs and creates `flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")`
- `PackedKVCache.plan()`: calls `wrapper.begin_forward(qo_indptr, kv_indptr, num_q_heads, num_kv_heads, head_dim, causal=False, q_data_type=dtype)` — called once, reused by all 28 attention layers in the same decode step

### 03 — Packing and Attending (`03_packing_and_attending.md`)
- `PackedKVCache.update(layer_idx, new_k, new_v)`: called once per attention layer; for each request, reshapes historical K from `[1, n_kv, L_i, dim]` to `[L_i, n_kv, dim]` (NHD layout), appends the new decode token `new_k[i].unsqueeze(0)` — shape `[1, n_kv, dim]`; concatenates all segments into `[total_kv_tokens, n_kv, dim]`
- The `+1` in `kv_indptr` revisited: the new token is appended inside `update()` before FlashInfer sees the ragged tensor; `kv_indptr[i+1] - kv_indptr[i] = kv_len_i + 1`
- `PackedKVCache.write_back()`: after the full forward pass, `_new_k[layer_idx]` holds `[B, n_kv, dim]`; reshaped to `[1, n_kv, 1, dim]` and appended to each `req.kv_cache` via `torch.cat` on `dim=2`
- `PackedKVCache.end_forward()`: calls `wrapper.end_forward()` to release FlashInfer's internal state before the next decode step

### 04 — The Attention Dispatch (`04_the_attention_dispatch.md`)
- Why the old `hasattr(kv_cache, "wrapper")` duck-type check was replaced: dispatch inferred from the cache type couples `attention.py` to every cache implementation; each new backend required a new branch
- `AttnBackend` enum in `model/config.py`: `SDPA` and `FLASHINFER`; set from `config.yml` via `ModelRunner.__init__` → `Qwen3ForCausalLM.from_pretrained(attn_backend=...)`; the backend is selected once at model load, not per forward call
- `ForwardBatch` dataclass in `forward_batch.py`: carries `mode` (`ForwardMode.PREFILL` / `DECODE`), `kv_cache`, and `attention_mask`; replaces the two separate arguments `(kv_cache, attention_mask)` throughout the model stack
- `model/backend.py`: `SDPABackend` (all paths use `F.sdpa`) and `FlashInferBackend` (prefill → `F.sdpa`; decode → FlashInfer ragged); `make_backend(config)` factory called once in `Qwen3Attention.__init__`
- `Qwen3Attention.forward()` is now a clean thin wrapper: QKV proj → QK norm → transpose → RoPE → `self.backend.forward(q, k, v, self.layer_idx, forward_batch)` → merge heads → output proj; no `if/else` dispatch in the attention layer

### 05 — The Full Loop (`05_the_full_loop.md`)
- End-to-end trace of one decode step with two concurrently running requests, naming every component in execution order
- Step 1 — Request arrival and prefill: server enqueues `Req`, scheduler prefills with `ForwardBatch(mode=PREFILL, kv_cache=PerReqKVCache(), attention_mask=mask)`; first token sampled; request enters `_running`
- Step 2 — `decode_step` entry: `kv_lens` collected; `last_toks [B, 1]` and `pos_ids [B, 1]` built per request; `PackedKVCache(reqs, workspace)` constructed; `plan()` calls `begin_forward` once
- Step 3 — Forward with `ForwardBatch(mode=DECODE, kv_cache=pack_kv, attention_mask=None)`: 28 attention layers each call `self.backend.forward(...)` → inside `FlashInferBackend.forward()`, `update()` builds the ragged tensor, `wrapper.forward()` attends over real tokens only
- Step 4 — `write_back()` and `end_forward()`: new token K/V appended to each `PerReqKVCache`; FlashInfer state released; sampling per request; newly finished requests returned to scheduler

### 06 — What Comes Next (`06_whats_next.md`)
- The remaining cost: `PackedKVCache.update()` gathers all per-request historical K/V tensors every decode step — a `torch.cat` of all segments into a new contiguous buffer; this is an `O(total_kv_tokens)` copy that grows with the number of active tokens across all requests
- Memory fragmentation persists: `PerReqKVCache` still grows by `torch.cat` per decode step, causing repeated allocations and fragmentation of GPU memory
- Layer 8 (paged KV cache): physical KV memory is divided into fixed-size pages; each request is assigned pages from a global pool; FlashInfer reads directly from the page table without any gather copy
- The backend-object design makes this straightforward to add: a new `PagedKVCacheBackend` class in `backend.py`, a new `PAGED` variant in `AttnBackend`, and a new entry in `make_backend()` — `attention.py` stays untouched

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `forward_batch.py` — `ForwardMode` enum and `ForwardBatch` dataclass; mirrors SGLang's `forward_batch_info.py`
- `model/backend.py` — `SDPABackend`, `FlashInferBackend`, `make_backend()`; mirrors SGLang's `srt/layers/attention/` backend classes
- `sglang_reference.md` — maps Layer 7 concepts to SGLang source: `PackedKVCache` → `ModelRunner.forward_batch` with `flashinfer.BatchPrefillWithRaggedKVCacheWrapper`; `ForwardBatch` → SGLang's `ForwardBatch`; backend objects → `AttentionBackend` subclasses

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `AttnBackend` enum | `model/config.py` — `SDPA` / `FLASHINFER`; read from `config.yml` |
| `ForwardMode` + `ForwardBatch` | `forward_batch.py` — carries `(mode, kv_cache, attention_mask)` |
| `make_backend()` factory | `model/backend.py` — returns `SDPABackend` or `FlashInferBackend` |
| `SDPABackend.forward()` | `model/backend.py` — wraps `_sdpa_forward`; used for all SDPA paths |
| `FlashInferBackend.forward()` | `model/backend.py` — PREFILL → `_sdpa_forward`; DECODE → FlashInfer ragged |
| `Qwen3Attention.__init__` | `model/attention.py` — `self.backend = make_backend(config)` |
| `Qwen3Attention.forward()` | `model/attention.py` — calls `self.backend.forward(q, k, v, layer_idx, forward_batch)` |
| Workspace pre-allocation | `model_runner.py` — `self._workspace = torch.empty(_WORKSPACE_MB * 1024**2, dtype=torch.uint8, device=DEVICE)` |
| `PackedKVCache` class | `kv_cache.py` — `class PackedKVCache:` |
| `qo_indptr` construction | `kv_cache.py` — `self.qo_indptr = torch.arange(B + 1, dtype=torch.int32, device=DEVICE)` |
| `kv_indptr` construction | `kv_cache.py` — `self.kv_indptr = torch.tensor(kv_cumsum, dtype=torch.int32, device=DEVICE)` |
| FlashInfer wrapper init | `kv_cache.py` — `flashinfer.BatchPrefillWithRaggedKVCacheWrapper(workspace, kv_layout="NHD")` |
| `plan()` / `begin_forward` | `kv_cache.py` — `self._wrapper.begin_forward(self.qo_indptr, self.kv_indptr, ...)` |
| `update()` — ragged pack | `kv_cache.py` — `def update(self, layer_idx, new_k, new_v)` |
| `write_back()` | `kv_cache.py` — `def write_back(self)` |
| `end_forward()` | `kv_cache.py` — `self._wrapper.end_forward()` |
| `prefill` ForwardBatch | `model_runner.py` — `ForwardBatch(mode=ForwardMode.PREFILL, kv_cache=kv, attention_mask=mask)` |
| `decode_step` ForwardBatch | `model_runner.py` — `ForwardBatch(mode=ForwardMode.DECODE, kv_cache=pack_kv, attention_mask=None)` |
| `config.yml` backend field | `config.yml` — `attn_backend: flashinfer`; read by `server.py` and forwarded to `ModelRunner` |
