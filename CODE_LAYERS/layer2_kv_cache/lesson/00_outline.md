# Layer 2 — Lesson Outline

## What This Lesson Covers

Layer 1 gave us full visibility into the decode loop. Layer 2 makes it faster — with a surgical change to `model.py` alone. The key insight is that on every decode step in Layer 1, the model recomputes key and value tensors for every token in the growing sequence, including all the tokens it already processed on previous steps. Layer 2 saves those tensors after each forward pass and hands them back on the next step, so the model only processes the single new token. `server.py` and `benchmark.py` are untouched. The benchmark numbers change dramatically; the server code does not change at all.

The sections follow the code: we start with what changed in `model.py`, then explain why that change is correct (the attention mechanism), then examine the supporting classes and the performance impact.

---

## Sections

### 01 — The Decode Loop Line by Line (`01_the_decode_loop.md`)
- Full walkthrough of `KVCacheModel.generate` in `model.py`
- Step 1: tokenize the prompt (same as Layer 1)
- Step 2: prefill — `model(input_ids=full_prompt, past_key_values=KVCache(), use_cache=True)` — builds the cache and produces the first token
- TTFT measured immediately after prefill: the first token is already available before the decode loop starts
- Step 3: decode loop — `input_ids=current_token` (shape `[1, 1]`), `past_key_values=past_kv` — the cache object grows by one position each step
- `past_kv = out.past_key_values` after each step: updating the reference to the now-extended cache
- EOS check and accumulation: same as Layer 1
- TPOT: average of `step_times` from the decode loop only (prefill is excluded, measured separately as TTFT)

### 02 — Keys, Values, and Queries in Attention (`02_keys_values_queries.md`)
- Why this matters: the cache is named after exactly two of the three attention projections
- The three projections: Q, K, V computed from each token's representation via learned weight matrices
- What each does: query as "what a token is asking for", key as "what it advertises", value as "what it contributes"
- The attention computation: `scores = Q @ Kᵀ / √d`, `weights = softmax(scores)`, `output = weights @ V`
- Causality: the causal mask means each token only attends to earlier positions — no future token can affect a past token's K or V
- Why K and V are cacheable: they depend only on what came before them and are fixed once computed
- Why Q is not cached: the query's job is already done once the output at that position is produced; only the new token's query matters each step

### 03 — What the KV Cache Is (`03_the_cache.md`)
- The redundant work in Layer 1: every decode step recomputes K and V for all prior tokens
- Why the redundancy is safe to eliminate: causality guarantees past K and V are immutable
- The cache as a saved scratchpad: store K and V after each step, pass them back next time
- What changes in the code: two arguments (`past_key_values`, `use_cache=True`) and one assignment (`past_kv = out.past_key_values`)
- Performance: decode cost drops from `O((L+k)²)` to `O(L+k)` per step

### 04 — `past_key_values` in HuggingFace (`04_past_key_values.md`)
- What `past_key_values` is: a cache object the model reads from and writes to during each forward pass
- HuggingFace's interface contract: each attention layer calls `past_key_values.update(new_k, new_v, layer_idx)` and gets back the full accumulated K and V
- How `out.past_key_values` is passed back to the next call: the loop holds a reference to the cache object
- Why we pass our own `KVCache` instead of HuggingFace's `DynamicCache`: visibility, and to prepare for later layers that change the storage strategy
- The `layer_idx` argument: why each of the 28 layers must store its K and V separately

### 05 — The `KVCache` Class (`05_kv_cache_class.md`)
- The two-level structure: `KVCache` holds one `LayerCache` per transformer layer
- `LayerCache`: stores the accumulated `keys` and `values` tensors for one layer, shape `[batch, n_kv_heads, seq_len, head_dim]`
- `LayerCache.update(new_k, new_v)`: appends via `torch.cat` and returns the full accumulated tensors — what HuggingFace's attention expects
- `KVCache.update(key_states, value_states, layer_idx)`: delegates to the right `LayerCache`, growing the list lazily
- Memory accounting: `KVCache.memory_bytes()` — how much GPU memory the cache occupies and how it grows with sequence length
- What Layer 3 will change here: replace `torch.cat` with a pre-allocated buffer

### 06 — Prefill and Decode in Layer 2 (`06_prefill_and_decode.md`)
- Prefill is unchanged in cost: the full prompt is processed in one forward pass, `O(L²)` attention work
- Prefill populates the cache as a side-effect: each attention layer calls `update` for every prompt token
- Decode is now genuinely different in code: `input_ids` is always `[1, 1]` — one new token — instead of the growing sequence
- The attention operation in decode: the new token's query attends over all cached K/V — `O(L+k)` work instead of `O((L+k)²)`
- Memory trade-off: TPOT drops dramatically; GPU memory usage grows with sequence length

### 07 — The Full Loop (`07_the_full_loop.md`)
- End-to-end trace of a single `generate()` call, connecting all prior sections in order
- Step 1 — Tokenize: apply chat template, encode to `[1, prompt_len]`, move to GPU
- Step 2 — Prefill: empty `KVCache()`, model call populates all 28 `LayerCache` entries, first token sampled from `logits[0, -1, :]`, TTFT recorded
- Step 3 — Decode loop: `[1, 1]` tensor each step, `LayerCache.update` appends via `torch.cat`, new query attends over full accumulated K/V at `O(L+k)` cost, EOS check, `step_times` accumulated
- Step 4 — Decode and return: TPOT from `step_times` average, `tokenizer.decode`, result dict returned to `server.py`

### 08 — What the Numbers Show (`08_benchmark_results.md`)
- Running the benchmark: same 20 ShareGPT requests, same seed, same hardware as Layer 1
- TTFT: unchanged — prefill is the same `O(L²)` computation
- TPOT: near-constant regardless of prompt length — each decode step processes one query token
- Reading per-request output: TTFT varies with prompt length, TPOT stays flat
- Memory cost per request from the server log
- Comparison table: Layer 1 vs Layer 2 TTFT and TPOT

### 09 — What Comes Next (`09_whats_next.md`)
- The remaining cost: `torch.cat` in `LayerCache.update` allocates a new tensor every decode step
- Layer 3: pre-allocated fixed buffer, written in-place — no allocation, no copy
- `server.py` and `benchmark.py` stay unchanged again — the improvement is isolated to `kv_cache.py`
- The larger challenge ahead: sharing a cache across concurrent requests and paged attention

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps layer2 concepts (KV cache, `past_key_values`, prefill/decode split) to their implementations in the SGLang source tree

---

## Key Code Anchors

| Concept | Location |
|---|---|
| Prefill forward pass | `model.py` line 109: `out = self.model(input_ids=input_ids, past_key_values=past_kv, use_cache=True)` |
| Cache populated after prefill | `model.py` line 111: `past_kv = out.past_key_values` |
| TTFT measurement | `model.py` line 115: `ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)` |
| Decode forward call | `model.py` line 141: `out = self.model(input_ids=current_token, past_key_values=past_kv, use_cache=True)` |
| Cache update after decode step | `model.py` line 147: `past_kv = out.past_key_values` |
| Sampling | `sampling.py`: `sample_next_token` |
| LayerCache accumulation | `kv_cache.py` line 61: `self.keys = torch.cat([self.keys, new_keys], dim=-2)` |
| KVCache HF interface | `kv_cache.py` line 110: `KVCache.update(key_states, value_states, layer_idx)` |
| Cache memory | `kv_cache.py` line 145: `KVCache.memory_bytes()` |
