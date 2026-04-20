# Layer 14 — Lesson Outline

## What This Lesson Covers

Layer 13 reduced weight memory with GPTQ quantization, leaving room for a second model alongside the target. But each decode step still commits exactly one token per request regardless of weight size — the fundamental bottleneck is the number of target model forward passes per output token, not the forward pass cost per se. Speculative decoding breaks this ratio: a small, cheap draft model generates N candidate tokens autoregressively; the large target model verifies all N+1 positions (last confirmed token plus N candidates) in a single batched extend pass. For greedy decoding, a draft token is accepted when the target's argmax agrees; the accepted prefix length `k ∈ [0, N]` plus one mandatory bonus token is committed per target call. Expected throughput gain is approximately `(1 + acceptance_rate × N)` target-model tokens per step, compared to 1 without speculation.

Layer 14 adds a single new file, `spec_runner.py`, containing the `SpecRunner` class. `SpecRunner` owns two `ModelRunner` instances — target and draft — each with an independent `KVPool` and `ReqToTokenPool`. For each request, a mirror `Req` object (`d_req`) tracks the draft model's KV state. At each speculation step: the draft model runs `decode_step` N times to produce N candidate tokens, the target model runs one `_verify_extend` (using the same `prefill_batch` machinery), `_accept_reject` computes the accepted prefix length, `_rewind_target_kv` and `_rewind_draft_kv` free the pool pages for rejected positions, and `kv_committed_len` is advanced by `accept_len + 1`.

`server.py` routes requests to `SpecRunner.prefill` and `SpecRunner.spec_decode_step` instead of `ModelRunner`. All layer-level files (`kv_cache.py`, `radix_cache.py`, `scheduler.py`, `forward_batch.py`, `model/`, `model_gptq/`) are unchanged.

The sections follow the speculation loop: the gap between decode throughput and forward pass cost, the two-model VRAM split, draft KV mirroring, the draft phase, the verify extend, the greedy accept/reject rule, KV rewind, statistics and acceptance rate, the full call trace, and the path to sampling-based verification for non-greedy decoding.

---

## Sections

### 01 — From One Token per Step to N+1 (`01_from_one_to_n_plus_one.md`)
- Layer 13's ceiling: one target model token committed per forward pass; `throughput = 1 / latency_of_target_forward`; the cost of the forward pass can be amortized across many requests with large batches but each request still waits for its own step
- Speculative decoding's offer: for greedy decoding, draft tokens are correct ~70% of the time; if N=5 and acceptance rate = 0.7, the expected committed tokens per target call is `1 + 5 × 0.7 = 4.5` vs 1; latency per generated token drops by ~4.5×
- The additional cost: N draft decode steps per target step; total cost is `N × draft_cost + target_verify_cost` per speculation cycle; worth it when `N × draft_cost ≪ target_verify_cost`, which holds when the draft model is ~3-5× smaller
- What determines acceptance rate: alignment between draft and target probability distributions at temperature 0 (greedy) — tokens where the two models agree; longer common subsequences produce higher acceptance rates for structured text (code, formulaic answers)
- What Layer 14 adds: `spec_runner.py` with `SpecRunner`; `server.py` wires to `SpecRunner`; all other files unchanged

### 02 — Two-Model VRAM Split (`02_two_model_vram_split.md`)
- VRAM budget: target model (Qwen3-1.7B GPTQ) ≈ 0.85 GB weights; draft model (Qwen3-0.6B fp16) ≈ 1.2 GB weights; on a 16 GB GPU, the remaining ~14 GB is split between two KV pools and the PyTorch CUDA context
- `SpecRunner.__init__` loads target first, then draft; `kv_memory_fraction` is split explicitly: `target_kv_fraction=0.35` and `draft_kv_fraction=0.45` of the free memory measured at each model's load time
- Why target loads first: after loading the target weights, `torch.cuda.mem_get_info()` reflects the consumed VRAM; the target KV pool is sized from that measurement; after the target pool is allocated, the draft model and its pool size from the remaining free memory
- `page_size` must match between target and draft: `compute_write_info` and `kv_indptr` construction both use `self.page_size`; mismatched page sizes would corrupt KV index arithmetic
- Independent `KVPool` and `ReqToTokenPool` per model: the draft's pool pages hold draft K/V; the target's pool pages hold target K/V; there is no cross-model KV sharing

### 03 — Draft KV Mirroring (`03_draft_kv_mirroring.md`)
- `d_req = Req(rid="draft_" + req.rid, input_ids=list(req.input_ids), ...)` created at prefill time for each target request; stored in `self._draft_reqs[id(req)]`; carries its own `slot_indices`, `req_pool_idx`, `kv_committed_len`, `output_ids`
- Why mirror instead of sharing: draft and target have different architectures (embedding dimension, FFN size, number of layers) and therefore different K/V shapes; separate pools and separate `ReqToTokenPool` rows are mandatory
- `d_req.output_ids = list(req.output_ids)` at the start of each `spec_decode_step`: syncs the confirmed token sequence to the draft req so `draft.decode_step` uses the correct last token as input; draft's own first-token prediction from prefill is discarded — the target's prediction is canonical
- `id(req)` as the dict key: avoids circular references between `Req` objects; works as long as target `Req` objects are not garbage-collected while a spec step is in flight
- `_draft_reqs` cleanup: `del self._draft_reqs[id(req)]` in `_cleanup_req`; both `target.kv_pool.free` and `draft.kv_pool.free` are called; `slot_indices` and `req_pool_idx` are cleared on both req objects

### 04 — The Draft Phase (`04_draft_phase.md`)
- `_draft_phase(d_req)`: runs `self.draft.decode_step([d_req])` exactly N times; each call is a standard paged decode step over the draft model — `begin_forward`, one token forward, `end_forward`, `req.slot_indices.append(new_slot)`, sampling
- Tracking new pages per step: `len_before = len(d_req.slot_indices)` is recorded before each decode call; `new_pages = d_req.slot_indices[len_before:]` captures any pages allocated at that step; `new_pages_per_step` accumulates these lists for use by `_rewind_draft_kv`
- Return value: `(draft_tokens [d1,...,dN], new_pages_per_step [[pages_step0], ..., [pages_stepN-1]])`; `draft_tokens` is the list of the draft model's greedy argmax outputs after each step
- Draft model's KV state after phase: the draft pool holds K/V for positions `[0 .. T + N - 1]` (prompt + first token + N draft tokens); if k tokens are accepted, positions `T+k .. T+N-1` are wasted and must be freed
- No batching across requests in the draft phase: each request runs its N draft steps independently; the draft model's small size means sequential calls are still fast relative to the target verify

### 05 — The Verify Extend (`05_verify_extend.md`)
- `_verify_extend(req, verify_tokens)` where `verify_tokens = [last_confirmed, d1, ..., dN]` (N+1 tokens); positions `kv_committed_len .. kv_committed_len + N`
- Uses exactly the same `compute_write_info` → `create_flashinfer_kv_indices_triton` → `begin_forward` → `ExtendKVCtx` → model forward pipeline as `prefill_batch` in Layer 11; no new FlashInfer API required
- Position IDs: `range(req.kv_committed_len, req.kv_committed_len + N + 1)` — each verify token is at its absolute sequence position; the last confirmed token is already in the KV pool from the previous step, but its position must be re-presented to produce the logit used to verify `d1`
- Output: `logits [1, N+1, vocab]` from the model; squeezed to `[N+1, vocab]`; `logits[i]` is the target model's distribution at position `kv_committed_len + i`; used by `_accept_reject` to compare with `draft_tokens[i]`
- The N+1 KV entries are written into the target pool by `ctx.store()` during the forward pass; positions for rejected tokens are freed immediately after by `_rewind_target_kv`

### 06 — Accept/Reject and KV Rewind (`06_accept_reject_rewind.md`)
- `_accept_reject(draft_tokens, verify_logits)`: `target_tokens = verify_logits.argmax(dim=-1).tolist()` — greedy predictions at all N+1 positions; walk `i ∈ [0, N-1]`: `accept_len += 1` while `target_tokens[i] == draft_tokens[i]`; stop at first mismatch
- Bonus token: `target_tokens[accept_len]` — the target's prediction at the first rejected position (or position N if all draft tokens were accepted); always emitted regardless of `accept_len`; `accept_len + 1` tokens committed per step
- `_rewind_target_kv(req, accept_len)`: `kept_kv_len = req.kv_committed_len + accept_len + 1`; `pages_needed = ceil(kept_kv_len / page_size)`; if `len(req.slot_indices) > pages_needed`: free tail pages and truncate `slot_indices`
- `_rewind_draft_kv(d_req, new_pages_per_step, accept_len)`: free `new_pages_per_step[accept_len:]` from the draft pool; these are the pages allocated for the N - accept_len rejected draft steps; `d_req.slot_indices` is updated to remove them
- After rewind: `req.kv_committed_len += accept_len + 1`; target pool reflects exactly the kept token positions; draft pool reflects the same positions (after `d_req.output_ids` sync at the next step's start)

### 07 — Statistics (`07_statistics.md`)
- `acceptance_rate = total_accepted_tokens / total_draft_tokens`; across all speculation steps, this is the fraction of draft tokens the target agreed with; for Qwen3 0.6B vs 1.7B at temperature 0, typical values are 0.65–0.75 on common text domains
- `tokens_per_step = (total_accepted_tokens + total_spec_steps) / total_spec_steps`; numerator counts all committed tokens (accepted drafts + bonus tokens); denominator counts target model calls; this is the speedup factor over autoregressive decoding
- `self.total_draft_tokens += self.N` per step (not per accepted token): all N draft tokens were generated regardless of how many were accepted; the draft cost is incurred in full for every speculation step
- `stats_str()` for server logs: `acceptance_rate`, `tokens_per_step`, `total_spec_steps`, `total_accepted`, `total_draft` — sufficient to evaluate whether the draft/target pairing is productive and whether N should be tuned

### 08 — The Full Loop (`08_the_full_loop.md`)
- End-to-end trace for one request with N=5, T=512 prompt tokens, acceptance rate ≈ 0.7
- Step 1 — Prefill: target `prefill_batch([req])` processes all 512 tokens, writes K/V to target pool, samples first token; `req.kv_committed_len = 512`; draft `prefill_batch([d_req])` processes the same 512 tokens, writes K/V to draft pool; `d_req.output_ids = [req.output_ids[-1]]`
- Step 2 — Draft phase: `_draft_phase` runs `draft.decode_step` 5 times; draft pool grows by 5 slots; `draft_tokens = [d1, d2, d3, d4, d5]`; `new_pages_per_step` records which pages were new at each step
- Step 3 — Verify extend: target processes `[t512, d1, d2, d3, d4, d5]` in one extend pass; target pool grows by 6 positions (1 for `t512` re-presented, 5 for the drafts); `logits [6, vocab]` returned
- Step 4 — Accept/reject: suppose `d1, d2, d3` match (`accept_len=3`), `d4` does not; `bonus = target_tokens[3]`; tokens `d1, d2, d3, bonus` committed to `req.output_ids`; `_rewind_target_kv` frees the 2 tail positions (`d4`, `d5`); `_rewind_draft_kv` frees `new_pages_per_step[3:]`; `req.kv_committed_len = 516`
- Step 5 — Repeat: 4 tokens were committed in one target forward; equivalent to 4 autoregressive decode steps; next iteration syncs `d_req.output_ids` and repeats

### 09 — What Comes Next (`09_whats_next.md`)
- The greedy constraint: `_accept_reject` uses `argmax` — it only works correctly for temperature=0 decoding; for temperature > 0, the correct rule (from Leviathan et al. 2023) uses rejection sampling from the ratio of target and draft distributions, requiring the full probability vectors, not just the argmax
- Batched speculation: Layer 14 processes each request's draft phase sequentially in a loop; batching multiple requests' draft phases into a single `draft.decode_step` call requires aligning their speculation positions, which complicates `_draft_reqs` bookkeeping and KV rewind
- Staged inference and pipeline parallelism: for very large target models (70B+), the target verify can be pipelined with the draft generation so that the GPU is never idle — this requires asynchronous orchestration beyond the synchronous `spec_decode_step` loop
- The established pattern: each layer adds one mechanism (chunked prefill, prefix caching, quantization, speculative decoding), one new file (scheduler additions, `radix_cache.py`, `model_gptq/`, `spec_runner.py`), and one benchmark metric; the benchmark measures exactly the throughput gain that the mechanism was designed to produce

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 14 concepts to SGLang source: `SpecRunner` → `SpecInfoBatch` / `EagleWorker` in `srt/speculative/`; `_accept_reject` → `verify_input` in `srt/speculative/eagle_utils.py`; `_rewind_target_kv` → `kv_cache rollback` in the spec worker; `acceptance_rate` → `spec_verify_stats`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `SpecRunner` class | `spec_runner.py` line 118: `class SpecRunner:` |
| Two-model init (target first) | `spec_runner.py` line 156: `self.target = ModelRunner(target_path, kv_memory_fraction=target_kv_fraction)` |
| Draft model init | `spec_runner.py` line 165: `self.draft = ModelRunner(draft_path, kv_memory_fraction=draft_kv_fraction)` |
| `_draft_reqs` dict | `spec_runner.py` line 176: `self._draft_reqs: Dict[int, Req] = {}` |
| `prefill` — target batch | `spec_runner.py` line 205: `self.target.prefill_batch(reqs)` |
| `prefill` — draft batch | `spec_runner.py` line 224: `self.draft.prefill_batch(d_reqs)` |
| `d_req.output_ids` sync | `spec_runner.py` line 262: `d_req.output_ids = list(req.output_ids)` |
| `spec_decode_step` entry | `spec_runner.py` line 244: `def spec_decode_step(self, reqs: List[Req]) -> List[Req]:` |
| Draft phase call | `spec_runner.py` line 265: `draft_tokens, new_pages_per_step = self._draft_phase(d_req)` |
| Verify extend call | `spec_runner.py` line 273: `verify_logits = self._verify_extend(req, verify_tokens)` |
| Accept/reject call | `spec_runner.py` line 277: `accept_len, bonus_token = self._accept_reject(...)` |
| KV committed update | `spec_runner.py` line 305: `req.kv_committed_len += accept_len + 1` |
| `_draft_phase` — new pages tracking | `spec_runner.py` line 374: `len_before = len(d_req.slot_indices)` |
| `_verify_extend` — compute_write_info | `spec_runner.py` line 408: `write_info = compute_write_info(...)` |
| `_accept_reject` greedy rule | `spec_runner.py` line 513: `target_tokens = verify_logits.argmax(dim=-1).tolist()` |
| Bonus token selection | `spec_runner.py` line 522: `bonus_token = target_tokens[accept_len]` |
| `_rewind_target_kv` | `spec_runner.py` line 529: `def _rewind_target_kv(self, req, accept_len)` |
| `_rewind_draft_kv` | `spec_runner.py` line 546: `def _rewind_draft_kv(self, d_req, new_pages_per_step, accept_len)` |
| `acceptance_rate` property | `spec_runner.py` line 333: `return self.total_accepted_tokens / self.total_draft_tokens` |
| `tokens_per_step` property | `spec_runner.py` line 342: `return (self.total_accepted_tokens + self.total_spec_steps) / self.total_spec_steps` |
