# Layer 14 — Summary

Layer 14 adds speculative decoding via a new `spec_runner.py` file containing `SpecRunner`, which orchestrates a small draft model and a large target model to commit more than one token per target forward pass. The scheduler, `KVPool`, `RadixCache`, `forward_batch.py`, and all model files are unchanged; `server.py` swaps `ModelRunner` for `SpecRunner` as its inference engine.

---

## From Layer 13 to Layer 14

In Layer 13, each decode step committed exactly one token per request regardless of how many requests were batched together:

```python
# Layer 13 — Scheduler.run decode path (simplified)
finished = self.model_runner.decode_step(self._running)
# One new token per request, one target forward pass per step.
```

In Layer 14, a single target forward pass can commit up to N+1 tokens:

```python
# Layer 14 — server.py / SpecRunner integration
finished = spec_runner.spec_decode_step(running_reqs)
# One target verify extend per step; accepts 0..N draft tokens + 1 bonus token.
```

The extra tokens come from N draft model decode steps run before the target verify. For `N=5` and a 70% acceptance rate, the expected committed tokens per target call is `1 + 5 × 0.7 = 4.5`. The total cost per step is `5 × draft_cost + target_verify_cost`. Because the draft model (Qwen3-0.6B) is roughly 3× smaller than the target (Qwen3-1.7B), `5 × draft_cost ≪ target_verify_cost`, and the net throughput per unit time increases substantially.

---

## Two-Model VRAM Split

`SpecRunner.__init__` loads the target model first, measures free VRAM, allocates its KV pool, then loads the draft model from the remaining memory:

```python
self.target = ModelRunner(
    target_path,
    page_size              = page_size,
    enable_prefix_caching  = enable_prefix_caching,
    kv_memory_fraction     = target_kv_fraction,   # 0.35
)

self.draft = ModelRunner(
    draft_path,
    page_size              = page_size,
    enable_prefix_caching  = enable_prefix_caching,
    kv_memory_fraction     = draft_kv_fraction,    # 0.45
)
```

`kv_memory_fraction` is measured at each model's load time from `torch.cuda.mem_get_info()`, so the fractions are relative to the free memory at that point — not to total GPU memory. The target loads first so its larger weight footprint is already accounted for when the draft's KV pool is sized. Each `ModelRunner` has its own `KVPool` and `ReqToTokenPool`; there is no cross-model KV sharing. `page_size` must match between the two models because the same `compute_write_info` and `kv_indptr` arithmetic is applied to both.

---

## Draft KV Mirroring

For each target `Req`, `SpecRunner` creates a mirror `Req` for the draft model at prefill time:

```python
d_req = Req(
    rid            = f"draft_{req.rid}",
    input_ids      = list(req.input_ids),
    max_new_tokens = 99_999,   # never auto-finish
    temperature    = 0.0,
    future         = req.future,
)
d_req.fill_ids         = list(req.input_ids)
d_req.extend_input_len = len(req.input_ids)
d_req.kv_committed_len = 0
```

`d_req` is stored in `self._draft_reqs[id(req)]` and carries its own `slot_indices`, `req_pool_idx`, and `kv_committed_len` — all pointing into the draft pool, not the target pool. After draft `prefill_batch` runs, the draft's own first-token prediction is discarded and replaced with the target's:

```python
d_req.output_ids = list(req.output_ids)   # [t0] from target
```

The confirmed token sequence is always determined by the target. The draft's role is purely to propose candidates ahead of the target's verification.

---

## The Draft Phase

At the start of each `spec_decode_step`, the confirmed state is synced into `d_req`:

```python
d_req.output_ids = list(req.output_ids)
```

Then `_draft_phase` runs N standard decode steps on the draft model:

```python
def _draft_phase(self, d_req: Req) -> Tuple[List[int], List[List[int]]]:
    draft_tokens       : List[int]       = []
    new_pages_per_step : List[List[int]] = []

    for _ in range(self.N):
        len_before = len(d_req.slot_indices)
        self.draft.decode_step([d_req])
        new_pages = list(d_req.slot_indices[len_before:])
        new_pages_per_step.append(new_pages)
        draft_tokens.append(d_req.output_ids[-1])

    return draft_tokens, new_pages_per_step
```

`new_pages_per_step` records which pool pages were allocated at each draft decode step. This list is used later by `_rewind_draft_kv` to free pages belonging to rejected steps. After N calls to `draft.decode_step`, the draft pool holds K/V for positions `0 .. T + N - 1` (prompt plus N new tokens), and `draft_tokens` contains `[d1, d2, ..., dN]` — the draft model's greedy predictions.

---

## The Verify Extend

`_verify_extend` runs a single extend forward of the target model over the N+1 tokens `[last_confirmed, d1, ..., dN]`:

```python
last_confirmed = req.output_ids[-1]
verify_tokens  = [last_confirmed] + draft_tokens   # N+1 tokens

verify_logits = self._verify_extend(req, verify_tokens)
```

Inside `_verify_extend`, the same machinery as `prefill_batch` is used — `compute_write_info`, the Triton `kv_indices` kernel, `begin_forward`, `ExtendKVCtx`, and `end_forward`. Position IDs cover `kv_committed_len .. kv_committed_len + N`. The N+1 tokens' K/V is written into the target pool:

```python
write_info = compute_write_info(
    kv_pool=self.target.kv_pool, rtp=self.target.req_to_token_pool,
    slot_indices=req.slot_indices, req_pool_idx=req.req_pool_idx,
    kv_committed_len=req.kv_committed_len, n_fill=n,   # n = N+1
)
# ...
logits_3d = self.target.model(ids_t, attention_mask=None,
                               kv_cache=ctx, position_ids=pos_t)
# [1, N+1, vocab]
return logits_3d[0]   # [N+1, vocab]
```

`logits[i]` is the target's distribution at position `kv_committed_len + i`. `logits[0]` is conditioned on the entire prompt and output so far and is used to verify `d1` — the first draft token — by checking whether the target's argmax at position 0 matches `d1`.

---

## Accept/Reject and KV Rewind

The greedy accept/reject rule walks the target's predictions from left to right:

```python
target_tokens = verify_logits.argmax(dim=-1).tolist()   # [N+1]

accept_len = 0
for i in range(N):
    if target_tokens[i] == draft_tokens[i]:
        accept_len += 1
    else:
        break

bonus_token = target_tokens[accept_len]   # target's correction / next
```

`accept_len ∈ [0, N]`. `bonus_token` is always emitted — it is the target's prediction at the first position where the draft was wrong (or at position N if all draft tokens were accepted). The committed tokens per step are `draft_tokens[:accept_len] + [bonus_token]`, always at least 1 and at most N+1.

After the accept/reject decision, rejected K/V positions are freed:

```python
# Target: keep only accept_len+1 of the N+1 written positions.
kept_kv_len  = req.kv_committed_len + accept_len + 1
pages_needed = math.ceil(kept_kv_len / self.page_size)
if pages_needed < len(req.slot_indices):
    self.target.kv_pool.free(req.slot_indices[pages_needed:])
    req.slot_indices = req.slot_indices[:pages_needed]

# Draft: free pages from the N - accept_len rejected steps.
for step_pages in new_pages_per_step[accept_len:]:
    d_req.slot_indices.remove(page)
    self.draft.kv_pool.free(step_pages)

req.kv_committed_len += accept_len + 1
```

After rewind, both pools reflect exactly the tokens that were committed. The draft pool is ready to extend from `kv_committed_len + accept_len + 1` on the next step (after `d_req.output_ids` is synced to `req.output_ids` at the top of the next `spec_decode_step` iteration).

---

## Statistics

`SpecRunner` tracks three counters across all steps:

```python
self.total_draft_tokens:    int = 0   # incremented by N every step
self.total_accepted_tokens: int = 0   # incremented by accept_len every step
self.total_spec_steps:      int = 0   # incremented by 1 every step
```

`acceptance_rate = total_accepted_tokens / total_draft_tokens` is the fraction of draft tokens the target agreed with. `tokens_per_step = (total_accepted_tokens + total_spec_steps) / total_spec_steps` counts all committed tokens (accepted drafts plus bonus tokens) divided by target model calls — this is the direct throughput multiplier relative to a non-speculative decode loop. Both metrics are available via `stats_str()` for server logging.

---

## The Full Loop

The server creates a `SpecRunner` instead of a `ModelRunner` and routes incoming requests through it. Consider a single request with a 512-token prompt and `N=5`.

At prefill, `spec_runner.prefill([req])` runs `target.prefill_batch([req])` — the standard 512-token extend forward, writing K/V into the target pool and sampling the first output token `t0`. Then `draft.prefill_batch([d_req])` runs the identical extend for the draft model, writing to the draft pool. `d_req.output_ids` is overwritten with `[t0]` from the target.

On the first `spec_decode_step`, `d_req.output_ids` is already `[t0]`. `_draft_phase` calls `draft.decode_step([d_req])` five times, each producing one draft token and potentially allocating new pool pages. `draft_tokens = [d1, d2, d3, d4, d5]`. `_verify_extend` then runs the target over `[t0, d1, d2, d3, d4, d5]` (positions 512–517) in one extend call, writing 6 positions of K/V into the target pool. `_accept_reject` computes `accept_len=3` (suppose `d1, d2, d3` match) and `bonus_token = target_tokens[3]`. Tokens `d1, d2, d3, bonus` are appended to `req.output_ids`. `_rewind_target_kv` frees the 2 tail pages for positions 516–517 from the target pool. `_rewind_draft_kv` frees the pages from draft decode steps 3 and 4. `req.kv_committed_len = 516`.

On subsequent steps the pattern repeats: 5 draft steps, 1 target extend, accept/reject, rewind, advance `kv_committed_len`. When `bonus_token == eos_id` or `len(req.output_ids) >= max_new_tokens`, `_cleanup_req` frees all target and draft pool pages and removes the `d_req` from `_draft_reqs`.

---

## What Comes Next

Layer 14 implements greedy speculative decoding — `_accept_reject` uses `argmax` and requires temperature=0 for correctness. For temperature > 0, the proper rejection-sampling rule from Leviathan et al. (2023) requires comparing the full token probability distributions of the draft and target at each position and resampling from the corrected distribution at the rejection site. This changes `_accept_reject` from a trivial argmax comparison to a distribution-ratio sampling operation but leaves all KV management, pool allocation, and the dual-`ModelRunner` architecture unchanged. The other open direction is batched draft phases: processing multiple requests' N draft steps in a single `draft.decode_step` call rather than sequentially, which requires aligning all requests' speculation positions and handling variable `accept_len` across requests in the same rewind step.
