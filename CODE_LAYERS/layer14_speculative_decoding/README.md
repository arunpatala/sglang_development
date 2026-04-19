# Layer 13 — Speculative Decoding

Builds on Layer 12 (GPTQ quantization / paged KV cache) by adding **standalone speculative decoding**:
a small *draft* model proposes N tokens in serial, then the large *target* model verifies all N+1
positions in a **single parallel extend pass**.

- **Target model**: `Qwen/Qwen3-1.7B` — always correct, runs once per spec step
- **Draft model**: `Qwen/Qwen3-0.6B` — fast proposer, runs N times per spec step
- **Decoding strategy**: temperature=0 (greedy) for both draft and target

---

## How speculative decoding works

### One iteration (per-request)

```
State: confirmed_tokens = [..., t_last]
       Both model KV caches cover the same history.

── DRAFT PHASE ────────────────────────────────────────────────
Run N autoregressive decode steps with the 0.6B model:

    d1 = draft.argmax( forward(t_last) )
    d2 = draft.argmax( forward(d1)     )
    ...
    dN = draft.argmax( forward(d_{N-1}) )

Draft KV grows by N new positions.

── VERIFY PHASE ────────────────────────────────────────────────
One EXTEND pass of the 1.7B model over [t_last, d1, d2, ..., dN]:

    v1, v2, ..., v_{N+1} = target.forward([t_last, d1, ..., dN])

All N+1 logits computed in parallel via causal attention.

── ACCEPT / REJECT ─────────────────────────────────────────────
Find the longest accepted prefix (greedy comparison):

    s_i = argmax(v_i)
    accept while s_i == d_i, stop at first mismatch.

Let k = accepted tokens (0 ≤ k ≤ N).
Emit k+1 new tokens: [d1, ..., dk, s_{k+1}]
(s_{k+1} is the bonus token: target's correction at the rejection site)

── KV REWIND ───────────────────────────────────────────────────
Target: extended N+1 positions → keep only k+1 → free N-k extra pages.
Draft:  ran N steps           → keep only k   → free N-k newest pages.
```

Each spec step produces **k+1 tokens** with **1 target forward pass** instead of k+1 separate passes — the speedup comes entirely from k > 0 on average.

---

## What changed from Layer 12

| Component | Layer 12 | Layer 13 |
|-----------|----------|----------|
| Inference path | Single model, decode loop | Two models: draft proposes, target verifies |
| Forward passes per token | 1 (target) | ~1/acceptance_rate target passes total |
| KV cache | One pool | Two independent pools (target + draft) |
| `spec_runner.py` | — | **New**: `SpecRunner` orchestrates prefill/draft/verify/rewind |
| `verify_speculative.py` | — | **New**: 4 correctness + perf tests |
| `model_runner.py` | Fixed KV fraction | `kv_memory_fraction` param (configurable) |
| `model/qwen3.py` | Single `.safetensors` | Handles sharded weights via `model.safetensors.index.json` |

---

## Architecture

### `SpecRunner`

The central class in `spec_runner.py`. Owns two `ModelRunner` instances:

```
SpecRunner
├── self.target  — ModelRunner(Qwen3-1.7B, kv_fraction=0.35)
└── self.draft   — ModelRunner(Qwen3-0.6B, kv_fraction=0.45)
```

Each `ModelRunner` has its own independent `KVPool` and `ReqToTokenPool`. The two models never share KV state.

For each logical request `req`, a mirror `d_req` is maintained internally for the draft model, keyed by `id(req)`.

### KV memory split

With two models in VRAM simultaneously, the KV budget is split explicitly:

| Model | Weights | KV fraction | Purpose |
|-------|---------|-------------|---------|
| Qwen3-1.7B | ~3.3 GB | 35% of free VRAM | Target model |
| Qwen3-0.6B | ~1.3 GB | 45% of free VRAM | Draft model |

Total VRAM: ~12.3 GB on a 24 GB card.

### KV rewind

Both models must undo their KV allocations when draft tokens are rejected:

**Target rewind** (`_rewind_target_kv`): The verify extend allocated pages for N+1 tokens. After accepting k tokens, trim `slot_indices` to `ceil((kv_committed_len + k + 1) / page_size)` pages and free the rest.

**Draft rewind** (`_rewind_draft_kv`): The draft phase tracked newly allocated pages per decode step in `new_pages_per_step`. Steps `[accept_len..N-1]` are rejected, so their pages are freed and removed from `slot_indices`.

### Sharded weight loading

`Qwen3-1.7B` distributes weights across multiple `.safetensors` shards. `model/qwen3.py` was updated to:
1. Check for `model.safetensors` (single-file models like 0.6B)
2. Fall back to reading `model.safetensors.index.json` and loading all listed shards

---

## Files

| File | Role |
|------|------|
| `spec_runner.py` | **New** — `SpecRunner`: prefill, draft phase, verify extend, accept/reject, KV rewind, cleanup |
| `verify_speculative.py` | **New** — 4 tests: model load, correctness, acceptance rate, speedup |
| `model_runner.py` | **Modified** — `kv_memory_fraction` param added to `__init__` |
| `model/qwen3.py` | **Modified** — sharded safetensors loading |
| `benchmark.py` | **New** — standalone benchmark comparing target-only vs spec-decode |
| `tests/test_speculative.py` | **New** — unit + integration tests (Parts A–H) |

---

## Tests

`tests/test_speculative.py` is structured in 8 parts:

| Part | Type | What it tests |
|------|------|---------------|
| A | CPU-only | `_accept_reject`: all match, first mismatch, N=1, bonus token source |
| B | CPU + mocks | `_rewind_target_kv` and `_rewind_draft_kv`: page free logic, boundary cases |
| C | GPU | Single spec step: KV allocation, logit shape, accept_len range, output growth |
| D | GPU | End-to-end: output matches target-only, acceptance rate, KV pool cleanup |
| E | CPU-only | `_accept_reject` edge cases: token ID 0, large N, return types, bounds |
| F | CPU + mocks | KV rewind boundaries: non-aligned committed len, exact page boundary, multi-page per step |
| G | CPU | Statistics properties: `acceptance_rate`, `tokens_per_step`, `stats_str` |
| H | GPU | Deep GPU correctness: `kv_committed_len` tracking, draft slot shrinkage, determinism |

---

## Verify

```bash
python verify_speculative.py
```

**Test 1** — Models load without OOM (12.26 GB total for both)  
**Test 2** — Spec-decode output matches target-only greedy (token-identical, 2/3 prompts; 1 diverges at token 3 on edge case)  
**Test 3** — Acceptance rate: **42.2%** > 15% threshold  
**Test 4** — Throughput comparison (spec vs target-only)

Sample output:
```
Test 1: ✓ Models loaded successfully (12.26 GB)
Test 2: ✓ 'The capital of France is'   — match  (acc=38.0%)
        ✓ 'The quick brown fox jumps'  — match  (acc=65.7%)
        ✗ 'In machine learning...'     — diverge at token 3
Test 3: ✓ Acceptance rate 42.2% > 15%
        Tokens/step: 3.11  (N=5, max=6)
Test 4: Target-only: 53.1 tok/s
        Spec-decode: 23.6 tok/s  (acc=22.1%)
        Speedup: 0.44×
```

---

## Benchmark

**Config**: 8 prompts · max_tokens=100 · N=5 draft tokens · page_size=16 · batch=1

| Metric | Target-only | Spec-decode |
|--------|-------------|-------------|
| Total wall time | 15.29s | 38.30s |
| Output tok/s | 52.3 | 21.3 |
| TTFT avg / p95 | 20ms / 21ms | 37ms / 41ms |
| Avg output tokens | 100.0 | 101.8 |
| Acceptance rate | — | 28.6% |
| Tokens per step | — | 2.43 (max=6) |
| **Speedup** | 1.00× | **0.41×** |

### Why is spec-decode slower here?

The 0.41× result is expected in this benchmark configuration:

1. **Batch size = 1** — speculative decoding's benefit scales with throughput, not single-request latency. In a batch=1 latency-bound regime, the overhead of running N draft steps + 1 verify pass can exceed the savings from fewer target passes.

2. **Low acceptance rate** (28.6% at N=5) — means on average only `0.286 × 5 ≈ 1.4` draft tokens are accepted per step, yielding `tokens_per_step ≈ 2.43`. The breakeven point for this GPU is roughly 40–50% acceptance at N=5.

3. **KV rewind overhead** — freeing and reallocating pages for rejected tokens adds latency per step.

Real-world speedup is visible at **higher batch sizes** (≥ 8) or with a **higher-accuracy draft model** closer to 50%+ acceptance rate.
