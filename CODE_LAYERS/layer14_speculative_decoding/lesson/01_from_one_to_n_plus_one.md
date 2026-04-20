# 01 — From One Token per Step to N+1

## From Layer 13 to Layer 14

Layers 9 through 13 progressively reduced overhead: the per-step GPU index copy (Layer 9), decode starvation from long prompts (Layer 11), redundant KV computation for shared prefixes (Layer 12), and weight memory (Layer 13). Each optimization improved throughput by removing wasted work. Layer 14 takes a different approach: instead of eliminating waste, it changes the fundamental ratio of output tokens to target model forward passes.

In Layer 13, every decode step commits exactly one token:

```python
# Layer 13 — Scheduler.run decode path
finished = self.model_runner.decode_step(self._running)
# one target forward pass → one output token per request
```

No matter how fast the 28-layer forward pass becomes, the number of target model evaluations required to generate `N_output_tokens` is exactly `N_output_tokens`. Halving layer latency halves total time, but does not change the token-per-evaluation count.

Layer 14 introduces speculative decoding, which changes the token-per-evaluation count:

```python
# Layer 14 — server.py / SpecRunner integration
finished = spec_runner.spec_decode_step(running_reqs)
# N draft decode steps + 1 target verify → (accept_len + 1) tokens committed
# Expected: 1 + N × acceptance_rate tokens per target evaluation
```

For `N=5` and acceptance rate 0.7: `1 + 5 × 0.7 = 4.5` tokens per target call. This is not a 4.5× speedup in wall-clock time per step (because the N draft steps add overhead), but the ratio of committed tokens to target model calls increases by 4.5×, which translates to a net throughput improvement when `N × draft_cost ≪ target_cost`.

---

## The Acceptance Rate

For greedy decoding (temperature=0), a draft token is accepted if and only if the target model's argmax at that position matches the draft token. The acceptance rate depends on how well the draft model's distribution aligns with the target's.

For Qwen3-0.6B vs Qwen3-1.7B at temperature 0 on natural language text: approximately 65–75% of draft tokens are accepted. The exact rate varies by domain — structured outputs (code, JSON), where the next token is highly constrained, yield higher acceptance rates (85%+); open-ended creative writing yields lower rates.

The acceptance rate compounds geometrically: if each draft token is accepted with probability `p`, then accepting all 5 draft tokens has probability `p^5`. The expected number of accepted tokens is `Σ_{k=0}^{N} k × (1-p) × p^k + N × p^N`, which simplifies to `p × (1 - p^N) / (1 - p)`. For `p = 0.7`, `N = 5`: approximately 2.69 accepted draft tokens plus one bonus token = 3.69 tokens per step. The `1 + N × p` formula overestimates slightly for finite N; the exact calculation gives 3.69 vs the approximation 4.5.

---

## The Total Cost

One `spec_decode_step` performs:

1. `N` draft decode steps (each a full forward pass through the draft model).
2. One target verify extend (one full forward pass through the target model over N+1 tokens).
3. KV rewind for rejected positions.

The total GPU time is approximately `N × T_draft + T_target(N+1)` where `T_draft` is the latency of one draft decode step and `T_target(N+1)` is the latency of a target extend over N+1 tokens. `T_target(N+1) ≈ T_target(1)` for small N because the extend is memory-bandwidth-bound at these sizes.

The speedup ratio over Layer 13 (which would take `N × T_target(1)` to produce the same N+1 tokens one at a time) is:

```
speedup ≈ (N+1) × acceptance_rate / ((N × T_draft / T_target) + 1)
```

For `N=5`, `acceptance_rate=0.7`, `T_draft / T_target ≈ 0.15` (draft is ~7× cheaper):

```
speedup ≈ (5+1) × 0.7 / ((5 × 0.15) + 1) = 4.2 / 1.75 ≈ 2.4×
```

A 2.4× throughput improvement with no change to the KV pool, scheduler, or model architecture.

---

## What Layer 14 Adds

A single new file `spec_runner.py` contains all of Layer 14. `SpecRunner` wraps two `ModelRunner` instances and implements `prefill`, `spec_decode_step`, and statistics tracking. `server.py` instantiates `SpecRunner` when `speculative_decoding` is configured.

Every other file — `kv_cache.py`, `radix_cache.py`, `scheduler.py`, `forward_batch.py`, `model/`, `model_gptq/`, `tokenizer.py` — is unchanged. The draft model and target model each have their own complete `KVPool`, `ReqToTokenPool`, and `RadixCache` (if enabled). The speculation logic operates entirely at the `SpecRunner` level, coordinating two independent `ModelRunner` instances.

The sections below explain each component in detail: the VRAM split between the two models, the draft KV mirror, the draft phase, the verify extend, the accept/reject rule, KV rewind, and statistics.
