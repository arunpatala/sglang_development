# 08 — What Comes Next

Layer 13 reduces weight memory by 4× through GPTQ quantization, freeing VRAM for larger KV pools and enabling the target model to coexist with a second, smaller model on the same GPU. This coexistence is exactly what Layer 14's speculative decoding exploits.

---

## The Remaining Bottleneck

GPTQ reduces the weight load per token but does not change the decode throughput formula. Each `decode_step` still commits exactly one token per request, regardless of how fast the 28-layer forward pass runs. The output token rate per request is `1 / (28_layers × layer_latency)`, and reducing layer latency by 3.8× (from the weight bandwidth reduction) directly improves throughput — but the structure remains sequential: one token out per model evaluation.

For applications requiring long generated sequences (code generation, document summarization), even a fast decode step must be repeated hundreds of times. The total output latency is `N_tokens × step_latency`. Halving `step_latency` halves total latency, but committing multiple tokens per step would reduce total latency by `N_tokens / (accepted_tokens_per_step)`.

---

## Speculative Decoding

Layer 14 adds a `SpecRunner` that owns two `ModelRunner` instances: a target (Qwen3-1.7B, optionally GPTQ) and a draft (Qwen3-0.6B). On each speculation step:

1. The draft model runs `decode_step` N times autoregressively, generating N candidate tokens. Draft decode is fast because Qwen3-0.6B has fewer parameters and lower FFN dimension.
2. The target model runs one `_verify_extend` call — a single batched extend over `[last_confirmed, d1, ..., dN]` (N+1 tokens). This is exactly `prefill_batch` machinery applied to the candidate tokens.
3. The greedy accept/reject rule compares target and draft argmax at each position. The longest accepted prefix `[d1, ..., dk]` plus one bonus token (the target's correction at the rejection site) is committed.

The expected committed tokens per target call is `1 + N × acceptance_rate`. For Qwen3 0.6B vs 1.7B at temperature 0, acceptance rates of 0.65–0.75 are typical, giving `1 + 5 × 0.7 = 4.5` tokens per target call with `N=5`.

The total step cost is `N × draft_cost + target_verify_cost`. GPTQ quantization makes both costs smaller — but most importantly it makes it feasible to fit both models on one GPU: the 1.7B GPTQ model takes approximately 1.25 GB and the 0.6B bfloat16 model takes approximately 1.2 GB; together they consume 2.45 GB, leaving more than 13 GB for two KV pools on a 16 GB card.

---

## What Changes in Layer 14

A single new file — `spec_runner.py` — contains the `SpecRunner` class. `SpecRunner.__init__` loads two `ModelRunner` instances with explicit `kv_memory_fraction` parameters that sum to less than 1 (e.g., `target_kv_fraction=0.35`, `draft_kv_fraction=0.45`). `spec_decode_step` orchestrates the draft phase, verify extend, accept/reject, and KV rewind in the correct order.

`server.py` instantiates `SpecRunner` instead of `ModelRunner` when `speculative_decoding` is enabled in the config. All K/V management, page allocation, `RadixCache`, `Scheduler`, and model files are unchanged. The speculative decoding mechanism operates entirely on top of the existing `prefill_batch` and `decode_step` infrastructure.
