# 09 — What Comes Next

Layer 14 achieves approximately 2–3× throughput improvement over Layer 13 by exploiting the cost asymmetry between the draft and target models. The core mechanism — run a cheap model N times, verify with the expensive model once — is the direct SGLang ancestor of production speculative decoding. Several significant extensions remain.

---

## Sampled (Non-Greedy) Speculative Decoding

Layer 14 uses the greedy accept rule: accept if `argmax(target) == draft_token`. This is correct only at temperature 0. For temperature > 0 (stochastic sampling), the greedy rule introduces a distribution mismatch: the target model's output distribution is not the same as the target's argmax. The accepted tokens come from the draft distribution filtered by the target's argmax check, which is different from sampling from the target distribution.

The correct non-greedy accept rule (from the original speculative decoding paper by Leviathan et al.) uses a ratio test:

```
accept d_j with probability  min(1, target_prob(d_j) / draft_prob(d_j))
```

If rejected, sample a correction from the "adjusted" distribution `max(0, target_prob - draft_prob)`. This rule guarantees that the output distribution matches the target model's distribution exactly, regardless of the draft model's distribution. The yield is the same as the greedy rule when both models agree; the correction step provides the right distribution when they disagree.

Implementing this requires retaining both target and draft logit distributions — not just the argmax — through the accept/reject phase. `SpecRunner` would need to run `softmax(logits / temperature)` for both models and implement the ratio test.

---

## Batched Speculation

Layer 14's `spec_decode_step` runs N sequential draft decode steps. For a single request, these cannot be parallelized — each step depends on the previous step's output token. But for a batch of B requests, all B requests can be processed together in each of the N draft steps (which is what `decode_step_for_dreqs([all_reqs], [all_dreqs])` does).

The remaining inefficiency is that the N draft steps are sequential in time. An alternative is tree-based speculation: at each position, instead of committing to one draft token, propose K candidates and evaluate them all in a branching tree. The target verify pass runs over the entire token tree in one extend call, and the longest accepted branch is chosen. This multiplies the effective N by K without multiplying the number of draft steps, at the cost of a more complex tree KV management and a larger verify extend.

Production SGLang implements tree-based Medusa decoding (a draft "head" attached to the target model) and EAGLE (a draft model with access to target hidden states) on top of the same paged KV infrastructure that Layer 14 establishes.

---

## Tensor Parallelism

Layer 14 runs both the target and draft models on a single GPU. For larger models (7B, 14B, 72B), tensor parallelism is required: the attention and FFN weight matrices are sharded across multiple GPUs, with `all-reduce` after the output projection and FFN. In this setting, `SpecRunner` would need to coordinate the draft and target models across multiple GPU ranks, with the draft model potentially running on fewer ranks than the target to balance cost.

The paged KV pool design from Layer 9 is directly compatible with tensor parallelism: each GPU shard holds its KV head subset, and `ReqToTokenPool` is replicated on each rank. The speculative decoding logic at the `SpecRunner` level is identical — only the underlying `ModelRunner` needs to distribute its linear operations.

---

## What Layers 9–14 Establish

Looking at the full series:

- **Layer 9**: GPU-resident page table + Triton index kernel — eliminates O(Σ tokens) CPU overhead per step.
- **Layer 11**: Chunked prefill + PrefillAdder — eliminates unbounded decode starvation.
- **Layer 12**: RadixCache + match_prefix — eliminates redundant KV computation for shared prefixes.
- **Layer 13**: GPTQLinear + gptq_gemm — reduces weight memory 4×, freeing VRAM for larger pools and enabling dual-model coexistence.
- **Layer 14**: SpecRunner + draft/target + accept/reject — increases committed tokens per target call from 1 to N×acceptance_rate+1.

Each layer adds one mechanism. Each mechanism is independent of the others — prefix caching works without speculative decoding; GPTQ works without prefix caching. Together they form the production SGLang stack: a layered, orthogonal set of optimizations on top of the same paged attention foundation.

The `KVPool`, `ReqToTokenPool`, `ForwardBatch`, `ExtendKVCtx`, `DecodeKVCtx`, `Triton kv_indices kernel`, and FlashInfer paged kernel are unchanged from Layer 9 through Layer 14. Understanding Layer 9 is understanding the load-bearing structure. Layers 11–14 are progressively richer features built on top of it.
