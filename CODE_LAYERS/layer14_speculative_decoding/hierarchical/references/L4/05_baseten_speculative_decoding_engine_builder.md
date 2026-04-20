# Baseten: Speculative Decoding Engine Builder Integration

**Source:** https://www.baseten.co/blog/speculative-decoding-engine-builder-integration/
**Authors:** Baseten engineering team (Justin Yi, lead)
**Published:** May 16, 2025
**Level:** L4 — Production deployment config; TensorRT-LLM based speculative decoding; two-tier configuration approach (pre-optimized or manual tuning); GPU memory allocation caveats
**Why here:** The Baseten article is the most practical "deploy speculative decoding to production right now" guide. Unique for: (1) showing a complete YAML config file for speculative decoding rather than just Python code, (2) explicitly naming when NOT to use speculative decoding (high load, already-lightweight models), and (3) the two-tier approach — default pre-optimized config for most users, full parameter access for power users.

---

## One-liner summary

Baseten's TensorRT-LLM Engine Builder integration adds speculative decoding to any production LLM deployment through a single config file, with pre-optimized defaults and full parameter access for tuning.

**Observed latency reduction:** up to 50% (2× speedup) with no effect on output quality.

---

## When speculative decoding helps (and when it doesn't)

Baseten explicitly names both sides — rare for vendor documentation:

### Good use cases
- **Large models in production** (Llama 3.1 70B or 405B) — where smaller sibling models (Llama 3.1 8B) are nearly as capable
- **Code generation** — highly repetitive, predictable outputs
- **Latency-sensitive applications** with aggressive SLAs (live translation, chatbots, coding assistants)
- **Low-to-moderate GPU load** — when the draft model has compute headroom

### When NOT to use speculative decoding
- **High load (GPU at or near 100%)** — running the draft model in addition to the large one creates bottlenecks
- **Already-lightweight LLMs** — the overhead of running two models outweighs the gains
- **Very small target models** — draft models available are too close in size to the target

This is the production reality check that L2 articles skip. The decision to use speculative decoding depends on your GPU utilization profile, not just the model family.

---

## Complete TensorRT-LLM config (Qwen 2.5 Coder 14B + 0.5B draft)

```yaml
model_metadata:
  tags:
  - openai-compatible
model_name: Qwen2.5-Coder-14B-Instruct (SpecDec)
resources:
  accelerator: H100
  cpu: '1'
  memory: 24Gi
  use_gpu: true
trt_llm:
  build:
    base_model: qwen
    checkpoint_repository:
      repo: Qwen/Qwen2.5-Coder-14B-Instruct
      source: HF
    max_seq_len: 10000
    plugin_configuration:
      paged_kv_cache: true
      use_paged_context_fmha: true
    speculator:
      speculative_decoding_mode: DRAFT_TOKENS_EXTERNAL
      checkpoint_repository:
          repo: Qwen/Qwen2.5-Coder-0.5B-Instruct
          source: HF
      num_draft_tokens: 4
  runtime:
    enable_chunked_context: true
    kv_cache_free_gpu_mem_fraction: 0.62
    request_default_max_tokens: 1000
    total_token_limit: 500000
```

### Key config fields to understand

| Field | Layer 14 equivalent | Notes |
|-------|--------------------|-|
| `speculative_decoding_mode: DRAFT_TOKENS_EXTERNAL` | Two-`ModelRunner` STANDALONE mode | Separate draft model |
| `num_draft_tokens: 4` | `num_spec_tokens` | k in the speedup formula |
| `kv_cache_free_gpu_mem_fraction: 0.62` | `kv_memory_fraction` split | Must accommodate BOTH models' KV caches |
| `paged_kv_cache: true` | `PagedKVPool` | Same paged attention as Layer 14 |

The `kv_cache_free_gpu_mem_fraction: 0.62` is a conservative setting — it leaves 38% of GPU memory for model weights (both draft and target), CUDA graphs, and other overhead. Setting this too high causes OOM when the KV caches grow at runtime.

---

## The two-tier production philosophy

Baseten explicitly chose NOT to treat speculative decoding as a black box:

**Tier 1:** Pre-optimized config files from Baseten engineers — "use this and it works."
**Tier 2:** Full parameter access for power users — "lift the hood when you need to."

This is a deliberate choice against the "fully automated black box" approach used by some cloud providers. The tradeoff: more complexity for the user, but visibility and control for production debugging.

> "While this certainly makes SpecDec easier to use, it fails to provide developers with the visibility or control needed for production AI."

**For Layer 14 readers:** This philosophy mirrors the book's approach — understand the mechanism so you can debug it, not just configure it.

---

## GPU memory allocation: the critical constraint

The most important practical constraint mentioned in the article, buried near the end:

> "Speculative decoding with draft models (e.g., using `DRAFT_TOKENS_EXTERNAL`) is an advanced feature that requires careful GPU memory allocation to accommodate both models simultaneously."

Specifically:
- Both draft and target models must fit in GPU VRAM simultaneously
- Both models' KV caches must fit alongside the weights
- `kv_cache_free_gpu_mem_fraction` governs the KV cache budget; the remaining fraction must accommodate both models' weights

The config above uses a **0.5B draft + 14B target on a single H100 (80GB)**:
- 14B model at fp16 ≈ 28 GB
- 0.5B model at fp16 ≈ 1 GB
- Remaining ~51 GB available for KV caches

With `kv_cache_free_gpu_mem_fraction: 0.62`, 62% of the ~51GB is allocated to KV caches ≈ ~31.6GB. This is why the config can support `max_seq_len: 10000` — each sequence's KV cache is bounded.

**For a 70B + 8B pairing on 2× H100s:** You'd need to reduce `kv_cache_free_gpu_mem_fraction` significantly and may need `kv_cache_free_gpu_mem_fraction: 0.45` or lower. Baseten recommends consulting their support team for large model pairs.

---

## Why code generation is the ideal use case

Baseten focuses on code generation in the demo. This reflects the fundamental alignment problem:

- **Code is repetitive and predictable** — a 0.5B model trained on code can reliably predict what tokens a 14B code model would write
- **Variable names, function calls, keywords, and boilerplate** are highly repetitive across code contexts
- **Draft acceptance rates for code** are typically in the 0.80–0.90+ range, compared to 0.65–0.75 for general text

The Qwen2.5-Coder-0.5B/14B pairing is specifically chosen because both models were trained on the same coding data with the same tokenizer — maximum distribution alignment.

---

## How this maps to Layer 14

| Baseten concept | Layer 14 code |
|----------------|---------------|
| `DRAFT_TOKENS_EXTERNAL` mode | `SpecRunner` with separate `DraftModelRunner` |
| `num_draft_tokens: 4` | `num_spec_tokens = 4` |
| `kv_cache_free_gpu_mem_fraction: 0.62` | `kv_memory_fraction` split between draft and target `KVPool`s |
| `paged_kv_cache: true` | `PagedKVPool` page allocation |
| When NOT to use: high GPU load | Throughput regime where SpecDecode-Bench shows gains disappear |
| When NOT to use: lightweight models | draft model's forward pass costs ≈ target's; `c ≈ 1`, speedup ≈ 1 |

---

## Limits of this article (for book context)

- TensorRT-LLM specific (not vLLM or SGLang) — the YAML config format does not translate directly
- Production pricing considerations (cost per generated token) are implicit, not quantified
- No benchmark numbers for the specific Qwen2.5 14B/0.5B pairing shown in the config
- Does not address the batch size effect — SpecDecode-Bench (L4/03) is the companion for that
- The "consult Baseten support" recommendation for large model pairs is a gap in the public documentation
