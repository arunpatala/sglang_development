# vLLM Speculative Decoding Docs (v0.8.5)

**Source:** https://docs.vllm.ai/en/v0.8.5/features/spec_decode.html
**Level:** L4 — Production engine docs parallel to SGLang; full Python + CLI configs for all four speculation modes; losslessness guarantees and floating-point caveats
**Why here:** vLLM is the other major production engine for speculative decoding, alongside SGLang. Reading both docs side-by-side reveals how the same algorithm (Layer 14's `SpecRunner`) is configured differently in two production systems. The vLLM docs also have the clearest discussion of the floating-point precision caveat that limits the mathematical losslessness guarantee in practice.

> **Important warning from the docs:** Speculative decoding in vLLM v0.8.5 is not yet optimized and does not usually yield inter-token latency reductions for all prompt datasets or sampling parameters. Not compatible with pipeline parallelism.

---

## Four speculation modes

### 1. Draft model (`model` key)

The Layer 14 equivalent — separate smaller model proposes tokens:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="facebook/opt-6.7b",
    tensor_parallel_size=1,
    speculative_config={
        "model": "facebook/opt-125m",
        "num_speculative_tokens": 5,
    },
)
```

CLI equivalent:
```bash
python -m vllm.entrypoints.openai.api_server \
    --model facebook/opt-6.7b \
    --speculative_config '{"model": "facebook/opt-125m", "num_speculative_tokens": 5}'
```

> **Note:** The old `--speculative_model` + separate params method is deprecated. Always use `--speculative_config` as a JSON dict.

### 2. N-gram matching (`method: "ngram"`)

No draft model — matches n-grams from the prompt:

```python
llm = LLM(
    model="facebook/opt-6.7b",
    speculative_config={
        "method": "ngram",
        "num_speculative_tokens": 5,
        "prompt_lookup_max": 4,
    },
)
```

Best for code editing, RAG, or any prompt-heavy workload where output repeats the input.

### 3. MLP Speculators

Draft model conditioning on both context vectors and sampled tokens (IBM-style; see `L3/01_pytorch_hitchhikers_guide.md`):

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3.1-70B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "ibm-ai-platform/llama3-70b-accelerator",
        "draft_tensor_parallel_size": 1,  # MLP speculators must use tp=1
    },
)
```

Available MLP speculator models on HuggingFace Hub:
- `ibm-ai-platform/llama3-8b-accelerator`
- `ibm-ai-platform/llama3-70b-accelerator`
- `ibm-ai-platform/llama2-70b-accelerator`
- `ibm-granite/granite-7b-instruct-accelerator`
- `ibm-granite/granite-20b-code-instruct-accelerator`

Constraint: MLP speculators currently must run **without tensor parallelism** (`draft_tensor_parallel_size=1`), even if the target uses TP.

### 4. EAGLE-based draft models

Draft model that uses hidden-state context from the target model:

```python
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "model": "yuhuili/EAGLE-LLaMA3-Instruct-8B",
        "draft_tensor_parallel_size": 1,  # EAGLE also requires tp=1 for draft
    },
)
```

Available EAGLE models (HuggingFace `yuhuili` collection):

| Base Model | EAGLE Model | Parameters |
|-----------|-------------|-----------|
| LLaMA3-Instruct 8B | `yuhuili/EAGLE-LLaMA3-Instruct-8B` | 0.25B |
| LLaMA3-Instruct 70B | `yuhuili/EAGLE-LLaMA3-Instruct-70B` | 0.99B |
| Qwen2-7B-Instruct | `yuhuili/EAGLE-Qwen2-7B-Instruct` | 0.26B |
| Vicuna-7B-v1.3 | `yuhuili/EAGLE-Vicuna-7B-v1.3` | 0.24B |

> **EAGLE known issue:** Observed speedup in vLLM is lower than the reference EAGLE implementation. Under investigation at [vllm-project/vllm#9565](https://github.com/vllm-project/vllm/issues/9565).

---

## Losslessness guarantees — three-tier breakdown

This is the most important section in the vLLM docs for understanding production correctness.

### Tier 1: Theoretical losslessness

The speculative sampling algorithm (Chen et al.) guarantees that the output distribution is identical to the target model — **up to floating-point precision limits**. This is the theoretical guarantee from the paper.

### Tier 2: Algorithmic losslessness (vLLM validates this)

vLLM's implementation is algorithmically validated to be lossless via two test suites:

1. **Rejection Sampler Convergence test:** Confirms samples from vLLM's rejection sampler converge to the target distribution. ([View test](https://github.com/vllm-project/vllm/blob/main/tests/samplers/test_rejection_sampler.py))

2. **Greedy Sampling Equality test:** Confirms greedy decoding with speculative decoding matches greedy decoding without it. All tests in [tests/spec_decode/e2e](https://github.com/vllm-project/vllm/tree/main/tests/spec_decode/e2e) verify this property.

### Tier 3: vLLM logprob stability (NOT guaranteed)

vLLM does **not** guarantee stable token log probabilities across runs. This can cause different outputs for the same request across different runs — even without speculative decoding.

**Why this matters for Layer 14:** The Layer 14 code produces outputs that are theoretically lossless (the math is correct), but in practice they may differ from non-speculative outputs due to:
- Floating-point precision differences between the batched verification forward pass and individual token forward passes
- Non-deterministic batching behavior in continuous batching systems

---

## Sources of output variation (practical gotchas)

These are the real-world reasons speculative decoding may produce different outputs from non-speculative generation:

| Cause | Effect | Mitigation |
|-------|--------|-----------|
| Floating-point precision | Different probabilities → different samples | Set `temperature=0` (greedy) for deterministic comparison |
| Batch size changes | Logprobs vary with batch composition | Use fixed batch sizes in testing |
| Non-deterministic GPU ops | Different kernel results across runs | Set random seeds; use deterministic kernels |
| Logprob instability in vLLM | Same request → different logprobs across runs | vLLM FAQ: accept this as a known limitation |

---

## Resources for vLLM contributors

The docs link these resources for engineers working on speculative decoding in vLLM:

- **"A Hacker's Guide to Speculative Decoding in vLLM"** (YouTube) — architectural walkthrough
- **Lookahead scheduling** (Google Doc) — how vLLM schedules speculative decoding requests
- **Batch expansion** (Google Doc) — how K draft tokens are batched for parallel verification
- **Dynamic speculative decoding** (GitHub issue #4565) — variable draft length implementation

These are the L5 entry points for contributing to vLLM's speculative decoding implementation.

---

## vLLM vs SGLang comparison (Layer 14 context)

| Feature | vLLM v0.8.5 | SGLang |
|---------|-------------|--------|
| Draft model mode | `"model"` key in `speculative_config` | `--speculative-algorithm STANDALONE` |
| N-gram mode | `"method": "ngram"` | `--speculative-algorithm NGRAM` |
| EAGLE support | Yes (with huggingface models) | Yes (EAGLE/EAGLE-2/EAGLE-3) |
| MLP speculators | Yes (IBM accelerator format) | Not explicitly listed |
| Pipeline parallelism | **Not supported** | Supported |
| Draft TP constraint | draft must use `tp=1` | Same constraint for EAGLE/STANDALONE |
| Optimization status | "Not yet optimized" (per docs) | Considered "among the fastest open-source" |
| Configuration API | JSON dict via `speculative_config` | CLI flags |

The same Layer 14 algorithm runs under both engines. The production difference is in scheduling, CUDA graph optimization, and the continuous batching integration — not in the speculative decoding algorithm itself.

---

## How this maps to Layer 14

| vLLM concept | Layer 14 code |
|-------------|---------------|
| `"model"` key in speculative_config | `SpecRunner` → `DraftModelRunner` instantiation |
| `"num_speculative_tokens"` | `num_spec_tokens` |
| `"draft_tensor_parallel_size": 1` | Layer 14 also runs draft on a single GPU (implied) |
| Rejection sampler convergence test | `_accept_reject()` loop — correctness validation |
| Floating-point precision caveat | Why Layer 14 uses greedy acceptance in default mode |
| N-gram mode | Extension: no separate `ModelRunner` needed |
| EAGLE mode | Extension: draft head inside target, not separate model |
