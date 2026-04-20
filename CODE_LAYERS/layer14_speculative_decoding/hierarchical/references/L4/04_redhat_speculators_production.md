# Speculators: Standardized, Production-Ready Speculative Decoding

**Source:** https://developers.redhat.com/articles/2025/11/19/speculators-standardized-production-ready-speculative-decoding
**Authors:** Red Hat Developer (Speculators team)
**Published:** November 19, 2025
**Level:** L4 — Production ecosystem; standardized speculator format; vLLM integration; EAGLE-3 model releases for Llama, Qwen3, Llama-4-Maverick
**Why here:** Addresses the ecosystem fragmentation problem: every speculator model was in a different format, different repo, different interface. Speculators v0.2.0 defines a standardized HuggingFace `speculators_config` that makes speculator deployment a one-liner. The performance numbers for Llama-4-Maverick (4.9× in high-throughput!) are a striking contrast to the typical 1.5–2.5× expected range.

---

## The problem Speculators solves

Despite speculative decoding's benefits, widespread production adoption was hampered by:

1. **Lack of standard format** — ecosystem fragmentation and complex hyperparameter management. Every speculator repo used its own format.
2. **Research code that doesn't scale** — most implementations were not hardened for production workloads.
3. **Model-specific speculator models** — state-of-the-art algorithms (EAGLE-3) need speculators trained to match the specific verifier model. No standardized training or distribution mechanism existed.

**Solution:** Speculators defines a standardized HuggingFace configuration (`speculators_config` in `config.json`) for speculator models. One format → vLLM compatibility automatically.

---

## Performance numbers (released with v0.2.0)

Math reasoning benchmark, across multiple models:

| Model | Hardware | Speedup (low-latency regime) | Peak speedup |
|-------|----------|------------------------------|-------------|
| Qwen3-32B | 2× A100 | 2–2.7× | — |
| Llama-3.3-70B-Instruct | 4× A100 | 2–2.7× | — |
| Llama-4-Maverick-17B-128E | 8× B200 | 2–2.7× | **4.9×** |

The Llama-4-Maverick result (4.9×) stands out: it's a large MoE model (17B active / 128 experts) with a speculator converted from NVIDIA's EAGLE-3. The high-throughput regime speedup reaching 4.9× is exceptional — most models see gains only in low-latency (batch=1) settings.

---

## Released speculator models (v0.2.0)

All available under `RedHatAI/` on HuggingFace Hub ([collection](https://huggingface.co/collections/RedHatAI/speculator-models-68c39684ac2649111619f068)):

| Base model | Speculator model |
|-----------|-----------------|
| Llama-3.1-8B-Instruct | `RedHatAI/Llama-3.1-8B-Instruct-speculator.eagle3` |
| Llama-3.3-70B-Instruct | `RedHatAI/Llama-3.3-70B-Instruct-speculator.eagle3` |
| Llama-4-Maverick-17B-128E-Instruct | `RedHatAI/Llama-4-Maverick-17B-128E-Instruct-speculator.eagle3` |
| Qwen3-8B | `RedHatAI/Qwen3-8B-speculator.eagle3` |
| Qwen3-14B | `RedHatAI/Qwen3-14B-speculator.eagle3` |
| Qwen3-32B | `RedHatAI/Qwen3-32B-speculator.eagle3` |
| gpt-oss-20b | `RedHatAI/gpt-oss-20b-speculator.eagle3` |

Typical speedup: **1.5–2.5×** across math reasoning, coding, summarization, RAG. Peak observed: **>4×**.

---

## How to use Speculators

### Deploy in one command

```bash
vllm serve --model RedHatAI/Qwen3-8B-speculator.eagle3
```

That's it. The `speculators_config` in the model's `config.json` tells vLLM everything it needs: which algorithm, which verifier architecture, which draft model configuration.

### Convert an existing EAGLE model to Speculators format

```python
from speculators.convert.eagle.eagle3_converter import Eagle3Converter

speculator_model = "nvidia/Llama-4-Maverick-17B-128E-Eagle3"
base_model = "meta-llama/Llama-4-Maverick-17B-128E-Instruct"
output_path = "Llama-4-Maverick-17B-128E-Instruct-speculator.eagle3"

converter = Eagle3Converter()
converter.convert(
    input_path=speculator_model,
    output_path=output_path,
    base_model=base_model,
    validate=True,
    norm_before_residual=False,
    eagle_aux_hidden_state_layer_ids=[1, 23, 44],
)
```

Then serve:
```bash
vllm serve --model Llama-4-Maverick-17B-128E-Instruct-speculator.eagle3 -tp 8
```

This conversion API is the ecosystem bridge: anyone who trains an EAGLE model can now make it instantly deployable with vLLM via Speculators format.

---

## Supported algorithms and verifier architectures (v0.2.0)

**Algorithms:**
- EAGLE
- EAGLE-3
- HASS (Harmonized Speculative Sampling)

**Verifier architectures:**
- Llama-3
- Llama-4
- Qwen3
- gpt-oss

Upcoming: Qwen3 MoE, Qwen3-VL.

---

## How to benchmark with GuideLLM

```bash
GUIDELLM_PREFERRED_ROUTE="chat_completions" \
guidellm benchmark \
  --target "http://localhost:8000/v1" \
  --data "RedHatAI/speculator_benchmarks" \
  --data-args '{"data_files": "math_reasoning.jsonl"}' \
  --rate-type sweep \
  --max-seconds 600 \
  --output-path "speculative_decoding_benchmark.json"
```

GuideLLM sweeps from synchronous (1 request at a time) to maximum throughput (hundreds of concurrent requests), 600 seconds per rate. This generates the latency-throughput curve that reveals where speculative decoding helps (low request rates) and where it degrades (saturated throughput).

---

## The ecosystem status it implies

As of November 2025, speculative decoding is transitioning from "research trick" to "standardized production component." The Speculators project signals:

1. **Standardization is happening:** HuggingFace format → universal compatibility
2. **Model family coverage is growing:** 7 models across 3 architectures in the first release
3. **Training pipeline is next:** v0.2.0 ships deployed models; v0.3+ will ship training code
4. **vLLM is the target production engine:** Single-command deployment via vLLM

**The missing piece (as of v0.2.0):** Training code is still preliminary. If you need to train your own speculator (e.g., for a fine-tuned model, or a model not in the RedHatAI collection), you're still relying on the original EAGLE training repos or IBM's fms-fsdp recipe (see L3/01).

---

## How this maps to Layer 14

| Speculators concept | Layer 14 equivalent |
|--------------------|---------------------|
| `speculators_config` in `config.json` | Layer 14's `SpecRunner` config structure |
| Draft model + verifier relationship | `DraftModelRunner` + `ModelRunner` in `spec_runner.py` |
| EAGLE-3 speculator head | Extension beyond Layer 14's two-model STANDALONE architecture |
| vLLM compatibility via unified format | Same acceptance/rejection math; different engine plumbing |
| `vllm serve --model RedHatAI/...speculator.eagle3` | One-liner that replaces Layer 14's full `SpecRunner` setup |
| GuideLLM sweep benchmark | More comprehensive than the `lesson/07_statistics.md` per-request tracking |

---

## Limits of this article (for book context)

- v0.2.0-specific: architecture support and model availability will evolve rapidly
- Training pipeline is not yet released — training a new speculator still requires original EAGLE repos
- Does not explain how the `speculators_config` format works internally (the standard format spec is not fully documented)
- Performance numbers (4.9× for Llama-4-Maverick) are with 8× B200 GPUs — not representative of typical H100 deployments
