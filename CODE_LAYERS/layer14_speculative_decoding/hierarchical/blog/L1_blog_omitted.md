# Omitted Sections — blog_speculative_decoding.md

**What this file is:** Content that was deliberately excluded from `blog_speculative_decoding.md` to preserve clarity and narrative flow for a general audience. Every section here came from `references/L1/COMBINED.md`. Nothing is lost — only deferred.

**Why these were omitted:** Each section is either (a) technically deep enough to belong in L2/L3 treatment, (b) too implementation-specific for an introductory article, or (c) audience-scoped narrowly enough that it would alienate the general reader the blog targets.

---

## §A — Advanced Variants: EAGLE-3, MTP, Tree Attention, Self-Speculative

*Source: COMBINED.md §8 — from NVIDIA Developer Blog (Sep 2025) and Google Research (Dec 2024)*

*Omitted because:* These are L3-depth architectural decisions. An L1 reader cannot act on them and they require understanding the baseline mechanism well first. The blog mentions them exist ("the field has kept going") and points here.

---

### EAGLE and EAGLE-3

**EAGLE (Extrapolation Algorithm for Greater Language-Model Efficiency):** Rather than using a separate second model, EAGLE operates at the **feature level**. A lightweight autoregressive prediction head ingests **hidden states** from the target model's internal layers — low, middle, and high-level embeddings — to predict next tokens. No separate model to train; no tokenizer mismatch risk; the draft head is coupled to the target.

**EAGLE-3 improvements over the baseline:**

- **Multi-layer fused features:** the drafting head takes embeddings from multiple layers of the target (not just the top), giving it richer signal about internal model state.
- **Dynamic draft tree:** candidate selection uses beam search over cumulative log probabilities — context-aware, not fixed-length.
- **Instance-adaptive drafting:** the head evaluates its own confidence as it builds the tree and stops drafting below a threshold, avoiding wasted computation on low-confidence speculations.
- **Tree attention:** the target model verifies the **entire candidate tree** in one forward pass, pruning invalid branches. Multiple continuation paths are checked simultaneously.

The EAGLE head = a lightweight Transformer decoder layer + a final linear layer. It generates a **tree** of candidates, not a single linear sequence. This is why tree attention is needed for verification.

---

### Multi-Token Prediction (MTP)

Used in DeepSeek-R1 style models. MTP bakes the drafting mechanism **into the model architecture itself**:

- The base model has multiple specialized prediction heads, each trained to predict a token at an increasing future offset.
- At generation time, each head drafts one future token.
- The main model verifies in order, keeping the longest accepted prefix.
- No separate draft model is needed — the prediction is architectural.

**How MTP differs from EAGLE:**
- MTP heads are trained **jointly with the base model** — they are part of the original training objective.
- EAGLE adds a lightweight head **after training**, extrapolating internal feature states.
- MTP requires the model to have been designed for it from the start. EAGLE can be applied to an existing pretrained model.

---

### The broader extension landscape

The draft-then-verify paradigm has been extended beyond the two-model baseline in several directions:

- **Distributed setups** — multiple draft models generating diverse candidate sequences simultaneously; the target verifies the best.
- **Knowledge distillation** — training the draft model specifically to mimic the target's output distribution, increasing acceptance rates beyond what a generic small model achieves.
- **Self-speculative decoding** — one model acts as both draft and target, using early exit layers or layer-skipping to generate cheap draft tokens from the same weights.
- **Tree attention** — verifying all draft tokens together via a shared prefix tree; parallel verification of branching candidate paths.
- **Image and speech generation** — the same draft-verify idea applied to non-text autoregressive models; any model that generates tokens sequentially can use this approach.

*For full implementation details: `blog_l3_technical.md`, and the L4/L5 references in `references/SURVEY/`.*

---

## §B — Production Systems: P99 Depth, MoE Specifics, Cascades, Draft Training

*Source: COMBINED.md §9 — from Nebius Blog*

*Omitted because:* The blog article covers the core P99 insight (tail latency, not throughput averages; speculative decoding reshapes the tail). This section contains the supporting details for readers who deploy or manage inference systems — too product-specific for the general audience the blog targets.

---

### Why MoE architectures stress the problem

Large Mixture-of-Experts models are uniquely punishing because they stretch multiple constraints simultaneously:

- Long context windows → **prefill dominates before decode even starts**. At ~10,000 token inputs, a large fraction of total latency is paid before the first output token is generated. This cost cannot be amortized by batching without directly worsening tail latency.
- Expert routing overhead → less predictable memory access patterns than dense models
- Non-trivial decode lengths → long, variable decode tails under load
- Sensitivity to queueing at moderate concurrency levels

Two systems with very different peak compute can exhibit surprisingly similar end-to-end latency for the same workload — because raw compute is no longer the bottleneck.

---

### Why replica scaling doesn't fix P99

The instinctive response to latency problems: add more replicas.

More replicas reduce queue depth and improve averages. They **rarely fix tails**. Under sustained load, long-context requests magnify small variations in execution time. Once the system crosses a concurrency threshold, P99 latency stops improving smoothly and **begins to cliff** — adding capacity beyond that point yields diminishing or negative returns for tail behavior.

This is why systems that appear healthy at low traffic **suddenly violate SLAs at moderate load**, despite looking fine in utilization dashboards.

---

### Streaming masks problems that non-streaming exposes

Many chat interfaces stream tokens as they are generated, surfacing partial output immediately. This masks prefill and early decode latency — the user sees something moving quickly, even if total generation is slow.

**Non-streaming products have no escape hatch.** If the user sees nothing until the full response is ready, end-to-end latency is the only metric that matters. Prefill delay is fully visible. Tail decode is fully visible.

This is why many MoE deployments succeed in demos (which are usually streaming) and fail in products (which often are not).

---

### Cascaded systems compound the budget

Real products rarely run a single model. Safety classifiers, guards, rerankers, and post-processors each consume part of the total latency budget. A system that barely meets a 10-second SLA in isolation is usually unusable inside a cascade.

Headroom matters. Tail behavior compounds across stages. **Speculative decoding's contribution to headroom** — not just mean latency, but predictable tail behavior — becomes the actual product argument for adopting it.

---

### Draft model training as production infrastructure

> *"Treating draft model training as part of the production pipeline, rather than an experiment, is what turns speculative decoding into a reliable architectural primitive."* — Nebius

A generic off-the-shelf draft model (same model family, smaller size) delivers real gains. A draft model that has been **post-trained on synthetic inputs resembling real production conversations** delivers:

- Higher and more consistent acceptance rates
- Tighter P99 tail behavior — less variance in how many draft tokens are accepted per request

The argument: draft model alignment should be treated as **serving infrastructure**, not a research exercise. Teams that invest in it get a reliable, bounded system. Teams that treat it as optional get unpredictable gains.

*For knowledge distillation approaches and evaluation methodology: L5 training references (`references/L5/`).*

---

### Design it in early

> *"For long-context, non-streaming systems, speculative decoding is not an optimization — it is a prerequisite for meeting real product SLAs."* — Nebius

Retrofitting speculative decoding into an existing serving stack is painful — it touches serving pipelines, batching behavior, memory management (KV budget split between draft and target), and observability tooling (acceptance rate monitoring).

Designed in from the beginning, teams can reason coherently about capacity, headroom, and failure modes before they become production incidents.

---

## §C — Practical Setup and Code Examples

*Source: COMBINED.md §11 (Chris Thomas) and §12 (NVIDIA)*

*Omitted because:* These are implementation-specific and library-specific. The blog article establishes the concept and confirms it's deployed — that's enough. Readers who want to run it need the implementation context first.*

---

### Local setup: llama.cpp + VS Code

Using the `llama-vscode` extension and `llama.cpp` HTTP server to run speculative decoding locally:

```bash
llama-server.exe \
  -md qwen2.5-coder-3b-q8_0.gguf \
  -m qwen2.5-coder-14b-q8_0.gguf \
  --port 8012 -c 2048 \
  --n-gpu-layers 99 -fa \
  -ub 1024 -b 1024 \
  -dt 0.1 --ctx-size 0 --cache-reuse 256
```

Key flags:
- `-md` — draft model (3B, fast)
- `-m` — main/target model (14B, authoritative)
- `--cache-reuse 256` — reuse KV cache across requests (important for acceptance rate)
- `-dt 0.1` — draft temperature (lower = more conservative/higher acceptance rate)

The `llama-vscode` extension exposes three accept gestures:
- **Tab** — accept the full suggestion
- **Shift+Tab** — accept only the first line
- **Ctrl/Cmd+Right** — accept the next word

This setup runs on mid-range consumer hardware. The 3B draft model is cheap enough that its cost is negligible relative to the 14B target, and the Qwen 2.5 Coder family has good alignment between sizes, leading to solid acceptance rates for code completion tasks.

---

### NVIDIA TensorRT EAGLE-3 configuration

For readers using NVIDIA's TensorRT-Model-Optimizer stack:

```python
import transformers
import modelopt.torch.opt as mto
import modelopt.torch.speculative as mtsp
from modelopt.torch.speculative.config import EAGLE3_DEFAULT_CFG

mto.enable_huggingface_checkpointing()

base_model = "meta-llama/Llama-3.2-1B"
model = transformers.AutoModelForCausalLM.from_pretrained(
    base_model, torch_dtype="auto", device_map="cuda"
)

config = EAGLE3_DEFAULT_CFG["config"]
config["eagle_architecture_config"].update({
    "hidden_size": model.config.hidden_size,
    "vocab_size": model.config.vocab_size,
    "draft_vocab_size": model.config.vocab_size,
    "max_position_embeddings": model.config.max_position_embeddings,
})

mtsp.convert(model, [("eagle", config)])
```

**What `mtsp.convert()` does:** attaches the EAGLE-3 drafting head to the base model, wiring the multi-layer feature extraction from the target's internal layers into the prediction head. The result is a single model object that can draft and verify internally.

Full tutorial: [TensorRT-Model-Optimizer/examples/speculative_decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb)

> **Note:** This uses NVIDIA's `modelopt` library, not SGLang or raw PyTorch. The Layer 14 book chapter uses SGLang's native speculative decoding configuration — the pattern is the same but the API differs.

---

## §D — Three Analogies: The Two That Were Dropped

*Source: COMBINED.md §3*

*Omitted because:* The blog uses one analogy (junior/senior developer) consistently throughout. Presenting multiple analogies for the same mechanism creates cognitive load and dilutes each one. These two are preserved here for teaching contexts where a different framing might land better.*

---

### Chief scientist and lab assistant *(NVIDIA)*

A **chief scientist** has deep expertise but is time-expensive. An **efficient assistant** handles routine experiments — rapidly working through checklists, running standard protocols, preparing samples.

The assistant works ahead; the scientist validates. The scientist never produces lower-quality results — they remain the final authority — but far fewer of their expensive steps are needed per output.

**When to use this analogy:** audiences from scientific or research backgrounds, or when explaining speculative decoding in the context of agent systems where the "senior" role resonates as "the model doing reasoning."

---

### Speculative execution in CPUs *(Google Research)*

Speculative execution is a longstanding CPU optimization: perform a task before you know it's needed, to increase concurrency.

```
Y = f(X)        # slow operation
Z = g(Y)        # slow, depends on Y
```

With a fast approximation `f*(X)`:
- Run `f(X)` and `g(f*(X))` in parallel.
- If `f*(X) == f(X)`: accept `g(f*(X))`, two steps done.
- If not: discard `g(f*(X))`, run `g(Y)` serially.

Output is identical either way. The accuracy of `f*` determines how often the parallel work is accepted.

Applied to LLMs: `f` = target model, `f*` = draft model, `g` = the next generation step.

**When to use this analogy:** audiences with systems programming or computer architecture backgrounds, where "speculative execution" is already a familiar concept and the LLM version is best understood as an instance of the same idea.

---

## §E — Key Quotes (Full Collection)

*Source: COMBINED.md §14*

*Omitted from blog because:* quotes are woven into the narrative of the blog article where they are most effective. A quotes section at the end of a blog reads as filler. Preserved here for reuse.*

> *"The key insight isn't just about making things faster — it's about making better use of the computing resources we already have."* — Chris Thomas

> *"Compared with standard autoregressive decoding, which produces one token per pass, this technique lets the system generate multiple tokens at once, cutting latency and boosting throughput without any impact on accuracy."* — NVIDIA

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research)

> *"Producing results faster with the same hardware also means that fewer machines are needed for serving the same amount of traffic, which translates yet again to a reduction in the energy costs of serving the same model."* — Google Research

> *"Throughput numbers hide all of this."* — Nebius

> *"Large MoE inference is not a throughput problem — it is an execution-path problem."* — Nebius

> *"The goal is not faster demos. It is systems whose behavior under stress is understood, measured, and bounded before users ever see them."* — Nebius
