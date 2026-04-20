# Speculative Decoding — L1 Combined Reference

**What this file is:** A synthesis of all L1 orientation articles for speculative decoding, combined into a single progressive narrative. Every concept from every source is included. The reading order moves from "why is this needed?" → "how does it work?" → "what the math guarantees" → "advanced variants" → "what it does in real products" → "where it's deployed."

**Sources synthesized:**
- `01` — Chris Thomas (Feb 2025): junior/senior developer analogy, VS Code demo
- `02` — NVIDIA Developer Blog (Sep 2025): chief scientist analogy, EAGLE-3, MTP, TensorRT code
- `03` — Google Research (Dec 2024): original authors, speculative sampling theory, production confirmation
- `04` — Nebius Blog: P99 tail latency, MoE production failures, speculative decoding as architecture

---

## 1. The Problem: Why LLMs Are Slow

Large Language Models generate text **one token at a time**. The sentence *"One small step for man, one giant leap for mankind"* is 12 tokens — the LLM must run 12 forward passes to produce it.

Each forward pass requires reading **all of the model's weights** from memory. For a large model, that is on the order of a terabyte of data per word produced. Since each token depends on all the tokens before it, they must be generated **one by one**, reading all weights again and again.

This process is **memory-bound**: speed is limited not by the GPU's arithmetic capability but by how quickly the model can read its parameters from High-Bandwidth Memory (HBM/DRAM).

Modern ML hardware can perform hundreds of **trillions of operations per second**. Memory bandwidth is around **trillions of bytes per second** — two orders of magnitude lower. The transformer performs only a **few operations per byte read** during inference, which means there are **spare computational resources sitting idle** while memory is the bottleneck.

GPUs offer massive parallelism, yet much of that power is unused during autoregressive generation — each token requires a full forward pass, reloading weights, and synchronizing memory. Even with powerful hardware, code completion and chat responses feel sluggish because **the bottleneck is not computational power, it is how efficiently we use it.**

---

## 2. Two Observations That Unlock the Solution

### Observation 1: Not all tokens are equally hard to generate

> *"What is the square root of 7? The square root of 7 is 2.646."*

Generating the second "7" is easy — it copies the one from the question. Generating "2.646" is hard — the model must compute or recall the answer.

Large models are better than small ones mainly in **difficult cases**. For **easy tokens** — function words, repeated phrases, predictable continuations — a small model can approximate the large model well. If we could route easy tokens to a small model and hard tokens to the large model, we would save significant compute without sacrificing quality.

### Observation 2: Idle compute exists

Since memory is the bottleneck during inference, we can **run extra computation in parallel** without slowing down the overall process. The memory reads are already happening; the arithmetic units are available. What we need is a way to use that idle compute productively.

Together, these observations motivate speculative decoding: use a **small, fast model** to do the easy work in parallel, and use the **large, accurate model** only to verify and correct.

---

## 3. Three Ways to Think About It: The Analogies

Different analogies work for different readers. All three describe the same mechanism.

### Analogy 1: Junior developer and senior developer
*(Chris Thomas)*

Think of two developers working together:
- A **junior developer** who quickly drafts code suggestions — fast but sometimes wrong.
- A **senior developer** who reviews those suggestions — slower but authoritative.

Instead of the senior developer writing everything from scratch (baseline autoregressive decoding), they can review multiple lines at once. The junior drafts a block; the senior scans it and accepts what's right, corrects the first mistake, and resumes from there.

**Result:** faster output with the same final quality as the senior working alone.

### Analogy 2: Chief scientist and lab assistant
*(NVIDIA)*

A **chief scientist** has a deep understanding of their field but is time-expensive. An **efficient assistant** handles routine experiments — rapidly working through checklists, setting up equipment, making preliminary measurements.

The assistant rapidly works through the checklist; the scientist validates and steps in to correct when necessary. The scientist never produces lower-quality results — they are still the final authority — but far fewer of their "expensive steps" are needed to produce the same output.

### Analogy 3: Speculative execution in CPUs
*(Google Research)*

Speculative execution is an established CPU optimization: perform a task before you know if it is needed, in order to increase concurrency.

In abstract form:

```
Y = f(X)        # slow operation
Z = g(Y)        # slow, depends on Y
```

If we have a fast approximation `f*(X)`:
- Run `f(X)` and `g(f*(X))` in parallel.
- When `f(X)` finishes: if `f*(X) == f(X)`, accept the result of `g(f*(X))`.
- If the approximation was wrong: discard and run `g(Y)` serially.

**Output is guaranteed to be identical either way.** The more accurate `f*`, the more concurrency gained. Applied to LLMs: `f` is the target model, `f*` is the draft model, `g` is the next generation step.

---

## 4. How Speculative Decoding Works: Step by Step

Two models work together:
- The **target model** — large, high-quality, the final authority. (e.g., Qwen 2.5 Coder 14B)
- The **draft model** — smaller, faster, often distilled from the target or from the same model family. (e.g., Qwen 2.5 Coder 1.5B or 3B)

### Step 1 — Draft generation

The draft model generates 3–12 candidate tokens **very quickly**. Because the draft model is much smaller, each of its forward passes takes a fraction of the time of the target model's forward pass.

Typically the draft model is a separate smaller model trained on the same data distribution, with the target's output as ground truth — so its predictions tend to align with the target's.

### Step 2 — Parallel verification

The target model processes the **entire input sequence plus all draft tokens simultaneously** in a single forward pass. It computes probability distributions for each position.

The KV cache holds previously computed values for the input sequence, so only the new speculated tokens incur compute cost during verification. This single forward pass of the target model checks all draft tokens at once — the same cost as generating one token, but producing information about many.

### Step 3 — Accept or reject

For each draft position, the target model compares:
- `P(Target)` — what the target assigns to the drafted token
- `P(Draft)` — what the draft model assigned to the same token

The rule:
- If `P(Target) ≥ P(Draft)`: token **accepted** — the target would have produced this token with at least as high probability as the draft did.
- If `P(Target) < P(Draft)`: token **rejected** — the draft was overconfident; discard this token and all subsequent draft tokens; revert to standard generation from the last accepted position.

A bonus token is sampled from the target model's distribution at the rejection point, so the system always makes forward progress.

**Worst case:** all draft tokens rejected → same as baseline (one target token per forward pass).  
**Best case:** all draft tokens accepted → many tokens committed per target model call.

### The correctness guarantee

Both the greedy variant (accept if the draft's top-1 matches the target's top-1) and the sampling variant (probabilistic acceptance based on `P(Target)/P(Draft)`) produce output that is **mathematically identical** to standard decoding.

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research)

This is the central promise: speculative decoding is **lossless**. Users see no degradation. The only risk is in misconfiguration (e.g., mismatched tokenizers), not in the algorithm itself.

---

## 5. What Speedup Looks Like: Concrete Numbers

**Original paper results** (Google Research, 2022–2023):
- Target model: **11B parameter T5-XXL** for translation
- Draft model: **60M parameter T5-small**
- Result: **~3× improvement in speed**

**Interactive timing example** (NVIDIA):
- Standard autoregressive decode: 3 tokens × 200 ms/pass = **600 ms**
- Speculative decode (2 draft tokens, 1 verification pass): verification pass ≈ 250 ms, 3 tokens committed = **250 ms**
- Users see responses materialize in faster, **multi-token chunks** rather than word by word — especially noticeable in chatbots

**Real-world range** (Chris Thomas, coding tools):
- **2–4× faster** suggestions in code completion
- Mid-range machines can run sophisticated code completion locally at usable speeds

The bottleneck matters: the draft model must be significantly faster than the target for the parallelism to pay off. In practice, draft models are 10–100× smaller than the target.

---

## 6. Three Factors That Determine Efficiency

*(Chris Thomas)*

1. **Draft model size:** The draft must be significantly smaller than the target so it generates tokens quickly. The smaller the draft model, the lower its per-token cost — but also the lower its accuracy.

2. **Draft length:** More draft tokens proposed per step = more potential speedup, but also more chances of rejection. A draft of 8 tokens that gets accepted 4 of them beats a draft of 3 tokens that gets accepted 3 of them.

3. **Acceptance rate:** The draft model must be well-aligned with the target so enough draft tokens are accepted. If the draft and target are from the same model family, share the same tokenizer and training data, acceptance rates are naturally higher.

These three factors interact: a smaller draft model generates faster but accepts less; a longer draft exploits more parallelism but compounds rejection risk. Tuning these is the practical job of deploying speculative decoding in production.

---

## 7. What the Hardware Is Actually Doing

*(Chris Thomas + NVIDIA)*

While the large model is loading its weights from memory (memory-bound), the smaller model is already generating suggestions (compute-bound — much cheaper). This parallel operation is what makes speculative decoding efficient:

- **Reducing memory access:** Instead of generating each token with the target model sequentially, the target model processes multiple draft tokens in a single pass — its parameters are read fewer times.
- **Parallel processing:** The target model evaluates multiple draft tokens in one forward pass (as a batch), leveraging the transformer's inherent parallelizability.
- **Compute/bandwidth balance:** LLMs are bandwidth-limited. Speculative decoding trades available idle compute for lower bandwidth requirements.

The key efficiency: **target model parameters are loaded fewer times per output token produced**.

---

## 8. Advanced Variants: Beyond the Baseline

The basic draft-target setup has inspired many extensions. These are mentioned here at L1 depth — deeper treatment is in L4/L5 references.

### EAGLE and EAGLE-3 *(NVIDIA)*

**EAGLE (Extrapolation Algorithm for Greater Language-Model Efficiency):** Rather than using a separate second model, EAGLE operates at the **feature level** — a lightweight autoregressive prediction head that ingests **hidden states** from the target model's internal layers to predict next tokens. No separate model to train; no tokenizer mismatch risk.

**EAGLE-3 improvements:**
- **Multi-layer fused features:** takes low, middle, and high-level embeddings from the target into the drafting head, giving it richer signal
- **Dynamic draft tree:** context-aware, beam-search candidate selection based on cumulative log probabilities
- **Instance-adaptive drafting:** the head evaluates its own confidence as it builds the tree; stops drafting below a threshold
- **Tree attention:** the target verifies the **entire candidate tree** in one pass, pruning invalid branches

The EAGLE head = a lightweight Transformer decoder layer + final linear layer, generating a **tree** of candidates rather than a single linear sequence.

### Multi-Token Prediction (MTP) *(NVIDIA)*

Used in DeepSeek-R1 style models. MTP uses **multi-head prediction baked into the model itself**:
- Each head drafts one future token at increasing distances
- The main model verifies in order, keeping the longest accepted prefix
- No separate draft model needed — the prediction is architectural

**Difference from EAGLE:** MTP uses specialized multi-token prediction heads trained jointly with the base model; EAGLE uses a single lightweight head added after training, extrapolating internal feature states.

### The broader landscape *(Google Research, Dec 2024)*

Speculative decoding has spawned many extensions, all sharing the draft-then-verify paradigm:
- **Distributed setups** — multiple draft models generating diverse guesses simultaneously
- **Knowledge distillation** — training the draft model specifically to mimic the target's distribution
- **Self-speculative decoding** — one model acts as both draft and target (using early exit layers or skipped layers)
- **Tree attention** — verifying all draft tokens together in a tree structure (prefix sharing)
- **Image and speech generation** — the same draft-verify idea applied to non-text autoregressive models

---

## 9. What Speculative Decoding Does to Real Products

*(Nebius Blog — this section is unique to the Nebius source)*

### The wrong mental model: throughput as the primary metric

Most inference discussions lead with throughput: tokens per second, requests per minute, accelerator utilization. For interactive products, **this framing is misleading**.

What users experience is **end-to-end latency**. What breaks products is not the mean, but the **tail**: P90 and P99 latency are the real acceptance criteria — whether or not they are written down.

> *"Throughput numbers hide all of this."*

### Why MoE models (and any large model) fail product SLAs

Large MoE models are uniquely punishing because they stretch multiple dimensions at once:
- Long context windows → **prefill dominates before decode even starts**
- Expert routing overhead → less predictable memory access patterns
- Non-trivial decode lengths → long, variable decode tails
- Sensitivity to queueing under moderate concurrency

**Prefill** is the dominant cost for long-context workloads. At input sizes of ~10,000 tokens, a large fraction of total latency is spent before the model generates a single output token. This cost is **paid in full for every request** and cannot be amortized by batching without directly impacting tail latency.

### Why adding replicas does not fix P99

The instinctive response to latency misses: add more replicas.

More replicas reduce queue depth and improve averages. But they **rarely fix the tail**. Under sustained load, long-context requests magnify small variations in execution time. Once the system crosses a concurrency threshold, P99 latency stops improving smoothly and **begins to cliff**. Adding capacity beyond that point yields diminishing or negative returns for tail behavior.

This is why systems that appear healthy at low traffic **suddenly violate SLAs at moderate load**, despite reasonable utilization metrics.

### Streaming masks problems that non-streaming exposes

Many chat interfaces stream tokens as they are generated — masking prefill and early decode latency by surfacing partial output quickly. **Non-streaming products have no escape hatch.** If the user sees nothing until the full response is ready, end-to-end latency is the only metric that matters.

This distinction explains why many deployments succeed in demos and fail in products.

### Cascaded systems tighten budgets further

Real products rarely run a single model. Safety classifiers, guards, rerankers, and post-processors sit before or after the primary model. Each stage consumes part of the latency budget. A system that barely meets a 10-second target in isolation is usually unusable inside a cascade. **Headroom matters. Tail behavior compounds.**

### How speculative decoding changes this

Speculative decoding is often presented as a throughput optimization. In long-context, non-streaming systems, **its real value is different: it reshapes the latency distribution**.

In a baseline setup, every output token is generated by the full model. For large models, each decode step incurs expert routing (in MoE), memory access, and synchronization costs. Under load, this creates **long, variable decode tails**.

Speculative decoding alters the execution path:
- A smaller draft model proposes multiple tokens ahead
- The full model verifies them **in chunks** rather than generating each independently
- Verification is cheaper than generation
- Multiple tokens can be accepted in a single step

Result: **fewer expensive operations on the critical path** → direct improvement to P90 and P99, exactly where large models tend to fail product SLAs.

### Quality does not have to be traded for speed

A common concern: speculative decoding requires aggressive quantization or sacrifices output quality. **This is not inherently true.**

Speculative decoding does not replace the full model. The full model remains the **final authority**. Draft tokens are verified and rejected if incorrect. The quality risk is concentrated in the draft model, not the main model — allowing the primary model to run in higher precision modes.

### Draft model training as production infrastructure

The effectiveness of speculative decoding depends heavily on the draft model. A generic draft model delivers gains. A draft model **shaped through post-training on synthetic inputs** that resemble real production conversations delivers:
- More consistent acceptance rates
- Tighter tail latency

> *"Treating draft model training as part of the production pipeline, rather than an experiment, is what turns speculative decoding into a reliable architectural primitive."*

### Design it in early

> *"Large MoE inference is not a throughput problem — it is an execution-path problem."*

The question is no longer how to shave milliseconds off average decode speed, but how to **reduce the amount of expensive work on the critical path**. That leads to architectural changes, not parameter tweaks.

Retrofitting speculative decoding late is painful — it touches serving pipelines, batching behavior, memory management, and observability. **Designed in from the beginning**, teams can reason coherently about capacity, headroom, and failure modes.

> *"For long-context, non-streaming systems, speculative decoding is not an optimization — it is a prerequisite for meeting real product SLAs."*

---

## 10. Where It Is Deployed Today

*(Chris Thomas + Google Research)*

- **Google AI Overviews (Google Search):** Speculative decoding is used in production to produce results faster while maintaining response quality. Confirmed by the original authors.
  > *"We have applied speculative decoding in a number of Google products, where we see remarkable speed-ups in inference, while maintaining the same quality of responses."*

- **Google products broadly:** Remains a significant part of Google's inference optimizations even as other techniques are added.

- **LM Studio:** Added speculative decoding support in its latest beta release, making it accessible to local model users.

- **Local coding assistants:** Via `llama.cpp` HTTP server with the `llama-vscode` extension. Developers running a 3B draft + 14B target on mid-range hardware get 2–4× faster code completions without cloud costs or privacy trade-offs.

- **Industrial serving stacks** (Nebius context): Teams working on long-context, non-streaming products are treating speculative decoding as a production pipeline requirement, not an experiment.

---

## 11. Practical Setup Example

*(Chris Thomas — unique to this source)*

Using the `llama-vscode` extension and `llama.cpp` HTTP server locally:

```bash
llama-server.exe \
  -md qwen2.5-coder-3b-q8_0.gguf \
  -m qwen2.5-coder-14b-q8_0.gguf \
  --port 8012 -c 2048 \
  --n-gpu-layers 99 -fa \
  -ub 1024 -b 1024 \
  -dt 0.1 --ctx-size 0 --cache-reuse 256
```

- `-md`: draft model (3B)
- `-m`: main/target model (14B)

The extension handles:
- **Tab** — accept full suggestion
- **Shift+Tab** — accept first line only
- **Ctrl/Cmd+Right** — accept next word

**Real-world benefit:** Developers no longer choose between speed (small model) or quality (large model) — they get both. Local processing improves privacy and reduces cloud costs.

---

## 12. Quick Implementation Sketch (NVIDIA TensorRT)

*(NVIDIA — unique to this source)*

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

Full tutorial: [TensorRT-Model-Optimizer/examples/speculative_decoding](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb)

> **Note:** This uses NVIDIA's `modelopt` library, not SGLang or raw PyTorch. Useful as an existence proof for L1 readers; the Layer 14 book uses SGLang's native speculative decoding configuration instead.

---

## 13. The Limits of Every Analogy

*(Chris Thomas + NVIDIA)*

All three analogies (junior/senior, scientist/assistant, CPU speculative execution) share one important flaw:

**Rejected draft tokens are never visible to the user.** Unlike a junior developer's edits, which you can see and explicitly accept or reject, the draft-target acceptance decision is **internal and automatic**. Users see only the final accepted tokens — the drafting and rejection process is invisible.

The "junior developer" metaphor also breaks when:
- The junior's suggestions are never shown to the user if rejected — unlike real code review
- Acceptance is not about quality judgment — it is about whether the senior would have produced the **identical** token at that position
- The junior can be wrong for any reason (vocabulary, style, context) — not just because the output is "bad"

For L1 readers: the key mental correction is that speculative decoding is a **mechanical process**, not a judgment process. The target model is not "reading" the draft and deciding if it's good — it is running its own forward pass and accepting or rejecting based on probability ratios.

---

## 14. Key Quotes Collected

> *"The key insight isn't just about making things faster — it's about making better use of the computing resources we already have."* — Chris Thomas

> *"Compared with standard autoregressive decoding, which produces one token per pass, this technique lets the system generate multiple tokens at once, cutting latency and boosting throughput without any impact on accuracy."* — NVIDIA

> *"The algorithm speeds up generation from autoregressive models by computing several tokens in parallel, without affecting output quality; in fact, the method guarantees an identical output distribution."* — Leviathan, Kalman, Matias (Google Research)

> *"Producing results faster with the same hardware also means that fewer machines are needed for serving the same amount of traffic, which translates yet again to a reduction in the energy costs of serving the same model."* — Google Research

> *"Throughput numbers hide all of this."* — Nebius

> *"Large MoE inference is not a throughput problem — it is an execution-path problem."* — Nebius

> *"The goal is not faster demos. It is systems whose behavior under stress is understood, measured, and bounded before users ever see them."* — Nebius

---

## Appendix: What Is Left Out and Why

### Left out: `05_google_cloud_five_techniques.md`

**Status:** Referenced in the L1 `README.md` as *"Landscape view: where speculative decoding fits alongside routing, disaggregation, quantization, and prefix caching. Best for readers who want the 'whole picture' quickly."*

**Why absent:** The file `05_google_cloud_five_techniques.md` does not exist in the `references/L1/` directory at the time this COMBINED.md was written. The README lists it as a fifth article but the file was not created. Its described content — positioning speculative decoding in the wider inference optimization landscape alongside routing, quantization, disaggregation, and prefix caching — is **not represented anywhere** in this combined document.

**Source URL:** https://cloud.google.com/blog/topics/developers-practitioners/five-techniques-to-reach-the-efficient-frontier-of-llm-inference

**Recommended action:** Download and add `05_google_cloud_five_techniques.md`, then add a Section 8.5 here covering the "efficient frontier" framing and where speculative decoding fits among the five techniques.

---

### Left out: KV rewind, page management, mirroring invariant

These are explicitly **L3 territory**, not L1. All four L1 sources treat the "draft tokens are rejected" step as a black box — none explain what happens to the KV cache when a rejection occurs, how memory pages are deallocated, or how the draft and target KV caches are kept in sync. See:
- `lesson/04` — KV cache architecture for speculative decoding
- `lesson/05` — the accept/reject loop
- `lesson/06` — KV rewind on rejection

---

### Left out: Specific production speedup numbers from Google

Google Research's blog post (source 03) confirmed deployment in AI Overviews and Google products broadly but did not disclose specific speedup numbers — only that they see *"remarkable speed-ups."* The original paper's 3× on T5-XXL is available; Google's production numbers are not public.

---

### Left out: Batched inference and throughput interaction

The Nebius article focuses on **tail latency** but does not cover the known tradeoff that speculative decoding speedup **shrinks with larger batch sizes** (a key finding in SpecDecode-Bench, L4/03). This is a nuance omitted at L1 intentionally — the batch size interaction is complex enough to belong in L4 treatment. L1 readers can assume the benefits described apply at batch size 1 or very small batches.

---

### Left out: Draft model alignment / training cost

The Nebius article introduces the concept of draft model training as production infrastructure (Section 9 above), but does not cover:
- Sequence-level knowledge distillation (Seq-KD)
- Online KD (real-time alignment during inference)
- How draft models are evaluated for alignment quality

These belong in the L5 training references (TorchSpec, vLLM Speculators library).
