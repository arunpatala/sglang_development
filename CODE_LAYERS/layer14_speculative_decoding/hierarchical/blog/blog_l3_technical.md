# Speculative Decoding: Algorithm, Hardware, and Production Architecture

*For ML engineers implementing, auditing, or extending speculative decoding systems. Assumes familiarity with transformer inference, KV caches, and basic probability.*

---

There is a pattern common to inference optimization: someone claims you can get significantly faster output without any quality loss, which sounds impossible, and turns out to be true but for a specific and understandable reason. Speculative decoding is one of those cases. The speedup is real, the losslessness proof is exact, and neither of those facts makes sense without first understanding what GPUs are actually doing — or not doing — during decode.

This post starts from hardware first principles, derives why the speedup exists before we even talk about an algorithm, walks through the algorithm and its correctness guarantee, and then follows it into production: advanced drafting mechanisms, KV cache management, configuration, and what to measure.

---

## 1. Why the speedup isn't magic: the hardware picture first

Before the algorithm, the hardware. Because if you don't understand the hardware situation, speculative decoding looks like a free lunch, and free lunches make engineers nervous. It isn't free — it just draws from a budget that was previously being wasted.

The roofline model characterizes any GPU workload by its **arithmetic intensity**: the ratio of floating-point operations performed per byte of data read from memory.

- If a workload has high arithmetic intensity, it's **compute-bound**: throughput scales with FLOPS.
- If it has low arithmetic intensity, it's **memory-bandwidth-bound**: throughput scales with HBM bandwidth, and adding more FLOPS does nothing.

Where does single-token transformer decode sit? A 70B parameter model in fp16 contains roughly 140 GB of weights. Generating one token requires reading most of those weights once — that's ~140 GB of HBM reads. The arithmetic performed for one decode step on one token is roughly `2 × 70B × 1 (batch size) ≈ 140 GFLOP`. Arithmetic intensity: `140 GFLOP / 140 GB = 1 FLOP/byte`.

An H100 SXM5's compute-to-bandwidth ratio is `989 TFLOPS / 3.35 TB/s ≈ 295 FLOP/byte`. Decode sits at **1 FLOP/byte against a roofline of 295 FLOP/byte** — less than 0.4% of theoretical compute utilization.

The CUDA cores are idle. The GPU is waiting for HBM reads. Every decode step, roughly 99.6% of available arithmetic throughput goes unused.

This is not a pathology — it's structural. The transformer architecture was designed for training (batched, compute-heavy) and the inference regime is fundamentally different. It means that during decode, you can schedule additional arithmetic on top of the HBM reads that are already happening, and as long as that arithmetic doesn't extend the critical path, it is essentially free.

**Speculative decoding is specifically a mechanism for using that idle compute.** A small draft model's forward passes run concurrently with the target model's memory reads. The draft is 10–100× smaller than the target, so its memory traffic is proportionally smaller and can be overlapped. You get additional tokens proposed at negligible additional cost — because the cost was being paid already.

Now, with that framing, let's look at the algorithm.

---

## 2. The algorithm: what's actually happening

### Notation

- `p(x | context)` — target model's token probability distribution
- `q(x | context)` — draft model's token probability distribution
- `γ` — draft length (number of tokens proposed per step)
- `x̃₁, x̃₂, ..., x̃_γ` — draft token sequence
- `α` — acceptance rate (expected fraction of draft tokens accepted per step)

### Step 1: Draft generation (cheap, serial, fast)

The draft model runs autoregressively for `γ` steps, starting from the current context `x₁..xₙ`:

```
x̃₁ ~ q(· | x₁..xₙ)
x̃₂ ~ q(· | x₁..xₙ, x̃₁)
...
x̃_γ ~ q(· | x₁..xₙ, x̃₁..x̃_{γ-1})
```

Each draft token is sampled from `q`, and the resulting distributions are stored. `γ` is typically 3–8 for general workloads.

Because the draft model is small, this runs fast — a 7B draft completes each of these steps at roughly 5–10% of the cost of a 70B target forward pass.

### Step 2: Parallel target verification (one pass, many positions)

Now the key operation. The target model runs a **single forward pass** over the full sequence `[x₁..xₙ, x̃₁, x̃₂, ..., x̃_γ]`.

The transformer is parallel across the sequence dimension. The KV cache covers the input prefix `x₁..xₙ` — those don't need to be recomputed. So this forward pass has the cost of processing `γ` new positions, which is nearly the same as processing one. The output: `γ+1` distributions from the target model — one per draft position, plus one bonus distribution at position `γ+1`.

This is the efficiency core: one expensive target forward pass yields `γ+1` probability assessments simultaneously, rather than requiring `γ+1` sequential passes.

### Step 3: Rejection sampling, position by position

The target now has both its own probability `p(x̃ᵢ | contextᵢ)` and the draft's probability `q(x̃ᵢ | contextᵢ)` for each position. The accept/reject decision is:

```
u ~ Uniform(0, 1)

if u < p(x̃ᵢ | contextᵢ) / q(x̃ᵢ | contextᵢ):
    accept x̃ᵢ                     # commit; move to next position
else:
    sample from corrected distribution:
        p'(x) = norm( max(0, p(x) - q(x)) )
    commit that sample; stop
```

If all `γ` tokens are accepted, sample one bonus token from the target's distribution at position `γ+1`.

**Why the corrected distribution `p'`?** The draft was overconfident at the rejected position — it assigned probability `q(x̃ᵢ)` higher than the target would have. Simply sampling from `p` at that point would be wrong — you'd be double-counting probability mass already "paid" by the accept/reject rule. The correction `max(0, p(x) - q(x))` zeroes out the mass where the draft was overconfident and redistributes only the residual. This is precisely what makes the marginal output distribution match `p` exactly.

### Correctness: why the output distribution is identical to the target's

The core theorem (Leviathan et al., 2022): for any draft model `q` and target model `p`, the rejection sampling accept/reject rule with corrected distribution `p'` produces an output sequence whose marginal distribution is identical to sampling from `p` alone.

> *"The algorithm guarantees an identical output distribution to standard sampling from the target model."*

This isn't "approximately correct" or "correct in expectation." The marginal distribution at every position is exactly `p`. Speculative decoding is lossless by construction.

### The greedy special case

At temperature=0 (argmax decoding), the probability ratio simplifies: the draft token is accepted if and only if the target's argmax matches the draft's chosen token:

```
accept x̃ᵢ  iff  argmax p(· | contextᵢ) == x̃ᵢ
```

No random draw, no distribution storage needed. This is the variant most commonly deployed in production serving frameworks (vLLM, SGLang, llama.cpp) because it's deterministic, fast, and avoids storing the full draft distributions during the speculation step.

---

## 3. How much faster, and why the numbers vary

The algorithm gives us the mechanism; now let's derive the expected speedup so we know what we're actually buying.

Expected tokens committed per target forward pass, assuming per-position acceptance probability `α` (constant for simplicity):

```
E[tokens_committed] = Σᵢ₌₁^γ αⁱ + 1  =  (1 − α^{γ+1}) / (1 − α)
```

At `α = 0.85`, `γ = 5`:

```
E[tokens_committed] = (1 − 0.85⁶) / (1 − 0.85) = (1 − 0.377) / 0.15 ≈ 4.15
```

But this is tokens committed per target pass, not wall-clock speedup. We also pay for the draft model's `γ` forward passes. If the draft costs `c` fraction of the target per token:

```
Effective speedup ≈ E[tokens_committed] / (1 + γ·c)
```

At `α = 0.85`, `γ = 5`, `c = 0.05` (draft is 5% of target cost per token):

```
Speedup ≈ 4.15 / (1 + 5 × 0.05) = 4.15 / 1.25 ≈ 3.3×
```

This is why draft model selection matters: `c` needs to be small. A draft that's 20% of the target's cost and achieves the same `α` yields:

```
Speedup ≈ 4.15 / (1 + 5 × 0.20) = 4.15 / 2.0 ≈ 2.1×
```

Same acceptance rate, half the gain — because the draft was too expensive.

### The batch size caveat

Everything above assumes small batch sizes. At larger batches, the target model's decode step becomes more compute-efficient: batch parallelism amortizes HBM reads across multiple sequences, pushing arithmetic intensity toward the compute-bound regime. As batch size grows, the memory-bandwidth headroom that speculative decoding exploits begins to close.

At batch size 64+, speculative decoding speedup can approach 1× — no gain, but also no regression, because the worst case (all drafts rejected) is still just baseline. The regime where speculative decoding provides maximum benefit is batch size 1–8, which is exactly the regime of interactive latency-sensitive products.

---

## 4. Advanced drafting: what production systems actually use

The draft-target baseline works well, but it introduces operational complexity (maintaining two model checkpoints, managing tokenizer alignment, capacity planning for two weight sets). This has driven a second generation of drafting mechanisms that either attach to the target model or bake the drafting directly into it.

### EAGLE-3: feature-level drafting, no separate model

The fundamental observation behind EAGLE: the target model's hidden states already carry rich information about what the next token should be. A small prediction head that reads those hidden states can draft next tokens more accurately than a separate small model, and without any of the tokenizer or distribution-mismatch problems.

The EAGLE head architecture:

```
EAGLE head = TransformerDecoderLayer(d_model) + Linear(d_model, vocab_size)

Input to head (EAGLE-3 multi-layer fusion):
  - Low-level embeddings from target layer ℓ₁   (syntactic features)
  - Mid-level hidden states from target layer ℓ₂  (semantic features)
  - High-level hidden states from target layer ℓ₃  (distribution-adjacent features)
  → concatenated and projected to d_model → fed into the EAGLE head
```

Using three layers rather than one (the EAGLE-3 improvement) matters because: early layers encode syntax and token identity, middle layers encode semantics, and late layers are closest to the output distribution. Feeding all three gives the head more signal and yields higher acceptance rates.

EAGLE-3 adds three further refinements:

**Dynamic draft tree.** Instead of proposing a fixed linear sequence of `γ` tokens, EAGLE-3 builds a tree of candidate continuations using beam search over cumulative log probabilities. A node is expanded if its score exceeds a threshold; the tree adapts its shape to the input rather than being fixed at configuration time.

**Instance-adaptive stopping.** The head tracks its own cumulative confidence as it builds the tree and halts when confidence drops below a learned threshold. This avoids committing draft computation to positions likely to be rejected — reducing draft overhead on difficult inputs.

**Tree attention in verification.** The target verifies the entire candidate tree in one forward pass. The attention mask is structured so that each tree node can attend to the input prefix plus its own ancestors, but not to sibling nodes or other branches. One forward pass → verification of the entire tree.

Training: the EAGLE head is trained with the target model frozen, minimizing KL divergence between the head's predicted distribution and the target's true distribution at each position. Since the head sees the same hidden states the target produces at inference time, the training signal is directly relevant.

### Multi-Token Prediction (MTP)

If EAGLE is a post-hoc attachment, MTP is the alternative: train the drafting capacity into the model from the start.

MTP adds `K` dedicated prediction heads to the transformer, each responsible for predicting the token at a specific future offset:

```
Head k predicts: P(x_{t+k} | x₁..xₜ)   for k = 1, 2, ..., K
```

During inference, these heads run in parallel with the main output head at each position `t`, drafting `K` tokens ahead simultaneously. Verification proceeds left-to-right: the main model's own forward pass (run on the next actual input position) produces the ground truth at each offset; the longest prefix where the head's top-1 matches the main model's top-1 is committed.

The architectural difference from EAGLE is fundamental. EAGLE is a separate artifact — it can be added to or removed from any target model. MTP heads are part of the base model checkpoint; they require the model to be trained with MTP in the objective. DeepSeek-R1 trains with MTP as a standard component, meaning draft capacity is available at inference time with no additional artifact to manage.

### Tree attention (general mechanism)

Both EAGLE-3 and any beam-search-based drafter produce tree-structured candidate sets rather than linear sequences. Tree attention is the general verification technique that makes single-pass tree verification efficient.

For a draft tree with depth `d` and branching factor `b`, there are `(b^{d+1} - 1)/(b - 1)` candidate nodes. The target builds an attention mask where node `i` attends to the input prefix and its ancestors. Non-ancestor positions are masked to zero. The full tree is verified in one target forward pass.

With branching factor 2–4, you verify `2^d - 1` or more candidates per target pass, dramatically increasing the probability of finding a long accepted prefix — because the tree's branches cover diverse continuations rather than committing to one linear path.

---

## 5. Production: where the implementation complexity actually lives

Understanding the algorithm and the variants gets you to a working single-request implementation. Getting to production means solving the engineering problems that don't appear in papers.

### KV cache management: the hidden cost

During each speculative step, two KV cache regions coexist:

- **Target KV cache** — the authoritative, committed cache for the prefix `x₁..xₙ`. This is the standard KV cache every serving framework manages.
- **Draft-extension region** — tentative KV entries computed during the draft model's forward passes, representing the speculated prefix `x̃₁..x̃_γ`. These entries are not yet committed.

On **acceptance**, the draft-extension entries are promoted into the target KV cache — they become part of the committed prefix. On **rejection**, the entries from the first rejected position onward must be discarded: a KV rewind that restores the cache to the state at the last accepted position.

For tree-structured drafts, each tree node has its own KV extension. The serving framework must track which nodes are in the active candidate set, and free rejected branches atomically when the target's verification pass returns. This requires either per-node KV page allocation (expensive but precise) or conservative over-allocation with deferred release.

**Memory budget implications:** A 70B target (≈140 GB fp16) plus a 7B draft (≈14 GB fp16) requires ≈154 GB of model weight memory before KV cache. On an 8×H100 node with 640 GB HBM, this leaves roughly 486 GB for KV cache across all concurrent requests — tighter than running the target alone, and the draft's KV cache (smaller but present) reduces available headroom further. Capacity planning must account for both.

### Serving framework configuration

**vLLM:**

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="Qwen/Qwen2.5-72B-Instruct",
    speculative_model="Qwen/Qwen2.5-7B-Instruct",
    num_speculative_tokens=5,
    speculative_draft_tensor_parallel_size=1,  # draft on fewer GPUs
)

sampling_params = SamplingParams(temperature=0.0, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

`speculative_draft_tensor_parallel_size=1` keeps the draft model on a single GPU if it fits — the draft is small enough that full tensor parallelism adds communication overhead without benefit.

**SGLang (server mode):**

```bash
python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-72B-Instruct \
  --draft-model-path Qwen/Qwen2.5-7B-Instruct \
  --num-draft-tokens 5 \
  --tensor-parallel-size 4
```

**NVIDIA TensorRT-LLM — EAGLE-3:**

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

`mtsp.convert()` modifies the model in-place: it attaches the EAGLE-3 head, hooks into the target's forward pass at the three hidden-state extraction points, and rewires the generation loop to emit a draft tree before each target verification pass. The resulting checkpoint contains base model weights plus the EAGLE head as a named submodule.

### Observability: acceptance rate is the primary health signal

Standard latency and throughput metrics don't tell you whether speculative decoding is doing anything useful. You need to instrument the speculation layer separately:

```
acceptance_rate      = tokens_accepted / tokens_drafted          # primary health signal
committed_per_step   = tokens_committed / target_passes          # effective parallelism
draft_overhead       = draft_time / total_step_time              # cost of drafting
speculative_efficiency = realized_speedup / theoretical_max_speedup
```

`acceptance_rate` is the one to watch. A healthy system in production sits above 0.75 for general conversational workloads. A drop to 0.50 or below is a signal that the draft and target have diverged — most commonly caused by a target model update that wasn't propagated to the draft model, or by a shift in traffic distribution that the draft wasn't trained to handle.

`draft_overhead` catches the opposite failure: a draft model that's too expensive. If draft time accounts for more than 20–25% of total step time, the draft is too large relative to the target, and `c` in the speedup formula has grown to the point where the efficiency gain is being consumed.

### Draft model training: why this is infrastructure, not a one-time experiment

A draft model pulled from a public model hub and paired with a target achieves moderate acceptance rates — typically 0.65–0.78 for conversational workloads. A draft model fine-tuned specifically to track the target achieves 0.80–0.90, with meaningfully tighter P99 latency as a result.

The training procedure is sequence-level knowledge distillation:

1. Sample inputs from production request logs (or a representative synthetic distribution).
2. Run the target model on each input to produce token-level probability distributions.
3. Fine-tune the draft model to minimize KL divergence against the target's distributions at each position.
4. Evaluate acceptance rate on a held-out slice of production traffic.
5. Deploy only if acceptance rate improves over the previous draft checkpoint.

The important operational lesson: acceptance rates degrade over time. As traffic distribution shifts — new prompt patterns, changed instruction formats, domain drift — a draft model trained once on historical data increasingly diverges from the target on live traffic. Teams that treat draft training as a continuous post-training pipeline maintain consistently high acceptance rates. Teams that treat it as a one-time setup find gains eroding without obvious cause.

---

## 6. What is provably preserved — and what is not

Being precise here matters for production commitments.

**What the algorithm guarantees:**

The output distribution at every position is identical to sampling from the target model alone. This holds for both temperature=0 (greedy) and all sampling temperatures, given the corrected rejection rule. The worst case — all drafts rejected — delivers exactly one target-model token per forward pass, identical to baseline autoregressive decoding. There is no regime where speculative decoding produces output worse than baseline.

**Where the guarantee doesn't extend:**

*Reproducibility at sampling temperatures > 0.* The accept/reject rule draws `u ~ Uniform(0, 1)` at each position. These draws are hardware-nondeterministic across runs, GPU types, and CUDA versions. The marginal distribution is correct, but exact token sequences will differ between runs unless temperature=0. If your application requires reproducible outputs, use greedy decoding — it is fully deterministic.

*Speedup at high batch sizes.* The roofline picture from Section 1 changes under batching. At batch size 64+, the target's decode step becomes meaningfully compute-bound, the memory-bandwidth headroom closes, and draft model overhead becomes proportionally more expensive. Speculative decoding speedup at large batch sizes can be 1.0–1.2× — useful for latency but not transformative. If you're operating at sustained high batch sizes, other optimizations (continuous batching tuning, quantization) may yield better returns.

*Memory headroom.* Two model weight sets must fit in the GPU memory budget alongside KV cache for all concurrent requests. For 405B+ target models, this requires careful topology decisions: draft on separate GPUs, pipeline parallelism for the target, dedicated KV cache partitioning. The memory budget is not automatic — it must be planned.

---

## Summary: the decisions that matter

Speculative decoding converts a structural inefficiency — idle CUDA cores during memory-bandwidth-bound decode — into committed token throughput, with a proof that the output distribution is exactly preserved. The mechanism is rejection sampling. The governing variable is acceptance rate `α`. The production value is concentrated in tail latency (P90, P99) at small batch sizes, exactly where interactive products are most sensitive.

The decisions that determine whether you get 3× or 1.5× in practice:

1. **Draft model alignment.** Same tokenizer, same model family, trained specifically to track the target's distribution on your traffic. Not an off-the-shelf download.

2. **Draft length γ.** 4–8 for general workloads. Tune empirically per task — predictable tasks (code completion, translation) tolerate longer drafts; open-ended generation needs shorter ones.

3. **Drafting mechanism.** Linear sequence (simplest, lowest overhead), tree attention (higher throughput per pass), EAGLE-3 (no separate model, best acceptance rates for models that support it).

4. **Acceptance rate as a production metric.** It should be visible in your dashboards. Drops are early warnings of model version drift. Floors (< 0.60) signal drafting mechanism selection or training problems.

5. **Batch size operating regime.** Speculative decoding pays off at batch sizes 1–16. If you're planning for high-concurrency batch serving, account for diminishing returns and size capacity accordingly.

---

*Sources: [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (2022)](https://arxiv.org/abs/2211.17192), [Google Research Blog (Dec 2024)](https://research.google/blog/looking-back-at-speculative-decoding/), [NVIDIA Developer Blog (Sep 2025)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/), [Nebius Blog](https://nebius.com/blog/posts/moe-spec-decoding), [TensorRT-Model-Optimizer EAGLE-3 example](https://github.com/NVIDIA/TensorRT-Model-Optimizer/blob/main/examples/speculative_decoding/example.ipynb)*
