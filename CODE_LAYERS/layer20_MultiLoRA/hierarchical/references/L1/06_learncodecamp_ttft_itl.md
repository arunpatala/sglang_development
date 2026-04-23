# Understanding LLM Inference Basics: Prefill and Decode, TTFT, and ITL

**Source:** https://learncodecamp.net/llm-inference-basics-prefill-decode-ttft-itl/
**Author:** LearnCodeCamp
**Level:** L1 (prerequisite) — TTFT/ITL definitions; compute-bound vs memory-bound; chunked prefill intro
**Why here:** The best pure prerequisite for PD disaggregation. Explains TTFT and ITL as user-facing metrics, defines why prefill is compute-bound and decode is memory-bound using arithmetic intensity, and introduces the KV cache role without assuming prior knowledge. Also covers the scheduler queuing insight (decode queue prioritised over prefill queue) and chunked prefill as the non-disaggregation mitigation. Read this before any other reference in Layer 19.

---

## The Two Phases of LLM Inference

### 1. Prefill (Prompt Processing)

When you send a prompt to an LLM, the model processes the **entire input sequence all at once**.

- The transformer processes every token in the prompt **in parallel**.
- For each layer, it computes attention over the full prompt and generates the **KV cache** — a stored representation of the prompt's keys and values for reuse during generation.
- No output tokens are generated yet; the model is building the internal state needed for generation.

### 2. Decode (Token Generation)

After prefill, the model generates output tokens **one by one** (autoregressive / sequential).

**Each decode step:**
1. Predict the next token.
2. Append that token to the sequence.
3. Compute new key and value for this token and add to the KV cache.
4. Recompute attention using all cached KV states (prompt + all generated tokens so far) plus the new ones.
5. Repeat until desired length or stop condition.

Because each new token depends on all previous ones, **decoding cannot be parallelised across output tokens** — you must generate them sequentially.

**Why the KV cache matters here:** Without it, every decode step would require re-computing attention over the entire growing sequence from scratch (O(n²) cost). With the cache, each decode step only needs to compute new keys/values and attend to the cached past — roughly O(n) total cost across the full generation.

---

## Key Performance Metrics

### Time to First Token (TTFT)

TTFT measures the delay from **when the request is submitted** until the **very first output token** appears in the response stream.

- TTFT ≈ prefill time + time to compute the first decode step.
- Long prompts → higher TTFT (prefill processes everything upfront).
- **Critical for perceived responsiveness**: users notice if the model "hangs" before starting to reply.

### Inter-Token Latency (ITL)

ITL (also called TPOT: Time Per Output Token) is the **average time between consecutive output tokens** once generation has started.

- ITL is dominated by the decode phase.
- Lower ITL = text streams faster = feels snappier.
- **Typical values on high-end hardware:** 20–100ms per token → 10–50 tokens per second.

### Together

| Metric | What it captures | Dominated by |
|---|---|---|
| TTFT | Time until model starts responding | Prefill |
| ITL / TPOT | How fast tokens stream | Decode |

- Low TTFT → quick start (model begins responding fast).
- Low ITL → fast streaming completion (text appears smoothly).

---

## Why the Split Matters: Asymmetric Workloads

**Prefill is compute-bound (FLOPs-bound):**
- Dominant operations: large matrix multiplications (Q @ Kᵀ and Attention @ V) over the full prompt length.
- Requires massive arithmetic operations — far more computation than data movement.
- GPU tensor cores are the bottleneck — the hardware is "embarrassingly parallel" here.
- GPU operates near peak FLOPs utilisation.

**Decode is memory-bound:**
- Each decode step generates only one token → matrix multiplications are tiny (shape [1 × d] against the cached KV of length n).
- Little arithmetic per step, but must load the **entire KV cache** from GPU memory for every step.
- Bottleneck: **moving large amounts of data with minimal work per byte** (memory bandwidth).
- Adding more compute capacity doesn't help; memory bandwidth is the constraint.

### Workload Asymmetry Examples

| Workload | TTFT | Total time | Bottleneck |
|---|---|---|---|
| Long prompt + short output | High (expensive prefill) | Prefill-dominated | Compute during prefill |
| Short prompt + long output | Low (fast prefill) | Decode-dominated | Memory BW during decode |
| Many concurrent users | Variable | ITL-sensitive | Decode pool saturation |

---

## Scheduler Behaviour in vLLM (Inference Engine Insight)

vLLM exploits the prefill/decode distinction in its scheduler:

- **Two queues**: waiting queue (new requests needing prefill) and running queue (ongoing decode).
- **Decode queue prioritised**: the scheduler prioritises the running (decode) queue over the waiting (prefill) queue. This prevents long compute-heavy prefills from starving latency-sensitive decode steps — which would otherwise cause visible stuttering in streaming responses.
- **Chunked prefill** (iterative prefill): instead of processing an entire long prompt in one massive batch slot, vLLM breaks it into smaller chunks. This caps the compute a single prefill can consume in one iteration, allowing decode steps from other requests to interleave and progress without long delays.

**Why simply adding more GPUs doesn't always help:** Decode-heavy workloads are often limited by **memory bandwidth**, not raw compute power. Adding GPUs with the same memory bandwidth spec doesn't proportionally improve decode throughput.

---

## Connection to PD Disaggregation

This article establishes the prerequisites for understanding why disaggregation is needed:

1. **TTFT and ITL are separate user-facing metrics** that correspond to separate hardware bottlenecks — prefill (compute) and decode (memory bandwidth).
2. **Scheduling tricks** (decode queue priority, chunked prefill) reduce but don't eliminate the tension between the two phases on the same GPU.
3. **The next step** — full disaggregation onto separate hardware pools — is motivated by these insights.

After reading this article, the reader is ready to understand:
- Why DistServe defines goodput in terms of both TTFT SLO and TPOT SLO simultaneously.
- Why the prefill pool can use a different TP configuration from the decode pool.
- Why `--chunked-prefill-size` is used on the prefill server in SGLang's disaggregated mode.

---

## Key Takeaways for Layer 19

- **Read this article first** — it ensures the reader understands TTFT, ITL/TPOT, and the compute-bound vs memory-bound distinction before encountering DistServe's goodput definition.
- TTFT ≈ prefill time + first decode step; ITL ≈ one decode step. These are not the same; disaggregation optimises them independently.
- Decode is always memory-bound regardless of batch size — the only way to raise decode GPU utilisation is to increase batch size (more concurrent requests amortising the HBM reads).
- Chunked prefill is the single-GPU mitigation; disaggregation is the multi-GPU solution. They are complementary, not competing.
- The KV cache is the bridge between the two phases — its existence and size are why disaggregation requires a network transfer step.
