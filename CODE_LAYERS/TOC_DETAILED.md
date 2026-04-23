# Building an Inference Engine from Scratch
### Detailed Table of Contents

---

## 1. "What does an inference engine look like?"

*Goal: By the end of this chapter, readers have a running, measurable, OpenAI-compatible server — HuggingFace model under the hood. This is the v0.0 baseline everything else is compared against.*

### 1.1 Training vs Inference
- **The fundamental difference** — Training updates weights; inference runs them. Different compute profiles, different bottlenecks, different engineering problems.
- **What an inference engine is** — A program that takes a prompt and returns generated text as fast as possible while serving many users.

### 1.2 The Black Box: `inferencecode.py`
- **Run it first, read it second** — `00_start.py` is 35 lines. Run it, see output, then read what each line does.
- **Three things that happen** — `tokenizer.encode()` → `model.generate()` → `tokenizer.decode()`. Each is a black box we will open in subsequent chapters.
- **Why the output changes every run** — Sampling is random; temperature, top-k, top-p introduced informally here.

### 1.3 Loading the Model
- **`AutoModelForCausalLM.from_pretrained()`** — What downloads, what loads, how long it takes.
- **Inspecting the model** — Parameter count, layer names, device placement, memory footprint.
- **The tokenizer** — What it does, why it's separate, a quick encode/decode smoke test.
- **`config.py`** — Single place to change `MODEL_NAME`, `PROMPT`, `SERVER_URL`; swap models across all files in one edit.

### 1.4 Sampling: Seeing the Knobs
- **Greedy vs stochastic** — Greedy is deterministic; run it twice, same output. Sampling is not.
- **Temperature** — Low: focused and repetitive. High: creative and sometimes incoherent.
- **Top-k and top-p** — Restricting the candidate pool by count vs by cumulative probability mass.
- **Behavioral demo** — `02_sampling.py` runs all combinations; observe outputs before any math.

### 1.5 A Minimal FastAPI Server
- **Why serve over HTTP** — Decouples the engine from the client; every benchmark tool speaks HTTP.
- **The `/v1/chat/completions` endpoint** — Receive a prompt, call `model.generate()`, return result. No streaming, no JSON schema complexity — just the request/response loop.
- **`03_server.sh`** — One command to start the server.
- **Testing with curl** — A one-liner to verify the server works end to end.
- **What is missing** — Explicitly naming what this server cannot do: concurrent requests, our own generate loop, KV cache, batching. These become the chapter titles that follow.

### 1.6 Measuring the Baseline
- **Time to First Token (TTFT)** — Measured server-side: time to generate 1 token = pure prefill latency.
- **Time Per Output Token (TPOT)** — Derived from full-run server-side elapsed minus TTFT, divided by (n_tokens - 1).
- **Tokens per second** — Overall throughput; the number tracked across every chapter.
- **HTTP-only benchmark** — `04_benchmark.py` hits the server and uses `_elapsed_s` from the response for both TTFT and full-run timing — same clock, no HTTP overhead mixed in. This is the yardstick for the rest of the book.

---

## 2. "How does generation actually work?"

*Goal: replace `model.generate()` with our own loop. Swap it into Chapter 1's server in one line. Confirm the benchmark numbers are identical — our loop is correct. The model is still HuggingFace's; Chapter 3 opens it.*

### 2.1 What Inference Actually Is
- **Autoregression** — Why the model generates one token at a time and feeds it back as input.
- **The generate loop** — The three-line core: forward pass → sample next token → append to sequence → repeat.

### 2.2 Prefill vs Decode: The Core Asymmetry
- **Prefill** — Processing the entire prompt in one forward pass; a large matrix multiply, compute-bound, fast per token.
- **Decode** — Generating one token at a time; a thin matrix-vector operation, memory-bandwidth-bound, slow per token.
- **The numbers** — Concrete timing: prefill for a 512-token prompt vs 512 individual decode steps; why decode dominates total latency.
- **Why this asymmetry is everything** — Every optimization in this book — KV cache, continuous batching, speculative decoding, flash attention — exists because of this gap. This is the insight the rest of the book is built on.

### 2.3 Sampling
- **Greedy decoding** — Always pick the highest probability token; deterministic but often boring.
- **Temperature** — Scaling logits before softmax to control sharpness of the distribution.
- **Top-k sampling** — Restricting to the k most likely tokens before sampling.
- **Top-p (nucleus) sampling** — Restricting to the smallest set of tokens whose cumulative probability exceeds p.
- **Putting it together** — A `sample()` function that takes logits and `SamplingParams` and returns the next token ID.

### 2.4 The Generate Loop
- **Tokenizing the prompt** — Converting the chat message to token IDs using the chat template.
- **The prefill step** — Running the full prompt through the HF model to get the first token's logits.
- **The decode loop** — Feeding one token at a time, collecting output tokens until EOS or max length.
- **Detokenizing** — Converting output token IDs back to a string, handling special tokens cleanly.
- **A complete `generate()` function** — End-to-end: string in, string out, ~50 lines of PyTorch.
- **Why this loop is deliberately naive** — The full prompt re-processed at every decode step; the redundancy that KV cache (Ch4) will fix. The FLOPs math: ~140x redundancy quantified explicitly.

### 2.5 The Swap: One Line, Before and After
- **Run the Chapter 1 benchmark first** — Record TTFT, TPOT, throughput against the server using `model.generate()`.
- **The one-line change** — In Chapter 1's server, replace `model.generate(...)` with our `generate(model, tokenizer, ...)`.
- **Run the benchmark again** — Numbers must be essentially identical; any divergence means our loop is wrong.
- **The payoff** — Side-by-side benchmark table: HF generate vs our generate. Same numbers = our loop is correct and we now own every line of the generation path.
- **What is still a black box** — The model itself. `model(input_ids).logits` is still HuggingFace. Chapter 3 opens that.

---

## 3. "What is the model doing?"

*Goal: By the end of this chapter, readers have a model built entirely from scratch in PyTorch — every layer hand-written, real weights loaded, logits verified against HuggingFace. The from-scratch model is then swapped into Chapter 2's generate loop — v0.1 is now fully ours. We get there in three zoom levels: black box → structure visible → every line understood.*

---

### Level 1 — The Black Box (start here)

### 3.1 Forward Pass and Reference Logits
- **Load pretrained model with HuggingFace** — `AutoModelForCausalLM.from_pretrained()`; the model as a black box that accepts tokens and returns logits.
- **Tokenize and run a forward pass** — Encode a prompt, call `model(input_ids)`, inspect the logits tensor.
- **What are logits?** — A score for every token in the vocabulary; the raw output we will learn to produce ourselves.
- **Verify and save the reference logits** — Store these numbers; everything we build in this chapter must reproduce them exactly.
- **Level 1 runnable artifact** — The complete Level 1 script: load, tokenize, forward pass, save logits. Readers can run this immediately before reading further.

### 3.2 Tokenization
- **What a tokenizer does** — Converts raw text to token IDs and back; the contract between the user and the model.
- **Byte-Pair Encoding (BPE)** — How the vocabulary is built and why subword tokenization matters.
- **Using a pretrained tokenizer** — Loading the tokenizer via HuggingFace; encode, decode, special tokens, chat template.
- **Tokenizer quirks that affect inference** — Padding, truncation, EOS/BOS, and why they matter for serving.

---

### Level 2 — The Structure (our model shell, HF internals)

### 3.3 Reading the Architecture
- **`config.json` as a blueprint** — Reading `hidden_size`, `num_layers`, `num_heads`, `num_kv_heads`, `intermediate_size`; understanding what each number controls.
- **The HuggingFace `modeling_llama.py` source** — A guided tour of the real implementation: where each layer lives, how they are wired together.

### 3.4 Our Model Shell
- **Sketching the class** — A `TinyLlamaModel` with the right structure: embedding, N decoder layers, final norm, LM head — each slot delegating to HuggingFace's own layer classes for now.
- **Loading weights into our shell** — Using `load_state_dict` with HF weights; confirming our shell produces identical logits to Level 1.
- **Introducing `check_logits()`** — A reusable helper: takes our model and the HF reference model, runs both on the same prompt, asserts `torch.allclose` within tolerance. Introduced once here, used after every swap in Level 3.
- **What to do when it fails** — Common failure modes: wrong layer order, missing transpose, dtype mismatch. How to bisect — isolate which layer's output first diverges.
- **Checkpoint 1** — Run `check_logits()`: our model logits must match the Level 1 reference to 4 decimal places. If they do, the structure is correct.
- **Level 2 runnable artifact** — The complete Level 2 script: our model shell + weight loading + `check_logits()` call. ~60 lines, self-contained.

---

### Level 3 — Every Line (replace each piece from scratch)

### 3.5 Embeddings
- **Token embeddings** — Mapping token IDs to vectors; the lookup table that is the first layer of every transformer.
- **Why there is no positional embedding here** — Models like Llama/Qwen use RoPE instead; setting up the intuition for section 3.7.
- **Replace and verify** — Swap HF embedding with our own; logits must still match.

### 3.6 RMSNorm
- **Why normalization is needed** — Training stability and inference consistency.
- **RMSNorm vs LayerNorm** — What is dropped, what is kept, and why modern LLMs prefer it.
- **Implementation** — A clean 10-line PyTorch version with learnable scale parameter.
- **Replace and verify** — Swap all RMSNorm layers; logits must still match.

### 3.7 Rotary Position Embedding (RoPE)
- **The problem with absolute positions** — Why learned position embeddings don't generalize.
- **The RoPE idea** — Rotating query and key vectors so relative position is encoded in the dot product.
- **Implementation** — Building the rotation matrix, applying it to Q and K, handling different sequence lengths.
- **Replace and verify** — Swap RoPE inside the attention block; logits must still match.

### 3.8 Attention
- **Scaled dot-product attention** — The core operation: Q, K, V projections, scaled dot product, softmax, output projection.
- **Multi-Head Attention (MHA)** — Splitting into heads, running attention in parallel, concatenating outputs. (TinyLLM uses MHA.)
- **Grouped Query Attention (GQA)** — How Qwen reduces KV heads; why this matters for memory at inference time. (Added when validating against Qwen.)
- **The causal mask** — Why we mask future tokens and how to implement it correctly.
- **Implementation** — A single `Attention` class in PyTorch that handles both MHA and GQA via config.
- **Replace and verify** — Swap attention in all layers; logits must still match.

### 3.9 MLP / Feed-Forward Block
- **The role of the FFN** — What the MLP adds that attention alone cannot.
- **SwiGLU / Gated MLP** — How the FFN works: two parallel projections, a gating nonlinearity, and a down projection.
- **Implementation** — A `GatedMLP` class matching `gate_proj`, `up_proj`, `down_proj` structure.
- **Replace and verify** — Swap MLP in all layers; logits must still match.

### 3.10 The Decoder Layer, Full Model, and Handoff
- **Pre-norm vs post-norm** — Why modern LLMs use RMSNorm before (not after) each sub-block.
- **Residual connections** — Why they exist and what breaks without them.
- **Assembling one layer** — Wiring RMSNorm → Attention → residual → RMSNorm → MLP → residual into a single `DecoderLayer`.
- **Stacking layers and the LM head** — How depth is just repetition; final RMSNorm and linear projection to vocabulary size.
- **Checkpoint 2** — Run `check_logits()`: fully from-scratch model with real weights loaded; logits must match Level 1 to 4 decimal places. Every line is now ours.
- **Validation across models** — Same code against SmolLM2-135M → passes. Add GQA → run against Qwen2.5-0.5B → Checkpoint 3.
- **Swap into Chapter 2's generate loop** — Replace HF's `AutoModelForCausalLM` with our from-scratch model inside `generate()`. Run `test_v01.py` → all tests pass. Tag `v0.1`.
- **Level 3 runnable artifact** — The complete from-scratch model as a single self-contained file: every layer, weight loader, and `check_logits()` included.

---

## 4. "Why does decode get slower?"

*Goal: Add KV cache to OurAttention. The generate loop now passes one token at a time through attention, reusing cached K,V tensors instead of recomputing them. TPOT drops and stays flat regardless of sequence length. This is the first real performance gain over the Ch3 baseline.*

### 4.1 The Problem: Decode Cost Grows with Length
- **Timing each decode step** — Run Ch3's generate loop, measure time per step. Step 1 = X ms, step 50 = 5X ms. Plot it.
- **Why it grows** — Every decode step recomputes K and V for all previous tokens. The attention matrix grows as O(N²).
- **Quantifying the waste** — At 100 tokens, 99% of the K/V computation is redundant. The FLOPs math made explicit.

### 4.2 The KV Cache Idea
- **What to cache** — After computing K and V for a token, store them. On the next step, only compute K and V for the new token and append.
- **Prefill path** — Process all prompt tokens at once, build the cache from scratch. Output: K, V tensors of shape `[layers, heads, seq_len, head_dim]`.
- **Decode path** — Process one new token, append its K and V to the cache, attend over the full (cached) sequence.
- **The cache as a data structure** — A simple list of `(K, V)` tensors, one entry per layer.

### 4.3 Modifying OurAttention
- **New signature** — `forward(x, kv_cache=None) → (out, updated_cache)`.
- **Prefill branch** — `kv_cache is None`: compute K, V for full sequence, return them as new cache.
- **Decode branch** — `kv_cache is not None`: compute K, V for new token only, cat with cached K, V, attend over full sequence.
- **GQA still works** — The repeat_interleave happens after the cat, same as before.
- **Verify** — `check_layer()` vs HF attention with `past_key_values` → max_diff < 1e-3 ✓

### 4.4 The New Generate Loop
- **Prefill step** — Run full prompt through model, collect per-layer `(K, V)` cache.
- **Decode loop** — Pass single new token + cache → get output + updated cache → repeat.
- **Each decode step is now O(1)** — Only one token's K and V computed per step. Timing plot: flat line.
- **Correctness** — `check_correctness()` vs Ch3 output → same token IDs ✓

### 4.5 Server and Benchmark
- **Server on port 8003** — Same FastAPI structure, cached generate loop inside.
- **Benchmark vs Ch3** — TTFT ≈ same (prefill unchanged), TPOT drops significantly, throughput improves.
- **The before/after table** — Side-by-side Ch3 vs Ch4 numbers. TPOT is the number to watch.

---

---

## 5. "Why is one request at a time not enough?"

*Goal: Replace the single-request server with a batched one. Requests queue up, the scheduler collects them into a static batch, runs one padded prefill followed by a shared decode loop, and returns results to all callers simultaneously. Throughput jumps. At the end, we measure the padding waste we introduced — and name it as the thing the next major chapter will fix.*

### 5.1 The Problem: One Request at a Time
- **Ch4 server processes requests serially** — while one request is decoding, every concurrent request waits.
- **GPU utilization during decode** — a single decode step is `[1, 1, hidden]`. The GPU is wildly underutilised.
- **Throughput vs latency** — TPOT is good (Ch4 fixed that); total tokens/sec is still limited by serial execution. Measure it.

### 5.2 Why Batching Helps
- **Matrix multiply vs matrix-vector** — batching `B` decode requests turns `[1, hidden]` into `[B, hidden]`, recovering GEMM vs GEMV efficiency.
- **The GPU is a throughput machine** — it is designed for wide parallel work. One token at a time is the worst case.
- **Concrete speedup estimate** — how much throughput improvement to expect from a batch of 4, batch of 8.

### 5.3 The Scheduler: Async Batch Loop
- **The insight** — the server does not need to wait for a batch to fill; it collects whatever arrived within a time window and runs it.
- **Dummy model demo** — `03_scheduler.py` uses `asyncio` + a fake `sleep(0.2)` model to show the mechanics with no model complexity.
  - Requests arrive at random intervals (simulated async clients)
  - Scheduler wakes every `BATCH_WAIT_MS`, drains up to `BATCH_SIZE` from the queue
  - Resolves each request's `asyncio.Future` when the batch completes
  - Printed timeline shows which requests land in which batch
- **The `asyncio.Future` pattern** — each HTTP handler creates a Future, puts its request on the queue, and awaits the Future. The background loop resolves it. Handler returns naturally.

### 5.4 Static Batch: Prefill and Decode
- **Left-padding** — prompts of different lengths padded to the same length; shorter prompts get PAD tokens on the left so all "last real tokens" align at position `[-1]`.
- **Attention mask** — `1` for real tokens, `0` for PADs; passed to attention so padded positions contribute nothing.
- **Batched prefill** — one forward pass with `input_ids=[B, max_prompt_len]`, builds per-request KV caches simultaneously.
- **Shared decode loop** — one forward pass per step with `input_ids=[B, 1]`; each request appends its own token. Track EOS flags per request; stop when `all(finished)`.
- **"Wait for the slowest"** — fast requests finish but their slot stays occupied until the longest request in the batch hits EOS. Per-step timing shows this directly.

### 5.5 Server and Benchmark
- **Server on port 8004** — same FastAPI structure; incoming requests queue up, background `batch_loop` drains and processes them, Futures resolved on completion.
- **Benchmark: Ch4 vs Ch5** — send `B` concurrent requests to each server. Ch5: TTFT goes up slightly (batch collect window), throughput goes up significantly.
- **The side-by-side table:**

```
              TTFT (ms)   TPOT (ms/tok)   tok/s (total)
Ch4  KV cache    12ms         6ms/tok         25
Ch5  + batching  20ms         6ms/tok        100   ← 4× throughput, B=4
```

### 5.6 Padding Waste: Motivating the Next Chapter
- **Measuring the waste** — after the batch runs, count actual tokens generated vs slots used. Print efficiency percentage.
- **Two kinds of waste** — prompt padding (shorter prompts padded to longest) and generation padding (fast requests idle while slow ones finish).
- **Concrete example** — one 10-token request, one 200-token request in the same batch: 95% of request A's generation slots are wasted.
- **The insight** — static batching is a rectangle. Requests have jagged shapes. The mismatch is waste. Continuous batching (later) breaks the rectangle.

---

## 6. "Where does the time go?"

*Goal: Profile the model and understand which operations dominate prefill vs decode. Build a GPU mental model — memory hierarchy, roofline — just enough to reason about bottlenecks. Every subsequent optimization chapter is justified by a number from this chapter.*

### 6.1 The Profiling Tool
- **`torch.profiler` setup** — `ProfilerActivity.CPU` and `ProfilerActivity.CUDA`; how to switch between CPU-only and GPU mode with one variable so all files work everywhere.
- **The cold-start trap** — first run timing is 5–10× higher than steady-state due to CUDA context init and JIT. Always warm up before profiling.
- **Reading the table** — `Self CUDA time` vs `CUDA total`; what `aten::mm`, `ampere_sgemm_*` mean; why high-level ops show 0 self-time. `with_flops=True` covers only matmul and conv — softmax/norm are not counted.
- **`cuda.synchronize()` is not optional** — CPU enqueues GPU work asynchronously; timing without a sync captures the queue time not the execution time. Demo: time the same op with and without sync.

### 6.2 Prefill vs Decode: Two Different Bottlenecks
- **Profiling prefill** — `torch.profiler` on a 512-token prompt. Attention and MLP projections dominate. High arithmetic intensity: many FLOPs per byte of memory read. Compute-bound.
- **Profiling decode** — same model, one new token. Weight matrices still read in full, but only one token's worth of compute. Low arithmetic intensity: memory-bound. The GPU is mostly waiting on HBM.
- **The numbers** — prefill utilises 60–80% of peak FLOPS; decode utilises 5–15%. Same hardware, same model, 10× different efficiency profile.

### 6.3 The Memory Math: FLOPs and Bytes from First Principles
- **Per-layer arithmetic intensity** — given `hidden_size`, `intermediate_size`, `num_heads`, compute FLOPs and bytes transferred for `q_proj`, `gate_proj`, and attention SDPA at both prefill (S=512) and decode (S=1).
- **The GEMM/GEMV split** — prefill `q_proj`: `[S, H] × [H, H]` → AI ≈ 200 FLOP/byte. Decode: `[1, H] × [H, H]` → AI ≈ 1 FLOP/byte. Same weights, 200× different hardware efficiency.
- **KV attention bytes** — reading K, V for N cached tokens: `2 × N × head_dim × num_heads × 2 bytes`. At N=1000 tokens, this is the dominant memory traffic of the decode step.
- **No GPU needed** — arithmetic intensity is computed from tensor shapes, not measurements. The math works on CPU.

### 6.4 Module Breakdown: MLP vs Attention vs Norm
- **Per-module timing with forward hooks** — register pre/post hooks on each `mlp` and `self_attn` sub-module. Aggregate time across all N layers. Print MLP%, Attn%, Norm%, Other%.
- **Vary sequence length** — at seq=32, MLP dominates (~60%). At seq=512, attention approaches MLP (~50%). At seq=1024+, attention dominates. The crossover point is where Flash Attention starts to matter.
- **CPU and GPU mode** — hooks use `time.perf_counter()` with `cuda.synchronize()` when available. Same code path, same output format.

### 6.5 KV Cache Length Sweep and the Roofline
- **Vary KV cache length** — run decode steps with KV caches of 50, 100, 200, 500, 1000 tokens. Measure total decode step time and attention-only time via hooks.
- **Attention scales linearly with KV length** — each decode step reads all K and V from HBM. At 1000 tokens vs 50 tokens: 20× more bytes read, 20× more time. MLP stays flat.
- **This is the Flash Attention motivation** — quantified as a real measured number, not theory.
- **The roofline** — GPU memory hierarchy (HBM → L2 → SRAM). Arithmetic intensity formula. ASCII chart plotting prefill and decode relative to the hardware ridge point. On CPU: print theoretical AI table only; GPU: add measured ridge point.

### 6.6 Batch Size Effect: Escaping the Memory-Bound Regime
- **Sweep batch sizes** — decode step with B=1, 2, 4, 8 requests simultaneously. Measure throughput (total tok/s) and compute theoretical arithmetic intensity for the weight matrices.
- **AI rises with batch size** — at B=1, `q_proj` AI ≈ 1 FLOP/byte (GEMV). At B=32, AI ≈ 30 FLOP/byte (approaching GEMM). Higher batch → more work per byte of weight loaded.
- **The crossover** — at what batch size does decode become compute-bound on this hardware? Print the answer.
- **Connection to Ch9** — continuous batching is what keeps batch size high in a real serving system. This file gives the reader a number: "at B=X, we recover Y% of peak hardware efficiency".

---

## 7. "Why is tiling the trick that makes fast kernels fast?"

*Goal: Build up to a working tiled GEMM in Triton from scratch. Tiling is the one idea that makes GEMM fast and will make Flash Attention fast in Ch8. Everything else in this chapter — Triton syntax, the dispatch chain — is scaffolding to get there.*

### 7.1 The Dispatch Chain: From Python to GPU Kernel
- **What `x @ W.T` actually does** — Python → `aten::mm` → ATen dispatcher (selects CUDA key) → `Blas.cpp` → `cublasGemmEx` → SASS kernel on GPU. Read the path from the profiler: `aten::mm` shows 0 self-time; `ampere_sgemm_128x128_ldg8_*` is the real kernel.
- **cuBLAS is a black box** — highly tuned hand-written assembly, not inspectable. This is why we need Triton: to see and control what happens inside the kernel.
- **One paragraph: torch.library** — how you would register a custom op to replace `aten::mm` with your own kernel. We will do exactly this at the end of the chapter.

### 7.2 The Problem with Naive GEMM
- **Naive GEMM in Python** — three nested loops: `C[i,j] += A[i,k] * B[k,j]`. Each output element reads one full row of A and one full column of B from HBM.
- **HBM read cost** — matrix A is read N times (once per column of B). Matrix B is read M times. Total HBM reads: proportional to M×K×N — far more bytes than the matrices contain.
- **Arithmetic intensity is fine on paper, terrible in practice** — AI formula gives ~200 FLOP/byte for prefill GEMM, but naive implementation reads each byte many times, achieving effectively much less. The GPU stalls waiting for data.
- **Benchmark it** — a simple Python/NumPy naive GEMM vs `torch.mm`. The gap is the cost of not tiling.

### 7.3 Triton Hello World: Vector Add
- **What Triton abstracts** — thread block indexing, shared memory layout, warp divergence, memory coalescing — all handled by the compiler. You write operations on blocks of data, not individual threads.
- **30-line kernel** — `@triton.jit`, `tl.program_id`, `tl.arange`, `tl.load` with mask, `tl.store`. Launch with `kernel[grid](...)`. The mask guards out-of-bounds: `offsets < n_elements`.
- **CPU fallback pattern** — if no GPU, fall through to `torch.add`. All chapter code runs on CPU; only kernel launch is GPU-only.
- **Benchmark vs `torch.add`** — Triton matches PyTorch at large sizes. Understand the warm-up/compile cost on first call.

### 7.4 Tiled GEMM in Triton
- **The tiling insight** — instead of reading each row of A once per output column, load a tile of A (`[BM, BK]`) into registers once and compute all `BN` output columns that need it. Then load the next tile. Each byte of A and B is read exactly `K/BK` times total instead of `N` and `M` times respectively. HBM reads shrink by `min(BM, BN)×`.
- **The kernel loop** — outer two loops (over M and N tiles) run in parallel as separate Triton programs. Inner loop (over K tiles) is sequential: load tile of A, load tile of B, `tl.dot` → accumulate into fp32 register block, advance pointers. Write output tile once at the end.
- **Pointer arithmetic** — strides passed explicitly because Triton sees raw pointers, not tensors. `a_ptrs = a_ptr + offs_m[:, None]*stride_am + offs_k[None, :]*stride_ak` builds a `[BM, BK]` pointer grid via broadcasting.
- **Benchmark** — Triton tiled GEMM vs `torch.matmul` (cuBLAS). At 2048×2048 FP16: Triton ≈ 180 TFLOPS, cuBLAS ≈ 220 TFLOPS (~82%). At 4096×4096: Triton ≈ 220 TFLOPS, cuBLAS ≈ 224 TFLOPS (~98%). Show why: L2 cache optimization (grouped program ordering) closes most of the gap.
- **Plug in** — replace `q_proj` in one decoder layer with our Triton GEMM via `torch.library.custom_op`. Verify token IDs match. The model still works — we now control the innermost operation.

### 7.5 Why Attention Cannot Be Tiled the Same Way — Yet
- **Standard SDPA writes [T, T] to HBM** — `Q×Kᵀ` produces a `[heads, T, T]` score matrix that is written to HBM, then read back for softmax, then read again for `×V`. At T=2048: 8 MB per head, 32 heads = 256 MB per layer per forward — all in slow HBM.
- **Why tiling GEMM didn't need softmax** — GEMM output is a sum: `C += A_tile × B_tile`. Each tile's contribution is independent; you just accumulate. Attention output is NOT a simple sum: softmax requires the global max across all T positions before you can normalise.
- **The block softmax problem** — if you try to tile attention the same way, you'd compute `softmax(Q×K[:BK]ᵀ)` per tile — but this is wrong because the denominator depends on all T keys, not just the tile. You can't normalise correctly without a second pass.
- **The teaser** — this is exactly what Chapter 8 solves. Online softmax maintains a running max and running normaliser as tiles arrive, enabling single-pass tiled attention. Flash Attention is tiled GEMM + this one trick.

---

## 8. "Why is attention slow, and how does Flash Attention fix it?"

*Goal: Understand the memory bottleneck in standard attention. Build Flash Attention bottom-up: online softmax in NumPy → tiled attention in NumPy → the same algorithm in Triton (applying Ch7 skills directly) → call the production library. The varlen API is what continuous batching will use.*

### 8.1 The Problem: The Score Matrix Lives in HBM
- **Standard SDPA** — compute Q×Kᵀ → materialise `[T,T]` score matrix in HBM → softmax → multiply by V. Two full HBM round-trips for the score matrix.
- **Profile it** — at T=2048, the score matrix is `2048×2048×2 bytes = 8MB` per head. With 32 heads, 256MB per layer per forward pass just for attention scores — all in slow HBM.
- **Arithmetic intensity of SDPA** — very low: many memory reads, relatively few FLOPs. Clearly memory-bound on the roofline. Ch7's `06_attention_problem.py` already showed this spike; now we fix it.

### 8.2 Online Softmax: The Core Trick
- **The problem with block-wise softmax** — softmax needs the global max across all T positions. You can't normalise correctly block-by-block without a second pass — exactly what `06_attention_problem.py` demonstrated.
- **Online softmax** — maintain a running max `m` and running normaliser `l` as each new block of scores arrives: `m_new = max(m, block_max)`, rescale previous sum by `exp(m - m_new)`, accumulate. One pass, numerically stable.
- **Implementation** — ~25 lines of pure Python/NumPy. No GPU needed. Verify outputs match `torch.softmax` exactly on random inputs.

### 8.3 Flash Attention: Tiled Computation in Python
- **The algorithm** — process Q in row-tiles. For each Q-tile, iterate over all K,V tiles. Apply online softmax update per tile. Accumulate weighted V into output tile. Never materialise the full `[T,T]` matrix.
- **Python implementation** — ~60 lines of NumPy. Tile size `BLOCK_T` is a hyperparameter. Show that output matches standard attention exactly for all sequence lengths.
- **Memory analysis** — `[T,T]` score matrix never written to HBM. Score tiles live in SRAM (registers) for their lifetime. HBM traffic drops from O(T²) to O(T×d). At T=2048 that is 256MB → ~8MB per layer.
- **Connection to Ch7** — this is tiled GEMM (8.3 inner loop = Ch7's K-tile loop) plus the online softmax update from 8.2. No new ideas — just two things the reader already knows, composed together.

### 8.4 Flash Attention in Triton
- **Applying Ch7 skills** — the Triton FA kernel has the same skeleton as `04_tiled_gemm.py`: outer loops over Q-tiles and K/V-tiles as separate programs, inner accumulation loop, pointer arithmetic with strides, write output once at the end.
- **The new piece: online softmax in Triton** — inside the K-tile loop: `scores = tl.dot(q_tile, k_tile)`, `m_new = tl.max(scores, axis=1)`, `m = tl.maximum(m, m_new)`, rescale accumulator `acc = acc * tl.exp(m_old - m)[:, None]`, `acc += tl.dot(p, v_tile)`. One extra 5-line block on top of the GEMM loop.
- **fp32 accumulators** — same discipline as Ch7: `acc` in fp32, inputs in fp16, store output in fp16.
- **CPU fallback** — if no GPU, call the NumPy implementation from 8.3. All scripts run on MacBooks.
- **Correctness check** — compare Triton FA output vs `torch.nn.functional.scaled_dot_product_attention` for multiple `(B, H, T, d)` shapes. Max error should be < 1e-2 in fp16.
- **Benchmark vs standard SDPA** — at T=512, 1024, 2048: show HBM bytes allocated (via `torch.cuda.memory_allocated`) and wall time. Triton FA should show sublinear memory growth and 1.5–3× speedup at long T.

### 8.5 Calling the Production Library
- **FlashAttention-2 / FlashInfer** — `flash_attn_func(q, k, v)` — identical interface to our Triton implementation. Readers now know exactly what is inside.
- **Benchmark vs our Triton FA and vs SDPA** — at T=512, 1024, 2048, 4096. Production library should be 1.2–1.5× faster than our Triton kernel (autotuning + optimised backward + warp-level tricks).
- **The varlen API** — `flash_attn_varlen_func(q, k, v, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k)`. No padding: all sequences concatenated into one tensor, with cumulative length offsets. This is the API continuous batching (Ch9) will use.

### 8.6 Server with Flash Attention
- **Swap SDPA** — replace `F.scaled_dot_product_attention` in `OurAttentionCached` with `flash_attn_func` (or our Triton FA as fallback). One-line change.
- **Benchmark vs Ch7** — at short contexts (T≤128) gain is small (attention is not the bottleneck). At T=512+ TPOT drops measurably — show the cross-over point.
- **The varlen preview** — note that Ch9 (continuous batching) will use `flash_attn_varlen_func` to process jagged batches with zero padding waste.

---

## 9. "Why does static batching leave GPU slots empty, and how do we fix it?"

*Goal: Replace static (rectangular) batching with iteration-level scheduling. Show the problem, build the fix in three progressive levels — serial → batched varlen → Orca combined forward — so the reader sees exactly what each upgrade buys and why varlen attention (Ch8) is the prerequisite that makes Orca possible.*

### 9.1 The Problem: Idle Slots and Padding Waste
- **Re-measure Ch5's waste** — run static batch with varying generation lengths; plot slot utilisation vs generation length variance. At high variance, utilisation can drop below 20%.
- **Two kinds of waste** — prompt padding (shorter prompts padded to longest in batch) and decode slot waste (fast requests sit idle while slow ones finish).
- **Quantified** — at B=4 with generation lengths [10, 50, 10, 50]: static utilisation = 60%. With continuous batching: ~100%.

### 9.2 The Orca Insight: Iteration-Level Scheduling
- **The checkpoint is free** — after every forward pass, each request either emitted EOS or didn't. Hook into this checkpoint to evict finished requests and admit new ones.
- **The three-phase loop** — (1) evict finished, (2) prefill new arrivals, (3) decode all running. Repeat every iteration.
- **Async integration** — each HTTP handler creates an `asyncio.Future`, enqueues the request, awaits the Future. The scheduler loop resolves it at EOS. The Ch5 Future pattern is unchanged — only the scheduling policy changes.
- **Demo** — `01_scheduler_demo.py` with a dummy model shows the loop mechanics. Print which requests enter and exit at each step. No model weights needed.

### 9.3 Padding Waste Measured
- **`02_padding_waste.py`** — load Ch5's static batch server, send requests with exponentially distributed generation lengths, measure actual slot efficiency %.
- **The number to beat** — at high variance (mean=50, max=500 tokens): Ch5 efficiency ~15–30%. Continuous batching target: >90%.

### 9.4 Level 1 — Continuous Batching: Scheduling Insight (Serial Prefill + Serial Decode)
- **One change at a time** — implement the three-phase loop. Prefill each new request individually (`F.sdpa`, single forward pass). Decode each running request individually (loop, one at a time).
- **Per-request KV cache** — each request has its own `dict` of KV tensors, freed when the request resolves. No shared buffer.
- **Correctness** — token IDs for a single request must match Ch5's output exactly.
- **What this shows** — the scheduling insight is unmistakable. No batching complexity obscures the loop.

### 9.5 Level 2 — Batched Varlen Decode, then Batched Varlen Prefill
- **Upgrade decode first** — replace the per-request decode loop with a single `flash_attn_varlen_func` call. `cu_seqlens_q = [0,1,2,...,B]`, `cu_seqlens_k` from each request's current KV length. All running requests decoded in one kernel call.
- **Upgrade prefill second** — replace per-request prefill with a single `flash_attn_varlen_func` call over all newly admitted requests. `cu_seqlens` built from prompt lengths. `causal=True`. One forward pass for N new arrivals instead of N sequential passes.
- **`split_kv_by_request`** — after the varlen prefill pass, the flat KV output is sliced back into per-request caches using `cu_seqlens` offsets. Show this index arithmetic explicitly.
- **Correctness** — output must match Level 1 for the same requests.
- **Benchmark** — N sequential prefills vs 1 varlen batch prefill. Decode: loop vs varlen. Show the wall time difference.

### 9.6 Level 3 — Combined Forward Pass (Orca Selective Batching)
- **One forward pass for everything** — concatenate ALL tokens (prefill + decode) into a flat `[total_T, hidden]` tensor. Linear layers (QKV proj, MLP gate/up/down) see all tokens in one matmul — weight matrices loaded from HBM once.
- **Attention still per-sequence** — `flash_attn_varlen_func` with mixed `cu_seqlens_q`/`cu_seqlens_k`: prefill requests have Q length = T_prompt; decode requests have Q length = 1. `causal=True` applies within each sequence independently — both cases are handled by the same kernel.
- **Why linear layers are already compatible** — `[total_T, hidden] @ [hidden, out]` is valid for any mix of prefill/decode tokens. The distinction only matters at attention time, and varlen handles it.
- **The hardware win** — weights loaded once instead of twice per step. At large batch sizes on A100 (2 TB/s HBM): ~2× fewer weight reads.
- **Correctness** — output must match Level 2.

### 9.7 Server and Benchmark
- **Server** — FastAPI on port 8009, using the Level 3 combined forward. Each handler creates a Future, background loop runs three-phase scheduling.
- **Benchmark vs Ch5** — send B concurrent requests with exponentially distributed generation lengths. Show TTFT distribution (queued requests wait longer under load) and total throughput. Sweep generation length variance to show where continuous batching wins most.
- **The interpretation** — static batching throughput collapses as variance grows; continuous batching holds steady. The combined forward (Level 3) vs separated passes (Level 1) shows the additional benefit of batching linear layers together.

---

## 10. "Why does memory run out before the GPU is full?"

*Goal: Replace Ch9's contiguous per-request KV cache with a flat token pool. Introduce dynamic slot allocation — one slot per generated token, freed immediately at EOS — eliminating the pre-allocation waste that limits concurrent request count. Then introduce page-size grouping as a performance optimisation, showing the tradeoff between allocation overhead and last-page waste. The reader ends up with a paged memory manager that lets the same GPU serve 4–8× more concurrent requests.*

### 10.1 The Problem: Pre-Allocation Waste

- **Ch9's allocation model** — each request receives a contiguous `list[(K, V)]` sized for its generation budget at admission time. Most of it stays empty.
- **Measure the waste** — with realistic exponential generation-length distributions, only 12–20% of reserved slots are ever used. The other 80% is locked until EOS.
- **Why this caps batch size** — VRAM fills with reservations, not tokens. Halving effective VRAM halves the number of concurrent requests → halves throughput.
- **Observable output**: table showing `requests × max_seq_len` vs `requests × actual_len` across three workloads (uniform, medium variance, high variance).

### 10.2 The Token-Level Pool: Dynamic Allocation

- **The core idea** — allocate one slot the moment a token's K and V are computed. Free slots immediately when the request finishes. Never hold memory for tokens not yet generated.
- **Data structures** — flat `kv_pool_k / kv_pool_v: [NUM_SLOTS, heads, head_dim]`; `slot_table[req_id, t]` maps position `t` to a physical slot; `free_slots` deque.
- **Write path** — `write_kv(req_id, t, k, v)`: pop one slot from `free_slots`, store in `slot_table[req_id, t]`, write K/V at that slot.
- **Free path** — on EOS: return all `slot_table[req_id, :seq_len]` slots to `free_slots`.
- **Observable output**: allocator demo — 4 requests with varying lengths, slot assignments printed per step, free-list size after each completion.

### 10.3 The Gather Path and Attention

- **Read path** — `gather_kv(req_id, seq_len)`: index `kv_pool` at `slot_table[req_id, :seq_len]` → one `torch.index_select` → contiguous `[seq_len, heads, dim]`.
- **Why gather is the only attention change** — linear layers are unchanged; attention just needs contiguous K, V. Gather produces exactly that.
- **CPU implementation** — gather then `F.scaled_dot_product_attention`. Correct, slightly slower than contiguous access.
- **GPU path** — paged attention kernel gathers and attends block-by-block in SRAM without materialising. Deferred to GPU section; CPU path is the teaching path.
- **Correctness check** — single request: token IDs must match Ch9's `generate_with_cache` exactly (greedy, temperature=0).
- **Observable output**: correctness pass printed, then step-by-step decode showing slot assignments and gather shapes.

### 10.4 Scheduler Integration

- **Three-phase loop unchanged** — only the KV write and free calls change. Ch9's evict/admit/decode loop is kept verbatim.
- **On prefill** — `write_kv()` for each prompt token; slots allocated progressively, not upfront.
- **On decode** — one `write_kv()` per new token generated across all running requests.
- **On EOS** — `free_slots()` for every slot in `slot_table[req_id]`. Freed immediately, available for the next admitted request.
- **Pool exhaustion** — if `free_slots` is empty mid-generation, raise `RuntimeError("pool exhausted — reduce MAX_CONCURRENT or increase NUM_SLOTS")`. Production preemption is Ch12 scope.
- **Observable output**: 8 concurrent requests with varying lengths; slot utilisation printed per step; final pool state shows near-100% utilisation.

### 10.5 Memory Benchmark: Ch9 vs Token-Level Pool

- **Measure concurrent capacity** — increase request count until Ch9 OOMs; token-level pool continues. Show the crossover point.
- **Slot utilisation** — `actual_slots_used / NUM_SLOTS` per step. Near 100% vs Ch9's 12–20%.
- **Throughput at capacity** — same GPU, same model: how many tok/s at the request count where Ch9 has already OOM'd.
- **Observable output**: side-by-side table — max concurrent requests, slot utilisation %, throughput at capacity for Ch9 vs Step 1.

### 10.6 Page-Size Optimisation

- **The allocation overhead problem** — at high throughput, one `allocate_slot()` per token per decode step adds up. For 64 concurrent requests at 1000 tok/s: 64,000 allocations/second.
- **Group into pages** — `free_slots` tracks page-aligned starts `[0, P, 2P, ...]`. One `allocate_page()` covers `P` tokens. `slot_table` interface unchanged.
- **`_page_to_token()`** — expand page start → all P slot IDs: `(page_start + arange(P))`.
- **Last-page partial fill** — only `seq_len % P` tokens in the final page are valid. Gather must use `slot_table[req_id, :seq_len]`, not the full page.
- **Trade-off table** — page size vs waste vs allocations-per-decode-step; `P=1` (token-level), `P=16` (vLLM default), `P=64` (TRT-LLM).
- **Observable output**: allocation call count for P=1, P=16, P=64 across 1000 decode steps; last-page waste measurement.

### 10.7 Flash Attention with Paged KV

- **The remaining gap** — Steps 1–2 use gather + `F.sdpa`: K/V must be materialised to a contiguous tensor before attention runs. On GPU this is an extra HBM round-trip per decode step.
- **`flash_attn_varlen_func` with `block_tables`** — reads K/V directly from the pool in SRAM, block-by-block, without materialising. Requires flash-attn ≥ 2.3.
- **Block table tensor** — shape `[MAX_REQS, MAX_BLOCKS]` where `MAX_BLOCKS = ceil(MAX_SEQ_LEN / PAGE_SIZE)`. Unused entries set to `-1`.
- **CPU fallback** — if flash-attn unavailable, fall back to gather + `F.sdpa` from Step 2. Print which path is active. Token IDs must match exactly.
- **Correctness check** — greedy output of `flash_attn_varlen_paged` must equal gather + `F.sdpa` output token-for-token.
- **Benchmark** — gather+sdpa vs paged flash-attn: wall time and HBM bytes allocated per decode step. Shows why the gather step matters at large batch sizes.
- **Observable output**: correctness pass printed; benchmark table (wall time, HBM bytes) for B=1, 4, 16 running requests.

### 10.8 Server and Benchmark

- **Server on port 8010** — FastAPI using Step 3 (flash-attn paged) on GPU, Step 2 (gather+sdpa) on CPU. Accepts `--backend [sdpa|flash_attn] --port N --mode [online|offline]`.
- **Benchmark vs Ch9** — send concurrent requests with exponential generation lengths. Show max sustainable request count before OOM and throughput at that count.
- **The interpretation** — Ch9 OOMs at B=8–12; Ch10 runs B=32–64 on the same hardware. Throughput 4–8× higher not from a faster kernel, but from fitting more requests.
- **Observable output**: benchmark table — Ch9 max-B, Ch10 max-B, throughput ratio; slot utilisation over time plot (ASCII).

---

## 11. "Why recompute what we've already computed?"

*Goal: Add a prefix cache on top of Ch10's paged pool. Requests that share a common prompt prefix (same system prompt, same few-shot examples, same RAG document) skip the prefill for the shared portion entirely — TTFT drops proportionally to the hit rate. The reader ends up with a hash-table-based prefix cache that integrates transparently into the paged scheduler.*

### 11.1 The Problem: Identical Prefill, Every Request
- **Re-measure prefill cost** — run the same 64-token system prompt through Ch10's scheduler 10 times; every request pays the full prefill cost. Wasted compute is proportional to prompt length × request count.
- **Who is affected** — API servers with a fixed system prompt; few-shot classification (same examples repeated); RAG (same document in every request); multi-turn chat (prior turns are a fixed prefix for the next turn).
- **Quantified** — at 80% hit rate, TTFT drops 37% (vLLM V1 benchmark). SGLang reports 2–5× throughput on few-shot workloads.

### 11.2 The Chained Hash: Identifying a Prefix Page
- **One hash per page** — a page of PAGE_SIZE tokens is the unit of caching. Only full pages are cached; the partial last page is never inserted.
- **Why chaining is necessary** — `hash(page_tokens_only)` is wrong: the same tokens at position 0 vs position 16 produce identical KV only by accident. RoPE encodes position, so position matters. The fix: `hash((parent_hash, tuple(page_tokens)))`. The parent hash chains all preceding pages in, making each page's hash position-aware.
- **Implementation** — pure Python `hash()`, deterministic within a process. No SHA-256 needed for single-tenant teaching purposes.
- **Verification** — same tokens at different position → different hash; same tokens at same position → same hash; partial page raises an assertion.

### 11.3 PrefixCache: The Data Structure
- **Flat dict** — `cache: dict[int, list[int]]` maps page hash → physical slot IDs (one per PAGE_SIZE tokens). All layers share the same slot IDs (Ch10 invariant).
- **Reference counting** — `ref_count[slot_id]` prevents eviction while a request is using the slot. Incremented on match, decremented on free.
- **No radix tree** — flat dict is sufficient for prompt-only caching. Radix tree (for branching / output caching) is deferred.
- **Demo** — insert 3 prefix pages, match a new request that shares 2, verify ref_counts are correct and slots are not double-freed.

### 11.4 Scheduler Integration: Match Before Prefill, Insert After
- **Match path** — before prefilling a new request, walk its pages and look up each hash. Return the matched length (in tokens) and the reused slot IDs. Skip prefill for matched tokens.
- **Prefill only the suffix** — `model(prompt_tokens[matched:])`; the matched KV is already in the pool. Assign newly allocated slots for the suffix.
- **Insert path** — after prefill completes, walk all full pages of the prompt and insert them into the cache. The partial last page is skipped.
- **Correctness check** — same prompt twice: both runs produce identical token IDs; second run logs `cache_hit=N tokens skipped`.

### 11.5 LRU Eviction
- **When to evict** — the pool has finite slots. When `alloc.n_free < needed`, evict the least-recently-used cache entry with `ref_count == 0`.
- **LRU tracking** — `last_used: dict[hash, float]` updated on every cache hit. Evict smallest timestamp first.
- **Safety** — only evict slots with `ref_count == 0`; an in-use slot is never evicted.
- **Steady-state demo** — 20 requests against a pool of 64 slots with 3 distinct prompts: hit rate stabilises after a warm-up period; eviction events printed.

### 11.6 Server and Benchmark
- **Server on port 8011** — same FastAPI + paged scheduler structure; `/health` adds `hit_rate` and `cached_pages` fields.
- **Benchmark vs Ch10** — 20 identical prompts: Ch10 TTFT is flat (full prefill every time); Ch11 TTFT drops after the first request (cache warm). Print cold TTFT, warm TTFT, and hit rate.
- **What we deferred** — output-token caching (multi-turn full reuse), radix tree for branching, copy-on-write, cross-process cache sharing.

---

## 12. Quantization
*Goal: Understand how GPTQ compresses model weights to 4 bits (W4A16), implement dequantization from scratch — INT32 unpacking to FP16 — and measure the quality cost of that compression on the same model at both precisions.*

### 12.1 Number Formats and the Memory Math
- **Bit layouts** — FP32 (1+8+23), FP16 (1+5+10), BF16 (1+8+7), INT8 (1+7), INT4 (1+3): what each bit buys you
- **Bytes-per-parameter table** — FP16 = 2 B, INT8 = 1 B, INT4 = 0.5 B; a 135M model shrinks from 270 MB to 68 MB
- **Why W4A16** — weights in INT4 halve HBM load per decode step; activations stay FP16 for accuracy; no INT4 matmul needed
- **Observable output**: table of dtype → bytes → model size in MB; memory bandwidth math for decode

### 12.2 The GPTQ Checkpoint: What's Inside
- **Load `quantize_config.json`** — read `bits=4`, `group_size=128`, `sym=False`, `desc_act`, `damp_percent` and explain each
- **Inspect layer tensors** — print shapes and dtypes of `qweight`, `scales`, `qzeros`, `g_idx` for one Linear layer
- **The packing ratio** — 8 INT4 values packed per INT32; `qweight` shape is `[in_features // 8, out_features]`
- **Per-group quantization** — 128 consecutive input features share one (scale, zero_point) pair
- **Observable output**: `quantize_config.json` pretty-printed; one layer's tensor shapes and dtypes; packing ratio calculation

### 12.3 Unpacking INT4 — Bit Manipulation from Scratch
- **8-into-1 packing** — `int32[i]` holds 8 signed 4-bit values in bits `[0:4], [4:8], ..., [28:32]`
- **Extraction** — `nibble_k = (int32 >> (4*k)) & 0xF`; convert from unsigned to signed: values ≥ 8 become `val - 16`
- **Verify round-trip** — repack unpacked values → must match original `qweight` exactly
- **qzeros unpacking** — same bit layout, same extraction; zeros are per-group INT4 offsets
- **Observable output**: step-by-step: one INT32 → 8 INT4s printed as table; round-trip ✓ assertion

### 12.4 Dequantization: INT4 → FP16
- **Per-group formula** — `W_fp16[i] = scale[group(i)] × (W_int4[i] − zero[group(i)])`
- **g_idx mapping** — `g_idx[i]` maps input feature `i` to its group index; handles non-sequential group assignment (`desc_act=True`)
- **Implementation** — pure PyTorch, no CUDA required; compare `W_fp16_ours` against `W_fp16_ref` (dequantized by transformers)
- **Numerical tolerance** — expect `allclose(atol=1e-3)` in FP16; small rounding differences from different dequant paths
- **Observable output**: one weight matrix reconstructed; max absolute error printed; `allclose ✓`

### 12.5 Quantized Linear Layer — Running the Model
- **`QuantizedLinear`** — stores `qweight`, `scales`, `qzeros`, `g_idx`; dequantizes in `forward()` then does FP16 matmul
- **Replace all Linear layers** — walk the model's `named_modules()`; swap `nn.Linear → QuantizedLinear` preserving bias
- **Correctness** — our `QuantizedLinear` logits must match transformers' `GPTQLinear` logits (`allclose(atol=0.05)`)
- **Observable output**: layer-by-layer replacement log; logit comparison against transformers GPTQ model

### 12.6 Quality Measurement
- **Perplexity** — run both FP16 (SmolLM-135M) and GPTQ model on wikitext-2 sentences; print perplexity delta
- **Greedy token accuracy** — feed 10 prompts; compare argmax token sequences; report % match
- **Size comparison** — `model.state_dict()` bytes: FP16 vs GPTQ; print 4× compression ratio
- **Key result**: GPTQ W4A16 achieves ~95–100% greedy accuracy vs FP16; ~4× smaller; slight perplexity increase
- **Observable output**: table — FP16 perplexity / GPTQ perplexity / delta / greedy accuracy % / model MB

### 12.7 Server and Benchmark
- **Server port 8012** — FastAPI serving GPTQ model; `--precision [fp16|gptq]` flag; `/health` returns `precision`, `model_mb`, `bits`, `group_size`
- **Benchmark** — run 20 prompts against both `--precision fp16` and `--precision gptq`; measure throughput tok/s and first-token latency
- **CPU note** — on CPU, GPTQ dequant adds overhead; throughput may be lower than FP16; on GPU smaller weight load = faster
- **Observable output**: benchmark table — precision → model MB → tok/s → greedy accuracy %

---

## 13. Speculative Decoding
*Goal: Use a small fast draft model (SmolLM-135M) to speculatively propose K tokens, then verify them all with the larger target model (SmolLM-360M) in a single forward pass — guaranteeing the output is identical to running the target alone while generating multiple tokens per target call.*

### 13.1 The Serial Cost of Autoregressive Decode
- **Decode is O(N) target calls** — every token requires a full forward pass; no inter-token parallelism
- **Memory-bandwidth bottleneck** — the GPU spends most time loading weights, not computing; decode is not compute-bound
- **Time grows linearly** — 10 tokens takes 10× longer than 1 token; no shortcut without changing the algorithm
- **Observable output**: `ms/token` for SmolLM-360M generating 1, 5, 10, 20 tokens; linear growth plotted

### 13.2 Draft Proposals — Running the Small Model First
- **SmolLM-135M generates K=4 tokens autoregressively** — 4 forward passes of the 135M model
- **Each draft step records token + probability** — the draft's confidence at each position
- **Draft is cheap** — 135M model is ~2.6× smaller; prints elapsed time vs one 360M forward pass
- **Observable output**: 4 draft tokens printed with ids, text, and probability at each step

### 13.3 Parallel Verification — One Target Forward Pass for K Tokens
- **Concatenate context + K draft tokens → single input** — the target sees all K+1 positions at once
- **Causal mask makes this exact** — logit at position L+i depends only on tokens before it; this is just a prefill step
- **Extract K+1 logits** — one per draft position plus the bonus position after the last draft token
- **Observable output**: target probability of each draft token printed; total time for one target pass vs K sequential passes

### 13.4 Greedy Acceptance / Rejection
- **Accept draft[i] iff == target argmax** — deterministic, no randomness needed under greedy decoding
- **On rejection**: replace with target argmax at that position; discard remaining draft tokens
- **Bonus token**: if all K accepted, take one extra token from target logits[K] — always correct, always free
- **Correctness**: output tokens must exactly match target-only greedy generation (token-level exact match ✓)
- **Observable output**: per-position decision table (accept ✓ / reject ✗ + correction); final text matches target-only

### 13.5 Sampling Acceptance / Rejection
- **Accept with probability** `min(1, p_target(x) / p_draft(x))` — tokens the target favours more than the draft are always kept
- **On rejection**: sample from residual distribution `max(0, p_target − p_draft) / Z` — redistributes mass the draft over-allocated
- **Mathematically exact**: combined procedure produces the target distribution exactly (rejection sampling theorem)
- **Observable output**: acceptance rate α printed per prompt; 200-sample frequency check shows token distribution matches target

### 13.6 Measuring Speedup and Acceptance Rate
- **Sweep K = 1, 2, 4, 6, 8** — measure actual tokens accepted per iteration and compare to theoretical `E[N] = (1 − α^{K+1}) / (1 − α)`
- **Acceptance rate α varies by prompt** — predictable prompts (code, templates) get higher α; open-ended prompts lower
- **CPU caveat**: no wall-clock speedup on CPU (memory-bandwidth saving requires GPU); algorithm cost is visible
- **Observable output**: table — K → theoretical E[N] → actual tokens/iter → α; formula matches within ~10%

### 13.7 Server and Benchmark
- **Server port 8013** — accepts `--draft_model`, `--target_model`, `--draft_k` flags; `/health` exposes `acceptance_rate`
- **Benchmark**: target-only server (port 8001) vs spec decode server (port 8013) on 20 prompts
- **GPU note**: speedup only measurable on GPU; on CPU spec decode is slower due to two-model overhead
- **Observable output**: table — precision → tok/s → acceptance rate → speedup ratio vs target-only

---

## 14. Parallelism
*[not yet detailed]*
- a. Multi-GPU primitives
- b. Data parallel
- c. Tensor parallel
- d. Other parallel
- e. Multi-node parallelism

---

## 15. Disaggregation
*[not yet detailed]*
- How disaggregation works
- When to use disaggregation
- Dynamic disaggregation with NVIDIA Dynamo

---

## 16. Advanced Scheduling
*[not yet detailed]*
- Chunked prefill
- Mixed batches
- CUDA graphs
- Overlap scheduling

---

## Appendices
*[not yet detailed]*

- A. Kernel fusion and compile, CUDA graphs
- B. Scheduling and overlap
- C. OpenAI API serving client
- D. Metrics / sampling
- E. Production, scaling, monitoring
- F. Roofline model, GPU analysis
- G. Chunked prefill, mixed phases
- H. MoE basics / inference

---

## Potential Chapters
*Topics researched from vLLM, SGLang, mini-sglang, nano-vllm source and recent literature (2025–2026). Each is a candidate to be promoted into a full numbered chapter.*

---

### Tier 1 — Must-have for a working engine

**CUDA Graphs**
Every production engine (vLLM, SGLang, mini-sglang, nano-vllm) captures the decode loop as a CUDA graph to eliminate Python kernel-launch overhead — often a 20–40% throughput gain. Covers: why CPU dispatch overhead dominates small batches, capturing per batch-size buckets, keeping prefill eager while graph-ing decode, and piecewise graphs for variable-length prefill shapes (SGLang's innovation).

**Chunked Prefill + Overlap Scheduling**
Sarathi-Serve's insight: split long prefills into chunks and interleave with decode steps so a single large prompt doesn't block the decode loop. Overlap scheduling (NanoFlow / mini-sglang's `overlap_loop`) pipelines CPU scheduling of batch N+1 while the GPU runs batch N. Together these fix TTFT collapse under mixed load without sacrificing throughput.

**Tensor Parallelism**
You cannot practically serve a 7B+ model without splitting weight matrices across GPUs. Covers Megatron-style column/row parallel splits, all-reduce placement (where it goes and what it costs), vocab sharding on the LM head, and how TP integrates with the existing paged scheduler and KV cache pool from Ch10.

**Streaming Output (SSE)**
Every production API streams tokens one-by-one over Server-Sent Events rather than returning a full response. Covers the SSE protocol, async generators, per-token flush, cancel-on-disconnect, and why TTFT and TPOT mean different things to users under streaming — the serving contract that makes latency numbers meaningful.

**Scheduler Preemption: Recompute vs. Swap**
When GPU memory fills mid-generation, the scheduler must either evict a request's KV blocks and requeue it (recomputation) or transfer those blocks to CPU RAM and restore them later (swapping). Covers the cost model for each strategy, when each wins (recompute for short sequences, swap for long), and how vLLM V1's unified `{req_id: num_tokens}` representation makes both clean to implement.

---

### Tier 2 — Significantly improve quality or production readiness

**Tensor Cores and Low-Precision Compute (FP16 → FP8 → INT8 → FP4)**
Tensor cores are dedicated matrix multiply-accumulate (MMA) units on each GPU SM that deliver 10–60× higher throughput than CUDA cores — but only for specific data types and only when matrix dimensions are aligned to multiples of 16/32/64. This chapter is the hardware foundation that makes Ch7 (Triton GEMM) and Ch12 (quantization) actually make sense: *why* does FP8 give ~2× throughput over BF16? *Why* do model hidden dims always need to be multiples of 128? *Why* is the roofline ridge point at a specific arithmetic intensity? Covers: (1) *the precision-throughput table* — FP32 (CUDA cores, 67 TFLOPS on H100) vs TF32 (989 TFLOPS) vs BF16/FP16 (1,979 TFLOPS) vs FP8 (3,958 TFLOPS) vs FP4 Blackwell (~5,000+ TFLOPS) — each tier roughly doubles throughput for the same memory bandwidth, shifting the roofline ridge point; (2) *alignment requirements* — tensor core instructions (WMMA / MMA) require matrix dimensions to be multiples of 16 (BF16), 32 (INT8/FP8), or 64 (FP4); misaligned dimensions fall back to CUDA cores silently — this is why every production model has hidden_size divisible by 128; (3) *W4A16 vs W8A8 vs FP8 — the critical distinction* — Ch12's GPTQ is W4A16: weights are INT4, activations are FP16; dequantization to FP16 happens before the GEMM so the tensor core still sees FP16; the weight compression saves memory bandwidth but does NOT unlock INT8/FP8 tensor core throughput; W8A8 (both weights and activations INT8) feeds INT8 tensor cores directly for full throughput gain; FP8 (H100 native) is training-free on some models and hits 2× BF16 throughput with near-identical quality; (4) *Marlin and fused dequant+GEMM kernels* — naive W4A16 dequantizes in a separate kernel before GEMM, paying an extra HBM round-trip; Marlin-style kernels fuse dequant into the GEMM register-tile loop so INT4 weights are expanded to FP16 inside SRAM, never written to HBM; this closes most of the gap between W4A16 quality and FP8 throughput; (5) *writing an FP8 GEMM in Triton* — extending Ch7's tiled GEMM to FP8: `tl.load` in FP8, `tl.dot` with fp32 accumulator, `tl.store` back in FP8; the scaling factor per tensor (per-tensor vs per-row vs per-channel) and why scaling granularity determines quality; (6) *updating the roofline* — the ridge point for FP8 GEMM is at higher arithmetic intensity than FP16 GEMM; decode steps (low arithmetic intensity) do not benefit from FP8 tensor cores; prefill steps (high arithmetic intensity) do; the chapter quantifies exactly where the crossover is for a given hidden size and batch size.

**torch.compile + Kernel Fusion**
`torch.compile` with `mode="reduce-overhead"` fuses RMSNorm, RoPE, sampler, and other hot ops automatically — all four are compiled in nano-vllm. Covers what gets fused and why, when it conflicts with CUDA graphs, piecewise compilation for variable-shape prefill, and how to profile the difference before and after.

**Advanced Speculative Decoding: EAGLE-2/3 + N-gram + Tree Attention**
Ch13 covers basic draft-and-verify; production uses EAGLE (a lightweight draft head trained on the target's hidden states, not a separate model) for 3–6× speedup, n-gram matching against the prompt itself for zero-cost drafting on repetitive completions, and tree attention to verify a branching tree of K candidates in one pass instead of a single chain.

**Multi-Token Prediction (MTP) Heads**
DeepSeek-V3 bakes extra prediction heads into the model at training time; at inference these heads draft tokens that the main head verifies, achieving 83% acceptance at k=1 and 72% at k=2 with no separate draft model. Covers the sequential MTP architecture, the "absorb" trick for efficient inference, and how this differs from EAGLE.

**Beyond GPTQ: FP8, AWQ, and KV Cache Quantization**
Ch12 covers W4A16 GPTQ; the 2025 production stack runs FP8 (W8A8) natively on H100/H200 with near-lossless quality at 2× compression, AWQ for better quality at the same bit-width, and INT8/FP8 KV cache quantization that shrinks the paged pool directly — often a better tradeoff than reducing weight precision further.

**Radix Tree Prefix Cache**
Ch11's flat hash dict works only for a fixed system prompt. A radix tree enables branching — multi-turn conversations, beam search, shared few-shot prefixes with different suffixes, and copy-on-write when a shared prefix forks. Covers tree construction, reference counting, LRU eviction on tree nodes, and the CUDA-accelerated key comparison in mini-sglang.

**Tiered KV Cache: CPU, NVMe, and Remote Storage (LMCache)**
When VRAM fills, proactively tier KV blocks to CPU DRAM → NVMe → network storage rather than simply evicting. LMCache (2025) achieves up to 15× throughput improvement on multi-turn workloads by pipelining KV transfers across tiers with compute. Covers the tiering policy, async offload, NVIDIA KVBM's block manager, and when tiering beats eviction-and-recompute.

**Cache-Aware Routing (Multi-Instance Load Balancing)**
Round-robin routing across multiple inference instances destroys prefix cache hit rates. Cache-aware routing maintains an approximate radix tree per worker and sends requests to the instance with the longest prefix match; SGLang's router achieves 2× throughput vs. round-robin for shared-prefix workloads. Covers approximate tree matching, PrefixHash consistent hashing, and the load-balance fallback.

**Structured Output / Constrained Decoding**
More than 50% of production API traffic uses `response_format: {type: "json_object"}` or a JSON schema. At every decode step, logits must be masked to only allow tokens consistent with the current grammar state. Covers FSM-based constrained decoding (outlines), pushdown automata (lm-format-enforcer), SGLang's grammar bitmask CUDA kernels, and the throughput cost of constraint checking.

**Batched Sampling Optimizations**
Ch2 introduces sampling conceptually; production engines use Triton kernels for batched top-k/top-p, repetition penalty, frequency/presence penalty, min-p, and logit bias across a batch of 64+ concurrent requests simultaneously. Covers the difference between per-token Python sampling and a fused batch sampler, and where sampling sits in the overall latency budget.

**KV Cache Compression / Selective Eviction**
Instead of storing all K,V tensors indefinitely, evict "unimportant" tokens mid-generation. H2O keeps heavy-hitters (tokens receiving the most attention historically); Scissorhands exploits persistence of importance; DMS (NeurIPS 2025) delays eviction to implicitly merge representations, achieving 8× compression with only 1K fine-tuning steps. Covers the importance scoring function, fixed memory budget allocation, and quality vs. compression tradeoffs.

---

### Tier 3 — Advanced / specialized

**PD Disaggregation (detailed)**
Ch15 outlines disaggregation; a full chapter would implement it. Prefill is compute-bound and wants compute-heavy GPUs; decode is memory-bandwidth-bound and wants memory-heavy ones. KV cache computed on the prefill node is transferred via RDMA (NIXL, Mooncake) to the decode node. Covers the transfer protocol, non-blocking KV send while GPU continues, router integration, and the TTFT/throughput tradeoff curve.

**MoE Inference + Expert Parallelism**
Over 60% of top open-source model releases in 2025 use MoE (DeepSeek-V3, Qwen3-MoE, Kimi K2). Covers token routing (topK gate), all-to-all expert dispatch, fused MoE Triton kernels, expert parallelism with DeepEP (77µs dispatch latency at EP=8 on H800), expert load imbalance and EPLB rebalancing, and why MoE serving is fundamentally different from dense model serving.

**MLA Attention (Multi-Head Latent Attention)**
DeepSeek-V2/V3's defining architectural choice: compress KV cache from `[layers, num_heads, head_dim]` down to a single low-rank latent vector per token, reducing KV memory 5–13× vs. standard GQA. Covers the latent projection, the "absorb" trick that avoids materializing full KV at runtime, FlashMLA and Cutlass MLA backends, and why MLA changes the paged pool structure.

**Multi-node Tensor Parallelism + All-Reduce Cost**
Within-node NVLink runs at 900 GB/s; cross-node InfiniBand/RoCE is 10–20× slower, fundamentally changing the optimal TP degree. Covers the all-reduce cost model (latency vs. bandwidth regimes), hierarchical all-reduce (NVRAR: 1.9–3.6× lower latency than NCCL), compute-communication overlap (TokenWeave's fused AllReduce-RMSNorm), and the rule of thumb for when cross-node TP stops being worth it.

**Context / Sequence Parallelism (Ring Attention + Ulysses)**
For contexts beyond 64K tokens a single GPU cannot hold the full attention map. Ring Attention scatters the sequence across GPUs and passes KV tiles peer-to-peer while overlapping communication with computation; DeepSpeed-Ulysses shards attention heads instead. Tetris (fine-grained SP for serving) achieves 4.35× lower TTFT. Covers the two approaches, when each wins, and how to integrate them with existing TP.

**Native Sparse Attention (NSA)**
DeepSeek's trainable sparse attention mechanism uses dynamic hierarchical sparsity (coarse token compression + fine-grained selection) to attend to only a fraction of the sequence. Achieves 11.6× decode speedup and 9.0× prefill speedup at 64K context length while maintaining full-attention quality. Covers the sparsity pattern, hardware-aligned tiling, and inference-time implications for the KV pool.

**Pipeline Parallelism**
For models too large to fit even with TP (>70B on a single node), split layers across GPU groups. SGLang's chunked pipeline parallelism achieves 3.31× prefill throughput on DeepSeek-V3. Covers pipeline stages, micro-batching to hide the pipeline bubble, async P2P communication, and the tradeoff between throughput and per-request latency introduced by pipelining.

**LoRA / Multi-Adapter Serving**
Production APIs serve thousands of fine-tuned LoRA adapters on a single base model. Covers batching requests with different LoRA adapters in one forward pass (requiring adapter weight merging at each layer), hot-swapping adapters between requests, managing the adapter memory pool, and the throughput cost of heterogeneous LoRA batches.

**Production Observability and SLO-Aware Scheduling**
A chapter on what it takes to run the engine under real traffic: Prometheus metrics (TTFT P50/P99, queue depth, GPU utilization, token throughput), health and readiness endpoints, graceful shutdown, SLO-aware scheduling (prioritize requests approaching their deadline), request cancellation on client disconnect, rate limiting, and multi-instance load balancing patterns.

**Semantic / Exact-Match Caching**
Beyond prefix caching (which reuses KV for shared prompt prefixes), semantic caching stores complete request→response pairs and serves cached answers for semantically equivalent queries without any model invocation — 5–20ms vs. 1–3s for inference. Covers exact-match via hash, approximate-match via embedding similarity, cache invalidation, and when semantic caching is appropriate vs. harmful (non-deterministic, personalized, real-time outputs).

---

### MLOps & Platform Layer — Above the Inference Engine

*These chapters cover the infrastructure that sits between end-users and inference engines. They are not about model weights or GPU kernels — they are about what it takes to operate inference at production scale: routing, billing, deployment, and continuous improvement. Each topic is implemented in real production systems (SGLang's sgl-model-gateway, vLLM's sleep API, Ray Serve) and represents concerns that appear immediately when you take a single-engine implementation and try to serve thousands of users.*

---

**LLM API Gateway and Model Platform**
A single inference engine is not a product — it is a component. The platform layer in front of it handles concerns no individual engine should own: routing requests across multiple engine instances and multiple model versions, validating and authenticating API keys (Bearer token, per-key rate budgets), accounting for token consumption per user/org for billing, A/B testing between model versions by splitting traffic by percentage or user cohort, circuit-breaking to a fallback model when the primary is overloaded, and providing a single OpenAI-compatible endpoint that hides the entire fleet topology from clients. SGLang's `sgl-model-gateway` (Rust) is a reference implementation: it exposes worker registration via a `/workers` API, implements four load-balancing policies (round_robin, cache_aware, power_of_two, bucket), uses token-bucket rate limiting per worker group, emits 40+ Prometheus metrics and OpenTelemetry traces, and optionally discovers backends from Kubernetes service endpoints. MLflow AI Gateway and LiteLLM solve the same problem at the provider-routing level (OpenAI, Anthropic, Bedrock behind one endpoint). A chapter would implement a minimal gateway in Python (FastAPI + async httpx): request validation, API key middleware, round-robin dispatch, per-user token counter, and a basic A/B traffic splitter — showing how this separates cleanly from the engine's core scheduler concerns.

**Benchmarking, Profiling, and Capacity Planning**
After building an inference engine, the immediately obvious question is: *is it fast?* — and answering it correctly requires a whole methodology. This chapter covers the full measurement stack: (1) *the benchmarking protocol* — `benchmark_serving.py`-style synthetic load generators that replay real request-length distributions (ShareGPT, WildChat, synthetic Poisson arrivals), sweep request rates from underloaded to saturated, and report throughput (req/s, tok/s), latency (TTFT P50/P99/P999, ITL, E2E), and goodput (fraction of requests meeting SLO); (2) *the throughput-latency Pareto curve* — how to plot it, why the "knee" tells you the engine's optimal operating point, and what happens past the knee (queue buildup, latency collapse); (3) *MFU (Model FLOP Utilization)* — vLLM's `perf.py` computes FLOPs analytically from model config (hidden size, layers, attention heads) and compares against measured throughput; an MFU < 30% on prefill means you are memory-bandwidth-bound on attention, not compute-bound on GEMMs; (4) *kernel-level profiling* — `torch.profiler` with Chrome trace export, NVIDIA Nsight Systems for CPU-GPU overlap visualization, CUDA event timers for individual kernel durations; how to identify whether your bottleneck is a specific kernel, a CPU dispatch stall, or an all-reduce; (5) *the roofline model revisited* — given your engine's measured arithmetic intensity per operation (attention vs. GEMM vs. all-reduce), plot where each falls on the roofline and whether a kernel is theoretically bound by memory bandwidth or compute; this tells you whether optimization effort should go into reducing memory traffic or increasing compute utilization; (6) *capacity planning* — given measured throughput at target SLOs, how many GPUs do you need for N req/s? how does adding TP affect the per-request cost? what is the break-even concurrency for PD disaggregation? A chapter would take the book's Ch9 server, run it through the full benchmark protocol, profile a specific bottleneck (e.g., the attention kernel), and show how to use the roofline to decide whether chunked prefill or CUDA graphs should be the next optimization.

**Online Weight Updates and RLHF Integration**
The distinction between training and inference is eroding. Modern RLHF pipelines (PPO, GRPO, DPO-online) alternate between generating rollouts with the inference engine and updating the model weights with the training process — often on the same GPU cluster, sometimes on the same GPUs. This creates a requirement no earlier chapter covers: the inference engine must accept hot weight updates mid-serving without restarting, without dropping in-flight requests, and without introducing a visible latency spike. SGLang implements this via `weight_sync/`: `UpdateWeightsFromTensorReqInput` carries serialized named tensors from the training process; the engine's `update_weights_from_tensor()` method applies them while the scheduler is quiesced; `FlattenedTensorBucket` batches many small tensors into one flat buffer to reduce IPC overhead; TP groups use `gather_object` to synchronize shard assignments. vLLM takes a complementary approach: a `sleep(level=1|2)` API suspends the engine and optionally offloads weights to CPU (level 2) to free GPU memory for a training step, then `wake()` reloads and resumes serving — the 3-process RLHF pattern runs actor, reference, and reward models in a shared memory pool with `sleep`/`wake` coordinating GPU time. Key concerns: (1) *weight update atomicity* — a request that starts a forward pass during a weight update must see either the old or new weights, not a mix; (2) *TP shard consistency* — all TP workers must apply the same update simultaneously; (3) *quantized weight reloading* — if the engine runs FP8 but training produces BF16, the update path must re-quantize in-flight; (4) *versioned weight sets* — for A/B testing two model checkpoints simultaneously. A chapter would implement a minimal `WeightUpdateServer` gRPC endpoint on top of the book's engine, demonstrate a complete PPO-style loop (generate → score → update → generate), and show the latency impact of naive vs. quiesced updates.

**Autoscaling and Elastic Serving**
A single engine instance cannot handle variable traffic economically: over-provisioning wastes GPU cost at night, under-provisioning causes SLO violations at peak. This chapter covers the full elastic serving stack: (1) *scale-to-zero* — vLLM's `sleep(level=2)` offloads weights to CPU and frees GPU memory entirely, enabling a running process to "hibernate" when idle and resume in ~5s (weight reload) vs ~120s (cold start from scratch); this makes scale-to-zero economically viable for bursty workloads; (2) *Kubernetes HPA with custom metrics* — the standard CPU/memory HPA metrics are useless for GPU inference; production systems expose custom metrics (request queue depth, GPU utilization, TTFT P99) via Prometheus and route them to HPA via `kube-prometheus-stack` + `prometheus-adapter`; (3) *replica-aware routing* — adding a replica must immediately receive traffic weighted by its queue depth, not after a fixed health-check interval; removing a replica must drain in-flight requests (graceful shutdown with a `SIGTERM` drain timeout); (4) *KV cache migration* — when scaling down, active sessions lose their KV state if the engine instance is killed; KV migration protocols (a subset of PD disaggregation) transfer live KV blocks to a surviving instance; (5) *elastic expert parallelism* — for MoE models, the number of expert-parallel shards can be increased at runtime by redistributing expert weights across newly added GPUs without restarting the model server; SGLang's EPLB load balancer recalculates expert-to-GPU assignment and broadcasts new routing tables; (6) *multi-LoRA autoscaling* — when adapter demand for a specific LoRA checkpoint spikes, the engine can replicate that adapter to additional workers and retire it when demand drops. A chapter would implement a minimal scale-to-zero controller in Python: a process that monitors queue depth via the engine's `/metrics` endpoint and calls `sleep()`/`wake()` to park/unpark the engine, then connects this to a simulated traffic pattern showing cost reduction vs. always-on serving.

---

### Architectural Variants — Different model families, different serving contracts

*These are not optimizations layered on top of a text LLM — they are fundamentally different model architectures that each require a distinct serving stack. Each would be a self-contained chapter explaining the architecture first, then how inference differs from everything the book has built so far.*

---

**Multimodal Serving: Vision-Language Models (VLMs)**
Models like LLaVA, Qwen2.5-VL, InternVL, and PaliGemma embed a vision encoder (ViT/CLIP/SigLIP) upstream of the LLM decoder. At prefill time the encoder runs over pixel patches to produce embedding vectors that replace placeholder token positions in the input sequence; the decoder then runs a normal KV-cache-based autoregressive loop over the merged token+patch sequence. The key engineering challenges that differ from text-only serving are: (1) the encoder's output is *embeddings*, not KV — a separate `encoder_cache` keyed by `mm_hash` stores computed patch embeddings so the same image across multiple requests pays the ViT cost only once; (2) the scheduler must track an *encoder compute budget* (ViT FLOPs) and *encoder cache capacity* in addition to the usual KV block budget; (3) chunked prefill interacts with image tokens non-trivially — an image chunk must be processed atomically, and large images (e.g. 576–3,136 tokens per image) can exceed a typical prefill chunk size; (4) M-RoPE (Multimodal RoPE) encodes 2D spatial position of patches rather than 1D sequence position; (5) ViT CUDA graph runners capture fixed-shape encoder passes for common image resolutions. The chapter would implement a minimal VLM pipeline on top of the book's existing text server: a ViT encoder pass, embedding merge, encoder cache, and modified scheduler budget — and benchmark the TTFT overhead of encoding a 448×448 image vs. a same-length text prompt.

**Image Diffusion Model Serving (Stable Diffusion, FLUX, DiT)**
Image diffusion models (Stable Diffusion, FLUX, PixArt, Hunyuan) have a completely different inference contract from autoregressive LLMs: there is no KV cache, no token-by-token generation, and no prefill/decode asymmetry. A request is one full *denoising loop*: text encoder (CLIP/T5) → sample Gaussian noise in latent space → N timesteps of DiT transformer forward passes → VAE decode to pixels. The bottleneck is the denoising loop: 20–50 forward passes of a large transformer over a spatial latent `[B, H/8, W/8, C]`. Key serving challenges: (1) *request batching* — requests with different image sizes produce different latent shapes; dynamic shape batching or bucketing is required; (2) *step-level parallelism* — PipeFusion splits image patches across GPUs and reuses stale activations from adjacent timesteps to reduce communication; xDiT combines sequence parallelism, PipeFusion, CFG parallelism, and data parallelism; (3) *CFG (Classifier-Free Guidance)* doubles the batch size (conditional + unconditional pass per step) — CFG parallelism splits these across GPUs; (4) *TeaCache / step skipping* reuses attention outputs from nearby timesteps to skip redundant computation; (5) *FP8 DiT + fused attention* (SageAttention, FlashAttention for 2D patches) are standard; (6) SLO-aware scheduling — TetriServe dynamically adjusts per-request parallelism degree based on deadlines. A chapter would build a minimal FLUX/SDXL serving loop showing the stage pipeline, batching policy, and benchmark against a naive sequential implementation.

**Text Diffusion Model Serving (dLLM: Masked Diffusion)**
Text diffusion models (LLaDA, Mercury, SDAR) are autoregressive's parallel cousin: instead of predicting one token at a time, they start with a fully masked output, then iteratively fill in tokens over multiple rounds using a bidirectional transformer that can see the whole (partially-unmasked) sequence. Mercury achieves >1,000 tokens/sec on H100 — up to 10× faster than autoregressive models — because many tokens are filled per forward pass. The key differences from standard LLM serving: (1) *no causal mask* — the transformer is bidirectional, so the full sequence (masked + unmasked) is visible at every step; (2) *no KV cache as primary bottleneck* — the sequence is fixed-length from the start, so the memory pattern is completely different; (3) *iterative denoising scheduler* — instead of a "generate until EOS" loop, the engine runs for a fixed number of denoising steps, each filling a subset of [MASK] tokens based on a confidence threshold (Low-Confidence or Joint-Threshold algorithms in SGLang's dLLM stack); (4) *block diffusion* — a hybrid: autoregressive across blocks, parallel within each block, giving a tunable tradeoff between parallelism and quality; (5) *full-sequence logits* — unlike causal LLMs which only need the last token's logit, dLLMs need logits for all masked positions at every step; (6) *request lifecycle differs* — a request enters with its full output length pre-allocated as [MASK] tokens, and the scheduler tracks how many mask tokens remain rather than how many tokens have been generated. SGLang implements this with `ForwardMode.DLLM_EXTEND`, dedicated `SchedulerDllmMixin`, and block-wise position construction. A chapter would implement a minimal masked-diffusion text generator from scratch, compare quality vs. speed vs. an autoregressive baseline, and show why the KV cache design changes completely.

**Audio / Speech Serving (Whisper-style Encoder–Decoder)**
Audio models like Whisper use a true encoder-decoder architecture: a convolutional + transformer encoder processes the raw audio spectrogram into fixed-length hidden states, then a separate autoregressive decoder cross-attends to those states to generate text tokens. This is the only architecture in the book where the encoder has a *separate KV cache* from the decoder (cross-attention KV, not self-attention KV), sized by `max_num_encoder_input_tokens`. The serving challenges differ from VLMs: (1) the encoder output length is fixed (1,500 hidden states for Whisper), so encoder compute is predictable; (2) the decoder's cross-attention KV is shared across all decode steps (it never changes), so it can be allocated once at prefill; (3) batch sizes for speech are typically smaller (30-second audio chunks), making throughput optimization different from text; (4) streaming transcription requires chunked audio processing with overlap between chunks. A chapter would implement the Whisper serving path as a contrast to the VLM path — showing how cross-attention KV differs from self-attention KV and how both fit into the paged pool.

**Embedding Model Serving**
Embedding models (e5-mistral, BGE-M3, nomic-embed, sentence-transformers) use the same transformer architecture as generative LLMs but with a completely different inference contract: a single forward pass over the input, pool the last-token hidden state, L2-normalize it, and return a dense float vector — no decode loop, no KV cache, no streaming. Both vLLM (`entrypoints/pooling/`) and SGLang (`serving_embedding.py`, `LlamaEmbeddingModel`) serve `/v1/embeddings` natively alongside generative endpoints. The serving stack is simpler than generation — requests are fixed-cost, batching is straightforward, and the bottleneck is pure matrix-multiply throughput rather than memory bandwidth — but it introduces three new task types not covered elsewhere: *bi-encoder* (dense retrieval, one vector per text), *cross-encoder* (reranking, one scalar score per query–document pair, vLLM's `ServingScores`), and *late interaction* (ColBERT-style multi-vector retrieval, vLLM's `v1/pool/late_interaction`). A chapter would implement a minimal `/v1/embeddings` endpoint on top of the book's existing model, explain the pooling strategies (last token vs. mean vs. CLS), benchmark batch throughput vs. the generative path, and show where this fits in a RAG pipeline — compute embeddings here, store and search vectors in a separate system (Qdrant, pgvector, etc.).

**Hybrid Architecture Serving: SSM + Full Attention (Mamba, Jamba, Falcon-H1)**
Hybrid models interleave full attention layers (standard KV cache) with SSM/linear attention layers (Mamba, Mamba2, GDN, RWKV) — and the serving engine must manage two fundamentally different state types simultaneously. Full attention layers produce a KV cache that grows O(N) with sequence length. SSM layers maintain a *fixed-size recurrent state* — a `conv_states` sliding window and an `ssm_states` hidden state — that is constant in memory regardless of how many tokens have been generated; this is the architectural advantage that makes hybrid models viable at very long contexts. The engineering challenges that differ from pure-attention serving: (1) *dual state manager* — the KV pool and SSM state pool coexist; `HybridReqToTokenPool` in SGLang and `MambaSpec` alongside standard `KVCacheSpec` in vLLM's block pool track both; (2) *dispatch per layer* — `HybridLinearAttnBackend` routes each layer's forward call to either the full attention backend or the linear attention backend based on `layer_id in full_attn_layers`; (3) *prefix caching breaks for SSM layers* — you cannot simply reuse K,V tensors; instead you must save the recurrent SSM state at chunk-aligned boundaries (`_init_track_conv_indices`, `_init_track_ssm_indices`), requiring alignment to `FLA_CHUNK_SIZE`; SGLang implements `mamba_radix_cache.py` as a separate radix cache that tracks SSM states alongside attention KV; (4) *speculative decoding requires SSM rollback* — if K draft tokens are proposed and only 2 accepted, the SSM recurrent state must be rolled back to position 2; `update_mamba_state_after_mtp_verify` fuses this with a Triton scatter kernel to avoid multiple kernel launches; (5) *CUDA graph capture* must handle both state types, with SSM cache indices padded separately from attention sequence lengths. A chapter would implement a minimal hybrid model (alternating full attention and Mamba2 layers) on top of the book's transformer, add the dual state manager, show why prefix caching requires SSM state snapshots, and benchmark memory usage vs. a pure-attention model at long context lengths — demonstrating the O(1) vs O(N) memory difference concretely.

**Agents: Tool Calling, Sessions, and Multi-Turn KV Persistence**
Tool calling is *mostly* tokens from the model's perspective — the model generates `<tool_call>{"name": "search", "args": {...}}</tool_call>` and the client executes it — but agents create four genuine engine-level concerns that go beyond what the book's existing server handles. (1) *Session KV persistence* — in a normal request the KV blocks are freed at EOS; in an agent session the engine must keep those blocks alive between turns so the next turn inherits the full conversation KV without recomputation. SGLang implements this as a first-class feature: `OpenSessionReqInput`/`CloseSessionReqInput` open and close a session; a `SessionSlot` dataclass holds `req_pool_idx`, `kv_committed_len`, and `kv_allocated_len` across turns; `SessionAwareCache` wraps the radix prefix cache to prevent eviction of in-session KV blocks; the `SessionController` manages the slot lifecycle. Without this, every agent turn re-prefills the entire conversation history — at turn 10 with 5K tokens each that is 50K tokens of redundant prefill per request. (2) *Session branching for parallel tool calls* — modern agent frameworks (OpenAI Responses API, LangGraph) issue multiple tool calls in parallel from the same context; each branch inherits the parent's KV, executes independently, and then the merged results are injected into a new turn; `SessionReqNode` in SGLang builds a tree of branching session requests and tracks ref-counts so shared KV is not freed until all branches complete. (3) *Engine-side tool execution (MCP integration)* — vLLM goes further than "just tokens": `entrypoints/mcp/tool.py` and `mcp/tool_server.py` integrate Model Context Protocol directly into the serving stack so the engine itself manages tool dispatch, result injection, and the agentic loop without round-tripping to the client application; this changes the request lifecycle from a single generate call to a multi-step engine-managed conversation. (4) *Traffic pattern and scheduler impact* — agent traffic is bursty and session-affine: a session generates tokens, then goes idle while the tool runs (0.1s–30s), then re-arrives with a longer prompt; the scheduler must handle frequent re-admission of the same session context, and cache-aware routing must send the same session to the same instance to preserve prefix cache hits across turns. A chapter would implement session KV persistence on top of the book's paged scheduler — a `SessionStore` that pins KV blocks at EOS and releases them on explicit close — and benchmark the TTFT reduction across multi-turn agent conversations (turn 1 vs turn 5 vs turn 10), showing that without session persistence TTFT grows linearly with conversation depth, and with it TTFT stays near-constant.

**Reasoning Model Serving (DeepSeek-R1, QwQ, o1-style)**
Reasoning models (DeepSeek-R1, QwQ-32B, Skywork-o1) are standard autoregressive transformers — the model architecture requires no changes to serve. What changes is the *operating point* and the *API contract*: these models emit 5,000–30,000 thinking tokens before producing a short final answer, which stresses every subsystem in the book at 10–60× the normal scale. The serving differences worth a chapter: (1) *KV memory collapse* — a 30K-token reasoning trace consumes as much KV pool as 30 normal requests; with even modest concurrency (B=8) the paged pool fills instantly; tiered KV offload (CPU/NVMe) stops being optional and becomes mandatory, and the scheduler must cap concurrent reasoning requests separately from normal requests; (2) *thinking token parsing and filtering* — the model emits tokens inside `<think>…</think>` delimiters (DeepSeek-R1) or a `reasoning_content` field (OpenAI o1 API); the streaming path must detect the delimiter mid-stream and either suppress thinking tokens from the client response, buffer them separately, or forward them as a separate SSE event type; this is a non-trivial stateful transform on the token stream; (3) *budget forcing* — the serving API (and some model fine-tunes) support a `thinking_tokens` or `max_reasoning_tokens` parameter that caps the thinking phase by injecting a closing delimiter token when the budget is hit, then forcing the model into answer mode; implementing this requires monitoring per-request generated-token count and overriding the logit mask at the budget boundary; (4) *TPOT vs TTFT interpretation changes* — for a normal request TTFT is the latency to first visible token; for a reasoning request with suppressed thinking, TTFT is the latency to the first *answer* token which may be 20+ seconds of GPU time after the prompt; the metrics layer must distinguish between first-generated-token latency and first-answer-token latency; (5) *speculative decoding profile shifts* — reasoning traces contain repetitive structured text (mathematical notation, step numbering, code blocks) where n-gram matching and EAGLE acceptance rates are significantly higher than on open-ended chat, making spec decode especially effective; (6) *batch size collapses under load* — at 30K output tokens per request, a batch of 4 reasoning requests holds the KV pool longer than 120 normal requests would, starving the scheduler of capacity; priority-based preemption (recompute short requests, swap long ones) becomes the key throughput lever. A chapter would measure all of these effects concretely: run the existing server from Ch9 against a reasoning model, show KV pool exhaustion, implement thinking-token detection and stream filtering, add a budget-forcing flag, and demonstrate that n-gram spec decode cuts TPOT by 2–3× on reasoning traces specifically.

---

## Next Generation

*Topics that are either just arriving in production (2025–2026) or will define inference engineering in the next 2–3 years. These are not speculative — each has working open-source code today. They are listed here because they sit just beyond the current book's scope but are the natural "what comes next" for a reader who has finished all the previous chapters.*

---

**NVIDIA Dynamo: Datacenter-Scale Disaggregated Inference**
vLLM and SGLang are single-cluster inference engines. NVIDIA Dynamo (open-sourced at GTC 2025) is the layer above them: a datacenter-scale orchestration framework that treats GPU clusters as a fluid pool, routing prefill and decode work to whichever nodes are best suited at any moment. Key ideas absent from everything earlier in the book: (1) *disaggregation at fleet scale* — Dynamo's KV-aware router tracks the KV cache state of every prefill and decode worker across the entire fleet, not just within a single instance; routing a request to the worker that already holds the most prefix-matching KV blocks eliminates the cross-node KV transfer cost; (2) *dynamic GPU reallocation* — Dynamo can reassign GPUs between prefill pools and decode pools in response to load shifts without restarting any process; it achieves 30× throughput improvement on DeepSeek-R1 on Blackwell B200 GPUs by continuously optimizing the prefill/decode GPU ratio; (3) *the disaggregated transfer path* — Dynamo uses NIXL (NVIDIA Inference Xfer Library), a UCX-based RDMA transport for zero-copy KV block migration between nodes, which it coordinates at the fleet level; (4) *CUDA MPS / multi-process serving* — Dynamo can co-locate multiple small models on the same GPU using MPS (Multi-Process Service) to improve GPU utilization for heterogeneous model fleets; (5) *elastic expert parallelism at fleet scale* — for MoE models, Dynamo can replicate hot experts across multiple nodes and drain cold ones, with the EPLB rebalancer now operating on fleet-wide expert load statistics rather than per-instance. A chapter would position Dynamo relative to the single-instance serving built in this book, explain the architectural jump from "engine" to "orchestration framework," and walk through the KV-aware routing algorithm as a fleet-scale generalization of the single-instance cache-aware routing from Ch11.

**Blackwell Architecture: FP4, NVLink 6, and What Changes**
The H100 assumptions embedded throughout this book — 3.35 TB/s HBM3 bandwidth, 1,979 TFLOPS BF16, 900 GB/s NVLink 4 — change substantially on Blackwell (B200/GB200, shipping 2025–2026). The engineering implications reach every layer of the stack built in this book: (1) *FP4 native tensor cores* (~5,000+ TFLOPS for FP4 NVFP4 format on B200) — FP4 doubles FP8's throughput but requires model-level calibration more careful than FP8; the first FP4-native inference kernels (ThunderKittens 2.0, cuBLAS B200 configs) are already available; (2) *HBM3e* (8 TB/s on GB200 NVL72) — more than 2× H100's bandwidth, which shifts the memory-bound/compute-bound crossover point for attention and decode-phase GEMMs; the roofline chapter's numbers need updating; (3) *NVLink 6* (1.8 TB/s bidirectional per GPU, 14.4 TB/s for GB200 NVL72 rack) — cross-GPU all-reduce is now fast enough to reconsider TP degree decisions that were bandwidth-limited on H100; the all-reduce cost model from the TP chapters changes; (4) *GB200 NVL72 as a single unit* — 36 Grace CPUs + 72 Blackwell GPUs connected with NVLink switches form a rack that looks like a single 1.4 PB/s memory bandwidth machine; the "within-node vs cross-node" TP distinction partially dissolves; (5) *TMA and warpgroup MMA on Blackwell* — new SASS-level instructions that ThunderKittens 2.0 exploits to match cuBLAS performance on GEMM. A chapter would re-run every benchmark from the book on B200 hardware, identify which optimizations become more important (FP4 quantization, NVLink-aware all-reduce fusion), which become less important (cross-node TP avoidance), and quantify what a "Blackwell-first" inference engine would look like differently.

**Compilation-Based Inference: TensorRT-LLM and MLC-LLM**
Everything in this book is *kernel-library-based* inference: the engine calls handwritten or Triton-generated kernels (Flash Attention, paged attention, GEMM) assembled at runtime. Two major production systems take the opposite approach — *compilation-based* inference — and understanding why illuminates a fundamental design tension. TensorRT-LLM (NVIDIA) compiles the entire model computation graph into a TensorRT engine: operator fusion, memory layout optimization, and kernel selection happen at compile time (2–4 hours per model), producing a binary that runs 10–15% faster than the best library-based approach because the compiler can fuse across operator boundaries that library APIs cannot cross. MLC-LLM (Tianqi Chen, Apache TVM) takes this further: a hardware-agnostic tensor computation description (TIR, Tensor IR) is compiled to optimized kernels for any target — NVIDIA CUDA, AMD ROCm, Apple Metal, ARM NEON, WebGPU, WASM — enabling the same model to run on an H100 and an iPhone from the same source. The tradeoff: compilation-based systems are slower to support new models (new operators require new compilation passes) and produce less debuggable outputs than a stack of named Triton kernels. A chapter would implement the same attention + FFN forward pass in both paradigms — Triton kernels (as done throughout this book) and TVM TIR — profile both on the same hardware, and explain structurally why the 10–15% compilation advantage exists and when it is worth the engineering cost.

**AI-Generated Kernels: The Coming Paradigm Shift**
The most disruptive near-term development for everything this book teaches: LLMs are beginning to write GPU kernels that match or exceed hand-tuned human implementations. KernelAgent (PyTorch, 2025) uses a multi-agent LLM system to iteratively write, profile, and refine Triton kernels, achieving 89% of H100 roofline efficiency and 2× speedup over default `torch.compile` on targeted kernels. More strikingly, NVIDIA demonstrated (2025) that DeepSeek-R1 with inference-time compute scaling can generate custom attention kernels for new architectural variants (relative position encodings, multimodal cross-attention) that outperform hand-tuned implementations by skilled engineers. The engineering implications: (1) *kernel authoring is shifting from manual craft to specification + search* — the human specifies the correctness contract (input/output shapes, mathematical semantics, hardware target) and the LLM agent explores the optimization space; (2) *the bottleneck moves from writing kernels to evaluating kernels* — the chapter on profiling and roofline analysis (Ch4 of the sidebook) becomes more important, not less, because you need to know if an AI-generated kernel is actually correct and optimal; (3) *the Triton skill remains essential* — AI kernel generators produce Triton code; you must understand Triton to evaluate, debug, and extend what they produce; (4) *custom hardware variants are now tractable* — a new attention variant (sparse pattern, new positional encoding, new quantization scheme) that would take a human engineer weeks to kernel-optimize can now be explored in hours. A chapter would run KernelAgent on a kernel from this book (e.g., the RMSNorm+RoPE fused kernel from Ch8), evaluate the generated result against the human-written version, and discuss what this means for the future of inference kernel engineering.

**Serverless and Elastic Inference: Cold Starts, Live Migration, and Scale-to-Zero**
The inference engine this book builds is a persistent process: it loads model weights into GPU memory on startup and serves requests indefinitely. Production platforms increasingly need something different — engines that can spin up in under a second, migrate live KV state between GPUs without dropping requests, and scale to zero when idle. Three systems from OSDI 2024 define the state of the art: (1) *ServerlessLLM* — achieves 10–200× faster cold starts by treating model checkpoints as a paged memory object that can be streamed from NVMe to GPU incrementally; rather than loading all 14 GB of a 7B model before serving the first request, ServerlessLLM streams the most-accessed layers first and begins serving as soon as the attention layers are loaded; (2) *Llumnix* — implements live migration of in-flight requests (including their KV cache) between serving instances with sub-50ms pause time; this enables the scheduler to drain an instance for maintenance or rebalance load across a cluster without dropping active requests; (3) *vLLM's sleep/wake API* — covered in the MLOps chapter, but its full implication is a path to true scale-to-zero: an instance that has served no requests for 30 seconds offloads its weights to CPU DRAM (`sleep(level=2)`), then wakes in ~5 seconds when a new request arrives (weight reload only, not disk load). A chapter would implement a minimal serverless wrapper around the book's engine: a `FastLoad` checkpoint format that streams layers in priority order, a `LiveMigrate` protocol that serializes the paged KV pool and deserializes it on a target instance, and a controller that manages a fleet of sleep/wake instances under synthetic bursty traffic.

**Planetary-Scale and Heterogeneous Inference**
The inference systems in this book assume a homogeneous datacenter cluster of identical H100 GPUs connected by NVLink and InfiniBand. Two emerging realities break this assumption: (1) *consumer GPU inference* — Prime Intellect (2025) demonstrated serving DeepSeek-R1 across consumer RTX 4090 GPUs connected over the public internet with 100–300ms cross-node latency (vs ~1µs for InfiniBand), using pipeline parallelism to hide the communication latency; at this latency the ring all-reduce approaches from Ch25 become completely untenable — the entire communication model must change; (2) *heterogeneous hardware fleets* — production deployments increasingly mix H100, A100, L40S, and Blackwell GPUs in the same fleet, routing prefill (compute-bound, prefers H100/Blackwell) to high-compute nodes and decode (bandwidth-bound, tolerates L40S) to high-bandwidth-per-dollar nodes; MLC-LLM's hardware-agnostic compilation enables the same model binary to run on any GPU in the fleet. A chapter would characterize how each of the book's core assumptions (homogeneous hardware, NVLink connectivity, fixed cluster topology) breaks in the heterogeneous case, implement a pipeline-parallel decode path tolerant of 100ms inter-stage latency, and discuss the routing and scheduling changes required for mixed-GPU fleets.

**ThunderKittens: The Layer Between Triton and CUTLASS**
The GPU sidebook teaches Triton as the implementation language for all kernels. But there is a growing middle layer between Triton's block-level abstractions and CUTLASS's C++ template metaprogramming: ThunderKittens (Stanford Hazy Research, v2.0 February 2026). ThunderKittens is a CUDA-embedded DSL that exposes warp-level programming directly — warp tiles, warpgroup MMA instructions, TMA async copies — without the verbosity of CUTLASS but with more hardware control than Triton's compiler allows. ThunderKittens 2.0 achieves cuBLAS-matching GEMM performance on NVIDIA B200, supports MXFP8 and NVFP4 natively, and is used in production Flash Attention and state space model kernels. The design philosophy: write kernels at the warp level (4 warps × 32 threads = 128 threads per warp group), express TMA copies explicitly, and describe warpgroup MMA tiles as typed `rt_bf16<16, 64>` register tiles rather than generic blocked pointers. A chapter would port two kernels from the GPU sidebook — the tiled GEMM (Ch11) and Flash Attention forward (Ch16) — to ThunderKittens, profile both against the Triton versions, and explain exactly where ThunderKittens gains (explicit TMA + warpgroup MMA scheduling, B200-native types) and where Triton wins (faster iteration, better portability, lower barrier to entry).
