# KV Cache Quantization — Combined Reference

**What this file is:** A synthesis of all L2, L3, and L4 reference material on KV cache quantization, combined into a single progressive narrative. The reading order moves from "why does KV cache quantization matter?" → "why is it harder than weight quantization?" → "what does the element distribution look like and why does it matter?" → "the scale granularity spectrum" → "production state in SGLang and vLLM" → "research frontier (sub-FP8, compute quantization)" → "quantization vs token dropping — which wins where" → "composability with HiCache, weights, and architecture" → "open problems."

**Sources synthesized:**
- L2/01 — KIVI NeurIPS 2024 (arXiv 2402.02750): abstract-level K/V distribution asymmetry and 2-bit results
- L2/02 — vLLM KV quantization source code: formats, KVQuantMode, CUDA kernel, extended scales, TurboQuant, sleep-mode bug
- L3/01 — KIVI NeurIPS 2024 (arXiv 2402.02750): full algorithmic detail, distribution study, results
- L3/02 — KVQuant NeurIPS 2024 (arXiv 2401.18079): per-channel K, pre-RoPE quantization, non-uniform datatypes, per-vector sparse
- L3/03 — ZipCache ECCV 2024 (arXiv 2405.14256): salient token identification, mixed-precision, channel-separable tokenwise
- L3/04 — SageAttention2 ICML 2025 (arXiv 2411.10958): attention compute quantization (INT4 QK, FP8 PV), orthogonality with storage
- L4/01 — Compression benchmark EMNLP 2024 (arXiv 2407.01527): quantization vs token dropping across 7 task categories
- L4/02 — KV cache management survey arXiv 2024 (arXiv 2412.19442): full taxonomy, MLA/GQA interaction, HiCache interaction, open problems
- L4/03 — LLM inference survey arXiv 2024 (arXiv 2408.03130): weight vs activation quantization hardness, accuracy-efficiency frontier

---

## 1. The Problem: KV Cache Dominates VRAM at Scale

KV cache grows with every token in every active request:

```
KV_bytes = num_layers × 2 × num_kv_heads × head_dim × num_tokens × bytes_per_element
```

For a representative production model (Llama-3.1-70B, GQA-8, 80 layers, 8 KV heads, head_dim=128) serving a batch of 32 requests at 8K context:

| Storage format | Bytes per token | Total KV for batch=32/ctx=8K |
|---|---|---|
| BF16 (2 bytes) | 327,680 bytes ≈ 0.32 MB | ~82 GB |
| FP8 (1 byte) | 163,840 bytes ≈ 0.16 MB | ~41 GB |
| FP4 (0.5 bytes) | ~0.08 MB | ~20 GB |

At BF16, a single 8×H100 node (640 GB) minus ~140 GB for model weights leaves ~500 GB for KV. That same 82 GB workload uses 16% of budget — but at batch=256 it is impossible. FP8 halves that constraint.

There is a second, subtler problem: **the decode step is memory-bandwidth-bound**. Each decode step loads the entire KV cache for all active tokens. On H100 (3.35 TB/s HBM bandwidth), loading 82 GB of KV takes ~24 ms per token. FP8 halves this to ~12 ms — a throughput improvement with no additional compute, just fewer bytes loaded.

The KIVI paper (NeurIPS 2024) frames the dual problem clearly: KV cache is the bottleneck in both **memory capacity** (fitting large batches) and **memory bandwidth** (loading all KV per decode step). GPU compute units idle while waiting for KV data — quantization solves a utilization problem, not just a capacity problem.

---

## 2. Why KV Quantization Is Fundamentally Harder Than Weight Quantization

The LLM inference survey (arXiv 2408.03130) provides the key conceptual framing that explains why the engineering of KV quantization is difficult.

**Weight quantization** operates on static tensors. Weights `W` are fixed after training. You calibrate once on a representative dataset, compute a per-layer scale that minimizes quantization error on calibration data, and fix that scale at inference time. The scale is always correct because `W` never changes.

**KV cache quantization is activation quantization**. The KV values `K`, `V` depend on the input tokens. Different inputs produce different K and V distributions. A single calibrated scale may not fit all possible inputs — especially problematic for:
- Out-of-distribution inputs (a scale calibrated on English text may be too tight for code inputs where K channels have larger dynamic ranges)
- Very long contexts (statistics shift over 128K+ tokens)
- Unusual prompts (math, foreign languages, specialized domains have different distributions)

This explains why SGLang's `--quantization-param-path` (static per-tensor scales) works well for models with stable activation ranges on typical inputs, but can cause accuracy degradation for domain-shifted or long-context workloads where the calibrated scale undershoots the actual activation range.

**The three engineering responses to this hardness:**

| Approach | Engine | Calibration | Tradeoff |
|---|---|---|---|
| Calibrated per-tensor scale (static) | SGLang, vLLM | **Required** (or ships with FP8 checkpoint) | Simple, zero runtime overhead; distribution-dependent; may fail on OOD inputs |
| Dynamic per-token-head scale | vLLM `fp8_per_token_head` | **Not needed** | More robust; ~3% scale storage overhead; computed at write time |
| Per-channel / per-vector (calibrated) | KIVI, KVQuant (research) | Mixed (KIVI: none; KVQuant: yes) | Best accuracy at sub-FP8; requires custom CUDA kernels not in production engines |

---

## 3. The K/V Distribution Asymmetry: Why Granularity Matters More Than Bit Width

KIVI (NeurIPS 2024, arXiv 2402.02750) ran the first comprehensive study of KV activation statistics across LLaMA, Falcon, and Mistral models. The results completely changed how KV quantization should be designed.

### K cache: channel-wise outliers

Plot K[head, token, :] as a heatmap across token positions. You see **vertical stripes** — specific feature dimensions (channels in `head_dim`) that consistently produce values 10–100× larger than the median, **across all tokens and all inputs**:

```
K distribution (schematic):
Channel 3:  [0.1, 0.2, 0.1, 0.2, ...]  ← normal, stable
Channel 7:  [8.2, 8.5, 8.1, 8.4, ...]  ← outlier channel (same position, all tokens)
Channel 64: [0.3, 0.2, 0.4, 0.3, ...]  ← normal, stable
```

These outlier channels are a **structural property of the model weights**, not random noise. They appear at the same channel indices regardless of input. This means:
- A **per-tensor scale** must accommodate channel 7's ±8.5 range → channel 3's ±0.2 values are quantized with step size 8.5/448 ≈ 0.019 — very coarse for such small values
- A **per-channel scale** lets channel 7 use 8.5/448 and channel 3 use 0.2/448 = 0.00045 — maximally accurate for both

### V cache: token-wise variation

Plot V[head, token, :] the same way. You see **horizontal stripes** — different tokens have different overall magnitudes, but within each token the distribution across channels is relatively uniform. No persistent per-channel outliers.

→ **V should be quantized per-token** (one scale per token covers the token's uniform range). Per-channel quantization adds overhead without benefit for V.

### The implication: per-tensor FP8 is the wrong granularity for both

Both SGLang (`--kv-cache-dtype fp8_e4m3`) and vLLM (`fp8_e4m3`) use **per-tensor FP8** — one scale per attention layer. This is the pragmatic starting point. KIVI's insight is that this granularity is suboptimal for K (too coarse: outlier channels inflate the scale) and unnecessary for V (per-token would work just as well).

The striking result: at **2-bit INT** with per-channel K and per-token V, KIVI achieves near-identical quality to BF16, with 2.6× memory reduction and 2.35–3.47× throughput. The bottleneck was never the number of bits — it was the granularity.

KVQuant (NeurIPS 2024) confirms this by pushing to **3-bit with < 0.1 perplexity degradation** using per-channel K plus additional techniques. The gap between per-tensor FP8 and the research frontier is primarily a granularity gap, not a hardware limitation.

---

## 4. The Scale Granularity Spectrum

Synthesizing across all references, there is a clear progression:

```
Coarser → → → → → → → → → → → → → → → → → → → → → → → Finer

Per-tensor     Per-head    Per-token    Per-token-head    Per-channel    Per-vector    Per-thread
(one per       (one per    (one per     (one per          (one per       (one per      (one per
 attn layer)    head)       token)       token×head)       head_dim)      group of N    thread tile)
                                                                          elements)

SGLang                                  vLLM dynamic                     KIVI K        SageAttention2
calibrated                              (fp8_per_token_head)             KVQuant K     (compute path)
(default)                                                                 KVQuant V
                                                                          (per-vector)
```

**Scale storage cost** grows from the left (~0 bytes) to the right (potentially 32%+ of KV memory for per-vector). **Accuracy** improves moving right. The production-research gap is visible: production engines occupy the left (per-tensor, per-token-head); research systems occupy the right (per-channel, per-vector).

**Why per-thread (SageAttention2) is finer than per-channel:** Each GPU thread block handles 16–64 elements of Q or K. The scale covers only that block's elements — this is the finest granularity physically possible without per-element scales. Thread-level quantization enables INT4 for Q×K matmuls with negligible accuracy loss.

### Calibration Required vs Not Required — by Granularity

The scale source (static calibrated vs dynamic computed) is independent of granularity but critical for deployment:

```
Scale source:
  Static / calibrated ────────────────────────────── Dynamic / online

  SGLang per-tensor         KVQuant per-channel     KIVI per-channel   vLLM per-token-head
  (--quantization-param-    (requires calibration   (computed from     (computed from actual
   path required for         dataset + custom        actual values,     values at write time,
   best accuracy)            CUDA kernels)           no dataset)        no dataset)
```

| | Calibration needed? | When scales are computed | Who does it |
|---|---|---|---|
| SGLang `fp8_e4m3` + auto | **No** — included in checkpoint | Offline, by model publisher | Model provider (HuggingFace) |
| SGLang `fp8_e4m3` + `--quantization-param-path` | **Yes** — you run it | Offline, ~30 min on 512 samples | You (llm-compressor / AWQ) |
| SGLang `fp8_e4m3` with no path | **No** — but scale=1.0 ⚠️ | Never — hardcoded | Nobody; accuracy risk |
| vLLM `fp8_per_token_head` | **No** | At inference, per write | Engine (fused in CUDA kernel) |
| vLLM `int8_per_token_head` | **No** | At inference, per write | Engine (fused in CUDA kernel) |
| vLLM `fp8_e4m3` (per-tensor) | Same as SGLang | Same as SGLang | Same as SGLang |
| KIVI 2-bit (research) | **No** | At inference, from actual values | Research kernel |
| KVQuant 3-bit (research) | **Yes** — small dataset | Offline, per-layer sensitivity | You (custom scripts) |
| ZipCache (research) | **No** | At inference, saliency per token | Research kernel |
| TurboQuant (vLLM only) | Unclear / checkpoint-bound | Depends on format | NVIDIA partner tooling |

---

## 5. Production State: SGLang vs vLLM

### Calibration Quick Reference

Before diving into implementation, the single most common question: **do I need to run calibration?**

```
Do you have an FP8 checkpoint from HuggingFace?
  (e.g. meta-llama/Meta-Llama-3.1-70B-Instruct-FP8, neuralmagic/*, Qwen/*-FP8)
         │
         ▼ Yes
  --kv-cache-dtype auto           ← No calibration needed. Scales ship with the model.
  ✅ Recommended. Best accuracy. Zero extra steps.

         │ No (you have a BF16 base model)
         ▼
  Do you want best accuracy?
         │
         ▼ Yes
  Run llm-compressor on 512 samples (~30 min on 8×H100)
  --kv-cache-dtype fp8_e4m3 --quantization-param-path /path/to/scales.json
  ✅ Recommended for production with BF16 base models.

         │ No / convenience / testing
         ▼
  --kv-cache-dtype fp8_e4m3       ← scale=1.0 fallback, logged as WARNING
  ⚠️  Acceptable for typical inputs. Accuracy risk for code, math, multilingual.

Are you on vLLM and have diverse inputs (multi-language, code+text)?
  --kv-cache-dtype fp8_per_token_head   ← No calibration. Dynamic per write.
  ✅ Recommended when calibration dataset is unavailable or inputs vary widely.
```

### Three-line summary

- **Calibrated scales from checkpoint** (`--kv-cache-dtype auto`): best accuracy, zero setup, requires an FP8 checkpoint.
- **Calibrated scales you generate** (`--quantization-param-path`): best accuracy on any checkpoint, requires running llm-compressor once.
- **Dynamic scales** (vLLM `fp8_per_token_head`): no calibration, adapts at inference time, ~3% scale storage overhead, only in vLLM.
- **No scales** (`fp8_e4m3` with no path): convenient, works for many models, accuracy risk on diverse or OOD inputs.

---

### SGLang: per-tensor calibrated FP8

Three relevant CLI flags and the pipeline they drive:

```
--kv-cache-dtype fp8_e4m3       # or "auto" to read config.json
--quantization-param-path ...   # optional JSON with per-layer k_scale/v_scale
```

Write path: `set_kv_buffer()` performs `BF16 → div_(k_scale) → .to(fp8) → .view(uint8) → index_put_()`. The uint8 workaround exists because PyTorch's `index_put_` is not implemented for float8 dtypes. FlashInfer/TRT-LLM dequantize inside the attention kernel using `bmm1_scale`. Scale=1.0 if no calibration path is provided — a known accuracy risk logged as a warning.

### vLLM: 13 formats, per-token-head dynamic, end-to-end FP8

The `KVQuantMode` enum (`NONE`, `FP8_PER_TENSOR`, `INT8_PER_TOKEN_HEAD`, `FP8_PER_TOKEN_HEAD`, `NVFP4`) provides clean dispatch. Key extensions beyond SGLang:

**Per-token-head dynamic:** scales computed at cache-write time from the actual token's activation values. No calibration needed. Scale storage adds `2 × block_size × num_kv_heads × 4 bytes` per page (~3% overhead for typical configs). The vLLM docs note this as the "most practical improvement" — no offline step, adapts to any input distribution.

**q_scale and prob_scale:** Beyond K and V storage, vLLM attaches scales to the Q-projection output and softmax probabilities, enabling the attention matmuls themselves to run in FP8 (not just storage). SGLang has only `k_scale`/`v_scale`.

**Fused CUDA kernel:** `reshape_and_cache_flash` fuses scale application + cast + scatter in one kernel. The `kv_scale_stride` argument dispatches between per-tensor (`stride=0`) and per-head (`stride=1`) scale shapes. SGLang's write path is Python-level (`div_()` + `.to()` + `.view()`).

**TurboQuant (NVIDIA Research):** `turboquant_k8v4` (K=INT8, V=INT4) and `turboquant_3bit_nc` (3-bit non-contiguous). Automatically skips the first 2 and last 2 attention layers (boundary layers have atypical distributions). Not in SGLang.

**Known bug:** `init_fp8_kv_scales()` resets all per-layer scales to 1.0 after a GPU wakes from idle sleep — calibrated checkpoint scales are lost until restart. A production correctness risk for deployments using `--quantization-param-path`.

### Quick comparison

| Feature | SGLang | vLLM |
|---|---|---|
| Per-tensor static FP8 | ✓ | ✓ |
| Per-token-head dynamic FP8 | — | ✓ |
| Per-token-head dynamic INT8 | — | ✓ |
| NVFP4 (Blackwell) | ✓ (experimental) | ✓ (WIP) |
| TurboQuant K8V4 / 3-bit | — | ✓ |
| q_scale (Q compute path) | — | ✓ |
| prob_scale (P compute path) | — | ✓ |
| MLA FP8 (DeepSeek) | ✓ (per-tile scales) | ✓ |
| Fused CUDA write kernel | Partial (TRT-LLM path) | ✓ |
| Sleep-mode scale reset bug | No | Yes |

---

## 6. The Pre-RoPE Gap: KVQuant's Hidden Accuracy Lever

KVQuant (NeurIPS 2024, arXiv 2401.18079) introduces four techniques for sub-4-bit quantization. Three of them (per-channel K, non-uniform datatypes, per-vector sparse) build on KIVI. The fourth — pre-RoPE K quantization — is the unique contribution with direct implications for production systems.

**The problem:** Rotary Positional Embedding (RoPE) rotates the Key vector by a position-dependent angle:
```
K_final = RoPE(W_K × input, position)
```
This rotation mixes the channel structure of K. The per-channel outlier pattern that exists in `W_K × input` is scrambled by the rotation → after RoPE, the same channel index no longer corresponds to the same semantic feature → per-channel statistics become position-dependent → calibrated per-channel scales become inaccurate over long contexts.

**KVQuant's solution:** Quantize K before RoPE is applied:
```
Standard:  input → W_K → RoPE → K_final → cache(K_final)    (post-RoPE cache — current production)
KVQuant:   input → W_K → K_pre → cache(quant(K_pre))         (pre-RoPE cache)
           at attention time: dequant(K_pre) → RoPE → compute Q×K^T
```

Pre-RoPE K has stable per-channel statistics across all token positions → per-channel scales calibrated once remain accurate throughout the context. Result: < 0.1 perplexity degradation at **3-bit** on Wikitext-2 and C4, enabling **1 million token contexts on a single A100-80GB**.

**The production gap:** Both SGLang and vLLM cache **post-RoPE K**. Implementing pre-RoPE caching requires splitting `apply_rotary_pos_emb()` from the KV cache write — the K projection goes to cache before rotation, and the rotation is applied at attention time on the dequantized K. This is a non-trivial change to the attention module forward pass. Neither engine has done this yet.

For FP8 per-tensor (the current production approach), this gap is less critical because the per-tensor scale calibration implicitly handles some of the position-dependency. For sub-4-bit quantization, pre-RoPE K is nearly mandatory for acceptable accuracy.

KVQuant also introduces **non-uniform per-layer datatypes** (minimizing a second-order sensitivity metric per layer) and **per-vector dense-and-sparse** (isolating the top-k magnitude outliers in FP16, quantizing the rest tightly). These enable 3-bit K + 4-bit V with near-identical quality to FP16.

**Calibration requirement:** KVQuant requires a small calibration dataset (~128–512 samples) to compute the per-layer non-uniform quantization grids. Significantly more involved than FP8 per-tensor calibration — custom scripts and CUDA kernels required; not a one-command workflow like llm-compressor.

---

## 7. The Research Frontier: Sub-FP8 Techniques

### KIVI (2-bit): the asymmetric algorithm

Beyond the distribution insight, KIVI's practical algorithm handles the fact that channel statistics need some tokens to stabilize. It maintains a residual buffer of the last `group_size` tokens in FP16, quantizing the older history to INT2:

```
Active context:
  [tokens 0..T-g-1] → INT2 quantized (per-channel K, per-token V)
  [tokens T-g..T-1] → FP16 residual (group_size = 32 or 64)
```

When the residual window advances, the oldest residual tokens are quantized and merged. During attention, INT2 history is dequantized and concatenated with FP16 residual before the Q×K^T matmul. **Calibration: not needed** — scale/zero are computed online from the actual token values at inference time.

Results: 2.6× memory reduction, 2.35–3.47× throughput, near-identical quality on LongBench, MMLU, HumanEval. KIVI-2 outperforms many token-dropping methods at the same memory budget because it retains **all** tokens — just at lower precision.

### ZipCache: mixed-precision via salient token identification

ZipCache (ECCV 2024, arXiv 2405.14256) occupies a unique design point between pure quantization and pure token dropping. Rather than quantizing all tokens uniformly, it identifies "salient" tokens and treats them differently:

1. **Saliency metric:** normalized attention score that accounts for causal mask structure — token i's saliency at position j = `softmax(Q_j × K_i^T) × (j+1)`. The normalization removes the bias toward early tokens that naive attention scores have.
2. **Mixed-precision layout:** salient tokens (score > adaptive threshold) kept at FP16; non-salient tokens quantized to INT2 or INT4.
3. **FlashAttention compatibility:** the metric can be approximated without computing the full O(seq²) attention matrix — compatible with memory-efficient attention backends.
4. **Channel-separable tokenwise quantization:** instead of per-group scales (one scale per G elements, overhead = 2 × total/G), it factorizes scales as `scale_total = scale_channel × scale_token` with overhead of `num_tokens + head_dim` parameters (~4× smaller than groupwise for typical sizes).
5. **Calibration: not needed** — saliency and scales are computed dynamically at inference from the current attention scores. No offline dataset required.

Result: 4.98× compression with 0.38% accuracy drop on Mistral-7B/GSM8K; 56.9% decode latency reduction (from fewer bytes loaded per decode step).

ZipCache's ideas are not in production engines yet, but they point at two future directions: (a) integrating saliency-aware precision into paged attention to decide which pages to keep in VRAM vs offload to HiCache, and (b) adapting the channel-separable scale format to reduce vLLM's per-token-head scale storage overhead.

### SageAttention2: quantizing the attention computation itself

SageAttention2 (ICML 2025, arXiv 2411.10958) addresses an orthogonal problem. KV storage quantization (Sections 1–7) saves memory by storing K and V in fewer bits. **Attention compute quantization** saves FLOPS by running the attention matmuls in fewer bits:

```
Storage quantization:  store K, V in FP8 (1 byte each) → 2× smaller KV cache
Compute quantization:  compute Q×K^T in INT4 → 2× more FLOP throughput

Both are orthogonal:
FP8 K/V storage + INT4 Q×K compute = memory saving + throughput improvement simultaneously
```

SageAttention2's three technical contributions:
1. **Per-thread INT4 for Q and K:** each warp's thread block (16–64 elements) gets its own scale — finer granularity than per-token, enabling INT4 without unacceptable rounding
2. **Q smoothing:** per-channel scale applied to Q before INT4 cast — same idea as KIVI's per-channel treatment of K storage
3. **Two-level FP8 accumulation for P×V:** partial sums in FP8 within each warp, reduce in FP32 — matches FP16 accumulation accuracy while using FP8 arithmetic for the bulk of computation

Result: ~3× faster than FlashAttention2 on RTX 4090; matches FlashAttention3 FP8 on H100 with better accuracy.

**The vLLM connection:** vLLM's `q_scale` and `prob_scale` parameters on `BaseKVCacheMethod` are exactly the production-path implementation of this compute quantization concept — scaling Q before Q×K^T and scaling softmax(P) before P×V. SGLang has not added these yet.

---

## 8. Quantization vs Token Dropping: Which Wins Where

The compression benchmark (EMNLP 2024, arXiv 2407.01527) is the only controlled study that compares KV quantization against every other KV compression approach — token dropping (H2O, SnapKV, StreamingLLM), prompt compression, linear-time models, and hybrid architectures — across 7 task categories on the same base models.

The headline finding: **there is no Pareto-optimal method across all tasks.**

### Quantization wins on retrieval tasks

On synthetic retrieval tasks (PassageRetrieval) that require the model to recall any token from the context:
- KV quantization (KIVI-2): minimal degradation — all tokens preserved, just at lower precision
- Token dropping (H2O, SnapKV): significant degradation — the dropped tokens may be exactly the ones needed

**Why:** Retrieval requires attending to tokens that receive low attention weight during prefill (they are not "important" by attention-weight metrics) but are critical for the final answer. Quantization is safe because it retains all tokens; dropping discards them permanently.

### Token dropping wins on summarization

On tasks where attention consistently concentrates on a small subset of tokens (summarization, few-shot classification):
- Token dropping: high retention with aggressive compression
- Quantization: needs very low bit-width to achieve the same memory reduction

**Why:** If 90% of attention weight concentrates on 20% of tokens, keeping only those tokens preserves nearly all the information. Quantizing all tokens equally wastes bits on tokens that barely contribute to the output.

### The practical decision table

| Workload | Recommended approach | Reasoning |
|---|---|---|
| Long-context RAG / document QA | FP8 quantization | Retrieval requires all tokens |
| Multi-turn chat | FP8 quantization | Which tokens matter next turn is unpredictable |
| Coding agent (long sessions) | FP8 quantization + HiCache | All history needed; HiCache handles capacity |
| Summarization at scale | Token dropping | Attention concentrates; dropping non-salient is safe |
| Streaming / infinite context | Token dropping (sliding window) | Cannot keep all context at any bit width |
| Maximum safety across all tasks | FP8 quantization + HiCache | No retrieval failures; 2× compression + tier capacity |

Small compression ratios (2–4×) are safe for all methods. Aggressive compression (8–16×):
- Quantization: degrades gracefully (all tokens retained at lower precision)
- Token dropping: catastrophic failure on retrieval tasks

---

## 9. Composability: How Quantization Multiplies with Other Optimizations

The KV cache management survey (arXiv 2412.19442) provides the most comprehensive treatment of how quantization interacts with the rest of the system stack.

### Composability with HiCache (Layer 17)

FP8 KV quantization and HiCache tiered storage are orthogonal: HiCache moves tensors across tiers; quantization makes each tensor smaller. Combining them gives multiplicative benefits:

- **VRAM:** 2× more tokens in GPU pool (FP8 vs BF16)
- **PCIe bandwidth:** 2× faster tier-2 loads (FP8 tensors are half the size)
- **CPU DRAM:** 2× more tokens in CPU pinned buffer
- **Storage I/O:** 2× faster reads/writes to tier-3 NVMe or remote storage

The κ_crit threshold (from the PCIe bottleneck analysis in Layer 17 references) effectively doubles with FP8: for the same PCIe bandwidth, you can offload 2× more tokens before the system becomes memory-bandwidth-bound.

### Composability with model architecture (GQA, MLA)

Modern model architectures already reduce KV size before quantization:

| Model | Architecture | KV reduction vs Llama-2 MHA @ BF16 |
|---|---|---|
| Llama-3.1-70B | GQA-8, BF16 | 8× smaller |
| Llama-3.1-70B | GQA-8, FP8 | **16× smaller** |
| DeepSeek-V3-671B | MLA, BF16 | ~60× smaller |
| DeepSeek-V3-671B | MLA, FP8 | **~120× smaller** |

GQA (Grouped Query Attention) reduces `num_kv_heads` — Llama-3's GQA-8 uses 8 KV heads instead of 64 query heads. MLA (Multi-head Latent Attention, DeepSeek) compresses KV into a low-rank latent vector `C` with `kv_lora_rank << head_dim × num_heads`. Applying FP8 quantization to either gives another 2× on top of the architectural reduction.

For MLA + FP8, SGLang's `MLATokenToKVPoolHost` uses a different path: per-tile FP32 scales embedded in each cache page alongside the FP8 data (`nsa_kv_cache_store_fp8`). The granularity is finer than the MHA per-layer-tensor approach.

### Composability with weight quantization

Weight quantization and KV cache quantization are independent (different tensors, different calibration pipelines):

| Config | VRAM for weights | VRAM for KV | Accuracy risk |
|---|---|---|---|
| BF16 weights + BF16 KV | 100% | 100% | None (baseline) |
| BF16 weights + FP8 KV | 100% | 50% | Low (calibrated) |
| FP8 weights + FP8 KV | ~50% | 50% | Low |
| GPTQ INT4 + FP8 KV | ~25% | 50% | Moderate |
| FP8 weights + KIVI 2-bit KV | ~50% | ~12.5% | Research; monitor |

AWQ pipelines collect KV activation statistics as a byproduct of weight quantization — the `quantization_param_path` JSON files are often generated alongside AWQ or llm-compressor weight quantization, making joint weight + KV quantization practical.

### Composability with attention compute (SageAttention2)

The fully quantized attention stack:
```
1. Compute K, V at BF16                           (model forward pass)
2. Store K, V as FP8                              (storage quantization — Layer 18)
3. At decode step: load K_fp8, V_fp8 from pool
4. INT4(Q) × INT4(K_fp8) → attn logits           (compute quantization — SageAttention2)
5. softmax → FP8(P) × FP8(V_fp8) → BF16 output  (compute quantization — SageAttention2)
```

Memory saving from step 2 and compute speedup from step 4 are independent. Combined: half the KV bandwidth (FP8 storage) and twice the attention FLOP throughput (INT4 compute).

---

## 10. The Full Quantization Taxonomy

From the KV cache management survey, KV quantization is token-level optimization in a three-level system:

```
KV Cache Management
├── Token-level
│   ├── KV cache selection (token dropping: H2O, StreamingLLM, SnapKV)
│   ├── Budget allocation (PyramidKV, RazorAttention)
│   ├── KV merging (KVMerger, CLA)
│   ├── KV quantization  ← Layer 18's focus
│   │   FP8 (production) → 4-bit (approaching) → 2-3-bit (research)
│   └── Low-rank decomposition (LoRA-style K/V factorization)
│
├── Model-level
│   ├── GQA / MQA (reduce num_kv_heads architecturally)
│   ├── MLA (DeepSeek-style latent compression)
│   └── Sliding window, sparse attention
│
└── System-level
    ├── PagedAttention / RadixAttention (memory management — Layer 12)
    ├── HiCache / LMCache (multi-tier storage — Layer 17)
    └── PD disaggregation (Layer 19)
```

The survey assesses precision tiers in 2025:
- **FP8**: Production-ready. Per-tensor (SGLang) and per-token-head (vLLM) both deployed.
- **4-bit**: Approaching production. TurboQuant (vLLM), NVFP4 (WIP). Research: WKVQuant, QJL, Gear.
- **2–3-bit**: Research stage. KIVI (2-bit), KVQuant (3-bit). Hardware-efficient kernels needed.
- **1-bit and below**: Research only. KVSharer, MiKV. Significant quality degradation.

---

## 11. Open Problems

The survey identifies five open problems that will drive the field in 2025–2027:

1. **Hardware-efficient sub-4-bit kernels:** Custom CUDA dequantization for INT2/INT3 that matches FP8 kernel speeds. KIVI and KVQuant have proof-of-concept kernels; production integration into FlashInfer or CUTLASS is missing.

2. **Pre-RoPE quantization in production:** KVQuant demonstrates pre-RoPE K quantization achieves significantly better accuracy at sub-4-bit. Both SGLang and vLLM cache post-RoPE K — changing this requires splitting `apply_rotary_pos_emb()` from the cache write in the attention module. No production engine has done this yet.

3. **Quantization-aware training (QAT) for KV cache:** All current methods are post-training quantization (PTQ). QAT for KV activations — fine-tuning the model to be robust to KV quantization noise — could enable more aggressive bit widths with no accuracy loss.

4. **Unified framework:** Combining quantization + token dropping + model architecture changes in one optimizable system. ZipCache's mixed-precision approach is a step in this direction but requires per-token precision decisions at runtime.

5. **Multimodal KV quantization:** Vision tokens (ViT patches, image embeddings) have fundamentally different statistical distributions from text tokens. Per-tensor scales calibrated on text are poorly suited for vision-language models. Specialized vision-aware quantization is an open problem.

---

## Key Quotes

> "KV cache grows linearly with batch size × context length. At larger batch sizes, KV cache exceeds model weight memory, and loading KV cache makes GPU compute units idle — a utilization problem, not just a memory problem." — KIVI paper

> "The outliers appear along the channel dimension [in K] and are stable across different tokens and inputs... This means K must be quantized per-channel. The V cache shows a smoother distribution that can be quantized per-token." — KIVI paper (the canonical asymmetry result)

> "If you quantize K after RoPE, the rotary transform has rotated the elements, mixing the channel structure. Quantizing K before RoPE gives more stable per-channel statistics across token positions." — KVQuant paper

> "KV cache quantization is a special case of activation quantization — fundamentally harder than weight quantization because activations change per input, weights don't." — LLM inference survey

> "There is no Pareto-optimal method across all tasks. Quantization is the safer choice for retrieval tasks; token dropping is more efficient when attention patterns are predictable." — Compression benchmark

> "GQA-8 + FP8 KV = 16× smaller KV cache than MHA + BF16. MLA + FP8 = ~120× smaller." — KV cache management survey

---

## Appendix: What Is Left Out and Why

### Left out: Linear-time models (Mamba, RetNet, Jamba)

The compression benchmark evaluates linear-time models alongside quantization. These are excluded here because they represent an architectural replacement for attention — they have no KV cache to quantize. They are a competing paradigm, not a quantization technique.

### Left out: KIVI L2 summary (merged into Section 3 and 7)

The L2/KIVI file covers the same content as L3/KIVI at a higher level. All unique content from L2 is incorporated into Sections 3 (asymmetry insight) and 7 (2-bit algorithm). The L2 file's "Why This Matters for SGLang/vLLM FP8 KV" section is absorbed into the granularity discussion in Sections 3 and 5.

### Left out: Detailed ZipCache benchmark methodology

ZipCache's specific benchmark numbers (37.3% prefill latency reduction, 19.8% GPU memory reduction on LLaMA3-8B at seq_len=4096) are partially covered in Section 7 (56.9% decode latency). The additional numbers are excluded because ZipCache is research-stage and the precise figures may not generalize across serving workloads.

### Left out: GPTQ, AWQ, GGUF weight quantization details

The LLM inference survey covers GPTQ and AWQ at length. These are **weight quantization** methods, upstream of KV cache quantization. Their interaction with KV quantization (GPTQ INT4 weights + FP8 KV cache, AWQ generating quantization_param_path as a byproduct) is covered in Section 9. The internal GPTQ/AWQ algorithms are excluded — they are not KV cache methods.

### Left out: Knowledge distillation and structured pruning

The LLM inference survey's taxonomy includes knowledge distillation and structured/unstructured pruning. These reduce model size or quality to improve inference efficiency through a different mechanism entirely (training a smaller model, removing heads/layers). Not relevant to KV cache quantization.

### Left out: Specific vLLM `CacheDType` string enumeration

The full list of 13 `CacheDType` strings from vLLM's `cache.py` (including variants like `fp8_inc`, `turboquant_k3v4_nc`, `turboquant_4bit_nc`) is covered in Section 5's comparison table at the level needed for design decisions. The full enumeration adds detail not needed for understanding the architecture.

### Left out: WKVQuant, QJL, Gear, KVSharer, MiKV

The KV cache management survey mentions these 4–8-bit quantization methods. They are research-stage with no production implementations in SGLang or vLLM. Their existence is acknowledged in the taxonomy table in Section 10. Detailed coverage would require paper-level treatment that goes beyond the scope of production-oriented synthesis.
