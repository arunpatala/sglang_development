# KV Cache Quantization — Omitted Material

**What this file is:** The full text of every section that was omitted from `COMBINED.md`. The appendix of `COMBINED.md` names each omission and briefly explains why it was excluded. This file preserves the complete original text so no source material is lost.

**Sources:** L2/01, L2/02, L3/01, L3/02, L3/03, L3/04, L4/01, L4/02, L4/03

---

## Omission 1: KIVI L2 — "Why This Matters for SGLang/vLLM FP8 KV" and Takeaways

**Source:** `L2/01_kivi_overview.md`
**Why omitted from COMBINED.md:** The L2 overview file covers the same content as L3 (the full paper). The production-specific framing section below was not duplicated in L3 and was not fully absorbed into COMBINED.md's granularity discussion.

### Why This Matters for SGLang/vLLM FP8 KV

Both SGLang (`--kv-cache-dtype fp8_e4m3`) and vLLM (`fp8_e4m3`) use **per-tensor FP8** for K and V — a single scale per attention layer. This is simpler and faster but ignores KIVI's insight:

- **Per-tensor FP8 K**: one scale for the entire K tensor. Channel outliers get quantized with the same scale as normal channels → outlier channels dominate the scale → normal channels lose precision.
- **KIVI-2 K**: per-channel scale → outlier channels get their own scale → normal channels retain precision.

This explains why FP8 KV with scale=1.0 (no calibration) performs worse: the scale is too large for normal channels. With calibrated checkpoint scales (from `llm-compressor`), the per-tensor scale is chosen to minimize the MSE across the full tensor, which partially compensates.

vLLM's `fp8_per_token_head` mode (dynamic per-token-head scales) is a step toward KIVI's insight — adapting the scale dynamically per token rather than using a fixed per-tensor scale.

### Key Takeaways (L2 framing)

- **K needs per-channel quantization** (channel outliers); **V needs per-token quantization** (token-wise variation).
- Per-tensor FP8 (current production approach) is a simplification of the optimal approach — it works well enough with calibration, but sub-optimal.
- 2-bit quantization is achievable with near-lossless quality if the granularity is right — this is the research direction that could push memory savings beyond the 2× of FP8.
- The `group_size` parameter in KIVI (how many elements share a scale) is the main accuracy/memory tradeoff knob.
- SGLang and vLLM do not yet implement per-channel K quantization; this remains a research gap in production systems.

---

## Omission 2: KIVI L3 — Exact Quantization Formulas and Hardware Implementation Details

**Source:** `L3/01_kivi_arxiv24.md`
**Why omitted from COMBINED.md:** COMBINED.md covers the K/V asymmetry insight and the algorithm at a high level. The exact min/max formulas and hardware-friendly implementation details are more technical than needed for the synthesis narrative.

### Quantization Granularity — Exact Formulas

```
K quantization:
  scale = (max(K[:, channel]) - min(K[:, channel])) / (2^bits - 1)
  zero  = -min(K[:, channel]) / scale
  → one (scale, zero) per channel per head
  → group elements along head_dim dimension

V quantization:
  scale = (max(V[token, :]) - min(V[token, :])) / (2^bits - 1)
  zero  = -min(V[token, :]) / scale
  → one (scale, zero) per token per head
  → group elements along head_dim dimension for token groups
```

### Hardware-Friendly Implementation Notes

- No backpropagation or gradient computation needed
- No tuning on calibration dataset
- Custom CUDA kernels for dequantization at attention time
- Compatible with existing attention implementations

### KIVI vs Production System — Full Comparison Table

| Aspect | KIVI | SGLang FP8 | vLLM fp8_per_token_head |
|---|---|---|---|
| K scale granularity | Per-channel | Per-tensor | Per-token-head |
| V scale granularity | Per-token | Per-tensor | Per-token-head |
| Bits | 2 | 8 | 8 |
| Memory saving | 8× vs BF16 | 2× vs BF16 | 2× vs BF16 |
| Scale source | Computed online | Calibrated checkpoint | Computed online (dynamic) |
| Tuning required | No | Yes (for best accuracy) | No |
| Production ready | Research | Yes | Yes |

vLLM's `fp8_per_token_head` is philosophically closer to KIVI (dynamic scales, no calibration) than SGLang's per-tensor FP8 (static calibrated scales). Neither implements KIVI's exact per-channel K / per-token V scheme at production scale yet.

---

## Omission 3: KVQuant L3 — Speedup Numbers, Non-Uniform Datatype Details, and Full Comparison Table

**Source:** `L3/02_kvquant_arxiv24.md`
**Why omitted from COMBINED.md:** The pre-RoPE technique (the unique contribution of KVQuant) is covered in full. The speedup numbers and comparison table are research-stage specifics that did not fit the synthesis narrative.

### KVQuant Speedup Results

- Custom CUDA kernels for KVQuant dequantization
- **~1.7× speedup** vs baseline FP16 matrix-vector multiply for LLaMA-7B
- Bottleneck shifts from memory bandwidth to compute at sub-4-bit

### Non-Uniform KV Cache Quantization — Full Detail

Standard quantization uses a **uniform grid** (equal spacing between quantization levels). But KV distributions are often:
- Clustered near zero with some large outliers
- Skewed (not symmetric)

KVQuant computes **per-layer sensitivity-weighted non-uniform datatypes** by minimizing a second-order sensitivity metric (similar to NLP's importance scores) on a calibration set. The resulting quantization grid has more levels near high-density regions of the distribution.

This is the same principle as applying GPTQ's optimal brain quantization to KV activations rather than weights.

### Per-Vector Dense-and-Sparse — Full Detail

Some individual vectors have extreme outliers that dominate the quantization range. KVQuant isolates these outliers separately:

```
For each vector v:
  1. Identify top-k outlier values (by magnitude)
  2. Store outliers in FP16 (sparse component)
  3. Quantize remaining values with tighter range (dense component)
  4. Reconstruct: dequant(dense) + sparse
```

This is similar to GPTQ/SqueezeLLM for weight quantization, applied to KV activations. The overhead of the sparse component is small (typically k=3–5 outliers per vector).

### Full Comparison: KIVI vs KVQuant

| Aspect | KIVI | KVQuant |
|---|---|---|
| Target precision | 2-bit | 3-bit |
| K quantization | Per-channel, post-RoPE | Per-channel, **pre-RoPE** |
| V quantization | Per-token | Per-token |
| Non-uniform grid | No | Yes (per-layer) |
| Outlier handling | Residual FP16 tokens | Per-vector sparse component |
| Calibration needed | No | Yes (small calibration set) |
| Context scaling | ~4× batch increase | **1M+ token contexts** |
| Production status | Research | Research |

KVQuant is more accurate at the same bit-width but requires calibration and is more complex to implement.

---

## Omission 4: ZipCache L3 — Detailed Benchmark Numbers, Scale Overhead Math, and Comparison Table

**Source:** `L3/03_zipcache_arxiv24.md`
**Why omitted from COMBINED.md:** COMBINED.md includes ZipCache's 56.9% decode latency reduction. The prefill latency, GPU memory numbers, and the exact scale overhead calculation are additional specifics that were excluded as research-stage.

### Channel-Separable Tokenwise Quantization — Scale Overhead Calculation

```
Standard groupwise quantization:
  → one (scale, zero) per group of G elements
  → overhead: 2 × (total_elements / G) parameters

Channel-separable tokenwise:
  → separate scale per (token × channel) = per-element-of-outer-product
  → but uses separable channel and token scales: scale_total = scale_channel × scale_token
  → overhead: num_tokens + head_dim parameters (vs num_tokens × head_dim / G for groupwise)
```

For typical sizes (head_dim=128, seq_len=8192, G=64):
- Groupwise overhead: 2 × 8192 × 128 / 64 = 32,768 parameters
- Channel-separable: 8192 + 128 = 8,320 parameters

**~4× reduction in scale storage overhead** while maintaining per-element accuracy.

### Normalized Attention Score Formula

```python
# Standard attention score:
score[i, j] = softmax(Q_i × K_j^T / sqrt(d))

# Normalized score (ZipCache):
# The lower-triangular structure of causal attention means each token
# has a different number of "competitors" for attention.
# Normalize by the expected attention if uniform:
normalized_score[i, j] = score[i, j] × (i + 1)  # adjust for position
```

This normalization makes saliency scores **comparable across token positions** — early tokens naturally receive more attention due to causal masking, so without normalization they appear falsely "salient."

### Full Benchmark Results: Prefill, Decode, and Memory

On LLaMA3-8B, input length 4096:

| Metric | ZipCache vs baseline |
|---|---|
| Prefill latency | **37.3% reduction** |
| Decode latency | **56.9% reduction** |
| GPU memory | **19.8% reduction** |

On Mistral-7B, GSM8k dataset:

| Compression ratio | Accuracy drop |
|---|---|
| **4.98×** compression | **0.38%** accuracy drop |

The decode latency reduction is large because fewer bytes need to be loaded per decode step — smaller cache → faster memory bandwidth → faster decode.

### Full Comparison Table: ZipCache vs Other Methods

| Method | All tokens same bits? | Saliency-aware? | FlashAttn compatible? |
|---|---|---|---|
| FP8 KV (SGLang/vLLM) | Yes (FP8) | No | Yes |
| KIVI | Yes (INT2) | No | Partially |
| KVQuant | Yes (INT3) | No (per-layer) | No (custom kernel) |
| **ZipCache** | No (mixed) | **Yes** | **Yes** |
| H2O / StreamingLLM | No (drop) | Yes (drop salient's history) | Yes |

ZipCache occupies a unique position: it's not "drop everything below threshold" nor "quantize everything to N bits" — it's "quantize non-salient aggressively, keep salient full-precision."

### ZipCache's Connection to Production Systems

ZipCache is not implemented in SGLang or vLLM — it's a research system. But its ideas influence production:

1. The **normalized saliency score** idea could be applied within SGLang's scheduler to decide which pages to write-through to HiCache's L3 tier vs evict — a cross-layer optimization between KV quantization and tiered storage.

2. The **channel-separable tokenwise quantization** reduces scale overhead — relevant for vLLM's `fp8_per_token_head` mode where scale tensors add meaningful memory overhead.

3. The **FlashAttention-compatible saliency** approach shows that mixed-precision quantization doesn't have to sacrifice attention backend compatibility.

---

## Omission 5: SageAttention2 L3 — Two-Level Accumulation, xformers Comparison, and Composability Pipeline Code

**Source:** `L3/04_sageattention2_icml25.md`
**Why omitted from COMBINED.md:** The composability concept is covered. The two-level accumulation detail, xformers baseline comparison, and the full pipeline code block are technical specifics that were excluded.

### Two-Level Accumulation for FP8 PV — Full Detail

FP8 accumulation suffers from precision loss in long sums. SageAttention2 uses a two-level scheme:

```
Level 1: Accumulate partial P×V products in FP8 within each warp
Level 2: Accumulate warp results in FP32 before final output
```

This matches the accuracy of FP16 accumulation while using FP8 arithmetic for the majority of operations.

### Full Throughput Results (including xformers)

| GPU | Method | Relative OPS |
|---|---|---|
| RTX 4090 | FlashAttention2 | 1× |
| RTX 4090 | xformers | 0.7× |
| RTX 4090 | **SageAttention2** | **~3×** |
| H100 (Hopper) | FlashAttention3 (fp8) | 1× |
| H100 (Hopper) | **SageAttention2** | **~1× (matches FA3 fp8)** |

### Models Tested for Accuracy Validation

"Comprehensive experiments confirm that our approach incurs negligible end-to-end metrics loss across diverse models, including those for language, image, and video generation."

Tested on: language models (LLaMA family), image generation (DiT-based), video generation (CogVideo, Wan).

### Full Composability Pipeline Code

```
Full pipeline:
1. Compute K, V at BF16
2. Store K, V as FP8 (storage quantization)
3. At decode step: load K_fp8, V_fp8
4. Use SageAttention2 compute path: INT4(Q) × INT4(K_fp8) → PV in FP8
5. Output in BF16
```

Memory saving (from step 2) + compute speedup (from step 4) are independent benefits.

---

## Omission 6: Full vLLM CacheDType Enumeration with All Variants

**Source:** `L2/02_vllm_kv_quant_docs.md`
**Why omitted from COMBINED.md:** COMBINED.md's comparison table covers the major formats. The complete literal type including variants like `fp8_inc`, `turboquant_k3v4_nc`, `turboquant_4bit_nc` was not needed for design decisions.

### Complete `CacheDType` Literal Type

```python
CacheDType = Literal[
    "auto",                  # Use model dtype (BF16 by default)
    "float16",               # FP16 storage
    "bfloat16",              # BF16 storage (same as auto for most models)
    "fp8",                   # FP8 generic (maps to e4m3 on CUDA)
    "fp8_e4m3",              # FP8 E4M3: 1 byte, range ±448 — best accuracy
    "fp8_e5m2",              # FP8 E5M2: 1 byte, range ±57344 — more range, less mantissa
    "fp8_inc",               # FP8 via llm-compressor integration
    "fp8_ds_mla",            # DeepSeek-style FP8 MLA packed layout
    "turboquant_k8v4",       # TurboQuant: K=INT8, V=INT4
    "turboquant_4bit_nc",    # TurboQuant: 4-bit non-contiguous
    "turboquant_k3v4_nc",    # TurboQuant: K=3-bit, V=4-bit NC
    "turboquant_3bit_nc",    # TurboQuant: 3-bit non-contiguous
    "int8_per_token_head",   # INT8 with dynamic per-(token,head) scales
    "fp8_per_token_head",    # FP8 with dynamic per-(token,head) scales
    "nvfp4",                 # NVFP4: packed FP4+FP8 block scales (Blackwell only)
]
```

### SGLang set_kv_buffer Python-Level Code (from L2/vLLM context)

```python
# SGLang: set_kv_buffer()
cache_k.div_(k_scale)       # scale down: BF16 → FP8 range
cache_k = cache_k.to(fp8)  # cast
cache_k = cache_k.view(uint8)  # store as uint8 (index_put workaround)
```

Key detail: `k_scale` can be shape `[1]` (per-tensor) **or** `[num_heads]` (per-head) in vLLM's kernel — the `kv_scale_stride` variable dispatches appropriately.

### vLLM CUDA Kernel Signature — Full Detail

```c
void reshape_and_cache_flash(
    torch::Tensor& key,        // [num_tokens, num_heads, head_size]
    torch::Tensor& value,
    torch::Tensor& key_cache,  // [num_blocks, block_size, num_heads, head_size]
    torch::Tensor& value_cache,
    torch::Tensor& slot_mapping,
    const std::string& kv_cache_dtype,
    torch::Tensor& k_scale,    // [1] for per-tensor, [num_heads] for per-head
    torch::Tensor& v_scale)
```

### vLLM Sleep Mode Bug — Full Code

When a vLLM GPU goes idle and then wakes up, calibrated scales are **reset to 1.0**:

```python
def init_fp8_kv_scales(self) -> None:
    # TODO: restore calibrated scales here in the future
    k_scale_val, v_scale_val = 1.0, 1.0
    # ... fills all attention layers with 1.0
```

If you deployed with `--quantization-param-path` and the GPU sleeps, the scales are lost until restart. This is a production-correctness risk.

---

## Omission 7: WKVQuant, QJL, Gear, KVSharer, MiKV — Sub-8-bit and Ultra-Low-bit Methods

**Source:** `L4/02_kv_cache_management_survey_24.md`
**Why omitted from COMBINED.md:** These are research-stage methods with no production implementations. COMBINED.md acknowledges their existence in the taxonomy table. Full detail requires paper-level treatment.

### Sub-8-bit Quantization Methods (full survey table)

| Method | Bits | K strategy | V strategy |
|---|---|---|---|
| KIVI | 2 | Per-channel | Per-token |
| KVQuant | 3 | Per-channel, pre-RoPE | Per-vector, non-uniform |
| ZipCache | Mixed | Channel-separable tokenwise | Salient=FP16, rest=INT2 |
| **WKVQuant** | 4 | Windowed quantization | Per-token |
| **QJL** | 4 | Random JL transform | Random JL transform |
| **Gear** | 4 | Low-rank + sparse outlier | Low-rank + sparse outlier |

**WKVQuant:** Uses windowed quantization — scales are computed over a sliding window of tokens rather than the full context. More robust to distribution shifts over very long contexts than global calibration.

**QJL (Johnson-Lindenstrauss):** Applies a random dimensionality-reducing transform to K and V before quantization. The JL lemma guarantees that distances are approximately preserved → attention scores are approximately correct → acceptable quality at 4-bit. No calibration needed; the randomness provides distribution-agnostic coverage.

**Gear:** Low-rank approximation of K/V matrices plus a sparse residual for outliers. Similar in spirit to KVQuant's dense+sparse but uses matrix decomposition rather than per-vector isolation. Better for attention patterns with global low-rank structure.

### 1-Bit / Ultra-Low-Bit Methods

Research-stage only. Methods: KVSharer (share K/V across layers), MiKV (progressive quantization). Quality degradation significant; not ready for production.

**KVSharer:** Shares K and V tensors across consecutive transformer layers — layers i and i+1 use the same K/V buffer. This halves the KV cache size without quantization noise, but assumes adjacent layers have similar attention patterns. Works for some model architectures; fails for others.

**MiKV (Mixed Intermediate KV):** Progressive quantization that assigns different bit widths to different token positions: recent tokens at FP8, middle-distance tokens at INT4, distant tokens at INT2. A dynamic pyramid of precision matching the empirically observed decay of attention weight with token distance.

**Survey assessment:** 1-bit and ultra-low-bit KV quantization achieves significant compression but with meaningful quality loss on most tasks. Not recommended for general-purpose production use. May be acceptable for specific workloads (highly repetitive content, summarization) where attention is highly concentrated.

---

## Omission 8: Linear-Time Models and Alternative Compression Paradigms

**Source:** `L4/01_longctx_compression_bench_24.md`
**Why omitted from COMBINED.md:** These represent architectural alternatives to attention with KV cache, not quantization techniques. They are competing paradigms, not methods for making the KV cache smaller.

### Full Taxonomy of Long-Context KV Compression

| Category | Examples | KV Memory | Quality |
|---|---|---|---|
| **KV cache quantization** | KIVI, KVQuant | Lossless (all tokens, fewer bits) | High with calibration |
| **Token dropping** | H2O, SnapKV, StreamingLLM | Lossy (fewer tokens, full bits) | Task-dependent |
| **Prompt compression** | Selective Context, LLMLingua | Reduces input length | High for summarization |
| **Linear-time models** | Mamba, RetNet | O(1) memory | Lower for retrieval tasks |
| **Hybrid architectures** | Jamba, Zamba | Selective attention | Moderate |
| **Attention approximation** | Sparse attention | Reduces attention cost | Moderate |
| **No compression** | Full attention | Baseline | Baseline |

### Linear-Time Models — Findings from Benchmark

**Mamba, RetNet** and similar state-space / linear attention models replace the O(n²) attention mechanism with O(n) recurrence. They have **no KV cache** in the traditional sense — the "state" is a fixed-size hidden vector updated incrementally.

**Benchmark finding:** Linear-time models perform poorly on **retrieval tasks** (PassageRetrieval, NeedleBench). The fixed-size state cannot retain arbitrary token information — it compresses the context into a lossy summary. For summarization and generation tasks, linear models are competitive. For precise retrieval, they fail.

**Hybrid architectures** (Jamba = Mamba + attention layers, Zamba = similar) partially recover retrieval ability by using attention on a subset of layers, but they still underperform full attention on demanding retrieval benchmarks.

### Prompt Compression Methods — Findings from Benchmark

**Selective Context, LLMLingua:** Remove semantically redundant tokens from the input before it is processed by the LLM. This reduces the effective context length rather than the KV cache of a given context.

**Benchmark finding:** Prompt compression works well for summarization and few-shot tasks but fails for tasks requiring precise token-level information (code, retrieval). The compression is lossy at the input level — information dropped before the model sees it cannot be recovered.

### Full Experimental Design

**10+ methods evaluated** across **7 task categories**:
1. **Single-document QA** (NarrativeQA, QuALITY)
2. **Multi-document QA** (HotpotQA, 2WikiMultiHopQA, MuSiQue)
3. **Summarization** (GovReport, MultiNews)
4. **Few-shot learning** (TREC, TriviaQA, SAMSum)
5. **Synthetic tasks** (PassageCount, PassageRetrieval)
6. **Code** (LCC, RepoBench-P)
7. **Long dialogue** (QMSum)

All methods evaluated on **the same base models** (Llama-2-7B-Chat, Llama-2-13B-Chat) with the same evaluation pipeline — eliminating confounders.

**Finding 4 (full):** Small compression ratios (2–4×) are safe for all methods. Aggressive compression (8–16×):
- Quantization: degrades gracefully (lower precision but all tokens retained)
- Token dropping: catastrophic failures on retrieval tasks (key tokens were dropped)
- Linear models: already at maximum compression (fixed state size)

---

## Omission 9: Weight Quantization Methods — GPTQ, AWQ, GGUF

**Source:** `L4/03_llm_inference_optimization_survey_24.md`
**Why omitted from COMBINED.md:** These are weight quantization methods — they quantize model parameters, not KV activations. Their interaction with KV cache quantization is covered in COMBINED.md Section 9. The internal algorithms are excluded as they are not KV cache methods.

### GPTQ (Optimal Brain Quantization)

- Layer-by-layer INT4 weight quantization using second-order information
- The `--quantization gptq` flag in SGLang/vLLM
- GPTQ-quantized models can be combined with FP8 KV cache: GPTQ weights + FP8 KV
- Algorithm: for each weight row, solve a constrained optimization to minimize output reconstruction error given the quantization constraint. Uses the inverse Hessian of the layer's output with respect to its weights.
- Requires a calibration set (~128 samples); takes 10–60 minutes for 70B models.

### AWQ (Activation-Aware Weight Quantization)

- Identifies and protects important weight channels (those with large activation magnitudes)
- Scales important channels up before quantization, scales them back down after → the quantization error on important channels is reduced
- AWQ pipelines also collect **KV activation statistics** as a byproduct — this is how the `quantization_param_path` JSON files are generated
- **AWQ + FP8 KV is the recommended joint quantization recipe for production**: AWQ quantizes weights, and the same calibration pass captures KV scales
- Available via `llm-compressor` and `AutoAWQ` libraries

### GGUF / llama.cpp Formats

- 2–8-bit weight quantization for CPU inference
- Separate from KV cache quantization (GGUF is for weights; CPU inference has its own KV handling via `llama.cpp`)
- Not directly relevant to SGLang/vLLM GPU serving
- GGUF's "k-quant" variants (Q4_K_M, Q5_K_S, etc.) use similar ideas to KIVI — per-block scales covering groups of weights — but applied to static weight matrices, not dynamic KV activations

---

## Omission 10: Knowledge Distillation and Pruning

**Source:** `L4/03_llm_inference_optimization_survey_24.md`
**Why omitted from COMBINED.md:** These are different inference optimization paradigms — they change the model itself, not how KV tensors are stored. Not relevant to KV cache quantization.

### Structured Pruning

Removes entire heads, layers, or neurons from the model:
- **Head pruning:** identifies attention heads that contribute little (by gradient magnitude or attention entropy) and removes them. Reduces `num_heads`, which reduces KV cache size proportionally.
- **Layer pruning:** removes entire transformer layers — the most aggressive form; fewer layers = fewer KV cache layers.
- **Neuron/FFN pruning:** removes individual neurons from feed-forward layers. Does not reduce KV cache.

**Interaction with KV quantization:** Head pruning and layer pruning reduce KV cache size architecturally (fewer heads/layers). Applying FP8 quantization on top gives multiplicative reduction. Some works combine pruning + quantization jointly (GPTQ on pruned models, etc.).

### Unstructured Pruning

Removes individual weight values (sets them to zero) rather than entire components:
- Creates sparse weight matrices that require sparse matrix-vector multiplication
- Does not reduce KV cache size — only affects weight matrix computation
- Hardware support for sparse matmul is limited on most GPUs (A100 2:4 structured sparsity is the main exception)
- Less relevant to KV cache optimization than structured pruning

### Knowledge Distillation

Trains a smaller "student" model to mimic a larger "teacher" model:
- The student has fewer parameters, fewer layers, or smaller hidden dimension
- Fewer layers directly reduces KV cache size (fewer K/V buffers)
- Distillation is a training-time operation — it produces a different model, not a different KV caching strategy
- **Interaction with KV quantization:** distilled models can then have their KV cache quantized using the same FP8 techniques. The combination (distill → quantize) can achieve very aggressive memory reduction while maintaining quality better than quantization alone on the original large model.

---

## Omission 11: Model Architecture Details — MQA, GQA Formulas, MLA Technical Detail

**Source:** `L4/02_kv_cache_management_survey_24.md`
**Why omitted from COMBINED.md:** COMBINED.md covers the memory impact (8× for GQA-8, 120× for MLA + FP8). The formulas and the MLA expansion mechanism were excluded as they are model-architecture content, not quantization content.

### MQA / GQA — Exact Formulas

Standard MHA: `num_kv_heads = num_query_heads`
MQA: `num_kv_heads = 1` (all heads share one K/V)
GQA: `num_kv_heads = num_query_heads / group_size`

**Memory impact (Llama-3.1-70B example):**

| Attention type | num_kv_heads | KV size vs MHA |
|---|---|---|
| MHA (GPT-2 style) | 64 | 1× (baseline) |
| GQA-8 (Llama-3) | 8 | 8× smaller |
| MQA (Falcon) | 1 | 64× smaller |

GQA already makes FP8 quantization even more impactful: a Llama-3 model with GQA-8 + FP8 KV uses **16× less KV memory** than the equivalent MHA model in BF16.

### Multi-head Latent Attention (MLA) — Expansion Mechanism

DeepSeek's MLA compresses KV into a **low-rank latent vector** before storing:

```
Standard MHA:
  store K, V separately at full head_dim each
  KV memory = 2 × num_layers × num_heads × head_dim × seq_len × bytes

MLA:
  store compressed latent C (much smaller) + rope_K separately
  at attention time: expand C → K, V using learned projection W_UK, W_UV
  KV memory = num_layers × (kv_lora_rank + rope_head_dim) × seq_len × bytes
```

The latent vector has dimension `kv_lora_rank << head_dim × num_heads` (e.g., kv_lora_rank=512 vs head_dim × num_heads = 128 × 128 = 16,384 for a 128-head model). This is why DeepSeek-V3-671B has an effective KV cache ~60× smaller than a hypothetical full-KV 671B model.

**FP8 quantization of MLA:** The tensor to quantize is the compressed latent C, not the full K/V. SGLang's `MLATokenToKVPoolHost` uses per-tile FP32 scales embedded alongside the FP8 data — the granularity is finer than the MHA per-layer-tensor approach because the latent dimension is small enough to make per-tile scales practical.

### Quantization + Token Dropping Interaction — Full Survey Text

ZipCache shows mixed-precision quantization (salient=FP16, non-salient=INT2) is feasible. The survey identifies this as an **active research area**: should non-salient tokens be quantized (rather than dropped) to enable recovery if needed? This combines:
- The "don't drop anything permanently" property of quantization
- The "focus bits on important tokens" property of saliency-based methods

The open question: at what compression ratio does "keep non-salient at 2-bit" outperform "drop non-salient entirely" for downstream task quality? The benchmark (arXiv 2407.01527) suggests the crossover is task-dependent — retrieval tasks strongly prefer keeping all tokens (quantized), while summarization tolerates aggressive dropping.

---

## Omission 12: Accuracy-Efficiency Frontier Diagram

**Source:** `L4/03_llm_inference_optimization_survey_24.md`
**Why omitted from COMBINED.md:** The composability table in Section 9 covers this content in tabular form. The ASCII diagram from the survey is preserved here in full.

```
Accuracy
  ↑
  │  BF16 weights + BF16 KV ●  (100% accuracy, 100% memory)
  │
  │  BF16 weights + FP8 KV  ●  (~99% accuracy, 50% KV memory)
  │
  │  GPTQ INT4 weights + FP8 KV ●  (~97% accuracy, 25% weight memory + 50% KV memory)
  │
  │  KIVI-2bit KV + INT4 weights ●  (~96% accuracy, 6.25% KV memory)
  │
  └──────────────────────────────────→ Memory efficiency
```

Each quantization decision moves along this frontier. The ideal is the rightmost point with acceptable accuracy for your use case.

### Practical Recommendations — Full Survey Text

**When to use FP8 KV cache:**

Use when:
- VRAM is the bottleneck (can't fit desired batch size or context length)
- Accuracy difference between FP8 and BF16 is acceptable (typically < 1% on standard benchmarks)
- You have a calibrated checkpoint (models from HuggingFace fp8 repos) or can tolerate scale=1.0

Avoid when:
- Model outputs are extremely sensitive to numerical precision (scientific, medical)
- Input distribution is very different from calibration data (specialized domains)
- Sub-0.1% accuracy is required (use BF16)

**When to go sub-4-bit (KIVI, KVQuant):**

Use when:
- 2× memory saving from FP8 isn't enough
- Running very long contexts (1M+ tokens) where BF16 is physically impossible
- Using custom CUDA kernels is acceptable
- Calibration dataset is available

**When to use dynamic per-token-head (vLLM fp8_per_token_head):**

Use when:
- Input distribution is diverse (multiple languages, domains, code + text)
- No calibration dataset is available or practical
- vLLM is the engine (SGLang doesn't support this yet)
