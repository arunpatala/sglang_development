# Layer 18 — Lesson Outline: KV Cache Quantization

## What This Lesson Covers

Layer 12 introduced `RadixCache` and `MHATokenToKVPool`: the KV cache stored in GPU VRAM as BF16 tensors. Layer 17 introduced HiCache: moving those BF16 tensors across tiers when VRAM fills. Layer 18 asks a different question — what if we stored each KV tensor value in **fewer bits** to begin with?

KV cache quantization stores the Key and Value tensors in FP8 (1 byte) or FP4 (0.5 byte) instead of BF16 (2 bytes). This halves or quarters the memory footprint of the KV cache directly — fitting more tokens in GPU VRAM, increasing throughput at larger batch sizes, and reducing the PCIe transfer cost when combined with HiCache tiering. The tradeoff is a small accuracy loss from the reduced numerical precision of the stored values.

This layer covers the full stack: why the KV cache dominates VRAM at scale → what float8 represents and why it is harder to use than float8 weight quantization → how SGLang implements FP8 KV (the `BaseKVCacheMethod`, `configure_kv_cache_dtype`, and `set_kv_buffer` paths) → how vLLM extends this with per-token-head dynamic scales, `q_scale`/`prob_scale`, and TurboQuant → what the research literature reveals about sub-FP8 quantization (KIVI, KVQuant, SageAttention2) → and how to choose the right configuration for your workload.

No new model execution logic is introduced. The model forward pass, attention kernels, and scheduler carry forward unchanged. The new work is entirely in how KV tensors are **encoded when written to the cache** and **decoded when read back** for attention — the write path in `set_kv_buffer()` and the read path inside the attention backend.

---

## Sections

### 01 — The Memory Bottleneck: Why KV Cache Quantization Exists (`01_memory_bottleneck.md`)

- KV cache size scales with `num_layers × 2 × num_heads × head_dim × num_tokens × bytes_per_element`
- At BF16 (2 bytes): Llama-3.1-70B serving a 10K-token context in a batch of 32 requires ~26 GB of KV cache — more than the model weights themselves
- The dual pressure: serving larger **batch sizes** (for throughput) and longer **context lengths** (for capability) both grow the KV cache in opposite directions, making VRAM the binding constraint at both extremes
- Why this is different from the HiCache problem (Layer 17): HiCache moves KV tensors to cheaper tiers when VRAM fills; quantization shrinks the KV tensors so more fit in VRAM in the first place — they are orthogonal and composable
- The decode bottleneck: at decode time, the GPU must load all KV layers for every active token in the batch on every step — this is memory-bandwidth-bound, not compute-bound; halving the KV size doubles the effective memory bandwidth available for decode
- The math: for Llama-3.1-70B with 80 layers, 64 KV heads, head_dim=128, and a batch of 32 requests each 2048 tokens long:
  - BF16: 80 × 2 × 64 × 128 × 32 × 2048 × 2 bytes ≈ 170 GB
  - FP8: same × 1 byte ≈ 85 GB — fits on a single 8×H100 node
  - FP4: same × 0.5 byte ≈ 42 GB — fits on 4×H100

### 02 — Float8 and the Quantization Taxonomy (`02_fp8_and_taxonomy.md`)

- What float8 is: the two FP8 formats used in deep learning hardware:
  - `fp8_e4m3`: 4 exponent bits, 3 mantissa bits — range ±448, preferred for activations (more precision, less range)
  - `fp8_e5m2`: 5 exponent bits, 2 mantissa bits — range ±57344, more range but less precision
- Why FP8 ≠ INT8: FP8 uses a floating-point encoding with an exponent, so it covers a wider dynamic range than INT8 at the cost of non-uniform quantization steps; this is important because KV activations can have outliers that INT8 would clip
- The per-tensor vs per-token vs per-channel scale problem: a single scale per attention layer (per-tensor) must accommodate outlier channels, leaving normal channels under-quantized; the KIVI paper's key finding:
  - **Key (K) cache**: channel-wise outliers (specific feature dimensions are consistently large across all tokens) → needs per-channel scale
  - **Value (V) cache**: token-wise variation (magnitudes vary by token, not by channel) → needs per-token scale
  - Production engines (SGLang, vLLM) use per-tensor for simplicity; KIVI achieves much better accuracy at the same or lower bit-width with asymmetric granularity
- Pre-RoPE vs post-RoPE quantization: rotary positional embedding (RoPE) rotates the K vectors, scrambling the channel-wise structure; quantizing K before RoPE (KVQuant's technique) gives more stable statistics; current production engines cache post-RoPE K — a known limitation
- The quantization design space:
  ```
  By precision:   FP8 (1 byte) > INT8 (1 byte) > FP4 (0.5 byte) > INT4 (0.5 byte) > INT2
  By granularity: per-tensor < per-channel < per-token-head < per-vector
  By scale source: calibrated (static) < dynamic (computed at cache-write time)
  ```
- Where Layer 18 focuses: FP8 and FP4 production paths in SGLang and vLLM; research directions for sub-FP8

### 03 — SGLang FP8 KV: The Write and Read Path (`03_sglang_fp8_kv.md`)

- `configure_kv_cache_dtype()` (`model_runner.py:2007`): maps `--kv-cache-dtype` string to a torch dtype; handles the `"auto"` path that reads `kv_cache_quant_algo` from the model's `config.json` to detect FP8 quantized checkpoints without an explicit flag
- `BaseKVCacheMethod` (`layers/quantization/kv_cache.py:16`): attaches `k_scale` and `v_scale` as `nn.Parameter` (initialized to -1.0) to each `RadixAttention` layer; loaded from the `--quantization-param-path` JSON file after model load; falls back to 1.0 with a warning if missing
- Loading scales from checkpoint (`model_runner.py:1288`): `model.load_kv_cache_scales(path)` reads per-layer scale JSON and populates `layer.k_scale_float` / `layer.v_scale_float`; the FP8-FNUZ path (AMD ROCm) doubles the scales due to the different FP8 encoding
- `MHATokenToKVPool` pool allocation (`memory_pool.py:662`): FP8 dtypes use `store_dtype = torch.uint8` because PyTorch's `index_put_()` is not implemented for `torch.float8_*`; the buffer holds uint8 bytes that are reinterpreted as FP8 via `.view(dtype)` at attention time
- **The write path** — `set_kv_buffer()` (`memory_pool.py:995`):
  ```python
  # 1. Scale down from BF16 range to FP8 range
  cache_k.div_(k_scale)
  cache_v.div_(v_scale)
  # 2. Cast to FP8 dtype
  cache_k = cache_k.to(fp8_e4m3fn)
  cache_v = cache_v.to(fp8_e4m3fn)
  # 3. Reinterpret as uint8 for index_put
  cache_k = cache_k.view(torch.uint8)
  # 4. Scatter into paged KV buffer
  ```
- **The read path**: `get_key_buffer()` returns the uint8 tensor; the attention backend calls `.view(fp8_dtype)` to recover the FP8 view; FlashInfer and TRT-LLM accept the FP8 KV directly with a `bmm1_scale` argument — dequantization is fused inside the attention kernel
- MLA + FP8 KV path (`memory_pool.py:1467`): `MLATokenToKVPool` with FP8 expands the KV cache dimension to include per-tile FP32 scale storage alongside the FP8 data; `nsa_kv_cache_store_fp8` handles the fused quant + packed write; dequantization uses `dequantize_k_cache_paged()` in `nsa_backend.py`
- Key code anchors:

| Concept | Location |
|---|---|
| `--kv-cache-dtype` CLI flag | `REPOS/sglang/python/sglang/srt/server_args.py:4169` |
| `--quantization-param-path` flag | `REPOS/sglang/python/sglang/srt/server_args.py:4159` |
| `configure_kv_cache_dtype()` | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:2007` |
| Load KV scales from checkpoint | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:1288` |
| `BaseKVCacheMethod` | `REPOS/sglang/python/sglang/srt/layers/quantization/kv_cache.py:16` |
| FP8 stored as uint8 | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:662` |
| `set_kv_buffer()` write path | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:995` |
| TRT-LLM fused FP8 write | `REPOS/sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py:552` |
| Triton FP8 KV kernel | `REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py` |
| `ModelOptFp8KVCacheMethod` | `REPOS/sglang/python/sglang/srt/layers/quantization/modelopt_quant.py:277` |

### 04 — vLLM's Extended Quantization Model (`04_vllm_kv_quant.md`)

- `CacheDType` literal type (`vllm/config/cache.py:18`): 13 supported KV formats vs SGLang's 5; adds `fp8_per_token_head`, `int8_per_token_head`, `nvfp4`, `turboquant_*`, `fp8_ds_mla`
- `KVQuantMode` enum (`vllm/v1/kv_cache_interface.py:30`): centralized dispatch (`NONE`, `FP8_PER_TENSOR`, `INT8_PER_TOKEN_HEAD`, `FP8_PER_TOKEN_HEAD`, `NVFP4`) replacing string comparison throughout the codebase
- **Per-token-head dynamic quantization** (`fp8_per_token_head`, `int8_per_token_head`): scales are computed dynamically **per (token, head)** at cache-write time in the fused CUDA kernel — no calibration dataset or `--quantization-param-path` required; `page_size_bytes` adds `2 × block_size × num_kv_heads × 4 bytes` for the float32 scale tensors; per-token-head is more principled than per-tensor because it adapts to each token's actual activation range
- **Extended scales: `q_scale` and `prob_scale`** (`vllm/model_executor/layers/quantization/kv_cache.py:40`): beyond K and V storage, vLLM attaches scales for the Q-projection output and the softmax probability matrix — enabling fully end-to-end FP8 attention where `Q×K^T` and `P×V` matmuls run in FP8 compute; SGLang does not have these compute-path scales
- **Fused CUDA kernel** `reshape_and_cache_flash` (`csrc/cache_kernels.cu:704`): scale application + cast to FP8 + scatter into paged layout in a single kernel; `k_scale` may be shape `[1]` (per-tensor) or `[num_heads]` (per-head) with `kv_scale_stride` dispatch; NVFP4 dispatches to a separate SM100-compiled kernel
- **TurboQuant** (vLLM-only): NVIDIA Research's custom mixed-precision format storing K in INT8 and V in INT4 (`turboquant_k8v4`) or both in 3-bit (`turboquant_3bit_nc`); first 2 and last 2 attention layers are automatically skipped (different activation statistics at boundaries); not yet in SGLang
- **Sleep-mode scale bug** (`vllm/v1/worker/gpu_model_runner.py:885`): `init_fp8_kv_scales()` resets all per-layer scales to 1.0 after a GPU wakes from idle sleep — calibrated scales from the checkpoint are lost; a known TODO in the codebase; a production-correctness risk for deployments using llm-compressor calibrated checkpoints

Key code anchors:

| Concept | Location |
|---|---|
| `CacheDType` — all 13 formats | `REPOS/vllm/vllm/config/cache.py:18` |
| `KVQuantMode` enum | `REPOS/vllm/vllm/v1/kv_cache_interface.py:30` |
| `page_size_bytes` with scale storage | `REPOS/vllm/vllm/v1/kv_cache_interface.py:132` |
| `BaseKVCacheMethod` (q/k/v/prob scales) | `REPOS/vllm/vllm/model_executor/layers/quantization/kv_cache.py:18` |
| Per-token-head scale deletion | `REPOS/vllm/vllm/model_executor/layers/quantization/kv_cache.py:57` |
| `reshape_and_cache_flash` CUDA kernel | `REPOS/vllm/csrc/cache_kernels.cu:704` |
| NVFP4 dispatch (SM100 only) | `REPOS/vllm/csrc/cache_kernels.cu:731` |
| FP8 read path + descale tensors | `REPOS/vllm/vllm/v1/attention/backends/flash_attn.py:743` |
| Sleep-mode scale reset | `REPOS/vllm/vllm/v1/worker/gpu_model_runner.py:885` |

### 05 — Research Frontiers: Sub-FP8 and Compute Quantization (`05_research_frontiers.md`)

- **KIVI — 2-bit Asymmetric KV Cache Quantization** (NeurIPS 2024, arXiv 2402.02750):
  - Core finding: K has channel-wise outliers (same channels are large across all tokens) → quantize K **per-channel**; V has token-wise variation → quantize V **per-token**
  - Algorithm: quantize all but the last `group_size` tokens (residual FP16); merge into INT2 once the residual window advances
  - Result: 2.6× peak memory reduction (including weights), 2.35–3.47× throughput, near-identical quality on LLaMA, Falcon, Mistral
  - Why it matters: demonstrates that per-tensor FP8 (current production approach) is leaving significant accuracy on the table — per-channel K quantization is fundamentally better

- **KVQuant — Sub-4-bit with Pre-RoPE Quantization** (NeurIPS 2024, arXiv 2401.18079):
  - Pre-RoPE K quantization: quantize K **before** the rotary positional embedding is applied; RoPE scrambles the channel structure of K making post-RoPE quantization harder; pre-RoPE K has more stable per-channel statistics
  - Non-uniform per-layer datatypes: fit the quantization grid to the actual (non-Gaussian) KV distribution
  - Per-vector dense-and-sparse: isolate extreme outliers in a sparse FP16 component, quantize the remainder tightly
  - Result: < 0.1 perplexity degradation at 3-bit; 1 million token context on a single A100-80GB
  - Implementation gap: both SGLang and vLLM currently cache **post-RoPE K**; implementing pre-RoPE quantization requires splitting the `apply_rotary_pos_emb()` call and the KV cache write — a non-trivial architectural change

- **SageAttention2 — INT4/FP8 Attention Compute** (ICML 2025, arXiv 2411.10958):
  - Quantizes the **attention matmul** itself, not just storage: Q and K cast to INT4 (per-thread granularity), P̃ and V to FP8
  - Q outlier smoothing: per-channel scale applied before INT4 cast (same insight as KIVI's per-channel K)
  - Two-level FP8 accumulation: partial sums in FP8, final reduce in FP32
  - Result: ~3× faster than FlashAttention2 on RTX 4090; matches FlashAttention3 FP8 on Hopper GPUs with better accuracy
  - Composability: FP8 KV storage (this layer) + SageAttention2 compute = fully quantized attention path; vLLM's `q_scale`/`prob_scale` is the production analog

- **ZipCache — Mixed-Precision with Salient Token Identification** (ECCV 2024, arXiv 2405.14256):
  - Identifies "salient" tokens (high normalized attention score) → keep at FP16; quantize the rest aggressively
  - Channel-separable tokenwise quantization reduces scale overhead vs groupwise
  - FlashAttention-compatible saliency metric (decoupled from full attention scores)
  - Result: 4.98× compression with only 0.38% accuracy drop on Mistral-7B/GSM8K; 56.9% decode latency reduction

- **NVFP4** (vLLM, Blackwell hardware):
  - Packed 4-bit data + FP8 block scales per head; 0.5 bytes per KV value
  - SM100/SM120 (Blackwell B100/B200) only — compiled separately from the main build
  - Currently WIP in vLLM (NotImplementedError in FlashInfer builder path)

### 06 — Accuracy, Configuration, and Practical Tradeoffs (`06_accuracy_and_config.md`)

- **Accuracy characterization**:
  - FP8 with calibrated scales (`--quantization-param-path`): < 0.5% degradation on MMLU, LongBench, HumanEval for calibrated Llama, Qwen, Mistral models
  - FP8 with scale=1.0 (no calibration): potentially visible degradation for tokens near the FP8 saturation limit; safe for models where the dynamic range happens to fit in FP8 (rare); always warn when missing
  - FP4 (`fp4_e2m1`, experimental): explicit accuracy drop warning in SGLang source; not recommended for production
  - Per-token-head dynamic FP8 (vLLM): comparable to calibrated per-tensor without the calibration overhead; better for diverse input distributions (multi-language, code+text)

- **Scale calibration workflow** (for `--quantization-param-path`):
  ```python
  # Using llm-compressor (neural-magic/llm-compressor) to generate scales
  from llmcompressor.modifiers.quantization import QuantizationModifier
  recipe = QuantizationModifier(
      targets="Linear",
      scheme="FP8",
      ignore=["lm_head"],
      kv_cache_scheme={"type": "float8", "strategy": "tensor"}
  )
  # Outputs: quantization_config.json with per-layer k_scale / v_scale
  ```
  - Many HuggingFace checkpoints include scales: `meta-llama/Meta-Llama-3.1-8B-Instruct-FP8`, `Qwen/Qwen2.5-72B-Instruct-GPTQ-Int4-FP8KV`, etc.
  - With `--kv-cache-dtype auto` + an FP8 checkpoint, SGLang reads `kv_cache_quant_algo` from `config.json` and loads scales automatically

- **SGLang launch commands**:
  ```bash
  # FP8 KV with auto-detected scales from checkpoint
  python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct-FP8 \
    --kv-cache-dtype auto \
    --tp 8

  # FP8 KV with explicit calibrated scales
  python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8_e4m3 \
    --quantization-param-path /path/to/kv_cache_scales.json \
    --tp 8

  # FP8 KV without calibration (convenient, slight accuracy risk)
  python -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-70B-Instruct \
    --kv-cache-dtype fp8_e4m3 \
    --tp 8
  ```

- **SGLang KV cache quantization flags**:

  | Flag | Default | Effect |
  |---|---|---|
  | `--kv-cache-dtype` | `"auto"` | Storage dtype: `auto`, `fp8_e4m3`, `fp8_e5m2`, `bf16`, `fp4_e2m1` |
  | `--quantization-param-path` | `None` | Path to JSON with per-layer `k_scale`/`v_scale` values |

- **Workload decision guide**:

  | Workload | Recommended KV dtype | Reasoning |
  |---|---|---|
  | Long-context RAG | `fp8_e4m3` with calibrated scales | All tokens preserved; 2× more KV fits in VRAM |
  | Multi-turn chat | `fp8_e4m3` with calibrated scales | Preserves all turns; 2× throughput |
  | Coding agent (variable input) | vLLM `fp8_per_token_head` | Code inputs have different distributions; dynamic scales adapt |
  | Sensitive production | `auto` (BF16) | No accuracy risk; use HiCache (Layer 17) for capacity instead |
  | Long contexts, tight VRAM | `fp8_e4m3` + HiCache | Composable: 2× compression + multi-tier capacity |

- **Interaction with weight quantization**: FP8 KV and weight quantization (GPTQ, AWQ, FP8 weights) are independent and composable:
  - BF16 weights + FP8 KV: safest — full weight precision, halved KV memory
  - FP8 weights + FP8 KV: maximum compression; recommended when throughput is the goal and calibrated checkpoints exist
  - GPTQ INT4 weights + FP8 KV: deepest compression; research-grade accuracy monitoring recommended

- **Interaction with HiCache (Layer 17)**: FP8 KV quantization reduces the bytes-per-token transferred over PCIe during HiCache L2 loads — the κ_crit threshold (from the PCIe bottleneck analysis) effectively doubles; for the same PCIe bandwidth, twice as many tokens can be offloaded before the system becomes memory-bandwidth-bound

- **What Layer 18 explicitly defers**:
  - Mooncake `TransferEngine` for PD disaggregation KV transfer (separate from KV storage format)
  - MoE-specific quantization (different attention shapes, not covered here)
  - Training-aware KV quantization (QAT for KV cache) — all methods here are PTQ
  - Hardware-specific FP4 deployment (Blackwell NVFP4, Grace-Hopper CXL memory)

---

## Supporting Files

- `summary.md` — narrative walkthrough of all six sections with worked examples, diagrams, and configuration recipes
- `01_memory_bottleneck.md` — KV cache size math, decode bandwidth bottleneck, how quantization helps
- `02_fp8_and_taxonomy.md` — float8 encoding, FP8 vs INT8 vs FP4, scale granularity taxonomy
- `03_sglang_fp8_kv.md` — complete SGLang write and read path with code anchors
- `04_vllm_kv_quant.md` — vLLM's extended model: KVQuantMode, per-token-head, q_scale/prob_scale, TurboQuant
- `05_research_frontiers.md` — KIVI, KVQuant, SageAttention2, ZipCache with detailed mechanism explanations
- `06_accuracy_and_config.md` — calibration workflow, launch commands, workload decision guide, interaction with HiCache

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `--kv-cache-dtype` choices | `REPOS/sglang/python/sglang/srt/server_args.py:4169` |
| `--quantization-param-path` warning | `REPOS/sglang/python/sglang/srt/server_args.py:4159` |
| `configure_kv_cache_dtype()` | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:2007` |
| Auto-detect FP8 from `config.json` | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:2008` |
| Load KV scales from JSON | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:1288` |
| Scale=1.0 warning | `REPOS/sglang/python/sglang/srt/model_executor/model_runner.py:1305` |
| `BaseKVCacheMethod.create_weights()` | `REPOS/sglang/python/sglang/srt/layers/quantization/kv_cache.py:30` |
| `process_weights_after_loading()` | `REPOS/sglang/python/sglang/srt/layers/quantization/kv_cache.py:47` |
| FP8 stored as uint8 (index_put workaround) | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:662` |
| `set_kv_buffer()` — full write path | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:995` |
| `MHATokenToKVPool` FP8 buffer | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:640` |
| `MLATokenToKVPool` FP8 NSA quant | `REPOS/sglang/python/sglang/srt/mem_cache/memory_pool.py:1467` |
| TRT-LLM fused FP8 set buffer | `REPOS/sglang/python/sglang/srt/layers/attention/trtllm_mha_backend.py:552` |
| Triton FP8 KV paged write kernel | `REPOS/sglang/python/sglang/srt/layers/attention/triton_ops/trtllm_fp8_kv_kernel.py:1` |
| NSA dequant for FP8 K | `REPOS/sglang/python/sglang/srt/layers/attention/nsa/dequant_k_cache.py:76` |
| FlashInfer tensor-core FP8 path | `REPOS/sglang/python/sglang/srt/layers/attention/flashinfer_backend.py:1680` |
| vLLM `CacheDType` (13 formats) | `REPOS/vllm/vllm/config/cache.py:18` |
| vLLM `KVQuantMode` enum | `REPOS/vllm/vllm/v1/kv_cache_interface.py:30` |
| vLLM per-token-head scale storage | `REPOS/vllm/vllm/v1/kv_cache_interface.py:132` |
| vLLM `BaseKVCacheMethod` (q/k/v/prob) | `REPOS/vllm/vllm/model_executor/layers/quantization/kv_cache.py:18` |
| vLLM `reshape_and_cache_flash` | `REPOS/vllm/csrc/cache_kernels.cu:704` |
| vLLM NVFP4 dispatch (SM100) | `REPOS/vllm/csrc/cache_kernels.cu:731` |
| vLLM FP8 read + descale tensors | `REPOS/vllm/vllm/v1/attention/backends/flash_attn.py:743` |
| vLLM sleep-mode scale reset | `REPOS/vllm/vllm/v1/worker/gpu_model_runner.py:885` |
