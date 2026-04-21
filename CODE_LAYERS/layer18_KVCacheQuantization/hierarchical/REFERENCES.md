# References — KV Cache Quantization

Organized by **reading level** (L1–L5) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

Layer 18 covers **KV cache quantization**: reducing the precision of stored Key and Value tensors (BF16 → FP8 → FP4) to fit more tokens in GPU VRAM, increase batch size, and improve throughput — with controlled accuracy tradeoffs. References cover: **the memory bottleneck that motivates quantization, the float8 formats and their properties, per-tensor vs per-token-head vs per-channel quantization, scale calibration, how SGLang and vLLM implement KV quant today, the research literature from INT8 to 2-bit KV, dynamic attention quantization (SageAttention), the broader KV compression taxonomy, and practical deployment tooling.**

---

## Quick navigation

| Reading level | What you'll find |
|---|---|
| **L1** | Why quantization, what FP8 is, 5-minute mental model |
| **L2** | SGLang and vLLM deployment docs, how to enable FP8 KV cache today |
| **L3** | KIVI (asymmetric 2-bit), KVQuant (sub-4-bit), ZipCache (salient token quant), SageAttention2 (INT4/FP8 compute), vLLM TurboQuant |
| **L4** | Comprehensive long-context compression benchmark, KV cache management survey, inference optimization survey |
| **L5** | Source code anchors in SGLang and vLLM |

---

## Primary sources: SGLang KV cache quantization

### SGLang `--kv-cache-dtype` CLI flag

- **URL:** https://docs.sglang.io/backend/server_arguments.html
- **Level:** L2
- **What it contributes:**
  - Supported choices: `auto`, `fp8_e5m2`, `fp8_e4m3`, `bf16`, `bfloat16`, `fp4_e2m1`.
  - `auto` reads `kv_cache_quant_algo` from the model's `config.json` and activates FP8 automatically for checkpoints that declare it.
  - `--quantization-param-path` points to a JSON file with per-layer `k_scale` / `v_scale` values; without it, scales default to 1.0 and accuracy may degrade for `fp8_e4m3`.
  - `fp4_e2m1` requires CUDA 12.8+ and PyTorch 2.8+; experimental with documented accuracy drop warning.

### SGLang FP8 KV cache test

- **URL:** https://github.com/sgl-project/sglang/blob/main/test/manual/quant/test_fp8_kvcache.py
- **Level:** L2
- **What it contributes:**
  - Integration test launching server with `--kv-cache-dtype fp8_e4m3` + `--quantization-param-path`.
  - Accuracy gates: MMLU score ≥ 0.65 for Llama, ≥ 0.3 for Qwen — confirms FP8 KV is viable with calibrated scales.

### SGLang source code anchors

**Level:** L5 (source code)

| File | Lines | What it does |
|---|---|---|
| `python/sglang/srt/server_args.py` | ~4169–4174 | CLI flag `--kv-cache-dtype` with choices and CUDA version requirements |
| `python/sglang/srt/server_args.py` | ~4159–4166 | `--quantization-param-path` flag and accuracy warning |
| `python/sglang/srt/model_executor/model_runner.py` | ~2007–2053 | `configure_kv_cache_dtype()` — maps string → torch dtype; handles auto detection |
| `python/sglang/srt/model_executor/model_runner.py` | ~1288–1309 | Load per-layer scales from JSON after model load; warning if missing |
| `python/sglang/srt/layers/quantization/kv_cache.py` | 1–82 | `BaseKVCacheMethod` — `k_scale`, `v_scale` as parameters, loaded from checkpoint |
| `python/sglang/srt/mem_cache/memory_pool.py` | ~662–665 | FP8 stored as `uint8` (PyTorch `index_put` limitation) |
| `python/sglang/srt/mem_cache/memory_pool.py` | ~995–1032 | `set_kv_buffer()` — `cache_k.div_(k_scale)` then `.to(dtype)` then `.view(uint8)` |

---

## Primary sources: vLLM KV cache quantization

### vLLM `CacheDType` and quantization modes

- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/config/cache.py
- **Level:** L2–L3
- **What it contributes:**
  - Full type alias: `auto`, `float16`, `bfloat16`, `fp8`, `fp8_e4m3`, `fp8_e5m2`, `fp8_inc`, `fp8_ds_mla`, `turboquant_k8v4`, `turboquant_4bit_nc`, `turboquant_k3v4_nc`, `turboquant_3bit_nc`, `int8_per_token_head`, `fp8_per_token_head`, `nvfp4`.
  - `fp8_per_token_head` / `int8_per_token_head`: dynamic per-(token, head) scales computed at cache-write time — no calibration dataset needed.
  - `turboquant_*`: NVIDIA Research custom formats (K in INT8, V in INT4 or 3-bit); boundary layers skipped automatically.
  - `nvfp4`: packed FP4 + FP8 block scales; Blackwell SM100+ only (WIP).
  - `--kv-cache-dtype-skip-layers`: force `auto` for specific named layers (e.g. first/last layers with different activation statistics).

### vLLM `KVQuantMode` enum

- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/v1/kv_cache_interface.py
- **Level:** L3
- **What it contributes:**
  - `KVQuantMode.NONE` / `FP8_PER_TENSOR` / `INT8_PER_TOKEN_HEAD` / `FP8_PER_TOKEN_HEAD` / `NVFP4` — centralized dispatch enum replacing string matching.
  - `page_size_bytes` for `is_per_token_head`: adds `2 × block_size × num_kv_heads × 4 bytes` for FP32 scale storage per page.
  - NVFP4 uses `nvfp4_kv_cache_full_dim` packed last dimension.

### vLLM `BaseKVCacheMethod` + q/prob scales

- **URL:** https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/quantization/kv_cache.py
- **Level:** L3
- **What it contributes:**
  - Extends SGLang's `k_scale`/`v_scale` with **`q_scale`** and **`prob_scale`** — enabling fully end-to-end FP8 attention (quantized Q matmul + softmax probabilities).
  - Per-token-head path deletes checkpoint scales and uses 1.0 (scales computed dynamically in kernel).
  - Warning: `fp8_e4m3` with `k_scale == 1.0` logged as potential accuracy issue.

### vLLM `reshape_and_cache_flash` CUDA kernel

- **URL:** https://github.com/vllm-project/vllm/blob/main/csrc/cache_kernels.cu
- **Level:** L3–L4
- **What it contributes:**
  - Fused CUDA kernel: scale application + cast to FP8/NVFP4 + scatter into paged KV layout — all in one kernel vs SGLang's Python-level `div_ + to() + view`.
  - `k_scale` shape: `[1]` (per-tensor) **or** `[num_heads]` (per-head) — `kv_scale_stride` dispatches appropriately.
  - NVFP4 dispatches to `reshape_and_cache_nvfp4_dispatch` (SM100 build only); raises `TORCH_CHECK` on older hardware.

### vLLM vllm source code anchors

**Level:** L5 (source code)

| File | Lines | What it does |
|---|---|---|
| `vllm/config/cache.py` | 18–34 | `CacheDType` Literal type — full list of supported KV formats |
| `vllm/v1/kv_cache_interface.py` | 30–76 | `KVQuantMode` enum + `get_kv_quant_mode()` mapper |
| `vllm/v1/kv_cache_interface.py` | 124–155 | `AttentionSpec.page_size_bytes` — adds scale storage for per-token-head |
| `vllm/model_executor/layers/quantization/kv_cache.py` | 18–173 | `BaseKVCacheMethod` with q/k/v/prob scales |
| `vllm/v1/attention/backends/flash_attn.py` | 743–767 | FP8 KV read path: `view(fp8_dtype)` + `q_descale`/`k_descale`/`v_descale` into FlashAttention |
| `csrc/cache_kernels.cu` | 704–769 | `reshape_and_cache_flash` — fused scale+cast+scatter CUDA kernel |
| `vllm/v1/worker/gpu_model_runner.py` | 885–926 | `init_fp8_kv_scales()` — resets scales to 1.0 after sleep; known TODO for calibrated scale restore |

---

## Research papers: KV cache quantization

### KIVI — Tuning-Free Asymmetric 2-bit KV Cache Quantization

- **URL:** https://arxiv.org/abs/2402.02750
- **Published:** February 2024 (arXiv); NeurIPS 2024
- **Authors:** Jiayi Yuan et al. (UCSD)
- **GitHub:** https://github.com/jy-yuan/KIVI
- **Level:** L3
- **What it contributes:**
  - **Core finding**: Key cache has channel-wise outliers → must quantize **per-channel**. Value cache is smoother → can quantize **per-token**.
  - This asymmetric strategy (K per-channel, V per-token) at **2-bit** achieves near-lossless quality on Llama, Falcon, Mistral.
  - **2.6× peak memory reduction** (including model weights), **2.35–3.47× throughput** on real workloads.
  - Enables **4× larger batch sizes** for the same VRAM budget.
  - Hardware-friendly implementation without tuning or calibration data.
  - **Key insight for Layer 18**: the K and V tensors have fundamentally different statistical distributions and need different quantization strategies. This is why per-tensor FP8 (one scale for all K, one for all V) is coarser than necessary — and why vLLM's per-token-head mode is more principled.

- **BibTeX:**
  ```bibtex
  @article{yuan2024kivi,
    title={KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache},
    author={Jiayi Yuan and others},
    journal={arXiv preprint arXiv:2402.02750},
    year={2024}
  }
  ```

### KVQuant — Sub-4-bit KV Cache for 10M Context Windows

- **URL:** https://arxiv.org/abs/2401.18079
- **Published:** January 2024 (arXiv); NeurIPS 2024
- **Authors:** Coleman Hooper et al. (UC Berkeley)
- **Level:** L3–L4
- **What it contributes:**
  - Four novel techniques to make sub-4-bit quantization viable:
    1. **Per-Channel Key Quantization** — quantize along the channel (head) dimension for K.
    2. **Pre-RoPE Key Quantization** — quantize K *before* rotary positional embedding to avoid positional encoding corrupting the quantization range.
    3. **Non-Uniform KV Cache Quantization** — per-layer sensitivity-weighted non-uniform datatypes.
    4. **Per-Vector Dense-and-Sparse Quantization** — isolate outliers separately per vector to minimize range skew.
  - **Result:** < 0.1 perplexity degradation at 3-bit on Wikitext-2 and C4.
  - Enables **1 million token context** on a single A100-80GB; **10 million tokens** on 8-GPU system.
  - Custom CUDA kernels: **~1.7× speedup** vs baseline FP16 matrix-vector multiply.
  - **Key insight for Layer 18**: Pre-RoPE quantization is a subtle but important detail — the rotary embedding changes the distribution of K after projection, making post-RoPE quantization harder. This explains why checkpoint-based scales (calibrated post-RoPE) can be inaccurate.

- **BibTeX:**
  ```bibtex
  @article{hooper2024kvquant,
    title={KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization},
    author={Coleman Hooper and others},
    journal={arXiv preprint arXiv:2401.18079},
    year={2024}
  }
  ```

### ZipCache — Salient Token Identification + Channel-Separable Quantization

- **URL:** https://arxiv.org/abs/2405.14256
- **Published:** May 2024 (arXiv)
- **Authors:** Yefei He et al.
- **Level:** L3
- **What it contributes:**
  - **Channel-separable tokenwise quantization**: reduces memory overhead of quantization parameters vs fine-grained groupwise quantization.
  - **Normalized attention score** as salient token metric: considers the lower-triangular structure of the attention matrix to identify which tokens are most important to preserve at full precision.
  - **Decoupled saliency metric**: doesn't require full attention scores — compatible with FlashAttention implementations.
  - **4.98× compression** on Mistral-7B / GSM8K with only **0.38% accuracy drop**.
  - **37.3% prefill latency reduction**, **56.9% decode latency reduction**, **19.8% GPU memory reduction** on LLaMA3-8B with 4096 input tokens.
  - **Key insight for Layer 18**: Mixes quantization with token dropping (salient tokens kept full-precision, others quantized heavily). This is the "mixed-precision" approach — the complement to uniform quantization (all tokens same dtype).

- **BibTeX:**
  ```bibtex
  @article{he2024zipcache,
    title={ZipCache: Accurate and Efficient KV Cache Quantization with Salient Token Identification},
    author={Yefei He and others},
    journal={arXiv preprint arXiv:2405.14256},
    year={2024}
  }
  ```

### SageAttention2 — INT4 QK + FP8 PV Attention Compute

- **URL:** https://arxiv.org/abs/2411.10958
- **Published:** November 2024 (arXiv); ICML 2025
- **Authors:** Jintao Zhang et al. (Tsinghua University)
- **GitHub:** https://github.com/thu-ml/SageAttention
- **Level:** L3
- **What it contributes:**
  - Quantizes the **attention computation itself** (not just KV storage): Q and K to **INT4** (thread-level granularity), P̃ and V to **FP8**.
  - **Outlier smoothing for Q**: preprocessing Q to reduce INT4 quantization error.
  - **Two-level accumulation** for FP8 P̃V to maintain accuracy.
  - **~3× faster than FlashAttention2**, ~4.5× faster than xformers on RTX 4090.
  - Matches FlashAttention3 (fp8) speed on Hopper GPUs with much higher accuracy.
  - **Negligible end-to-end metric loss** on language, image, and video generation models.
  - **Key insight for Layer 18**: SageAttention2 quantizes the *compute path* not just the *storage*. This is orthogonal to KV cache storage quantization — you can combine FP8 KV storage (for memory efficiency) with INT4/FP8 attention compute (for FLOP throughput). This is the direction vLLM's `q_scale`/`prob_scale` is heading.

- **BibTeX:**
  ```bibtex
  @inproceedings{zhang2024sageattention2,
    title={SageAttention2: Efficient Attention with Thorough Outlier Smoothing and Per-thread INT4 Quantization},
    author={Jintao Zhang and others},
    booktitle={ICML 2025},
    year={2025}
  }
  ```

---

## Benchmarks and surveys

### KV Cache Compression — Long Context Benchmark

- **URL:** https://arxiv.org/abs/2407.01527
- **Published:** July 2024 (arXiv)
- **Authors:** Jiayi Yuan et al.
- **GitHub:** https://github.com/henryzhongsc/longctx_bench
- **Level:** L4
- **What it contributes:**
  - Comprehensive benchmark of **10+ KV compression approaches** across **7 categories** of long-context tasks.
  - Taxonomy: KV cache quantization, token dropping, prompt compression, linear-time models, hybrid architectures.
  - **Key finding**: Different compression methods have complementary failure modes. Quantization preserves all tokens at lower precision; token dropping removes less-important tokens at full precision — neither dominates.
  - Side-by-side comparison in a controlled environment reveals **previously unknown phenomena** about how methods interact with task type.
  - **Key insight for Layer 18**: Places KV cache quantization in the broader KV compression landscape. Quantization is the "lossless across tokens, lossy in precision" approach, while token dropping is "lossless in precision, lossy across tokens." Both reduce memory; which you choose depends on workload.

- **BibTeX:**
  ```bibtex
  @article{yuan2024kvcompbench,
    title={KV Cache Compression, But What Must We Give in Return? A Comprehensive Benchmark of Long Context Capable Approaches},
    author={Jiayi Yuan and others},
    journal={arXiv preprint arXiv:2407.01527},
    year={2024}
  }
  ```

### KV Cache Management Survey — Token/Model/System Taxonomy

- **URL:** https://arxiv.org/abs/2412.19442
- **Published:** December 2024; revised July 2025
- **Authors:** Haoyang Li et al. (TreeAI Lab)
- **GitHub:** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
- **Level:** L4
- **What it contributes:**
  - 200+ paper survey categorized as token-level, model-level, system-level.
  - **Token-level quantization** section: KIVI, KVQuant, ZipCache, Gear, WKVQuant, and many more — with comparative analysis.
  - Frames KV cache quantization as a **token-level** optimization (changes how individual token KV values are stored) vs **system-level** (HiCache, LMCache) which changes where they're stored.
  - Quantization can be combined with system-level tiering: FP8 KV in GPU VRAM + FP8 KV in CPU RAM = half the PCIe bandwidth needed for offloading.
  - The curated paper list is the best starting point for finding new quantization methods.

- **BibTeX:**
  ```bibtex
  @article{li2024kvcachesurvey,
    title={A Survey on Large Language Model Acceleration based on KV Cache Management},
    author={Haoyang Li and others},
    journal={arXiv preprint arXiv:2412.19442},
    year={2024}
  }
  ```

### LLM Inference Optimization Survey — Quantization, Pruning, Distillation

- **URL:** https://arxiv.org/abs/2408.03130
- **Published:** August 2024 (arXiv)
- **Authors:** Leo Donisch, Sigurd Schacht, Carsten Lanquillon
- **Level:** L4
- **What it contributes:**
  - Taxonomy of LLM optimization: quantization (weight, activation, KV cache), pruning, knowledge distillation, architectural.
  - KV cache quantization placed in context of **activation quantization** — KV caches are intermediate activations, not weights, so they have different statistical properties (dynamic range varies by token/input, unlike static weights).
  - Explains why **post-training quantization (PTQ) of KV cache** is harder than weight PTQ: scales must handle unseen input distributions at inference time.
  - **Key insight for Layer 18**: KV cache quantization is fundamentally different from weight quantization. Weights are fixed after training; KV values change every request. This is why per-token-head dynamic scales (vLLM) are more robust than per-tensor calibrated scales (SGLang).

---

## Related: weight quantization methods that include KV calibration

### AWQ — Activation-Aware Weight Quantization

- **URL:** https://arxiv.org/abs/2306.00978
- **Published:** June 2023 (arXiv); MLSys 2024
- **Authors:** Ji Lin et al. (MIT HAN Lab)
- **Level:** L4
- **Why here (related, not directly):**
  - AWQ produces calibrated weight-only quantization using activation statistics. Many FP8-quantized checkpoints (e.g., Llama-3.1-FP8, Qwen2.5-FP8) include **KV cache scaling factors** alongside AWQ-style weight scales in their `quantization_config.json`.
  - Understanding AWQ explains *why* the `--quantization-param-path` JSON exists: the calibration pipeline (e.g., llm-compressor) runs AWQ or similar methods, collects activation statistics, and outputs per-layer `k_scale`/`v_scale` alongside weight scales.
  - The framework that generates KV scales is the same framework that generates weight scales.

---

## Tooling and frameworks

### llm-compressor (Neural Magic / Red Hat)

- **URL:** https://github.com/vllm-project/llm-compressor
- **Level:** L2–L3
- **What it contributes:**
  - The primary tool for generating `quantization_param_path` JSON files for SGLang and vLLM FP8 KV cache.
  - Runs a calibration forward pass through the model on a representative dataset, collects per-layer max-abs activation statistics, and outputs `k_scale`/`v_scale` per attention layer.
  - Supports FP8 weight + FP8 KV joint calibration (what the official Meta and Qwen FP8 checkpoints on Hugging Face use).
  - **Key command:**
    ```python
    from llmcompressor.modifiers.quantization import QuantizationModifier
    recipe = QuantizationModifier(
        targets="Linear",
        scheme="FP8",
        ignore=["lm_head"],
        kv_cache_scheme={"type": "float8", "strategy": "tensor"}
    )
    ```

### Hugging Face FP8 model checkpoints

- **URL:** https://huggingface.co/models?search=fp8
- **Level:** L2
- **What it contributes:**
  - Meta's `meta-llama/Meta-Llama-3.1-8B-Instruct-FP8` and similar models include `quantization_config.json` with `kv_cache_quant_algo: "FP8"` — SGLang's `--kv-cache-dtype auto` picks this up automatically.
  - The `config.json` or `quantization_config.json` pattern is what SGLang's `configure_kv_cache_dtype()` reads to auto-detect FP8 KV.
  - These checkpoints serve as the reference for "FP8 KV with calibrated scales" — the production-recommended configuration.

---

## The quantization design space: a map

```
KV Cache Quantization Approaches
├── By precision
│   ├── FP8 (e4m3 / e5m2) — 1 byte       ← SGLang, vLLM: --kv-cache-dtype fp8_e4m3
│   ├── INT8 per-token-head               ← vLLM: --kv-cache-dtype int8_per_token_head
│   ├── FP4 / NVFP4 — 0.5 byte           ← vLLM: nvfp4 (Blackwell only); SGLang: fp4_e2m1
│   ├── INT4 (K) + FP8 (PV) compute      ← SageAttention2 (compute path)
│   └── 2-bit / 3-bit KV                 ← KIVI, KVQuant (research, not yet in engines)
│
├── By scale granularity
│   ├── Per-tensor (one scale per layer)  ← SGLang FP8, vLLM fp8_e4m3
│   ├── Per-channel (one scale per head)  ← KIVI K-cache, vLLM per-head scale
│   ├── Per-token-head (one per token×head) ← vLLM fp8_per_token_head (dynamic, no calibration)
│   └── Per-vector / group               ← KVQuant, ZipCache
│
├── By scale source
│   ├── Calibrated from checkpoint       ← --quantization-param-path JSON
│   ├── Dynamic at runtime               ← vLLM per-token-head, SageAttention2
│   └── Default 1.0 (uncalibrated)       ← works but risks accuracy loss for fp8_e4m3
│
└── Combined with token selection
    ├── Uniform (all tokens quantized equally) ← FP8 KV cache
    └── Mixed precision (salient full, rest quantized) ← ZipCache
```

---

## Recommended reading order

**Fast path — enable FP8 KV today (30 min):**
1. SGLang `--kv-cache-dtype` CLI docs
2. SGLang source: `configure_kv_cache_dtype()` + `set_kv_buffer()`
3. KIVI abstract (understand the K per-channel, V per-token insight)

**Standard path — understand the design space (2 hours):**
1. SGLang + vLLM source anchors (both engines)
2. KIVI full paper (2-bit, asymmetric quantization, distribution analysis)
3. KVQuant paper (sub-4-bit, pre-RoPE quantization)
4. KV cache management survey (taxonomy, situate in landscape)

**Research path — push the frontier (6+ hours):**
1. All of the above
2. ZipCache (mixed-precision + salient token identification)
3. SageAttention2 (INT4/FP8 compute quantization, not just storage)
4. Long-context compression benchmark (controlled comparison across methods)
5. LLM inference optimization survey (broader context)

---

## What Layer 18 defers

The following are mentioned but not fully covered in Layer 18:

- **Weight quantization** (GPTQ, AWQ, FP8 weight) — Layer 18 focuses on KV cache (activation) quantization only.
- **Token dropping** (H2O, StreamingLLM, SnapKV) — these remove tokens rather than quantize; covered partially in the KV cache survey.
- **Speculative decoding + quantization interactions** — an open research area.
- **Training-aware KV quantization** (QAT for KV cache) — all methods here are PTQ.
- **Hardware-specific formats** (Blackwell NVFP4, Grace-Hopper NVLink) — mentioned but hardware details deferred.
