# References — Multi-LoRA Inference

Organized by **reading level** (L1–L5) and **category**. Use this when writing or extending lesson content, understanding production precedents, or designing exercises.

Layer 20 covers **single-adapter and multi-LoRA inference**: the technique of serving multiple LoRA fine-tuned variants of a base model in a single batched forward pass. References therefore cover: **the LoRA fine-tuning method itself, the kernel mathematics of batched multi-adapter inference (segmented GEMM / SGMV), the two seminal serving systems (Punica and S-LoRA), production implementations in SGLang and vLLM, the LoRAX open-source server, advanced systems (dLoRA, CaraServe, InfiniLoRA), serverless LoRA challenges (ServerlessLoRA, Predictive-LoRA), unified fine-tuning + serving (Loquetier), and the HuggingFace PEFT ground truth used in verification.**

---

## Quick navigation

| Reading level | What you'll find |
|---|---|
| **L1** | Blog posts and practitioner guides: LoRAX technical analysis, SageMaker multi-LoRA walkthrough, intro to serving fine-tuned variants at scale |
| **L2** | Production framework docs: SGLang LoRA serving, vLLM LoRA serving, HuggingFace PEFT reference |
| **L3** | Foundational papers: LoRA original (Hu et al. 2021), Punica (MLSys 2024), S-LoRA (MLSys 2024) |
| **L4** | Research systems: dLoRA (OSDI 2024), CaraServe (2024), Loquetier (NeurIPS 2025), InfiniLoRA (2026), ServerlessLoRA (2025), Predictive-LoRA (2025) |
| **L5** | Surveys and source code anchors: comprehensive LoRA review, SGLang implementation internals, vLLM Punica wrapper |

---

## Primary sources: LoRA foundations

### LoRA: Low-Rank Adaptation of Large Language Models

- **arXiv:** https://arxiv.org/abs/2106.09685
- **Published:** June 2021 (arXiv); accepted ICLR 2022
- **Authors:** Edward Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
- **GitHub:** https://github.com/microsoft/LoRA
- **Level:** L3 (foundational)
- **What it contributes:**
  - **The foundational paper for all LoRA serving work.** Proposes injecting trainable rank-decomposed matrices `A ∈ ℝ^{r×d}` and `B ∈ ℝ^{d×r}` into frozen transformer layers, where `r ≪ min(d_in, d_out)`.
  - For GPT-3 175B: LoRA at `r=4` reduces trainable parameters by **10,000×** and GPU memory by **3×** vs full fine-tuning, with on-par or better downstream task quality.
  - Introduces `scaling = lora_alpha / r` as a hyperparameter controlling delta magnitude independent of rank — used verbatim in all subsequent implementations.
  - The key inference insight: no additional latency vs the merged weight matrix `W + BA`, because the delta can be pre-merged at deploy time or computed as a residual add. Serving with un-merged adapters enables switching between adapters without weight reloading.
  - **Why it matters for Layer 20**: every line of `lora.py` (`apply()`, `scaling`, `A_weights`, `B_weights`) directly implements this paper's Equation (3). The PEFT checkpoint format is derived from this paper's conventions.

- **BibTeX:**
  ```bibtex
  @article{hu2021lora,
    title={LoRA: Low-Rank Adaptation of Large Language Models},
    author={Edward J. Hu and others},
    journal={arXiv preprint arXiv:2106.09685},
    year={2021}
  }
  ```

### HuggingFace PEFT — Parameter-Efficient Fine-Tuning Library

- **URL:** https://huggingface.co/docs/peft/en/index
- **LoRA guide:** https://huggingface.co/docs/peft/en/developer_guides/lora
- **GitHub:** https://github.com/huggingface/peft
- **Level:** L2 (production reference)
- **What it contributes:**
  - The canonical open-source implementation of LoRA (and other PEFT methods) for HuggingFace models. `PeftModel` wraps any `AutoModelForCausalLM` with LoRA adapters using a simple two-call API: `LoraConfig(r=8, lora_alpha=32, target_modules=[...])` + `get_peft_model(base, config)`.
  - **Checkpoint format**: PEFT saves adapters as `adapter_model.safetensors` with keys like `base_model.model.model.layers.{i}.self_attn.q_proj.lora_A.weight` — the exact format parsed by `lora.py:_load_weights()` in this layer.
  - **Ground truth for verification**: `verify_lora.py` loads a `PeftModel` and compares logits against our custom implementation. Near-identical deltas (`A max_diff ≤ 0.00012`, `B max_diff ≤ 0.000002`) confirmed our weight loading and `apply()` are correct.
  - Supports dynamic rank selection, DoRA, LoRA+, and quantized LoRA (QLoRA) as extensions of the base LoRA approach.
  - PEFT's inference path is: `base_out = linear(x)` → `lora_out = lora_B(lora_A(x)) * scaling` → `return base_out + lora_out`. This is identical to our `apply()` implementation.

---

## Seminal multi-LoRA serving systems

### Punica: Multi-Tenant LoRA Serving

- **Paper:** Chen et al., "Punica: Multi-Tenant LoRA Serving," MLSys 2024
- **MLSys proceedings:** https://proceedings.mlsys.org/paper_files/paper/2024/hash/054de805fcceb78a201f5e9d53c85908-Abstract-Conference.html
- **PDF:** https://www.yongjiwu.me/assets/pdf/mlsys24-punica.pdf
- **GitHub:** https://github.com/punica-ai/punica
- **Published:** 2024 (MLSys '24)
- **Authors:** Lequn Chen, Zihao Ye, Yongji Wu, Danyang Zhuo, Luis Ceze, Arvind Krishnamurthy (University of Washington)
- **Level:** L3 (seminal)
- **What it contributes:**
  - **The foundational kernel paper for multi-LoRA serving.** Introduces **Segmented Gather Matrix-Vector Multiplication (SGMV)** — a custom CUDA kernel that batches LoRA delta computation across requests using different adapters in a single kernel launch.
  - **Core observation**: during decode, each request generates one token and the serving bottleneck is memory bandwidth, not compute. Running a separate GEMM per adapter wastes kernel launch overhead. SGMV amortises this by processing all adapters in one pass, gathering the correct A/B matrices per request from a pool.
  - **SGMV kernel**: takes `x[i]` (input for request `i`), `A[w[i]]` and `B[w[i]]` (adapter weights indexed by `w[i]` from a weight pool), and accumulates `(x @ A.T) @ B.T * scaling` for each segment in one CUDA kernel.
  - **Result**: 12× higher throughput vs state-of-the-art LoRA serving at the time; adds only 2ms latency per token vs dedicated per-adapter serving.
  - **Weight pool design**: all adapter A and B matrices are pre-loaded into a contiguous GPU buffer; `w[i]` is a per-token index into this buffer — this is the direct predecessor of S-LoRA's unified paging and SGLang's `lora_b_weights` pool.
  - **Integrated into vLLM**: vLLM's `PunicaWrapperGPU` class wraps the SGMV kernel; SGLang uses a variant as its `csgmv` LoRA backend.
  - **Why it matters for Layer 20**: our `apply()` is the naive version of what SGMV does; Section 02 of this layer explains how the mask-based approach compares to SGMV.

- **BibTeX:**
  ```bibtex
  @inproceedings{chen2024punica,
    title={Punica: Multi-Tenant LoRA Serving},
    author={Lequn Chen and others},
    booktitle={MLSys 2024},
    year={2024}
  }
  ```

### S-LoRA: Serving Thousands of Concurrent LoRA Adapters

- **Paper:** Sheng et al., "S-LoRA: Serving Thousands of Concurrent LoRA Adapters," MLSys 2024
- **MLSys proceedings:** https://proceedings.mlsys.org/paper_files/paper/2024/hash/906419cd502575b617cc489a1a696a67-Abstract-Conference.html
- **arXiv:** https://arxiv.org/abs/2311.03285
- **GitHub:** https://github.com/S-LoRA/S-LoRA
- **Project page:** https://sky.cs.berkeley.edu/project/s-lora/
- **Published:** 2024 (MLSys '24)
- **Authors:** Ying Sheng, Shiyi Lin, Joseph E. Gonzalez, Ion Stoica, Lianmin Zheng (UC Berkeley, Stanford)
- **Level:** L3 (seminal)
- **What it contributes:**
  - **The foundational memory management paper for multi-LoRA serving.** Addresses the scale problem: how do you serve *thousands* of adapters when they cannot all fit in GPU VRAM simultaneously?
  - **Unified paging**: extends the paged KV cache idea to adapter weights. A single unified memory pool manages both KV cache pages and adapter A/B weight pages, eliminating external fragmentation.
  - **Adapter eviction**: adapters not currently in use are offloaded to CPU RAM via LRU eviction; loaded on-demand from a background thread (overlap H2D transfer with compute).
  - **Heterogeneous batching**: a single forward pass can serve requests using different adapters at different ranks — the GEMM computation is rank-heterogeneous, handled by custom CUDA kernels.
  - **Tensor parallelism for LoRA**: novel TP sharding strategy for LoRA weights: A matrices are replicated (not sharded) because they are small; B matrices are sharded column-wise for QKV and row-wise for O — different from base model TP sharding.
  - **Results**: serves up to **2000 adapters simultaneously**, 4× higher throughput than HuggingFace PEFT, several orders of magnitude more adapters than vLLM.
  - **SGLang implementation**: `sglang_multi_lora_implementation.md` documents SGLang's production implementation, which builds on S-LoRA's unified paging and Punica's SGMV kernels.
  - **Why it matters for Layer 20**: the gap between our minimal mask-based implementation (one static adapter) and S-LoRA (thousands of adapters, memory pool, LRU eviction) is the full scope of what `sglang_multi_lora_implementation.md` covers.

- **BibTeX:**
  ```bibtex
  @inproceedings{sheng2024slora,
    title={{S-LoRA}: Serving Thousands of Concurrent LoRA Adapters},
    author={Ying Sheng and others},
    booktitle={MLSys 2024},
    year={2024}
  }
  ```

---

## Production framework implementations

### SGLang LoRA Serving Documentation

- **URL:** https://sgl-project.github.io/advanced_features/lora.html
- **GitHub (initial PR):** https://github.com/sgl-project/sglang/pull/1307
- **Dynamic load/unload PR:** https://github.com/sgl-project/sglang/pull/7446
- **Level:** L2
- **What it contributes:**
  - Complete configuration reference for SGLang's production multi-LoRA serving, based on S-LoRA unified paging + Punica/cSGMV kernels.
  - **Key server arguments**:
    - `--lora-paths`: list of adapters to load at startup (can include HF Hub IDs)
    - `--max-loras-per-batch`: maximum adapters active per batch (default: 8)
    - `--max-lora-rank`: maximum rank supported; auto-inferred from loaded adapters
    - `--lora-backend`: GEMM backend (`triton` or `csgmv`, default: `csgmv`)
    - `--lora-eviction-policy`: `lru` (LRU, default) or `fifo`
    - `--enable-lora-overlap-loading`: overlap H2D transfer with GPU compute
  - **OpenAI-compatible API**: `model` field syntax `"base_model_name:adapter_name"` in `/v1/chat/completions` — clients select the adapter per-request without server changes.
  - **Dynamic loading** (added 2025): `/load_lora_adapter` and `/unload_lora_adapter` endpoints allow adapter hot-swap without server restart.
  - **cSGMV backend**: SGLang's default LoRA GEMM kernel; a variant of Punica SGMV with chunk-based processing (`--max-lora-chunk-size 16`) to control memory usage.
  - **Why it matters for Layer 20**: the implementation this layer's code is a simplified version of. `sglang_multi_lora_implementation.md` documents the internals at the code level.

### vLLM Multi-LoRA Serving

- **URL:** https://docs.vllm.ai/en/latest/features/lora.html
- **Punica wrapper API:** https://docs.vllm.ai/en/latest/api/vllm/lora/punica_wrapper/
- **Level:** L2
- **What it contributes:**
  - vLLM's production LoRA serving, using the Punica SGMV kernel (`PunicaWrapperGPU`) as the default GEMM backend.
  - **Configuration**: `--enable-lora`, `--max-loras` (max adapters in GPU memory), `--max-lora-rank`, `--max-cpu-loras` (max adapters in CPU memory for LRU).
  - **API**: per-request adapter selection via `model` field in the API request (same syntax as SGLang: `base:adapter`).
  - `PunicaWrapperBase` manages token-to-adapter index tensors (`lora_indices_t`, `sampler_indices_t`) and provides the interface to the SGMV kernel — the production equivalent of our `lora_mask` approach.
  - The `compute_meta()` function collapses consecutive same-adapter tokens into single SGMV segments — the key optimisation that eliminates per-token GEMM launches for requests sharing an adapter.
  - Supports CPU platforms (XPU, HPU) via `PunicaWrapperXPU` and `PunicaWrapperCPU` subclasses, with platform-specific GEMM implementations.

### HuggingFace PEFT LoRA Developer Guide

- **URL:** https://huggingface.co/docs/peft/en/developer_guides/lora
- **Level:** L2 (reference)
- **What it contributes:**
  - Detailed walkthrough of PEFT's LoRA implementation: `LoraConfig`, `get_peft_model()`, `PeftModel.from_pretrained()`, and the checkpoint save/load API.
  - **target_modules**: how PEFT identifies which linear layers to adapt — by name string match against module names in the base model. This explains why adapter checkpoint keys contain the full module path (e.g., `self_attn.q_proj`).
  - **lora_A is always initialized randomly (Gaussian); lora_B is initialized to zero** — the adapter output is exactly zero at initialization, matching the base model output. Training nudges B away from zero. This is why in verification, a freshly initialized (untrained) adapter produces zero delta.
  - **DoRA** (Weight-Decomposed Low-Rank Adaptation): a PEFT extension that decomposes the weight update into magnitude and direction components; produces better quality than vanilla LoRA at the same rank but adds a small inference overhead.
  - **QLoRA integration**: PEFT supports attaching LoRA to bitsandbytes 4-bit quantized base models — the adapter is stored in BF16 while the base model is in NF4/FP4.

### LoRAX — Multi-Adapter Inference Server

- **GitHub:** https://github.com/predibase/lorax
- **Technical analysis:** https://pub.towardsai.net/the-architectural-paradigm-of-multi-adapter-inference-a-technical-analysis-of-lorax-567c2f4851f0
- **Blog:** https://medium.com/@saimudhiganti/lorax-serve-1000-fine-tuned-models-on-one-gpu-heres-how-62336a64de4b
- **Level:** L1–L2
- **What it contributes:**
  - LoRAX is an open-source inference server (Predibase) purpose-built for multi-LoRA serving, with Docker-ready deployment and an OpenAI-compatible API. It is the most accessible production-ready multi-LoRA server for practitioners.
  - **"Dedicated Model Problem" framing**: LoRAX's key conceptual contribution is naming the inefficiency of deploying one GPU-per-adapter. With LoRAX, one GPU holds one base model and serves hundreds of adapters as "lightweight dynamic overlays."
  - **Dynamic just-in-time loading**: adapters are loaded from HuggingFace Hub or S3 on first request without server restart — the first request for an adapter incurs a cold-start penalty, subsequent requests use the cached adapter.
  - **Supports**: LLaMA, Mistral, Qwen, and other models; 4-bit quantization; multiple concurrent adapters; streaming responses.
  - **Architecture**: base model stays resident; adapters are swapped in/out of GPU memory using a weight pool and LRU eviction — the same design as S-LoRA.
  - **SageMaker deployment**: Amazon SageMaker integrates LoRAX as a managed endpoint option, enabling "serve 100 fine-tuned models for the price of 1" — a practical cloud deployment reference.

---

## Research systems: advancing the state of the art

### dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving

- **Paper:** Wu et al., "dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving," USENIX OSDI 2024
- **USENIX page:** https://www.usenix.org/conference/osdi24/presentation/wu-bingyang
- **GitHub:** https://github.com/LLMServe/dLoRA-artifact
- **Published:** July 2024 (USENIX OSDI '24)
- **Authors:** Bingyang Wu, Ruidong Zhu, Zili Zhang, Peng Sun, Xuanzhe Liu, Xin Jin (Peking University, Shanghai AI Lab)
- **Level:** L4
- **What it contributes:**
  - Identifies two new inefficiencies in S-LoRA's static batching approach:
    1. **Merge/unmerge tradeoff**: when all requests in a batch use the same adapter, it is more efficient to *merge* the adapter weights into the base model (eliminating separate GEMM) than to run S-LoRA's segmented path. S-LoRA always uses the segmented path regardless of request mix.
    2. **Load imbalance**: autoregressive generation produces variable-length outputs, causing worker replicas to finish at different times even with uniform input distribution.
  - **Credit-based batching algorithm**: dynamically decides when to merge vs unmerge based on the fraction of requests using each adapter, avoiding unnecessary GEMM overhead.
  - **Request-adapter co-migration**: when a worker becomes overloaded with long-decode requests, dLoRA migrates both the request state *and* the adapter weights to a less-loaded replica — maintaining adapter affinity across migration.
  - **Results**: up to **57.9× higher throughput** vs vLLM, **26.0×** vs HuggingFace PEFT, **1.8× lower average latency** vs S-LoRA on mixed-adapter workloads.
  - **OSDI 2024** — top-tier systems venue (concurrent with S-LoRA's MLSys publication).

- **BibTeX:**
  ```bibtex
  @inproceedings{wu2024dlora,
    title={dLoRA: Dynamically Orchestrating Requests and Adapters for LoRA LLM Serving},
    author={Bingyang Wu and others},
    booktitle={USENIX OSDI 2024},
    year={2024}
  }
  ```

### CaraServe: CPU-Assisted and Rank-Aware LoRA Serving

- **Paper:** Li et al., "CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference"
- **arXiv:** https://arxiv.org/abs/2401.11240
- **Published:** January 2024
- **Level:** L4
- **What it contributes:**
  - Addresses the **cold-start problem** in multi-LoRA serving: when an adapter is requested but not in GPU memory, the request must wait for H2D transfer before execution can begin. S-LoRA masks this with background prefetching, but still incurs head-of-line blocking.
  - **CPU-assisted execution**: during the prefill phase, CaraServe begins LoRA computation on the CPU (where the adapter is already resident in DRAM) while the H2D transfer is in flight. Once the adapter arrives on GPU, execution switches to GPU for the token-generation (decode) phase. Net effect: prefill latency includes CPU computation but hides the H2D transfer.
  - **Rank-aware preemptive scheduling**: ranks vary across adapters (e.g., some requests use `r=8`, others `r=64`). High-rank adapters require more compute per token. CaraServe schedules requests with awareness of rank, prioritising SLO-critical requests and preempting lower-priority ones.
  - **Results**: **1.4× speedup** in average serving latency vs S-LoRA; **up to 99% SLO attainment** across diverse rank distributions.
  - Directly relevant to Layer 20: our implementation does not handle cold-start because the adapter is loaded once at startup. CaraServe is the production answer to cold-start at scale.

- **BibTeX:**
  ```bibtex
  @article{li2024caraserve,
    title={CaraServe: CPU-Assisted and Rank-Aware LoRA Serving for Generative LLM Inference},
    author={Li and others},
    journal={arXiv preprint arXiv:2401.11240},
    year={2024}
  }
  ```

### Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving

- **arXiv:** https://arxiv.org/abs/2511.00101
- **GitHub:** https://github.com/NJUDeepEngine/Loquetier
- **Published:** October 2025 (arXiv); NeurIPS 2025 (poster)
- **Authors:** NJU Deep Engine team
- **Level:** L4
- **What it contributes:**
  - Addresses a gap left by S-LoRA and Punica: they serve inference-only workloads. In practice, models are continuously fine-tuned as new data arrives — fine-tuning and inference workloads need to coexist on the same GPU cluster.
  - **Virtualized module**: a runtime abstraction that isolates per-adapter PEFT modifications; multiple adapters share the frozen base model with no weight copying.
  - **SMLM (Segmented Multi-LoRA Multiplication) kernel**: extends SGMV to merge the fine-tuning and inference computation paths, enabling batching of both gradient-required and inference-only tokens in the same kernel invocation.
  - **Results**: up to **3.0× inference throughput** vs S-LoRA; **46.4× higher SLO attainment** vs HuggingFace PEFT on unified fine-tuning + inference tasks.
  - **NeurIPS 2025** — top-tier ML venue.

- **BibTeX:**
  ```bibtex
  @inproceedings{loquetier2025,
    title={Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving},
    booktitle={NeurIPS 2025},
    year={2025}
  }
  ```

### ServerlessLoRA: Minimizing Latency and Cost in Serverless Inference for LoRA-Based LLMs

- **arXiv:** https://arxiv.org/abs/2505.14468
- **URL:** https://arxiv.org/html/2505.14468v1
- **Published:** May 2025
- **Level:** L4
- **What it contributes:**
  - **Serverless LoRA problem**: cloud functions (AWS Lambda, Azure Functions) invoke LoRA inference on-demand with cold starts. 99% of parameters are duplicated across LoRA functions (shared base model) but each function loads its own copy — causing massive redundancy, long TTFT, and resource contention.
  - **Backbone sharing**: a shared base model process handles all LoRA functions; function-specific A/B matrices are loaded into the shared process's GPU memory on demand.
  - **Contention-aware batching**: when multiple functions request the same base model simultaneously, ServerlessLoRA batches them together and applies per-function masks — conceptually similar to our `lora_mask` but across serverless function boundaries.
  - **Results**: **86% reduction in TTFT**, **89% reduction in cost** vs independent per-function loading.
  - **Why it matters**: demonstrates that the "backbone sharing + per-request LoRA selection" pattern (which our Layer 20 implements) is the correct abstraction for cloud/serverless LoRA at scale.

### Predictive-LoRA: Proactive Adapter Management for Serverless Inference

- **arXiv:** https://arxiv.org/abs/2512.20210
- **URL:** https://arxiv.org/html/2512.20210v1
- **Published:** December 2025
- **Level:** L4
- **What it contributes:**
  - Addresses S-LoRA's reactive eviction policy: adapters are only evicted when memory pressure forces it, causing cold-start delays when they are needed again.
  - **LSTM-based traffic prediction**: predicts which adapters will be requested next based on historical traffic patterns; proactively prefetches predicted adapters before their requests arrive, hiding the H2D transfer latency entirely.
  - **Page-based adapter memory management**: adapters are stored and evicted at page granularity (not all-or-nothing), reducing fragmentation and improving GPU utilisation to above 87%.
  - **Results**: **1.52× higher throughput** vs S-LoRA, **35% reduction in average TTFT**, **up to 68% reduction in cold-start latency**.

### InfiniLoRA: Disaggregated Multi-LoRA Serving for Large Language Models

- **arXiv:** https://arxiv.org/abs/2604.07173
- **URL:** https://arxiv.org/html/2604.07173v1
- **Published:** April 2026
- **Authors:** Researchers from SJTU, ByteDance, SMU, HKUST, NUS
- **Level:** L4
- **What it contributes:**
  - **Motivation**: as MoE models (DeepSeek-V3, Mixtral) become dominant, LoRA adapter sizes explode — a single LoRA for DeepSeek-V3 (671B, 256 experts) is orders of magnitude larger than for a dense 7B model. Existing coupled LoRA serving designs (S-LoRA, dLoRA) struggle because the LoRA GEMM latency becomes comparable to the base model forward pass.
  - **Disaggregated LoRA serving**: decouples the LoRA computation from the base model inference by introducing a **shared LoRA Server** — a separate set of GPU workers that handle only the A/B matrix GEMMs while the base model workers run the transformer layers. The two communicate via GPU-initiated RDMA.
  - **SLO-driven provisioning**: the LoRA Server's capacity is automatically scaled based on the latency SLO, adapter ranks, and request rates.
  - **Hardware-specialized LoRA kernels**: exploits the Tensor Memory Accelerator (TMA) on Hopper GPUs for the segmented LoRA GEMM — achieves memory-bandwidth-limited performance for the SGMV operation.
  - **Results**: **3.05× average increase in serviceable request rate** under strict latency SLOs; **54% more adapters satisfying SLO requirements**.
  - **Why it matters**: InfiniLoRA represents the next evolutionary step — applying PD disaggregation ideas (Layer 19) to LoRA serving (Layer 20) in response to MoE model scale.

---

## Surveys: LoRA in context

### Low-Rank Adaptation for Foundation Models: A Comprehensive Review

- **arXiv:** https://arxiv.org/abs/2501.00365
- **Published:** December 2024 / January 2025
- **Level:** L5 (survey)
- **What it contributes:**
  - **Comprehensive taxonomy of LoRA variants** across training, fine-tuning, and inference:
    - Architectural variants: DoRA (magnitude-direction decomposition), LoHa (Hadamard product), VeRA (shared frozen matrices), LoKr (Kronecker product), Flora (stochastic gradient update)
    - Rank selection: dynamic rank (adapting `r` per layer during training), automated rank search
    - Quantization: QLoRA (4-bit NF4 base + BF16 adapter), LoftQ (joint quantization and LoRA initialization)
    - Multi-task: mixture-of-LoRA-experts, task-arithmetic LoRA merging
  - Surveys the inference serving literature: Punica, S-LoRA, dLoRA, CaraServe, LoRAX.
  - **Theoretical grounding**: explains why fine-tuning updates are low-rank in practice (the "intrinsic dimensionality" of fine-tuning subspaces) — the mathematical basis for why `r=4–16` is sufficient for most tasks despite `d` being 1024–8192.
  - Good reference for understanding LoRA's position relative to other PEFT methods (prefix tuning, adapters, prompt tuning, IA³).

---

## Source code anchors: SGLang multi-LoRA internals

The full SGLang multi-LoRA implementation is documented in `sglang_multi_lora_implementation.md`. Key source files for direct study:

### `lora/lora_manager.py` — Lifecycle manager

- **File:** `REPOS/sglang/python/sglang/srt/lora/lora_manager.py`
- **Level:** L5
- **Key anchors:**
  - `LoRAManager.__init__()`: allocates the 3D A/B weight pool buffers `[max_slots, rank, hidden]`; initialises the LRU registry.
  - `load_adapter()`: downloads adapter safetensors → H2D copy → register in pool slot.
  - `evict()`: LRU eviction — frees pool slots when new adapter requests exceed pool capacity.

### `lora/lora_registry.py` — Per-batch adapter bookkeeping

- **File:** `REPOS/sglang/python/sglang/srt/lora/lora_registry.py`
- **Level:** L5
- **Key anchors:**
  - `LoRARegistry.prepare_batch_info()`: builds `LoRABatchInfo` for the current batch — maps each token to a pool slot index; constructs `weight_indices[i]` and `seg_lens` tensors for the SGMV kernel.
  - `LoRABatchInfo`: the per-batch descriptor passed to `lora/layers.py`; equivalent to our `lora_mask` but richer (slot indices, segment lengths, sorted token order).

### `lora/layers.py` — LoRA-wrapped linear layers

- **File:** `REPOS/sglang/python/sglang/srt/lora/layers.py`
- **Level:** L5
- **Key anchors:**
  - `LoRALinear.forward()`: dispatches to the cSGMV or Triton SGMV kernel using `batch_info.weight_indices`.
  - Wraps `nn.Linear` — the base projection runs first; the SGMV delta is added as a residual.

### `lora/triton_ops/` — Triton SGMV kernels

- **Directory:** `REPOS/sglang/python/sglang/srt/lora/triton_ops/`
- **Level:** L5
- **Key anchors:**
  - `sgmv_shrink.py`: the "A" pass — computes `x @ A[w[i]].T` per segment.
  - `sgmv_expand.py`: the "B" pass — computes `h @ B[w[i]].T * scaling` per segment and adds to base output.
  - Both kernels use `tl.load()` with per-segment pointer offsets from the weight pool.

---

## What this layer explicitly does not cover

| Topic | Why excluded | Where to find it |
|---|---|---|
| LoRA training / fine-tuning | Layer 20 focuses on inference only | HuggingFace PEFT docs, QLoRA paper |
| DoRA, LoHa, VeRA, LoKr | Inference compatible but training-focused variants | Comprehensive review survey (L5) |
| Multi-modal LoRA (vision) | Extends LoRA to vision encoders; not in this codebase | PEFT multimodal docs |
| Quantized base models + LoRA (QLoRA) | Base model quantization is Layer 18 territory | QLoRA paper (arXiv 2305.14314) |
| LoRA + PD disaggregation | InfiniLoRA (L4 above) introduces this; not in our implementation | InfiniLoRA (arXiv 2604.07173) |
| CUDA graphs with multi-LoRA | Complex — `LoRABatchInfo` tensors must be static for graph replay | SGLang source, `lora_manager.py` |
| Weight merging at deploy time | Eliminates inference overhead but loses per-request switching | HuggingFace PEFT `merge_adapter()` |
