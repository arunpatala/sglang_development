# References — Prefill-Decode (PD) Disaggregation

Organized by **reading level** (L1–L5) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

Layer 19 covers **Prefill-Decode disaggregation**: the system-design technique of routing the computationally distinct prefill phase and the memory-bandwidth-bound decode phase onto separate GPU pools, connected by a high-speed KV cache transfer layer. References therefore cover: **the root cause of prefill-decode interference, the goodput framing that motivates disaggregation, the two seminal papers (DistServe and Splitwise), the KV transfer engines that make it practical (Mooncake, NIXL), SGLang's production implementation, vLLM's connector abstraction, NVIDIA Dynamo as an enterprise orchestration layer, DeepSeek's production deployment case study, and the ongoing debate between aggregation and disaggregation architectures.**

---

## Quick navigation

| Reading level | What you'll find |
|---|---|
| **L1** | 9 articles, blog posts, and tools: 5-minute mental models (Jackson MZ), KV cache size math (Jarvis Labs), roofline calculator (interactive), production proof (Perplexity 435M queries/month), TTFT/ITL explainer (LearnCodeCamp), compute-vs-memory analysis (TDS, NADDOD, BentoML), lecture slides from DistServe's PI (Hao Zhang) |
| **L2** | SGLang and vLLM deployment docs, launch commands, connector overview, LMSYS blog |
| **L3** | DistServe, Splitwise, Mooncake, NIXL, NVIDIA Dynamo — core mechanisms and benchmarks |
| **L4** | SARATHI (chunked prefill), TaiChi (aggregation vs disaggregation analysis), P/D-Serve, PPD for multi-turn, TraCT (CXL shared memory) |
| **L5** | Source code anchors: `disagg_utils.py`, `kv_transfer/`, `mooncake_connector.py`, `nixl_connector.py`, SGLang router |

---

## Primary sources: SGLang PD disaggregation

### SGLang PD Disaggregation Official Documentation

- **URL:** https://docs.sglang.io/advanced_features/pd_disaggregation.html
- **Level:** L2
- **What it contributes:**
  - The canonical launch-level reference for enabling PD disaggregation in SGLang.
  - Two supported transfer backends at launch: **Mooncake** (RDMA-based, recommended for production) and **NIXL** (network-agnostic, UCX/LIBFABRIC selectable).
  - `--disaggregation-mode prefill` and `--disaggregation-mode decode` flags: how a single `launch_server` binary becomes a phase-specific worker.
  - `--disaggregation-ib-device`: which RDMA NIC to use for KV transfer (e.g., `mlx5_roce0`).
  - Router setup: `sglang_router.launch_router --pd-disaggregation --prefill ... --decode ...` — the single entry point clients connect to.
  - Heterogeneous TP with GPU staging buffer: when prefill TP=4 and decode uses DP attention with a different TP size, enables 2–5× throughput improvement over default per-token slice.
  - ASCEND NPU support via Mooncake backend.
  - Environment variables for Mooncake NVLink transport (`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK`) — recommended for NVL72 deployments.

### SGLang PD Disaggregation Design Document

- **URL:** https://docs.sglang.io/advanced_features/pd_disaggregation_design.html
- **GitHub:** https://github.com/sgl-project/sglang/blob/main/docs/advanced_features/pd_disaggregation_design.md
- **Level:** L3
- **What it contributes:**
  - Full architectural walkthrough of SGLang's disaggregation implementation.
  - The handshake protocol between a prefill server and decode server: how they pair, how the decode server pre-allocates KV cache pages and signals the prefill server to begin the forward pass.
  - KV transfer sequence: prefill completes forward pass → writes KV pages to registered memory → notifies decode server → decode server pulls via RDMA.
  - Interaction with SGLang's RadixCache and HiCache: prefill nodes run HiCache to maximise GPU prefix hits; decode nodes offload KV cache asynchronously after generation.
  - Page-granular transfer: KV pages are the unit of transfer, matching the paged memory model of the underlying `MHATokenToKVPool`.
  - `DecodeKVCacheOffloadManager`: how the decode server handles KV cache growth without running out of VRAM during long decode sequences.

### SGLang DeepSeek Deployment Blog — 96 H100 GPUs

- **URL:** https://lmsys.org/blog/2025-05-05-large-scale-ep
- **Published:** May 5, 2025
- **Authors:** SGLang Team (LMSYS Org)
- **Level:** L2–L3
- **What it contributes:**
  - The canonical production deployment recipe for large-scale PD disaggregation with expert parallelism: 12 nodes × 8 H100 GPUs (3 prefill nodes + 9 decode nodes).
  - Measured throughput: **52,300 input tokens/second** and **22,300 output tokens/second** per node for 2,000-token input sequences on DeepSeek-V3.
  - Explains why PD disaggregation is essential for DeepSeek-V3's MoE architecture: without it, DeepEP's `auto` mode cannot simultaneously support all dispatch patterns needed for prefill and decode phases.
  - DeepEP (deep expert parallelism communication library by DeepSeek) integration: how `EPLB` (Expert Parallelism Load Balancing) works with PD disaggregation.
  - Practical implementation detail: prefill server and decode server pair via handshake, establishing local sender/receiver; decode server pre-allocates KV cache → signals prefill → prefill runs forward pass → KV transferred.
  - Result tables comparing 4 configurations (TP16×6, PD disagg, PD disagg + simulated MTP, aggregated baseline) — the primary benchmark for evaluating SGLang's disaggregation implementation.

---

## L1 introductions and mental models

> **Downloaded:** All 9 L1 articles are saved as local `.md` files in [`hierarchical/references/L1/`](references/L1/). Start with `references/L1/README.md` for reading-order guidance.

### Prefill Is Compute-Bound. Decode Is Memory-Bound. — Towards Data Science

- **URL:** https://towardsdatascience.com/prefill-is-compute-bound-decode-is-memory-bound-why-your-gpu-shouldnt-do-both/
- **Local:** [`references/L1/01_tds_compute_vs_memory.md`](references/L1/01_tds_compute_vs_memory.md)
- **Author:** Gokul Chandra Purnachandra Reddy
- **Published:** April 15, 2026
- **Level:** L1
- **What it contributes:**
  - Opens with a production failure story: a 64 H100 K8s cluster where tensor cores hit 92% during prefill, then dropped to 30% during decode — the mismatch that motivates disaggregation.
  - **Arithmetic intensity explained**: prefill computes a full S×S attention matrix per head per layer (16.7M multiply-adds at S=4,096), saturating tensor cores at 200–400 ops/byte. Decode computes a single 1×(S+t) dot product per head, dropping arithmetic intensity to 60–80 ops/byte — always below the GPU memory bandwidth ceiling.
  - Quantifies the gap: InfoQ technical analysis (September 2025) measured prefill at **90–95% GPU utilisation** and decode at **20–40%**, with 3–4× better energy efficiency per operation during prefill.
  - Traces adoption timeline: DistServe paper (OSDI 2024) → production at Perplexity, Meta, LinkedIn, Mistral → NVIDIA Dynamo → vLLM/SGLang/TensorRT-LLM native support.
  - Provides a concrete **request lifecycle diagram**: 4K-token prompt → prefill worker (150–400ms compute-bound) → 1.34 GB KV cache RDMA transfer (27–107ms) → decode worker (3–9s memory-bound) → output stream.
  - **When NOT to disaggregate**: 5 specific conditions including short prompt/output ratios, single-GPU setups, batch-only workloads.

### Disaggregated Prefill-Decode: The Architecture Behind Meta's LLM Serving — Jarvis Labs Blog

- **URL:** https://jarvislabs.ai/blog/llm-optimization-disaggregated-prefill-decode
- **Local:** [`references/L1/02_jarvislabs_meta_pd.md`](references/L1/02_jarvislabs_meta_pd.md)
- **Author:** Vishnu Subramanian (Jarvis Labs), January 29, 2026
- **Level:** L1
- **What it contributes:**
  - Written as a practitioner's research notes — explains the problem from first principles without assuming prior knowledge of disaggregation.
  - Provides the most concrete **KV cache size calculation** available in any L1 article:
    ```
    LLaMA-3.1-70B: 80 layers × 8 KV heads × 128 dims × 2 (K+V) × 2 bytes = 327,680 bytes/token
    4K-token prompt → 4,096 × 327,680 = 1.34 GB of KV cache
    ```
  - **Network bandwidth table**: compares 1 GbE (10.7s transfer), 10 GbE (1.07s), 25 GbE (0.43s), 100 GbE (0.11s), InfiniBand HDR (54ms), NVLink within-node (2.2ms) — makes the case for why InfiniBand/NVLink is a hard requirement, not an option.
  - Flowchart diagrams (Mermaid format) for: the phase mismatch, the disaggregated architecture, independent scaling scenarios (short-prompts/long-responses vs long-prompts/short-responses), and the vLLM router architecture.
  - Lists production adopters: Meta, LinkedIn, Mistral, HuggingFace (vLLM-based), NVIDIA Dynamo.
  - **Per-GPU-type optimization table**: prefill workers (H100, large batches) vs decode workers (high-memory-BW GPUs, continuous batching, persistent KV cache).
  - **Good use cases vs avoid cases**: production high-traffic → yes; development/testing → standard vLLM instead.

### Practical Guide to LLM: Prefill & Decode Disaggregation — Better Programming (Medium)

- **URL:** https://betterprogramming.pub/practical-guide-to-llm-prefill-decode-disaggregation-bd7f9ee4eaf5
- **Local:** [`references/L1/03_jackson_mz_practical.md`](references/L1/03_jackson_mz_practical.md)
- **Author:** Jackson MZ (Research Engineer, ex-DeepMind, ex-Google), April 5, 2025
- **Level:** L1
- **What it contributes:**
  - The shortest useful introduction — 2-minute read with the core mental model and a link to an implementation in under 99 lines of code.
  - One-sentence framing for each phase: "Prefill: calculate full attention for user input prompt." / "Decode: autoregressive output token generation."
  - One-line justification for disaggregation: "To make sure inference is not compute- and bandwidth-bound at the same time, we want to separate them."
  - Flow: `Prefill cluster → KV cache → Queue → Decode cluster → output response`.
  - Points to vLLM's disaggregated prefill diagram as the canonical visual.
  - Best used as the **very first read** before any other reference — sets up the mental model in 2 minutes.

### Understanding Prefill-Decode Disaggregation in LLM Inference Optimization — NADDOD (Medium)

- **URL:** https://naddod.medium.com/understanding-the-prefill-decode-disaggregation-in-llm-inference-optimization-5c11223a5360
- **Local:** [`references/L1/04_naddod_pd_overview.md`](references/L1/04_naddod_pd_overview.md)
- **Published:** August 22, 2025
- **Level:** L1
- **What it contributes:**
  - 5-minute structured overview covering: what each phase is, why disaggregation makes sense, what the advantages are, and what the key challenge (KV transfer overhead) is.
  - Explains the hardware matching argument: prefill stages use "GPUs with high computing power and suitable for large-scale parallel computing"; decode stages use "GPUs with larger video memory and faster response times."
  - Explains why mixing parallelism strategies is inefficient: "prefill is suited to tensor parallelism to reduce latency, while decode is suited to pipeline parallelism to increase throughput."
  - Introduces RDMA acceleration as the solution to KV transfer latency in a single paragraph.
  - Useful for readers who find the TDS article too long — provides the same core insight in 5 minutes.

### Prefill-Decode Disaggregation — BentoML LLM Inference Handbook

- **URL:** https://bentoml.com/llm/inference-optimization/prefill-decode-disaggregation
- **Local:** [`references/L1/05_bentoml_pd_handbook.md`](references/L1/05_bentoml_pd_handbook.md)
- **Level:** L1–L2
- **What it contributes:**
  - Concise handbook-style overview: what PD disaggregation is, why it makes sense, when not to use it.
  - Key benefits enumerated: dedicated resource allocation, parallel execution, independent scaling, hardware specialisation (e.g., use different GPU SKUs for compute-bound vs memory-bound phases).
  - **When disaggregation is not a silver bullet**: if workloads are small or GPU setup is not tuned, performance can drop 20–30% vs collocated serving.
  - Data transfer cost framing: disaggregation requires moving KV caches rapidly and reliably between workers; if transfer latency exceeds the gain from interference elimination, overall performance degrades.
  - Lists current KV transfer methods: NIXL, CXL, NVMe-oF.
  - References DistServe and SARATHI as the two canonical precedents.

### Understanding LLM Inference Basics: Prefill and Decode, TTFT, and ITL — LearnCodeCamp

- **URL:** https://learncodecamp.net/llm-inference-basics-prefill-decode-ttft-itl/
- **Local:** [`references/L1/06_learncodecamp_ttft_itl.md`](references/L1/06_learncodecamp_ttft_itl.md)
- **Level:** L1 (prerequisite reading)
- **What it contributes:**
  - The cleanest explainer of TTFT and ITL (inter-token latency, also called TPOT) as user-facing metrics — before disaggregation is introduced.
  - Defines TTFT: "delay from when the request is submitted until the very first output token appears." Explains why long prompts increase TTFT.
  - Defines ITL: "average time between consecutive output tokens once generation has started." Typical values: 20–100ms/token (10–50 tokens/sec) on high-end hardware.
  - Explains the asymmetry: long prompt + short output → high TTFT but fast completion; short prompt + long output → low TTFT but decode-dominated total time.
  - Introduces chunked prefill as the non-disaggregation mitigation in the same article.
  - Best read **before** any disaggregation article — ensures readers understand what TTFT and ITL are and why they're the right metrics to optimise.

### LLM Inference Performance Estimator — Interactive Roofline Calculator

- **URL:** https://joursbleu.github.io/llm-perf-model/
- **Local:** [`references/L1/07_llm_perf_estimator.md`](references/L1/07_llm_perf_estimator.md)
- **Level:** L1 (interactive tool)
- **What it contributes:**
  - Web-based interactive calculator for prefill latency, decode throughput, memory usage, and TTFT — all computed via roofline analysis.
  - Input: model (LLaMA, Mistral, DeepSeek, etc.), device (H100, A100, etc.), quantization (FP16, INT8, FP8, INT4, 3-bit, 2-bit), prompt length, output length, batch size, TP degree, FlashAttention on/off.
  - Output: prefill latency, decode speed (tokens/sec), total time, model memory (weights + KV cache per request).
  - Shows **roofline plot** directly, marking each operation (prefill vs decode) on the arithmetic intensity axis — makes the compute-bound vs memory-bound distinction visible and interactive.
  - Also includes per-operation layer breakdown and multi-GPU scaling comparison.
  - Useful formula reference: `Decode AI ≈ 1 FLOP/byte` (always memory-bound), `Prefill AI ≈ SeqLen` (scales with prompt length).
  - Recommended: enter your exact model and GPU to see the roofline before reading DistServe — makes the motivation for disaggregation immediately concrete.

### Perplexity AI: Disaggregated Serving at 400M Queries/Month — NVIDIA Spotlight

- **URL:** https://developer.nvidia.com/blog/spotlight-perplexity-ai-serves-400-million-search-queries-a-month-using-nvidia-inference-stack/
- **Local:** [`references/L1/08_perplexity_nvidia_spotlight.md`](references/L1/08_perplexity_nvidia_spotlight.md)
- **Published:** December 5, 2024
- **Level:** L1 (production case study)
- **What it contributes:**
  - The earliest public confirmation that disaggregated serving has moved to production at a hyperscale consumer-facing LLM service: **Perplexity AI handles 435+ million search queries per month** using NVIDIA H100 SXM GPUs with disaggregated prefill and decode.
  - States the production outcome directly: "This technique significantly boosts overall system throughput while meeting SLAs, translating to lower cost per token."
  - Confirms the hardware heterogeneity argument: "this technique gives Perplexity the flexibility to use different NVIDIA GPU products for each inference phase given its specific hardware resource requirements."
  - Uses NVIDIA Triton Inference Server (now part of Dynamo) + TensorRT-LLM as the serving stack.
  - Best used as **motivation for adoption**: if you're asking "is this production-ready or still research?", this article answers directly.

### Hao Zhang (UCSD): Disaggregating Prefill and Decode — CMU LLM Systems Lecture Slides

- **URL:** https://llmsystem.github.io/llmsystem2025spring/assets/files/llmsys-24-disaggregating_prefill_decode_hao_zhang-c0e55139d20512a2348783423397cc7f.pdf
- **Local:** [`references/L1/09_hao_zhang_lecture_slides.md`](references/L1/09_hao_zhang_lecture_slides.md)
- **Author:** Hao Zhang (UCSD, co-author of DistServe)
- **Level:** L1–L2 (lecture slides, more visual than a paper)
- **What it contributes:**
  - Slides from the CMU LLM Systems course — authored by DistServe's PI. This is the most direct explanation of the motivation behind the paper from its creator.
  - Defines goodput visually: "Throughput = completed request / time. Goodput = completed request within SLO / time. Low goodput: 3 rps even though Throughput = 10 rps under SLO criteria."
  - Explains the interference problem with diagrams: one prefill request saturating compute for hundreds of ms while decode requests wait.
  - Traces the history: "2023 end: Published and open sourced at UCSD (Hao's lab), with a concurrent work from Microsoft (not open source). 2024: OSS integration is slower compared to CB/paged attention as no significant gain was observed."
  - Includes the "Continuous Batching vs. Disaggregation" clarification: CB optimises throughput; disaggregation optimises goodput under SLOs. They are complementary, not competing.
  - **Best used as a companion to the DistServe paper** (L3/01) — the slides explain the intuitions that the paper presents formally.

---

## Seminal systems papers

### DistServe: Disaggregating Prefill and Decoding for Goodput-Optimized LLM Serving

- **Paper:** Zhong et al., "DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving," USENIX OSDI 2024
- **arXiv:** https://arxiv.org/abs/2401.09670
- **Conference page:** https://www.usenix.org/conference/osdi24/presentation/zhong-yinmin
- **Blog post:** https://hao-ai-lab.github.io/blogs/distserve/
- **Published:** July 2024 (USENIX OSDI '24)
- **Authors:** Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang (Peking University, UC San Diego)
- **Level:** L3
- **What it contributes:**
  - **The foundational paper for PD disaggregation** as a system-level design principle. Introduces the concept of **goodput** — the number of requests completed per second while satisfying both TTFT and TPOT SLO constraints simultaneously — as a better metric than raw throughput.
  - **Prefill-decoding interference**: quantifies how co-locating prefill with decode increases average TPOT by up to 40% on production workloads, because prefill compute waves block decode memory accesses.
  - **Resource coupling problem**: in a collocated system, the same parallelism configuration (TP, PP) must be used for both phases; disaggregation allows different TP/PP plans optimised per phase.
  - DistServe assigns prefill and decoding to different GPUs; after the first token is computed, the KV cache is transferred from the prefill instance to the decode instance via a point-to-point transfer.
  - Each phase manages its own copy of the model weights and its own KV cache pool.
  - **Benchmark results**: on various LLMs (OPT-13B, OPT-66B, OPT-175B) across chatbot, summarisation, and coding workloads, DistServe serves **4.48× more requests** under latency constraints vs state-of-the-art collocated systems, or meets **10.2× tighter SLO** at the same request rate.
  - Introduces the term "disaggregation" in the LLM serving context — all subsequent papers, SGLang, vLLM, and NVIDIA Dynamo cite this as the foundational reference.
  - **OSDI 2024** — top-tier systems venue.

- **BibTeX:**
  ```bibtex
  @inproceedings{zhong2024distserve,
    title={DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving},
    author={Yinmin Zhong and others},
    booktitle={USENIX OSDI 2024},
    year={2024}
  }
  ```

### Splitwise: Efficient Generative LLM Inference Using Phase Splitting

- **Paper:** Patel et al., "Splitwise: Efficient Generative LLM Inference Using Phase Splitting," ISCA 2024
- **arXiv:** https://arxiv.org/abs/2311.18677
- **PDF:** https://homes.cs.washington.edu/~patelp1/papers/splitwise-isca24.pdf
- **GitHub (SplitwiseSim):** https://github.com/Mutinifni/splitwise-sim
- **Published:** June 2024 (ISCA '24)
- **Authors:** Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, Ricardo Bianchini (University of Washington + Microsoft)
- **Level:** L3
- **What it contributes:**
  - **Independent concurrent discovery** of phase splitting (same idea as DistServe, published the same year). Splitwise uses production traces from two LLM inference services at **Microsoft Azure**, giving it a real-world workload distribution grounding that DistServe's synthetic benchmarks cannot provide.
  - Characterises the prompt-computation (prefill) vs token-generation (decode) phases across production traffic: prompt lengths, output lengths, concurrency, and GPU utilisation patterns.
  - Proposes splitting these phases onto separate machines, enabling **phase-specific resource management**: different hardware SKUs for prefill (compute-optimised) vs decode (memory-bandwidth-optimised).
  - **SplitwiseSim**: a discrete-event simulator for evaluating cluster-level PD disaggregation policies — still used in research to evaluate routing strategies.
  - **Benchmark result**: Splitwise clusters under performance SLOs achieve **1.76× better throughput** vs collocated baseline.
  - **Hardware heterogeneity insight**: prefill is better served on H100 (high FLOPS) while decode can tolerate lower-compute GPUs with high HBM bandwidth (e.g., A100 vs H100 tradeoff), suggesting mixed-hardware clusters for cost efficiency.
  - A prototype implementation of Splitwise's KV-cache transfer mechanism in vLLM (GitHub PR #2809) was the first public implementation of inter-instance KV transfer — the direct predecessor to vLLM's `kv_transfer/` module.
  - **ISCA 2024** — top-tier computer architecture venue.

- **BibTeX:**
  ```bibtex
  @inproceedings{patel2024splitwise,
    title={Splitwise: Efficient Generative LLM Inference Using Phase Splitting},
    author={Pratyush Patel and others},
    booktitle={ISCA 2024},
    year={2024}
  }
  ```

---

## KV transfer engines: the infrastructure layer

### Mooncake: A KVCache-Centric Disaggregated Architecture for LLM Serving

- **Paper:** "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving," USENIX FAST 2025
- **arXiv:** https://arxiv.org/abs/2407.00079
- **GitHub:** https://github.com/kvcache-ai/Mooncake
- **Docs:** https://kvcache-ai.github.io/Mooncake/
- **Published:** February 2025 (USENIX FAST '25); arXiv preprint July 2024
- **Authors:** Moonshot AI (Kimi team) + MadSys @ Tsinghua University
- **Level:** L3
- **What it contributes:**
  - **Production origin**: Mooncake is the serving platform for **Kimi** (Moonshot AI's flagship LLM service), handling exponential traffic growth in production. Not a research prototype — this is a paper about what they built and deployed.
  - **Architecture**: separates prefill and decode clusters; groups underutilised CPU, DRAM, SSD, and RDMA resources of the GPU cluster to implement a **disaggregated KV cache pool** — effectively combining PD disaggregation (Layer 19) with tiered KV cache (Layer 17) in one system.
  - **Transfer Engine**: the core component — provides a unified interface for batched data transfer across multiple storage devices and network links. Supports TCP, RDMA (InfiniBand/RoCEv2/eRDMA/NVIDIA GPUDirect), NVMe-oF, NVLink, intra-node NVLink, CXL/shared-memory, and EFA.
  - **GPUDirect RDMA**: enables direct GPU-to-GPU KV transfer without CPU involvement; topology-aware path selection chooses NICs to avoid PCIe/UPI bottlenecks.
  - **Multi-NIC pooling and retry**: uses multiple RDMA NICs simultaneously to saturate inter-node bandwidth; automatic failover when a NIC temporarily fails.
  - **KVCache-centric scheduler**: balances maximising effective throughput while minimising total costs; uses cache-aware load balancing to route requests to prefill workers whose cached prefix overlaps with the incoming request.
  - **Benchmark**: on 40 GB of KV cache (equivalent to LLaMA3-70B with 128k tokens), Mooncake Transfer Engine achieves high bandwidth across RDMA, NVLink, and TCP paths.
  - **Integrations as of 2026**: vLLM (v1 + v0), SGLang, TensorRT-LLM, LMCache, LMDeploy, xLLM, Ascend, and others. NIXL officially supports Mooncake Transfer Engine as a backend plugin (May 2025).
  - **Production result** (Kimi K2, July 2025): 128 H200 GPUs with PD disaggregation and large-scale expert parallelism → **224k tokens/sec prefill** and **288k tokens/sec decode** throughput.
  - **SGLang integration**: SGLang officially supports Mooncake Transfer Engine (April 2025); Mooncake is the default recommended transfer backend for SGLang PD disaggregation.
  - **FAST 2025** — top-tier storage systems venue.

- **BibTeX:**
  ```bibtex
  @inproceedings{moonshot2025mooncake,
    title={Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving},
    author={Moonshot AI and MadSys},
    booktitle={USENIX FAST 2025},
    year={2025}
  }
  ```

### NIXL: NVIDIA Inference Xfer Library

- **GitHub:** https://github.com/ai-dynamo/nixl
- **NVIDIA Technical Blog:** https://developer.nvidia.com/blog/enhancing-distributed-inference-performance-with-the-nvidia-inference-transfer-library/
- **Docs:** https://github.com/ai-dynamo/nixl/blob/main/docs/nixl.md
- **Released:** March 2025 (v1.0.0: March 2026)
- **Level:** L2–L3
- **What it contributes:**
  - **NIXL is the KV transfer library used by NVIDIA Dynamo** and supported natively by SGLang, vLLM, TensorRT-LLM, and others; it is the vendor-agnostic alternative to Mooncake.
  - **Core abstraction**: Transfer Agent — unifies three concepts: (1) Memory Section (HBM, DRAM, NVMe, distributed storage through a single buffer-list API), (2) Transfer Backend Interface (abstracts UCX/GDS/TCP/NVLink/etc.), (3) Metadata Handler (exchanges registration metadata between distributed agents).
  - **Supported backends**: UCX (RDMA InfiniBand/RoCEv2, GPUDirect RDMA), NVIDIA Magnum IO GPUDirect Storage, TCP, NVMe-oF, NVLink — all through a single plugin architecture.
  - **Three use cases** addressed: disaggregation (KV blocks between prefill/decode workers), long-context KV cache loading (from NVMe/cloud storage), and wide expert parallelism (MoE all-to-all).
  - **Async transfer design**: `batch_transfer_async_write()` / `transfer_check_status()` — non-blocking submission with polling, matching the compute-transfer overlap pattern used in HiCache (Layer 17).
  - **Metadata exchange via ETCD**: allows agents on different nodes to discover each other and exchange buffer registration metadata without involving the data path.
  - **NIXLBench** and **KVBench**: built-in profiling tools — KVBench automatically computes exact KV cache I/O sizes for supported models (LLaMA, Mistral, etc.) and generates ready-to-run NIXLBench commands.
  - **SGLang integration**: SGLang supports `--disaggregation-mode` with the NIXL backend via `SGLANG_DISAGGREGATION_NIXL_BACKEND` environment variable (choices: UCX, LIBFABRIC).
  - **vLLM integration**: `NixlConnector` in `vllm/distributed/kv_transfer/` — one of 6 supported disaggregation connectors.
  - NIXL was already a key component of NVIDIA Dynamo, TensorRT-LLM, vLLM, SGLang, Anyscale Ray, LMCache, and more at its 1.0 release.

---

## Production frameworks and deployments

### NVIDIA Dynamo — Distributed Inference Framework

- **Technical Blog:** https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models
- **Architecture Docs:** https://docs.nvidia.com/dynamo/design-docs/overall-architecture
- **GitHub:** https://github.com/ai-dynamo/dynamo
- **Announced:** March 18, 2025 (GTC 2025)
- **Level:** L2–L3
- **What it contributes:**
  - **Enterprise-scale disaggregated serving**: Dynamo treats prefill and decode workers as first-class entities, with a KV-aware router, planner, and discovery plane that manages them independently at datacenter scale.
  - **Four key innovations**:
    1. Disaggregated prefill and decode onto separate GPUs.
    2. LLM-aware request routing — routes requests to workers whose KV cache overlaps maximally with the incoming request prefix, reducing recomputation.
    3. Dynamic GPU worker allocation — elastically scales the prefill and decode pools in response to traffic shifts.
    4. KV Block Manager (KVBM) — multi-tier cache management for long-context workloads (extends effective cache beyond HBM).
  - **NIXL as the KV transfer layer** (default): end-to-end request flow in disaggregated mode: Client → Frontend → Router (chooses prefill worker) → Prefill computes KV → transfer metadata returned → Router (chooses decode worker) → Decode receives KV via NIXL → generation loop.
  - **Benchmark**: on DeepSeek-R1 on NVIDIA GB200 NVL72 with disaggregated serving, Dynamo increases requests served by **up to 30×** vs collocated baseline.
  - **Framework agnostic**: supports vLLM, SGLang, and TensorRT-LLM backends — the orchestration layer is separate from the inference engine.
  - **Rust runtime** for performance-sensitive components; Python for backend integration and extensibility.
  - Acknowledges DistServe, Splitwise, Mooncake as prior work in its architecture documentation.

### vLLM Disaggregated Prefilling

- **URL:** https://docs.vllm.ai/en/latest/features/disagg_prefill.html
- **Level:** L2–L3
- **What it contributes:**
  - Official vLLM documentation for disaggregated prefilling — marked experimental but deployed in production at scale.
  - **6 supported connector types** (as of 2025–2026):
    1. `ExampleConnector` — reference implementation for custom connectors.
    2. `LMCacheConnectorV1` — LMCache-based with NIXL underlying KV transmission.
    3. `NixlConnector` — direct NIXL integration (UCX/LIBFABRIC backends).
    4. `P2pNcclConnector` — NCCL-based direct P2P GPU transfer (simpler setup, lower bandwidth than RDMA).
    5. `MooncakeConnector` — Mooncake Transfer Engine.
    6. `MultiConnector` — chain multiple connectors.
  - **Key abstraction**: `Connector` (allows KV consumer to retrieve KV caches from producer) and `LookupBuffer` (`insert` + `drop_select` semantics).
  - **Architecture note from docs**: "Disaggregated prefill DOES NOT improve throughput" — it improves latency SLO compliance and TTFT/TPOT decoupling, not raw throughput.
  - Implementation is entirely in `vllm/distributed/kv_transfer/`.
  - `--kv-transfer-config` JSON flag: specifies `kv_connector`, `kv_role` (`kv_producer` or `kv_consumer`), `kv_parallel_size`, `kv_buffer_size`, `kv_port`.
  - Benchmarks available in `benchmarks/disagg_benchmarks/` — the canonical comparison point for vLLM disaggregation performance.

### DeepSeek-V3 Technical Report — Inference and Deployment Section

- **Paper:** DeepSeek-AI, "DeepSeek-V3 Technical Report"
- **arXiv:** https://arxiv.org/abs/2412.19437
- **Published:** December 2024 (arXiv); February 2025 (v2)
- **Level:** L3
- **What it contributes:**
  - Section 3.4 ("Inference and Deployment") is one of the few public technical descriptions of PD disaggregation at the scale of a frontier model (671B parameters, MoE with 37B activated per token).
  - **DeepSeek's production disaggregation strategy**: prefilling nodes run with chunked-prefill enabled; decoding nodes run with a separate KV cache pool and a different parallelism configuration.
  - **Prefilling section (3.4.1)**: describes how the prefill cluster manages KV cache generation for long prompts; uses expert parallelism to handle the 256-expert MoE; explains why the prefill parallelism configuration (EP=64, TP=4) differs from the decode configuration.
  - Motivates the need for disaggregation in the context of MoE: different experts are active during prefill vs decode, making collocated batching inefficient because the batch composition is fundamentally different.
  - The inference deployment details in this report became the blueprint that the SGLang team used to build the 96-H100 deployment described in the LMSYS blog (above).
  - **MLA** (Multi-head Latent Attention) and its KV cache implications: MLA compresses the KV cache by projecting into a lower-dimensional latent space, reducing KV transfer costs in a PD disaggregated setup — an architectural choice that directly benefits disaggregation.

---

## The aggregation-versus-disaggregation debate

### SARATHI: Chunked Prefill as an Alternative to Disaggregation

- **Paper:** Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills," arXiv 2308.16369
- **Extended version:** "Sarathi-Serve: Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve," USENIX OSDI 2024
- **arXiv:** https://arxiv.org/abs/2308.16369
- **Published:** August 2023 (arXiv); July 2024 (OSDI '24)
- **Authors:** Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Ramachandran Ramjee (Microsoft Research India)
- **Level:** L3–L4
- **What it contributes:**
  - **Chunked prefill**: instead of processing an entire prefill request in one large batch step, split it into equal-sized chunks and interleave each chunk with decode requests. This reduces head-of-line blocking without disaggregating to separate hardware.
  - **Decode-maximal batching**: construct a batch using one prefill chunk that saturates GPU compute, then fill remaining slots with as many decode requests as fit in memory — decode requests "piggyback" on the compute wave.
  - **Results**: up to 10× better decode throughput on LLaMA-13B (A6000); 4.25× on LLaMA-33B (A100); 1.91× end-to-end throughput on GPT-3 with pipeline parallelism.
  - **Relation to disaggregation**: SARATHI is the primary **aggregation-side** optimization cited in the disaggregation debate. It shows that carefully scheduled collocated serving can reduce — but not eliminate — prefill-decode interference. TaiChi (below) uses SARATHI-Serve as the aggregation baseline in its comparison.
  - SARATHI's chunked prefill is implemented in both vLLM and SGLang as `--chunked-prefill-size`; it is a prerequisite for efficient aggregated serving and is also used in disaggregated setups (the prefill node in SGLang's PD disagg uses `--chunked-prefill-size=32768`).
  - **Why it matters for Layer 19**: SARATHI defines the "control condition" against which disaggregation gains are measured.

### TaiChi: Unifying PD Aggregation and Disaggregation

- **Paper:** Wang et al., "Prefill-Decode Aggregation or Disaggregation? Unifying Both for Goodput-Optimized LLM Serving"
- **arXiv:** https://arxiv.org/abs/2508.01989
- **Submitted:** August 2025
- **Authors:** Chao Wang et al. (Sun Yat-sen University, Huawei Cloud, CUHK)
- **Level:** L4
- **What it contributes:**
  - **The most comprehensive comparative analysis** of PD aggregation vs PD disaggregation, framing the debate as a function of SLO (service level objective) constraints:
    - **PD aggregation** (Orca, SARATHI-Serve) is optimal under **tight TTFT + relaxed TPOT**: all GPUs can do prefill quickly, decode interference is tolerable.
    - **PD disaggregation** (DistServe, Splitwise) is optimal under **tight TPOT + relaxed TTFT**: dedicated decode GPUs ensure no prefill interference, at the cost of higher TTFT (fewer instances handling prefill).
    - Under **balanced TTFT and TPOT SLOs**, neither pure approach is optimal.
  - **TaiChi**: proposes a hybrid-mode inference system that unifies both: dynamically switches workers between P-heavy and D-heavy modes based on real-time SLO violation signals.
  - **Results**: improves goodput by up to **77%** over SOTA systems; reduces TTFT by up to **13.2×** (vs PD disaggregation) and TPOT by up to **1.69×** (vs PD aggregation).
  - Implemented on vLLM; planned to be open-sourced.
  - **Why it matters**: this paper is the clearest framing of when to use disaggregation vs aggregation in production — essential context for any deployment decision.

### PPD Disaggregation for Multi-Turn LLM Serving

- **Paper:** "Not All Prefills Are Equal: PPD Disaggregation for Multi-turn LLM Serving"
- **arXiv:** https://arxiv.org/abs/2603.13358
- **Submitted:** March 2026
- **Level:** L4
- **What it contributes:**
  - **Key insight**: in multi-turn conversations, "turn-2+ prefills" (append-prefills on top of an existing KV cache) are fundamentally different from "turn-1 prefills" (full prompt prefills). Append-prefills are shorter, have a high KV cache hit ratio, and may not need to be routed to the prefill pool at all.
  - **PPD** (Prefill Prefill-capable Decode) architecture: decode workers are made capable of handling short append-prefills locally, avoiding the KV transfer cost entirely when the request's KV state is already on the decode worker.
  - Achieves the best Pareto-frontier (TTFT vs TPOT tradeoff) compared to fixed PD routing on both synthetic and real multi-turn datasets.
  - Directly relevant to Layer 19: identifies a structural inefficiency in pure PD disaggregation for multi-turn workloads — a gap that SGLang's prefix-caching-aware router and HiCache try to address.

### P/D-Serve: Disaggregated LLM at Scale

- **Paper:** "P/D-Serve: Serving Disaggregated Large Language Model at Scale"
- **arXiv:** https://arxiv.org/abs/2408.08147
- **Published:** August 2024
- **Level:** L4
- **What it contributes:**
  - Addresses cluster-scale operational challenges of PD disaggregation: how to handle prefill node failures, how to manage KV transfer load balancing, how to scale the decode pool independently.
  - Introduces a disaggregation-aware request scheduler that accounts for KV transfer latency in routing decisions.
  - Relevant to production deployments where PD disaggregation involves dozens to hundreds of workers per pool.

---

## Alternative KV transfer approaches

### TraCT: CXL Shared Memory for Rack-Scale KV Transfer

- **Paper:** "TraCT: Disaggregated LLM Serving with CXL Shared Memory KV Cache at Rack-Scale"
- **arXiv:** https://arxiv.org/abs/2512.18194
- **Submitted:** December 2025
- **Level:** L4
- **What it contributes:**
  - **Alternative to RDMA**: proposes CXL Type-3 shared memory as the transport substrate for KV transfer between prefill and decode workers, eliminating RDMA card requirements at rack scale.
  - CXL shared memory provides load/store-accessible pooled DRAM visible to all nodes in a rack, enabling zero-copy KV cache sharing without explicit point-to-point transfer.
  - Compares against NIXL/UCX and LMCache on Dynamo (DeepSeek-R1-Distill-Llama-8B): TraCT reduces TTFT and achieves higher peak throughput vs NIXL for workloads where prefix reuse is high.
  - **Relevance**: CXL is a nascent alternative to RDMA for clusters with CXL-attached memory pools (e.g., rack-scale CXL switches); not yet production-deployed at hyperscaler scale but represents the direction of post-RDMA KV transfer infrastructure.
  - Implemented as a NIXL alternative within the NVIDIA Dynamo framework — uses vLLM's KV connector layer.

---

## Surveys: situating PD disaggregation in the broader landscape

### From Attention to Disaggregation: Tracing the Evolution of LLM Inference

- **Paper:** "From Attention to Disaggregation: Tracing the Evolution of LLM Inference"
- **arXiv:** https://arxiv.org/abs/2511.07422
- **Submitted:** November 2025
- **Level:** L3–L4
- **What it contributes:**
  - Comparative analysis of three disaggregated serving frameworks: DistServe (research-first, goodput-optimised), AIBrix (cloud-native production orchestration), and NVIDIA Dynamo (enterprise-scale hardware-software co-design).
  - Traces the architectural evolution from paged attention (vLLM) to prefill-decode disaggregation as the industry standard.
  - Frames disaggregation as having three distinct archetypes: research-first (goodput optimisation, DistServe), cloud-native (cost-effectiveness, AIBrix), and enterprise-scale (hardware acceleration, Dynamo).
  - Good reference for understanding *why* different organisations chose different architectures despite all using disaggregation.

### A Survey on Efficient Inference for Large Language Models (Roofline Analysis)

- **Paper:** Yuan et al., "LLM Inference Unveiled: Survey and Roofline Model Insights"
- **arXiv:** https://arxiv.org/abs/2402.16363
- **Published:** February 2024
- **Level:** L3–L4
- **What it contributes:**
  - **Roofline model analysis of LLM inference**: formally classifies prefill as compute-bound and decode as memory-bandwidth-bound using the arithmetic intensity ratio (FLOPs per byte moved from HBM).
  - **Prefill arithmetic intensity** ~ `2 × sequence_length × d_model / (2 × d_model)` = `sequence_length` — scales with input length, making long prefills very compute-bound.
  - **Decode arithmetic intensity** ~ `1` for a single-token decode step — always at or below the memory bandwidth ceiling.
  - This mathematical grounding is the theoretical basis for why the two phases benefit from different hardware and why interference is unavoidable when they share GPUs.
  - The survey is cited in PPD disaggregation, TaiChi, and NVIDIA Dynamo documentation as the roofline reference.

---

## Source code: SGLang PD disaggregation implementation

### `disagg_utils.py` — Core disaggregation primitives

- **File:** `REPOS/sglang/python/sglang/srt/managers/disagg_utils.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `KVCacheTransferMeta`: descriptor for a batch of KV pages to transfer — contains token indices, page slots, layer range, and dst_addr for RDMA write.
  - Handshake protocol: `DisaggPrefillBootstrapMeta` — the message the prefill server sends to the decode server to initiate a session.
  - `DisaggMode` enum: `PREFILL` / `DECODE` / `NULL` — the three modes a server can run in.

### `prefill_server.py` / `decode_server.py` — Phase-specific server logic

- **Files:** `REPOS/sglang/python/sglang/srt/managers/prefill_server.py`, `decode_server.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `PrefillServer.run_batch()`: executes the forward pass, gathers KV page addresses, calls the transfer engine to write KV to the decode server's registered memory.
  - `DecodeServer.pre_alloc_kv_cache()`: pre-allocates page slots for incoming KV pages before signalling the prefill server to begin — avoids page allocation races.
  - `DecodeKVCacheOffloadManager`: manages the decode server's KV cache when long decode sequences exceed available VRAM; integrates with HiCache when enabled.

### `mooncake_connector.py` — Mooncake transfer backend

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/disagg_backends/mooncake_connector.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `MooncakeTransferEngine` wrapper: initialises `mooncake.engine.TransferEngine` with local hostname, metadata server (etcd), protocol, and device names.
  - `register_kv_pages()`: registers GPU KV cache page buffers for RDMA access; called once at server startup.
  - `send_kv_cache()`: submits a batch write of KV pages from prefill GPU VRAM to decode GPU VRAM via `batch_transfer_async_write()`.
  - NVLink transport path (`SGLANG_MOONCAKE_CUSTOM_MEM_POOL=NVLINK`): bypasses PCIe for intra-NVL72-rack transfers.

### `nixl_connector.py` — NIXL transfer backend

- **File:** `REPOS/sglang/python/sglang/srt/mem_cache/disagg_backends/nixl_connector.py`
- **Level:** L5 (source study)
- **Key anchors:**
  - `NixlTransferAgent`: wraps the NIXL Python API; one agent per server process.
  - `register_memory()`: registers KV pool pages with the NIXL agent for RDMA access; exchanges metadata with decode agent via ETCD.
  - `transfer_kv_pages()`: issues `batch_transfer_async_write()` calls for each page in the batch; polls `transfer_check_status()` for completion.
  - Backend selection: `SGLANG_DISAGGREGATION_NIXL_BACKEND` environment variable (UCX default, LIBFABRIC alternative).

### `sglang_router` — PD-aware load balancer

- **Package:** `REPOS/sglang/sglang_router/`
- **Level:** L4–L5 (source study)
- **Key anchors:**
  - `launch_router --pd-disaggregation`: runs the router in PD mode — all incoming requests are first assigned to a prefill instance, then relayed to a decode instance.
  - **Routing policy**: currently `round_robin` between prefill instances; future direction is prefix-aware routing (route to the prefill instance with the highest KV cache overlap with the request).
  - `--prefill` / `--decode` URL lists: supports multiple prefill and decode instances per router — the primary mechanism for scaling each pool independently.

### `vllm/distributed/kv_transfer/` — vLLM KV connector framework

- **Directory:** `REPOS/vllm/vllm/distributed/kv_transfer/`
- **Level:** L4–L5 (source study)
- **Key anchors:**
  - `kv_connector.py`: `BaseKVConnector` abstract class — `send_kv_caches()`, `recv_kv_caches()`, `close()`.
  - `nixl/`: `NixlConnector` — uses `nixl.NixlAgent`; supports UCX and LIBFABRIC backends.
  - `mooncake/`: `MooncakeConnector` — wraps `mooncake.engine.TransferEngine`.
  - `p2p/`: `P2pNcclConnector` — NCCL-based direct P2P transfers; simpler setup, lower bandwidth than RDMA.
  - `kv_transfer_config.py`: `KVTransferConfig` dataclass — parsed from `--kv-transfer-config` JSON; contains `kv_connector`, `kv_role`, `kv_parallel_size`, `kv_buffer_size`, `kv_port`, `kv_connector_extra_config`.

---

## What this layer explicitly does not cover (deferred or adjacent)

| Topic | Why excluded | Where |
|---|---|---|
| Chunked prefill implementation details | Covered as context for disaggregation; deeper mechanics are in SARATHI/SGLang scheduler | External reading |
| HiCache + PD disaggregation interaction | Covered in Layer 17 (HiCache) with a note on PD integration | Layer 17 |
| KV cache quantization during transfer | FP8 KV reduces transfer bytes; covered as a separate topic | Layer 18 |
| Speculative decoding with PD disaggregation | An open research area; not yet standardised in production | Future layer |
| Multi-modal PD disaggregation (EPD: Encode-Prefill-Decode) | SGLang introduced EPD with Mooncake in Dec 2025; extends disaggregation to vision encoders | Future layer |
| Weight-sharing between prefill/decode nodes | Research direction for reducing GPU memory overhead; not yet production-deployed | External reading |
| Expert parallelism (EP) internals | DeepEP, DeepGEMM, EPLB — MoE-specific parallelism; overlaps with PD but is a separate topic | External reading |
