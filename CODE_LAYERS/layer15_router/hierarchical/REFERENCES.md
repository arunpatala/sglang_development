# References — LLM Router / Gateway

Organized by **reading level** (L1–L5 from `WRITING_GUIDE/PERSONAS.md`) and **category**. Use this when writing or extending lesson content, locating production precedents, or designing exercises.

---

## Foundational theory papers

> **Downloaded:** Both foundational papers are available in `references/SURVEY/`. See `references/SURVEY/README.md`.

### Mitzenmacher — the original power-of-two-choices proof (IEEE TPDS 2001)

- **Title:** The Power of Two Choices in Randomized Load Balancing
- **Author:** Michael Mitzenmacher (Harvard)
- **Venue:** IEEE Transactions on Parallel and Distributed Systems, Vol. 12, No. 10, Oct. 2001
- **DOI:** 10.1109/71.963420
- **PDF:** https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf
- **Level:** L4 (theory spine) / L3 (implication only)
- **What it contributes:**
  - Proves that sampling 2 servers at random and routing to the least-loaded one reduces maximum queue depth from O(log n / log log n) to O(log log n) — exponential improvement over random assignment.
  - This is the direct theoretical ancestor of SGLang's `PowerOfTwoPolicy` and Layer 15's `LeastLoadPolicy`.
  - The "d=2 vs d=3" result explains why `router.py` samples exactly two workers and not more.

### Mitzenmacher, Richa, Sitaraman — survey of two-choice techniques (2001)

- **Title:** The Power of Two Random Choices: A Survey of Techniques and Results
- **Authors:** Michael Mitzenmacher, Andrea W. Richa, Ramesh Sitaraman
- **PDF:** https://eecs.harvard.edu/~michaelm/postscripts/handbook2001.pdf
- **Level:** L4
- **What it contributes:**
  - Comprehensive treatment of the d-choice paradigm beyond the supermarket model.
  - Covers dynamic task assignment to servers, memory emulation, and routing.
  - Good background reading before implementing `LeastLoadPolicy._pick_worker`.

---

## Academic papers on LLM routing and scheduling

### Preble — first distributed LLM serving platform with global KV scheduling (ICLR 2025)

- **Title:** Preble: Efficient Distributed Prompt Scheduling for LLM Serving
- **Authors:** Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, Yiying Zhang
- **Venue:** ICLR 2025
- **arXiv:** https://arxiv.org/abs/2407.00023
- **PDF:** https://openreview.net/pdf?id=meKEKDhdnx
- **Local file:** `references/SURVEY/03_preble_iclr2025_full.md` (full paper, PDF → markdown)
- **Level:** L4
- **What it contributes:**
  - Proposes the E2 (Exploitation + Exploration) algorithm: route requests to replicas that already hold a matching KV prefix, but occasionally route to underloaded replicas to prevent hotspots.
  - Maintains a **global radix tree** across GPUs — conceptually identical to Layer 15's `RadixTrie` but at cluster scale.
  - 1.5×–14.5× improvement in average latency, 2×–10× in p99 latency over SOTA baselines.
  - Maps directly to `PrefixCacheAwarePolicy`: the `cache_threshold` and `balance_abs_threshold` dials in `config.yml` implement the same exploitation/exploration tradeoff.
  - Demonstrates that single-GPU prefix caching (vLLM/SGLang RadixAttention) does not extend automatically to distributed deployments — the router must maintain its own prefix state.

### Srivatsa et al. — Intelligent Router for LLM Workloads (arXiv 2024/2025)

- **Title:** Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing
- **Authors:** (IBM Research)
- **arXiv:** https://arxiv.org/abs/2408.13510
- **Local file:** `references/SURVEY/04_intelligent_router_ibm_full.md` (full paper, PDF → markdown)
- **Level:** L4
- **What it contributes:**
  - Empirical study establishing that round-robin routing is a strong baseline but breaks down for decode-heavy workloads.
  - RL-based and workload-guided routing strategies reduce end-to-end latency by up to 15% over round-robin.
  - Key insight: mixing requests with very different decode lengths on the same instance hurts throughput — the "dedicated small-large split" is an alternative to prefix-cache-aware routing.
  - Includes a Join-Shortest-Queue baseline that directly corresponds to Layer 15's `LeastLoadPolicy`.
  - Good empirical backing for why `round_robin` should be the starting point before tuning.

### Xia et al. — SkyWalker: locality-aware cross-region load balancer (EUROSYS 2026)

- **Title:** SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference
- **Authors:** Heming Xia et al. (LMSYS)
- **arXiv:** https://arxiv.org/abs/2505.24095
- **Venue:** EUROSYS 2026
- **Local file:** `references/SURVEY/05_skywalker_eurosys2026_full.md` (full paper, PDF → markdown)
- **Level:** L4–L5
- **What it contributes:**
  - Shows that for a 2-replica round-robin setup, memory usage divergence can reach 2.64× — motivating `LeastLoadPolicy` and `PrefixCacheAwarePolicy`.
  - Benchmark baseline set: Random, Round Robin, Least Load, Consistent Hashing, SGLang Router — same four policies Layer 15 implements (minus consistent hashing).
  - Introduces selective prompt sharing: transfers KV cache blocks between replicas when prefix overlap is high, avoiding full recomputation — the next step beyond Layer 15's stateless prefix routing.
  - Confirms SGLang Router is the production baseline for single-region prefix-aware routing.

### A Survey of LLM Inference Systems (arXiv 2025)

- **Title:** A Survey of LLM Inference Systems
- **arXiv:** https://arxiv.org/html/2506.21901v1
- **Local file:** `references/SURVEY/06_survey_llm_inference_2025_full.md` (full paper, PDF → markdown)
- **Level:** L3–L4
- **What it contributes:**
  - Section on multi-replica scheduling covers Round Robin, Power-of-Two, Preble, and Disaggregated (PD) serving.
  - Shows how load balancing fits into the broader inference stack: batching → KV memory → scheduling → routing.
  - Preble is positioned as the reference for cache-persistent routing; Mitzenmacher's PoT is the reference for load-only routing.
  - Good survey chapter to read before Layer 15 and before Layer 16 (PD disaggregation).

---

## Survey papers

### Zheng et al. — RadixAttention and SGLang (SOSP 2024 / NeurIPS 2024)

- **Title:** SGLang: Efficient Execution of Structured Language Model Programs
- **Authors:** Lianmin Zheng et al. (LMSYS / UC Berkeley)
- **arXiv:** https://arxiv.org/abs/2312.07104
- **Local file:** `references/SURVEY/07_sglang_radixattention_full.md` (full paper, PDF → markdown)
- **Level:** L3 (background on RadixAttention) / L5 (source study)
- **What it contributes:**
  - Introduces RadixAttention, the KV cache data structure that makes prefix caching viable inside a single engine.
  - Layer 15's `RadixTrie` is a simplified router-side approximation of this structure (text matching, not block-level token matching).
  - Understanding RadixAttention explains *why* routing to the same engine for the same prefix helps: the engine's RadixAttention will hit in memory and skip recomputation.
  - The `cache_threshold` parameter in `config.yml` is calibrated against the RadixAttention hit rate.

---

## Production router and gateway implementations (L4–L5 reading)

### SGLang Model Gateway — primary production reference

- **Docs:** https://docs.sglang.io/advanced_features/sgl_model_gateway.html
- **GitHub:** https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway
- **Router CLI:** https://sgl-project.github.io/advanced_features/router.html
- **Level:** L4–L5
- **Key files mapping to Layer 15:**
  - `sgl-model-gateway/src/policies/round_robin.rs` ↔ `router.py` `RoundRobinPolicy`
  - `sgl-model-gateway/src/policies/power_of_two.rs` ↔ `router.py` `LeastLoadPolicy`
  - `sgl-model-gateway/src/policies/cache_aware.rs` ↔ `router.py` `PrefixCacheAwarePolicy`
  - `sgl-model-gateway/src/core/worker.rs` ↔ `router.py` `Worker` dataclass
  - `sgl-model-gateway/src/routers/http/router.rs` ↔ `router.py` `Router.route()`
- **Relevant CLI params:** `--policy`, `--cache-threshold`, `--balance-abs-threshold`, `--worker-urls`
- **Production features not in Layer 15:** circuit breaker, Prometheus metrics, gRPC routing, PD disaggregation, Kubernetes service discovery, mTLS.

### vLLM Router — derived from SGLang gateway, adds PD disaggregation

- **Blog:** https://vllm.ai/blog/vllm-router-release
- **GitHub:** https://github.com/vllm-project/vllm/tree/main/vllm/router
- **Level:** L4–L5
- **What it contributes:**
  - Built in Rust, forked from SGLang model gateway, simplified to work with vLLM backends.
  - Adds consistent hashing (session-level stickiness) and Power-of-Two as first-class policies.
  - PD disaggregation support: the router manages separate prefill worker pools and decode worker pools and orchestrates the two-phase handoff — the Layer 16 concept previewed in `lesson/09_whats_next.md`.
  - TTFT benchmark: vLLM Router is 2,000 ms faster than llm-d and Kubernetes-native routing in their benchmark.
  - Shows that the Layer 15 `router.py` design (FastAPI proxy + policy + worker state) is a faithful minimal version of a real production router.

### Ray Serve PrefixCacheAffinityRouter — cluster-level prefix routing

- **Docs:** https://docs.ray.io/en/releases-2.54.0/serve/llm/user-guides/prefix-aware-routing.html
- **Level:** L3–L4
- **What it contributes:**
  - Multi-tier routing strategy identical in structure to Layer 15: (1) check load balance gap against `imbalanced_threshold`; (2) if balanced, use prefix tree to pick the best-match replica; (3) if imbalanced, route to the least-loaded replica.
  - The `imbalanced_threshold` ↔ Layer 15's `balance_abs_threshold`; the `match_rate_threshold` ↔ Layer 15's `cache_threshold`.
  - Demonstrates that the two-guard design (load check before cache affinity) is the correct production pattern, not Layer 15-specific simplification.
  - Uses a distributed prefix tree actor (Ray distributed object) — the scaled-out version of Layer 15's in-process `RadixTrie`.

### llm-d — Kubernetes-native KV-cache-aware routing

- **Blog (precise scheduling):** https://llm-d.ai/blog/kvcache-wins-you-can-see
- **Red Hat guide:** https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference
- **Level:** L3–L4
- **What it contributes:**
  - Production benchmark: precise prefix-cache scheduling is 57× faster TTFT than approximate scheduling in a B2B workload with 150 enterprise customers.
  - Demonstrates 87.4% cache hit rate achievable when routing is prefix-aware.
  - Shows what happens when prefix caching is *not* reflected in the router: standard load balancers scatter related requests, wiping out all KV cache reuse within individual vLLM pods.
  - Architecture: Kubernetes Gateway API Inference Extension (GAIE) with an Endpoint Picker Plugin (EPP) — the production version of `router.py`'s health + routing loop.

---

## Explainer blogs and tutorials (L1–L2 reading)

| Level | Link | Why useful |
|-------|------|------------|
| L1 | [Portkey: LLM proxy vs AI gateway](https://portkey.ai/blog/llm-proxy-vs-ai-gateway) | Clear taxonomy: proxy (lightweight, dev-time) vs gateway (production control plane). Good framing before explaining why Layer 15 is a "gateway." |
| L1 | [pkgpulse: Portkey vs LiteLLM vs OpenRouter 2026](https://www.pkgpulse.com/blog/portkey-vs-litellm-vs-openrouter-llm-gateway-2026) | Concise comparison of real LLM gateway products; shows what Layer 15 deliberately omits (semantic caching, guardrails, multi-provider). |
| L2 | [OptiVerse: Prefix-Aware Routing — Cache-Conscious Request Distribution](https://www.optiversetech.com/blog/prefix-aware-routing) | Best standalone explainer of prefix-aware routing. Covers the cache-blind → cache-aware transition, RadixAttention context, and compares SGLang, vLLM, and Ray Serve implementations. |
| L2 | [llm-d: KV-Cache Wins You Can See](https://llm-d.ai/blog/kvcache-wins-you-can-see) | Quantifies why naive load balancing breaks KV cache reuse in distributed deployments; 57× TTFT gap between approximate and precise scheduling. Good motivation for `PrefixCacheAwarePolicy`. |
| L2 | [Red Hat: Master KV cache aware routing with llm-d](https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference) | Practitioner walkthrough of llm-d configuration; good for L2 "why stateless routing is not enough." |
| L3 | [Ray Serve prefix-aware routing docs](https://docs.ray.io/en/releases-2.54.0/serve/llm/user-guides/prefix-aware-routing.html) | Best short technical description of the two-guard design (load check → prefix match). Includes runnable Python config. Good L3 bridge before reading `router.py` `PrefixCacheAwarePolicy`. |

---

## Production engine gateway docs (L4 reading)

### SGLang Router CLI reference

- **Docs:** https://sgl-project.github.io/advanced_features/router.html
- **Level:** L4
- **Key configuration mapped to Layer 15 `config.yml`:**

| SGLang CLI flag | Layer 15 `config.yml` key | Default |
|---|---|---|
| `--policy` | `router.policy` | `cache_aware` |
| `--cache-threshold` | `router.cache_threshold` | `0.5` |
| `--balance-abs-threshold` | `router.balance_abs_threshold` | `32` |
| `--worker-urls` | `router.workers[*].url` | — |
| `--host` / `--port` | `router.host` / `router.port` | `127.0.0.1:30000` |
| `--max-concurrent-requests` | (not in Layer 15) | `64` |

### SGLang Model Gateway full docs

- **Docs:** https://docs.sglang.io/advanced_features/sgl_model_gateway.html
- **Level:** L4–L5
- **Covers beyond Layer 15:**
  - gRPC routing with native Rust tokenization
  - Prefill-decode (PD) disaggregation mode
  - Circuit breaker and retry logic
  - Prometheus metrics + OpenTelemetry tracing
  - Kubernetes service discovery
  - Multi-model inference gateway
  - Language bindings (Python / Go)

---

## Blogs and articles — full list by level

### L1 — Orientation (no code, concept + use case)

> **Downloaded:** All three L1 articles are available in `references/L1/`. See `references/L1/README.md` for reading order.

| Title | Link | Local file | Why useful |
|-------|------|-----------|------------|
| Portkey: LLM proxy vs AI gateway | https://portkey.ai/blog/llm-proxy-vs-ai-gateway | `references/L1/01_portkey_llm_proxy_vs_ai_gateway.md` | Defines the gateway concept; distinguishes simple proxy from full gateway (governance, routing, observability). |
| Portkey AI Gateway introduction | https://portkey-ai-gateway.mintlify.app/introduction | `references/L1/03_portkey_ai_gateway_introduction.md` | Sub-1ms overhead; 250+ providers; shows the scale production gateways operate at — good contrast to Layer 15's teaching-scale `router.py`. |
| pkgpulse: Portkey vs LiteLLM vs OpenRouter 2026 | https://www.pkgpulse.com/blog/portkey-vs-litellm-vs-openrouter-llm-gateway-2026 | `references/L1/02_pkgpulse_portkey_litellm_openrouter.md` | Positions three real gateways; explains when you need a gateway vs a proxy; shows what Layer 15 deliberately omits. |

### L2 — Definitions + motivation (mechanism, minimal code, examples)

> **Downloaded:** All four L2 articles are available in `references/L2/`. See `references/L2/README.md` for reading order.

| Title | Link | Local file | Why useful |
|-------|------|-----------|------------|
| OptiVerse: Prefix-Aware Routing | https://www.optiversetech.com/blog/prefix-aware-routing | `references/L2/01_optiverse_prefix_aware_routing.md` | Best standalone explainer: why round-robin wastes KV cache; how prefix trees work in routers; compares SGLang, vLLM, Ray Serve. |
| llm-d: KV-Cache Wins You Can See | https://llm-d.ai/blog/kvcache-wins-you-can-see | `references/L2/02_llmd_kvcache_wins_you_can_see.md` | Shows the 57× TTFT gap; explains precise vs approximate prefix scoring; motivates the `RadixTrie` in `router.py`. |
| vLLM Router release blog | https://vllm.ai/blog/vllm-router-release | `references/L2/03_vllm_router_release.md` | Confirms Layer 15's design is a faithful minimal port of the production router (vLLM Router forked from SGLang gateway). |
| Red Hat: KV cache aware routing with llm-d | https://developers.redhat.com/articles/2025/10/07/master-kv-cache-aware-routing-llm-d-efficient-ai-inference | `references/L2/04_redhat_kv_cache_aware_routing_llmd.md` | Configuration walkthrough + 87.4% cache hit rate benchmark; good "what this gives you" motivation. |

### L3 — Mechanism level (pseudocode, invariants, configuration)

> **Downloaded:** All three L3 articles are available in `references/L3/`. See `references/L3/README.md` for reading order.

| # | Title | Link | Local file | Why useful |
|---|-------|------|-----------|------------|
| 01 | Ray Serve prefix-aware routing docs | https://docs.ray.io/en/releases-2.54.0/serve/llm/user-guides/prefix-aware-routing.html | `references/L3/01_ray_serve_prefix_cache_affinity_router.md` | Best technical description of the two-guard policy (load check → prefix match); runnable config; parameter reference. |
| 02 | SGLang Router CLI reference | https://docs.sglang.io/advanced_features/sgl_model_gateway.html | `references/L3/02_sglang_router_cli_reference.md` | Policy reference, parameter table, deployment modes (co-launch vs separate). Essential companion to `config.yml`. |
| 03 | Intelligent Router paper (arXiv 2408.13510) | https://arxiv.org/abs/2408.13510 | `references/L3/03_intelligent_router_llm_workloads.md` | Empirical study comparing Round Robin, Decode Balancer, and RL-based routing; round-robin baseline validation. |

### L4 — Production + systems (real stacks, benchmarks, tradeoffs)

> **Downloaded:** All four L4 articles are available in `references/L4/`. See `references/L4/README.md` for reading order.

| # | Title | Link | Local file | Why useful |
|---|-------|------|-----------|------------|
| 01 | SGLang Model Gateway full docs | https://docs.sglang.io/advanced_features/sgl_model_gateway.html | `references/L4/01_sglang_model_gateway_docs.md` | Primary production reference; all features beyond Layer 15 scope; source code mapping to `router.py`. |
| 02 | Preble (ICLR 2025) | https://arxiv.org/abs/2407.00023 | `references/L4/02_preble_distributed_prompt_scheduling.md` | 1.5×–14.5× latency reduction; E2 algorithm = Layer 15's two-guard design. |
| 03 | SkyWalker (EUROSYS 2026) | https://arxiv.org/abs/2505.24095 | `references/L4/03_skywalker_cross_region_load_balancer.md` | Cross-region routing; benchmark baseline set matches Layer 15 policies exactly; SGLang Router = recognized production baseline. |
| 04 | A Survey of LLM Inference Systems (2025) | https://arxiv.org/html/2506.21901v1 | `references/L4/04_survey_llm_inference_systems.md` | §5.2.2 positions Round Robin, PoT, Preble, and PD disaggregation in the full inference stack. |

### L5 — Build track (source code, integration, contribution)

> **Note:** L5 references are GitHub repositories and source files. Not downloaded; use git clone or browse online.

| Title | Link | Why useful |
|-------|------|------------|
| SGLang sgl-model-gateway source | https://github.com/sgl-project/sglang/tree/main/sgl-model-gateway | Rust implementation of all three Layer 15 policies + circuit breaker + Prometheus; canonical source for `router.py` design. |
| sglang-router Python package | https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/router | PyO3 bindings wrapping the Rust gateway; shows how to embed the router in a Python launch script. |
| vllm-router source | https://github.com/vllm-project/vllm/tree/main/vllm/router | Fork of sgl-model-gateway; adds consistent hashing + PD disaggregation orchestration. |
| Portkey AI Gateway source | https://github.com/Portkey-AI/gateway | TypeScript / Hono framework; shows a production multi-provider gateway with fallback chains, semantic caching, guardrails. |
| LiteLLM proxy source | https://github.com/BerriAI/litellm | Python; 100+ provider support; budget tracking; good contrast to Layer 15's single-model-family, single-region scope. |
| Preble source (on vLLM + SGLang) | https://arxiv.org/abs/2407.00023 | Referenced in the paper; distributed scheduler on top of vLLM/SGLang; global radix tree implementation at cluster scale. |

---

## How this maps to the hierarchical model

| Cluster | Key references |
|---------|---------------|
| **15a why** — motivation | Portkey proxy vs gateway (L1); OptiVerse prefix-aware routing (L2); llm-d KV-Cache Wins (L2); Preble §1 motivation (L4) |
| **15b worker abstraction** — `Worker`, `SelectWorkerInfo` | SGLang `core/worker.rs` (L5); Intelligent Router paper baselines (L4); SkyWalker benchmark baselines (L4) |
| **15c round-robin policy** — `RoundRobinPolicy` | SGLang `policies/round_robin.rs` (L5); Intelligent Router §A.1.5 (L4); SkyWalker Round Robin baseline (L4) |
| **15d least-load policy** — `LeastLoadPolicy` | Mitzenmacher 2001 PoT (L4 theory); SGLang `policies/power_of_two.rs` (L5); Intelligent Router §A.2.1 Join Shortest Queue (L4) |
| **15e prefix-cache-aware policy** — `RadixTrie`, `PrefixCacheAwarePolicy` | Preble E2 algorithm (L4); SGLang `policies/cache_aware.rs` (L5); Ray Serve PrefixCacheAffinityRouter docs (L3); OptiVerse blog (L2); llm-d 57× benchmark (L4) |
| **15f HTTP proxy loop** — `Router.route()`, `check_health` | SGLang `routers/http/router.rs` (L5); vLLM Router blog (L4); llm-d architecture (L4) |
| **15g what comes next** — PD disaggregation, streaming | vLLM Router PD section (L4); SGLang Gateway PD mode (L4); SkyWalker selective KV transfer (L4); Preble multi-GPU global scheduler (L4) |

---

## See also

- `WRITING_GUIDE/PERSONAS.md` — which reference depth fits which reader level.
- `WRITING_GUIDE/HIERARCHICAL.md` — how these references attach to topic nodes as L1–L5 artifacts.
- `lesson/00_outline.md` — full section list with code anchors into `router.py`.
- `lesson/summary.md` — blog-post-style narrative covering all Layer 15 components.
- `CODE_LAYERS/layer14_speculative_decoding/hierarchical/REFERENCES.md` — reference format this file mirrors.
