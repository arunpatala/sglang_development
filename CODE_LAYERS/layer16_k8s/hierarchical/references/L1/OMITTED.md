# LLM Router / Gateway — Omitted Content

**What this file is:** Full text of every passage, section, and code example that was left out of `COMBINED.md`. Content is organized by source file, with a heading explaining why each piece was omitted. Nothing is summarized — if it appears here, it is verbatim from the original source.

**Sources:**
- L1/01 — `01_portkey_llm_proxy_vs_ai_gateway.md`
- L1/02 — `02_pkgpulse_portkey_litellm_openrouter.md`
- L1/03 — `03_portkey_ai_gateway_introduction.md`
- L2/02 — `02_llmd_kvcache_wins_you_can_see.md`
- L2/03 — `03_vllm_router_release.md`
- L2/04 — `04_redhat_kv_cache_aware_routing_llmd.md`

---

## From L1/01 — Portkey: LLM Proxy vs AI Gateway

### Omitted: When to use each pattern

**Why omitted from COMBINED.md:** The proxy-vs-gateway feature table (Section 2) already implies the decision. These lists add no new information beyond what the table already communicates, and are phrased as sales copy.

> **When should you use an LLM proxy?**
>
> Use an LLM proxy when:
> - You're building a prototype or internal tool.
> - You're only using one model provider.
> - You're not worried yet about governance or compliance.

> **When should you use an AI gateway?**
>
> Use an AI gateway when:
> - You're deploying to production and need predictable behavior.
> - Multiple teams or apps use LLMs.
> - You need observability and auditability.
> - You care about cost and rate limits.
> - You're using multiple providers.
> - You need to enforce guardrails.
> - You're in a regulated environment.

---

## From L1/02 — PkgPulse: Portkey vs LiteLLM vs OpenRouter

### Omitted: The Multi-LLM Problem

**Why omitted from COMBINED.md:** This section frames the *multi-provider* routing problem (different vendors, rate limits, outages). Layer 15 solves a different problem — distributing load across replicas of the *same* model. Including this risked blurring the distinction. The COMBINED.md §3 table captures the scope difference without expanding into why multi-provider routing is needed.

> In 2026, using a single LLM provider is risky:
> - **Rate limits**: OpenAI's 429 errors during peak hours kill production apps.
> - **Outages**: Any provider can go down; no fallback = 100% downtime.
> - **Cost optimization**: Route simple queries to cheaper models (GPT-4o Mini, Claude Haiku).
> - **SDK fragmentation**: OpenAI, Anthropic, and Google each have their own SDKs.
> - **Observability gaps**: No unified view of costs, latency, and errors across providers.
>
> LLM gateways solve all of this with a single unified API.

---

### Omitted: LiteLLM full description and configuration

**Why omitted from COMBINED.md:** The YAML config is specific to multi-provider routing (OpenAI vs. Anthropic keys). Layer 15 uses a different config format (`workers:` list with `url:` and `engine:` fields). Including this would give readers a misleading mental model of what configuring Layer 15's router looks like. The routing strategy names from LiteLLM (`least-busy`, `latency-based-routing`) were noted in the §3 table only.

> LiteLLM is a Python library AND proxy server that translates any LLM call to OpenAI format. Self-host the proxy and get unified routing, load balancing, cost tracking, and fallbacks — with full data ownership.
>
> ```yaml
> # litellm-config.yaml
> model_list:
>   - model_name: gpt-4o
>     litellm_params:
>       model: openai/gpt-4o
>       api_key: os.environ/OPENAI_API_KEY
>   - model_name: claude-sonnet
>     litellm_params:
>       model: anthropic/claude-sonnet-4-5
>       api_key: os.environ/ANTHROPIC_API_KEY
>   # Load-balanced group
>   - model_name: best-available
>     litellm_params:
>       model: openai/gpt-4o
>
> router_settings:
>   routing_strategy: "least-busy"
>   num_retries: 3
>   timeout: 30
> ```
>
> **Routing strategies**: `round-robin`, `least-busy`, `latency-based-routing`, `usage-based-routing`, `cost-based-routing`.
>
> LiteLLM is the most-starred LLM gateway on GitHub (15k+ stars). Proxy adds ~10-20ms latency.

---

### Omitted: Portkey enterprise features

**Why omitted from COMBINED.md:** Semantic caching, guardrails, and virtual keys are enterprise gateway features that Layer 15 does not implement and does not have a direct analog for. Including them would expand the scope beyond instance routing into the governance/security layer that belongs to a full AI gateway — a different layer of the stack entirely.

> Portkey is the most feature-rich option — designed for enterprise production with:
> - Semantic caching (cache similar, not just identical prompts — up to 40% cost reduction)
> - Guardrails for content filtering and PII detection
> - Virtual keys (one Portkey key per team, underlying provider keys centrally managed)
> - Advanced routing: weighted load balancing, conditional routing, automatic failover

---

## From L1/03 — Portkey AI Gateway Introduction

### Omitted: Universal LLM Integration section

**Why omitted from COMBINED.md:** The list of 45+ providers (OpenAI, Azure, Anthropic, Bedrock, Groq, Ollama, etc.) is specific to Portkey's multi-provider scope. Layer 15 connects to a fixed list of same-model backends. The provider diversity is irrelevant and could mislead readers into thinking Layer 15 needs to handle multi-provider abstraction.

> Integrate with any LLM in under 2 minutes. Unified OpenAI-compatible API for 250+ models across 45+ providers including:
> - OpenAI, Azure OpenAI, Anthropic Claude
> - Google Gemini, AWS Bedrock, Cohere
> - Together AI, Groq, Perplexity, Mistral
> - Ollama, Hugging Face, and many more

---

### Omitted: Production-Grade Reliability section

**Why omitted from COMBINED.md:** Automatic retries, fallbacks, and request timeouts are gateway features that Layer 15 implements only partially (`_health_loop` removes failed workers but does not retry individual requests with exponential backoff). Including these would raise the question of why Layer 15 doesn't implement them — a distraction at L1 depth.

> - **Automatic Retries**: Up to 5 times with exponential backoff
> - **Fallbacks**: Automatically switch to backup providers on failures
> - **Load Balancing**: Distribute requests across multiple API keys or providers
> - **Request Timeouts**: Set granular timeouts to manage latencies

---

### Omitted: Advanced Features section

**Why omitted from COMBINED.md:** Guardrails, multi-modal support, smart caching, conditional routing, and MCP Gateway are all features of the gateway tier that sits *above* the instance-routing tier Layer 15 implements. Including them at L1 would expand scope in a direction that doesn't connect to any component in `router.py`.

> - **Guardrails**: Verify LLM inputs and outputs with 40+ pre-built guardrails
> - **Multi-modal Support**: Text, vision, audio, image generation, and real-time APIs
> - **Smart Caching**: Reduce costs and improve latency with response caching
> - **Conditional Routing**: Route requests based on custom logic and conditions
> - **MCP Gateway**: Centralized control plane for Model Context Protocol servers

---

### Omitted: Routing & Load Balancing code examples

**Why omitted from COMBINED.md:** The Python config API (weighted targets by provider) is Portkey-specific and uses a multi-provider abstraction that Layer 15 does not have. The `"weight": 0.7` / `"weight": 0.3` pattern is the Portkey analog of Layer 15's `RoundRobinPolicy` with unequal weights — but the API is entirely different. Presenting this code without the mapping would confuse readers about Layer 15's config format.

```python
# Portkey load balancing config (multi-provider)
config = {
  "strategy": {
    "mode": "loadbalance",
  },
  "targets": [
    {"provider": "openai", "api_key": "sk-***", "weight": 0.7},
    {"provider": "anthropic", "api_key": "sk-ant-***", "weight": 0.3}
  ]
}
client = client.with_options(config=config)
```

```python
# Portkey multi-provider fallback config
config = {
  "strategy": {"mode": "fallback"},
  "targets": [
    {"provider": "openai", "api_key": "sk-***"},
    {"provider": "anthropic", "api_key": "sk-ant-***"}
  ]
}
```

---

### Omitted: Portkey architecture section

**Why omitted from COMBINED.md:** The Hono framework, TypeScript runtime, Cloudflare Workers, and plugin system are implementation details of Portkey's specific product. The `COMBINED.md` §9 already uses Portkey's *conceptual* architecture (middleware pipeline → provider system → routing) as a mapping to `router.py`'s class hierarchy. The full implementation specifics below add no additional insight for someone reading Layer 15.

> Built with:
> - **Hono Framework**: Fast, lightweight web framework supporting multiple runtimes
> - **Provider System**: Modular provider implementations with standardized interfaces
> - **Middleware Pipeline**: Request validation, caching, logging, and routing
> - **Plugin System**: Extensible guardrails for content filtering and validation
>
> Runs on: Node.js, Cloudflare Workers, Docker, Kubernetes, and more.

---

## From L2/02 — llm-d: KV Cache Wins You Can See

### Omitted: llm-d KVEvents engine internals

**Why omitted from COMBINED.md:** This section describes the *engine side* of cache tracking — how vLLM pods emit `KVEvents` and how llm-d consumes them into a global index. Layer 15's `RadixTrie` is the *router side* equivalent: the router maintains its own prefix index from request history rather than receiving live cache events from the engine. Including the engine-side implementation would require explaining why Layer 15 can't receive `KVEvents` (it's a teaching artifact, not a production system), which is a distraction at L1/L2 depth.

This content is the correct starting point for anyone building the engine→router feedback loop as a Layer 15 extension.

> llm-d creates a **global view of the cluster's KV-cache**, allowing it to treat the disaggregated memory as a single, manageable pool.
>
> ### How It Works: KVEvents
>
> Each vLLM pod continuously emits `KVEvents` — live feed of all physical cache changes:
> 1. `kvevents.Pool`: Consumes the event stream, maintains a KV-Block Index (map of block-hashes to pod + memory medium).
> 2. `kvcache.Index`: Higher-level index used by the scheduler. Maps logical token sequences (prefixes) to pods.
>
> ### The Precise Prefix-Cache Scorer
>
> For every incoming request, the scorer:
> 1. Retrieves the most extended cached token sequence from `PrefixStore`.
> 2. Outputs a "cache affinity score" for each pod — directly representing the computational work that can be saved.
> 3. Combines the cache affinity score with load awareness to make the final routing decision.

---

## From L2/03 — vLLM Router Release

### Omitted: Consistent Hashing policy description

**Why omitted from COMBINED.md:** Consistent hashing maps routing keys (session ID, user ID) to workers via a hash ring. It achieves similar KV cache locality to prefix-aware routing via a simpler mechanism (no prefix index required), but it misses cross-user cache reuse (different users with the same system prompt do not benefit). Layer 15 omits this policy in favour of explicit `PrefixCacheAwarePolicy`, which is more general. Including consistent hashing at L1/L2 depth would require explaining *why* Layer 15 chose the harder approach — which belongs at L3+.

> **Consistent Hashing**: Key policy for maximizing performance. Requests with the same routing key (e.g., session ID, user ID) are "sticky" — always routed to the same worker replica, maximizing KV cache reuse.

---

### Omitted: Prefill/Decode Disaggregation section (full)

**Why omitted from COMBINED.md:** PD disaggregation is the Layer 16 concept. It splits the compute-bound prefill phase and the memory-bound decode phase onto different worker pools, with the router orchestrating handoff between them. Layer 15's `router.py` has no decode worker group, no prefill worker group, and no state transfer mechanism. Including this at L1/L2 depth would introduce a concept that requires its own layer to understand correctly.

> ### 2. Native Support for Prefill/Decode Disaggregation
>
> The router orchestrates the **prefill/decode (P/D) disaggregation** architecture:
> 1. Routes new requests to the prefill worker group.
> 2. Upon completion, directs the request state to the appropriate decode worker for token generation.
> 3. Supports both NIXL and NCCL-based (with ZMQ discovery) disaggregation backends.
>
> This is the **Layer 16 concept** previewed in `lesson/09_whats_next.md`.

---

### Omitted: Enterprise-Grade Resiliency detail

**Why omitted from COMBINED.md:** The COMBINED.md §9 table already lists circuit breakers, Kubernetes service discovery, and Prometheus metrics as "not in Layer 15." The full descriptions below expand on each, but at L1/L2 depth the table is sufficient. This content is the starting point for extending `router.py` with production hardening.

> - **Kubernetes Service Discovery**: Automatic discovery and routing to vLLM worker pods using label selectors.
> - **Fault Tolerance**: Configurable retry logic (with exponential backoff and jitter) and circuit breakers.
> - **Observability**: Built-in Prometheus endpoint (`/metrics`) with metrics on request volume, latency, error rates, and worker health.
>
> Production features not in Layer 15:
> - Circuit breaker (Layer 15 has basic health-check removal only)
> - Kubernetes service discovery (Layer 15 uses static config)
> - Prometheus metrics (Layer 15 uses Python logging only)

---

## From L2/04 — Red Hat / llm-d: Master KV Cache Aware Routing

### Omitted: GAIE / Endpoint Picker Plugin configuration

**Why omitted from COMBINED.md:** The Gateway API Inference Extension (GAIE) and Endpoint Picker Plugin (EPP) are Kubernetes-native production infrastructure. The YAML config defines a plugin that runs inside a Kubernetes gateway controller — a deployment model entirely different from Layer 15's standalone Python HTTP server. Including this would require explaining the Kubernetes gateway API, which is several layers removed from `router.py`.

This is the correct next step for anyone deploying Layer 15-style routing into a real Kubernetes cluster.

> llm-d introduces the **Gateway API Inference Extension (GAIE)** with an Endpoint Picker Plugin (EPP):
>
> - **Session-aware routing**: Maintains request consistency for optimal cache reuse.
> - **Prefix-aware scoring**: Routes requests based on prompt similarity and cache warmth.
>
> ```yaml
> # plugins.yaml configuration
> plugins:
>   - name: "cache-aware-router"
>     type: "external_processor"
>     config:
>       discovery:
>         label_selector: "llm-d.ai/inferenceServing=true"
>       cache:
>         type: "in-memory-lru"
>         max_size: 10000
>       routing:
>         algorithm: "prefix-aware"
>         session_affinity: true
> ```

---

### Omitted: vLLM pod configuration for optimal prefix caching

**Why omitted from COMBINED.md:** These are vLLM engine-side flags — they configure the inference server, not the router. Layer 15's `router.py` does not set or read these flags; they are a prerequisite configuration for the backend engines that the router connects to. Including them in the router document would blur the boundary between router (Layer 15) and inference engine (Layer 13/14). Relevant when setting up a real deployment.

> ```yaml
> args:
>   - "--enable-prefix-caching"          # Enable KV-cache prefix reuse
>   - "--block-size=16"                   # Optimal block size for cache efficiency
>   - "--gpu-memory-utilization=0.7"     # Reserve memory for cache storage
>   - "--max-model-len=4096"             # Match expected prompt lengths
>   - "--kv-cache-dtype=auto"            # Automatic cache data type optimization
> ```

---

## When to read this file

| Goal | Read |
|------|------|
| Understand Layer 15's `router.py` | `COMBINED.md` |
| Extend Layer 15 with live engine cache events | §"llm-d KVEvents engine internals" above |
| Deploy Layer 15's routing into Kubernetes | §"GAIE / EPP configuration" and §"vLLM pod configuration" above |
| Add consistent hashing to `router.py` | §"Consistent Hashing policy" above |
| Add circuit breaker / Prometheus to `router.py` | §"Enterprise-Grade Resiliency detail" above |
| Add P/D disaggregation (Layer 16) | §"Prefill/Decode Disaggregation" above |
| Build a multi-provider gateway above Layer 15 | §"Multi-LLM Problem", §"LiteLLM config", §"Portkey features" above |
