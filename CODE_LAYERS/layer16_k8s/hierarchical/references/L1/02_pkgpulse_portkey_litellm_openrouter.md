# Portkey vs LiteLLM vs OpenRouter: LLM Gateway Comparison 2026

**Source:** https://www.pkgpulse.com/blog/portkey-vs-litellm-vs-openrouter-llm-gateway-2026
**Date:** 2026
**Level:** L1 — Orientation
**Why here:** Shows the breadth of real LLM gateway products and what Layer 15 deliberately omits. Good for framing scope: Layer 15 targets a single-model-family, two-engine setup — a slice of what these products do.

---

## TL;DR

Managing multiple LLM providers — OpenAI, Anthropic, Gemini, Mistral — is complex: different SDKs, different pricing, different reliability.

- **LiteLLM**: Open-source Python proxy. One OpenAI-compatible API for 100+ LLMs. Self-host and route anywhere.
- **Portkey**: Enterprise-grade AI gateway. Semantic caching, guardrails, advanced observability. Managed or self-hosted.
- **OpenRouter**: SaaS marketplace. One API key, 200+ models, pay per token. No infrastructure to manage.

**Decision guide:**
- Enterprise production teams → Portkey
- Self-hosted infrastructure control → LiteLLM
- Instant multi-model access with zero setup → OpenRouter

---

## The Multi-LLM Problem

In 2026, using a single LLM provider is risky:
- **Rate limits**: OpenAI's 429 errors during peak hours kill production apps.
- **Outages**: Any provider can go down; no fallback = 100% downtime.
- **Cost optimization**: Route simple queries to cheaper models (GPT-4o Mini, Claude Haiku).
- **SDK fragmentation**: OpenAI, Anthropic, and Google each have their own SDKs.
- **Observability gaps**: No unified view of costs, latency, and errors across providers.

LLM gateways solve all of this with a single unified API.

---

## LiteLLM: Open-Source Universal Proxy

LiteLLM is a Python library AND proxy server that translates any LLM call to OpenAI format. Self-host the proxy and get unified routing, load balancing, cost tracking, and fallbacks — with full data ownership.

```yaml
# litellm-config.yaml
model_list:
  - model_name: gpt-4o
    litellm_params:
      model: openai/gpt-4o
      api_key: os.environ/OPENAI_API_KEY
  - model_name: claude-sonnet
    litellm_params:
      model: anthropic/claude-sonnet-4-5
      api_key: os.environ/ANTHROPIC_API_KEY
  # Load-balanced group
  - model_name: best-available
    litellm_params:
      model: openai/gpt-4o

router_settings:
  routing_strategy: "least-busy"
  num_retries: 3
  timeout: 30
```

**Routing strategies**: `round-robin`, `least-busy`, `latency-based-routing`, `usage-based-routing`, `cost-based-routing`.

LiteLLM is the most-starred LLM gateway on GitHub (15k+ stars). Proxy adds ~10-20ms latency.

---

## Portkey: Enterprise AI Gateway

Portkey is the most feature-rich option — designed for enterprise production with:
- Semantic caching (cache similar, not just identical prompts — up to 40% cost reduction)
- Guardrails for content filtering and PII detection
- Virtual keys (one Portkey key per team, underlying provider keys centrally managed)
- Advanced routing: weighted load balancing, conditional routing, automatic failover

---

## Key stats

| Feature | LiteLLM | Portkey | OpenRouter |
|---|---|---|---|
| Provider support | 100+ | 25+ | 200+ |
| Open source | Yes (MIT) | Partial | No |
| Self-hosted | Yes | Yes / Cloud | Cloud only |
| Semantic caching | No | Yes | No |
| Proxy latency | ~10-20ms | ~5ms | ~50-100ms |
| Routing strategies | 6 | Weighted + conditional | Fixed |

---

## Relevance to Layer 15

Layer 15's `router.py` is conceptually closest to the LiteLLM proxy in architecture: a self-hosted Python HTTP server that routes to a fixed set of same-model backend engines using configurable policies. The key differences from LiteLLM:
- Layer 15 routes to engines serving the **same model** (load distribution), not different models.
- Layer 15's `PrefixCacheAwarePolicy` is more specialized than LiteLLM's generic routing strategies.
- Layer 15 has no multi-provider abstraction layer.
