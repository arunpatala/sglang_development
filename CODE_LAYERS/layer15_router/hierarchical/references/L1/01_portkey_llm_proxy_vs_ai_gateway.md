# LLM Proxy vs AI Gateway: What's the Difference and Which One Do You Need?

**Source:** https://portkey.ai/blog/llm-proxy-vs-ai-gateway
**Author:** Portkey (Drishti Shah)
**Date:** 2025
**Level:** L1 — Orientation
**Why here:** Clearest available definition of the gateway concept. Establishes the "control plane for LLM traffic" framing used throughout Layer 15. Good first read before opening `router.py`.

---

## Summary

As more teams build with large language models, managing how these models are accessed, used, and scaled has become a core part of the stack. This has given rise to two infrastructure patterns: the **LLM proxy** and the **AI gateway**.

---

## What is an LLM proxy?

An LLM proxy is a lightweight middleware layer that sits between your application and the underlying LLM provider (like OpenAI, Anthropic, or Cohere). It does one job well: forward and shape LLM requests. It often comes with a few useful extras, like caching, logging, or token counting, but it's fundamentally built for simple, fast routing.

**Common features of an LLM proxy:**
- Request forwarding and routing: Easily switch between model endpoints.
- Token tracking: Estimate or log tokens used per request.
- Basic logging: Capture inputs, outputs, and metadata for debugging.
- Response caching: Reduce cost and latency for repeated prompts.

LLM proxies are best suited for early-stage projects or internal tools, where speed of iteration matters more than governance or enterprise-level control.

---

## What is an AI gateway?

An AI gateway is a production-grade infrastructure layer designed to manage, govern, and optimize all LLM traffic across an organization. While it includes basic proxying and routing, it goes far beyond that, offering centralized control, security, observability, and team-level governance.

It is like a **control plane for LLM usage**. Every request, regardless of model provider, team, or use case, flows through the gateway, which applies rules, logs activity, and ensures safe, cost-efficient usage.

**Additional capabilities of an AI gateway:**
- Access controls: Define who can use which models, and under what conditions.
- Guardrails: Enforce content policies, block jailbreaks, prevent PII leakage.
- Audit logs: Structured, queryable history of every request.
- LLM Observability: Track latency, error rates, token usage across providers.
- Rate limits and budget enforcement: Apply spending caps per app, team, or use case.
- Multi-provider abstraction: Easily route across OpenAI, Claude, Mistral, Azure, and more.
- Prompt management: Store, version, and test prompts centrally.

---

## LLM proxy vs AI gateway: feature comparison

| Feature | LLM Proxy | AI Gateway |
| --- | --- | --- |
| Request routing | ✔︎ | ✔︎ |
| Basic caching | ✔︎ | ✔︎ (advanced, configurable) |
| Token tracking | ✔︎ | ✔︎ |
| Logging | ✔︎ (limited) | ✔︎ (structured, queryable) |
| Access control | ✘ | ✔︎ (role-based, multi-tenant) |
| Guardrails & moderation | ✘ | ✔︎ (jailbreak detection, filtering) |
| Audit logs | ✘ | ✔︎ |
| Budget controls | ✘ | ✔︎ (per team/app/model) |
| Rate limiting | ✘ | ✔︎ (configurable and enforceable) |
| Multi-provider support | ✔︎ | ✔︎ |
| Prompt management | ✘ | ✔︎ (centralized + version control) |
| Observability | ✘ | ✔︎ (latency, error, usage analytics) |

---

## When should you use an LLM proxy?

Use an LLM proxy when:
- You're building a prototype or internal tool.
- You're only using one model provider.
- You're not worried yet about governance or compliance.

---

## When should you use an AI gateway?

Use an AI gateway when:
- You're deploying to production and need predictable behavior.
- Multiple teams or apps use LLMs.
- You need observability and auditability.
- You care about cost and rate limits.
- You're using multiple providers.
- You need to enforce guardrails.
- You're in a regulated environment.

---

## Relevance to Layer 15

Layer 15's `router.py` is at the **proxy end of this spectrum**: it does one job (route requests to backend engines based on a policy) without guardrails, audit logs, or multi-provider abstraction. The SGLang Model Gateway adds circuit breakers, Prometheus metrics, and Kubernetes discovery on the way toward a full gateway. This article gives the vocabulary to position Layer 15 correctly in that spectrum.
