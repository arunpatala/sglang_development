# Welcome to Portkey AI Gateway

**Source:** https://portkey-ai-gateway.mintlify.app/introduction
**Author:** Portkey AI
**Level:** L1 — Orientation
**Why here:** Shows what a production gateway looks like at the feature level. The "10 billion tokens/day" and "sub-1ms latency" numbers give concrete scale context for why Layer 15's `router.py` is a teaching-scale artifact.

---

## What is Portkey AI Gateway?

The AI Gateway is an open-source, lightweight solution designed for fast, reliable, and secure routing to 1600+ language, vision, audio, and image models. Built with Hono framework for TypeScript/JavaScript, it processes **over 10 billion tokens daily** in production environments.

- **Blazing Fast**: Sub-1ms latency with a tiny 122kb footprint
- **Battle Tested**: Processing 10B+ tokens daily in production
- **Enterprise Ready**: Enhanced security, scale, and custom deployments

---

## Why Choose AI Gateway?

### Universal LLM Integration

Integrate with any LLM in under 2 minutes. Unified OpenAI-compatible API for 250+ models across 45+ providers including:
- OpenAI, Azure OpenAI, Anthropic Claude
- Google Gemini, AWS Bedrock, Cohere
- Together AI, Groq, Perplexity, Mistral
- Ollama, Hugging Face, and many more

### Production-Grade Reliability

- **Automatic Retries**: Up to 5 times with exponential backoff
- **Fallbacks**: Automatically switch to backup providers on failures
- **Load Balancing**: Distribute requests across multiple API keys or providers
- **Request Timeouts**: Set granular timeouts to manage latencies

### Advanced Features

- **Guardrails**: Verify LLM inputs and outputs with 40+ pre-built guardrails
- **Multi-modal Support**: Text, vision, audio, image generation, and real-time APIs
- **Smart Caching**: Reduce costs and improve latency with response caching
- **Conditional Routing**: Route requests based on custom logic and conditions
- **MCP Gateway**: Centralized control plane for Model Context Protocol servers

---

## Key Capabilities: Routing & Load Balancing

```python
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

### Multi-Provider Fallbacks

```python
config = {
  "strategy": {"mode": "fallback"},
  "targets": [
    {"provider": "openai", "api_key": "sk-***"},
    {"provider": "anthropic", "api_key": "sk-ant-***"}
  ]
}
```

---

## Architecture

Built with:
- **Hono Framework**: Fast, lightweight web framework supporting multiple runtimes
- **Provider System**: Modular provider implementations with standardized interfaces
- **Middleware Pipeline**: Request validation, caching, logging, and routing
- **Plugin System**: Extensible guardrails for content filtering and validation

Runs on: Node.js, Cloudflare Workers, Docker, Kubernetes, and more.

---

## Relevance to Layer 15

Portkey's architecture (middleware pipeline → provider system → routing) mirrors the class hierarchy in `router.py`:
- Middleware Pipeline ↔ `Router.route()` + health loop
- Provider System ↔ `Worker` dataclass
- Routing ↔ `LoadBalancingPolicy` ABC + three concrete policies

The key difference: Portkey routes across providers (OpenAI vs Anthropic); Layer 15 routes across instances of the same provider/model. The load-balancing config (`weight`-based) is Portkey's equivalent of Layer 15's `round_robin` with unequal weights.
