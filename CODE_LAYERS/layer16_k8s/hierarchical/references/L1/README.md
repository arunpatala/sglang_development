# L1 References: LLM Router / Gateway

**Level:** L1 — Orientation (no code, concept + use case)

**Reader profile:** Knows what an API is. Has heard of OpenAI. Wants to understand what a "router" or "gateway" is before looking at any code. Satisfied when they can explain in one sentence why you need a gateway in front of your LLM engines.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_portkey_llm_proxy_vs_ai_gateway.md` | Portkey | Clearest definition of proxy vs gateway. The "control plane for LLM traffic" framing used in Layer 15. |
| 02 | `02_pkgpulse_portkey_litellm_openrouter.md` | PkgPulse | Shows real LLM gateway products and what Layer 15 deliberately omits (semantic caching, guardrails, multi-provider). |
| 03 | `03_portkey_ai_gateway_introduction.md` | Portkey AI Gateway | Production scale: 10B tokens/day, sub-1ms latency. Good contrast to Layer 15's teaching-scale `router.py`. |

---

## Recommended reading order

**Fast path (10 min):** 01 → 02
- 01 for the vocabulary: proxy vs gateway, what each provides.
- 02 for market context: what products exist, what Layer 15 covers vs what it omits.

**Thorough path (20 min):** 01 → 03 → 02
- 03 adds production scale and architecture context.

---

## How these map to Layer 15

| Layer 15 concept | Most relevant L1 reference |
|-----------------|---------------------------|
| "What is a router?" (lesson/01) | 01 (proxy vs gateway framing) |
| "Why not just use round-robin?" (lesson/01) | 01 (stateless proxy limitations), 02 (LiteLLM routing strategies) |
| "What does a production gateway add?" | 03 (Portkey production features) |

---

## Common L1 limits to name for readers

These articles **do not explain**:
- How prefix caching works inside a single engine (RadixAttention, vLLM APC).
- Why routing to the same engine for the same prefix reduces latency.
- What the `RadixTrie` in `router.py` does or why it's needed.
- The difference between routing across providers (LiteLLM/Portkey) vs routing across instances of the same model (Layer 15).

Those live in L2 (mechanism articles) and L3 (lesson files).
