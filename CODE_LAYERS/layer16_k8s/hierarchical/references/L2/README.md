# L2 References: LLM Router / Gateway

**Level:** L2 — Definitions + motivation (mechanism, minimal code, examples)

**Reader profile:** Knows AI terminology, comfortable with Python, wants the routing mechanism explained clearly before reading code. Satisfied when key terms are locked in, intuition is solid, and they can explain prefix-aware routing to a colleague.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_optiverse_prefix_aware_routing.md` | OptiVerse | Best standalone explainer: why round-robin wastes KV cache, how prefix trees work in routers, compares SGLang/vLLM/Ray Serve. |
| 02 | `02_llmd_kvcache_wins_you_can_see.md` | llm-d | 57× TTFT gap benchmark; precise vs approximate prefix scoring; motivates `RadixTrie` in `router.py`. |
| 03 | `03_vllm_router_release.md` | vLLM Blog | Confirms Layer 15's design is a faithful minimal port of the production router (vLLM Router forked from SGLang gateway). |
| 04 | `04_redhat_kv_cache_aware_routing_llmd.md` | Red Hat Developer | Configuration walkthrough + 87.4% cache hit rate benchmark; practitioner framing. |

---

## Recommended reading order

**Fast path (30 min):** 01 → 02
- 01 for the cleanest mechanism explanation + three production implementations compared.
- 02 for the strongest quantitative motivation (57× TTFT gap).

**Thorough path (60 min):** 01 → 02 → 04 → 03
- 04 adds concrete configuration examples.
- 03 connects Layer 15 to the production vLLM Router lineage.

---

## How these map to Layer 15

| Layer 15 component | Most relevant L2 reference |
|-------------------|---------------------------|
| `RoundRobinPolicy` | 01 (round-robin wasted computation problem), 02 (random/round-robin = weakest baseline) |
| `LeastLoadPolicy` | 02 (load-aware scheduling), 04 (load-aware routing tier) |
| `PrefixCacheAwarePolicy` | 01 (three-production-implementation comparison), 02 (precise prefix scheduling) |
| `RadixTrie` | 01 (SGLang RadixAttention radix tree), 02 (llm-d kvcache.Index) |
| `cache_threshold` | 01 (match_rate_threshold: 0.1 in Ray Serve), 02 (87.4% cache hit rate target) |
| `balance_abs_threshold` | 01 (imbalanced_threshold in Ray Serve), 02 (load-awareness gate in llm-d) |
| `_health_loop` | 03 (vLLM Router fault tolerance and health monitoring) |

---

## Common L2 limits to name for readers

These articles **do not explain**:
- The exact Python implementation of `RadixTrie.insert` and `RadixTrie.match_len`.
- How `config.yml` maps to `router.py` initialization.
- How `httpx.AsyncClient` is used to forward requests.
- The three-state circuit breaker (Closed → Open → HalfOpen) in SGLang production.

Those live in L3 (lesson files) and L4 (SGLang/vLLM source docs).
