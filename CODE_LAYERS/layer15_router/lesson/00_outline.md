# Layer 15 — Lesson Outline

## What This Lesson Covers

Layer 14 committed more tokens per target forward pass using speculative decoding, but the inference process still ran on a single engine instance: one scheduler, one model runner, one KV pool. If that engine is busy, new requests wait. Layer 15 introduces a gateway process — `router.py` — that sits in front of multiple engine instances and forwards each request to a chosen backend based on a routing policy. The engine (`server.py`) is unchanged; the new work is entirely in the router. Three policies are implemented: `RoundRobinPolicy` alternates engines in sequence, `LeastLoadPolicy` routes to the engine with fewest in-flight requests, and `PrefixCacheAwarePolicy` maintains a radix trie per engine and routes requests with matching prompt prefixes to the same engine so that engine's `RadixCache` (from Layer 12) continues to hit.

Layer 15 adds a single new file, `router.py`, containing six classes and one background task. Configuration is extended with a `router:` block in `config.yml` that lists backend URLs, the chosen policy, and policy tuning parameters. `server.py`, `scheduler.py`, `model_runner.py`, `kv_cache.py`, `radix_cache.py`, and all model files are unchanged.

The sections follow the router's call path: why a single engine is insufficient, the `Worker` abstraction and load tracking, the `LoadBalancingPolicy` interface, each of the three policy implementations, the `Router` class that wires policy to httpx forwarding, the background health checker, the full request trace from client to engine and back, and the path to disaggregated prefill/decode.

---

## Sections

### 01 — From One Engine to Many (`01_from_one_engine_to_many.md`)
- Layer 14's ceiling: one engine processes requests sequentially through its scheduler; throughput is bounded by a single GPU's forward pass throughput; adding requests beyond `max_running` creates a waiting queue
- The router's offer: N engines behind a gateway multiply the effective `max_running` and KV pool capacity; the gateway adds no model weights and negligible latency (~0.5 ms per forward call overhead)
- The routing problem: which engine to choose for each request determines whether the RadixCache from Layer 12 helps or hurts; random assignment halves effective cache hit rate for N=2; prefix-aware assignment preserves it
- What Layer 15 adds: `router.py` with `Worker`, `LoadBalancingPolicy`, `RoundRobinPolicy`, `LeastLoadPolicy`, `PrefixCacheAwarePolicy`, `RadixTrie`, and `Router`; `config.yml` extended with `router:` block; all engine files unchanged

### 02 — Worker and Load Tracking (`02_worker_and_load_tracking.md`)
- `Worker` dataclass: `url`, `name`, `load` (in-flight request count), `healthy` (last health check result); mirrors SGLang's `BasicWorker` struct with `load_counter: AtomicUsize` and `healthy: AtomicBool`
- Load increment/decrement pattern: `worker.load += 1` before forwarding, `worker.load -= 1` in `finally`; mirrors SGLang's `WorkerLoadGuard` RAII struct (increment on creation, decrement on drop)
- Why load tracking matters: `LeastLoadPolicy` reads `worker.load` directly at selection time; without accurate counts, a slow engine accumulates requests while a fast engine sits idle
- `healthy` field: set by the background health checker every 10 seconds; all policies filter `w.healthy == True` before selection; an engine that crashes is excluded from routing without restarting the gateway

### 03 — The LoadBalancingPolicy Interface (`03_load_balancing_policy.md`)
- `LoadBalancingPolicy` abstract base class: one required method `select(workers, info) → Worker | None`; two optional hooks `needs_prompt() → bool` and `on_request_complete(worker, success)`
- `SelectWorkerInfo` dataclass: carries `prompt_text` extracted from the request body; only `PrefixCacheAwarePolicy` sets `needs_prompt()` to True; other policies receive an empty string and ignore it
- Why `needs_prompt()` is a flag rather than always extracting: JSON parsing and string concatenation of chat messages adds ~0.1 ms per request; round-robin and least-load do not need it; the flag makes extraction opt-in per policy
- Mirrors SGLang's `LoadBalancingPolicy` trait in `src/policies/mod.rs`: `select_worker(&self, workers, info) → Option<usize>`; `needs_request_text() → bool`; `on_request_complete(url, success)`

### 04 — RoundRobinPolicy (`04_round_robin_policy.md`)
- Counter `% len(healthy)`: `self._counter` increments on every `select()` call; the result indexes into the filtered healthy worker list; counter never resets — wrapping is natural via modulo
- No knowledge of load or cache: the policy is correct in the sense that it distributes requests evenly over time, but it does not react to engine speed differences or KV cache state
- When to use: homogeneous engines under predictable, stateless traffic (e.g. embedding requests, requests with no shared prefixes); also useful as a baseline to compare against prefix-cache-aware throughput
- Mirrors SGLang's `RoundRobinPolicy` in `src/policies/round_robin.rs`: `AtomicUsize::fetch_add(1, Ordering::Relaxed) % healthy_len`

### 05 — LeastLoadPolicy (`05_least_load_policy.md`)
- `min(healthy, key=lambda w: w.load)`: picks the engine with the fewest in-flight requests at selection time; for N=2 this always compares both engines; for N>2 this is O(N) scan over healthy workers
- Degenerates from SGLang's PowerOfTwo: SGLang's `PowerOfTwoPolicy` picks two random workers and routes to the lighter one — O(1), avoids global scan on large fleets; for N=2 the result is identical
- Load as a proxy for engine busyness: a request that generates 1000 tokens holds `worker.load` elevated much longer than a request that generates 10 tokens; request count is an imperfect proxy but requires no communication with the engine
- When to use: engines with heterogeneous request lengths where round-robin would send long requests to an already-busy engine; also correct for mixed GPU/CPU deployments where one engine is inherently slower

### 06 — PrefixCacheAwarePolicy and RadixTrie (`06_prefix_cache_aware_policy.md`)
- `RadixTrie` per engine: character-level trie built from prompt text; `insert(text)` adds each character as a node; `match_len(text)` returns the longest prefix of text already present; `size()` returns total node count
- Why character-level rather than token-level: SGLang's `CacheAwarePolicy` also stores raw text characters — tokenization is expensive and unnecessary for routing; the trie only needs to identify which engine has seen a similar prompt, not an exact token match
- Two-stage selection: Stage 1 is a load imbalance guard — if `max(load) - min(load) > balance_abs_threshold`, ignore cache affinity and route to the least-loaded engine; Stage 2 is cache-affinity routing — find the engine with the longest prefix match, route there if `match_ratio >= cache_threshold`, otherwise route to the engine with the smallest trie
- Why the imbalance guard comes first: without it, a stalled engine could accumulate all requests for a popular prefix while the other engine sits idle; the guard ensures neither engine starves regardless of cache state
- After selection: `self._trees[chosen.url].insert(prompt)` records the prompt in the chosen engine's trie so future requests with the same prefix find a match; the trie grows with traffic and never shrinks (no eviction in this layer)
- Mirrors SGLang's `CacheAwarePolicy` in `src/policies/cache_aware.rs` + `Tree` in `src/policies/tree.rs`

### 07 — The Router and HTTP Forwarding (`07_router_and_http_forwarding.md`)
- `Router` class: owns `workers: list[Worker]`, `policy: LoadBalancingPolicy`, `client: httpx.AsyncClient`; one method `route(request, path)` handles all forwarding
- Prompt extraction: if `policy.needs_prompt()`, parse the request JSON body; for `/v1/chat/completions` concatenate all message `content` fields; for `/generate` or `/v1/completions` use the `prompt` field directly; pass to `SelectWorkerInfo`
- Load guard pattern: `worker.load += 1` before `await self.client.request(...)`, `worker.load -= 1` in `finally`; the `finally` block guarantees decrement even on httpx timeout or client disconnect; mirrors SGLang's RAII `WorkerLoadGuard`
- Header forwarding: strip `host` and `content-length` (stale for the new connection); forward all other headers including `authorization`, `content-type`, and custom headers; pass query params unchanged
- `/health` endpoint: polls `/health` on every engine via `check_health()`, updates `worker.healthy`, returns aggregate status 200 if all healthy or 207 if any are degraded; includes per-engine load and trie sizes for `prefix_cache_aware`

### 08 — The Full Loop (`08_the_full_loop.md`)
- End-to-end trace for one `/v1/chat/completions` request with `policy: prefix_cache_aware`, two engines, engine A holding a warm cache for "You are a helpful assistant"
- Step 1 — Request arrives at router port 8200: FastAPI catch-all route `/{path:path}` matches; body is read once with `await request.body()`
- Step 2 — Prompt extraction: JSON parsed, message contents joined → `"You are a helpful assistant. What is 2+2?"`
- Step 3 — Policy selects engine A: `PrefixCacheAwarePolicy.select()` checks load balance (both engines at 0), then checks trie match — engine A's trie matches 31 characters of the system prompt, engine B matches 0; ratio = 0.71 > 0.5 threshold → engine A chosen; prompt inserted into engine A's trie
- Step 4 — Forward: `engine_a.load = 1`; httpx POST to `http://localhost:8114/v1/chat/completions`; engine A's scheduler receives the request, RadixCache hits on the system prompt prefix, skips those prefill tokens
- Step 5 — Response: engine returns JSON; httpx response forwarded back; `engine_a.load = 0`; `policy.on_request_complete()` called
- Step 6 — Background health check (every 10 s): `_health_loop` fires, polls `/health` on both engines, updates `worker.healthy`; if engine B is down, it is excluded from future `select()` calls

### 09 — What Comes Next (`09_whats_next.md`)
- Prefill/decode disaggregation: the gateway currently treats each engine as a single unit; in production SGLang (`PdRouter` in `src/routers/http/pd_router.rs`), prefill workers and decode workers are registered separately; the router sends the prompt to a prefill worker, receives the first token and KV cache transfer metadata, then routes decode steps to a decode worker — the two roles can run on different hardware
- Streaming responses: the current `route()` method collects the full response body before returning; real deployments use SSE (server-sent events) streaming; httpx supports streaming with `client.stream()` and the router forwards chunks as they arrive, reducing time-to-first-token at the client
- The established pattern: each layer adds one mechanism (chunked prefill, prefix caching, quantization, speculative decoding, routing), one new file (`scheduler.py` additions, `radix_cache.py`, `model_gptq/`, `spec_runner.py`, `router.py`), and one benchmark metric; the benchmark measures exactly the throughput gain that the mechanism was designed to produce

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps Layer 15 concepts to SGLang source: `Worker` → `BasicWorker` in `src/core/worker.rs`; `LoadBalancingPolicy` → `LoadBalancingPolicy` trait in `src/policies/mod.rs`; `RoundRobinPolicy` → `src/policies/round_robin.rs`; `LeastLoadPolicy` → `src/policies/power_of_two.rs`; `PrefixCacheAwarePolicy` + `RadixTrie` → `src/policies/cache_aware.rs` + `src/policies/tree.rs`; `Router.route()` → `Router::proxy_request` in `src/routers/http/router.rs`; `WorkerLoadGuard` → `try/finally` in `router.py`

---

## Key Code Anchors

| Concept | Location |
|---|---|
| `Worker` dataclass | `router.py` line 108: `class Worker:` |
| `load` increment (WorkerLoadGuard equivalent) | `router.py` line 400: `worker.load += 1` |
| `RadixTrie` class | `router.py` line 138: `class RadixTrie:` |
| `RadixTrie.insert` | `router.py` line 145: `def insert(self, text: str) -> None:` |
| `RadixTrie.match_len` | `router.py` line 155: `def match_len(self, text: str) -> int:` |
| `SelectWorkerInfo` dataclass | `router.py` line 179: `class SelectWorkerInfo:` |
| `LoadBalancingPolicy` ABC | `router.py` line 191: `class LoadBalancingPolicy(ABC):` |
| `needs_prompt()` flag | `router.py` line 201: `def needs_prompt(self) -> bool:` |
| `RoundRobinPolicy` | `router.py` line 216: `class RoundRobinPolicy(LoadBalancingPolicy):` |
| `RoundRobinPolicy.select` | `router.py` line 221: `def select(self, workers, info):` |
| `LeastLoadPolicy` | `router.py` line 242: `class LeastLoadPolicy(LoadBalancingPolicy):` |
| `PrefixCacheAwarePolicy` | `router.py` line 279: `class PrefixCacheAwarePolicy(LoadBalancingPolicy):` |
| Stage 1: imbalance guard | `router.py` line 304: `if max(loads) - min(loads) > self._balance_abs_threshold:` |
| Stage 2: cache threshold check | `router.py` line 320: `if ratio >= self._cache_threshold:` |
| Trie insert after routing | `router.py` line 328: `self._trees[chosen.url].insert(prompt)` |
| `Router` class | `router.py` line 354: `class Router:` |
| `Router._extract_prompt` | `router.py` line 367: `async def _extract_prompt(self, body: bytes) -> str:` |
| `Router.route` | `router.py` line 382: `async def route(self, request: Request, path: str) -> httpx.Response:` |
| `Router.check_health` | `router.py` line 416: `async def check_health(self) -> dict[str, Any]:` |
| Background health loop | `router.py` line 439: `async def _health_loop(router: Router) -> None:` |
| Policy factory | `router.py` line 454: `def _make_policy(name: str, workers: list[Worker]) -> LoadBalancingPolicy:` |
