# Layer 15 — Summary

Layer 15 adds a policy-based HTTP gateway in front of multiple engine instances via a new `router.py` file containing six classes and one background task. The engine (`server.py`), scheduler, model runner, KV cache, radix cache, and all model files are unchanged; clients that previously talked directly to one engine now talk to the router, which forwards to one of N engines based on a routing policy.

---

## From Layer 14 to Layer 15

In Layer 14, a client sent requests directly to a single engine. That engine's scheduler queued requests and processed them one batch at a time:

```python
# Layer 14 — client calls one engine directly
POST http://localhost:8114/v1/chat/completions
# One scheduler, one model runner, one KV pool.
# Max throughput bounded by one GPU's forward pass speed.
```

In Layer 15, a gateway process accepts the request and forwards it to one of two (or more) engines based on a policy:

```python
# Layer 15 — client calls the router
POST http://localhost:8200/v1/chat/completions
#   → Router.route()
#   → policy.select(workers, info)  →  Worker(url="http://localhost:8114")
#   → httpx forward to engine A
#   ← response proxied back
```

The router itself holds no model weights, no KV cache, and no scheduler. Its only job is to pick an engine and forward the bytes. The structural difference is that the forwarding decision — which engine — is made by a pluggable policy class, and that choice has a direct effect on whether the receiving engine's `RadixCache` from Layer 12 produces a hit.

---

## Worker and Load Tracking

The `Worker` dataclass is the router's view of one backend engine. It holds four fields: `url` (the engine's base address), `name` (a human label for logs and health output), `load` (the number of requests currently in-flight to that engine), and `healthy` (the result of the most recent `/health` poll).

```python
@dataclass
class Worker:
    url: str
    name: str
    load: int    = 0
    healthy: bool = True
```

The `load` field is the router's equivalent of SGLang's `AtomicUsize load_counter` in `BasicWorker`. It is incremented before each forwarded request and decremented in a `finally` block after the response returns, so it accurately reflects the number of requests the engine is currently processing regardless of how long each takes. SGLang wraps this pattern in a `WorkerLoadGuard` RAII struct that calls `increment_load` on construction and `decrement_load` on drop. In `router.py` the pattern appears directly in `Router.route`:

```python
worker.load += 1           # WorkerLoadGuard::new → increment_load
try:
    response = await self.client.request(...)
finally:
    worker.load -= 1       # WorkerLoadGuard::drop → decrement_load
```

The `healthy` field is updated by the background health checker every ten seconds. All three policies filter for `w.healthy == True` before selecting, so an engine that crashes is automatically excluded from routing without restarting the gateway.

---

## The LoadBalancingPolicy Interface

`LoadBalancingPolicy` is an abstract base class with one required method and two optional hooks:

```python
class LoadBalancingPolicy(ABC):

    @abstractmethod
    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        ...

    def on_request_complete(self, worker: Worker, success: bool) -> None:
        pass

    def needs_prompt(self) -> bool:
        return False
```

`select` receives the full worker list (not pre-filtered) and a `SelectWorkerInfo` dataclass carrying `prompt_text`. It returns the chosen `Worker` or `None` if no healthy engine is available (the router responds 503 in that case). `needs_prompt` tells the router whether to spend time extracting the prompt from the request body — `RoundRobinPolicy` and `LeastLoadPolicy` return `False` and receive an empty string; `PrefixCacheAwarePolicy` returns `True` and uses the text to query its tries. `on_request_complete` is called after every request and is a no-op in all three policies implemented here; it is the hook for future stateful policies (e.g. exponential backoff on repeated failures to a specific engine).

This interface mirrors SGLang's `LoadBalancingPolicy` trait in `src/policies/mod.rs`, which declares `select_worker`, `needs_request_text`, and `on_request_complete` with identical semantics.

---

## RoundRobinPolicy

`RoundRobinPolicy` maintains a single integer counter. On each `select` call it filters the worker list to healthy engines, picks the engine at index `counter % len(healthy)`, and increments the counter:

```python
class RoundRobinPolicy(LoadBalancingPolicy):

    def __init__(self) -> None:
        self._counter = 0

    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        healthy = [w for w in workers if w.healthy]
        if not healthy:
            return None
        chosen = healthy[self._counter % len(healthy)]
        self._counter += 1
        return chosen
```

The policy is stateless beyond the counter. It distributes requests evenly across engines over time regardless of how long each request takes, which means a slow engine can accumulate more concurrent requests than a fast one. Round-robin is appropriate when request latency is uniform (embedding models, fixed-length completions) or when engines are identical and load is expected to be balanced by the application layer. It serves as the baseline against which the other two policies are measured.

---

## LeastLoadPolicy

`LeastLoadPolicy` picks the engine with the smallest `load` value at the moment of selection:

```python
class LeastLoadPolicy(LoadBalancingPolicy):

    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        healthy = [w for w in workers if w.healthy]
        if not healthy:
            return None
        return min(healthy, key=lambda w: w.load)
```

Because `load` reflects in-flight requests and not completed tokens, a single long-running request (1000 output tokens) and a short request (10 output tokens) both increment `load` by exactly 1. The policy is therefore an imprecise proxy for actual engine busyness, but it reacts correctly to the most important case: when one engine is overwhelmed and the other is idle, the next request goes to the idle engine. SGLang's `PowerOfTwoPolicy` achieves the same outcome by picking two random workers and routing to the lighter one — an O(1) algorithm that avoids scanning all workers on large fleets. For two workers the results are identical.

---

## PrefixCacheAwarePolicy and RadixTrie

`PrefixCacheAwarePolicy` is the most important policy. Its purpose is to route requests with shared prompt prefixes to the same engine so that engine's `RadixCache` produces a KV hit and skips the corresponding prefill tokens. Without this policy, two requests with the same 512-token system prompt sent to different engines would both compute the full 512-token prefill; with it, the second request reuses the KV pages already stored by the first.

The policy maintains one `RadixTrie` per engine URL. `RadixTrie` is a character-level trie that stores prompt text verbatim — not token IDs:

```python
class RadixTrie:

    def insert(self, text: str) -> None:
        node = self._root
        for ch in text:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
                self._size += 1
            node = node.children[ch]
        node.end = True

    def match_len(self, text: str) -> int:
        node = self._root
        matched = 0
        for ch in text:
            if ch not in node.children:
                break
            node = node.children[ch]
            matched += 1
        return matched
```

Using raw characters rather than token IDs is the same design choice SGLang makes in `src/policies/tree.rs`. Tokenization at routing time would add per-request latency and a tokenizer dependency in the router process; character-level matching is sufficient to identify which engine has seen a similar prompt.

Selection proceeds in two stages. Stage 1 is a load imbalance guard: if the difference between the busiest and least-busy engine exceeds `balance_abs_threshold`, cache affinity is ignored and the request goes to the least-loaded engine to prevent starvation:

```python
loads = [w.load for w in healthy]
if max(loads) - min(loads) > self._balance_abs_threshold:
    return min(healthy, key=lambda w: w.load)
```

Stage 2 is cache-affinity routing: find the engine whose trie matches the longest prefix of the prompt, compute the match ratio, and route to that engine if the ratio exceeds `cache_threshold`. If no engine has a good match, route to the engine with the smallest trie (the one with the most available cache capacity):

```python
best  = max(healthy, key=lambda w: self._trees[w.url].match_len(prompt))
ratio = self._trees[best.url].match_len(prompt) / len(prompt)

if ratio >= self._cache_threshold:
    chosen = best
else:
    chosen = min(healthy, key=lambda w: self._trees[w.url].size())

self._trees[chosen.url].insert(prompt)
```

The final `insert` records this prompt in the chosen engine's trie so that future requests with the same prefix find a match. The trie grows monotonically in this layer; SGLang's production implementation adds a background LRU eviction task (`src/policies/utils.rs`) that trims the trie to `max_tree_size` nodes periodically.

---

## The Router and HTTP Forwarding

`Router` owns the worker list, the active policy, and an `httpx.AsyncClient`. Its `route` method is the central dispatch:

```python
async def route(self, request: Request, path: str) -> httpx.Response:
    body = await request.body()

    info = SelectWorkerInfo()
    if self.policy.needs_prompt():
        info.prompt_text = await self._extract_prompt(body)

    worker = self.policy.select(self.workers, info)
    if worker is None:
        raise HTTPException(status_code=503, detail="No healthy workers available")

    target_url = f"{worker.url}/{path}"
    headers = {k: v for k, v in request.headers.items()
               if k.lower() not in ("host", "content-length")}

    worker.load += 1
    success = False
    try:
        response = await self.client.request(
            method=request.method, url=target_url,
            headers=headers, content=body,
            params=dict(request.query_params),
        )
        success = response.is_success
        return response
    finally:
        worker.load -= 1
        self.policy.on_request_complete(worker, success)
```

The body is read once and passed both to `_extract_prompt` (for trie matching) and to `httpx` as the forwarded payload. The `host` and `content-length` headers are stripped because they belong to the client-router connection, not the router-engine connection; all other headers including `authorization` and `content-type` are forwarded unchanged. The `finally` block guarantees that `worker.load` is decremented even if httpx times out or the client disconnects mid-stream.

---

## The Full Loop

The gateway starts by loading config from the `router:` block in `config.yml`, constructing `Worker` objects for each listed URL, instantiating the configured policy, and running an initial health check before accepting traffic. A background `asyncio.Task` polls `/health` on every engine every ten seconds and updates `worker.healthy`.

Consider a request `POST http://localhost:8200/v1/chat/completions` with the body `{"messages":[{"role":"system","content":"You are a helpful assistant"},{"role":"user","content":"What is 2+2?"}], "max_tokens":32}`. The `policy` is `prefix_cache_aware`. Engine A has previously seen a request with the same system prompt; engine B has not.

FastAPI's catch-all route `/{path:path}` matches and calls `Router.route(request, "v1/chat/completions")`. The body is read. Because `policy.needs_prompt()` is `True`, `_extract_prompt` parses the JSON and produces `"You are a helpful assistant What is 2+2?"`. `policy.select` is called: both engines have `load=0` so the imbalance guard does not fire. Engine A's trie matches 29 characters (`"You are a helpful assistant"`), engine B's trie matches 0. The match ratio for engine A is `29/42 = 0.69 > 0.5`. Engine A is chosen. The prompt is inserted into engine A's trie.

`worker.load` for engine A becomes 1. The httpx client posts the original body to `http://localhost:8114/v1/chat/completions`. Engine A's scheduler receives the request. Its `RadixCache` finds the system prompt prefix in its trie and skips those prefill tokens, writing only the user turn's KV. The engine returns a JSON response. The router forwards it back to the client. `worker.load` drops to 0. `on_request_complete` is called (no-op for this policy).

The next request with the same system prompt again reaches the router. Engine A's trie now contains the full `"You are a helpful assistant What is 2+2?"` string. Any future prompt that starts with `"You are a helpful assistant"` will match engine A's trie and be routed there, compounding the RadixCache benefit as traffic accumulates.

---

## What Comes Next

Layer 15 implements the simplest form of multi-engine routing: each engine handles both prefill and decode for every request it receives. The natural next step is prefill/decode disaggregation — SGLang's `PdRouter` in `src/routers/http/pd_router.rs` — where prefill workers and decode workers are registered separately and the router sends the prompt to a prefill worker, receives the first token and KV transfer metadata, then routes all subsequent decode steps to a decode worker. This allows prefill to run on a high-memory GPU optimised for long-context processing while decode runs on a GPU optimised for low-latency generation, doubling effective hardware utilisation. The `Worker` and `LoadBalancingPolicy` classes from this layer carry forward unchanged; only the router's forwarding logic gains a two-phase dispatch step.
