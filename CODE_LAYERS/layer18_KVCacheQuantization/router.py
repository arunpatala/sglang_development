"""
Layer 15 — Router: policy-based HTTP gateway in front of multiple engine instances.

Mirrors sgl-model-gateway (REPOS/sglang/sgl-model-gateway/src/) in Python:

    Client
      │  POST /v1/chat/completions
      ▼
    Router  (this file, port 8200)
      │  policy.select(workers, prompt) → Worker
      │  worker.load += 1
      │  httpx.forward(worker.url, request)
      │  worker.load -= 1
      ▼
    Engine A (server.py, port 8114)  or  Engine B (server.py, port 8115)

Three policies — all implement LoadBalancingPolicy.select():
  round_robin        → RoundRobinPolicy        (sgl: RoundRobinPolicy)
  least_load         → LeastLoadPolicy         (sgl: PowerOfTwoPolicy)
  prefix_cache_aware → PrefixCacheAwarePolicy  (sgl: CacheAwarePolicy + Tree)

Configuration is read from config.yml under the `router:` key.
CLI args override individual router fields:
    python router.py                          # all values from config.yml
    python router.py --policy round_robin     # override policy
    python router.py --port 8201             # override gateway port
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx
import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

# ── Config ─────────────────────────────────────────────────────────────────────

_CONFIG_FILE = Path(__file__).parent / "config.yml"


def _load_config(cli_overrides: dict) -> dict:
    with open(_CONFIG_FILE) as f:
        cfg = yaml.safe_load(f)
    rcfg: dict = cfg.get("router", {})
    rcfg.update({k: v for k, v in cli_overrides.items() if v is not None})
    return rcfg


def _parse_args() -> dict:
    p = argparse.ArgumentParser(description="Layer 15 router")
    p.add_argument("--host",      type=str,   default=None)
    p.add_argument("--port",      type=int,   default=None)
    p.add_argument("--policy",    type=str,   default=None,
                   choices=["round_robin", "least_load", "prefix_cache_aware"])
    p.add_argument("--log-level", type=str,   default=None, dest="log_level")
    p.add_argument("--cache-threshold",       type=float, default=None, dest="cache_threshold")
    p.add_argument("--balance-abs-threshold", type=int,   default=None, dest="balance_abs_threshold")
    args = p.parse_args()
    return {k: v for k, v in vars(args).items() if v is not None}


_ARGS = _parse_args()
_CFG  = _load_config(_ARGS)

HOST                  = _CFG.get("host",                  "0.0.0.0")
PORT                  = int(_CFG.get("port",              8200))
LOG_LEVEL             = _CFG.get("log_level",             "warning")
POLICY_NAME           = _CFG.get("policy",                "prefix_cache_aware")
CACHE_THRESHOLD       = float(_CFG.get("cache_threshold", 0.5))
BALANCE_ABS_THRESHOLD = int(_CFG.get("balance_abs_threshold", 32))
WORKER_CFGS           = _CFG.get("workers", [])

# ── Logging ────────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Worker ─────────────────────────────────────────────────────────────────────
# Mirrors: src/core/worker.rs — Worker trait + BasicWorker struct
#
# Fields:
#   url     — base URL of the engine  (sgl: Worker::url())
#   name    — human label for logs
#   load    — in-flight request count (sgl: BasicWorker::load_counter AtomicUsize)
#   healthy — last known health state (sgl: BasicWorker::healthy AtomicBool)
#
# Thread safety: load is read/written under a threading.Lock acquired by Router.
# In production SGLang, AtomicUsize gives lock-free safety; here a single asyncio
# event loop serialises access, so no lock is needed for correctness, but we keep
# the field explicit to match the concept.


@dataclass
class Worker:
    url: str
    name: str
    load: int   = field(default=0, repr=True)
    healthy: bool = field(default=True, repr=True)


# ── RadixTrie ──────────────────────────────────────────────────────────────────
# Mirrors: src/policies/tree.rs — Tree struct
#
# Character-level radix trie. Stores raw prompt text (not token IDs) so no
# tokenizer is needed at routing time — same design decision as sgl-model-gateway.
#
# Methods used by PrefixCacheAwarePolicy:
#   insert(text)        — add text into the trie, creating nodes as needed
#   match_len(text)     — length of the longest prefix of text present in the trie
#   size()              — total node count (proxy for "how full is this cache")
#
# SGLang's Tree also does LRU eviction on a background task. We omit that here;
# the trie is bounded by traffic volume on a 2-engine dev setup.


class _TrieNode:
    __slots__ = ("children", "end")

    def __init__(self) -> None:
        self.children: dict[str, _TrieNode] = {}
        self.end: bool = False


class RadixTrie:
    """Character-level radix trie for prompt prefix tracking."""

    def __init__(self) -> None:
        self._root = _TrieNode()
        self._size = 0

    def insert(self, text: str) -> None:
        """Insert text into the trie. Each character is one node."""
        node = self._root
        for ch in text:
            if ch not in node.children:
                node.children[ch] = _TrieNode()
                self._size += 1
            node = node.children[ch]
        node.end = True

    def match_len(self, text: str) -> int:
        """Return the length of the longest prefix of text present in the trie."""
        node = self._root
        matched = 0
        for ch in text:
            if ch not in node.children:
                break
            node = node.children[ch]
            matched += 1
        return matched

    def size(self) -> int:
        """Total node count — proxy for how much cache content this worker has seen."""
        return self._size


# ── SelectWorkerInfo ────────────────────────────────────────────────────────────
# Mirrors: src/policies/mod.rs — SelectWorkerInfo struct
#
# Carries per-request context that policies can use for their decision.
# Only prefix_cache_aware uses prompt_text; other policies ignore it.


@dataclass
class SelectWorkerInfo:
    prompt_text: str = ""


# ── LoadBalancingPolicy (abstract base) ────────────────────────────────────────
# Mirrors: src/policies/mod.rs — LoadBalancingPolicy trait
#
# One required method: select(workers, info) → Worker | None
# Optional hook:       on_request_complete(worker, success) — for stateful policies
# Optional flag:       needs_prompt() → bool — tells Router to extract prompt text


class LoadBalancingPolicy(ABC):

    @abstractmethod
    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        """Select a worker for the incoming request. Return None if none available."""
        ...

    def on_request_complete(self, worker: Worker, success: bool) -> None:
        """Called after each request. Override for stateful bookkeeping."""

    def needs_prompt(self) -> bool:
        """Return True if select() requires info.prompt_text to be populated."""
        return False


# ── RoundRobinPolicy ───────────────────────────────────────────────────────────
# Mirrors: src/policies/round_robin.rs — RoundRobinPolicy
#
# Atomic counter % len(healthy_workers).
# Completely stateless beyond the counter — no knowledge of load or cache.
#
# SGLang uses AtomicUsize::fetch_add(1, Ordering::Relaxed); we use a plain int
# because a single asyncio event loop serialises all selects.


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


# ── LeastLoadPolicy ────────────────────────────────────────────────────────────
# Mirrors: src/policies/power_of_two.rs — PowerOfTwoPolicy
#
# SGLang picks two random workers and routes to the lighter one (power-of-two
# choices). For exactly two workers that always compares both, so we simplify to
# min(workers, key=load) — identical result for N=2, correct generalisation for N>2.
#
# In SGLang, load comes from LoadMonitor (token-level metrics from each worker's
# /get_loads endpoint). Here we use in-flight request count (worker.load) which
# is accurate without a background poller.


class LeastLoadPolicy(LoadBalancingPolicy):

    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        healthy = [w for w in workers if w.healthy]
        if not healthy:
            return None
        return min(healthy, key=lambda w: w.load)


# ── PrefixCacheAwarePolicy ─────────────────────────────────────────────────────
# Mirrors: src/policies/cache_aware.rs — CacheAwarePolicy
#          src/policies/tree.rs        — Tree (radix trie)
#
# Core idea: route requests with the same prompt prefix to the same engine so
# that engine's RadixCache (from Layer 12) has a KV hit and skips prefill.
#
# Decision logic (matches SGLang's two-stage approach):
#
#   Stage 1 — load imbalance guard (SGLang: balance_abs_threshold check)
#     if max(load) - min(load) > balance_abs_threshold:
#         → route to least-loaded engine (don't sacrifice balance for cache)
#
#   Stage 2 — cache-affinity routing (SGLang: approximate tree match)
#     best  = engine whose trie has the longest prefix match with the prompt
#     ratio = best_match_len / len(prompt)
#     if ratio > cache_threshold:
#         → route to best  (likely cache hit)
#     else:
#         → route to engine with smallest trie (most free cache capacity)
#
#   After routing: insert prompt into chosen engine's trie.
#
# Parameters:
#   cache_threshold       — match ratio needed to prefer the cache-hit engine (default 0.5)
#   balance_abs_threshold — load gap that overrides cache affinity          (default 32)


class PrefixCacheAwarePolicy(LoadBalancingPolicy):

    def __init__(
        self,
        worker_urls: list[str],
        cache_threshold: float = 0.5,
        balance_abs_threshold: int = 32,
    ) -> None:
        self._trees: dict[str, RadixTrie] = {url: RadixTrie() for url in worker_urls}
        self._cache_threshold       = cache_threshold
        self._balance_abs_threshold = balance_abs_threshold

    def needs_prompt(self) -> bool:
        return True

    def select(self, workers: list[Worker], info: SelectWorkerInfo) -> Worker | None:
        healthy = [w for w in workers if w.healthy]
        if not healthy:
            return None

        # ── Stage 1: load imbalance guard ──────────────────────────────────────
        # If one engine is significantly busier, ignore cache affinity and route
        # to the lighter engine so neither engine starves.
        if len(healthy) > 1:
            loads = [w.load for w in healthy]
            if max(loads) - min(loads) > self._balance_abs_threshold:
                return min(healthy, key=lambda w: w.load)

        # ── Stage 2: cache-affinity routing ────────────────────────────────────
        prompt = info.prompt_text
        if not prompt:
            # No text to match on — fall back to least load
            return min(healthy, key=lambda w: w.load)

        # Find the engine whose trie has the longest prefix match
        def _match(w: Worker) -> int:
            return self._trees[w.url].match_len(prompt)

        best   = max(healthy, key=_match)
        ratio  = _match(best) / len(prompt)

        if ratio >= self._cache_threshold:
            # Good match — route here, cache hit likely
            chosen = best
        else:
            # Poor match — route to engine with smallest trie (most cache capacity free)
            chosen = min(healthy, key=lambda w: self._trees[w.url].size())

        # Record prompt into chosen engine's trie for future routing
        self._trees[chosen.url].insert(prompt)
        return chosen

    def stats(self) -> dict[str, Any]:
        return {
            url: {"trie_size": t.size()}
            for url, t in self._trees.items()
        }


# ── Router ─────────────────────────────────────────────────────────────────────
# Mirrors: src/routers/http/router.rs — Router struct
#
# Owns the worker list, the active policy, and an httpx.AsyncClient.
# On each request:
#   1. Extract prompt text if the policy needs it (needs_prompt())
#   2. Call policy.select() to pick a worker
#   3. worker.load += 1  (mirrors WorkerLoadGuard::new → increment_load)
#   4. Forward the full request body to worker.url via httpx
#   5. worker.load -= 1  in finally  (mirrors WorkerLoadGuard::drop → decrement_load)
#   6. Call policy.on_request_complete() for stateful bookkeeping
#
# Health checks: /health on the router polls /health on every engine and reports
# their status. Unhealthy engines are excluded from routing by policy.select().


class Router:

    def __init__(
        self,
        workers: list[Worker],
        policy: LoadBalancingPolicy,
        client: httpx.AsyncClient,
    ) -> None:
        self.workers = workers
        self.policy  = policy
        self.client  = client
        self._health_lock = asyncio.Lock()

    async def _extract_prompt(self, body: bytes) -> str:
        """Pull prompt text from the request body for cache-aware routing."""
        try:
            data = json.loads(body)
            # /v1/chat/completions  →  concatenate all message contents
            if "messages" in data:
                parts = [m.get("content", "") for m in data["messages"] if isinstance(m, dict)]
                return " ".join(parts)
            # /generate or /v1/completions  →  plain prompt field
            if "prompt" in data:
                return str(data["prompt"])
        except Exception:
            pass
        return ""

    async def route(self, request: Request, path: str) -> httpx.Response:
        """Select a worker and forward the request. Mirrors Router::proxy_request."""
        body = await request.body()

        info = SelectWorkerInfo()
        if self.policy.needs_prompt():
            info.prompt_text = await self._extract_prompt(body)

        worker = self.policy.select(self.workers, info)
        if worker is None:
            raise HTTPException(status_code=503, detail="No healthy workers available")

        target_url = f"{worker.url}/{path}"
        headers = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ("host", "content-length")
        }

        worker.load += 1
        success = False
        try:
            response = await self.client.request(
                method  = request.method,
                url     = target_url,
                headers = headers,
                content = body,
                params  = dict(request.query_params),
            )
            success = response.is_success
            return response
        finally:
            worker.load -= 1                               # WorkerLoadGuard::drop
            self.policy.on_request_complete(worker, success)

    async def check_health(self) -> dict[str, Any]:
        """Poll /health on every engine and update worker.healthy."""
        results: dict[str, Any] = {}
        for w in self.workers:
            try:
                r = await self.client.get(f"{w.url}/health", timeout=3.0)
                w.healthy = r.is_success
                results[w.name] = {"url": w.url, "healthy": w.healthy, "load": w.load}
            except Exception as exc:
                w.healthy = False
                results[w.name] = {"url": w.url, "healthy": False, "error": str(exc)}
        return results


# ── Background health checker ───────────────────────────────────────────────────
# Mirrors: src/core/worker.rs — HealthChecker background task
#
# Polls every engine on HEALTH_INTERVAL_SECS. Updates worker.healthy so that
# policy.select() naturally skips dead engines on the next request.

HEALTH_INTERVAL_SECS = 10


async def _health_loop(router: Router) -> None:
    while True:
        await asyncio.sleep(HEALTH_INTERVAL_SECS)
        try:
            await router.check_health()
        except Exception as exc:
            logger.warning("Health check loop error: %s", exc)


# ── FastAPI app ─────────────────────────────────────────────────────────────────

app    = FastAPI(title="Layer 15 — Router")
_router: Router = None   # set at startup


def _make_policy(name: str, workers: list[Worker]) -> LoadBalancingPolicy:
    if name == "round_robin":
        return RoundRobinPolicy()
    if name == "least_load":
        return LeastLoadPolicy()
    if name == "prefix_cache_aware":
        return PrefixCacheAwarePolicy(
            worker_urls           = [w.url for w in workers],
            cache_threshold       = CACHE_THRESHOLD,
            balance_abs_threshold = BALANCE_ABS_THRESHOLD,
        )
    raise ValueError(f"Unknown policy: {name!r}. Choose round_robin | least_load | prefix_cache_aware")


@app.on_event("startup")
async def startup() -> None:
    global _router
    logger.info(
        "Starting router  host=%s  port=%d  policy=%s  workers=%s",
        HOST, PORT, POLICY_NAME, [w["url"] for w in WORKER_CFGS],
    )
    workers = [Worker(url=w["url"], name=w.get("name", w["url"])) for w in WORKER_CFGS]
    policy  = _make_policy(POLICY_NAME, workers)
    client  = httpx.AsyncClient(timeout=120.0)
    _router = Router(workers=workers, policy=policy, client=client)

    # Initial health check — mark engines as healthy/unhealthy before serving
    await _router.check_health()

    # Launch background health poller
    asyncio.create_task(_health_loop(_router))
    logger.info("Router ready  policy=%s  backends=%s", POLICY_NAME, [w.url for w in workers])


@app.on_event("shutdown")
async def shutdown() -> None:
    if _router:
        await _router.client.aclose()


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> JSONResponse:
    """Aggregate health of all backend engines."""
    statuses = await _router.check_health()
    all_healthy = all(v["healthy"] for v in statuses.values())
    content: dict[str, Any] = {
        "status":  "ok" if all_healthy else "degraded",
        "policy":  POLICY_NAME,
        "workers": statuses,
    }
    if isinstance(_router.policy, PrefixCacheAwarePolicy):
        content["trie_stats"] = _router.policy.stats()
    return JSONResponse(content, status_code=200 if all_healthy else 207)


@app.api_route(
    "/{path:path}",
    methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
)
async def proxy(request: Request, path: str):
    """Forward every non-health request to a selected backend engine."""
    response = await _router.route(request, path)
    return JSONResponse(
        content    = response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text,
        status_code= response.status_code,
        headers    = {
            k: v for k, v in response.headers.items()
            if k.lower() not in ("content-encoding", "transfer-encoding", "content-length")
        },
    )


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "router:app",
        host      = HOST,
        port      = PORT,
        log_level = LOG_LEVEL,
    )
