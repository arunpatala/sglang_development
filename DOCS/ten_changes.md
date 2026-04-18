# Ten SGLang Changes — Increasing Difficulty

All changes are in `REPOS/sglang/python/sglang/` (editable install — no reinstall needed, just restart server).

---

## 1. Change the startup banner
**File:** `srt/entrypoints/http_server.py:2031`
**Change:** Edit the `"The server is fired up and ready to roll!"` string.
**Test:** Restart server, see new message in logs.
**Status:** [x]

## 2. Log every incoming chat request
**File:** `srt/entrypoints/http_server.py:1491`
**Change:** Add `logger.info(f">>> chat request: {request.messages[-1].content[:80]}")` inside `openai_v1_chat_completions`.
**Test:** Send a request, see log line in server terminal.
**Status:** [x]

## 3. Add a custom `/hello` endpoint
**File:** `srt/entrypoints/http_server.py` (anywhere near other routes)
**Change:** Add a new GET route:
```python
@app.get("/hello")
async def hello():
    return {"hello": "from my sglang", "status": "ok"}
```
**Test:** `curl http://localhost:30000/hello`
**Status:** [x]

## 4. Add a field to `/v1/models` response
**File:** `srt/entrypoints/http_server.py:1638`
**Change:** Return raw dict with `sglang_version` and `attention_backend` fields added.
**Test:** `curl http://localhost:30000/v1/models`
**Status:** [x]

## 5. Add a request counter
**File:** `srt/entrypoints/http_server.py`
**Change:** Add a module-level `_request_count = 0`, increment it in `openai_v1_chat_completions`, expose it at `/stats/requests`.
**Test:** Send 3 requests, hit `/stats/requests`, see count=3.
**Status:** [x]

## 6. Enforce a max prompt length
**File:** `srt/entrypoints/http_server.py:1491`
**Change:** Before dispatching, check `len(request.messages[-1].content) > 500` and return a 400 error with a message.
**Test:** Send a long prompt, get error. Send short prompt, works.
**Status:** [x]

## 7. Change the default sampling temperature
**File:** `srt/sampling/sampling_params.py`
**Change:** Find the default `temperature` field and change it (e.g. `0.7` → `0.1`). Observe more deterministic outputs on requests that don't specify temperature.
**Test:** Send identical prompts multiple times, compare variance.
**Status:** [x]

## 8. Add a per-request latency log
**File:** `srt/entrypoints/http_server.py` + `srt/entrypoints/openai/protocol.py`
**Change:** `time.perf_counter()` around `handle_request()`, log + inject `latency_ms` into `UsageInfo`.
**Test:** `usage.latency_ms` appears in the JSON response.
**Status:** [x]

## 9. Add a simple in-memory response cache
**File:** `srt/entrypoints/http_server.py`
**Change:** Add a `dict` keyed by prompt hash. On cache hit, return stored response immediately without calling `handle_request()`. Log cache hit/miss.
**Test:** Send identical prompt twice — second response is instant and logs "cache hit".
**Status:** [ ]
