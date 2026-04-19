"""
Layer 0 — Test Client
======================
Sends requests to the Layer 0 server and measures latency and throughput.

Three tests:
  1. Sequential — one request at a time, measure latency per request
  2. Concurrent — fire N requests simultaneously, see how the server degrades
  3. KV cache comparison — compare use_cache=False vs True (edit server.py)

Run:
    python test_client.py
    python test_client.py --url http://localhost:8100 --n 5
"""

import argparse
import concurrent.futures
import json
import statistics
import time
from pathlib import Path

import requests
import yaml

_HERE = Path(__file__).parent


def _load_config(path: str) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    with open(p) as f:
        return yaml.safe_load(f) or {}


_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--config", default=str(_HERE / "config.yml"))
_pre_args, _ = _pre.parse_known_args()

cfg = _load_config(_pre_args.config)

_host = cfg.get("host", "localhost")
_port = cfg.get("port", 8100)

parser = argparse.ArgumentParser()
parser.add_argument("--config", default=str(_HERE / "config.yml"), help="Path to YAML config file")
parser.add_argument("--url", default=f"http://{_host}:{_port}")
parser.add_argument("--n", type=int, default=4, help="Number of concurrent requests")
args = parser.parse_args()

BASE = args.url
N = args.n

PROMPTS = [
    "What is 2 + 2?",
    "Name the planets in the solar system.",
    "What is the speed of light?",
    "Who wrote Romeo and Juliet?",
    "What is the boiling point of water in Celsius?",
    "What is the largest ocean on Earth?",
    "Who painted the Mona Lisa?",
    "What is the chemical symbol for gold?",
]


def send(prompt: str, max_new_tokens: int = 32) -> dict:
    # Send OpenAI-style messages list — same format as the real OpenAI API.
    # The server applies the chat template internally, just like SGLang does.
    messages = [{"role": "user", "content": prompt}]
    t0 = time.perf_counter()
    r = requests.post(
        f"{BASE}/generate",
        json={"messages": messages, "max_new_tokens": max_new_tokens},
        timeout=120,
    )
    r.raise_for_status()
    wall_ms = round((time.perf_counter() - t0) * 1000, 1)
    data = r.json()
    data["wall_ms"] = wall_ms  # client-side end-to-end latency
    return data


def print_result(i: int, d: dict):
    print(
        f"  [{i:02d}] {d['latency_ms']:>7.0f}ms server | {d['wall_ms']:>7.0f}ms wall | "
        f"{d['completion_tokens']:>3} tokens | "
        f"{round(d['completion_tokens'] / (d['latency_ms'] / 1000)):>4} tok/s | "
        f"{d['text'][:50]!r}"
    )


# ---------------------------------------------------------------------------
# Check server is up
# ---------------------------------------------------------------------------
try:
    requests.get(f"{BASE}/health", timeout=5).raise_for_status()
except Exception as e:
    print(f"Server not reachable at {BASE}: {e}")
    print("Start the server first:  python server.py")
    raise SystemExit(1)

print(f"\nLayer 0 server at {BASE}\n")

# ---------------------------------------------------------------------------
# Test 1 — Sequential requests
# ---------------------------------------------------------------------------
print("=" * 70)
print(f"TEST 1: Sequential ({N} requests, one at a time)")
print("=" * 70)
print(f"  {'idx':>3}  {'server ms':>9}  {'wall ms':>7}  {'tokens':>6}  {'tok/s':>5}  output")

latencies = []
t_total = time.perf_counter()
for i, prompt in enumerate(PROMPTS[:N]):
    d = send(prompt, max_new_tokens=32)
    latencies.append(d["latency_ms"])
    print_result(i, d)

total_s = time.perf_counter() - t_total
print(f"\n  Total wall time: {total_s:.1f}s")
print(f"  Median latency:  {statistics.median(latencies):.0f}ms")
print(f"  Throughput:      {N / total_s:.2f} req/s  (limited by sequential blocking)")

# ---------------------------------------------------------------------------
# Test 2 — Concurrent requests
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print(f"TEST 2: Concurrent ({N} requests fired simultaneously)")
print("=" * 70)
print("  Note: Layer 0 server is synchronous — concurrent requests queue up.")
print("  The second request waits for the first to finish completely.")
print(f"  {'idx':>3}  {'server ms':>9}  {'wall ms':>7}  {'tokens':>6}  output")

prompts_to_use = (PROMPTS * 4)[:N]
results = [None] * N
wall_latencies = []

t_all = time.perf_counter()
with concurrent.futures.ThreadPoolExecutor(max_workers=N) as pool:
    futures = {pool.submit(send, p, 32): i for i, p in enumerate(prompts_to_use)}
    for fut in concurrent.futures.as_completed(futures):
        i = futures[fut]
        d = fut.result()
        results[i] = d
        wall_latencies.append(d["wall_ms"])

total_concurrent_s = time.perf_counter() - t_all

for i, d in enumerate(results):
    print_result(i, d)

print(f"\n  Total wall time (concurrent): {total_concurrent_s:.1f}s")
print(f"  Total wall time (sequential above): {total_s:.1f}s")
print(
    f"  Ratio: {total_concurrent_s / total_s:.2f}x  "
    f"(ideal would be 1.0x if truly parallel, but Layer 0 has no concurrency)"
)
print(f"  Max wall latency: {max(wall_latencies):.0f}ms  (this is head-of-line blocking)")

# ---------------------------------------------------------------------------
# Summary / what to notice
# ---------------------------------------------------------------------------
print()
print("=" * 70)
print("OBSERVATIONS (what Layer 0 teaches you to feel, not just read about)")
print("=" * 70)
print("""
  1. Sequential latency is purely dominated by GPU compute.
     There is no caching, no batching, no parallelism.

  2. Concurrent requests do NOT run in parallel — the server processes them
     one at a time. The last concurrent request waits for all prior ones.
     This is head-of-line blocking. A 200-token request blocks a 5-token one.

  3. The server is doing redundant work. Every request recomputes K and V
     for every token from scratch. If you send the same prompt twice, all
     that attention computation is repeated identically.

  4. GPU utilization is very low. During decode (generating one token at a
     time), the GPU is mostly waiting for memory reads, not computing.
     Batching multiple sequences together would fix this (Layer 2).

Next step: edit server.py, change use_cache=False to use_cache=True, and
re-run this client. That is Layer 1: you add exactly one line and get a
significant speedup — the KV cache.
""")

# ---------------------------------------------------------------------------
# Check stats endpoint
# ---------------------------------------------------------------------------
stats = requests.get(f"{BASE}/stats").json()
print(f"Server stats: {json.dumps(stats, indent=2)}")
