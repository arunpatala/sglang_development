# 05 — Observability: What HiCache Reports

## What This Section Covers

Before tuning HiCache, you need to know whether it is working. This section covers the three observability surfaces SGLang exposes for HiCache: structured log output from the scheduler, Prometheus metrics for each tier, and Grafana dashboard integration. It ends with an interpretation guide — given a set of metric readings, what do they mean and what action should you take.

---

## Layer 1: Scheduler Log Output

Every scheduler logging interval, SGLang prints a one-line status summary. Without HiCache, it looks like:

```
#reqs: 12 | throughput: 43.1 tok/s | #cached-token: 8192 | #running: 12
```

`#cached-token` is the count of GPU-resident prefix-cached tokens at that instant. When the GPU pool fills and eviction begins, this number plateaus and stays near the pool maximum. There is no way to tell from this line alone whether a prefix was served from GPU cache or re-prefilled.

With HiCache enabled, the `_log_hicache_stats()` method (`scheduler_metrics_mixin.py:694`) runs on every logging cycle and writes additional fields into `self.stats`:

```python
# scheduler_metrics_mixin.py:694
def _log_hicache_stats(self: Scheduler):
    if not self.enable_hierarchical_cache:
        return
    host_pool = getattr(self.tree_cache, "token_to_kv_pool_host", None)
    self.stats.hicache_host_used_tokens = host_pool.size - host_pool.available_size()
    self.stats.hicache_host_total_tokens = host_pool.size
```

These are published to Prometheus each cycle (see below). Separately, the `RadixCacheMetricsCollector` and `StorageMetricsCollector` increment counters each time an eviction or load-back completes.

---

## Layer 2: Prometheus Metrics — Complete Reference

### GPU Pool Metrics (always available)

These metrics exist regardless of whether HiCache is enabled. They reflect the GPU KV pool state.

| Metric | Type | Source line | Meaning |
|---|---|---|---|
| `sglang:cache_hit_rate` | Gauge | `metrics_collector.py:272` | Fraction of input tokens matched from GPU-resident prefix cache; recalculated each scheduler step |
| `sglang:kv_used_tokens` | Gauge | `metrics_collector.py:298` | GPU KV pool tokens currently in active use (being computed) |
| `sglang:kv_available_tokens` | Gauge | `metrics_collector.py:286` | GPU KV pool tokens free (neither active nor radix-cached) |
| `sglang:kv_evictable_tokens` | Gauge | `metrics_collector.py:292` | GPU KV pool tokens radix-cached and evictable (warm but not locked) |

`kv_used + kv_available + kv_evictable = max_total_num_tokens` (the GPU pool size).

When HiCache is working correctly, you expect `kv_evictable_tokens` to stay high and `kv_available_tokens` to stay low — the pool is full of reusable prefix pages. When eviction pressure is high, `kv_evictable_tokens` drops toward zero and `kv_available_tokens` rises as pages are freed (or offloaded to CPU).

### Host Pool Metrics (HiCache tier 2)

Created only when `enable_hierarchical_cache=True` (`metrics_collector.py:671`).

| Metric | Type | Source line | Meaning |
|---|---|---|---|
| `sglang:hicache_host_used_tokens` | Gauge | `metrics_collector.py:673` | Tokens currently occupying the CPU pinned buffer |
| `sglang:hicache_host_total_tokens` | Gauge | `metrics_collector.py:679` | Total capacity of the CPU pinned buffer |

These are updated by `_log_hicache_stats()` each scheduler tick. The ratio `host_used / host_total` is the CPU pool fill percentage.

### Eviction and Load-Back Metrics (HiCache tier 2 I/O)

Created by `RadixCacheMetricsCollector` (`metrics_collector.py:1573`), active when `enable_hierarchical_cache=True`.

| Metric | Type | Source line | Meaning |
|---|---|---|---|
| `sglang:evicted_tokens_total` | Counter | `metrics_collector.py:1639` | Cumulative tokens moved GPU → CPU (eviction events) |
| `sglang:eviction_duration_seconds` | Histogram | `metrics_collector.py:1631` | Time taken per GPU→CPU eviction; buckets from 1 ms to 1 s |
| `sglang:load_back_tokens_total` | Counter | `metrics_collector.py:1652` | Cumulative tokens moved CPU → GPU (load-back events) |
| `sglang:load_back_duration_seconds` | Histogram | `metrics_collector.py:1644` | Time taken per CPU→GPU load; buckets from 1 ms to 1 s |

The histogram bucket configuration for both metrics can be overridden via environment variables `SGLANG_BUCKET_EVICTION_DURATION` and `SGLANG_BUCKET_LOAD_BACK_DURATION` (comma-separated floats in seconds).

### Storage I/O Metrics (HiCache tier 3)

Created by `StorageMetricsCollector` (`metrics_collector.py:1466`), active when `hicache_storage_backend` is set.

| Metric | Type | Source line | Meaning |
|---|---|---|---|
| `sglang:prefetched_tokens_total` | Counter | `metrics_collector.py:1476` | Cumulative tokens moved from storage → CPU by the prefetch thread |
| `sglang:backuped_tokens_total` | Counter | `metrics_collector.py:1482` | Cumulative tokens moved from CPU → storage by the backup thread |
| `sglang:prefetch_pgs` | Histogram | `metrics_collector.py:1506` | Pages per prefetch batch; buckets [1, 5, 10, 50, 100] |
| `sglang:backup_pgs` | Histogram | `metrics_collector.py:1512` | Pages per backup batch; buckets [1, 5, 10, 50, 100] |
| `sglang:prefetch_bandwidth` | Histogram | `metrics_collector.py:1519` | Storage → CPU bandwidth in GB/s; buckets [0.1, 0.5, 1, 5, 10, 50, 100] |
| `sglang:backup_bandwidth` | Histogram | `metrics_collector.py:1526` | CPU → storage bandwidth in GB/s; same buckets |

---

## Layer 3: PromQL Queries

Copy these directly into Grafana or `curl http://localhost:29000/metrics | grep sglang:` to check current values.

### GPU cache health

```promql
# Overall prefix cache hit rate — primary health indicator
sglang:cache_hit_rate

# GPU pool breakdown (should sum to max_total_num_tokens)
sglang:kv_used_tokens
sglang:kv_evictable_tokens
sglang:kv_available_tokens
```

### HiCache tier-2 (CPU pool)

```promql
# CPU pool fill percentage (alert if sustained > 90%)
sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens * 100

# Eviction rate: GPU → CPU tokens per second
rate(sglang:evicted_tokens_total[1m])

# Load-back rate: CPU → GPU tokens per second
rate(sglang:load_back_tokens_total[1m])

# P99 CPU → GPU load latency in milliseconds
histogram_quantile(0.99, rate(sglang:load_back_duration_seconds_bucket[5m])) * 1000

# P99 GPU → CPU eviction latency in milliseconds
histogram_quantile(0.99, rate(sglang:eviction_duration_seconds_bucket[5m])) * 1000
```

### HiCache tier-3 (storage)

```promql
# Storage prefetch rate: tokens per second being loaded from storage
rate(sglang:prefetched_tokens_total[1m])

# Storage backup rate: tokens per second being written to storage
rate(sglang:backuped_tokens_total[1m])

# Average storage prefetch bandwidth (GB/s)
histogram_quantile(0.50, rate(sglang:prefetch_bandwidth_bucket[5m]))

# Average storage backup bandwidth (GB/s)
histogram_quantile(0.50, rate(sglang:backup_bandwidth_bucket[5m]))
```

---

## Interpretation Guide

| What you observe | What it means | What to do |
|---|---|---|
| `cache_hit_rate` high (> 0.7), `load_back_tokens_total` growing fast | HiCache is working well — many prefix hits served from CPU instead of re-prefill | No action needed; this is the desired state |
| `cache_hit_rate` high, `load_back_tokens_total` flat | All hits are GPU-resident (tier 1 only); CPU pool not yet pressured | Workload has low eviction rate; HiCache cost is near zero |
| `cache_hit_rate` low, `evicted_tokens_total` high | Requests don't share prefixes; heavy eviction churn with no reuse | HiCache adds overhead without benefit; review workload prefix sharing |
| `hicache_host_used_tokens / total` near 100% sustained | CPU pool is full; new evictions from GPU have nowhere to go (fall through to storage or discard) | Increase `--hicache-ratio` or add more host RAM |
| `load_back_duration_seconds` P99 > 200 ms | PCIe bandwidth saturation or NUMA affinity issue | Check PCIe topology with `nvidia-smi topo -m`; try pinning GPU and CPU to the same NUMA node |
| `prefetched_tokens_total` growing, load-back latency stable | Storage prefetch is successfully hiding tier-3 read latency | Prefetch policy is working; monitor `prefetch_bandwidth` for storage headroom |
| `prefetched_tokens_total` flat, `load_back_tokens_total` high | Storage backend not being reached or prefetch is too slow | Check storage backend connectivity; increase `prefetch_threshold` |
| Both `evicted_tokens_total` and `load_back_tokens_total` low | Workload has low cache pressure overall | HiCache adds no benefit; consider disabling if RAM is constrained |

---

## Layer 4: Grafana Dashboard

The pre-built Grafana dashboard JSON at `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` includes a `Cache Hit Rate` panel that plots `sglang:cache_hit_rate`. This is the most visible indicator of overall cache effectiveness.

To monitor HiCache specifically, add three panels manually:

**Panel 1: CPU Pool Utilisation**
- Type: Gauge
- Query: `sglang:hicache_host_used_tokens / sglang:hicache_host_total_tokens * 100`
- Thresholds: green < 70%, yellow 70–90%, red > 90%
- Alert: `> 90% for 5m` → "CPU pool near capacity, consider increasing hicache-ratio"

**Panel 2: Tier-2 Transfer Rates**
- Type: Time series, two lines
- Query A: `rate(sglang:evicted_tokens_total[1m])` — label "GPU→CPU evictions/s"
- Query B: `rate(sglang:load_back_tokens_total[1m])` — label "CPU→GPU loads/s"
- The ratio B/A over time shows what fraction of evictions are being usefully loaded back. A sustained high ratio confirms temporal locality.

**Panel 3: Load-Back Latency P99**
- Type: Time series
- Query: `histogram_quantile(0.99, rate(sglang:load_back_duration_seconds_bucket[5m])) * 1000`
- Unit: milliseconds
- Alert: `> 200ms for 5m` → "Load-back latency high, investigate PCIe bandwidth"

Setup instructions for the full monitoring stack (Prometheus + Grafana via Docker Compose): `REPOS/sglang/examples/monitoring/README.md`.

To verify metrics are being exported before Grafana is set up:

```bash
curl http://localhost:30000/metrics | grep hicache
# Expected output:
# sglang:hicache_host_used_tokens{...} 12345.0
# sglang:hicache_host_total_tokens{...} 20480.0
```

---

## Key Files Referenced

| File | What it shows |
|---|---|
| `REPOS/sglang/python/sglang/srt/observability/scheduler_metrics_mixin.py:694` | `_log_hicache_stats()` — reads host pool state into stats |
| `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:272` | `sglang:cache_hit_rate` Gauge definition |
| `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:671` | `hicache_host_used_tokens` / `hicache_host_total_tokens` Gauge definitions |
| `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1466` | `StorageMetricsCollector` — tier-3 I/O counters and histograms |
| `REPOS/sglang/python/sglang/srt/observability/metrics_collector.py:1573` | `RadixCacheMetricsCollector` — eviction and load-back counters |
| `REPOS/sglang/examples/monitoring/grafana/dashboards/json/sglang-dashboard.json` | Pre-built Grafana dashboard (includes cache hit rate panel) |
| `REPOS/sglang/examples/monitoring/README.md` | Docker Compose setup for Prometheus + Grafana |
