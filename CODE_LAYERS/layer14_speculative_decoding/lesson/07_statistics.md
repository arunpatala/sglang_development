# 07 — Statistics and Measurement

Speculative decoding's benefit depends critically on the acceptance rate. A low acceptance rate means most draft work is wasted; a high acceptance rate means each target call commits many tokens. `SpecRunner` exposes running statistics that quantify both the acceptance rate and the realized token yield per step.

---

## The Four Counters

```python
# spec_runner.py — SpecRunner.__init__
self._total_accepted: int = 0   # cumulative accepted draft tokens across all spec steps
self._total_proposed: int = 0   # cumulative draft tokens proposed (= N × steps)
self._total_tokens:   int = 0   # cumulative committed tokens (accepted + bonus)
self._total_steps:    int = 0   # cumulative spec_decode_step calls
```

These are lifetime counters — they accumulate across all requests and never reset. This means `acceptance_rate` and `tokens_per_step` report server-lifetime averages, not per-request or per-window statistics. For a long-running server, the statistics converge to stable values as the sample size grows.

---

## acceptance_rate

```python
@property
def acceptance_rate(self) -> float:
    if self._total_proposed == 0:
        return 0.0
    return self._total_accepted / self._total_proposed
```

`acceptance_rate` ∈ [0, 1]. It answers: "of all the draft tokens proposed so far, what fraction did the target model agree with?"

For a homogeneous workload (all requests with the same prompt distribution), `acceptance_rate` is stable. For mixed workloads — some structured (code generation), some open-ended (chat) — the average reflects the mixture. Code generation typically achieves 0.80–0.90; open chat achieves 0.65–0.75.

`acceptance_rate` does not capture the position dependence: acceptance rates are typically higher for early positions (closer to the confirmed context) and lower for later positions (where draft errors compound). A compound acceptance rate of `p^5` for N=5 with `p=0.7` gives `0.7^5 = 0.168` probability that all 5 are accepted, but the expected yield (3.69) already accounts for this via the per-position calculation.

---

## tokens_per_step

```python
@property
def tokens_per_step(self) -> float:
    if self._total_steps == 0:
        return 1.0
    return self._total_tokens / self._total_steps
```

`tokens_per_step` is the key efficiency metric. It answers: "how many committed tokens (on average) does one `spec_decode_step` produce?"

The theoretical maximum is `N + 1` — all N draft tokens accepted plus one bonus. The theoretical minimum is `1` — all N draft tokens rejected, only the bonus committed. In practice, with `N=5` and `acceptance_rate=0.7`, `tokens_per_step ≈ 3.7`.

The speedup over Layer 13 (one token per target call) is approximately `tokens_per_step / (1 + N × draft_cost_ratio)` where `draft_cost_ratio = T_draft / T_target`. For the values above: `3.7 / (1 + 5 × 0.15) ≈ 3.7 / 1.75 ≈ 2.1×`.

---

## Logging and Benchmarking

```python
# spec_runner.py — SpecRunner.log_stats
def log_stats(self) -> None:
    logger.info(
        f"SpecRunner stats: "
        f"acceptance_rate={self.acceptance_rate:.3f} "
        f"tokens_per_step={self.tokens_per_step:.2f} "
        f"total_steps={self._total_steps} "
        f"total_tokens={self._total_tokens}"
    )
```

`server.py` calls `spec_runner.log_stats()` periodically (e.g., every 100 steps or on request completion). The logs allow offline analysis of whether the draft model choice was good for the workload.

The counter update in `spec_decode_step`:

```python
self._total_accepted += result.accept_len
self._total_proposed += self._n_speculative_tokens
self._total_tokens   += result.accept_len + 1
self._total_steps    += 1
```

This update happens once per `spec_decode_step` call regardless of how many requests are in the batch. For a batch of B requests, this undercounts `_total_accepted` and `_total_proposed` by a factor of B: the counters treat the B-request batch as a single "step." The properties `acceptance_rate` and `tokens_per_step` remain correct ratios, but `_total_steps` is the number of batch steps, not the number of per-request spec steps. For a fair per-request comparison, divide `_total_tokens` by the number of requests that participated.

---

## Dynamic N Adjustment (Not in Layer 14)

A natural extension is to adaptively tune `N` based on the observed acceptance rate:

- If `acceptance_rate > 0.8`: increase `N` from 5 to 7 — high acceptance means more draft tokens are worth proposing.
- If `acceptance_rate < 0.5`: decrease `N` from 5 to 3 — low acceptance means draft overhead outweighs the yield benefit.

Layer 14 exposes the statistics but leaves `N` fixed at `cfg.num_speculative_tokens`. Production systems (SGLang's speculative decoding implementation) implement exponential moving average acceptance rates and online `N` adjustment. This is a policy decision that requires careful hysteresis to avoid oscillation.

Section 08 traces a complete `spec_decode_step` for two requests to show all six phases — draft, verify, accept/reject, rewind, bonus, statistics — in sequence.
