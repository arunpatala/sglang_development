# Speculative Decoding Math: Algorithms & Speedup Limits

**Source:** https://mbrenndoerfer.com/writing/speculative-decoding-math-acceptance-criterion
**Author:** Michael Brenndoerfer
**Published:** January 17, 2026
**Read time:** 53 minutes (interactive Jupyter-style article)
**Level:** L3 — Full mathematical derivation of the acceptance criterion, speedup formula, and optimal draft length; with runnable code
**Why here:** The most rigorous publicly available mathematical treatment of speculative decoding outside of the original papers (Chen et al., Leviathan et al.). Derives everything from first principles: the acceptance criterion `α(x) = min(1, p/q)`, the residual distribution, expected tokens per iteration `E[N] = (1 - α^{k+1}) / (1 - α)`, the speedup formula `S = E[N] / (1 + k/c)`, and the optimal draft length. Companion to `lesson/06_accept_reject_rewind.md` and `lesson/07_statistics.md`.

---

## Why the math matters (article's framing)

Speculative decoding must produce outputs **statistically indistinguishable** from target-model-only sampling — not an approximation. The mathematics tells you exactly why naive shortcuts fail:

- **Naive shortcut 1:** Accept when both models agree, reject when they disagree → biases output toward the intersection of the two distributions.
- **Naive shortcut 2:** Always accept draft tokens and spot-check with the target occasionally → changes the output distribution to match the draft, not the target.

The mathematical framework rules both out and derives the unique correct criterion.

---

## Section 1: The Acceptance Criterion

### Setup

Let:
- `p(x)` = target model's probability distribution over vocabulary V
- `q(x)` = draft model's probability distribution over vocabulary V
- Both distributions sum to 1

When the draft model proposes token `x` (sampled from `q`), we need an acceptance probability `α(x)` such that — across both accepted and resampled-on-rejection tokens — the final output follows exactly `p(x)`.

### The acceptance criterion

```
α(x) = min(1, p(x) / q(x))
```

- If `p(x) ≥ q(x)`: accept always (the target wants this token *at least as much* as the draft)
- If `p(x) < q(x)`: accept with probability `p(x)/q(x)` (the draft overproposed this token)

This is classical **rejection sampling** (von Neumann, 1950s) applied to token selection.

### The residual distribution (what to do on rejection)

When a token is rejected, we cannot simply resample from `q` — that would keep trying to sample tokens the draft overestimates. Instead, sample from the **residual distribution**:

```
p_resid(x) = max(0, p(x) - q(x)) / Z

where Z = Σ_x max(0, p(x) - q(x))   (normalizing constant)
```

Geometric interpretation: the residual distribution captures the portions of the target histogram that "stick out above" the draft histogram — tokens the draft underestimated. Sampling from `p_resid` on rejection is precisely what's needed to restore the missing probability mass.

Key identity: `Z = 1 - P(accept)`. The residual mass equals the rejection probability. Everything adds up.

### Complete algorithm for one token

1. Draft model samples `x ~ q(x)`
2. Sample uniform `u ~ Uniform(0, 1)`
3. If `u < min(1, p(x)/q(x))`: **accept** — output `x`
4. Else: **reject** — sample replacement from `p_resid(x)` and output that

### Worked example (from the article)

Vocabulary: {A, B, C, D, E}

| Token | p(x) | q(x) | α(x) = min(1, p/q) |
|-------|------|------|---------------------|
| A | 0.40 | 0.20 | 1.00 (always accept) |
| B | 0.30 | 0.50 | 0.60 |
| C | 0.15 | 0.15 | 1.00 |
| D | 0.10 | 0.10 | 1.00 |
| E | 0.05 | 0.05 | 1.00 |

Draft proposes B (q(B) = 0.50). u = 0.72 > 0.60 → reject.

Residual: `max(0, p-q)` = {A: 0.20, B: 0, C: 0, D: 0, E: 0}. Normalized: p_resid(A) = 1.0.

On rejection of B, we always sample A. This is correct: A had probability 0.40 in the target but only 0.20 in the draft — the rejection mechanism recovers the missing 0.20 by redirecting all rejected-B probability mass to A.

**Verification:** P(output A) = P(draft A) × 1.0 + P(draft B) × P(reject B) × 1.0 = 0.20 + 0.50 × 0.40 × 1.0 = 0.20 + 0.20 = 0.40 = p(A). ✓

---

## Section 2: Expected Speedup Analysis

### Expected acceptance probability α

The overall expected acceptance probability for one draft token:

```
α = E_{x~q}[min(1, p(x)/q(x))]
  = Σ_x q(x) · min(1, p(x)/q(x))
  = Σ_x min(q(x), p(x))
```

**Interpretation:** α equals the total probability mass overlap between the two distributions. Range: [0, 1].
- α = 1: distributions identical → every draft token accepted
- α = 0: disjoint distributions → no draft tokens accepted
- In practice: α ∈ [0.6, 0.9] for well-matched model pairs

### Expected tokens per iteration E[N]

Given draft length `k`, under the i.i.d. assumption (each position has the same acceptance rate):

```
E[N] = (1 - α^{k+1}) / (1 - α)
```

Derivation: N = 1 + (number of accepted draft tokens). The j-th draft token is accepted only if all j-1 prior tokens were also accepted (probability αʲ). By linearity of expectation:

```
E[N] = 1 + Σ_{j=1}^{k} αʲ = Σ_{j=0}^{k} αʲ = (1 - α^{k+1}) / (1 - α)
```

Limits:
- As α → 1: E[N] → k+1 (accept all k draft tokens + bonus verification token)
- As α → 0: E[N] → 1 (reject immediately, get one token from residual)

### The speedup formula

Let `c = T_target / T_draft` (cost ratio; how much faster the draft model is).

```
S = E[N] / (1 + k/c)
  = [(1 - α^{k+1}) / (1 - α)] / (1 + k/c)
```

Numerator = benefit (expected tokens per iteration).
Denominator = cost (iteration cost in target-model-pass units).

When draft model is free (c → ∞): `S ≈ (1 - α^{k+1}) / (1 - α)` — the theoretical maximum.

### Optimal draft length table (from article's code)

| α | c | Optimal k | Speedup |
|---|---|-----------|---------|
| 0.6 | 10 | 3 | 1.67× |
| 0.6 | 20 | 4 | 1.92× |
| 0.6 | 50 | 6 | 2.17× |
| 0.7 | 10 | 4 | 1.98× |
| 0.7 | 20 | 6 | 2.35× |
| 0.7 | 50 | 8 | 2.76× |
| 0.8 | 10 | 6 | 2.47× |
| 0.8 | 20 | 8 | 3.09× |
| 0.8 | 50 | 11 | 3.82× |
| 0.9 | 10 | 10 | 3.43× |
| 0.9 | 20 | 13 | 4.67× |
| 0.9 | 50 | 19 | 6.37× |

**Reading the table:** For Layer 14's default config (`num_spec_tokens = 3`, which implies k=3), at α=0.7 and c=10, the table says the optimal is k=4 at 1.98×. k=3 would be slightly suboptimal — you're leaving some speedup on the table by not using one more draft token.

---

## Section 3: Draft Quality Effects

### Connection to total variation distance

```
α = 1 - TV(p, q)
```

where TV(p, q) = ½ Σ_x |p(x) - q(x)| (total variation distance).

This is not a coincidence — it's a mathematical identity. The acceptance probability equals one minus the fundamental statistical divergence between the two distributions. Higher draft quality → smaller TV distance → higher acceptance rate.

**Practical meaning:** If you draw a sample from either distribution at random, an optimal observer can identify which one it came from with probability at most `TV(p,q)` better than chance. When acceptance rates are 80%, the models are hard to tell apart 80% of the time.

### Quality-speed tradeoffs (the counterintuitive result)

The optimal draft model is NOT necessarily the one with the highest acceptance rate:

| Scenario | α | c | k=5 speedup |
|----------|---|---|-------------|
| Tiny draft (125M) | 0.40 | 100 | ~1.3× |
| Medium draft (7B) | 0.75 | 10 | ~2.4× |
| Large draft (30B) | 0.90 | 2.5 | ~1.8× |

> "The medium draft model — somewhat worse but much faster — outperforms a nearly perfect but slower draft model."

This is the key practical insight. A 30B draft for a 70B target might achieve 90% acceptance but only runs 2.5× faster than the target → diminishing speedup. A 7B draft achieves 75% acceptance but runs 10× faster → higher net speedup.

### Empirical acceptance rate ranges

| Context type | Typical α | Notes |
|-------------|-----------|-------|
| Code completion | 0.80–0.90 | Repetitive, predictable |
| Factual text | 0.75–0.85 | Formulaic phrasing |
| Creative writing | 0.60–0.70 | More exploration |
| Low temperature | approaches 1.0 | Both models pick same argmax |
| High temperature | lower | Both models explore differently |

---

## Section 4: Optimal Draft Length

### Why there's an interior optimum

E[N] grows *sublinearly* in k (geometric series → saturates at 1/(1-α)).
The iteration cost `1 + k/c` grows *linearly* in k.

→ There is always a finite optimal k* where marginal benefit = marginal cost.

The first-order condition (treating k as continuous):

```
d/dk [(1 - α^{k+1}) / (1 + k/c)] = 0

Numerator of derivative = 0:
-α^{k+1} ln(α) · (1 + k/c) = (1 - α^{k+1}) · (1/c)
```

No closed-form solution — solve numerically. For typical values (α ≈ 0.7, c ≈ 10), optimal k ∈ [4, 8]. This is why k=5 is the most common production default.

### Adaptive draft length strategies

| Strategy | Description | Benefit |
|----------|-------------|---------|
| Fixed k | Default for most systems | Simple, predictable |
| Adaptive k | Increase k when acceptance rates are high, decrease when low | 10–20% throughput gain |
| Tree-based | Generate multiple candidate continuations as a tree | Higher effective α at cost of memory |

---

## Code reference (runnable Python from the article)

```python
import numpy as np

def acceptance_probability(p: np.ndarray, q: np.ndarray) -> float:
    """α = Σ_x min(p(x), q(x)) — the probability mass overlap."""
    return np.sum(np.minimum(p, q))

def sample_with_rejection(p, q, draft_token):
    """Apply the acceptance criterion to a single draft token."""
    accept_prob = min(1.0, p[draft_token] / q[draft_token])
    u = np.random.uniform()
    if u < accept_prob:
        return draft_token, True
    else:
        residual = np.maximum(0, p - q)
        residual = residual / residual.sum()
        new_token = np.random.choice(len(p), p=residual)
        return new_token, False

def expected_tokens(alpha: float, k: int) -> float:
    """E[N] = (1 - α^{k+1}) / (1 - α)"""
    return (1 - alpha ** (k + 1)) / (1 - alpha)

def speedup(alpha: float, k: int, c: float) -> float:
    """S = E[N] / (1 + k/c)"""
    return expected_tokens(alpha, k) / (1 + k / c)

def find_optimal_k(alpha: float, c: float, max_k: int = 20):
    """Grid search for optimal draft length."""
    best_k, best_s = 1, speedup(alpha, 1, c)
    for k in range(2, max_k + 1):
        s = speedup(alpha, k, c)
        if s > best_s:
            best_s, best_k = s, k
    return best_k, best_s
```

**Verification experiment:** 100,000 samples through `sample_with_rejection` with known p and q → empirical distribution matches target p with max deviation < 0.001. The mathematical guarantee holds numerically.

---

## Section 5: Practical Limitations (from article's "Limitations" section)

These are the gotchas that the math doesn't capture:

| Limitation | Explanation |
|-----------|-------------|
| **i.i.d. assumption false** | Real acceptance rates vary by position and context — code has higher α than creative text within the same generation |
| **Full softmax required** | Computing p_resid requires full vocabulary softmax on rejection — incompatible with some memory-efficient generation tricks |
| **KV cache rollback** | Rejected tokens leave KV cache pages that must be deallocated — bugs here silently corrupt the output distribution |
| **Multi-GPU communication** | When draft and target run on separate GPUs, communicating logit tensors adds latency that reduces effective c |
| **Memory constraints** | Both models must fit in VRAM simultaneously; for large targets, the draft model may require quantization |
| **Continuous batching interaction** | Batch composition changes every iteration; diverse batch members have lower joint acceptance rates |

The KV cache rollback point connects directly to `lesson/06_accept_reject_rewind.md` — this is the bug the `_kv_rewind()` call prevents.

---

## How this maps to Layer 14

| Math concept | Layer 14 code |
|-------------|---------------|
| `α(x) = min(1, p(x)/q(x))` | Acceptance criterion in `_accept_reject()` |
| Residual distribution on rejection | `torch.multinomial(residual, 1)` in accept/reject loop |
| `E[N] = (1 - α^{k+1}) / (1 - α)` | `_total_accepted / _total_proposed` tracks empirical α in `lesson/07_statistics.md` |
| KV rewind on rejection | `_kv_rewind()` call — the bug this corrects is the "KV cache rollback" gotcha |
| `num_spec_tokens = k` | The k in the speedup formula |
| Draft model cost ratio c | Ratio of target/draft forward pass times — informs choice of draft model size |
| Optimal k* grid search | Dynamic speculation (EAGLE-2) adapts k at runtime based on observed α |

---

## Limits of this article (for book context)

- Mathematical derivations assume single-token position acceptance rates are i.i.d. — relaxing this assumption is an open research area (tree speculation, EAGLE-2 handle this partially)
- Does not address batched speculative decoding (multiple sequences simultaneously) — Layer 14's `SpecRunner` handles this, the single-request math doesn't directly generalize
- Interactive visualizations and quiz are in the web article — not reproducible from this markdown file
- Optimal k formulas are for static α — adaptive strategies (lesson/08) go further
