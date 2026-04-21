# The Power of Two Choices in Randomized Load Balancing

**Source:** https://www.eecs.harvard.edu/~michaelm/postscripts/tpds2001.pdf
**Author:** Michael Mitzenmacher (Harvard University)
**Venue:** IEEE Transactions on Parallel and Distributed Systems, Vol. 12, No. 10, October 2001
**DOI:** 10.1109/71.963420
**Level:** L4 (theory spine) / L3 (implication only)
**Why here:** The mathematical foundation for `LeastLoadPolicy`. Proves that sampling 2 servers and routing to the least-loaded one gives exponential improvement over random assignment. Every LLM router that implements "power of two choices" (SGLang `PowerOfTwoPolicy`, vLLM Router PoT, TetriInfer) cites this paper.

---

## Abstract

We consider the **supermarket model**: customers arrive as a Poisson stream of rate λn (λ < 1) at a collection of n servers. Each customer chooses d servers independently and uniformly at random and waits for service at the one with the **fewest customers**.

Key finding: Having d = 2 choices leads to **exponential improvements** in expected wait time over d = 1, whereas having d = 3 choices is only a **constant factor** better than d = 2.

---

## The Core Theorem

For d = 2 choices, the expected time a customer spends in the system is bounded above by:

$$\sum_{i=1}^{\infty} \frac{\lambda^{d^i - d} \cdot d}{d-1} + o(1)$$

For d = 1 (random assignment): max queue length scales as O(log n / log log n) with n servers.
For d = 2 (power of two): max queue length scales as O(log log n) — **exponential improvement**.
For d = 3: only O(1) better than d = 2. Not worth the extra probe cost.

**The practical implication:** Sample 2 workers at random. Route to the less loaded one. This provides near-optimal load distribution with minimal coordination overhead.

---

## Why d=2 and Not d=All

The paper's most important engineering insight:

> "Having d = 2 choices leads to exponential improvements in the expected time a customer spends in the system over d = 1, whereas having d = 3 choices is only a constant factor better than d = 2. We discuss the possible implications for system design."

For a router with N workers:
- **Checking all N workers**: O(N) overhead per request, but optimal placement.
- **Checking 2 workers**: O(1) overhead per request, but still O(log log N) max queue — exponentially better than O(log N / log log N).
- **Checking 3 workers**: Marginally better than 2, but the overhead is 50% higher.

This is why Layer 15's `LeastLoadPolicy` samples exactly 2 workers, not 3 or all:

```python
class LeastLoadPolicy(LoadBalancingPolicy):
    def _pick_worker(self, workers):
        # Sample 2 workers at random (Mitzenmacher 2001)
        a, b = random.sample(workers, 2)
        return a if a.in_flight <= b.in_flight else b
```

---

## The Balls-and-Bins Model

The supermarket model is equivalent to the balls-and-bins problem:
- n balls thrown into n bins.
- With one random choice per ball: max load is O(log n / log log n).
- With two random choices per ball (pick the less-loaded bin): max load is O(log log n).

This applies to any load distribution problem where:
- Work units (requests) arrive dynamically.
- Multiple workers (servers) have varying load.
- Routing decisions must be made quickly with limited global information.

---

## Brief History

The earliest application to load balancing: Eager, Lazowska, and Zahorjan (1986) — empirical evidence that two-choice policies perform well.

The rigorous analytical demonstration: Karp, Luby, and Meyer auf der Heide (1992, 1996).

The Mitzenmacher (2001) paper: Full proof for the supermarket model using fluid limit (differential equations) analysis. Independently derived by Vvedenskaya, Dobrushin, and Karpelevich (1996).

---

## Connection to LLM Routing

The supermarket model maps to LLM inference routing:
- **Customers** = inference requests
- **Servers** = GPU worker instances
- **Service time** = time to generate the response (unpredictable, auto-regressive)
- **Queue length** = in-flight requests per worker (`Worker.in_flight`)

The unpredictability of LLM output length (unlike exponential service times in the supermarket model) makes the two-choice heuristic especially valuable — you can't predict which worker will finish first, but sampling two and picking the less-loaded one is always better than random.

---

## Citations in LLM Routing Papers

This paper is cited in:
- **SkyWalker (EUROSYS 2026)**: As the foundation for least-load routing baselines.
- **TetriInfer (Hu et al., 2024)**: Adopts power-of-two load balancing for the decode worker pool in P/D disaggregation.
- **Block (2025)**: Uses PoT as baseline, proposing a more complex approach.
- **A Survey of LLM Inference Systems (2025)**: As the canonical reference for "power-of-two load balancing".
- **SGLang sgl-model-gateway**: `PowerOfTwoPolicy` (Rust) implements this algorithm.
- **vLLM Router**: `PowerOfTwo (PoT)` policy implements this algorithm.

---

## What This Paper Does Not Cover

- Prefix caching: the supermarket model treats all customers as equivalent. LLM requests are NOT equivalent — those with matching prefixes benefit from routing to specific workers.
- The `cache_threshold` / `balance_abs_threshold` dials: these are not in Mitzenmacher's model. They are production additions by SGLang/Preble to handle the cache-vs-load tradeoff.
- Multi-turn conversation locality: the supermarket model is stateless per customer. LLM serving has session state (KV cache).

Mitzenmacher 2001 establishes the **floor**: always do at least as well as random by sampling two workers. `PrefixCacheAwarePolicy` builds on top by incorporating KV cache state when the load is balanced.
