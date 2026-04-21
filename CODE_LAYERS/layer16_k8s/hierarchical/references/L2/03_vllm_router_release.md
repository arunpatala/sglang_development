# vLLM Router: A High-Performance and Prefill/Decode Aware Load Balancer for Large-scale Serving

**Source:** https://vllm.ai/blog/vllm-router-release
**Author:** vLLM team
**Date:** 2025
**Level:** L2 — Definitions + motivation (L4 for PD disaggregation section)
**Why here:** Confirms that Layer 15's `router.py` design (FastAPI proxy + policy + worker state) is a faithful minimal version of a real production router. The vLLM Router was forked from the SGLang model gateway — same codebase that Layer 15 studies.

---

## Summary

The vLLM Router is a high-performance, lightweight load balancer built in Rust for minimal overhead. It acts as an intelligent, state-aware load balancer that sits between clients and a fleet of vLLM workers, either in a Kubernetes or bare-metal GPU cluster.

> "The vLLM Router is derived from a fork of the SGLang model gateway, modified and simplified to work with vLLM."

---

## Core Architecture and Capabilities

### 1. Intelligent Load Balancing Strategies

The vLLM Router provides multiple load balancing algorithms:

- **Consistent Hashing**: Key policy for maximizing performance. Requests with the same routing key (e.g., session ID, user ID) are "sticky" — always routed to the same worker replica, maximizing KV cache reuse.
- **Power of Two (PoT)**: Low-overhead random-choice policy that provides excellent load distribution. ↔ Layer 15's `LeastLoadPolicy`.
- **Round Robin & Random**: Standard policies for stateless load distribution. ↔ Layer 15's `RoundRobinPolicy`.

### 2. Native Support for Prefill/Decode Disaggregation

The router orchestrates the **prefill/decode (P/D) disaggregation** architecture:
1. Routes new requests to the prefill worker group.
2. Upon completion, directs the request state to the appropriate decode worker for token generation.
3. Supports both NIXL and NCCL-based (with ZMQ discovery) disaggregation backends.

This is the **Layer 16 concept** previewed in `lesson/09_whats_next.md`.

---

## Enterprise-Grade Resiliency

- **Kubernetes Service Discovery**: Automatic discovery and routing to vLLM worker pods using label selectors.
- **Fault Tolerance**: Configurable retry logic (with exponential backoff and jitter) and circuit breakers.
- **Observability**: Built-in Prometheus endpoint (`/metrics`) with metrics on request volume, latency, error rates, and worker health.

Production features not in Layer 15:
- Circuit breaker (Layer 15 has basic health-check removal only)
- Kubernetes service discovery (Layer 15 uses static config)
- Prometheus metrics (Layer 15 uses Python logging only)

---

## Benchmark Analysis

**Llama 3.1 8B with 8 Prefill pods + 8 Decode pods:**
- vLLM Router throughput: **25% higher** than llm-d, **100% higher** than K8s-native load balancer.
- vLLM Router TTFT: Close to K8s-native, **1,200 ms faster** than llm-d.

**DeepSeek V3 with 1 Prefill pod (TP8) + 1 Decode pod (TP8):**
- vLLM Router throughput: Close to llm-d, **100% higher** than K8s-native.
- vLLM Router TTFT: **2,000 ms faster** than llm-d and K8s-native.

---

## Connection to Layer 15 Source Code

The vLLM Router is derived from `sgl-model-gateway` (the primary reference for Layer 15):

| vLLM Router | SGLang gateway | Layer 15 `router.py` |
|---|---|---|
| `PowerOfTwoPolicy` | `policies/power_of_two.rs` | `LeastLoadPolicy` |
| `RoundRobinPolicy` | `policies/round_robin.rs` | `RoundRobinPolicy` |
| `ConsistentHashingPolicy` | (similar) | (not in Layer 15) |
| Worker health loop | `routers/http/router.rs` | `_health_loop` |
| `/metrics` endpoint | Prometheus integration | (not in Layer 15) |

The key insight: Layer 15's `router.py` is the Python teaching analog of what the vLLM Router implements in Rust for production.
