# Revisiting Session Affinity in Kubernetes

**Source:** https://medium.com/@rajeshlagishetty/session-affinity-in-kubernetes-899e243f1ead
**Author:** Rajesh Lagishetty
**Date:** January 2025
**Level:** L2 — Conceptual + configuration
**Why here:** Explains the three main session affinity approaches in Kubernetes (ClientIP, IPVS Source Hashing, Nginx Cookie) with their limitations. Directly motivates Layer 16's lesson/08 decision guide: why `X-Session-ID` header hashing is more reliable than ClientIP when clients are behind NAT.

---

## Overview

Kubernetes Service load balancing primarily depends on the proxy and its configuration. Currently, IPVS mode supports algorithms like round-robin (rr) and least connections (lc).

Session affinity ensures a client's requests are consistently routed to the same backend pod — important when maintaining state (like a router's radix tree cache).

---

## Method 1: `sessionAffinity: ClientIP`

The simplest approach — configure on the Service spec:

```yaml
kind: Service
apiVersion: v1
metadata:
  name: sglang-router
spec:
  selector:
    app: sglang-router
  ports:
    - name: http
      protocol: TCP
      port: 30000
      targetPort: 30000
  sessionAffinity: ClientIP
  sessionAffinityConfig:
    clientIP:
      timeoutSeconds: 3600   # 1 hour sticky session
```

**How it works:** `kube-proxy` hashes the client's source IP and routes consistently to the same backend pod.

**Limitation:** Fails if source IPs are NATed — clients behind a corporate proxy or VPN all share the same source IP, routing every request to the same pod and creating a hotspot.

---

## Method 2: IPVS Source Hashing (sh)

An alternative to ClientIP affinity. Advantage: no timeout (ClientIP has a configurable timeout after which sessions are reassigned).

Configure in the `kube-proxy` ConfigMap:

```yaml
apiVersion: kubeproxy.config.k8s.io/v1alpha1
kind: KubeProxyConfiguration
mode: ipvs
ipvs:
  scheduler: sh   # Source Hashing
```

**Same limitation as ClientIP:** Fails with NATed source IPs.

---

## Method 3: Nginx Ingress Cookie Affinity

The most reliable approach for user-facing services. Ingress-NGINX creates a cookie with a randomly generated key mapped to a specific upstream pod.

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sglang-router
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/session-cookie-name: "ROUTER_STICKY"
    nginx.ingress.kubernetes.io/affinity-mode: "balanced"
spec:
  ingressClassName: nginx
  rules:
    - host: llm.example.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: sglang-router
                port:
                  number: 30000
```

**How it works:** NGINX creates the `INGRESSCOOKIE` (or custom name) cookie containing a consistent hash → same upstream pod. Subsequent requests include the cookie → same backend.

**Advantage:** Works regardless of client IP; works behind NAT.

**Limitation:** Cookie is client-browser-level — API clients must persist and send the cookie.

---

## Method 4: Header-based hashing (`upstream-hash-by`)

For API clients (not browsers), use `X-Session-ID` or `X-User-ID` header hashing:

```yaml
annotations:
  nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_session_id"
```

Clients set `X-Session-ID: <user-or-session-id>` and NGINX hashes it to a consistent upstream.

**Critical caveat (2025):** `upstream-hash-by` is only deterministic within a single ingress-nginx replica. Multiple ingress controller replicas may have different endpoint orderings, causing the same hash value to map to different backends. For true stickiness: keep ingress-nginx at 1 replica OR use cookie affinity.

---

## Decision guide (Layer 16)

| Scenario | Recommended approach |
|---|---|
| Internal tooling, clients have unique IPs | `sessionAffinity: ClientIP` on K8s Service |
| Users behind corporate NAT / VPN | Nginx cookie affinity or header-based hash |
| API clients that support custom headers | `upstream-hash-by: "$http_x_session_id"` |
| Single ingress-nginx replica | Any of the above |
| Multiple ingress-nginx replicas | Cookie affinity only (header hash is non-deterministic) |

**The critical quote:** "Both approaches of Affinity / Source Hashing will fail if source IPs are NATed."

This is why Layer 16's lesson/08 recommends `X-Session-ID` header hashing as the production default — it works regardless of network topology.
