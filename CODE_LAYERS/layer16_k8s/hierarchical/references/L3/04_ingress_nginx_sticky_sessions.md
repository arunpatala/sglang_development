# Sticky Sessions — Ingress-NGINX Controller

**Source:** https://kubernetes.github.io/ingress-nginx/examples/affinity/cookie/
**Author:** Kubernetes Ingress-NGINX Project
**Level:** L3 — Configuration reference
**Why here:** Official reference for cookie-based session affinity in Ingress-NGINX. Layer 16's lesson/08 uses cookie affinity (`INGRESSCOOKIE`) to stick a user session to a specific router replica. This doc explains all annotation options and the INGRESSCOOKIE mechanism.

---

## Overview

Session affinity ensures a client's requests are consistently routed to the same backend pod. In Ingress-NGINX, this is implemented via cookies.

---

## Annotation Reference

| Annotation | Description | Values |
|---|---|---|
| `nginx.ingress.kubernetes.io/affinity` | Enable affinity | `cookie` (only supported type) |
| `nginx.ingress.kubernetes.io/affinity-mode` | Stickiness behavior | `balanced` (default) or `persistent` |
| `nginx.ingress.kubernetes.io/session-cookie-name` | Cookie name | string (default: `INGRESSCOOKIE`) |
| `nginx.ingress.kubernetes.io/session-cookie-max-age` | Cookie lifetime | seconds |
| `nginx.ingress.kubernetes.io/session-cookie-expires` | Legacy expiry (browser compat) | seconds |
| `nginx.ingress.kubernetes.io/session-cookie-secure` | Set Secure flag | `"true"` or `"false"` |
| `nginx.ingress.kubernetes.io/session-cookie-path` | Cookie path scope | string (default: matched path) |
| `nginx.ingress.kubernetes.io/session-cookie-domain` | Cookie domain | string |
| `nginx.ingress.kubernetes.io/session-cookie-samesite` | SameSite attribute | `None`, `Lax`, `Strict` |
| `nginx.ingress.kubernetes.io/session-cookie-change-on-failure` | New cookie on backend failure | `true` or `false` (default: `false`) |
| `nginx.ingress.kubernetes.io/affinity-canary-behavior` | Canary affinity behavior | `sticky` (default) or `legacy` |

---

## Affinity modes

### `balanced` (default — Layer 16 uses this)
When the backend pod pool grows or shrinks, NGINX redistributes sessions across pods. Some existing sessions will be remapped. Good for router replicas where you want even distribution.

### `persistent`
Maximum stickiness — sessions never move until the cookie expires. Risk: overloaded pods if one user is much heavier than others.

---

## Deployment Example (Layer 16 Router)

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sglang-router
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/affinity: "cookie"
    nginx.ingress.kubernetes.io/affinity-mode: "balanced"
    nginx.ingress.kubernetes.io/session-cookie-name: "SGLANG_ROUTER_SESSION"
    nginx.ingress.kubernetes.io/session-cookie-max-age: "3600"
    nginx.ingress.kubernetes.io/session-cookie-secure: "true"
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

---

## Validation

```bash
kubectl describe ing sglang-router
```

Test cookie is set:
```bash
curl -I https://llm.example.com/v1/models
# Response includes:
# Set-Cookie: SGLANG_ROUTER_SESSION=a9907b79b248140b56bb13723f72b67697baac3d; ...
```

Subsequent requests with the cookie go to the same router replica:
```bash
curl -H "Cookie: SGLANG_ROUTER_SESSION=a9907b79b248140b56bb13723f72b67..." \
  https://llm.example.com/v1/completions
```

---

## How INGRESSCOOKIE works

1. Client sends first request (no cookie).
2. NGINX selects an upstream using consistent hashing.
3. NGINX creates `INGRESSCOOKIE=<hash>` where the hash maps to the selected upstream pod.
4. Cookie is set in the response `Set-Cookie` header.
5. Client includes cookie in subsequent requests.
6. NGINX looks up the hash → routes to same upstream.

**If the upstream pod is removed:** The consistent hash changes → NGINX selects a new upstream and sets a new cookie. No stale routing.

**If the backend pool grows:** `balanced` mode redistributes some sessions to new pods. `persistent` mode keeps all sessions on original pods (overload risk).

---

## Caveats

**1. Service pointing to multiple Ingresses:**
When a Service points to more than one Ingress, with only one containing affinity configuration, the first-created Ingress is used. May cause affinity to not work if the non-affinity Ingress was created first.

**2. Multiple ingress-nginx replicas:**
Cookie affinity works correctly regardless of how many ingress-nginx replicas are running — the cookie value encodes the upstream directly (consistent hash), not a replica-local table. This makes cookie affinity superior to `upstream-hash-by` for HA ingress setups.

**3. API clients must send cookies:**
Browser clients handle cookies automatically. API clients (curl, Python httpx/requests) must be configured to persist and send cookies. Use `requests.Session()` in Python or `--cookie-jar` in curl.

---

## `upstream-hash-by` (header-based hashing)

An alternative to cookie affinity — hash a request header instead of using a cookie:

```yaml
annotations:
  nginx.ingress.kubernetes.io/upstream-hash-by: "$http_x_session_id"
```

Clients send `X-Session-ID: <user-id>` and NGINX hashes it to a consistent upstream.

**Critical limitation:** Only deterministic within a single ingress-nginx replica. Multiple ingress-nginx replicas may have different endpoint orderings — the same hash value can map to different backends across replicas. For true cross-replica stickiness, use cookie affinity.

---

## Mapping to Layer 16

| Concept | Layer 16 location |
|---|---|
| Cookie-based router stickiness | `lesson/08_high_availability.md` §"Mitigation option 2" |
| `affinity-mode: balanced` | Preferred for router replicas to allow rebalancing |
| `INGRESSCOOKIE` mechanism | Explained in lesson/08 as "consistent hash → same replica" |
| `upstream-hash-by` caveats | lesson/08 §"What to avoid with multiple ingress replicas" |
