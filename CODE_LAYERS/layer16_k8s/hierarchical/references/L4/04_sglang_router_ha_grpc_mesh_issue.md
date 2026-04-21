# SGLang Router HA: gRPC Mesh Design (Issue #10839)

**Source:** https://github.com/sgl-project/sglang/issues/10839
**Related PR:** https://github.com/sgl-project/sglang/pull/14108
**Related issue:** https://github.com/sgl-project/sglang/issues/10341 (Router Roadmap)
**Author:** SGLang Project maintainers
**Date:** September 2025 (closed November 2025, implementation in PR #14108)
**Level:** L4–L5 — Internal design document
**Why here:** This is the design document for the router's HA state layer — why multiple router replicas degrade cache performance (as described in Layer 16's lesson/08), and what the fix is. Explains the CRDTs, gRPC mesh, and eventual consistency approach.

---

## The Problem: Multi-Replica Cache Degradation

When the SGLang router is deployed with N replicas for HA, each replica:
1. Runs its own watcher loop (service discovery works fine — pods are watched independently)
2. Maintains its own local **radix tree** (prefix cache map)
3. Has no awareness of what other replicas have cached

**Consequence:** A user's second request may land on a different router replica (if ingress load-balances without session affinity). The second replica has an empty cache for that user's prefix, causing a cache miss even though another replica (and a worker!) already has the data.

Issue #18058 (February 2026) explicitly confirms:
> "Is multi-router replica data synchronization available already? **No.**"

---

## The Solution: gRPC Bidirectional Mesh

Design goal: **no external datastore** (no Redis, no etcd). All sync happens in-process via a gRPC mesh.

### Architecture

Each router replica:
1. Listens on `--router-mesh-port` (default: configurable, e.g., 9100) for gRPC connections from peers
2. At startup, connects to all known peers (`--ha-server-peers "router-1:9100,router-2:9100"`)
3. Streams state updates via bidirectional gRPC to all peers
4. Applies received updates locally using CRDT merge semantics

### State Being Synchronized

| State type | CRDT type | Sync priority |
|---|---|---|
| Worker registry (pod set) | OR-Set (add-wins) | Critical — must converge quickly |
| Per-model radix trees | LWW-Register per prefix | Eventually consistent |
| Rate-limit buckets (token buckets) | PN-Counter or gossip | Eventually consistent |
| Router membership (who is alive) | SWIM gossip | For liveness detection |

**OR-Set (Observed-Remove Set):** Worker additions are always applied; removals require seeing the specific add token. Prevents split-brain where one replica removes a worker the other just added.

**LWW-Register (Last-Write-Wins Register):** For radix tree nodes, the most-recently-observed state wins. Slightly lossy but efficient; a cache entry may briefly appear "missing" during partition, then reappear after sync.

---

## Implementation (PR #14108)

### Rust dependency
```toml
[dependencies]
rust-crdt = "7.3"           # CRDT primitives (OR-Set, LWW, etc.)
tonic = "0.11"              # gRPC for Rust
```

### Key components

```rust
// Router mesh configuration
pub struct MeshConfig {
    pub mesh_port: u16,              // Port for gRPC peer connections
    pub peers: Vec<String>,          // peer1:9100,peer2:9100
    pub sync_interval: Duration,     // How often to push state to peers
    pub max_sync_batch: usize,       // Max entries per sync message
}

// State sync message
pub struct StateSyncMessage {
    pub worker_registry_delta: OrSetDelta<WorkerId>,
    pub radix_tree_deltas: Vec<RadixTreeDelta>,
    pub rate_limit_deltas: Vec<RateBucketDelta>,
    pub sender_id: RouterId,
    pub timestamp: Timestamp,
}
```

### Consistency model
- **Worker registry**: eventual consistency, target convergence < 5s
- **Radix trees**: eventual consistency; acceptable to have stale data for 30s
- **Rate limits**: approximate — within 10% of true rate, not exact enforcement
- **No strong consensus**: no leader election, no Raft, no quorum writes

**Trade-off:** During a network partition, each isolated router replica continues serving with its local state. After the partition heals, states merge. A rate limit might temporarily allow ~10% more traffic than configured. Cache hit rates drop during partition.

---

## SGLang Router Roadmap (Issue #10341)

The broader roadmap for the SGLang router (Sept 2025):

### Completed (as of Q4 2025)
- [x] Service discovery (K8s watcher loop)
- [x] New worker management API (`/workers/{id}`)
- [x] Policy registry per model family
- [x] Data-parallel aware routing
- [x] Multi-model support (`--model-registry`)
- [x] gRPC mesh HA (PR #14108)

### In Progress / Pending
- [ ] Full radix-tree sync across replicas (partially done in PR #14108; not production-stable as of Feb 2026)
- [ ] Semantic model selection (route by model capability, not just name)
- [ ] Data mesh component (federated KV cache across clusters)
- [ ] KV cache sharing between workers (beyond radix routing)

---

## Practical guidance for Layer 16

### What works today (early 2026)
- **N router replicas, session affinity** → each user sticks to one replica → no cache degradation → use ingress cookie or `X-Session-ID` hashing
- **N router replicas, no session affinity** → 1/N cache miss rate per cross-replica request → acceptable for low N (2–3 replicas)
- **1 router replica, K8s Deployment** → zero cache degradation; single point of failure if pod crashes (mitigated by `minReadySeconds` and `rollout strategy`)

### What's coming (post-v2 SGLang)
- **gRPC mesh with OR-Set worker registry** → all replicas converge on same worker list; cache routing still per-replica until radix tree sync is stable
- **Full radix tree sync** → true HA without session affinity; estimate mid-2026

### Recommendation
For lesson/08's "HA design decision":
- **Simple HA (recommended):** 2 replicas + ingress cookie affinity → effectively independent routers, each fully warm for their user segment
- **Scale-out HA (future):** 5+ replicas + gRPC mesh sync → not yet production-stable; follow issue #10839 for updates
