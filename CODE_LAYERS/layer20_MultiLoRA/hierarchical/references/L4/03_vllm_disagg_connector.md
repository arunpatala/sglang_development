# vLLM Disaggregated Prefilling: Connector Architecture

**Source:** https://docs.vllm.ai/en/latest/features/disagg_prefill.html
**GitHub:** https://github.com/vllm-project/vllm/tree/main/vllm/distributed/kv_transfer
**Example scripts:** https://github.com/vllm-project/vllm/tree/main/examples/online_serving
**Level:** L4 — vLLM's disaggregation connector framework; alternative implementation reference
**Why here:** vLLM's connector architecture provides the clearest open-source specification of what a KV transfer abstraction layer needs to support. Its 6-connector design (NIXL, Mooncake, NCCL, LMCache, Example, Multi) maps exactly to the different transfer protocols and storage backends covered in Layer 19. For engineers extending SGLang's disaggregation support or evaluating alternative connectors, vLLM's implementation is the most detailed public reference.

---

## Why vLLM's Implementation Matters

vLLM's disaggregated prefilling was the first open-source implementation of inter-instance KV transfer (building on Splitwise's prototype PR #2809). Today:
- It supports 6 different connector types covering every major KV transfer protocol.
- Its Connector/LookupBuffer abstraction is the reference design for how any inference framework should expose KV transfer.
- SGLang's PD disaggregation follows the same conceptual architecture (prefill node = KV producer, decode node = KV consumer) but with tighter integration into SGLang's scheduler.

---

## Why Disaggregated Prefilling (vLLM's Own Answer)

From the vLLM docs:

> Two main reasons: (1) Tuning TTFT and ITL separately — disaggregated prefilling puts prefill and decode phases inside different vLLM instances, giving flexibility to assign different resources and optimization strategies. (2) Satisfying different SLO requirements for different applications.

> **Important**: Disaggregated prefill DOES NOT improve throughput. It improves latency SLO compliance and decouples TTFT from ITL.

This note is critical: disaggregation is a **latency architecture choice**, not a throughput optimisation. Total tokens processed per GPU-hour is roughly the same; what changes is which requests meet their SLO constraints.

---

## The 6 Supported Connectors

### 1. NixlConnector

```bash
--kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_producer","kv_parallel_size":2}'
```

Uses NIXL (NVIDIA Inference Xfer Library) with UCX backend. Supports RDMA InfiniBand/RoCEv2. The default high-performance connector for GPU-to-GPU KV transfer in production clusters.

Backend selection: `--kv-transfer-config '{"kv_connector":"NixlConnector","NIXL_Backend":"LIBFABRIC"}'`

### 2. MooncakeConnector

```bash
--kv-transfer-config '{"kv_connector":"MooncakeConnector","kv_role":"kv_producer"}'
```

Uses Mooncake Transfer Engine. Supports RDMA, NVLink, and TCP. Identical in capability to NixlConnector but uses Mooncake's multi-NIC pooling and topology-aware path selection.

### 3. P2pNcclConnector

```bash
--kv-transfer-config '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer",
  "kv_rank":0,"kv_parallel_size":2,"kv_buffer_size":"1e9","kv_port":"14579",
  "kv_connector_extra_config":{"proxy_ip":"...","proxy_port":"30001","send_type":"PUT_ASYNC"}}'
```

Uses NCCL for direct P2P GPU transfers. Simpler setup (no RDMA NIC required), but lower bandwidth than RDMA. Good for clusters without InfiniBand where bandwidth is limited to PCIe.

### 4. LMCacheConnectorV1

```bash
--kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

LMCache-based connector with support for multiple storage backends including NIXL as the underlying KV transmission. Enables both cross-engine KV sharing and disaggregated prefilling in a unified system.

### 5. ExampleConnector

Reference implementation for writing custom connectors. Located in `examples/offline_inference/disaggregated-prefill-v1/`. Used for testing and as a starting template.

### 6. MultiConnector

```bash
--kv-transfer-config '{"kv_connector":"MultiConnector","kv_role":"kv_both",
  "kv_connector_extra_config":{"connectors":[
    {"kv_connector":"NixlConnector",...},
    {"kv_connector":"LMCacheConnectorV1",...}
  ]}}'
```

Chains multiple connectors in priority order. Allows fallback from RDMA to NCCL if RDMA is unavailable, or combining NIXL transfer with LMCache persistent storage.

---

## The Connector/LookupBuffer Abstraction

All connectors implement the same abstract interface in `vllm/distributed/kv_transfer/kv_connector/base.py`:

### Connector

```python
class BaseKVConnector(ABC):
    def send_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches, hidden_or_intermediate_states
    ) -> None: ...

    def recv_kv_caches_and_hidden_states(
        self, model_executable, model_input, kv_caches
    ) -> Tuple[torch.Tensor, bool]: ...

    def close(self) -> None: ...
```

**`send_kv_caches`**: called on the prefill instance after the forward pass completes. Writes KV tensors to the transfer buffer for the decode instance to receive.

**`recv_kv_caches`**: called on the decode instance before the forward pass. Receives KV tensors from the prefill instance. Returns `(hidden_states, bypass_model_exec)` — if `bypass_model_exec=True`, the decode instance skips its own prefill computation (it already has the KV from the transfer).

### LookupBuffer

```python
class KVLookupBufferBase(ABC):
    def insert(self, input_tokens, roi, key, value, hidden) -> None: ...
    def drop_select(self, input_tokens, roi) -> Tuple[...]: ...
    def close(self) -> None: ...
```

`insert`: prefill instance inserts KV cache into the buffer for the decode instance to retrieve.

`drop_select`: decode instance atomically selects and removes the KV cache matching its request's token sequence. "Drop" ensures each KV cache is consumed exactly once (the decode instance that retrieves it owns it).

---

## The `--kv-transfer-config` JSON

Full configuration schema:

```python
class KVTransferConfig(BaseModel):
    kv_connector: str                  # Connector class name
    kv_role: str                       # "kv_producer", "kv_consumer", or "kv_both"
    kv_rank: int = 0                   # Rank within the transfer group
    kv_parallel_size: int = 2          # Total size of transfer group (producer + consumer)
    kv_buffer_size: float = 1e9       # Transfer buffer size in bytes (1 GB default)
    kv_port: str = "14579"             # Port for connector communication
    kv_connector_extra_config: dict    # Connector-specific configuration
```

**`kv_role`**: the most important field.
- `kv_producer`: this instance is the prefill worker; it sends KV to consumers.
- `kv_consumer`: this instance is the decode worker; it receives KV from producers.
- `kv_both`: this instance runs both phases (used for P2P or multi-turn scenarios).

---

## Implementation Directory Structure

```
vllm/distributed/kv_transfer/
├── kv_connector/
│   ├── base.py                   # BaseKVConnector abstract class
│   ├── factory.py               # ConnectorFactory — maps name to class
│   ├── simple_connector.py      # Wraps LookupBuffer into Connector interface
│   └── worker_connector.py      # Per-worker connector state
├── kv_lookup_buffer/
│   ├── base.py                  # KVLookupBufferBase abstract class
│   └── simple_buffer.py        # In-memory dictionary implementation
├── kv_transfer_agent.py         # Manages connector lifecycle per process
└── kv_transfer_config.py        # KVTransferConfig dataclass
```

Each connector implementation lives in `vllm/distributed/kv_transfer/<connector_name>/`:
- `nixl/` — NixlConnector
- `mooncake/` — MooncakeConnector
- `p2p/` — P2pNcclConnector
- `lmcache_integration/` — LMCacheConnectorV1

---

## Comparison: vLLM vs SGLang Disaggregation Architecture

| Aspect | vLLM | SGLang |
|---|---|---|
| Transfer abstraction | `BaseKVConnector` (6 implementations) | `DisaggTransferBackend` (Mooncake, NIXL) |
| Configuration | JSON via `--kv-transfer-config` | CLI flags `--disaggregation-mode`, `--disaggregation-ib-device` |
| Router | External (Ray Serve, Dynamo, custom) | Built-in `sglang_router` |
| MoE support | General MoE | DeepEP + EPLB integration |
| Cache-aware routing | Not built in | Round-robin (prefix-aware planned) |
| Multi-connector chaining | MultiConnector | Not yet supported |

---

## Key Takeaways for Layer 19

- vLLM's connector architecture is the **reference spec** for what a KV transfer abstraction must provide: send/recv semantics, async transfer, and atomic lookup-buffer semantics.
- The **`drop_select` pattern** is the correct way to handle ownership: each KV cache is retrieved by exactly one decode instance, preventing double consumption.
- `kv_role="kv_both"` enables a single instance to run both phases — useful for testing or for multi-turn scenarios where the previous decode state is reused as the next prefill input.
- The **6 connectors** cover all major scenarios: RDMA (NIXL, Mooncake), NCCL (P2P without RDMA), LMCache (with persistent storage), multi-backend fallback (MultiConnector).
- **Disaggregated prefill does NOT improve throughput** — vLLM's documentation makes this explicit. The gain is latency SLO compliance under mixed workloads.
