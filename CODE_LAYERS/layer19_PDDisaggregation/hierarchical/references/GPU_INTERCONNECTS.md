# GPU Interconnects: Intra-Node, Inter-Node, and Cluster Level

**Purpose:** Reference document covering every GPU interconnect technology relevant to LLM serving and PD disaggregation — generations, speeds, and AWS instance mapping.

**Context:** KV cache transfer speed during PD disaggregation is entirely determined by which interconnect sits between the prefill and decode workers. This document is the reference for §9 (KV Transfer Tax) and §10–12 (Mooncake, Dynamo, NIXL) in `COMBINEDL3L4.md`.

---

## The Three Scopes

```
┌──────────────────────────────────────────────────────────────────┐
│  CLUSTER LEVEL (Scale-Out)                                       │
│  Rack ↔ Rack via InfiniBand / RoCEv2 / Ethernet                  │
│  100 Gbps → 3,200 Gbps per node                                  │
│                                                                  │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  INTER-NODE LEVEL (Scale-Out)                             │  │
│  │  Node ↔ Node via EFA/NIC                                  │  │
│  │  25 GB/s (HDR) → 400 GB/s (p5 8×NIC)                    │  │
│  │                                                           │  │
│  │  ┌──────────────────────────────────────────────────────┐ │  │
│  │  │  INTRA-NODE LEVEL (Scale-Up)                        │ │  │
│  │  │  GPU ↔ GPU via NVLink + NVSwitch                    │ │  │
│  │  │  160 GB/s (P100) → 1,800 GB/s (B200) per GPU       │ │  │
│  │  │  GPU ↔ CPU via PCIe                                 │ │  │
│  │  │  32 GB/s (Gen3) → 242 GB/s (Gen6) per slot         │ │  │
│  │  └──────────────────────────────────────────────────────┘ │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

**The bandwidth gap across scopes (2025 production numbers):**

| Scope | Technology | Typical BW |
|---|---|---|
| Intra-node GPU↔GPU | NVLink 4 (H100) | 900 GB/s |
| Intra-rack GPU↔GPU | NVLink 5 Switch (NVL72) | ~1,800 GB/s |
| Inter-node GPU↔GPU | InfiniBand NDR + GPUDirect | 50 GB/s |
| Inter-node (plain) | 100 GbE TCP | ~6 GB/s |

NVLink is **18× faster** than NDR InfiniBand. NDR InfiniBand is **8× faster** than 100 GbE TCP. This cascade explains why disaggregation topology planning matters so much.

---

## Level 1 — Intra-Node

### NVLink — GPU to GPU Direct

NVLink is NVIDIA's proprietary point-to-point wire between GPUs inside a server or rack. It bypasses PCIe and the CPU entirely. Born from the observation that PCIe was a hard ceiling for multi-GPU workloads.

**All NVLink generations:**

| Gen | GPU Architecture | Year | Links / GPU | BW per GPU (bidir) | Latency | Notes |
|---|---|---|---|---|---|---|
| 1.0 | Pascal (P100) | 2016 | 4 | **160 GB/s** | ~100–300 ns | First gen; 5× PCIe Gen3 |
| 2.0 | Volta (V100) | 2018 | 6 | **300 GB/s** | ~100–300 ns | CPU–GPU and GPU–GPU; used in Summit supercomputer |
| 3.0 | Ampere (A100) | 2020 | 12 | **600 GB/s** | ~100–300 ns | 50 GT/s per lane; 2× V100 |
| 4.0 | Hopper (H100) | 2022 | 18 | **900 GB/s** | ~100–300 ns | PAM4 signaling; current production standard |
| 5.0 | Blackwell (B200/GB200) | 2024–25 | 18 | **1,800 GB/s** | ~100–300 ns | 100 GT/s per lane; 14× PCIe Gen5 |
| 6.0 | Rubin | 2026 | 36 | **3,600 GB/s** | ~100–300 ns | 2× NVLink 5; 14× PCIe Gen6 |

**Key ratio:** NVLink 4.0 (H100) is 14× faster than PCIe Gen5. NVLink 5.0 doubles again.

**Signaling progression:**
- NVLink 1/2/3: NRZ encoding
- NVLink 4/5: PAM4 (more bits per signal cycle — same reason InfiniBand HDR→NDR doubled bandwidth)

---

### NVSwitch — From Point-to-Point to Full Mesh

A single NVLink is point-to-point (one GPU pair). **NVSwitch** is the silicon hub that creates a **full non-blocking all-to-all fabric** inside the node — every GPU talks to every other at full speed simultaneously, with no sharing or collision.

```
Without NVSwitch (ring / tree):    GPU0 → GPU1 → GPU2 → GPU3
  Bandwidth: shared, latency: O(n hops)

With NVSwitch (full mesh):         GPU0 ═╗
                                   GPU1 ═╬═ NVSwitch ═╦═ GPU4
                                   GPU2 ═╣            ╠═ GPU5
                                   GPU3 ═╝            ╠═ GPU6
                                                       ╚═ GPU7
  Bandwidth: full per pair, latency: 1 hop
```

**NVSwitch generations:**

| Gen | Era | Max GPUs in Domain | Aggregate BW | Notes |
|---|---|---|---|---|
| Gen1 | V100 / HGX-2 | 8 | ~2.4 TB/s total | First NVSwitch chip |
| Gen2 | A100 / DGX A100 | 8 | **4.8 TB/s** | 600 GB/s per GPU |
| Gen3 | H100 / DGX H100 | 8 | **7.2 TB/s** | 900 GB/s per GPU |
| Gen4 (NVL72) | GB200 / NVL72 rack | **72** | **130 TB/s** | Entire rack = 1 logical GPU |
| Gen5 (NVL72) | GB300 / Rubin NVL72 | **72** | **260 TB/s** | 3,600 GB/s per GPU |

**The NVL72 leap:** instead of 8 GPUs in one NVLink domain, the entire rack of **72 GPUs** behaves as a single logical GPU. KV cache transfer within an NVL72 rack is effectively free — 1–2 ms for a 1.34 GB KV cache vs 27 ms over InfiniBand NDR.

---

### PCIe — GPU to CPU

PCIe connects GPUs to the host CPU and DRAM. Relevant for KV transfer in two cases:
1. **No GPUDirect**: data must cross PCIe twice (GPU → CPU RAM → NIC, and reverse on the other side). Halves effective bandwidth.
2. **CPU DRAM for KV offload**: PCIe bandwidth limits how fast KV caches can be evicted to or recalled from CPU DRAM (HiCache / Mooncake KV Store).

**All PCIe generations:**

| Gen | Year | BW (x16, bidir) | Notable GPU era |
|---|---|---|---|
| PCIe 3.0 | 2010 | **32 GB/s** | Pascal, Volta, Turing |
| PCIe 4.0 | 2017 | **64 GB/s** | Ampere (A100) |
| PCIe 5.0 | 2021 | **128 GB/s** | Hopper (H100) |
| PCIe 6.0 | 2023 | **242 GB/s** | Blackwell (B200) |

**CXL (Compute Express Link):** an emerging standard on top of PCIe 5/6 for cache-coherent CPU–GPU and memory expansion. Allows GPU direct access to remote CPU DRAM as if it were local memory. Not yet in production AI clusters but relevant for Mooncake's KV cache pool design (CXL-attached DRAM as a KV tier between HBM and NVMe).

---

## Level 2 — Inter-Node

When prefill and decode workers are on **different servers**, KV cache crosses the network. This is the primary bottleneck for most production PD disaggregation deployments today.

### InfiniBand — The HPC Standard

Purpose-built for high-performance computing. Native RDMA integrated into the protocol stack — no TCP/IP overhead. Credit-based lossless flow control (zero packet drops). Proprietary (NVIDIA/Mellanox hardware: ConnectX NICs, Quantum/Spectrum switches).

**All InfiniBand generations (4x link, the standard):**

| Gen | Acronym | Year | BW / port (4 lanes) | BW in GB/s | Latency | Connector |
|---|---|---|---|---|---|---|
| Single Data Rate | SDR | 2001 | 10 Gbps | 1.25 GB/s | — | Legacy |
| Double Data Rate | DDR | 2005 | 20 Gbps | 2.5 GB/s | — | Legacy |
| Quad Data Rate | QDR | 2007 | 40 Gbps | 5 GB/s | — | QSFP |
| Fourteen Data Rate | FDR | 2011 | 56 Gbps | 7 GB/s | ~1.0 µs | QSFP+ |
| Enhanced Data Rate | EDR | 2015 | 100 Gbps | **12.5 GB/s** | ~0.8 µs | QSFP28 |
| High Data Rate | HDR | 2017 | 200 Gbps | **25 GB/s** | ~0.6 µs | QSFP56 |
| Next Data Rate | NDR | 2021 | 400 Gbps | **50 GB/s** | ~0.5 µs | QSFP112 / OSFP |
| eXtreme Data Rate | XDR | 2024–25 | 800 Gbps | **100 GB/s** | <0.5 µs | OSFP |
| GDR (planned) | GDR | ~2027 | 1.6 Tbps | 200 GB/s | — | — |
| LDR (planned) | LDR | ~2030 | 3.2 Tbps | 400 GB/s | — | — |

**Reading the table:**
- "4x" = 4 lanes bundled. HDR = 4 × 50 Gbps = 200 Gbps. NDR = 4 × 100 Gbps = 400 Gbps.
- Multi-port NICs (2×200G, 2×400G) double per-server bandwidth.
- The signaling jump HDR→NDR used PAM4 (same as NVLink 4.0): doubles bits per cycle without changing the wire count.

**Current production:** NDR (H100/Blackwell-era clusters). p5.48xlarge uses 8× NDR NICs = 3,200 Gbps total per instance. p4de uses 1× HDR NIC = 400 Gbps.

**Application-layer latency:** InfiniBand 1–2 µs vs RoCEv2 5–10 µs vs TCP 50 µs. For KV transfer (large payloads, not small messages), latency matters less than bandwidth — but P99 latency spikes matter for TTFT tail.

---

### RoCEv2 — RDMA over Ethernet

The same RDMA verb interface as InfiniBand, carried over standard Ethernet instead of proprietary IB fabric. Requires Priority Flow Control (PFC) or DCQCN congestion control to prevent packet drops (RDMA retransmits the entire message on a drop).

**RoCEv2 vs InfiniBand:**

| Metric | InfiniBand NDR | RoCEv2 (400G Ethernet) |
|---|---|---|
| Bandwidth / port | 400 Gbps | 400 Gbps |
| Latency (application) | **1–2 µs** | 5–10 µs |
| Tail latency (P99) | Very stable (credit-based) | Variable under congestion |
| Packet loss | Zero (hardware credit-based) | Requires PFC tuning |
| Hardware cost | Higher (proprietary Quantum switches) | Lower (commodity Ethernet) |
| Ecosystem | NVIDIA/Mellanox only | Broadcom, Marvell, Intel, NVIDIA |
| AWS equivalent | InfiniBand via EFA | EFA (uses RoCEv2 internally) |
| Mooncake/NIXL support | Yes (UCX backend) | Yes (UCX backend) |

**AWS EFA (Elastic Fabric Adapter):** AWS's implementation of RDMA over its own low-latency fabric. Exposes RDMA semantics to applications (via libfabric) but the underlying transport is proprietary. Supports GPUDirect RDMA. Used on p4de (400 Gbps EFA) and p5 (3,200 Gbps EFA).

---

### Ethernet — The Fallback

Standard TCP/IP Ethernet. High CPU overhead (kernel stack involvement), high latency (50 µs), no GPUDirect — all KV data must bounce through CPU RAM on both ends.

| Speed | BW | Transfer (1.34 GB KV cache) | Verdict |
|---|---|---|---|
| 1 GbE | 125 MB/s | 10.7 s | Unusable |
| 10 GbE | 1.25 GB/s | 1.07 s | Unusable |
| 25 GbE | 3.1 GB/s | 430 ms | Too slow |
| 100 GbE (TCP) | ~6 GB/s effective | **~220 ms** | Borderline (long context only) |
| 400 GbE (TCP) | ~25 GB/s effective | **~54 ms** | Acceptable with GPUDirect |
| 400 GbE (RoCEv2) | ~40 GB/s | **~34 ms** | Good |

GPUDirect on 100 GbE RoCEv2 gets you to ~12.5 GB/s (wire rate), cutting the 220 ms to ~107 ms. The CPU bypass is the primary reason Mooncake and NIXL exist — their first job is enabling GPUDirect to eliminate the PCIe double-crossing.

---

## Level 3 — Cluster Scale

At cluster scale (multiple racks), the same NICs connect through a multi-tier switch fabric.

### Rail-Optimized Topology (NVIDIA SuperPOD standard)

Each GPU gets its own dedicated NIC on its own "rail." GPU 0 on every server connects to Rail 0's leaf switch; GPU 1 to Rail 1; and so on. Traffic within a rail never crosses the spine.

```
DGX Node:  GPU0  GPU1  GPU2  GPU3  GPU4  GPU5  GPU6  GPU7
              │     │     │     │     │     │     │     │
          Rail-0 Rail-1 Rail-2 Rail-3 Rail-4 Rail-5 Rail-6 Rail-7
              │     │     │     │     │     │     │     │
           Leaf switches (one per rail, up to 32 nodes per rail)
                           │
                       Spine switches
```

Result: near-linear scaling to 256+ GPUs. A 256-GPU InfiniBand NDR cluster achieves ~92% of theoretical bandwidth for all-reduce — 370 GB/s effective on a 400 Gbps fabric.

**KV-aware routing in Dynamo** (§11 of COMBINEDL3L4): the router tracks which prefill workers have which token prefixes cached. Requests are sent to workers with the highest prefix overlap, keeping traffic on the same rail where possible — reducing spine hops and improving effective KV transfer bandwidth.

### NVLink Switch System — Rack-to-Rack NVLink

For NVL72 deployments, NVIDIA extends NVLink beyond a single rack using **NVLink External Link Modules (ELMs)** — copper or optical cables carrying NVLink signals between adjacent racks. Multiple NVL72 racks can form a single NVLink domain of up to **576 GPUs** at near-intra-rack speeds.

| Domain | GPUs | Aggregate BW |
|---|---|---|
| Single NVL72 rack | 72 | 130 TB/s |
| 8× NVL72 racks (NVLink domain) | 576 | ~1 PB/s |

This is why NVIDIA Dynamo's benchmark shows **30× throughput improvement** over collocated H100 — the KV transfer between disaggregated prefill and decode workers, when both are in NVL72 racks, is NVLink-speed rather than InfiniBand-speed.

---

## KV Transfer Times (LLaMA-3.1-70B, 4,096-token prompt = 1.34 GB KV cache)

This is the number that determines whether PD disaggregation is net-positive within a given TTFT budget.

| Technology | Scope | Bandwidth | Transfer Time | PD Viable? |
|---|---|---|---|---|
| NVLink 5 (NVL72, same rack) | Intra-rack | ~1,000 GB/s | **~1 ms** | Yes — nearly free |
| NVLink 4 (same 8-GPU server) | Intra-node | 900 GB/s | **~1.5 ms** | Yes — free |
| NVLink Switch (rack-to-rack) | Intra-cluster | ~200 GB/s | **~7 ms** | Yes |
| InfiniBand XDR (GPUDirect) | Inter-node | 100 GB/s | **~13 ms** | Yes |
| InfiniBand NDR (GPUDirect) | Inter-node | 50 GB/s | **~27 ms** | Yes |
| InfiniBand HDR (GPUDirect) | Inter-node | 25 GB/s | **~54 ms** | Yes |
| RoCEv2 400G (GPUDirect) | Inter-node | ~40 GB/s | **~34 ms** | Yes |
| 100 GbE TCP (CPU-mediated) | Inter-node | ~6 GB/s | **~220 ms** | Borderline |
| 10 GbE TCP | Inter-node | ~1 GB/s | ~1,340 ms | No |

**TTFT budget check:** for a 500 ms TTFT SLO with 200 ms prefill time, you have 300 ms for transfer. InfiniBand HDR (54 ms) uses 18% of the budget — comfortable. 100 GbE TCP (220 ms) uses 73% — barely viable and leaves no headroom. 10 GbE is not viable at any budget.

**Splitwise amortisation threshold:** disaggregation is net-positive only when transfer cost < interference-elimination gain. For prompts > ~500 tokens generating > ~50 output tokens, InfiniBand NDR is always net-positive.

---

## AWS GPU Instance Comparison

Four instance families covering three GPU generations and two NVLink generations.

| | **g5.12xlarge** | **p4d.24xlarge** | **p4de.24xlarge** | **p5.48xlarge** |
|---|---|---|---|---|
| **GPU** | 4× A10G | 8× A100 40 GB | 8× A100 **80 GB** | 8× H100 80 GB |
| **GPU memory total** | 96 GB | 320 GB HBM2 | **640 GB HBM2e** | 640 GB HBM3 |
| **GPU memory BW** | 4× 600 GB/s GDDR6 | 8× 1.6 TB/s | 8× **2.0 TB/s** | 8× **3.35 TB/s** |
| **GPU compute (FP16)** | 4× 31 TFLOPS | 8× 77 TFLOPS | 8× 77 TFLOPS | 8× 989 TFLOPS (FP8) |
| **CPU** | AMD EPYC 7R32 | Intel Xeon 8275CL | Intel Xeon 8275CL | AMD EPYC 9454P |
| **vCPUs** | 48 | 96 | 96 | 192 |
| **RAM** | 192 GB | 1,152 GB | 1,152 GB | 2,048 GB |
| **GPU–GPU (intra-node)** | **PCIe only (no NVLink)** | NVSwitch @ **600 GB/s** | NVSwitch @ **600 GB/s** | NVSwitch @ **900 GB/s** |
| **Inter-node network** | 40 Gbps ENA | 400 Gbps EFA | 400 Gbps EFA | **3,200 Gbps EFA** (8× NICs) |
| **GPUDirect RDMA** | **No** | Yes | Yes | Yes |
| **NICs per instance** | 1 | 1 | 1 | **8** |
| **Local NVMe** | 3.8 TB | 8 TB (8× 1 TB) | 8 TB | 30.7 TB (8× 3.84 TB) |
| **On-demand price** | ~$5.67/hr | ~$32.77/hr | ~$32.77/hr | ~$98/hr |
| **Price / GPU** | ~$1.42/hr | ~$4.10/hr | ~$4.10/hr | ~$12.25/hr |

### Intra-Node Topology

**g5.12xlarge — No NVLink:**
```
GPU0  GPU1  GPU2  GPU3
  └─────┴──PCIe──┴─────┘
            CPU
   (shared bus, ~64 GB/s total, through CPU)
```
All GPU-to-GPU communication must route through the CPU. Effective inter-GPU bandwidth: ~32–64 GB/s shared across all pairs.

**p4de.24xlarge / p4d.24xlarge — NVSwitch (A100):**
```
GPU0 ═══╗
GPU1 ═══╬══ NVSwitch ══╦═ GPU4
GPU2 ═══╣              ╠═ GPU5
GPU3 ═══╝              ╠═ GPU6
                        ╚═ GPU7
  Each pair: 600 GB/s bidirectional, simultaneously, no CPU involved
```

**p5.48xlarge — NVSwitch (H100):**
Same full-mesh topology as p4de, but NVLink 4.0 gives **900 GB/s** per GPU pair. Each of the 8 GPUs also has its **own dedicated NIC** (one 400G NIC per GPU = 8× 400G = 3,200 Gbps total).

### Inter-Node Capability

| Instance | KV Transfer (1.34 GB) | PD Disaggregation |
|---|---|---|
| g5.12xlarge | ~268 ms (TCP, CPU-mediated) | Not suitable |
| p4de.24xlarge | **~27 ms** (HDR EFA, GPUDirect) | Good |
| p5.48xlarge | **~3 ms** (NDR EFA ×8, GPUDirect) | Excellent |

The p5's 8 dedicated NICs mean each GPU's KV transfer uses its own NIC at full bandwidth — no NIC sharing. This is the rail-optimized topology implemented at the instance level.

### HBM Comparison: Why It Matters for Decode

Decode is **memory-bandwidth-bound**. ITL ∝ 1/HBM bandwidth.

| GPU | Memory Type | HBM BW | Decode ITL (relative) |
|---|---|---|---|
| A10G (g5) | GDDR6 | 600 GB/s | 1× (baseline) |
| A100 (p4d) | HBM2 | 1,600 GB/s | **~2.7× faster** |
| A100 80GB (p4de) | HBM2e | 2,000 GB/s | **~3.3× faster** |
| H100 (p5) | HBM3 | 3,350 GB/s | **~5.6× faster** |

A10G uses **GDDR6** (not HBM) — the same memory technology as consumer gaming GPUs. Fine for small-model inference, but decode throughput is limited to ~600 GB/s, vs 3.35 TB/s on H100. For workloads where TPOT (ITL) is the SLO, H100 is ~5× better at decode throughput.

### PD Disaggregation Role by Instance

| Instance | Role | Why |
|---|---|---|
| **g5.12xlarge** | Not suitable for PD disagg at scale | No GPUDirect, no NVLink, 40 Gbps ENA is too slow |
| **p4de.24xlarge** | Decode pool (cost-optimised) | High HBM2e bandwidth at lower cost/GPU; 640 GB fits many concurrent KV caches |
| **p4de.24xlarge** | Prefill pool (budget) | A100 compute is strong; EFA GPUDirect transfers at HDR speed |
| **p5.48xlarge** | Prefill pool (primary) | Highest FP8/FP16 FLOPS; 3,200 Gbps EFA for fast KV handoff |
| **p5.48xlarge** | Decode pool (premium) | Highest HBM3 bandwidth; 8× dedicated NICs = no bottleneck receiving KV |

**Splitwise heterogeneous cluster applied to AWS:**
- Prefill: p5.48xlarge (H100, max FLOPS)
- Decode: p4de.24xlarge (A100 80 GB, max HBM at lower cost)
- Expected outcome: ~2.35× more throughput at same cost vs all-p5, matching Splitwise's benchmark

---

## Technology Summary Table

| Technology | Scope | Current Best | Latency | RDMA | GPU-Native |
|---|---|---|---|---|---|
| NVLink | Intra-node GPU↔GPU | 1,800 GB/s (NVLink 5) | 100–300 ns | N/A | NVIDIA only |
| NVSwitch | Intra-node full mesh | 130 TB/s (NVL72) | <500 ns | N/A | NVIDIA only |
| NVLink Switch | Intra-rack rack↔rack | ~200 GB/s / link | ~1 µs | N/A | NVIDIA only |
| PCIe | GPU↔CPU | 242 GB/s (Gen6) | ~1 µs | No | Universal |
| InfiniBand NDR | Inter-node | 50 GB/s / NIC | 0.5 µs | Native | Universal |
| InfiniBand XDR | Inter-node | 100 GB/s / NIC | <0.5 µs | Native | Universal |
| RoCEv2 (400G) | Inter-node | ~40 GB/s / NIC | 5–10 µs | Via PFC | Universal |
| AWS EFA | Inter-node | 400 GB/s / instance (p5) | ~2–5 µs | Yes | AWS only |
| 100 GbE TCP | Inter-node | ~6 GB/s effective | 50 µs | No | Universal |
| CXL | CPU↔memory | ~64–128 GB/s | ~1 µs | Coherent | Emerging |

---

## References

- NVIDIA NVLink official page: https://www.nvidia.com/en-us/data-center/nvlink/
- NVIDIA GB200 NVL72 specs: https://www.nvidia.com/en-us/data-center/gb200-nvl72/
- Wikipedia NVLink: https://en.wikipedia.org/wiki/NVLink
- InfiniBand generations guide: https://network-switch.com/blogs/networking/infiniband-cables-types-speeds-connectors
- RoCEv2 vs InfiniBand 2025: https://dataoorts.com/roce-v2-vs-infiniband-compare-for-gpu-clusters/
- AWS g5 instances: https://aws.amazon.com/ec2/instance-types/g5/
- AWS p4d/p4de instances: https://aws.amazon.com/ec2/instance-types/p4/
- AWS p5 instances: https://aws.amazon.com/ec2/instance-types/p5/
- NVLink scale-up networking (2025): https://introl.com/blog/nvlink-scale-up-networking-gpu-interconnect-infrastructure-2025
- Blackwell platform interconnect guide: https://dev.to/aicplight/from-node-to-superpod-interconnect-and-optical-design-considerations-for-nvidia-blackwell-platforms-2pgn
- Splitwise (ISCA 2024): hardware heterogeneity results → `L3/02_splitwise_isca24.md`
- Mooncake (FAST 2025): GPUDirect RDMA, topology-aware NIC selection → `L3/03_mooncake_fast25.md`
- NVIDIA Dynamo + NIXL: NIXLBench, KVBench → `L3/04_nvidia_dynamo_nixl.md`
