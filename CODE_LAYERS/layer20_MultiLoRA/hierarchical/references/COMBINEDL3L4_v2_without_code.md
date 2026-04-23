# Chapter: Prefill-Decode Disaggregation in Large Language Model Serving

**Format:** Textbook-style prose chapter. Concepts build sequentially; each section establishes the ground for the next. Systems (DistServe, Splitwise, SARATHI, Mooncake, NIXL, Dynamo, TaiChi, vLLM) are cited as examples within a concept-driven narrative, not as the organizing principle.

---

## 1. The Two Phases of LLM Inference

To understand why the serving of large language models is architecturally challenging, we must begin with the most fundamental observation about how inference works: a single request to a language model involves two computationally distinct phases that have almost nothing in common from the perspective of hardware utilization.

### 1.1 The Prefill Phase

When a user submits a prompt, the model must first process the entire input sequence before it can generate a single output token. This processing step is called the **prefill phase**, and it is characterized by a single property that dominates everything else: the model can process all input tokens simultaneously, in parallel, in a single forward pass.

During prefill, the model computes three vectors — Query, Key, and Value — for every token in the input, across every attention layer. The Key and Value vectors for every token are then stored in GPU high-bandwidth memory (HBM) as the **KV cache**, which will serve as the model's working memory for the remainder of the request. The KV cache exists for a practical reason: without it, every subsequent decode step would need to recompute attention over the entire growing sequence from scratch, an operation whose cost scales quadratically with sequence length. By materializing and storing the K and V tensors once during prefill, every decode step can run in O(n) time rather than O(n²).

From a hardware perspective, the prefill phase is dominated by large matrix multiplications that multiply the full sequence length against the model's weight matrices. For a 4,096-token prompt processed by a model like LLaMA-3.1-70B, the arithmetic intensity — defined as floating-point operations per byte of memory accessed — sits in the range of 200 to 400 FLOP per byte. This value places prefill far to the right of the roofline model's ridge point, firmly in the compute-bound regime. The GPU's tensor cores operate at 90 to 95 percent utilization. The HBM bus, by contrast, is barely touched — the computation requires only the model weights, which are accessed once and reused across all input tokens. Adding more memory bandwidth to a prefill-only GPU would produce almost no improvement.

### 1.2 The Decode Phase

After prefill completes, the model enters the **decode phase**, in which it generates output tokens one at a time, sequentially. This sequential constraint is not an engineering choice that can be optimized away — it is a mathematical property of autoregressive generation. Each output token depends on all previous output tokens, which means the model cannot begin computing token $t+1$ until it knows the value of token $t$.

In each decode step, the model reads the entire KV cache from HBM, attends over the full accumulated sequence, generates a probability distribution over the vocabulary, samples the next token, appends its Key and Value vectors to the cache, and repeats. The dominant cost of each step is reading the KV cache, and that cache grows by one row with every token generated. For a request generating 512 output tokens, the final decode step reads a KV cache that is 512 tokens larger than what was present after prefill.

From a hardware perspective, each individual decode step is tiny. The matrix multiplications that dominate prefill are here replaced by vector-matrix products with a batch dimension of one per active request. The arithmetic intensity collapses to approximately 1 FLOP per byte, regardless of sequence length — a value so low that it places decode at the extreme left of the roofline model, pinned against the memory-bandwidth wall. The GPU's tensor cores operate at 20 to 40 percent utilization even when the decode batch is large, because the mathematical structure of autoregressive generation simply does not produce enough arithmetic to keep them busy. What limits decode throughput is not compute — it is the rate at which HBM can deliver KV cache data to the attention kernels.

### 1.3 Why the Asymmetry Is Not Incidental

It is tempting to treat the difference between prefill and decode as a matter of degree — prefill is more compute-intensive, decode is more memory-intensive, and both phases happen on the same hardware, so perhaps the hardware just averages across them. This framing misses how severe the asymmetry actually is.

Splitwise, a research effort from Microsoft published at ISCA 2024, characterized both phases from real Azure production traces across two LLM serving services. Their data confirmed what theoretical analysis predicts: even with continuous batching across many concurrent requests, the decode phase consistently underutilizes GPU compute. The paper's conclusion was direct — token generation phases do not require the compute capability of the latest GPUs. An H100 GPU, designed to deliver 80 teraFLOPS of FP16 throughput, sits at 20 to 40 percent compute utilization during decode, because the workload's structure cannot consume more. You are, in this regime, paying for 60 teraFLOPS that are doing nothing. This is not a tuning problem. It is a fundamental consequence of the mathematical structure of autoregressive decoding.

The SPAD research group at UT Austin validated the hardware insensitivity from the opposite direction. They ran controlled experiments modifying the hardware characteristics of the GPU used for each phase. When they reduced HBM memory bandwidth by 40 percent on the chip running prefill, the measured prefill latency increased by only 17 percent — confirming that prefill barely uses bandwidth at all. When they reduced compute capacity by 50 percent on the chip running decode, the measured decode latency increased by only 22 percent — confirming that decode barely uses compute at all. These are not small effects. They demonstrate that the two phases are not simply different points on the same workload spectrum. They are in fundamentally different hardware regimes, and the optimal hardware for each phase has almost no overlap with the optimal hardware for the other.

---

## 2. The Interference Problem: What Happens When Both Phases Share a GPU

Given the structural asymmetry described above, the question becomes: what happens when you run both phases on the same pool of GPUs, as essentially every LLM serving system did before 2024?

### 2.1 The Mechanism of Interference

In a standard monolithic serving deployment, a single GPU pool processes incoming requests with a technique called continuous batching. New requests enter the batch at prefill and transition to decode as they produce tokens. At any given moment, the batch contains a mix of requests that are currently in prefill and requests that are currently in decode. This mixing is the source of the problem.

When a new request arrives and its prefill begins, the GPU must allocate substantial compute time to processing the input sequence. A 4,096-token prompt requires hundreds of milliseconds of dense matrix computation. During this time, every decode request that is actively generating tokens must wait. From the perspective of a user who submitted a request 30 tokens ago and is watching the streaming output, the text simply stops appearing mid-sentence. The prefill of a completely different user's long prompt has stalled their output generation. This is the fundamental form of the interference: the compute-bound workload (prefill) occupies the hardware, starving the memory-bandwidth-bound workload (decode) of the scheduling time it needs.

The interference runs in both directions. When the decode queue is heavily loaded with many concurrent active generations, new requests must queue behind them before their prefill can begin. Even when the server is not operating near its throughput capacity, the waiting time before a new request's prefill starts — the time-to-first-token, or TTFT — can be substantial. The problem is not capacity; it is scheduling priority conflict between two workloads with incompatible resource profiles.

### 2.2 The Resource Coupling Problem

Beyond the scheduling-level interference just described, there is a deeper architectural problem that scheduling cannot fix: **resource coupling**. The two phases require not just different scheduling priorities but different hardware configurations.

The tensor parallelism (TP) degree — the number of GPUs across which a model's layers are split — has opposite optimal values for the two phases. A lower TP degree means fewer all-reduce communication operations per layer, which reduces the per-request TTFT. A higher TP degree means each GPU holds a smaller shard of the model, which reduces the amount of HBM each GPU must read per decode step, lowering ITL. These are opposite requirements. In a collocated pool, you must choose a single TP degree that serves both phases, and that value is necessarily suboptimal for both.

Similarly, the batch size strategy that maximizes prefill efficiency — processing multiple long prompts simultaneously to achieve high compute utilization — is incompatible with the batch size strategy that minimizes decode ITL, which requires keeping individual decode steps as short as possible. A scheduling system that tries to serve both simultaneously makes both worse than they would be in dedicated pools.

### 2.3 Quantifying the Damage

DistServe, published at USENIX OSDI 2024, measured the interference effect directly under production-representative load and SLO constraints. The findings were stark. Collocated systems showed three to five times higher TPOT (time per output token) variance compared to disaggregated systems, even when the most sophisticated aggregation-side scheduling technique — chunked prefill, described in the next section — was applied. The decode ITL spiked in direct positive correlation with the prefill batch size, a relationship that is visible in measurements across multiple model sizes and request distributions. TTFT rose above SLO thresholds even at request rates well below the server's peak throughput capacity, because decode queue occupancy was preventing prefill requests from being processed promptly.

These are not edge cases or pathological loads. They are the normal operating conditions of a serving system under any meaningful load, because any useful deployment will have concurrent requests in both phases simultaneously.

---

## 3. The Aggregation Baseline: Can Better Scheduling Solve the Problem?

Before accepting that the problem requires an architectural solution — that is, physically separating the two phases — it is worth asking whether improved scheduling within the collocated architecture can close the gap. SARATHI, published at USENIX OSDI 2024, represents the definitive attempt to answer this question.

### 3.1 Chunked Prefill: The Core Mechanism

The central idea of SARATHI is **chunked prefill**: instead of processing a large prefill request in a single uninterrupted forward pass, split the prefill into fixed-size chunks and interleave each chunk with a batch of decode steps. A prefill request for 4,096 tokens, for example, might be processed as eight chunks of 512 tokens each. Between each chunk, the scheduler allows decode requests to run, so that no single prefill completely monopolizes the GPU for its full duration.

The practical effect is a form of time-sliced scheduling. The maximum time that any decode request can be stalled by a single prefill is now the time required to process one chunk, not the time required to process the entire prompt. For a long prompt that would otherwise stall decode for 800 milliseconds, chunking it into 8 steps reduces the maximum stall to approximately 100 milliseconds per chunk. Decode requests piggyback on the compute gaps within each chunk's batch.

SARATHI also reduces pipeline bubble overhead in pipeline-parallel configurations. In multi-GPU pipeline parallelism, the variable duration of different prefill requests creates irregular microbatch sizes that leave pipeline stages idle while waiting for the previous stage to complete. Chunked prefill creates uniformly-sized microbatches, reducing pipeline bubbles by up to 6.29× on GPT-3 sized models at pipeline depth 8, and delivering up to 1.91× end-to-end throughput improvement from this structural effect alone.

### 3.2 What SARATHI Achieves

SARATHI's measured results are substantial. For LLaMA-13B on A6000 GPUs, decode throughput improves by up to 10× compared to standard continuous batching, because decode requests no longer spend most of their time waiting behind long prefill operations. End-to-end throughput improves by up to 1.33×. For larger models like LLaMA-33B on A100 GPUs, the improvements are 4.25× for decode throughput and 1.25× for end-to-end throughput.

These are real and significant gains, achievable without any change to the serving infrastructure — no additional hardware, no separate pools, no network transfers. This is why chunked prefill is now a standard feature in all major inference frameworks, including SGLang, vLLM, and TensorRT-LLM.

### 3.3 The Hard Limits of Scheduling

However, SARATHI's gains come with a ceiling that no scheduling improvement can break through, because the ceiling is not a scheduling limit — it is a physics limit.

Even a one-token prefill chunk requires exclusive GPU access for a full forward step. During that step, no decode request can run. At high prefill rates, with many concurrent long-prompt requests arriving, the residual interference from chunk-level stalls remains significant. DistServe measured three to five times higher TPOT variance with chunked prefill compared to disaggregated serving — meaning that even SARATHI's best-case output still produces substantially more variance than a system where decode is never interrupted by prefill at all.

More fundamentally, chunked prefill does nothing about resource coupling. Both phases still run on the same hardware with the same TP configuration. The tensor cores that sit idle during decode are still sitting idle. The HBM bandwidth that goes unused during prefill is still going unused. The optimal hardware for each phase is still the same suboptimal compromise. SARATHI is the correct answer to the question "how do we reduce scheduling interference?" but it cannot answer the question "how do we eliminate it?" and it cannot address the underlying hardware mismatch.

It is worth noting that SARATHI and disaggregation are not mutually exclusive — they operate at different levels of the stack. In SGLang's disaggregated deployment, the prefill server itself uses chunked prefill internally (controlled by `--chunked-prefill-size`) to prevent any single large prompt from occupying the prefill GPU so long that it delays the KV transfer handoff to the decode server. Chunked prefill within a dedicated prefill pool is still useful; it just can no longer cause decode interference, because decode runs on entirely separate hardware.

---

## 4. The Correct Optimisation Target: Goodput

Before examining the disaggregated solution in detail, it is important to establish the correct metric by which to measure success. This matters because the conventional metric — raw throughput in requests per second — systematically obscures the interference problem and makes suboptimal solutions appear satisfactory.

### 4.1 Why Raw Throughput Misleads

A serving system can achieve high throughput while violating the latency constraints that users actually care about. Suppose a system processes 100 requests per second, but 30 percent of those requests experience TTFT longer than the service-level objective (SLO) specifies, or TPOT that causes the streaming output to stutter. From a user-experience perspective, those 30 percent of requests are failures — they did not receive the service the system promised. But raw throughput counts them as successes, because they were processed.

In a collocated serving system under moderate load, the interference described in the previous sections tends to produce correlated latency violations: when TTFT is violated because the prefill queue is backed up, TPOT tends to also be violated because the decode queue that was backing up the prefill queue is itself congested. The system fails both metrics simultaneously, but its raw throughput number tells you nothing about this.

### 4.2 Goodput: The Principled Alternative

DistServe introduced **goodput** as the correct optimisation target for LLM serving: the number of requests that satisfy both the TTFT SLO and the TPOT SLO per unit time. A request contributes to goodput only if TTFT is at or below the TTFT threshold and TPOT (equivalently, ITL — inter-token latency) is at or below the TPOT threshold, with both conditions holding simultaneously.

This definition captures what users and operators actually care about: not how many requests the system touches, but how many it serves correctly. The goodput metric exposes exactly the trade-off that collocated serving systems face: as you accept more requests to improve throughput, you push more of them outside the SLO box in both dimensions simultaneously, collapsing goodput even as raw throughput climbs.

DistServe's experimental results are reported in goodput, which is why their headline number — 7.4 times more requests per second at the same SLO constraints — is so large. Under raw throughput comparison, the improvement would appear smaller, because the baseline's raw throughput counts requests that are failing SLOs. When measured correctly, with only SLO-satisfying requests counted, the advantage of eliminating interference is far more pronounced. This is a deliberate methodological choice that reflects what the improvement actually means in production.

---

## 5. The Disaggregated Solution: Separating the Phases

With the problem established precisely and the limits of scheduling-based mitigation understood, the architectural solution becomes clear: the two phases must run on separate hardware that can be independently configured and independently scaled.

### 5.1 The Architecture

A disaggregated serving system replaces the monolithic GPU pool with three distinct components. The first is a **router**, which serves as the single client-facing entry point. It receives incoming requests, maintains awareness of the state of all downstream workers, routes each request to an available prefill worker, and — after prefill completes — coordinates the transfer of the KV cache to an appropriate decode worker. The router holds no model state and performs no inference computation.

The second component is the **prefill pool**, a set of GPU workers configured specifically for the prefill workload. These workers run the model forward pass over input prompts, accumulate the KV cache, and immediately transfer it to a decode worker upon completion. They never generate output tokens. Because they are dedicated to a single workload, they can be configured with a tensor parallelism degree chosen specifically to minimize per-request TTFT, and they can run batching strategies optimized for compute throughput without concern for decode interference.

The third component is the **decode pool**, a separate set of GPU workers configured specifically for the decode workload. These workers receive KV caches from prefill workers and run autoregressive token generation until the request is complete. They never process input prompts. They can be configured with a tensor parallelism degree chosen specifically to maximize HBM utilization and token generation throughput, and they can maintain large concurrent request batches without any scheduling interference from prefill operations.

The full lifecycle of a request in this architecture is as follows: the client sends a request to the router; the router selects an available prefill worker and forwards the request; the prefill worker processes the prompt, builds the KV cache, and transfers it across the network to a decode worker; the decode worker begins autoregressive generation and streams output tokens back to the client through the router. At no point do the prefill and decode computations interfere with each other, because they are running on entirely separate hardware.

### 5.2 What Separation Enables

The immediate benefit is the elimination of scheduling interference. Because the decode pool never runs prefill, decode ITL is perfectly stable — there is no workload on the decode GPU that can preempt or delay a decode step. Because the prefill pool never runs decode, prefill throughput is not limited by decode queue occupancy.

But the more significant benefit is the removal of resource coupling, which enables optimisations that were structurally impossible in a collocated system. The prefill pool can use a different tensor parallelism degree than the decode pool. In DistServe's experiments, prefill workers used a lower TP to minimize per-request TTFT through reduced all-reduce overhead, while decode workers used a higher TP to maximize HBM utilization across shards. This configuration produced each pool operating at or near its hardware optimum for its respective workload — something that cannot happen when a single TP configuration must serve both.

Similarly, the two pools can scale independently. If the TTFT SLO is being violated, the prefill pool can be scaled up without adding decode capacity — and the cost of the additional capacity goes entirely toward addressing the TTFT problem, rather than being shared between prefill and decode. This independent scaling granularity is not available when both phases share a pool.

DistServe's evaluation across OPT-13B, OPT-66B, and OPT-175B, measured against vLLM with continuous batching as the baseline, showed 7.4 times more goodput at the same SLO thresholds, and the ability to satisfy SLO constraints 12.6 times tighter than the baseline at the same request rate. The improvement was consistent across chatbot, document summarisation, and coding workloads.

---

## 6. Hardware Heterogeneity: The Cost Consequence of Separation

Once the two phases are running on separate hardware, an additional dimension of optimization becomes available: the two pools need not use the same GPU model. This insight — that the optimal hardware for each phase is different — has direct and quantifiable cost implications.

### 6.1 The Mismatch at the Silicon Level

An H100 SXM GPU costs approximately thirty thousand dollars and delivers 80 teraFLOPS of FP16 throughput alongside 3.35 terabytes per second of HBM3 bandwidth. During decode, the 80 teraFLOPS goes almost entirely unused, because the arithmetic intensity of autoregressive attention is approximately 1 FLOP per byte. The chip's compute resources sit idle while its memory bandwidth runs at or near saturation. In other words, a significant fraction of what you paid for — specifically, the tensor core silicon that dominates the H100's cost — is delivering zero value during decode.

An A100 80GB GPU costs approximately fifteen thousand dollars and delivers 77 teraFLOPS of FP16 throughput alongside 2.0 terabytes per second of HBM2e bandwidth. For a decode-only worker, this chip provides approximately 60 percent of the decode throughput of an H100 at approximately 50 percent of the cost. The per-dollar decode throughput ratio favors the A100 over the H100 for decode work.

For prefill work, the situation reverses. Prefill is compute-bound, so 80 teraFLOPS vs 77 teraFLOPS matters, and the additional FLOP budget of newer architectures (H100 delivers substantially higher FP8 throughput than A100 delivers in any precision) directly reduces per-request TTFT. Prefill does not benefit meaningfully from additional HBM bandwidth beyond what is needed to load model weights, so the bandwidth advantage of HBM3 over HBM2e is secondary.

### 6.2 The Splitwise Measurement

Splitwise formalized this reasoning and backed it with experimental data. Their evaluation compared a heterogeneous cluster using H100 GPUs for the prefill pool and A100 GPUs for the decode pool against a homogeneous cluster using H100 GPUs for both pools. At the same total cost budget, the heterogeneous cluster delivered 2.35 times more throughput. Equivalently, at the same throughput target, the heterogeneous cluster achieved the target at 20 percent lower cost. These numbers hold under production-representative workload distributions from real Azure service traces.

The intuition is straightforward: by moving decode workloads to cheaper hardware that is sufficient for the memory-bandwidth-bound workload, you free up H100 budget to be used for what H100s actually do well — compute-intensive prefill. The result is that every dollar in the cluster contributes more to overall throughput.

This insight has been confirmed in practice. Perplexity AI, which operates disaggregated serving at 435 million search queries per month across more than 20 AI models, explicitly cited hardware heterogeneity as one of the key benefits of their disaggregated deployment, noting that disaggregation gives them the flexibility to use different GPU products for each inference phase given each phase's specific hardware resource requirements.

### 6.3 The AWS Mapping

On AWS, this translates to a practical cluster design. The prefill pool maps naturally to `p5.48xlarge` instances (8× H100 SXM, 3,200 Gbps EFA networking, 640 GB HBM3 total), which provide maximum FP16 and FP8 FLOP throughput for compute-intensive prefill work. The decode pool maps to `p4de.24xlarge` instances (8× A100 80GB HBM2e, 400 Gbps EFA with GPUDirect RDMA, 640 GB HBM2e total), which provide sufficient memory bandwidth for decode at lower cost per GPU. The Splitwise result of 20 percent cost reduction at the same throughput becomes a real procurement decision: for a cluster of this scale, the savings over a homogeneous H100 fleet are substantial.

---

## 7. The Transfer Problem: Moving the KV Cache

Disaggregation introduces a problem that does not exist in collocated serving: the KV cache, built by the prefill worker in its local VRAM, must be physically moved across a network to the decode worker's VRAM before token generation can begin. This transfer is the primary cost that disaggregation adds, and it determines whether the interference-elimination benefit is large enough to justify the architectural complexity.

### 7.1 The Size of the Cache

The KV cache size is determined by the model architecture and the prompt length. For each token in the input, each attention layer generates one Key vector and one Value vector, each of dimension `n_kv_heads × head_dim`. The total KV cache size is therefore the product of the number of layers, the number of key-value heads, the head dimension, two (for K and V), the precision in bytes per element, and the number of tokens in the prompt.

For LLaMA-3.1-70B using grouped-query attention with 8 KV heads per layer, 80 layers, 128-dimensional heads, and FP16 precision, a 4,096-token prompt produces a KV cache of approximately 1.34 gigabytes. A 128K-token context produces approximately 40 gigabytes. For DeepSeek-V3, which uses multi-head latent attention (MLA) — an architectural choice that compresses the KV representation — a comparable context produces a smaller cache, though still in the gigabyte range.

These are the payloads that must cross the network between prefill and decode workers on every single request. At a serving rate of 100 requests per second with 4K-token prompts, the LLaMA-70B system needs to transfer 134 gigabytes of KV data per second through the network connecting the prefill and decode pools.

### 7.2 The Network Speed Requirement

For a serving system targeting a TTFT SLO of 500 milliseconds with a prefill computation time of 200 milliseconds, the remaining budget for KV transfer is 300 milliseconds. With a 1.34 GB payload, that requires at minimum 4.5 gigabytes per second of effective network bandwidth — effective, meaning the bandwidth available for the actual KV data after protocol overhead, and including the GPU-to-NIC path on both ends.

Standard 100 Gigabit Ethernet provides 12.5 gigabytes per second at wire rate, but in the absence of GPUDirect RDMA, the actual KV transfer follows a path that crosses PCIe twice: from prefill GPU VRAM to CPU RAM via PCIe, then from CPU RAM to the NIC, then through the network, then from the NIC into the remote CPU RAM, then from remote CPU RAM to the decode GPU VRAM via PCIe. Each PCIe crossing consumes bandwidth from the PCIe bus, which is shared across all devices connected to the same CPU socket. The effective throughput of this CPU-mediated path on a 100 GbE link is approximately 6 gigabytes per second — roughly half the wire rate — due to PCIe bandwidth sharing and CPU involvement.

At 6 gigabytes per second, a 1.34 GB transfer takes approximately 220 milliseconds. For a 500 ms TTFT SLO with 200 ms prefill time, this uses 73 percent of the total TTFT budget on the transfer alone, leaving almost no margin. For longer prompts or tighter SLOs, this is simply not viable.

### 7.3 GPUDirect RDMA: The Enabling Technology

The solution to the CPU-mediated path problem is **GPUDirect RDMA**, a capability that allows the network interface card to read directly from and write directly to GPU VRAM, bypassing the CPU and CPU DRAM entirely. With GPUDirect RDMA, the transfer path becomes: prefill GPU VRAM → RDMA NIC → network → RDMA NIC → decode GPU VRAM. The PCIe crossings still exist, but they are short (NIC to GPU on the same root complex, rather than NIC to CPU to GPU across NUMA boundaries), and they carry only the transfer payload, not the CPU's interrupt handling overhead.

On InfiniBand NDR (400 Gbps, 50 gigabytes per second effective), a 1.34 GB KV cache transfer with GPUDirect RDMA takes approximately 27 milliseconds — 8 times faster than the CPU-mediated 100 GbE path. On InfiniBand HDR (200 Gbps, 25 gigabytes per second), the same transfer takes approximately 54 milliseconds. Within a single server using NVLink (where prefill and decode run on different GPUs of the same machine), the 900 gigabytes per second bandwidth of NVLink 4.0 reduces the transfer to approximately 1.5 milliseconds — effectively free. Within an NVL72 rack using rack-scale NVLink, the speed is comparable, at approximately 1 millisecond.

GPUDirect RDMA is not optional for production disaggregation. It is the technology that makes the transfer cost acceptable. This is why the entire software ecosystem for KV transfer — Mooncake, NIXL, and the vLLM connector abstractions — is built on top of it.

---

## 8. The Transfer Layer: Three Approaches

The transfer problem has two dimensions that must be solved simultaneously. The first is the physical transport dimension: given GPUDirect RDMA as the foundation, how do you use the available network hardware as efficiently as possible? The second is the abstraction dimension: how do you expose KV transfer as a reusable interface that inference frameworks can use without coupling themselves to a specific transport technology? The ecosystem has developed three complementary answers, each operating at a different level of the stack.

### 8.1 The Production Transfer Engine: Mooncake

Mooncake, published at USENIX FAST 2025, is the KV transfer engine built by Moonshot AI for the Kimi LLM service. It represents a production-system approach: start with the physical reality of inference server hardware and solve every obstacle to maximum transfer throughput.

The first obstacle Mooncake addresses is NIC-GPU path non-uniformity. Modern inference servers have multiple CPU sockets, NUMA nodes, PCIe root complexes, GPUs, and RDMA NICs. Not all NIC-GPU paths are equivalent. A NIC attached to the same PCIe root complex as a given GPU provides full bandwidth for transfers to or from that GPU's VRAM. A NIC attached to a different PCIe root complex, requiring a UPI or NUMA interconnect hop to reach the GPU, provides substantially less — often 50 percent or less of the peak bandwidth. On a server with 8 GPUs and 4 NICs spread across 2 NUMA nodes, naive NIC selection (for example, always using NIC 0 regardless of which GPU holds the KV cache) can halve effective transfer bandwidth.

Mooncake solves this with topology-aware NIC selection. During initialization, each server generates a topology matrix that maps every GPU to every NIC with their affinity scores and measured bandwidth. When a KV transfer is requested, the engine consults this matrix and selects the NIC (or NICs) with the highest-bandwidth path to the source GPU. This is not a one-time calibration — the topology can be updated as NICs are added or removed, and the engine re-evaluates path selection for each transfer.

The second obstacle is single-NIC bandwidth limits. A single NDR InfiniBand NIC provides approximately 50 gigabytes per second of effective RDMA bandwidth. For a 40 GB KV cache — LLaMA-70B at 128K context — transferring over a single NIC takes approximately 800 milliseconds, which is almost certainly beyond any reasonable TTFT budget. Modern inference servers commonly have 4 to 8 NICs. Mooncake exploits this with multi-NIC pooling: large transfers are internally sliced into segments, each assigned to a different NIC based on topology affinity, with all segments submitted in parallel. Completion is tracked per segment and the transfer is considered complete only when all segments have arrived. For servers with 8 NICs, this produces up to 8 times the per-NIC bandwidth — approximately 400 gigabytes per second aggregate, reducing the 40 GB transfer from 800 ms to approximately 100 ms.

The third dimension Mooncake addresses is transport heterogeneity. InfiniBand RDMA is the right transport for inter-node GPU-to-GPU transfers. But within a rack of GB200 NVL72, all 72 GPUs are interconnected via rack-scale NVLink at approximately 1,000 gigabytes per second, far exceeding what any InfiniBand NIC can provide. Mooncake supports pluggable transport backends — RDMA for inter-node, NVLink (MNNVL) for intra-rack, NVMe-oF for loading KV from persistent storage — with the same API exposed to the calling inference framework regardless of which backend is active.

In production on the Kimi service, Mooncake's optimizations produced 75 percent more requests handled under SLO constraints compared to the pre-disaggregation baseline, and up to 525 percent throughput increase for long-context workloads. Kimi K2, deployed in July 2025 on 128 H200 GPUs, achieved 224,000 tokens per second of prefill throughput and 288,000 tokens per second of decode throughput — numbers that would not be possible without efficient KV transfer enabling the two pools to operate at near-independent efficiency.

### 8.2 The Vendor-Agnostic Library: NIXL

NIXL (NVIDIA Inference Xfer Library) takes a different design goal than Mooncake. Where Mooncake is optimized for production deployment on specific hardware and maximizes throughput through hardware-specific techniques, NIXL is designed to be a general-purpose KV transfer library that any inference framework can use, regardless of the underlying interconnect technology.

Each server running a NIXL-enabled inference engine instantiates a **Transfer Agent**, which manages all KV transfer operations for that process. The Transfer Agent presents a unified memory abstraction across all memory types it can access: GPU VRAM (registered for GPUDirect RDMA), CPU DRAM (pinned and registered), local NVMe, and remote storage (NVMe-oF or object storage). From the caller's perspective, all of these are addressed through the same buffer-list API — a list of (address, length) pairs that describe the source or destination of a transfer. The caller does not specify which transport to use or how the data will physically move; the Transfer Agent selects the appropriate backend based on the memory types of the source and destination.

The key capability that NIXL exposes explicitly, and that distinguishes it from lower-level RDMA libraries, is **asynchronous non-blocking transfers**. A caller submits a transfer operation and receives a handle. The handle can be polled later to check completion status. This allows the inference engine's decode worker to begin executing forward-pass computation on the layers of the KV cache that have already arrived while the remaining layers are still in transit over the network. The compute and the transfer overlap in time, reducing the effective wall-clock cost of the transfer below what the raw bandwidth calculation would suggest.

NIXL also handles metadata exchange between Transfer Agents. Before data can flow via RDMA, both endpoints must register their memory with the RDMA subsystem and exchange memory registration keys. NIXL manages this through an etcd-backed metadata cache, so that the first transfer between two endpoints pays the registration overhead and all subsequent transfers can proceed immediately with the cached credentials.

NIXL serves three distinct use cases within the same interface: KV cache transfer for disaggregated prefill-decode serving (the primary use case discussed in this chapter), loading KV caches from persistent storage to GPU VRAM for long-context requests that cannot hold the entire context in VRAM simultaneously, and moving expert activations across GPUs for the all-to-all communication patterns in mixture-of-experts model serving. All three use the same Transfer Agent and the same buffer-list API.

### 8.3 The Framework Abstraction: vLLM's Connector Interface

While Mooncake and NIXL solve the transport problem at the systems level, inference frameworks need a stable API surface that lets them express "this worker is producing KV caches" or "this worker is consuming KV caches" without coupling to a specific transport. vLLM's disaggregated prefilling implementation, which evolved from Splitwise's GitHub PR #2809 (the first public prototype of inter-instance KV transfer in vLLM), provides this abstraction through its connector architecture.

The central interface is `BaseKVConnector`, an abstract class with two methods. The `send_kv_caches_and_hidden_states` method is called on the prefill instance after its forward pass completes; it is responsible for writing the KV tensors to whatever transfer buffer the connector maintains. The `recv_kv_caches_and_hidden_states` method is called on the decode instance before its forward pass; it blocks until KV tensors are available and returns them along with a boolean flag called `bypass_model_exec`. When this flag is `True`, the decode instance skips its own prefill computation entirely, because the KV cache has arrived from the remote prefill instance. This flag is the critical design decision — it allows the framework to check a single boolean and skip an entire model phase without needing to know anything about the transfer mechanism that delivered the data.

The `KVLookupBufferBase` interface governs the ownership protocol for the shared transfer buffer. It exposes two operations: `insert`, called by the prefill instance to place the KV cache into the buffer, and `drop_select`, called by the decode instance to atomically retrieve and remove the KV cache for a given request. The "drop" in `drop_select` is not incidental — it ensures that each KV cache is consumed by exactly one decode instance, preventing double consumption or concurrent access races. This ownership semantics is correct for disaggregated serving: once the decode instance has the KV cache, the prefill instance no longer needs it, and the buffer slot can be reused.

vLLM ships six concrete implementations of this connector interface, each addressing the same transfer problem for a different infrastructure context. `NixlConnector` uses RDMA via the UCX library and is the default for production clusters with InfiniBand or RoCEv2 networking. `MooncakeConnector` uses Mooncake's Transfer Engine with full topology-aware and multi-NIC capabilities. `P2pNcclConnector` uses NCCL's peer-to-peer communication primitives for clusters that lack RDMA NICs but have NVLink or PCIe connectivity. `LMCacheConnectorV1` integrates NIXL with the LMCache persistent KV storage system for unified handling of cross-engine KV sharing and long-context storage. `MultiConnector` chains any combination of the above, enabling fallback chains such as RDMA primary and NCCL fallback if RDMA is unavailable. `ExampleConnector` provides a reference implementation as a starting template for custom connectors.

Together, these six connectors define the full design space for KV transfer: RDMA-native, RDMA with topology optimization, NCCL-based, storage-backed, chained, and custom. The vLLM documentation is direct about what this mechanism optimises: "Disaggregated prefill does not improve throughput. It improves latency SLO compliance and decouples TTFT from ITL." This statement is worth holding onto, because it correctly identifies what disaggregation is for.

---

## 9. The Orchestration Problem: What Transfer Alone Cannot Solve

Once the transfer layer is in place, a new class of problems becomes visible at production scale. These are not transport problems — the data is moving correctly and efficiently. They are **coordination problems**: how does the system know which worker is available, which one has the right KV cache prefix already cached, and how should pool sizes adjust as traffic patterns shift?

### 9.1 The Four Planes of Production Orchestration

NVIDIA Dynamo, released at GTC 2025, represents the most complete public statement of what production orchestration for disaggregated serving requires. Dynamo's central architectural claim is that managing a disaggregated serving system requires four separate planes of communication, each handling a different concern, and that conflating any two of these concerns in the same mechanism leads to scaling failures.

The **Request Plane** is the data path that every disaggregated system already has: client requests flow to the router, the router forwards them to prefill workers, the KV cache flows to decode workers, and token streams flow back to clients. This is the layer that Mooncake, NIXL, and the vLLM connectors address. It is necessary but not sufficient.

The **Discovery Plane** handles worker registration and health monitoring. In a small deployment with a handful of statically configured workers, the router can simply be given a list of URLs at startup. But at production scale, workers are added and removed dynamically as load changes, and individual worker processes die without warning. Without dynamic discovery, every topology change requires a manual router restart. Dynamo solves this with etcd-backed worker registration: each worker publishes a record containing its endpoint URL, its role (prefill or decode), the model it serves, and its current load state, with a time-to-live (TTL) of ten seconds. Workers refresh their TTL continuously while running. When a worker dies, its TTL expires, and the router automatically removes it from the routing table within one TTL period — approximately ten seconds. New workers become available to receive traffic within the same period after they complete startup and publish their record.

The **Event Plane** handles asynchronous state propagation. The router needs to know which prefill workers have which KV cache prefixes already computed, so it can route requests with repeated context (system prompts, conversation history) to the worker that will avoid recomputation. Workers need to know when SLO violations are occurring so they can adjust behavior. These information flows cannot be implemented as polling at scale — with fifty or more workers, continuous polling of all worker state by the router would generate more traffic than the actual inference requests. Dynamo implements an event bus where prefill workers publish their KV cache prefix state, and the router subscribes and maintains a continuously updated routing table. When a worker's KV cache state changes, it publishes an event; the router updates its table; subsequent routing decisions reflect the new state. This is asynchronous, push-based, and scales without polling overhead.

The **Control Plane (the Planner)** handles dynamic pool sizing based on live SLO violation rates. If TTFT P95 begins rising above the SLO threshold, the Planner issues a scale-up command to the prefill pool — adding workers or reassigning GPUs from underutilized resources. If TPOT violations appear, it grows the decode pool. The Planner implements a feedback control loop with configurable reaction time (to avoid overreacting to transient spikes) and cooldown (to prevent oscillation after a scale-up). SGLang's current disaggregated serving mode has no equivalent; scaling is fully manual. Dynamo's Planner represents what production autoscaling for disaggregated inference looks like when the system is mature.

### 9.2 KV-Aware Routing

Beyond the four-plane architecture, Dynamo's router implements a capability that provides significant TTFT reduction at no additional hardware cost: **KV-aware routing**. Each prefill worker maintains a local KV cache (analogous to SGLang's RadixCache) that stores the KV tensors for recently processed token prefixes. When the same system prompt, document context, or conversation history appears in multiple requests, the worker that processed the first request has the KV for the shared prefix already computed and stored. Routing subsequent requests with the same prefix to that worker allows it to skip recomputing the shared portion of the context — directly reducing TTFT by the time that would have been spent on the repeated prefill computation.

Dynamo's router tracks the prefix cache state of all prefill workers through the Event Plane and routes each incoming request to the worker with the highest prefix overlap against the request's input. SGLang's current router uses round-robin assignment; prefix-aware routing is a planned upgrade that would bring similar TTFT reduction without any change to the transfer or compute infrastructure.

---

## 10. When Disaggregation Is Not the Answer

Every preceding section has argued for disaggregation as the superior architecture. But this conclusion is conditional, and the condition is important: disaggregation is superior when the SLO that is being violated, or that requires the most margin, is the TPOT SLO. When the binding constraint is the TTFT SLO, the situation is more nuanced.

### 10.1 The SLO Regime Framework

TaiChi, published as an arXiv preprint in August 2025, provides the most rigorous analysis of when disaggregation outperforms aggregation and when it does not. The core insight is that the two architectures occupy different regions of the SLO feasibility space, and the optimal architecture depends on which SLO is tight.

When the TTFT SLO is tight and the TPOT SLO is relaxed — as in a conversational chatbot application where users expect the model to begin responding within one second but tolerate moderate token generation speed — disaggregation is actually harmful. In a disaggregated system, only the prefill pool handles new requests. If the prefill pool has, say, 4 GPUs and the decode pool has 12, then only 4 GPUs are available to drive down TTFT. In a collocated system with the same 16 GPUs, all 16 contribute to prefill, yielding a lower TTFT per request at the same request rate. The TPOT interference from the collocated system is acceptable, because the TPOT SLO is relaxed. Disaggregation, by dedicating most GPUs to decode, starves the prefill pool of resources precisely when TTFT is the constraint.

When the TPOT SLO is tight and the TTFT SLO is relaxed — as in a code generation application where users care about smooth, fast token streaming but are willing to wait a few seconds for the first token — disaggregation is clearly superior. The decode pool is never interrupted by prefill, so TPOT is perfectly stable. The higher TTFT caused by routing prefill through a separate pool is acceptable, because users have already consented (or have been designed around) a slightly longer initial wait.

When both SLOs are tight simultaneously — as in agentic AI systems where every step of a multi-step workflow has real-time requirements on both initial response and output speed — neither pure disaggregation nor pure aggregation is optimal. Disaggregation causes TTFT violations from insufficient prefill pool capacity; aggregation causes TPOT violations from prefill interference. The system is in a regime where the two SLOs are in structural conflict, and a static architecture cannot satisfy both simultaneously at the same request rate.

### 10.2 TaiChi's Solution: Dynamic Switching

TaiChi's response to the SLO regime problem is a dynamic architecture that continuously reassigns workers between prefill-heavy and decode-heavy roles based on live SLO violation measurements. Each worker is classified as either P-heavy (handling primarily prefill batches with occasional decode) or D-heavy (handling primarily decode batches with occasional prefill), and this classification is adjusted in real time by an SLO monitoring loop. When the monitoring system observes rising TTFT violations, it reassigns some D-heavy workers to P-heavy, increasing effective prefill capacity. When it observes TPOT violations, it reassigns some P-heavy workers to D-heavy. At steady state, the system maintains the minimum assignment of each type that satisfies both SLOs simultaneously.

TaiChi's experimental evaluation on DeepSeek-R1 and LLaMA-70B, compared against the respective state-of-the-art for each SLO regime, showed goodput improvements of up to 77 percent. More specifically, compared to pure disaggregation (the wrong architecture for tight TTFT), TaiChi reduced TTFT by up to 13.2 times. Compared to pure aggregation (the wrong architecture for tight TPOT), TaiChi reduced TPOT by up to 1.69 times. These are not marginal improvements — they reflect the cost of using the wrong static architecture for a given SLO regime.

### 10.3 The Decision Map

Given the SLO regime framework, the choice of architecture can be reduced to a decision based on the relative tightness of the two SLO constraints. A conversational chatbot application with a 1-second TTFT SLO and a relaxed 200 ms TPOT SLO should use aggregation with chunked prefill. A code generation application with a relaxed 3-second TTFT SLO and a tight 50 ms TPOT SLO should use disaggregation. An agentic reasoning system with tight constraints on both should use a TaiChi-style hybrid or provision a large enough disaggregated cluster that the prefill pool is not the bottleneck for TTFT. Batch-only workloads with no latency requirements should use aggregation to maximize throughput. RAG systems with long retrieval-augmented prompts and tight TPOT requirements should use disaggregation.

There is one category of workload where disaggregation is mandatory regardless of SLO regime: Mixture-of-Experts models like DeepSeek-V3. These models use expert parallelism communication libraries (in DeepSeek's case, DeepEP) that implement different communication dispatch modes for prefill and decode — a high-throughput "normal" mode for the large batches produced during prefill, and a low-latency mode for the single-token batches produced during decode. These modes are mutually exclusive: a worker cannot optimally serve both simultaneously. Only by running prefill and decode on separate workers, each using its optimal dispatch mode exclusively, can the model achieve full performance. For MoE models, disaggregation is not a latency optimization — it is a correctness optimization.

---

## 11. Production Evidence: Does It Work at Scale?

The theoretical arguments and experimental results described in the preceding sections are validated by multiple production deployments at substantial scale.

Perplexity AI operates disaggregated serving for 435 million search queries per month across more than 20 AI models simultaneously, running on H100 SXM GPUs. This deployment began in late 2024 and is the largest publicly confirmed production deployment of PD disaggregation. NVIDIA's spotlight report on this deployment confirmed both of the key claimed benefits: higher throughput within the SLO constraints (reducing cost per token), and the operational freedom to use different GPU products for each phase based on each phase's actual hardware requirements. The estimated cost savings on the Related-Questions feature alone are approximately one million dollars per year — a concrete financial validation of the cost arithmetic discussed in section 6.

The SGLang team, in collaboration with the LMSYS organization, deployed DeepSeek-V3 (a 671-billion parameter MoE model) across 96 H100 GPUs in May 2025. The deployment used a 3:9 prefill-to-decode ratio — 24 GPUs for prefill across 3 nodes, 72 GPUs for decode across 9 nodes — using Mooncake as the transfer engine over InfiniBand NDR. The measured throughput was 52,300 tokens per second per node for input processing and 22,300 tokens per second per node for output generation. The team reported this as the highest published throughput for DeepSeek-V3 at that time. The 3:9 ratio was not chosen arbitrarily — it reflects the measured observation that for this workload and model, the decode pool was the throughput bottleneck, and adding more decode capacity relative to prefill capacity maximized overall goodput.

Moonshot AI's Kimi K2 deployment in July 2025 used 128 H200 GPUs with Mooncake's Transfer Engine and InfiniBand, achieving 224,000 tokens per second of prefill throughput and 288,000 tokens per second of decode throughput. These numbers represent the current publicly known upper bound for disaggregated LLM serving throughput.

Beyond these three deployments, disaggregated serving has been confirmed in production at Meta (using vLLM), LinkedIn (using vLLM), Mistral (using vLLM), and HuggingFace (using vLLM). NVIDIA has committed to disaggregation as a first-class architectural primitive in Dynamo, its enterprise inference serving framework. Within eighteen months of the first academic publications on the topic, the technique has achieved broad production adoption across the industry.

---

## 12. The Decision Framework: When and How to Disaggregate

The preceding sections have developed the complete conceptual framework for PD disaggregation. This final section synthesizes the key decision criteria.

### 12.1 The Five Checks

The first question to answer is whether disaggregation is the right architecture at all, given the constraints of the specific deployment. Five checks provide the answer.

The first check is the decode-to-prefill time ratio. If less than 70 percent of wall-clock GPU time is spent in decode, the interference-elimination benefit is moderate and may not justify the architectural complexity. If more than 85 percent of GPU time is in decode, the tensor cores are sitting idle for most of each request's lifetime, and disaggregation allows those resources to be redirected to prefill capacity.

The second check is the SLO regime. Apply TaiChi's framework: is the binding constraint TTFT, TPOT, or both? A tight TTFT constraint with relaxed TPOT may be better served by aggregation. A tight TPOT constraint with relaxed TTFT is the canonical use case for disaggregation. Both constraints tight requires either a large enough disaggregated cluster that the prefill pool is not a TTFT bottleneck, or a hybrid architecture.

The third check is the transfer cost. Using the formula developed in section 7, calculate the KV cache size for the median prompt length. Divide by the available RDMA bandwidth (after running KVBench against the actual cluster) to determine the transfer time. Verify that this transfer time fits within the TTFT budget after accounting for prefill computation time. If the cluster lacks RDMA-capable NICs, the effective bandwidth drops substantially and the transfer time may consume too much of the TTFT budget to be viable.

The fourth check is cluster scale. Below approximately 16 total GPUs, the scheduling overhead and operational complexity of maintaining two pools typically outweigh the utilization gains. Above 32 GPUs under sustained load, the cost savings compound at scale and the case for disaggregation strengthens.

The fifth check is the prefix cache hit rate. If the majority of requests share a system prompt or conversation history, the decode workers may already hold most of the KV from previous turns in their local cache. Routing every request through a remote prefill pool in this scenario adds a network round-trip for data that is effectively local. A hybrid approach — running prefill locally on decode workers for high-cache-hit requests, and routing only cold requests through the prefill pool — may outperform full disaggregation.

### 12.2 The Cost Arithmetic

If all five checks are favorable, the cost arithmetic is straightforward. Disaggregation reduces per-token serving cost through two independent mechanisms. The first is interference elimination: more requests complete within SLO bounds, so the effective goodput at the same hardware budget increases, reducing cost per correctly-served request. The second is hardware heterogeneity: the decode pool can run on less expensive hardware (A100 instead of H100, for example) without quality loss, reducing the per-GPU cost of the cluster's largest pool. Splitwise's measurements suggest 20 percent cost reduction at equivalent throughput as the combined effect of these two mechanisms in a well-designed heterogeneous cluster. Infrastructure analysts have estimated 15 to 40 percent total cluster cost reduction at production scale.

The eighteen-month adoption arc from first publication to broad production deployment is, in itself, a form of evidence. Engineering organizations do not adopt architectural complexity without a compelling return. The fact that Perplexity, Meta, LinkedIn, Mistral, Moonshot AI, HuggingFace, and the SGLang team all independently reached the same conclusion — that disaggregation is worth deploying — is strong evidence that the cost arithmetic works in practice, not just in benchmarks.
