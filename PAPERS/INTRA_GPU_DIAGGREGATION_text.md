# **Nexus: Proactive Intra-GPU Disaggregation of Prefill and Decode in LLM Serving** 

## Xiaoxiang Shi[∗] 

lambda7xx@gmail.com Independent Researcher 

Junjia Du junjia001@e.ntu.edu.sg Nanyang Technological University 

## **Abstract** 

Current prefill–decode (PD) disaggregation is typically deployed at the level of entire serving engines[1] , assigning separate GPUs to handle prefill and decode phases. While effective at reducing latency, this approach demands more hardware. To improve GPU utilization, Chunked Prefill mixes prefill and decode requests within the same batch, but introduces phase interference between prefill and decode. 

While existing PD disaggregation solutions separate the phases across GPUs, we ask: can the same decoupling be achieved within a single serving engine? The key challenge lies in managing the conflicting resource requirements of prefill and decode when they share the same hardware. In this paper, we first show that chunked-prefill requests cause interference with decode requests due to their distinct requirements for GPU resource. Second, we find that GPU resource exhibits diminishing returns—beyond a saturation point, increasing GPU allocation yields negligible latency improvements. This insight enables us to split a single GPU’s resources and dynamically allocate them to prefill and decode on the fly, effectively disaggregating the two phases within the same GPU. 

Across a range of models and workloads, our system Nexus achieves up to 2.2× higher throughput, 20× lower TTFT, and 2.5× lower TBT than vLLM, and outperforms SGLang with up to 2× higher throughput, 2× lower TTFT, and 1.7× lower TBT, and achieves 1.4× higher throughput than vLLM-disaggregation with only half the number of GPUs. 

## **1 Introduction** 

Transformer-based [54] Large Language Models (LLMs) [4, 11, 18, 41] have achieved state-of-the-art performance on a wide range of tasks, from natural language understanding to code synthesis [2, 19, 20, 27, 29, 40, 51, 58, 59]. The success has also driven their integration into latency-sensitive applications such as chatbots [11, 18, 19, 40, 41, 51], search assistants [44], and AI-augmented IDEs [10]. In interactive 

> ∗Both authors contributed equally to this work. 

> 1We use the term _serving engine_ to denote a unit of GPUs that manages exactly one complete copy of the model weights. 

## Colin Cai[∗] 

cai9@berkeley.edu Independent Researcher 

## Zhihao Jia 

zhihao@cmu.edu Carnegie Mellon University 

**==> picture [117 x 32] intentionally omitted <==**

**==> picture [116 x 32] intentionally omitted <==**

|**(a)**Monolithic.<br>**(c)**Intra-engine disaggregation.|**(b)**PD Disaggregated.|**(b)**PD Disaggregated.|**(b)**PD Disaggregated.|**(b)**PD Disaggregated.|
|---|---|---|---|---|
||**System**|**TTFT**|**TBT**|**Util.**|
||Monolithic<br>Disagg.<br>Nexus|✗<br>✓<br>✓|✗<br>✓<br>✓|✓<br>✗<br>✓|
||**(d)**System characteristics.||||



**Figure 1. Design evolution of LLM inference systems.** Comparison between monolithic, disaggregated, and intra-engine disaggregated designs. Ap is the prefill phase of request A; Bd, Cd, and Dd are the decode phases of requests B, C, and D. 

settings, even small delays matter a lot: humans perceive latencies above one second as disruptive [67], and sub-second improvements has been shown to substantially boost engagement [16]. Therefore, latency has become a critical performance metric for LLM serving. 

LLM inference consists of two distinct stages with heterogeneous resource demands: _prefill_ and _decode_ . In the prefill stage, the model processes the entire prompt in a single forward pass to produce the first output token while populating the key-value (KV) cache. This stage is typically _computebound_ [42, 68], dominated by large matrix multiplications that fill GPU compute units. In contrast, the decode stage generates tokens one at a time, attending to all previously cached KV states. The decode stage, processing one token per request in each forward pass, features lightweight compute but requires reading the full model weights and the entire KV cache, making it heavily constrained by memory bandwidth [24, 25, 42, 68]. 

These two stages naturally give rise to two critical latency metrics in LLM serving: _time-to-first-token (TTFT)_ , the delay before the first output token is produced, determined by the prefill stage; and _Time-between-Tokens (TBT)_ , the latency between subsequent tokens, dictated by each iteration of the decode loop. These metrics directly impact the user experience in latency sensitive applications. Optimizing TTFT and TBT has driven a wave of innovations in LLM serving systems, spanning scheduling, batching, and kernel design [1, 15, 24, 25, 30–33, 39, 50, 56, 57, 61, 63, 64, 66, 67]. 

1 

Existing LLM serving systems fall into two broad classes based on how they place prefill and decode execution (Figure 1a, 1b). 

_Monolithic systems_ [1, 25, 30, 52, 63, 66, 68] execute both stages within a single engine. To improve utilization, recent designs like Sarathi-Serve [1] adopt _chunked prefill_ , where long prompts are split into shorter chunks and batched alongside decode tokens. This improves token throughput and reduces TBT by increasing batch efficiency. However, this design mixes compute heavy prefill and memory sensitive decode operations in the same batch, causing interference between the prefill and decode and increasing TBT (see Section 3.1 for details). 

_Disaggregated systems_ [23, 42, 45, 67] assign prefill and decode to separate engines, transferring KV cache data between them. This eliminates interference and achieves consistently low TTFT and TBT. But it sacrifices efficiency [15]: decode engines are often underutilized, and multi-GPU deployment incurs additional hardware and communication overhead, especially when KV state is large. 

This paper asks a simple question: _Can a single engine achieve low TTFT and TBT without sacrificing GPU utilization?_ We answer “yes” by introducing _intra-engine logical disaggregation_ , which separates prefill and decode execution within a single serving engine. This design is grounded in three observations. First, mixing compute heavy prefill and memor bound decode in chunked batches creates finegrained interference that increases TBT. Second, both stages exhibit diminishing returns beyond moderate compute allocations overprovisioning is wasteful, and equal partitioning is inefficient. Third, modern accelerators [34, 36] now provide sufficient on device memory and compute to support disaggregation within a single engine, avoiding the communication overheads of cross-GPU designs. 

However, enabling this form of disaggregation raises new challenges. Unlike traditional multi-engine systems, we must partition and schedule GPU resources _dynamically within a single serving engine_ , while minimizing contention and adapting to workload shifts in _sub-second timescales_ . Prefill and decode pressure evolve continuously with prompt length, KV cache footprint, and request mix, requiring fine-grained orchestration that scales. 

To address these challenges, we present _Nexus_ (Figure 1c), a monolithic LLM serving engine that achieves intra-engine prefill–decode disaggregation. Nexus is built on three components: **(1)** A lightweight analytical cost model predicts latency as a function of resource allocation, prompt length, and cache usage. **(2)** A greedy search algorithm consults this model to select low-latency resource partitions in real time, requiring only a few closed-form evaluations per update. **(3)** Two phase-specific schedulers, shortest-prompt-first for prefill and FCFS for decode, exploit their differing characteristics to optimize TTFT and TBT. 

**==> picture [242 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill Phase Decode Phase<br>good and<br>be  So<br>x N x N<br>FFN Layer  FFN Layer<br>Attention  Attention<br>Linear Linear<br>Attention Attention<br>Key Key<br>Value Value<br>KQV Linear KV Cache KQV Linear KV Cache<br>Paris is so good<br>LLM will be<br>**----- End of picture text -----**<br>


**Figure 2. Inference process of transformer-based LLMs.** Red boxes indicate compute-bound operations (KQV Linear, Prefill Attention, Attention Linear, and FFN Layer), while the orange box (Attention) represents a memory-bound operation. Auxiliary components such as LayerNorm are omitted for clarity. 

Together, these mechanisms allow Nexus to match the high utilization of monolithic designs while achieving the isolation benefits of disaggregated systems—without incurring cross-device transfers or relying on additional hardware. 

We implement Nexus by extending vLLM [25], adding finegrained resource partitioning and concurrent prefill–decode execution within a single engine. Nexus runs on commodity GPUs without kernel modification, supports decoder-only LLMs, and requires no specialized hardware. 

## **This paper makes the following contributions:** 

- We identify the limitations of existing solutions in serving LLMs and propose intra-engine PD disaggregation as a solution. 

- We develop a lightweight, adaptive scheduling mechanism that enables _sub-second resource repartitioning_ and phase-aware prioritization, making intra-engine PD disaggregation practical under real-world, dynamic workloads. 

- We implement Nexus as a drop-in vLLM extension and evaluate it on production-scale LLMs and traffic. Nexus achieves up to **2.2** × and **2** × higher throughput, **20** × and **2** × lower TTFT, and **2.5** × and **1.7** × lower TBT than vLLM and SGLang respectively and outperforms disaggregated vLLM with half the GPU resources. 

## **2 Background** 

## **2.1 Architecture of Transformer-based LLM** 

Most large language models (LLMs) [3, 5, 14] adopt a decoderonly Transformer architecture [54], consisting of a stack of layers that apply self-attention and feed-forward networks (FFNs). During inference, LLMs operate in two distinct phases: _prefill_ and _decode_ . Figure 2 illustrates the computation involved in each phase. The left panel depicts the prefill phase, where the prompt is processed to produce the first token and KV cache. The right panel shows the decode phase, where tokens are generated autoregressively using 

2 

the cached KV states. Each Transformer layer includes both attention-related operations and dense operations [68]. The latter consist of the linear projections for queries, keys, and values (Q/K/V), the output projection following attention, and the FFN sublayer. All these dense operations are compute bound [42, 68]. 

## **2.2 Dense Operations** 

**Q/K/V Projections.** Before attention, the input _𝑋_ is projected by weight matrices _𝑊𝑄_ , _𝑊𝐾_ , and _𝑊𝑉_ to produce the corresponding Query, Key, and Value tensors. _𝑊𝑄,𝑊𝐾,𝑊𝑉_ ∈ R _[𝑑]_[×] _[𝑑]_ where d is the hidden size. 

**==> picture [194 x 12] intentionally omitted <==**

Without caching, all _𝐿_ tokens are processed, leading to _𝑂_ ( _𝐿𝑑_[2] ) compute. With caching, only the _𝑛_ new tokens require projection, reducing the cost to _𝑂_ ( _𝑛𝑑_[2] ). 

**Attention Output Projection.** The attention output _𝐴_ ∈ R _[𝑛]_[×] _[𝑑]_ is projected using: 

**==> picture [171 x 12] intentionally omitted <==**

The cost is _𝑂_ ( _𝑛𝑑_[2] ), or _𝑂_ ( _𝐿𝑑_[2] ) if no caching is used. 

**Feed-Forward Network (FFN).** Each token is independently processed by a two-layer MLP with a non-linear activation (typically GELU): 

FFN( _𝑥_ ) = GELU( _𝑥𝑊_ 1 + _𝑏_ 1) _𝑊_ 2 + _𝑏_ 2 (3) where _𝑊_ 1 ∈ R _[𝑑]_[×] _[𝑑]_[ff] , _𝑊_ 2 ∈ R _[𝑑]_[ff][×] _[𝑑]_ , and typically _𝑑_ ff = 4 _𝑑_ . The cost is _𝑂_ ( _𝑛𝑑_ · _𝑑_ ff), or _𝑂_ ( _𝐿𝑑_ · _𝑑_ ff) in the absence of caching. Given the size of _𝑑_ ff, this is usually the most FLOP-intensive component of the layer. 

## **2.3 Attention Operation** 

Self-attention computes token-level dependencies by comparing queries with keys and applying attention weights to values. Given query _𝑄_ ∈ R _[𝑛]_[×] _[𝑑]_ , key _𝐾_ ∈ R _[𝐿]_[×] _[𝑑]_ , and value _𝑉_ ∈ R _[𝐿]_[×] _[𝑑]_ , attention is computed as: 

**==> picture [209 x 25] intentionally omitted <==**

The attention computation has two main components: - Computing similarity scores _𝑆_ ∈ R _[𝑛]_[×] _[𝐿]_ : cost _𝑂_ ( _𝑛𝐿𝑑_ ), - Applying softmax and aggregating over values: cost _𝑂_ ( _𝑛𝐿𝑑_ ). 

Thus, the overall attention complexity is _𝑂_ ( _𝑛𝐿𝑑_ ). In the absence of KV caching, this becomes _𝑂_ ( _𝐿_[2] _𝑑_ ). 

**Prefill vs. Decode Attention.** Assuming cache is enabled, the difference between prefill and decode arises from the size of _𝑛_ . In the _prefill phase_ , a chunk of _𝑛_ new tokens is processed in parallel. The attention computation involves matrix-matrix multiplications and has cost _𝑂_ ( _𝑛𝐿𝑑_ ), which is compute-bound [68] and benefits from parallelism. In the _decode phase_ , only one new token is processed at a time 

**==> picture [242 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
Kernel A Kernel B Kernel BKernel C<br>Block 0 Block 0 Block 2 Block 4 Block 0 Block 2<br>Block 1 Block 1 Block 3 Block 5 Block 1 Block 3<br>Block<br>Scheduler SM Exec.Exec. Exec.Exec. SM Exec.Exec. Exec.Exec. SM Exec.Exec. Exec.Exec.<br>L1 Cache / Shared Mem L1 Cache / Shared Mem L1 Cache / Shared Mem<br>L2 Cache<br>GPU<br>DRAM<br>**----- End of picture text -----**<br>


**Figure 3. Simplified GPU execution model.** Modern GPUs share a global kernel queue, with SMs (streaming multiprocessors) dynamically fetching kernels to execute. Concurrently executing kernels compete for shared memory bandwidth. 

( _𝑛_ = 1), resulting in a matrix-vector multiplication (GEMV) with cost _𝑂_ ( _𝐿𝑑_ ). While the FLOP count is low, this operation is memory-bound [68] due to repeated access to the model weights and the growing KV cache. 

These contrasting patterns lead to different system bottlenecks [24, 25]: prefill benefits from parallelism, while decode demands cache-efficient execution. 

## **2.4 Batching in LLM Serving** 

Modern LLM serving systems must coordinate prefill and decode phases with fundamentally different latency objectives. Prefill performance directly determines TTFT, while decode responsiveness governs TBT. Since both phases contend for GPU resources. Existing LLM serving systems fall into two broad categories: Monolithic Systems and PD Disaggregation systems, depending on how they manage prefill and decode execution. 

_**Monolithic System.**_ Monolithic system [25, 66] such as Sarathi-Serve [1](Figure 1a) splits long prompts into fixedsize chunks and mixing them with decode tokens in a shared batch queue, these systems construct mixed-phase batches that improve GPU utilization and reduce head-of-line blocking. 

_**PD Disaggregation.**_ To isolate prefill and decode execution, some systems adopt prefill–decode disaggregation [23, 42, 45, 67](Figure 1b), assigning each phase to separate serving engine. This strategy enables independent scheduling and eliminates interference between prefill and decode, particularly in multi-GPU environments. 

## **2.5 GPU Execution Model** 

Modern LLM inference relies heavily on GPU acceleration. Figure 3 presents a simplified view of GPU execution, abstracting low-level scheduling mechanisms in favor of architectural components most relevant to LLM serving workloads. 

GPUs consist of multiple Streaming Multiprocessors (SMs), each with its own block executor, L1 cache, and register file, while sharing a unified L2 cache and off-chip DRAM. Kernels are submitted via a global software queue and consists of 

3 

||<br>Prefll-only<br>0.132<br>2<br>0.02%<br>Decode-only<br>0.015<br>563<br>5.80%<br>Mixed<br>0.251<br>9150<br>94.18%<br>**(a)**Statistics by Batch Types<br>**(b)**Latency breakdown by kernel.|
|---|---|



**Figure 4. Latency impact of mixed prefill–decode batches.** (a) Prefill-only and decode-only batches have predictable latency, but mixed batches cause 8×–10× slowdown due to interference. (b) Kernel-level profiling reveals that even lightweight decode kernels experience inflated runtimes when co-executed with prefill. This highlights the fine-grained contention caused by chunked batching. 

many blocks. The blocks carry operations and are scheduled onto available executors in SMs by a hardware-level scheduler, which is largely opaque to software and does not provide preemptive control or fine-grained prioritization. 

## **3 Motivation** 

Modern LLM serving must balance two asymmetric phases: compute-heavy prefill and memory-bound decode. In the following section, we examine limitations of current systems that mix or isolate these phases (§3.1), explore how both exhibit diminishing returns with added resources (§3.2), and highlight the inefficiency of static resource partitioning under memory bandwidth contention (§3.3). Together, these insights motivate intra-engine disaggregation with GPU resourcereallocation on the fly. 

**Setup.** Unless stated otherwise, we evaluate with Long Data Collections Workload (§6.1) using Qwen2.5–3B on a single NVIDIA L20 GPU. Requests follow a Poisson arrival at 2.5 req/s. We use NVIDIA MPS [38] to control SM partitioning. 

## **3.1 Limitation of Existing Solutions** 

Existing LLM serving systems can fall into two architectural categories: _monolithic execution_ and _disaggregation_ . 

**Disaggregated systems** [42, 45, 67] places prefill and decode to separate engines, eliminating resource contention and stabilizing latency. However, the clean separation come with steep hardware costs: multiple engines must maintain the full model replica, prefill engine’s memory is wasted, and decode often underutilize their assigned GPUs. Worse, coordination is non-trivial. Under dynamic workloads, KV cache eviction and recomputation have been shown to severely inflate both TTFT and TBT [15]. 

**Monolithic systems** [1, 25, 63, 66] adopt chunked prefill, batching prefill and decode requests to improve utilization. Although the strategy increases throughput, it does not account for the distinct compute and memory behavior of each 

**==> picture [241 x 59] intentionally omitted <==**

**----- Start of picture text -----**<br>
Normalized P&D Latency Prefill Breakdown Decode Breakdown<br>3 Prefill Decode 40 KQV Attn Score Attn Linear 1.5<br>2 Ideal (1 / ratio) 20 FFN (FFN) 1.0<br>0.5<br>1 0.4 0.6 0.8 1.0 0 0.4 0.6 0.8 1.0 0.4 0.6 0.8 1.0<br>SM Ratio SM Ratio SM Ratio<br>(a)  Normalized latency. (b)  Prefill breakdown. (c)  Decode breakdown.<br>Norm. Lat.<br>Latency (ms) Latency (ms)<br>**----- End of picture text -----**<br>


**Figure 5. Diminishing returns in prefill and decode with increasing SM allocation.** (a) End-to-end latency for prefill and decode flattens well before full SM usage. (b) Prefill kernels (e.g., FFN, KQV, attention linear) show varied sensitivity to SM scaling, with FFN benefiting the most. (c) Decode kernels saturate quickly, confirming that decode is memory-bound and gains little from additional compute. 

phase, causing phase interference. To quantify this, we categorize batches into _prefill-only_ , _decode-only_ , and _mixed_ , and measure their iteration times (Figure 4a). Despite having similar token counts, mixed batches average 250 ms, compared to just 15 ms for decode-only batches. 

This slowdown arises from full prefill computation (e.g., KQV projection, FFN) blocking lightweight decode kernels in the same batch. As shown in Figure 4b, linear kernel latency in mixed batches is up to 10× higher than in decode-only batches. Since decode cannot proceed until all prefill kernels are completed, this inflates TBT by more than 8×. 

**Insight 1.** Chunked prefill improves utilization but introduces fine-grained interference within batches. PD disaggregation avoids this but wastes resources. We aim to achieve both isolation and efficiency within a single engine. 

## **3.2 Diminishing Returns in Compute Allocation** 

While the Section 3.1 highlights the limitations of both chunked prefill and inter-engine PD disaggregation, we take a step further by exploring an intra-engine PD approach through GPU resource sharing. In this design, prefill and decode are assigned different portions of the same GPU. To allocate GPU resources effectively under prefill–decode separation, we analyze how each phase scales with compute in isolation. We run pure prefill and decode batches under varying SM ratios, measuring both end-to-end latency and per-kernel runtimes. 

As shown in Figure 5, prefill latency closely follows the 1 idealized scaling model _𝑇_ ∝ _𝑟_[,][with][diminishing][returns] emerging gradually. For instance, increasing SM allocation from 30% to 40% reduces latency by over 25%, but the gain drops to just 10% between 70% and 80%. Further investigation in Figure 5b shows that compute-heavy layers such as FFN continue to benefit, while others such as KQV and projection flatten out earlier, around 60%. 

Decode exhibits much sharper diminishing returns. Increasing SMs from 30% to 40% improves latency by only 10%, and beyond 50%, additional SMs yield less than 3% improvement per 10% increment. This behavior is expected given 

4 

**==> picture [242 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
Prefill KV Len. Over Time<br>Decode Lat. Over Prefill KV Len. 10000<br>0.150 8000<br>0.145 6000<br>0.140<br>4000<br>0.135<br>0.130 2000<br>0<br>1000 4000 7000 10000 0 500 1000 1500 2000<br>Prefill KV Cache Length Duration(s)<br>(a)  Impact of prefill KV length on (b)  Observed variation in prefill<br>decode latency. KV length during execution.<br>Latency (ms) Prefill KV Len.<br>**----- End of picture text -----**<br>


**Figure 6. Memory contention’s impact and variability.** (a) Decode latency increases as prefill KV length grows due to shared memory bandwidth pressure; (b) Prefill KV length fluctuates significantly over time, making contention difficult to predict statically. 

decode is memory-bound and fails to utilize added SMs effectively. Figure 5c confirms this: attention and projection layers show minimal runtime reduction with more compute. 

These trends suggest a key systems insight: instead of time-slicing all SMs between the two stages, we can spatially partition them to avoid overprovisioning either phase. Notably, disaggregated systems, which allocate entire GPUs to each phase, operate on the far right of these curves, where additional compute offers diminishing marginal benefit. 

**Insight 2.** Both prefill and decode saturate well before full GPU allocation. To improve efficiency, systems should allocate only the SMs needed to meet each phase’s demand. 

## **3.3 Limitation of Static Partitioning** 

Section 3.2 shows that prefill and decode phases exhibit sharply diminishing returns with increasing SM allocation, motivating finer-grained partitioning. However, even a wellchosen _static_ SM partition is suboptimal at runtime. This is because compute demand, while often estimable from inputs like chunk size or sequence length, does not capture dynamic memory behavior. Prefill and decode contend for bandwidth in ways that depend on evolving KV cache sizes, prompt distributions, and decode lengths that emerge only at runtime. 

To illustrate this, we co-execute chunks of prefill[2] and pure decodes under a fixed SM partition. As we can see from Figure 6a, increasing the prefill KV length from 2000 to 10000 also increases the latency of the exact same decode batch by 36%. This slowdown stems from memory bandwidth contention: prefill attention layers perform large KV reads, which overlap with the decode stage’s latency-critical memory access. 

Moreover, Figure 6b shows that prefill memory traffic is highly irregular, fluctuating significantly over time and making contention both difficult to predict and workload dependent. 

> 2Following Sarathi-Serve [1], long prompts are split into smaller segments and executed alongside decode requests. These chunks of prefill tasks read and write to the KV cache. 

**==> picture [241 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
Controller GPU worker<br>Greedy Search Rp  Rd Scheduler ExecutorPrefill<br>Cost Model Decode<br>Executor<br>Serving Engine<br>PT  KVp  Bd   KVd<br>**----- End of picture text -----**<br>


**Figure 7. System architecture of Nexus.** _𝑅𝑝_ and _𝑅𝑑_ denote the SM ratios allocated to prefill and decode, respectively. _𝑃𝑇_ represents the chunk prompt length of prefill, _𝐾𝑉𝑝_ is the total KV cache used during prefill, _𝐵𝑑_ is the decode batch size, and _𝐾𝑉𝑑_ is the total KV cache used during decoding. 

These highlight a core limitation: even when compute demands are static and analyzable, bandwidth pressure varies at runtime due to asynchronous prefill/decode evolution. Static SM partitioning fails to respond to this variation, leading to avoidable contention and degraded latency. 

**Insight 3.** Logical PD disaggregation with static partitioning is insufficient. To adapt to emergent memory contention and shifting runtime demands, systems must have fine-grained, dynamic SM reallocation. 

## **4 Design** 

We present Nexus, a lightweight monolithic LLM serving system that enables _intra-engine prefill–decode disaggregation_ . Unlike prior systems that either co-batch prefill and decode or isolate them across GPUs, Nexus partitions GPU resources dynamically, executing both phases concurrently but independently within a single engine. 

As shown in Figure 7, Nexus introduces three core mechanisms to enable this fine-grained separation: 

- **Dynamic SM Partitioning** (§4.1): A runtime cost model estimates per-phase iteration latency based on compute scaling and memory contention. We formalize a dualobjective optimization problem and solve it efficiently via a greedy search. 

- **Stability Control** (§4.2): To reduce the overhead of frequent re-partitioning, we apply a hysteresis-style buffer zone that filters out insignificant SM ratio changes. 

- **Phase-Specific Scheduling** (§4.3): With prefill and decode isolated, we deploy customized schedulers to optimize TTFT and TBT jointly. 

These components form a closed-loop control system that continuously adapts GPU resource allocation to match workload demands, preserving high utilization without reintroducing interference. 

## **4.1 Dynamic SM Partitioning via Cost Model** 

To execute prefill and decode concurrently without introducing significant interference or wasting resources, Nexus must decide at runtime how to split GPU computes between 

5 

the two phases. This is challenging: exhaustive search is too slow for inference loops, and simple rules fail under dynamic memory contention. 

To solve this, Nexus combines three components: (1) an analytical cost model that predicts latency under any SM split, (2) a dual-mode optimization objective that shifts focus based on runtime signals, and (3) a greedy search algorithm that efficiently selects SM partitions with just a few cost model queries. 

**4.1.1 Cost Model.** Nexus’s cost model estimates the latency of prefill and decode under SM allocations without execution, enabling rapid exploration of latency tradeoffs. 

Each iteration consists of multiple operators with different bottlenecks. For example FFNs are compute-bound [68], while adecode ttention may be memory-bound [68] as KV grows. These characteristics shift with workload, so we model each phase’s latency as a sum over operators: 

**==> picture [205 x 56] intentionally omitted <==**

This operator-level modeling captures shifting bottlenecks, such as decode attention flipping between compute- and memory-bound, without collapsing structure as coarse stagelevel models would. 

**Compute Latency.** We estimate the compute latency of each operator _𝑜_ ∈ PrefillOps ∪ DecodeOps based on its FLOP count _𝑐𝑜_ and the SM ratio _𝑟_ assigned to its stage. While latency ideally scales inversely with compute share (1/ _𝑟_ ), realworld performance can deviates depending on kernel(§3.2). 

To model this, we use a two-regime saturation–decay curve 

- _Sub-saturation:_ Latency scales near-inversely with _𝑟_ until a saturation threshold _𝑅_ sat; 

- _Post-saturation:_ Additional SMs yield diminishing returns, modeled by a decay coefficient _𝜆_ . 

**==> picture [236 x 38] intentionally omitted <==**

**==> picture [11 x 10] intentionally omitted <==**

where _𝐶_ is the peak throughput of the GPU. 

We extract _𝑅_ sat and _𝜆_ per operator from end-to-end measurements of the full stage (prefill or decode) under varying SM allocations. 

**Memory Access Latency.** As shown in §3.3, memory contention can significantly affect latency when prefill and decode execute concurrently on shared hardware. To capture this effect, we model decode memory latency as a function of (1) temporal overlap with prefill, and (2) the relative memory bandwidth demands of each phase. 

Let _𝑇_ prefill denote total prefill duration, and _𝑇_ prefill[attn][be the] estimated time on memory-bound attention layers. Then the probability that decode overlaps with prefill attention is: 

**==> picture [149 x 29] intentionally omitted <==**

We conservatively assume that prefill’s dense layers consume memory bandwidth during the remaining time, yielding: 

**==> picture [67 x 10] intentionally omitted <==**

Assuming full bandwidth saturation during each overlap window, we allocate effective bandwidth to decode based on its share of memory traffic. Let _𝑚𝑑_ be the total memory bytes accessed by decode attention, _𝑚𝑝_ 1 the bytes accessed by prefill attention, and _𝑚𝑝_ 2 those accessed by prefill’s dense operators.[3] Then the effective bandwidth for decode is: 

**==> picture [235 x 61] intentionally omitted <==**

This formulation captures two important dynamics: (1) contention grows with total memory traffic, reducing effective bandwidth; (2) allocating more SMs to decode slows prefill (via compute contention), stretching _𝑇_ prefill, which reduces _𝑃_ attn and mitigates contention. This feedback is integrated into the overall cost model and guides SM partitioning decisions. 

While decode involves multiple operators, we model memory contention only for attention, which dominates bandwidth usage. Other components are lightweight or computebound, and do not significantly impact contention. For prefill, we estimate memory latency assuming peak bandwidth, and use the resulting memory-bound segments only to compute _𝑃_ attn. This separation avoids circular dependencies while capturing the dominant interaction between phases. 

**4.1.2 Optimization Objective.** Given the cost model, Nexus selects an SM partition that balances performance and memory pressure. However, since prefill and decode run concurrently and compete for resources, optimizing both simultaneously is infeasible. 

To resolve this, Nexus formulates a _dual-objective latency optimization_ : prioritize one phase while constraining the other to remain within a slowdown budget. This allows flexible tradeoffs between TBT and TTFT, depending on workload state. The choice is also guided by runtime signals, such as KV cache usage, to avoid pathological behaviors like OOM. 

3We estimate memory access volume for linear layers using known parameter sizes from the model architecture. For attention, we calculate KV memory traffic based on the number of cached tokens tracked by vLLM, multiplied by the per-token key/value size determined from model dimensions. 

6 

**Formulation.** Let _𝑇_ prefill( _𝑅𝑝_ ) and _𝑇_ decode ( _𝑅𝑑_ ) denote the estimated latencies under a given SM split _𝑅𝑝_ and _𝑅𝑑_ = 1 − _𝑅𝑝_ . We define two optimization modes: 

**==> picture [98 x 11] intentionally omitted <==**

**==> picture [115 x 50] intentionally omitted <==**

**==> picture [93 x 10] intentionally omitted <==**

**==> picture [136 x 49] intentionally omitted <==**

Here, _𝑇_[min] denotes the ideal latency when a stage is allocated all SMs, and _𝛼, 𝛽 >_ 1 are slack variables controlling tolerable slowdowns in the non-prioritized stage. 

**Runtime Switching.** We select the objective mode based on live KV cache usage _𝐾𝑉𝑢_ : when _𝐾𝑉𝑢_ is low, favoring prefill accelerates prompt ingestion; when _𝐾𝑉𝑢_ is high, prioritizing decode helps reduce memory pressure by completing generations and evicting KV. This feedback mechanism enables resource allocation to respond to workload phases and memory constraints. 

**==> picture [238 x 31] intentionally omitted <==**

This mode-switching behavior is key to handling dynamic workloads. Rather than rely on fixed priorities or static thresholds, Nexus adapts its scheduling objective based on system state, enabling both high throughput and memory safety. 

**4.1.3 Greedy SM Search.** Since prefill and decode iterations run in sub-second times, we do not attempt to globally solve the constrained optimization; instead, we use a twophase greedy adjustment that is fast, robust, and effective in practice. 

**Phase 1: Constraint Satisfaction.** Starting from the current allocation, the algorithm reduces the SM share of the prioritized stage until the non-prioritized stage’s latency constraint is satisfied (lines 21–23 in Algorithm 1). 

**Phase 2: Target Optimization.** Once within the feasible region, the algorithm gradually increases the SM share of the prioritized stage, improving its latency as long as the constraint remains satisfied (lines 24–30). 

**Efficiency.** The search typically converges within 2–4 cost model evaluations and imposes negligible latency overhead. This fast feedback loop allows Nexus to adapt to runtime contention and shifting workloads. 

**Algorithm 1** Nexus’s SM Partitioning with Greedy Search and Bufer Control 

**==> picture [238 x 154] intentionally omitted <==**

- 14 **end procedure** 

- 15 **procedure** AdjustPartition( _target_ , _𝑅𝑝_[cur] _[, 𝑅] 𝑑_[cur][)] 16 Let _other_ ← (target is prefill? decode : prefill) 

- 17 Let Slack ← (target is prefill? _𝛽_ : _𝛼_ ) 18 _𝑇_ other[opt][←][CostModel][(] _[other][,]_[ 100][)] 19 _𝑅_ cur ← (target is prefill? _𝑅𝑝_[cur] : _𝑅𝑑_[cur][)] 20 _𝑅_ ← _𝑅_ cur 

- _⊲_ Phase 1: Decrease until constraint is satisfied 

- 21 **while** CostModel( _other,_ 100 − _𝑅_ ) _>_ Slack · _𝑇_ other[opt] **[do]** 22 _𝑅_ ← _𝑅_ − 1 23 **end while** 

- _⊲_ Phase 2: Increase target share until constraint is at limit 

- 24 **while** _𝑅 <_ 100 **do** 

- 25 _𝑇_ other ← CostModel( _other,_ 100 −( _𝑅_ + 1)) 26 **if** _𝑇_ other _>_ Slack · _𝑇_ other[opt] **[then]** 27 **break** 28 **end if** 29 _𝑅_ ← _𝑅_ + 1 

- 30 **end while** 

- 31 **return** (target is prefill? ( _𝑅,_ 100 − _𝑅_ ) : (100 − _𝑅, 𝑅_ )) 

- 32 **end procedure** 

_Note:_ CostModel(phase _, 𝑅_ ) estimates phase latency under _𝑅_ ; all ratios are expressed as percentages of total SMs; _𝛽_ is tolerance for primary objective’s deviation from optimal; _𝛿_ is the buffer to avoid frequent switches. 

## **4.2 Hysteresis-Based SM Repartitioning for Stability** 

While selecting the optimal SM split is critical for reducing prefill–decode interference, the timing and stability of repartitioning are equally important. Green Contexts provide logical SM isolation, but switching partition states is not free: transitions can introduce temporary underutilization, overcommitment, or execution stalls. 

**Pitfall 1: Synchronous switching.** A natural design is to synchronize partition changes at global checkpoints. However, as shown in Figure 8a, this introduces idle “bubbles,” where some streams stall waiting for others to reach the switch point. It hurts SM utilization and inflates latency. 

7 

**==> picture [241 x 44] intentionally omitted <==**

**==> picture [241 x 65] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a)  Synchronous solution.<br>(b)  Naive asynchronous solution.<br>**----- End of picture text -----**<br>


**==> picture [241 x 44] intentionally omitted <==**

**(c)** Nexus combines asynchronous switching and hysteresis. 

**Figure 8. Mechanisms for SM partition switching.** Comparison between synchronous, asynchronous, and our asynchronous with hysteresis approach. 

**Pitfall 2: Naive asynchronous switching.** Letting streams switch independently avoids global stalls, but creates new problems: SM oversubscription or underutilization depending on timing (Figure 8b). This makes system sensitive to transient workload shifts, causing back-and-forth toggling and instability. 

**Our design: Buffered asynchronous switching.** To mitigate these issues, Nexus adopts a buffered asynchronous switching policy. The runtime controller tracks the lastapplied SM ratio and only triggers repartitioning when the new target differs by more than a threshold _𝛿_ . This hysteresisstyle buffer (Figure 8c) smooths out transient fluctuations and suppresses excessive reconfiguration. The logic is embedded in Algorithm 1, line 9-13. 

This simple mechanism retains the adaptability of finegrained decisions while avoiding the instability of overreactive switching. Alternative approaches like reducing update frequency or increasing step size proved brittle in practice: the former sacrifices responsiveness; the latter risks overshooting optimal ratios. Buffered switching provides a robust, low-overhead tradeoff. 

## **4.3 Phase-Specific Scheduler** 

Since Nexus separates prefill and decode into concurrent batches, we can further exploit this with phase-specific optimizations. Particularily, we employee tailored scheduler policies that address the distinct latency and resource profiles. 

**4.3.1 Prefill Scheduler.** TTFT is governed by the prefill stage, which must complete a full forward pass before emitting output. When scheduling prompt requests of varying lengths, naïve policies can introduce head-of-line (HoL) blocking: short prompts are delayed by long ones. This effect significantly impacts latency-sensitive workloads and motivates length-aware scheduling. 

**Algorithm 2** Nexus’s Shortest Prompt First (SPF) Scheduler 

|1|**Input:**Request queue_𝑄_, batch limit_𝐵_, age|decay factor_𝛼_|
|---|---|---|
|2|**Output:**The next prefll batch to run||
|3 <br>4|**procedure**SPF_Schedule(**Queue**_𝑄_,**Int**_𝐵_,**Float**_𝛼_)<br>**for all**_𝑟_∈_𝑄_**do**||
|5|_𝑟._remaining←_𝑟._prompt_len−_𝑟._preflled_len||
|6|_𝑟._score←_𝑟._remaining−_𝛾_·_𝑟._age|_⊲_Antistarvation|
|7|**end for**||
|8<br>9|_𝑄_sorted ←SortBy(_𝑄,𝑟_→_𝑟._score)<br>_𝑏𝑎𝑡𝑐ℎ_←[ ]||
|10|_𝑡𝑜𝑡𝑎𝑙_←0||
|11|**for**_𝑟_∈_𝑄_sorted **do**||
|12|**if**_𝑡𝑜𝑡𝑎𝑙_+_𝑟._remaining ≤_𝐵_**then**||
|13|Append(_𝑏𝑎𝑡𝑐ℎ,𝑟_)||
|14|_𝑡𝑜𝑡𝑎𝑙_←_𝑡𝑜𝑡𝑎𝑙_+_𝑟._remaining||
|15|**else**||
|16|**break**||
|17|**end if**||
|18<br>19|**end for**<br>**return**batch||
|20|**end procedure**||



At each scheduling tick, the system selects a subset of pending requests whose combined prompt lengths fit within a token budget. Decoupling prefill from decode enables phasespecific scheduling, allowing us to optimize directly for TTFT. 

The prompt of each request is known. Hence, we introduce a greedy Shortest Prompt First (SPF) heuristic that ranks requests by an age-adjusted score: 

**==> picture [175 x 11] intentionally omitted <==**

where _𝑙𝑖_ is the prompt length, _𝑎𝑖_ is arrival time, _𝑡_ is the current time, and _𝛾_ controls the anti-starvation trade-off between responsiveness (low _𝛾_ ) and fairness (high _𝛾_ ). 

Requests are sorted by score and added greedily until the cumulative token limit is reached. This favors short prompts while gradually promoting delayed long requests, achieving a practical balance between latency and fairness. The full procedure is shown in Algorithm 2. 

While SPF is simple in design, its deployment is facilitated by our system’s phase isolation: in monolithic schedulers, prefill prioritization is harder to isolate and tune due to shared queues and coupled resource contention. In §6.5, we show that SPF significantly reduces TTFT compared to FCFS and helps offset performance regressions when SMs are reallocated to decode under memory pressure. 

**4.3.2 Decode Scheduler.** The decode phase controls TBT. Unlike prefill, decode scheduling operates at a finer granularity, with each request contributing only a single token to the active batch. Thus, we adopt a simple First-Come-FirstServe (FCFS) policy. FCFS ensures fairness, incurs minimal overhead, and avoids token-level starvation. While more sophisticated schemes that considers context window size or memory bandwidth are theoretically possible, we find that 

8 

|**Dataset**||**Mean**|**P50**|**P95**|**P99**|
|---|---|---|---|---|---|
|**Long Data Collections**|In<br>Out|5905<br>180|5461<br>159|9292<br>339|9817<br>454|
||In|3832|3575|6460|6894|
|**ArXiv Summarization**|Out|200|181|357|443|
|**ShareGPT**|In<br>Out|496<br>97|432<br>37|970<br>383|1367<br>474|



**Table 1. Characteristics of Workloads.** Distributions of input and output lengths of various datasets from different serving scenarios. 

their impact is limited in practice as they are considered in SM partitioning. 

## **5 Implementation** 

We implement Nexus on top of vLLM v1-0.8.1 [55], modifying approximately 6K lines of Python and CUDA/C++ code. Our changes enable phase-separated execution within a single engine or a single GPU, with minimal disruption to the original engine architecture. The maximum batch size and chunk size for prefill of Nexus are same as those of vLLM.. 

**Concurrent Execution.** We launch prefill and decode phases as separate coroutines, each managing its own GPU stream and scheduler. The main loop coordinates their execution and handles shared metadata updates. Because both phases share worker threads, we guard critical state to prevent inconsistencies during overlapping execution. 

**Per-Phase Scheduler.** We extend vLLM’s unified scheduler with pluggable logic for phase-specific algorithms. Prefill and decode queues are maintained independently. Both schedulers are configured to use the vLLM’s default config. The SPT implementation has the default _𝛾_ set to 15. 

**SM Partitioning and Runtime Switching.** We use CUDA Green Context [35] to partition the SM. Since CUDA Green Context does not provide a Python API, we implement a PyTorch extension using approximately 150 lines of CUDA code to expose this functionality to Python. We leverage this extension to dynamically reassign SM groups at runtime. To avoid reconfiguration overhead, Nexus pre-instantiates all partition layouts during initialization and switches among them as decided by the algorithm. The decaying _𝜆_ for each operator in cost model (§4.1) is obtained by profiling prefill and decode offline, and is done for each model and workload configuration. Since SPT scheduler heavily optimizes TTFT, we have a tight 1.1 _𝛽_ slack for decode, and 1.3 _𝛼_ slack for prefill. The _𝐾𝑉_ switch threshold is set to be 70% of all available KV cache memory. 

design choices of Nexus and show the effectiveness of each component. 

## **6.1 Experimental Setup** 

**Testbed.** Our evaluations are run on a workstation with Intel Xeon Platinum 8457C CPU (45 cores), two NVIDIA-L20 GPUs with 48GB DDR6 RAM each, and 200GB of CPU memory. The GPUs use driver 570.124.04 together with CUDA 12.8. All benchmarks were executed under PyTorch-2.6.0. 

**Model.** We use Qwen2.5–3B and LLaMA3.1–8B for singleGPU experiments, and Qwen2.5–14B for dual-GPU setups. These popular open-source LLMs span a range of KV cache sizes and compute intensities, enabling evaluation under diverse resource pressures. 

**Workloads.** We construct three workloads to emulate realworld LLM serving, combining datasets with diverse usage patterns and token length characteristics (Table 1). Similar to prior work [25, 67], the arrival pattern of requests is generated by a Poisson process. 

- **Long Data Collections [53]** : It mixes multi-turn QA and summarization, characterized by long prefill lengths and moderate decode demands. Evaluated on Qwen2.5–3B. 

- **Arxiv [6]** : It uses ArXiv Summarization [6] (full paper and abstract pairs) to model long-input, short-output tasks with stable token patterns. Evaluated on Qwen2.5–3B. 

- **Mixed** : It Combines 60% ShareGPT [46] (short, interactive prompts) and 40% Long Data Collections to induce token length and KV cache variability, stressing scheduling and memory. Evaluated on LLaMA3.1–8B and Qwen2.5–14B. 

**Metrics.** We report the mean and 95th percentile of three latency metrics: TTFT, TBT, and Normalized Latency. Normalized Latency is defined as the end-to-end latency divided by the number of output tokens, reflecting per-token serving efficiency across variable-length requests. A well-optimized serving system should maintain low normalized latency under high request loads. 

**Baselines.** We compare Nexus against four representative LLM serving engines, all configured with the same tensor parallelism, chunked prefill, and scheduling budget for fair comparison. 

- **vLLM (v1.0.8.1).** A throughput-optimized serving engine with FCFS scheduling, continuous batching [62], Page Attention, and chunked prefill [1]. 

- **FastServe (v0.0.8.1).** Implements a multi-level feedback queue with skip-join to resolve head-of-line blocking [56]. We reimplement it atop vLLM due to lack of public code and enable CPU swap (120GB) for each device with recomputation fallback. 

## **6 Evaluation** 

In this section, we first examine the end-to-end performance of Nexus under various workloads. Then, we evaluate the 

9 

- **SGLang (v0.4.4.post1)**[4] **.** A latency-optimized engine using Radix Attention for KV reuse [66]. Supports chunked prefill, Page Attention, and FCFS scheduling. 

- **vLLM-P/D (v1.0.8.5).** An extension of vLLM with prefill–decode disaggregation via LMCache [7, 28, 60]. We evaluate one prefill and one decode instance on separate GPUs to model single-layer PD setups. 

## **6.2 End-to-end Performance** 

We first evaluate end-to-end performance on a single GPU using three workloads with Qwen2.5-3B and Llama3-1-8B (Section 6.2.1). All systems use one L20 GPU, except DistvLLM, which uses two. Then, we report multi-GPU results (Section 6.2.2). 

**6.2.1 End-to-End Single-GPU Performance.** Figure 9 reports the end-to-end single GPU performance of Nexus and all baselines across three workloads (top to bottom). Figure 9 yields three key conclusions, which we discuss in detail below. 

**TTFT.** As shown in Figure 9 (columns 3–4), Nexus achieves the lowest or near-lowest average TTFT across all workloads. It improves TTFT by 2–20× over vLLM and up to 1.6× over SGLang through SPF scheduling and dynamic SM reallocating. The latter reallocates GPU resources at runtime to reduce prefill–decode contention, further amplifying SPF’s benefits. vLLM and SGLang suffer from head-of-line blocking under FCFS, though SGLang fares better due to its optimized runtime. FastServe reduces average TTFT via skip-join MLFQ, but hurts P95 due to deprioritizing long prompts. Compared to vLLM-P/D, which avoids contention by separating phases across GPUs, Nexus matches its TTFT in Mixed Workload and remains within 10% on Long Data Collections and Arxiv while using a single GPU. 

To ensure fairness, Nexus includes a tunable anti-starvation mechanism within its prefill scheduler. Under current settings, it improves P95 TTFT by 2–3× over vanilla vLLM and narrows the gap with SGLang and vLLM-P/D in Long Data Collections and Arxiv Workloads, while maintaining consistent advantages in average latency. In Mixed Workload, Nexus shows worse tail TTFT due to high prompt length diversity, which increases batching variability and makes it harder to protect long requests without hurting throughput. 

**TBT.** TBT reflects the responsiveness of steady-state decoding, and is particularly sensitive to memory bandwidth and scheduling efficiency. As we can see from Figure 9 (columns 5–6), vLLM-P/D achieves the best average and P95 TBT by fully separating prefill and decode onto dedicated GPUs. Among single-GPU systems, Nexus consistently ranks at or near the top across all workloads. 

4For fairness, we select the same evaluation timepoint as vLLM for SGLang; the corresponding commit is 3c09548. 

FastServe degrades sharply under load as it needs to fall back on recomputation. vLLM, which co-schedules prefill and decode, suffers from intra-batch interference, trailing Nexus by 1.24×–1.48×. SGLang improves over vLLM via Radix Attention and runtime optimizations, closely matching Nexus in Long Data Collections Workload, slightly surpassing in Arxiv Workload, but falling behind in Mixed Workload where prompt diversity intensifies decode imbalance. 

While Nexus lags in P95 TBT on Arxiv Workload, this anomaly stems from the workload’s relatively uniform input lengths and moderate output size (Table 1), which result in minimal memory pressure. As a consequence, the system’s dynamic SM partitioning is less likely to trigger aggressive decode resource shifts for few tailed requests. 

**Throughput.** To summarize end-to-end performance, we also measure maximum sustainable throughput as the highest arrival rate that each system can handle without violating token latency constraints. 

From Figure 9 (columns 1–2), Nexus consistently delivers the highest throughput among single-GPU systems. In Long Data Collections and Arxiv Workloads, it achieves 1.5–1.8× higher throughput than vLLM and 1.18-1.27× higher than SGLang, reflecting more efficient resource scheduling under uniform or moderately variable request patterns. 

In Mixed Workload, where prompt diversity and scheduling imbalance are most pronounced, Nexus demonstrates its largest gain: it achieves 1.9× higher throughput than vLLM, 1.8× over SGLang, and even 1.4× over vLLM-P/D, despite the latter using two GPUs and full-phase disaggregation. This is driven by Nexus’s head-light design which aggressively prioritizing short requests and adapting GPU resource allocation dynamically, enabling it to avoid contention and return early outputs with minimal delay. 

By serving more requests under the same compute budget, Nexus not only improves user-perceived latency but also scales to higher load than other systems, evidencing its strength in practical high-throughput serving scenarios. 

**6.2.2 End-to-End Multi-GPU Performance.** Due to space limits, we present multi-GPU results for Qwen-2.5-14B on the Mixed Workload only, as trends on other ones are similar. Since FastServe already performs poorly in single-GPU tests, we exclude it here and compare against stronger baselines: vLLM, SGLang, and vLLM-P/D. 

As shown in FIgure 10, Nexus achieves the highest throughput, 2.2× over vLLM and 2× over SGLang, while using the same hardware. These gains come from efficient intra-engine phase separation and the ability to maintain high concurrency without overwhelming shared compute or memory resources. 

More importantly, this throughput does not come at the cost of latency. Nexus delivers 2–3× lower average TTFT and 1.5–2× lower TBT than vLLM and SGLang, showing strong responsiveness across both prefill and decode phases. At the 

10 

**==> picture [504 x 293] intentionally omitted <==**

**----- Start of picture text -----**<br>
FastServe vLLM SGLang Nexus vLLM-P/D(2GPU)<br>Normalized latency P95 Normalized latency Avg TTFT P95 TTFT Avg TBT P95 TBT<br>3 6 800 1500<br>0.6 0.6<br>2 4 1000<br>400 0.4 0.4<br>1 2 500<br>0.2 0.2<br>0 0 0 0 0.0 0.0<br>0 2 4 2 4 2 4 2 4 2 4 2 4<br>3 6 500<br>800 0.6 0.6<br>2 4<br>250 0.4 0.4<br>400<br>1 2<br>0.2 0.2<br>0 0 0 0 0.0 0.0<br>2 4 2 4 2 4 2 4 2 4 2 4<br>3 10 1500<br>0.6 0.6<br>300<br>2 1000<br>5 0.4 0.4<br>1 150 500<br>0.2 0.2<br>0 0 0 0 0.0 0.0<br>1 2 3 1 2 3 1 2 3 1 2 3 1 2 3 1 2 3<br>Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s)<br>(s/token) (s/token) Latency(s) Latency(s) Latency(s)<br>Long Data Collection<br>Arxiv<br>(s/token) (s/token) Latency(s) Latency(s) Latency(s)<br>Mixed<br>(s/token) (s/token) Latency(s) Latency(s) Latency(s)<br>**----- End of picture text -----**<br>


**Figure 9. End-to-end results on Single GPU.** All systems use a single L20 GPU, except vLLM-P/D which uses two. This figure compares three workloads: Long Data Collection and Arxiv use Qwen-2.5–3B(first two rows), and Mixed uses Llama-3–1.8B(third row). The first and second columns report the average and 95th-percentile normalized latency—lower is better for throughput. The third and fourth columns report the average and 95th-percentile TTFT, while the fifth and sixth columns show the average and 95th-percentile TBT. 

**==> picture [241 x 102] intentionally omitted <==**

**----- Start of picture text -----**<br>
vLLM SGLang Nexus vLLM-P/D<br>3 1.0<br>2 500<br>0.5<br>1 250<br>0 1 2 0 1 2 0.0 1 2<br>1.0<br>200<br>1000<br>100 0.5<br>0 1 2 0 1 2 0.0 1 2<br>Arrival Rate (req/s) Arrival Rate (req/s) Arrival Rate (req/s)<br>(s/token) TTFT(s) TBT(s)<br>Normalized Latency<br>(s/token) P95 TTFT(s) P95 TBT(s)<br>P95 Normalized Latency<br>**----- End of picture text -----**<br>


**Figure 10. End-to-end results for Multi-GPU.** Run using Mixed Workload on two NVIDIA L20 GPUs with Qwen2.5–14B. All systems use two L20 GPU. The top row presents average normalized latency, TTFT, and TBT, while the bottom row shows corresponding P95 metrics. 

tail, P95 TTFT is slightly higher than SGLang but matches vLLM, while P95 TBT is nearly identical across all systems. 

One surprising result is the poor performance of vLLMP/D despite its disaggregated architecture. Its aggressive prefill overwhelms the decode stage and saturates the transfer buffer, leading to frequent cache evictions and recomputation. Nexus, by contrast, avoids these issues through adaptive SM partitioning which dynamically changes load to sustain decoding throughput even under pressure. 

**==> picture [242 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
3000<br>1500 FastServe<br>vLLM-P/D<br>1000 vLLM<br>SGLang 1500<br>Nexus<br>500<br>× × × × × ×<br>0 0<br>512 1024 2048 1024 2048 4096<br>Total Jobs Total Jobs<br>(a)  Long Data Collections. (b)  Mixed.<br>MakeSpan(s) MakeSpan(s)<br>**----- End of picture text -----**<br>


**Figure 11. Offline Inference.** Run on a single L20 under Long Data Collections and Mixed Workloads with 3B and 8B models repsectively. X means timeout. All systems use a single L20 GPU, except vLLM-P/D with two. 

In sum, Nexus offers the best latency–throughput tradeoff among all baselines, scaling to larger models while preserving per-token responsiveness and avoiding the coordination pitfalls of more fragmented systems. 

## **6.3 Offline Inference under Heterogeneous Prompts** 

In offline settings where requests are handled in large batches, throughput should be prioritized over latency. To evaluate this scenario, we submit all requests at once and measure endto-end makespan. As shown in Figure 11, Nexus achieves 5–50% lower makespan than vLLM and SGLang on Long Data Collections, which features uniformly long requests that benefit from phase separation and adaptive GPU resource use. In 

11 

**==> picture [241 x 79] intentionally omitted <==**

**----- Start of picture text -----**<br>
3.00 Execution 10.00<br>Scheduler<br>2.00 Wait 5.00<br>1.00<br>0.00 0.00<br>System System<br>(a)  Long Data Collections. (b)  Mixed.<br>FastServevLLM-P/D vLLM SGLang Nexus FastServevLLM-P/D vLLM SGLang Nexus<br>Latency(s/token) Latency(s/token)<br>**----- End of picture text -----**<br>


**Figure 12. Breakdown of Inference Overheads.** Run on a single L20 under Long Data Collections and Mixed Workloads with 3B and 8B models repsectively. X means timeout. All systems use a single L20 GPU, except vLLM-P/D with two. 

Mixed Workload, with highly variable input lengths, Nexus still outperforms vLLM by 5% but lags SGLang by 8–15% due to its stronger tail control. FastServe times out under both workloads. vLLM-P/D acheives 15%-35% lower makespan than Nexus but needs more GPU. 

## **6.4 Latency Breakdown** 

To better understand the sources of Nexus’s performance gains, Figure 12 decomposes normalized token latency into scheduling, queuing, and execution stages. 

**Scheduling Overhead.** All systems incur minimal scheduling latency. Nexus’s dual-queue design introduces no measurable overhead, confirming its coordination logic is lightweight. 

**Execution Time.** Execution latency under Nexus closely matches that of vLLM and SGLang. Although Nexus runs prefill and decode in separate batches, which leads to reading model weights more than once, its batch level separation and dynamic resource use help reduce contention, keeping the overhead low. vLLM-P/D achieves the lowest execution latency due to full disaggregation, but at the cost of using twice GPUs. 

**Queuing Delay.** Waiting time dominates total latency under load, and here Nexus demonstrates its greatest advantage. In Long Data Collections, Nexus reduces waiting time by 4× over vLLM and 2× over SGLang. In Mixed Workload, which involves greater request variability, it improves further, achieving 5× lower wait time than monolithic baselines, and 2× less than vLLM-P/D. These gains stem from Nexus’s effective shortest-prompt-first scheduling and adaptive GPU resource allocation, allowing it to maintain concurrency without overloading shared resources. 

## **6.5 Ablation Study** 

We ablate Nexus’s two core components: (1) dynamic SM changing and (2) phase-specific scheduling, with focus on Shortest-Prompt-First (SPF) for prefill. Figure 13 shows the results. 

**Baseline:FCFS Scheduling(Naive Intra-Engine PD Disaggregation).** The baseline (PF-DF-Wo-SC) is the intra-engine PD disaggregation that uses FCFS for both prefill and decode without dynamic SM changing. It suffers from head-of-line 

**==> picture [241 x 72] intentionally omitted <==**

**----- Start of picture text -----**<br>
PF-DF-Wo-SC PF-DF-W-SC Nexus-Wo-SC Nexus<br>0.3<br>500<br>0.2<br>0 0.1<br>1.5 2.0 2.5 3.0 1.5 2.0 2.5 3.0<br>Arrival Rate (req/s) Arrival Rate (req/s)<br>Avg TTFT(s) Avg TBT(s)<br>**----- End of picture text -----**<br>


**Figure 13. Ablation Study.** Run with Mixed Workload on Llama3.1-8B using a single L20 GPU. _PF-DF-Wo-SC_ is the intraengine PD disaggregation that uses FCFS for both prefill and decode scheduling, without dynamical GPU SM changing. _PF-DF-W-SC_ is the intra-engine PD disaggregation that FCFS scheduling for both prefill and decode but enables dynamical GPU SM changing. _NexusWo-SC_ denotes our system with dynamical GPU SM changing disabled. 

(HOL) blocking in prefill and persistent resource contention between phases, resulting in poor TTFT and TBT. 

**Effect of Dynamical SM Changing.** Enabling dynamic SM changing (PF-DF-W-SC) improves TBT by 14% over the baseline by assigning more compute to decode when GPU memory becomes bottneleck. However, TTFT degrades by 30% due to delayed prefill execution during decode intervals. This showcases the inevitable TBT and TTFT tradeoff under naive FCFS. 

**Effect of Prefill Scheduling (SPF without SM Changing).** Applying SPF to prefill (Nexus-Wo-SC) dramatically improves TTFT (up to 90% reduction over baseline) by mitigating HOL blocking. However, TBT worsens, due to unresolved GPU resource contention between prefill and decode. SPF helps responsiveness, but lacks decode-phase control. 

**Combined Design: SPF + Dynamical SM Changing** When both techniques are used (Nexus), TTFT improves by 23% over SPF-only, while TBT drops by 26%, achieving optimality. Unlike FCFS+switching, TTFT does not regress, as SPF reduces prefill HoL blocking and can benefit from less contention. Dynamical SM changing amplifies gains without introducing new overhead. 

## **7 Related Work** 

**Monolithic LLM Serving Systems.** Early LLM serving engines focus on maximizing throughput and memory efficiency within a unified execution model. Orca [62] introduces continuous batching to reduce head-of-line blocking. vLLM [25] eliminates KV cache fragmentation via PagedAttention, while SGLang [66] reduces memory usage through RadixAttention. SarathiServe [1] mixes chunks of prefill and decode requests to better utilize GPU resources. However, all of these designs treat prefill and decode as indistinguishable units within a shared queue. In contrast, **Nexus** decouples the two phases at the batching level, enabling independent, phase-specific execution and scheduling while preserving compatibility with existing attention mechanisms. 

**LLM Scheduling Frameworks.** Recent works propose sophisticated schedulers to balance latency and throughput. 

12 

FastServe [56] uses MLFQ with skip-join to avoid prompt variance stalls. VTC [47] and QLM [43] target fairness and SLO adherence. Llumnix [50] accelerates dynamic scaling via migration-aware scheduling. LightLLM [17] and Preble [48] improve memory reuse via future prediction or prompt sharing. Yet, all treat each request as a monolithic scheduling unit. **Nexus** introduces dual schedulers for prefill and decode, each tailored to its phase’s latency sensitivity and compute intensity. This decoupling enables tighter queue control and more efficient GPU resource utilization. 

**Engine-Level PD Disaggregation Systems.** Several systems physically disaggregate prefill and decode across GPUs to better match their resource profiles. Splitwise [42] statically assigns phases to different hardware tiers; DistServe [67] improves TTFT/TBT via tiered GPU scheduling; Mooncake [45] serves cached KV blocks from storage to reduce compute load; TetriInfer [23] routes requests by latency class to isolated replicas. **Nexus** achieves similar benefits without incurring cross-engine complexity. By performing lightweight intra-engine disaggregation, it enables low-latency execution and efficient KV reuse within a single serving engine. **Intra-GPU PD Disaggregation Systems.** There are some prior works on intra-engine PD disaggregation [9, 26]. Drift [9] introduces an adaptive gang scheduling mechanism, a contentionfree performance model, and an SLO-aware dispatching policy to enable intra-engine prefill–decode separation. Bullet [26] proposes a comprehensive system that includes: (1) a performance estimator for building a profile-augmented analytical model; (2) an SLO-aware scheduler to dynamically balance the compute load between prefill and decode phases; and (3) a resource manager capable of delivering fast yet accurate resource configurations. The key differences between their and Nexus are as follows: Nexus employs a contention-based cost model to estimate latency, formulates a dual-objective optimization problem, and uses a greedy search to determine the optimal SM partitioning. In addition, our scheduler is phase-aware and explicitly considers the distinct characteristics of prefill and decode stages, which contrasts with others’ SLO-aware scheduling approach. 

**Intra-GPU PD Disaggregation Systems.** Recent systems like Bullet [26], Drift [9], and semi-PD [22] explore intraGPU disaggregation to reduce cross-device overhead. Bullet builds profile-augmented latency models and uses SLOdriven feedback loops to tune SM partitioning reactively. Drift applies phase-tagged gang scheduling under a contentionfree assumption, combining static modeling with latencyaware dispatch. Semi-PD fits inverse-linear latency curves and adjusts SM ratios through runtime feedback control based on latency violations. **Nexus** takes a proactive approach. First, it introduces a contention-aware analytical model that explicitly captures both diminishing compute returns and dynamic memory bandwidth interference at the 

operator level. Second, it formulates intra-GPU resource allocation as a dual-objective optimization problem guided by runtime KV-cache usage and solved via fast greedy search. Third, Nexus uses a one-time profiling pass to calibrate perkernel latency scaling curves, but avoids offline workload tracing or in-deployment feedback fitting, enabling generalization across dynamic traffic and prompt structures. **GPU Multiplexing.** Recent work explores fine-grained GPU sharing via spatial or temporal partitioning. NVIDIA MPS [38] and MIG [37] provide coarse-grained isolation, while GreenContext [35] enables dynamic intra-process SM partitioning. Systems like GPUlet, Orion, REEF, and Bless [8, 21, 49, 65] improve utilization for small models through temporal multiplexing. MuxServe [13] proposes spatial-temporal multiplexing to efficiently serve multiple LLMs, while NanoFlow [68] and Liger [12] introduce kernel level parallelism that enables concurrent execution of compute bound, network bound, and memory bound operations. PoD [24] fuses prefill attention and decode attention into one kernel to reduce overhead. 

**Nexus** differs in three key aspects: (1)it disaggregates prefill and decode into different batch and executes these batch concurrently; and (2) it dynamically reallocate SMs across prefill and decode stages in a single engine, adapting to workload shifts; and (3) it introduces intra-engine, phase-specific schedulers for coordinated but decoupled execution. This design enables fine-grained responsiveness while maximizing intra-GPU parallelism. 

## **8 Conclusion** 

While the asymmetric resource demands of prefill and decode phases in LLM inference is well recognized, its implications for intra-GPU coordination have remained underexplored. In this work, we analyze how co-executing these phases under coarse-grained batching strategies leads to resource imbalance and performance interference. To address this, we propose Nexus, a novel SM-partitioned execution framework with dynamic resource allocation and phase-specific schedulers. Our system consistently improves end-to-end throughput and latency across diverse LLM workloads, outperforming state-of-the-art serving systems. 

## **References** 

- [1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S. Gulavani, Alexey Tumanov, and Ramachandran Ramjee. 2024. Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve. In _18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024_ , Ada Gavrilovska and Douglas B. Terry (Eds.). USENIX Association, 117–134. https://www.usenix.org/conference/osdi24/ presentation/agrawal 

- [2] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, et al. 2023. Qwen technical report. _arXiv preprint arXiv:2309.16609_ (2023). 

- [3] Jinze Bai, Shuai Bai, Yunfei Chu, Zeyu Cui, Kai Dang, Xiaodong Deng, Yang Fan, Wenbin Ge, Yu Han, Fei Huang, Binyuan Hui, Luo Ji, Mei Li, Junyang Lin, Runji Lin, Dayiheng Liu, Gao Liu, Chengqiang Lu, 

13 

   - Keming Lu, Jianxin Ma, Rui Men, Xingzhang Ren, Xuancheng Ren, Chuanqi Tan, Sinan Tan, Jianhong Tu, Peng Wang, Shijie Wang, Wei Wang, Shengguang Wu, Benfeng Xu, Jin Xu, An Yang, Hao Yang, Jian Yang, Shusheng Yang, Yang Yao, Bowen Yu, Hongyi Yuan, Zheng Yuan, Jianwei Zhang, Xingxuan Zhang, Yichang Zhang, Zhenru Zhang, Chang Zhou, Jingren Zhou, Xiaohuan Zhou, and Tianhang Zhu. 2023. Qwen Technical Report. arXiv:2309.16609 [cs.CL] https://arxiv.org/ abs/2309.16609 

- [4] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are FewShot Learners. In _Advances in Neural Information Processing Systems 33: Annual Conference on Neural Information Processing Systems 2020, NeurIPS 2020, December 6-12, 2020, virtual_ , Hugo Larochelle, Marc’Aurelio Ranzato, Raia Hadsell, Maria-Florina Balcan, and HsuanTien Lin (Eds.). https://proceedings.neurips.cc/paper/2020/hash/ 1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html 

- [5] Tom B. Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, Sandhini Agarwal, Ariel Herbert-Voss, Gretchen Krueger, Tom Henighan, Rewon Child, Aditya Ramesh, Daniel M. Ziegler, Jeffrey Wu, Clemens Winter, Christopher Hesse, Mark Chen, Eric Sigler, Mateusz Litwin, Scott Gray, Benjamin Chess, Jack Clark, Christopher Berner, Sam McCandlish, Alec Radford, Ilya Sutskever, and Dario Amodei. 2020. Language Models are Few-Shot Learners. arXiv:2005.14165 [cs.CL] https://arxiv.org/abs/2005.14165 

- [6] ccdv. 2025. arxiv-summarization. https://huggingface.co/datasets/ ccdv/arxiv-summarization. 

- [7] Yihua Cheng, Kuntai Du, Jiayi Yao, and Junchen Jiang. 2024. Do Large Language Models Need a Content Delivery Network? _arXiv preprint arXiv:2409.13761_ (2024). 

- [8] Seungbeom Choi, Sunho Lee, Yeonjae Kim, Jongse Park, Youngjin Kwon, and Jaehyuk Huh. 2022. Serving Heterogeneous Machine Learning Models on Multi-GPU Servers with Spatio-Temporal Sharing. In _Proceedings of the 2022 USENIX Annual Technical Conference, USENIX ATC 2022, Carlsbad, CA, USA, July 11-13, 2022_ , Jiri Schindler and Noa Zilberman (Eds.). USENIX Association, 199–216. https://www.usenix. org/conference/atc22/presentation/choi-seungbeom 

- [9] Weihao Cui, Yukang Chen, Han Zhao, Ziyi Xu, Quan Chen, Xusheng Chen, Yangjie Zhou, Shixuan Sun, and Minyi Guo. 2025. Optimizing SLO-oriented LLM Serving with PD-Multiplexing. _CoRR_ abs/2504.14489 (2025). doi:10.48550/ARXIV.2504.14489 arXiv:2504.14489 

- [10] Cursor. 2025. Cursor. https://www.cursor.com/. 

- [11] DeepSeek. 2025. DeepSeek. https://www.deepseek.com/. 

- [12] Jiangsu Du, Jinhui Wei, Jiazhi Jiang, Shenggan Cheng, Dan Huang, Zhiguang Chen, and Yutong Lu. 2024. Liger: Interleaving Intra- and Inter-Operator Parallelism for Distributed Large Model Inference. In _Proceedings of the 29th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming, PPoPP 2024, Edinburgh, United Kingdom, March 2-6, 2024_ , Michel Steuwer, I-Ting Angelina Lee, and Milind Chabbi (Eds.). ACM, 42–54. doi:10.1145/3627535.3638466 

- [13] Jiangfei Duan, Runyu Lu, Haojie Duanmu, Xiuhong Li, Xingcheng Zhang, Dahua Lin, Ion Stoica, and Hao Zhang. 2024. MuxServe: Flexible Spatial-Temporal Multiplexing for Multiple LLM Serving. In _Forty-first International Conference on Machine Learning, ICML 2024, Vienna, Austria, July 21-27, 2024_ . OpenReview.net. https: //openreview.net/forum?id=R0SoZvqXyQ 

- [14] Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Amy Yang, Angela Fan, Anirudh Goyal, Anthony Hartshorn, Aobo Yang, Archi Mitra, Archie Sravankumar, Artem Korenev, Arthur Hinsvark, Arun Rao, Aston Zhang, Aurélien Rodriguez, Austen Gregerson, Ava Spataru, Baptiste Rozière, Bethany Biron, Binh Tang, Bobbie Chern, Charlotte Caucheteux, Chaya Nayak, Chloe Bi, Chris Marra, Chris McConnell, Christian Keller, Christophe Touret, Chunyang Wu, Corinne Wong, Cristian Canton Ferrer, Cyrus Nikolaidis, Damien Allonsius, Daniel Song, Danielle Pintz, Danny Livshits, David Esiobu, Dhruv Choudhary, Dhruv Mahajan, Diego Garcia-Olano, Diego Perino, Dieuwke Hupkes, Egor Lakomkin, Ehab AlBadawy, Elina Lobanova, Emily Dinan, Eric Michael Smith, Filip Radenovic, Frank Zhang, Gabriel Synnaeve, Gabrielle Lee, Georgia Lewis Anderson, Graeme Nail, Grégoire Mialon, Guan Pang, Guillem Cucurell, Hailey Nguyen, Hannah Korevaar, Hu Xu, Hugo Touvron, Iliyan Zarov, Imanol Arrieta Ibarra, Isabel M. Kloumann, Ishan Misra, Ivan Evtimov, Jade Copet, Jaewon Lee, Jan Geffert, Jana Vranes, Jason Park, Jay Mahadeokar, Jeet Shah, Jelmer van der Linde, Jennifer Billock, Jenny Hong, Jenya Lee, Jeremy Fu, Jianfeng Chi, Jianyu Huang, Jiawen Liu, Jie Wang, Jiecao Yu, Joanna Bitton, Joe Spisak, Jongsoo Park, Joseph Rocca, Joshua Johnstun, Joshua Saxe, Junteng Jia, Kalyan Vasuden Alwala, Kartikeya Upasani, Kate Plawiak, Ke Li, Kenneth Heafield, Kevin Stone, and et al. 2024. The Llama 3 Herd of Models. _CoRR_ abs/2407.21783 (2024). doi:10.48550/ARXIV.2407.21783 arXiv:2407.21783 

- [15] Jingqi Feng, Yukai Huang, Rui Zhang, Sicheng Liang, Ming Yan, and Jie Wu. 2025. WindServe: Efficient Phase-Disaggregated LLM Serving with Stream-based Dynamic Scheduling. In _Proceedings of the 52nd Annual International Symposium on Computer Architecture_ . 1283–1295. 

- [16] GigaSpaces. 2023. _Amazon Found Every 100ms of Latency Cost them 1% in Sales_ . https://www.gigaspaces.com/blog/amazon-found-every100ms-of-latency-cost-them-1-in-sales Accessed: 2025-05-28. 

- [17] Ruihao Gong, Shihao Bai, Siyu Wu, Yunqian Fan, Zaijun Wang, Xiuhong Li, Hailong Yang, and Xianglong Liu. 2025. Past-Future Scheduler for LLM Serving under SLA Guarantees. In _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2025, Rotterdam, Netherlands, 30 March 2025 - 3 April 2025_ , Lieven Eeckhout, Georgios Smaragdakis, Katai Liang, Adrian Sampson, Martha A. Kim, and Christopher J. Rossbach (Eds.). ACM, 798–813. doi:10.1145/3676641.3716011 

- [18] Google. 2025. gemini. https://gemini.google.com/. 

- [19] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Ruoyu Zhang, Runxin Xu, Qihao Zhu, Shirong Ma, Peiyi Wang, Xiao Bi, et al. 2025. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. _arXiv preprint arXiv:2501.12948_ (2025). 

- [20] Daya Guo, Qihao Zhu, Dejian Yang, Zhenda Xie, Kai Dong, Wentao Zhang, Guanting Chen, Xiao Bi, Yu Wu, YK Li, et al. 2024. DeepSeekCoder: When the Large Language Model Meets Programming–The Rise of Code Intelligence. _arXiv preprint arXiv:2401.14196_ (2024). 

- [21] Mingcong Han, Hanze Zhang, Rong Chen, and Haibo Chen. 2022. Microsecond-scale Preemption for Concurrent GPU-accelerated DNN Inferences. In _16th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2022, Carlsbad, CA, USA, July 11-13, 2022_ , Marcos K. Aguilera and Hakim Weatherspoon (Eds.). USENIX Association, 539–558. https://www.usenix.org/conference/osdi22/ presentation/han 

- [22] Ke Hong, Lufang Chen, Zhong Wang, Xiuhong Li, Qiuli Mao, Jianping Ma, Chao Xiong, Guanyu Wu, Buhe Han, Guohao Dai, Yun Liang, and Yu Wang. 2025. semi-PD: Towards Efficient LLM Serving via Phase-Wise Disaggregated Computation and Unified Storage. arXiv:2504.19867 [cs.CL] https://arxiv.org/abs/2504.19867 

14 

- [23] Cunchen Hu, Heyang Huang, Liangliang Xu, Xusheng Chen, Jiang Xu, Shuang Chen, Hao Feng, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, and Yizhou Shan. 2024. Inference without Interference: Disaggregate LLM Inference for Mixed Downstream Workloads. _CoRR_ abs/2401.11181 (2024). doi:10.48550/ARXIV.2401.11181 arXiv:2401.11181 

- [24] Aditya K. Kamath, Ramya Prabhu, Jayashree Mohan, Simon Peter, Ramachandran Ramjee, and Ashish Panwar. 2025. POD-Attention: Unlocking Full Prefill-Decode Overlap for Faster LLM Inference. In _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2025, Rotterdam, Netherlands, 30 March 2025 - 3 April 2025_ , Lieven Eeckhout, Georgios Smaragdakis, Katai Liang, Adrian Sampson, Martha A. Kim, and Christopher J. Rossbach (Eds.). ACM, 897–912. doi:10.1145/3676641.3715996 

- [25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient Memory Management for Large Language Model Serving with PagedAttention. In _Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023_ , Jason Flinn, Margo I. Seltzer, Peter Druschel, Antoine Kaufmann, and Jonathan Mace (Eds.). ACM, 611–626. doi:10.1145/ 3600006.3613165 

- [26] Zejia Lin, Hongxin Xu, Guanyi Chen, Xianwei Zhang, and Yutong Lu. 2025. Bullet: Boosting GPU Utilization for LLM Serving via Dynamic Spatial-Temporal Orchestration. arXiv:2504.19516 [cs.DC] https:// arxiv.org/abs/2504.19516 

- [27] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. 2024. Deepseek-v3 technical report. _arXiv preprint arXiv:2412.19437_ (2024). 

- [28] Yuhan Liu, Hanchen Li, Yihua Cheng, Siddhant Ray, Yuyang Huang, Qizheng Zhang, Kuntai Du, Jiayi Yao, Shan Lu, Ganesh Ananthanarayanan, et al. 2024. Cachegen: Kv cache compression and streaming for fast large language model serving. In _Proceedings of the ACM SIGCOMM 2024 Conference_ . 38–56. 

- [29] Michael Luo, Sijun Tan, Roy Huang, Ameen Patel, Alpay Ariyak, Qingyang Wu, Xiaoxiang Shi, Rachel Xin, Colin Cai, Maurice Weber, et al. 2025. Deepcoder: A fully open-source 14b coder at o3-mini level. _Notion Blog_ (2025). 

- [30] Yixuan Mei, Yonghao Zhuang, Xupeng Miao, Juncheng Yang, Zhihao Jia, and Rashmi Vinayak. 2025. Helix: Serving large language models over heterogeneous gpus and network via max-flow. In _Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1_ . 586–602. 

- [31] Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, et al. 2024. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ . 932–949. 

- [32] Xupeng Miao, Chunan Shi, Jiangfei Duan, Xiaoli Xi, Dahua Lin, Bin Cui, and Zhihao Jia. 2024. Spotserve: Serving generative large language models on preemptible instances. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2_ . 1112–1127. 

- [33] Nvidia. 2024. FasterTransformer. https://github.com/NVIDIA/ FasterTransformer. 

- [34] Nvidia. 2025. B200. https://resources.nvidia.com/en-us-blackwellarchitecture. 

- [35] Nvidia. 2025. CUDA Driver API: Green COntexts. https: //docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__GREEN_ _CONTEXTS.html. 

- [36] Nvidia. 2025. H100. https://resources.nvidia.com/en-us-hopperarchitecture/nvidia-h100-tensor-c. 

- [37] Nvidia. 2025. MIG. https://www.nvidia.com/en-sg/technologies/multiinstance-gpu/. 

- [38] Nvidia. 2025. MPS. https://docs.nvidia.com/deploy/mps/contents. html. 

- [39] Hyungjun Oh, Kihong Kim, Jaemin Kim, Sungkyun Kim, Junyeol Lee, Du-Seong Chang, and Jiwon Seo. 2024. ExeGPT: Constraint-Aware Resource Scheduling for LLM Inference. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS 2024, La Jolla, CA, USA, 27 April 2024- 1 May 2024_ , Rajiv Gupta, Nael B. Abu-Ghazaleh, Madan Musuvathi, and Dan Tsafrir (Eds.). ACM, 369–384. doi:10.1145/ 3620665.3640383 

- [40] OpenAI. 2023. GPT-4 Technical Report. _CoRR_ abs/2303.08774 (2023). doi:10.48550/ARXIV.2303.08774 arXiv:2303.08774 

- [41] OpenAI. 2025. ChatGPT. https://chatgpt.com. 

- [42] Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah, Íñigo Goiri, Saeed Maleki, and Ricardo Bianchini. 2024. Splitwise: Efficient Generative LLM Inference Using Phase Splitting. In _51st ACM/IEEE Annual International Symposium on Computer Architecture, ISCA 2024, Buenos Aires, Argentina, June 29 - July 3, 2024_ . IEEE, 118–132. doi:10. 1109/ISCA59077.2024.00019 

- [43] Archit Patke, Dhemath Reddy, Saurabh Jha, Haoran Qiu, Christian Pinto, Chandra Narayanaswami, Zbigniew Kalbarczyk, and Ravishankar K. Iyer. 2024. Queue Management for SLO-Oriented Large Language Model Serving. In _Proceedings of the 2024 ACM Symposium on Cloud Computing, SoCC 2024, Redmond, WA, USA, November 20-22, 2024_ . ACM, 18–35. doi:10.1145/3698038.3698523 

- [44] perplexity. 2025. perplexity. https://www.perplexity.ai/. 

- [45] Ruoyu Qin, Zheming Li, Weiran He, Jialei Cui, Feng Ren, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. 2025. Mooncake: Trading More Storage for Less Computation — A KVCachecentric Architecture for Serving LLM Chatbot. In _23rd USENIX Conference on File and Storage Technologies (FAST 25)_ . USENIX Association, Santa Clara, CA, 155–170. https://www.usenix.org/conference/fast25/ presentation/qin 

- [46] ShareGPT. 2025. ShareGPT. https://huggingface.co/datasets/ anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ ShareGPT_V3_unfiltered_cleaned_split.json. 

- [47] Ying Sheng, Shiyi Cao, Dacheng Li, Banghua Zhu, Zhuohan Li, Danyang Zhuo, Joseph E. Gonzalez, and Ion Stoica. 2024. Fairness in Serving Large Language Models. In _18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024_ , Ada Gavrilovska and Douglas B. Terry (Eds.). USENIX Association, 965–988. https://www.usenix.org/conference/ osdi24/presentation/sheng 

- [48] Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, and Yiying Zhang. 2024. Preble: Efficient Distributed Prompt Scheduling for LLM Serving. _CoRR_ abs/2407.00023 (2024). doi:10.48550/ARXIV. 2407.00023 arXiv:2407.00023 

- [49] Foteini Strati, Xianzhe Ma, and Ana Klimovic. 2024. Orion: Interference-aware, Fine-grained GPU Sharing for ML Applications. In _Proceedings of the Nineteenth European Conference on Computer Systems, EuroSys 2024, Athens, Greece, April 22-25, 2024_ . ACM, 1075–1092. doi:10.1145/3627703.3629578 

- [50] Biao Sun, Ziming Huang, Hanyu Zhao, Wencong Xiao, Xinyi Zhang, Yong Li, and Wei Lin. 2024. Llumnix: Dynamic Scheduling for Large Language Model Serving. In _18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024_ , Ada Gavrilovska and Douglas B. Terry (Eds.). USENIX Association, 173–191. https://www.usenix.org/conference/osdi24/ presentation/sun-biao 

15 

- [51] Kimi Team, Angang Du, Bofei Gao, Bowei Xing, Changjiu Jiang, Cheng Chen, Cheng Li, Chenjun Xiao, Chenzhuang Du, Chonghua Liao, et al. 2025. Kimi k1. 5: Scaling reinforcement learning with llms. _arXiv preprint arXiv:2501.12599_ (2025). 

- [52] SGLang team. 2025. SGLang-v0.4. https://lmsys.org/blog/2024-12-04sglang-v0-4/. 

- [53] TogetherComputer. 2025. Long-Data-Collections. https://huggingface. co/datasets/togethercomputer/Long-Data-Collections. 

- [54] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is All you Need. In _Advances in Neural Information Processing Systems 30: Annual Conference on Neural Information Processing Systems 2017, December 4-9, 2017, Long Beach, CA, USA_ , Isabelle Guyon, Ulrike von Luxburg, Samy Bengio, Hanna M. Wallach, Rob Fergus, S. V. N. Vishwanathan, and Roman Garnett (Eds.). 5998–6008. https://proceedings.neurips.cc/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html 

- [55] vLLM team. 2025. vLLM-v1. https://blog.vllm.ai/2025/01/27/v1-alpharelease.html. 

- [56] Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. 2023. Fast Distributed Inference Serving for Large Language Models. _CoRR_ abs/2305.05920 (2023). doi:10.48550/ARXIV. 2305.05920 arXiv:2305.05920 

- [57] Jiaming Xu, Jiayi Pan, Yongkang Zhou, Siming Chen, Jinhao Li, Yaoxiu Lian, Junyi Wu, and Guohao Dai. 2025. Specee: Accelerating large language model inference with speculative early exiting. In _Proceedings of the 52nd Annual International Symposium on Computer Architecture_ . 467–481. 

   - [65] Shulai Zhang, Quan Chen, Weihao Cui, Han Zhao, Chunyu Xue, Zhen Zheng, Wei Lin, and Minyi Guo. 2025. Improving GPU Sharing Performance through Adaptive Bubbleless Spatial-Temporal Sharing. In _Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025_ . ACM, 573–588. doi:10.1145/3689031.3696070 

   - [66] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark W. Barrett, and Ying Sheng. 2023. Efficiently Programming Large Language Models using SGLang. _CoRR_ abs/2312.07104 (2023). doi:10.48550/ARXIV.2312.07104 arXiv:2312.07104 

   - [67] Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. 2024. DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving. In _18th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2024, Santa Clara, CA, USA, July 10-12, 2024_ , Ada Gavrilovska and Douglas B. Terry (Eds.). USENIX Association, 193– 210. https://www.usenix.org/conference/osdi24/presentation/zhongyinmin 

   - [68] Kan Zhu, Yilong Zhao, Liangyu Zhao, Gefei Zuo, Yile Gu, Dedong Xie, Yufei Gao, Qinyu Xu, Tian Tang, Zihao Ye, Keisuke Kamahori, ChienYu Lin, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. 2024. NanoFlow: Towards Optimal Large Language Model Serving Throughput. _CoRR_ abs/2408.12757 (2024). doi:10.48550/ARXIV.2408. 12757 arXiv:2408.12757 

- [58] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. _arXiv preprint arXiv:2505.09388_ (2025). 

- [59] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. 2024. Qwen2.5 Technical Report. _arXiv preprint arXiv:2412.15115_ (2024). 

- [60] Jiayi Yao, Hanchen Li, Yuhan Liu, Siddhant Ray, Yihua Cheng, Qizheng Zhang, Kuntai Du, Shan Lu, and Junchen Jiang. 2024. CacheBlend: Fast Large Language Model Serving with Cached Knowledge Fusion. _arXiv preprint arXiv:2405.16444_ (2024). 

- [61] Zihao Ye, Lequn Chen, Ruihang Lai, Wuwei Lin, Yineng Zhang, Stephanie Wang, Tianqi Chen, Baris Kasikci, Vinod Grover, Arvind Krishnamurthy, and Luis Ceze. 2025. FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving. _arXiv preprint arXiv:2501.01005_ (2025). https://arxiv.org/abs/2501.01005 

- [62] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. 2022. Orca: A Distributed Serving System for Transformer-Based Generative Models. In _16th USENIX Symposium on Operating Systems Design and Implementation, OSDI 2022, Carlsbad, CA, USA, July 11-13, 2022_ , Marcos K. Aguilera and Hakim Weatherspoon (Eds.). USENIX Association, 521–538. https://www.usenix.org/ conference/osdi22/presentation/yu 

- [63] Lingfan Yu, Jinkun Lin, and Jinyang Li. 2025. Stateful large language model serving with pensieve. In _Proceedings of the Twentieth European Conference on Computer Systems_ . 144–158. 

- [64] Haochen Yuan, Yuanqing Wang, Wenhao Xie, Yu Cheng, Ziming Miao, Lingxiao Ma, Jilong Xue, and Zhi Yang. 2025. NeuStream: Bridging Deep Learning Serving and Stream Processing. In _Proceedings of the Twentieth European Conference on Computer Systems_ . 671–685. 

16 

