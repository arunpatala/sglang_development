# Chapter: Multi-LoRA Inference — Serving Many Fine-Tuned Models from a Single GPU

**Format:** Textbook-style prose chapter. Concepts build sequentially from first principles; each section establishes the ground for the next. Research systems (Punica, S-LoRA, dLoRA, CaraServe, ServerlessLoRA, Predictive-LoRA, Loquetier, InfiniLoRA) are cited as examples within a concept-driven narrative, not as the organizing principle. No code blocks; all numbers and mechanisms described in prose.

---

## 1. The Problem: Fine-Tuning Is Cheap, Serving Is Not

Modern large language models are trained to be generalists. They understand language, reason across domains, write code, summarize documents, and answer questions. But for most production applications, generality is not enough. A legal document analysis tool needs a model that reliably follows citation conventions. A customer support agent needs a model that knows the company's product catalog and tone. A code generation assistant needs a model that understands a specific internal API. The solution to this gap — training a generalist model to be good at a specific task — has a well-established name: fine-tuning.

The engineering infrastructure around fine-tuning has matured to the point where producing a specialized model is genuinely inexpensive. A team of two engineers can fine-tune a 7-billion parameter base model on a task-specific dataset in a single afternoon, on a single GPU, at a hardware cost measured in tens of dollars. At a company operating hundreds of distinct products and features, each with its own specialized language task, this produces exactly the natural outcome one would expect: hundreds or thousands of fine-tuned models, each tailored to a specific use case.

The problem does not arise during training. It arises the moment those models need to serve live traffic.

### 1.1 The Weight Size Problem

A 7-billion parameter model stored in half-precision floating point — the standard representation for inference — occupies approximately fourteen gigabytes of GPU high-bandwidth memory. This is not a tunable parameter. It is determined by the model architecture and the storage format. Before the model can process a single request, before any input has been seen, those fourteen gigabytes must be resident in GPU memory. This is the fixed cost of loading the model — paid once at startup, every time.

If a company has one thousand fine-tuned models, and each model is a full separate copy of the 7-billion parameter base model with its weights updated by fine-tuning, the inference infrastructure requirement is straightforward to calculate: one thousand separate GPU instances, each holding one separate copy of one model, each occupying approximately fourteen gigabytes. At current market rates for high-performance inference hardware, this amounts to millions of dollars per month in operating costs, the vast majority of which is paid for GPU memory that is sitting idle at any given moment — because most of the one thousand fine-tuned models are not receiving requests simultaneously.

The serving crisis is not a shortage of models. It is the mismatch between the economics of creating fine-tuned models and the economics of serving them.

### 1.2 Why Fine-Tuning Does Not Automatically Solve the Serving Cost

The research community recognized the training cost problem before the serving cost problem and produced an elegant solution: parameter-efficient fine-tuning. The central insight of PEFT is that full fine-tuning — updating all of the model's parameters — is almost always unnecessary. The model's general linguistic and reasoning capabilities, learned during large-scale pretraining, do not need to change when adapting to a specific domain. Only a small set of task-specific adjustments are required.

Low-Rank Adaptation, or LoRA, is the most widely adopted parameter-efficient fine-tuning method. It dramatically reduces the number of trainable parameters by restricting the adaptation to low-rank corrections that are added on top of frozen base weights. A fine-tune that would require updating 7 billion parameters in the traditional approach requires updating fewer than 10 million parameters with LoRA — a reduction of three orders of magnitude. The adapter file that encodes these corrections is typically between 50 and 200 megabytes, compared to the 14 gigabytes of the full model. Training cost, storage cost, and training time all fall proportionally.

But here is the catch: LoRA solved the training cost without automatically solving the serving cost. The most straightforward way to use a LoRA adapter at inference time is to merge it back into the base model — arithmetically combining the adapter corrections with the frozen weights to produce a single updated model. After merging, the resulting model is indistinguishable from a traditionally fine-tuned model at inference time, and can be served with zero additional overhead. But after merging, the adapter and the base model are no longer separable. You are left with one thousand merged models, each fourteen gigabytes, each requiring its own GPU instance. You have solved the training cost problem. The serving cost problem remains exactly as it was.

---

## 2. The Key Insight: Separating the Base Model from the Adapter

The fundamental shift that makes efficient multi-adapter serving possible is a reordering of the arithmetic. Instead of merging the adapter into the model before serving, we keep the base model and the adapter separate throughout inference and compute their contributions independently on every forward pass.

### 2.1 The Decomposed Forward Pass

For any linear layer in the transformer — an attention projection, for example — the standard forward pass computes an output vector by multiplying the input vector against the layer's weight matrix. In a traditionally fine-tuned model, this weight matrix has been updated by training; in a merged LoRA model, the adapter correction has been permanently added to it. In both cases, the forward pass is a single matrix multiplication.

The LoRA decomposed forward pass instead computes two terms and adds them. The first term is the same matrix multiplication against the frozen base weights — the same computation that the base model would perform. The second term is the adapter's contribution: the input is first projected down into a very small intermediate space of dimension equal to the adapter's rank, then projected back up to the original dimension. The result of this second computation is the adapter's correction — the delta that the fine-tuning process learned. The final output is the sum of the base model's output and the adapter's delta.

The critical observation is that the base model computation is identical for every request, regardless of which adapter is being used. Two requests in the same batch — one using adapter A and one using adapter B — execute exactly the same base model forward pass and differ only in which adapter delta is added to the result. This means the base model weights need to be stored exactly once, and the base model computation needs to be executed exactly once per forward pass, regardless of how many different adapters are active in the current batch.

### 2.2 The Storage Consequence

If the base model is stored once and the adapters are stored separately, the storage arithmetic transforms dramatically. One thousand customers each with their own adapter requires one copy of the fourteen-gigabyte base model plus one thousand copies of the fifty-megabyte adapter file. Total memory: approximately fourteen gigabytes plus fifty gigabytes — roughly sixty-four gigabytes, not fourteen terabytes. The per-adapter storage cost has fallen from fourteen gigabytes to fifty megabytes. An H100 GPU with eighty gigabytes of high-bandwidth memory can, in principle, hold the base model and all fifty megabytes adapters for all one thousand customers simultaneously — though in practice, the KV cache required during inference limits how many adapters can be resident at once, as we will discuss in a later section.

The economic implication is immediate. A service that previously required one thousand separate GPU instances can now be served by a small cluster of shared GPU servers, each holding one copy of the base model and a pool of the most recently used adapters. Hardware costs fall by roughly two orders of magnitude.

---

## 3. The Mathematics of Low-Rank Adaptation

Understanding why this decomposition works — why the adapter's correction can be well-approximated by two small matrices — requires engaging with the core hypothesis of the LoRA paper, published by Hu and colleagues at Microsoft in 2021 and presented at ICLR 2022.

### 3.1 The Low-Rank Hypothesis

When a pre-trained model is fine-tuned on a specific task, the weight updates it learns are not arbitrary. The model already possesses rich representations of language, syntax, and semantics. The task-specific fine-tuning does not rewire these representations from scratch; it adjusts how they are used and combined. The hypothesis underlying LoRA is that this adjustment — the correction that fine-tuning learns — has low intrinsic dimensionality. Formally, if you represent the correction as a matrix of changes to each weight, the claim is that this matrix has an effective rank much smaller than its nominal dimensions.

To understand why this matters, consider that a matrix with rank one can be written as the outer product of two vectors — a compact representation that requires storing only as many parameters as the sum of the matrix's dimensions, rather than their product. A rank-eight matrix requires eight times as many parameters as rank-one, but still far fewer than the full matrix. If the actual correction learned by fine-tuning has an effective rank of eight — meaning it lies almost entirely within an eight-dimensional subspace of the space of all possible corrections — then representing it as a product of two small matrices loses almost no information while achieving an enormous parameter reduction.

The LoRA paper provided empirical validation for this hypothesis through a series of experiments that directly measured the dimensionality of the optimal fine-tuning correction for several large language models. By analyzing the singular value spectrum of the learned correction matrices, the authors confirmed that the vast majority of the explained variance was concentrated in a very small number of dimensions — often fewer than eight — regardless of the original matrix dimensions. This is the empirical foundation that justifies using low-rank matrices to represent fine-tuning corrections in practice.

### 3.2 The Rank and the Scaling Factor

The rank of the adapter — typically called r — controls the trade-off between the adapter's expressive power and its size. A rank of one produces the smallest possible adapter but can only represent a one-dimensional correction subspace. A rank of sixty-four produces a more expressive adapter at thirty-two times the storage cost. In practice, rank values of eight or sixteen capture most of the benefit of fine-tuning for standard NLP tasks, and rank-eight adapters are the most common choice in publicly released checkpoints.

The LoRA paper introduced a scaling factor, conventionally written as alpha divided by r, that controls the magnitude of the adapter's correction relative to the base model's output. This scaling is important because the optimal correction magnitude varies across tasks and models, and the scaling factor provides a way to control it without changing the adapter's rank or architecture. When the scaling factor equals one — which happens when the alpha hyperparameter is set equal to r — the adapter's correction enters the forward pass at unit weight. In the widely used adapter for Qwen3-0.6B used in Layer 20's implementation, the rank is eight and alpha is thirty-two, producing a scaling factor of four. This means the adapter's correction enters the forward pass at four times the unit weight — a strong adaptation that reflects the degree of task specialization the fine-tuning learned.

### 3.3 Which Layers to Adapt

A transformer consists of attention layers and feed-forward layers, each containing multiple linear projections. LoRA can be applied to any of these projections. The original paper experimented with different configurations and found that targeting more projections at a lower rank consistently outperformed targeting fewer projections at a higher rank, given the same total parameter budget. This finding has been robustly confirmed by subsequent work. Modern adapters typically target all seven projection matrices in each transformer layer — the query, key, value, and output projections in the attention module, and the gate, up, and down projections in the feed-forward module — at a relatively low rank. The adapter for the Qwen3-0.6B model used in Layer 20's minimal implementation follows this convention, applying rank-eight corrections to all seven projection types across all twenty-eight layers of the model.

---

## 4. The Compute Challenge: Batching Across Different Adapters

Establishing that the base model and adapters should be computed separately addresses the storage problem but introduces a new computational challenge. When a batch of requests arrives, each specifying a different adapter, the base model forward pass is trivially batched — all tokens execute the same operations against the same weights, and standard GPU matrix multiplication handles this efficiently. The adapter computation, however, must vary per request: each token must be multiplied against the adapter weights corresponding to its specific adapter, which are different for each request in the batch.

### 4.1 The Naive Implementation and Its Failure Mode

The straightforward approach to this problem is to loop over the adapters present in the current batch, select the tokens belonging to each adapter, compute that adapter's delta for those tokens as a separate matrix multiplication, and then sum the results. If the batch contains tokens belonging to ten different adapters, this approach launches ten separate matrix multiplication operations.

The problem is that each GPU matrix multiplication operation carries overhead beyond the actual arithmetic cost. The GPU's execution engine must schedule the operation, allocate registers, load the operands into cache, and manage synchronization with subsequent operations. For large matrix multiplications — the kind that dominate the base model forward pass — this overhead is negligible relative to the arithmetic cost. For the small matrix multiplications that arise from adapter delta computation with a low rank, the overhead is substantial. A rank-eight adapter delta requires a down-projection to eight dimensions followed by an up-projection back to the model dimension — two matrix multiplications whose arithmetic intensity is orders of magnitude lower than the base model's attention and feed-forward operations. Launching ten of these, sequentially, for ten different adapters in a batch, wastes far more time on GPU scheduling overhead than on actual arithmetic.

The consequence is that naive multi-adapter batching collapses GPU utilization. The GPU processes the base model forward pass at high efficiency, then spends most of the remaining time launching and completing tiny per-adapter matrix multiplications, each with startup overhead comparable to the actual computation time. Measurements from Punica's evaluation confirmed this: a naive multi-model serving approach using vLLM achieved a throughput that was twelve times lower than the optimized approach described next.

### 4.2 The Segmented Gather Matrix-Vector Multiplication

Punica, published at MLSys 2024 by Chen and colleagues from the University of Washington and Duke University, introduced the computational primitive that all production multi-LoRA systems now use: the Segmented Gather Matrix-Vector Multiplication, or SGMV.

The insight behind SGMV is that all the adapter delta computations for a given forward pass can be fused into a single GPU kernel launch, rather than launched as separate operations per adapter. The kernel takes as input the entire batch of token activations, a mapping that associates each token with its adapter, and a contiguous buffer containing the weight matrices for all adapters present in the batch. It then groups tokens by adapter — this is the "segmented" part of the name — and computes the delta for each group using the appropriate weights. The "gather" refers to the operation of collecting the weight matrices for each segment from potentially non-contiguous memory locations. The entire operation — across all adapters, all tokens, all groups — completes in a single kernel invocation.

The impact on GPU utilization is immediate and substantial. The scheduling overhead that dominates naive per-adapter looping is eliminated: the GPU pays it once, not ten times. The kernel can interleave computation across segments in a way that keeps GPU threads occupied throughout, avoiding the idle periods between separate kernel launches. The kernel also enables prefetching of adapter weight data for upcoming segments while the current segment is being computed, improving effective memory bandwidth utilization. Punica's evaluation confirmed a twelve times throughput improvement over the naive approach, at the cost of only two milliseconds of additional latency per token — an overhead that is constant regardless of how many adapters are in the batch.

The SGMV kernel is designed for the prefill phase, where a batch can contain many tokens from the same request. For the decode phase, where each request contributes exactly one token per step, a related kernel called BGMV — Batched Gather Matrix-Vector Multiplication — is optimized for the different memory access pattern that arises when the batch consists entirely of single-token computations. Both are now standard components of SGLang, vLLM, and LoRAX.

### 4.3 The Masked Alternative and Its Trade-offs

For a system serving exactly one adapter alongside the base model — the design of Layer 20's pedagogical implementation — a simpler approach than SGMV is available. Instead of computing the adapter delta only for tokens that need it, compute the delta for every token in the batch, and then multiply the result by a binary mask: one for tokens that are using the adapter, zero for tokens that are using only the base model. Tokens with a zero mask contribute no adapter correction to the output; tokens with a one mask receive the full correction.

This masked approach is computationally wasteful but algorithmically correct. For a batch of one hundred tokens where ten need the adapter and ninety do not, the delta computation is performed for all one hundred tokens, but ninety of those computations are immediately zeroed out. Ninety percent of the adapter computation is wasted. For a single-adapter system under typical mixed traffic, this waste is acceptable relative to the simplicity benefit. For a system serving hundreds of adapters with arbitrary batch composition, the waste is intolerable — SGMV becomes mandatory.

The masked approach is the correct pedagogical choice for Layer 20 precisely because it makes the logic visible: every token in the batch is treated uniformly by the matrix operations, and the adapter versus base distinction is enforced only at the final addition step. SGMV is the correct production choice because it avoids computing corrections for tokens that will only mask them away.

---

## 5. The Memory Management Challenge: More Adapters Than VRAM

The SGMV kernel resolves the compute problem. A GPU using SGMV can efficiently handle batches containing tokens from dozens of different adapters, with throughput near what would be achieved if only one adapter were present. But a new problem immediately arises: each adapter's weight matrices must be loaded into GPU memory before SGMV can access them. For a system serving one hundred adapters simultaneously, all one hundred adapters' weight matrices must be in GPU high-bandwidth memory at the same time.

### 5.1 The VRAM Budget

An H100 GPU with eighty gigabytes of HBM has, after reserving space for the base model's fourteen gigabytes, approximately sixty-six gigabytes of space available for the KV cache and adapter weights combined. A rank-eight adapter applied to all seven projection modules of a 7-billion parameter model occupies approximately fifty megabytes. One hundred such adapters require five gigabytes. This is readily accommodated — after all adapter weights are loaded, fifty-nine gigabytes remain for the KV cache, more than enough for typical workloads.

The picture changes as the adapter pool grows. One thousand adapters require fifty gigabytes. At this point, only sixteen gigabytes remain for the KV cache, severely limiting the number of concurrent requests the system can hold. Two thousand adapters would require one hundred gigabytes — more than the GPU's total capacity, even before accounting for the base model. The simple approach of loading all adapters into GPU memory simultaneously fails at scale.

### 5.2 The Two-Level Cache Hierarchy

The solution is the same technique that CPU operating systems have used for decades to address the mismatch between the working set of active pages and the size of physical memory: a two-level cache hierarchy with demand paging.

Punica, the first system to demonstrate efficient multi-LoRA serving, assumed all adapters fit in GPU memory simultaneously. S-LoRA, published at MLSys 2024 by Sheng and colleagues from UC Berkeley, extended Punica's approach to support adapter catalogs orders of magnitude larger than GPU memory by keeping adapters in CPU DRAM as the secondary tier and loading them to GPU VRAM on demand when a request that needs them arrives.

The mechanics are straightforward. The GPU holds a pool of adapter weight matrices — the working set of currently active adapters. The CPU holds a larger pool of recently used adapters that have been evicted from the GPU. Remote storage holds the full adapter catalog, arbitrarily large. When a request arrives for an adapter that is not in the GPU pool, the adapter is fetched from whichever tier holds it — CPU if it has been recently used and evicted, remote storage if it has not been used recently — and loaded into the GPU pool. If the GPU pool is full when a new adapter must be loaded, the least recently used adapter is evicted to CPU memory. The KV-cache analogy is exact: GPU is page frames, CPU is swap space, and the full catalog is the backing store.

S-LoRA's experimental results demonstrate the practical impact of this design. On a single A100 80GB GPU serving LLaMA-7B with rank-8 adapters, S-LoRA supported more than two thousand concurrent active adapters — compared to approximately sixteen for naive per-adapter loading that fills VRAM — at four times the throughput of HuggingFace PEFT, which serves only one adapter at a time. The "several orders of magnitude" improvement in supported adapter count reflects the difference between being constrained by GPU VRAM capacity and being constrained by disk capacity.

### 5.3 Unified Paging: Managing Two Competing Consumers

S-LoRA's most technically important contribution is not simply the two-level cache; it is the observation that adapter weights and KV cache entries compete for the same GPU memory, and that managing them as separate pools — with fixed pre-allocated sizes for each — is inherently wasteful and fragile.

The problem with separate pools is over-provisioning. If you reserve twenty gigabytes for adapters and forty gigabytes for KV cache, you will encounter situations where the adapter pool is half empty while the KV cache is full and new requests are being rejected for insufficient cache space. The twenty gigabyte adapter reservation is idle while the KV cache is the bottleneck, and there is no mechanism to transfer the unused reservation across the pool boundary.

S-LoRA's Unified Paging treats adapter weight pages and KV cache pages as equivalent items in a single shared pool. Both types of pages can occupy any slot in the pool. When the pool is full and a new item needs space — whether it is a KV cache page for a new request or an adapter weight page for a newly loaded adapter — the eviction policy selects the least recently used item regardless of whether it is a cache page or an adapter page. The pool dynamically rebalances: during periods of heavy concurrent generation (many active requests accumulating KV cache), adapter pages are evicted to CPU; during periods with many adapter cold starts and light generation load, the pool fills with adapter pages. Neither resource artificially constrains the other.

This design also handles the adapter size heterogeneity problem. Adapters with different ranks have different sizes — a rank-four adapter is half the size of a rank-eight adapter, which is a quarter the size of a rank-thirty-two adapter. Managing arbitrary-sized allocations in a fixed-capacity pool leads to fragmentation: after a sequence of load and evict operations with adapters of different sizes, the free space is scattered in unusable small gaps. Unified Paging addresses this by using fixed-size memory pages, allocating adapters as collections of pages. Because all pages are the same size, any page can be used for any purpose, and the pool never fragments regardless of the size distribution of the adapters it has served.

---

## 6. The Merge-or-Separate Decision: When Batching Is Not Always Optimal

S-LoRA established that the unmerged, SGMV-based approach — keeping adapter and base model separate, batching the adapter computation across adapters — is the correct default architecture for multi-LoRA serving. But a 2024 paper from Peking University and Shanghai AI Lab, dLoRA, published at USENIX OSDI 2024, challenged this default and showed that it is not always optimal.

### 6.1 The Hidden Cost of Always-Unmerged Serving

The SGMV kernel adds a small but real cost to every forward pass: the down-projection and up-projection that compute the adapter delta, even when they are efficiently fused across adapters, are additional computation that the base model forward pass does not require. For a system where all requests in the batch use the same adapter, this cost provides no batching benefit — there is only one adapter to batch across. You are paying the adapter computation cost on every token and gaining no efficiency from doing so over the alternative of merging the adapter into the base model weights before serving.

The merged alternative eliminates the adapter computation cost entirely. When the adapter is merged, the forward pass is a single matrix multiplication against the combined weights, with zero additional overhead. A merged model is computationally identical to a model that was traditionally fine-tuned — no extra layers, no separate computations, no overhead. For a workload where virtually all requests use the same adapter, merging is strictly better than the SGMV approach.

The dLoRA paper formalized the conditions under which merging is preferable. In workloads where the adapter request distribution is highly skewed — most requests using one of a small number of dominant adapters, with a long tail of adapters receiving occasional traffic — dedicated merged replicas for the dominant adapters outperform the SGMV approach on average latency, because the dominant-adapter requests avoid the SGMV overhead entirely. The long-tail adapters, which receive infrequent traffic, can be served by a shared SGMV-based pool. The optimal system uses both modes simultaneously.

### 6.2 Dynamic Switching via Credits

The dLoRA system implements this insight with a continuous monitoring mechanism: a credit system that tracks the request arrival rate per adapter over time. When an adapter's credit score — a weighted measure of recent request volume — exceeds a threshold, dLoRA merges that adapter into the base weights of a dedicated worker replica and routes all requests for that adapter to that replica. When the credit score falls below a lower threshold (maintaining hysteresis to prevent oscillation), the adapter is unmerged and the replica returns to the shared SGMV pool.

This design adapts to real workload shifts. An adapter that is quiet at night but receives heavy traffic during business hours will be in merged mode during the day and unmerged mode at night. An adapter that temporarily becomes popular due to a product launch will be promoted to a merged replica within minutes of the traffic spike, and returned to the shared pool when the spike subsides. The credit system is, in essence, a real-time signal that answers the question that dLoRA identified as unanswered by all prior work: for this specific adapter at this moment in time, is it cheaper to pay the merge-and-dedicate overhead or to absorb the SGMV per-token overhead?

dLoRA's experimental results on LLaMA-2-7B and LLaMA-2-13B showed up to 57.9 times throughput improvement over vLLM and 1.8 times lower average latency compared to S-LoRA. The improvement over S-LoRA reflects exactly the benefit of avoiding SGMV overhead for high-traffic adapters — an improvement that S-LoRA's always-unmerged policy leaves on the table.

### 6.3 Request-Adapter Co-Migration

dLoRA also identified a second problem that S-LoRA and Punica ignored: load imbalance across worker replicas. In a multi-replica deployment, requests are initially distributed across replicas by the router. But requests have different output lengths — some generate ten tokens, some generate a thousand. Over time, replicas that received long-output requests accumulate larger KV caches and fall behind replicas that received short-output requests. The imbalance compounds: the overloaded replica accepts fewer new requests (because its KV cache is nearly full), while the underloaded replica sits partially idle.

The solution dLoRA introduces is request-adapter co-migration: moving an in-progress request — along with its current KV cache — from an overloaded replica to an underloaded one. The KV cache is typically small relative to the total traffic the migration prevents accumulating, and the adapter weights can be co-transferred at the same time if the destination replica does not already hold them. The migration decision is made by a cost model that compares the expected latency benefit of rebalancing against the latency cost of the transfer, ensuring migration is only performed when it is net positive.

---

## 7. The Cold-Start Problem: Hiding Adapter Loading Latency

Both Punica and S-LoRA address adapter loading in similar ways: when a request arrives for an adapter that is not in the GPU pool, queue the request and start loading the adapter. Once loading completes, process the request. This approach is correct but introduces what the CaraServe paper, published in 2024 by Li and colleagues, called the cold-start problem: the time spent loading an adapter from CPU to GPU is dead time that the user experiences as TTFT latency, and for systems with large adapter catalogs and diverse access patterns, cold starts are frequent.

### 7.1 The Time Structure of a Cold-Start Request

The timeline of a cold-start request in S-LoRA is sequential: the adapter is loaded from CPU to GPU, consuming hundreds of milliseconds depending on adapter size and PCIe bandwidth; only then does prefill begin; only after prefill does decode begin. The adapter loading time is entirely outside the useful work the GPU performs. The user is waiting, the GPU is transferring data, and neither computation nor generation is happening.

CaraServe's insight is that the GPU's prefill computation and the PCIe transfer are not mutually dependent. The GPU can run prefill on a different request while the PCIe bus transfers the adapter for this request — but this is not what standard scheduling does. Standard scheduling, which processes one request at a time within a serving slot, blocks the GPU on the PCIe transfer. CaraServe's alternative is to begin the cold-start request's prefill on the CPU in parallel with the GPU-bound PCIe transfer. The CPU is available, the adapter weights are already in CPU memory (the intermediate tier), and although the CPU is slower at attention computation, the parallel overlap means the effective TTFT is the maximum of the CPU prefill time and the GPU loading time rather than their sum.

### 7.2 CPU-Assisted Prefill

CaraServe's CPU-assisted prefill mechanism works as follows. When a request arrives for a cold adapter, CaraServe begins the PCIe transfer of the adapter weights from CPU memory to GPU memory. Simultaneously, it begins executing the request's prefill on the CPU, using the CPU-resident adapter weights that are still present before transfer completes. The CPU builds the KV cache for the input tokens as it processes them. When the PCIe transfer completes and the adapter is available on the GPU, CaraServe transfers the CPU-built KV cache to the GPU and switches the request to GPU-based decode. The total TTFT is reduced to the maximum of the CPU prefill time and the GPU loading time — an overlap that is beneficial whenever loading takes longer than the GPU would have taken to perform prefill alone.

The mechanism is not universally beneficial. When adapter loading is fast (small adapter, high PCIe bandwidth, short prompt) or when the CPU is particularly slow relative to the GPU, the sequential approach may actually complete faster than the overlapped approach due to the overhead of managing two simultaneous operations. CaraServe implements a per-adapter profiling system that determines the crossover point — the prompt length above which CPU-assisted prefill becomes beneficial — and applies it adaptively based on the adapter's rank and the current system load.

CaraServe's evaluation on LLaMA-7B showed 1.4 times lower average latency than S-LoRA and 99 percent SLO attainment even under high cold-start rates — compared to approximately 85 percent for S-LoRA without CPU assistance. The 99 percent figure is remarkable for a system serving thousands of adapters, where cold starts cannot be avoided.

### 7.3 Rank-Aware Scheduling

CaraServe also introduced a scheduling policy that prior work had overlooked: request priority should account for adapter rank. A rank-sixty-four adapter requires more loading time than a rank-eight adapter. A request with a rank-sixty-four adapter and a tight TTFT deadline needs to be scheduled earlier than a request with a rank-eight adapter and the same deadline, because it needs more lead time to complete loading before the deadline. A scheduler that treats all requests identically regardless of adapter rank will systematically under-prioritize high-rank adapter requests and over-miss their TTFT SLOs.

CaraServe's rank-aware priority assignment combines the adapter's rank, the estimated loading and computation time, and the request's SLO deadline into an urgency score. Requests with higher urgency — defined as the ratio of estimated processing cost to time remaining before the deadline — are scheduled first. This is a standard earliest-deadline-first variant with cost awareness, applied to the specific two-dimensional cost structure of adapter loading plus prefill computation.

---

## 8. The Serverless Dimension: Backbone Sharing and Predictive Loading

The cost problem that motivates multi-LoRA serving becomes especially acute in serverless deployment models, where each serving function runs in an isolated container, each container loads its own model, and users pay for the full load time and memory footprint regardless of whether the resources are shared.

### 8.1 The Weight Redundancy Problem at Scale

In a serverless deployment without backbone sharing, every function instance that serves a LoRA-based model loads a complete copy of the base model. If ten LoRA functions are concurrently active — one for each customer, or one for each product feature — ten complete copies of the base model occupy GPU memory across the system, even though all ten base models are identical. For a 14-gigabyte base model, ten concurrent functions consume 140 gigabytes of GPU memory for data that is 99 percent redundant. The 50-megabyte adapters that actually differentiate the functions represent 0.35 percent of the total memory footprint.

ServerlessLoRA, published in 2025 by Sui and colleagues, quantified this directly and proposed secure backbone sharing: a mechanism that allows multiple LoRA serving functions to share a single base model instance while maintaining complete isolation of their adapter weights, KV caches, and intermediate activations. Each function can only read the shared base model — it cannot read or write any other function's private state. The shared base model is loaded once; each function contributes only its adapter's fifty megabytes. For ten concurrent functions, the shared architecture requires approximately 14.5 gigabytes instead of 140 gigabytes — a 90 percent reduction in VRAM consumption, which directly translates to cost reduction because fewer GPU instances are needed.

ServerlessLoRA's evaluation on an industrial workload trace showed an 86 percent reduction in time-to-first-token and an 89 percent reduction in cost per million tokens compared to the baseline serverless deployment, driven almost entirely by backbone sharing and its consequence: many more LoRA functions can coexist on the same GPU without contention.

### 8.2 Reactive vs. Proactive Loading

Both ServerlessLoRA and the earlier S-LoRA use reactive adapter loading: an adapter is loaded only after a request for it arrives. For workloads with predictable temporal patterns — many real services show consistent daily, weekly, and event-driven access cycles — reactive loading guarantees that the first request for any adapter after an idle period will experience a cold start, regardless of how predictable that request was.

Predictive-LoRA, published in 2025 by Tang and colleagues, addressed this with proactive loading based on traffic prediction. Their system uses a lightweight neural network — specifically a Long Short-Term Memory network, chosen for its ability to model temporal dependencies in time-series data — that ingests historical per-adapter request rates and predicts, at each time step, which adapters are likely to be needed in the near future. Adapters predicted to be above a demand threshold are prefetched from CPU to GPU memory during idle periods, before the requests that need them arrive.

The LSTM is intentionally small — a few hundred parameters, executing in under one millisecond on CPU — and is continuously updated via online learning from recent traffic history. Its advantage over simpler predictors is its ability to model long-range temporal patterns. A moving average predictor can detect that adapter access is increasing but cannot recognize that adapter traffic follows a weekly cycle. An LRU policy makes no predictions at all — it reacts only to what has happened, not what is about to happen. The LSTM can learn that a specific adapter is popular on Tuesday mornings and Wednesday afternoons, and can begin prefetching it on Monday evening. Predictive-LoRA's evaluation on real Microsoft Azure Functions traces showed 68 percent fewer cold starts than S-LoRA's reactive policy and 1.52 times higher throughput.

### 8.3 Fragmentation and Page-Based Memory Management

Predictive-LoRA also addressed a fragmentation problem that S-LoRA's Unified Paging partially solves but does not eliminate in its most general form. When adapters of highly variable rank are loaded and evicted repeatedly, the free space in the GPU memory pool can become scattered in fragments that are individually too small to accommodate new allocations, even when the total free space is sufficient. This is the classical memory fragmentation problem, and it can cause out-of-memory failures even when overall utilization is well below 100 percent.

Predictive-LoRA's solution is borrowed directly from operating systems design: fixed-size pages, with all allocations rounded up to a multiple of the page size. Because every page is the same size, any combination of free pages can always accommodate a new allocation of the same total size, regardless of how fragmented the free space is geometrically. The VRAM utilization cost of the fixed-page approach — rounding up small adapters to a larger page size wastes some space — is outweighed by the elimination of fragmentation-induced failures. Predictive-LoRA reported VRAM utilization consistently above 87 percent across heterogeneous adapter rank distributions, compared to 60 to 70 percent for S-LoRA under similar conditions.

---

## 9. The Training-Serving Unification Challenge

All systems discussed to this point — Punica, S-LoRA, dLoRA, CaraServe, ServerlessLoRA, and Predictive-LoRA — address a single workload: inference serving. They assume that adapter weights are static artifacts, computed offline by a training process and deployed to a serving system. But an increasingly common production pattern breaks this assumption: continuous learning systems, reinforcement learning from human feedback pipelines, and personalization systems all require the serving infrastructure to simultaneously receive new training data, update adapter weights, and serve the updated adapters to live traffic. Training and serving must coexist on the same hardware.

### 9.1 The Resource Contention Problem

When training and serving run simultaneously on the same GPU, they compete for the same resources. Training a LoRA adapter requires forward passes to compute activations, backward passes to compute gradients, and optimizer steps to update weights — all of which consume GPU memory proportional to the model size and the batch size used for training. Serving requires forward passes and KV cache storage for concurrent users. In a naive implementation, either task must be paused while the other runs, or both run slowly due to memory pressure.

Loquetier, published at NeurIPS 2025 by Zhang and colleagues from Nanjing University, addressed this unification problem with an architectural abstraction they called the Virtualized Module. Each layer in the model that has LoRA applied to it is wrapped in a Virtualized Module that understands both the serving forward pass and the training forward and backward pass. The Virtualized Module maintains a pool of adapters, each with a mode — serving or training — and applies the appropriate computation path when processing each token. A serving token computes the adapter delta without gradient tracking. A training token computes the adapter delta with full gradient tracking and contributes to the backward pass for weight updates.

The key consequence is that training tokens and serving tokens can coexist in the same forward pass batch. The GPU computes the base model output once for all tokens and then branches at the adapter computation: serving tokens receive their delta without gradient overhead, training tokens receive their delta with gradient tracking, and the combined batch occupies the GPU at higher utilization than either workload alone would achieve. Loquetier's evaluation showed three times the inference throughput of the state-of-the-art co-serving system and 46.4 times the SLO attainment of HuggingFace PEFT on unified training-plus-serving workloads. The 46.4 times improvement reflects how catastrophically PEFT's sequential single-adapter approach fails when mixed with concurrent serving traffic.

---

## 10. The MoE Frontier: When Adapters Are No Longer Small

Every system described in this chapter makes an assumption that is central to the entire architecture: adapter weights are small. A rank-eight adapter applied to a 7-billion parameter dense model occupies approximately fifty megabytes. Against the fourteen-gigabyte base model, this is 0.36 percent of the total weight volume. Adapter loading is fast. The GPU pool can hold dozens of adapters alongside the base model. The KV cache competition with adapter weights is manageable.

This assumption breaks down completely for Mixture-of-Experts models.

### 10.1 How MoE Changes the Adapter Size Arithmetic

In a Mixture-of-Experts architecture, the feed-forward component of each transformer layer is replaced by a collection of expert networks, each a full feed-forward layer. Each token, during the forward pass, is routed to a small subset of these experts — typically two or four out of sixty-four — while the remaining experts are not invoked. The total parameter count of all experts is much larger than a dense model's feed-forward layer, but only a fraction of those parameters are activated per token, keeping per-token computation manageable.

When LoRA is applied to an MoE model, the adapter must cover all expert layers in the model, not just the feed-forward layer. If the model has sixty-four experts and thirty-two transformer layers, and LoRA is applied to all feed-forward projections, the adapter contains sixty-four times as many feed-forward parameters per layer as a dense model adapter would. For a model like DeepSeek-V2 or Mixtral-8×7B with rank-eight adapters applied to all projections, a single adapter can occupy several gigabytes — not fifty megabytes. The assumption that adapters are small, which the entire preceding architecture relies on, is violated by roughly two orders of magnitude.

### 10.2 The Disaggregated LoRA Solution

InfiniLoRA, published in April 2026 by Chen and colleagues, proposed a response to the MoE scaling problem that mirrors the Prefill-Decode Disaggregation solution discussed in the companion chapter on PD disaggregation: move the problem component — the LoRA computation — onto dedicated separate hardware.

In InfiniLoRA's architecture, a cluster of specialized LoRA Server GPUs holds all adapter weights and is responsible exclusively for computing adapter deltas. The base model GPUs hold the frozen base model weights and the KV caches, and run the base model forward passes. When a base model GPU needs the adapter delta for a given layer during a forward pass, it sends the intermediate activation tensor to the LoRA Server, which computes the delta and returns it. The base model GPU applies the delta and continues to the next layer. The LoRA computation is entirely offloaded.

The consequences are significant. The base model GPUs no longer need to hold any adapter weights, which frees VRAM that can be allocated to larger KV caches, enabling larger concurrent batch sizes and higher GPU utilization. The LoRA Server can be provisioned independently of the base model cluster — if adapter catalog size grows, only the LoRA Server needs to scale, not the base model deployment. The LoRA computation can be parallelized across LoRA Server GPUs using tensor parallelism, matching the degree used by the base model cluster.

InfiniLoRA's implementation uses GPU-initiated communication for the activation transfers — a mechanism that allows the GPU to directly send and receive data via RDMA without CPU involvement, reducing communication latency by approximately an order of magnitude compared to CPU-mediated transfers. On a cluster of H100 GPUs serving DeepSeek-V2, InfiniLoRA achieved 3.05 times the serviceable request rate under strict latency SLOs compared to S-LoRA's coupled approach, and 100 percent SLO attainment compared to S-LoRA's 46 percent. The improvement reflects the fundamental change in memory pressure: base model GPUs in the InfiniLoRA deployment used only 64 percent of their VRAM for base weights and KV cache, compared to 90 percent in S-LoRA's coupled approach — that additional 26 percent of VRAM went directly to larger KV caches and larger batch sizes.

### 10.3 The Disaggregation Lineage

InfiniLoRA represents a natural extension of the disaggregation principle to the LoRA axis. The same logic that motivated separating prefill from decode — that two computations with different resource profiles should not share hardware — applies here to the base model computation and the adapter delta computation. For dense models with small adapters, the two computations share hardware comfortably because the adapter's resource footprint is negligible. For MoE models with large adapters, the adapter's resource footprint is significant enough to justify dedicated hardware, just as decode's memory-bandwidth profile eventually justified dedicated hardware separate from prefill.

The progression from Layer 20's minimal single-adapter implementation to full production multi-LoRA serving to InfiniLoRA's disaggregated design mirrors the progression from simple batch inference to PD disaggregation in the base model layer. In both cases, the path begins with collocated shared-hardware systems that are simple but suboptimal, proceeds through optimization of the shared approach (SGMV, Unified Paging, chunked prefill), and arrives at dedicated hardware for each workload component — the only architecture that eliminates the resource competition entirely.

---

## 11. Production Reality: The Systems That Exist Today

The research results described in the preceding sections are not speculative. The core techniques — SGMV-based multi-adapter batching, LRU adapter eviction, Unified Paging — are deployed in all three major inference frameworks: SGLang, vLLM, and LoRAX (now a production offering from Predibase). Their deployment in these frameworks means they are running in production at thousands of organizations worldwide.

SGLang implements both the Triton-based SGMV kernel and a Chunked SGMV variant for different serving configurations. Its LoRA serving is configured through a set of command-line arguments that map directly to the theoretical concepts: the maximum number of adapters in the GPU pool at once, the eviction policy (LRU or FIFO), the kernel backend, and the maximum supported rank. These parameters directly trade off KV cache space against adapter pool size, and the documentation explicitly notes that the optimal configuration depends on the specific traffic distribution of the deployment.

vLLM implements a companion BGMV kernel optimized for decode-phase adapter computation and an LRU-evicted adapter pool managed by a LoRAManager component. Its LoRAResolver plugin interface allows operators to define custom adapter loading policies — fetching adapters from S3, a private HuggingFace mirror, or a database — using the same interface regardless of the source. The in-place reload capability, which replaces adapter weights atomically without changing the adapter's registration, enables reinforcement learning pipelines that continuously improve adapters while serving production traffic.

LoRAX, Predibase's open-source multi-adapter inference server, combines multi-adapter batching with structured output generation (via Outlines) and speculative decoding (via Lookahead LoRA) in a single system designed for production extraction and structured inference tasks. Its benchmarks, published on AWS SageMaker with Mistral-7B and fifty simulated adapter variants, confirmed that throughput with fifty concurrent adapters is statistically indistinguishable from throughput with a single adapter — validating the theoretical claim that SGMV's overhead is constant with respect to the number of adapters in the batch.

---

## 12. The Decision Framework: Choosing the Right Architecture

The nine preceding sections have developed the conceptual landscape for multi-LoRA serving from first principles. This final section synthesizes the decision criteria.

### 12.1 When Does Separation Pay Off?

The decomposed forward pass approach — keeping base model and adapters separate — is strictly superior to merged serving whenever two conditions hold simultaneously: the number of adapters required is large enough that loading each as a separate model instance is economically unreasonable, and the batch mix contains requests from multiple different adapters frequently enough that SGMV's ability to batch heterogeneous adapter computation provides meaningful throughput benefit.

The first condition is almost always satisfied for systems with more than eight to ten adapters. Below that count, pre-loading all adapters at server startup into GPU memory is feasible, and the full weight of multi-adapter infrastructure (LRU eviction, SGMV kernels, two-level cache management) adds complexity without proportionate benefit. Above that count, the economics of full per-adapter model instances become untenable, and the architectural investment in multi-adapter serving pays for itself rapidly.

The second condition is satisfied by virtually any realistic serving system with diverse users or applications. A system where 90 percent of requests go to one adapter and 10 percent to another will benefit less than a system with uniform adapter access, but even the highly skewed case benefits from avoiding the storage and instance cost of separate deployments — the savings from sharing the base model exist regardless of batch composition.

### 12.2 When Is the Merge Preferable?

dLoRA's insight — that merging is preferable for high-traffic adapters — introduces a nuance to the above framework. If the deployment has a small number of dominant adapters (those receiving more than, say, 20 percent of traffic individually) alongside a long tail of infrequent adapters, a hybrid architecture is optimal: merge the dominant adapters into dedicated worker replicas for zero-overhead hot-path serving, and maintain a shared SGMV pool for the long tail. This is exactly what dLoRA implements with its credit system, and the result is substantially lower average latency than always-unmerged serving at the same hardware budget.

For deployments with genuinely uniform adapter access — no adapter receiving more than a few percent of traffic — the always-unmerged SGMV approach is optimal, because no adapter is dominant enough to justify the merge-and-dedicate overhead.

### 12.3 Cold Starts and SLO Guarantees

If the deployment has a large adapter catalog (more than a few hundred adapters) and the catalog is more than can be fully loaded into CPU memory, cold starts from remote storage are inevitable. CaraServe's CPU-assisted prefill reduces the latency cost of these cold starts but does not eliminate it. Predictive-LoRA's proactive loading, if the traffic pattern is sufficiently regular and predictable, can reduce cold start frequency to near zero by loading adapters before they are needed.

For deployments where strict TTFT SLO compliance is required even at high cold-start rates, some combination of CPU-assisted prefill (to reduce cold-start cost) and predictive loading (to reduce cold-start frequency) is necessary. Neither alone is sufficient if the cold-start frequency is high and the SLO is tight.

### 12.4 The MoE Exception

For deployments on Mixture-of-Experts base models, the entire preceding framework should be reassessed with the corrected adapter size in mind. If adapter weights are gigabytes rather than megabytes, the KV cache competition is severe, the GPU pool holds only a handful of adapters simultaneously, and the loading-time arithmetic changes dramatically. For these deployments, InfiniLoRA's disaggregated LoRA Server approach — decoupling adapter computation onto dedicated hardware — is the architecture that matches the problem's actual structure. The correct analogy is PD disaggregation applied to the LoRA axis rather than the prefill-decode axis.

### 12.5 The Correct Baseline Comparison

It is worth closing with a quantitative reminder of what is at stake. A service with one thousand customer-specific adapters of a 7-billion parameter model, using dedicated per-adapter GPU instances at current H100 pricing, costs approximately two million dollars per month in GPU infrastructure. The same service using a well-configured multi-adapter serving system with ten to twenty shared GPU servers costs approximately twenty thousand dollars per month — a 98 percent reduction. These numbers assume 24-hour operation at modest but real utilization rates.

The savings are not marginal. They are the difference between a product line that is economically viable and one that is not. This is the ultimate reason why the research community produced eight papers on multi-LoRA serving in four years, why all three major inference frameworks have full LoRA serving implementations, and why the technique has moved from academic publication to broad production deployment in less time than many software engineering best practices take to propagate through the industry.
