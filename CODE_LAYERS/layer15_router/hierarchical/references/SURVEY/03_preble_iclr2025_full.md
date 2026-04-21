PREBLE: EFFICIENT DISTRIBUTED PROMPT SCHEDULING FOR LLM SERVING 

## **Vikranth Srivatsa** _[∗]_ **, Zijian He** _[∗]_ **, Reyna Abhyankar, Dongming Li, Yiying Zhang** 

University of California, San Diego 

## ABSTRACT 

Prompts to large language models (LLMs) have evolved beyond simple user questions. For LLMs to solve complex problems, today’s practices are to include domain-specific instructions, illustration of tool usages, and/or long context such as textbook chapters in prompts. As such, many parts of prompts are repetitive across requests. Recent works propose to cache and reuse KV state of prompts. However, they are all confined to a singleGPU optimization, while production LLM serving systems are distributed by nature. 

This paper proposes Preble, the first distributed LLM serving platform that targets and optimizes for prompt sharing. We designed a distributed scheduling system that co-optimizes KV state reuse and computation load-balancing with a new scheduling algorithm and a hierarchical scheduling mechanism. Our evaluation of Preble with real workloads and request arrival patterns on two open-source LLMs shows that Preble outperforms the SOTA serving systems by 1.5 _×_ to 14.5 _×_ on average latency and 2 _×_ to 10 _×_ on p99 latency. 

## **1 Introduction** 

Recently, new capabilities and use cases of LLMs have created two common features not seen in traditional LLM usages. First, prompts to LLMs are significantly longer than generated sequences. For example, questions about a long document (Li et al., 2023a) or a video clip Xiao et al. (2021) are answered by LLMs with short answers. As another example, detailed instructions and illustrations for LLMs are vital in accomplishing complex tasks like solving advanced math problems Yao et al. (2023b). The latest LLMs like OpenAI o1 have built-in reasoning capabilities; even though the end users’ prompts do not need to be long, these models internally add more context like chain-of-thought prompting Wei et al. (2024), lengthening the total context length OpenAI (2024a). As demonstrated by o1 and other model usage, oftentimes, the longer a prompt is, the better quality the model can generate. Long prompts with short generations imply that the prefill phase significantly outweighs the decoding phase. Thus, improving the prefill phase performance is crucial to the overall performance of LLM serving systems. 

Second, prompts are partially shared across requests. For example, a long document or video is often queried many times with different questions Li et al. (2023a); different requests using the same tools share tool instructions in tool-augmented LLMs Hao et al. (2023); chain- or tree-structured prompting calls an LLM in steps, with each subsequent step reusing context from previous steps Yao et al. (2023b); Zhang et al. (2023); Yao et al. (2023a). Real-world production systems like Anthropic have also reported the wide existence of long and shared prompts Anthropic (2024). 

Recent works Zheng et al. (2023b); Gim et al. (2024); Moore & Li (2024) propose to cache computed keyvalue (KV) state in GPU memory and reuse the cached KV when a new request sharing a prompt prefix 

*Equal contribution 

Published as a conference paper at ICLR 2024 

arrives. These works aim to improve the serving performance of LLMs with long and shared prompts in a _single GPU_ setting. However, in-production LLM serving systems typically utilize a distributed set of GPUs to serve user requests. Current distributed LLM serving systems are not prompt-cache-aware; they attempt to distribute LLM computation load equally across GPUs to achieve high cluster-level GPU utilization. Yet, this distribution could result in requests with shared prefixes being sent to different GPUs, causing KV computation at all these GPUs that could otherwise be avoided if prefixes are cached and reused on the same GPU. On the other hand, a naive solution that always sends requests with shared prefixes to the same GPU would result in imbalanced loads and low overall GPU utilization because the GPU that initially serves a request with a popular prefix will accumulate a huge load of new requests all trying to reuse the calculated prefix KV. 

To properly design a distributed LLM serving system for long and shared prompts, we first perform a comprehensive study of five typical LLM workloads: LLM with tool calling Guo et al. (2024), LLM as embodied agents in virtual environments Huang et al. (2022), LLM for code generation Nijkamp et al. (2023), embedded video QA Xiao et al. (2021), and long document QA Li et al. (2023a). We find their prompts to be 37 _×_ to 2494 _×_ longer than generated sequences, and 85% to 97% tokens in a prompt are shared with other prompts. Additionally, a request often shares prompts with multiple other requests at different amounts ( _e.g._ , one prefix sharing being the initial system prompt, one sharing being the system prompt plus tool A’s demonstration, and one being the system prompt plus tool A and tool B’s demonstrations). We also analyze the Azure LLM request trace Patel et al. (2024) and found these end-user traces to have large prompt-to-output ratios as well. Additionally, this trace shows that requests arrive at varying speeds over time and across LLM usages, making designing a distributed LLM serving system challenging. 

Based on our findings, we propose a distributed LLM request scheduling algorithm called _E2_ (standing for _Exploitation + Exploration_ ) that co-designs model computation load-balancing and prefix-cache sharing. E2 allows requests to _exploit_ ( _i.e._ , reuse) computed prompt prefixes on the same GPU but also gives chances for requests with shared prefixes to _explore_ other GPUs. E2 chooses exploitation when the amount of recomputation saved is larger than that of new computation, which happens when the number of shared prefix tokens is larger than the remaining non-shared tokens. Otherwise, E2 chooses exploration. For exploitation, we send the request to the GPU that caches the longest-matched prefix. 

When E2 decides to explore GPUs, it chooses the GPU with the lightest “load”, using a _prompt-aware load_ definition we propose. This prompt-aware load includes three parts all calculated as GPU computation time. The first part is a GPU’s computation load in a recent time window _H_ , which is measured by the total prefill time and decode time incurred by all the requests in _H_ . The second part is the cost of evicting existing KVs on the GPU to make memory space to run the new request. The third part is the cost of running the new request on the GPU. When calculating these three parts, we account for prompt sharing that have or would occur on the GPU by treating the reusable computation as zero cost. E2 picks the GPU with the lowest sum of the three parts to explore, which balances loads while accounting for cached prompt behavior. 

Centered around the E2 scheduling algorithm, we build _Preble_ , a distributed LLM serving system that aims to provide high serving throughput and low request average and tail latency for long and shared prompts. Preble consists of a global, request-level scheduler and a per-GPU, iteration-level scheduler. Apart from E2, Preble incorporates several novel designs to tackle practical LLM challenges. First, with the basic E2 algorithm, a prefix is cached at a GPU after its initial assignment until its eviction. However, the amount of requests sharing it can change over time, which can cause load imbalance. To mitigate this issue, Preble detects load changes and redirects requests from a heavily loaded GPU to a light GPU. If the load hitting a popular prefix increases beyond what a single GPU can handle, Preble automatically scales (autoscales) the prefix by replicating it on multiple GPUs. 

Second, prefill and decoding phases have different computation needs, as discovered and tackled by a set of recent works Agrawal et al. (2023); Zhong et al. (2024); Patel et al. (2024); Agrawal et al. (2024). We 

2 

Published as a conference paper at ICLR 2024 

propose a new way to solve this problem based on our insight that a prompt hitting a cached prefix can be treated as decoding-phase computation, while a missed prompt can be treated as prefill-phase computation because of the high prompt-to-decoding token length ratio. Thus, our global scheduler tries to mix the two types of requests on a GPU to balance prefill and decoding computation needs. 

Finally, unlike existing works that either honor request fairness or maximize prefix matching, Preble aims to achieve high prefix reusing while ensuring fairness, which is important in multi-tenancy environments. We achieve this by assigning priorities to waiting requests based on their prefix cache hit ratio and giving each priority their respective quota of requests to serve. 

We implement Preble as a standalone layer on top of slightly modified vLLM Kwon et al. (2023) and SGLang Zheng et al. (2023b), two popular open-source LLM serving systems both supporting single-GPU prefix caching. We evaluate Preble using our studied five workloads and the Azure request arrival pattern with the Mistral 7B model Jiang et al. (2023) and the Llama-3 70B model Meta (2024) on a fourNvidia-A6000 GPU cluster and an eight-Nvidia-H100 GPU server. Our results show that Preble outperforms SGLang by 1.5 _×_ -14.5 _×_ and 2 _×_ -10 _×_ on average and p99 average request latency and similarly when compared to vLLM. 

Overall, this paper makes the following key contributions: **(1)** The first study of LLM workloads with long and shared prompts, resulting in four key insights. **(2)** E2, a new LLM request scheduling algorithm with the idea of exploitation and exploration integration. **(3)** Preble, the first distributed LLM serving system that targets long and shared prompts. **(4)** A comprehensive evaluation of Preble and SOTA LLM serving systems on two popular open-source LLMs, five real workloads, and two GPU clusters. Preble is publicly available at https://github _._ com/WukLab/preble. 

**==> picture [430 x 103] intentionally omitted <==**

**----- Start of picture text -----**<br>
System: You are AutoGPT, you can use many tools… tool use 1835:42; 85%; 39financial_statement..Tool 1.  User: .. Could you provide me basic daily data…? LLMcall  werans “You are a computer science programmer … program generation Here is an example: 3871:190; 97%; 126Programming Problem 1User  LLMcall call  1st parallel code 2nd parallel generation<br>You have access of the following tools: {‘type’: ‘object’, …{‘parameters’: …{‘parameters’: Tool 3. .. {‘parameters’: Tool 2. ..… User: …?User: …? problem, --Input--, --outpu—, example codeHere is another example: … User Problem 2 LLM code generation…<br>video QA 9865:4; 88%; 8.6 call LLM answer: 0.<br>virtual environment 2285:16; 97%; 48 “What happened to the baby…?  walk away<br>“You are in the middle of a Here are two examples…room with kitchen itemsYour task is to: … LLMcall  stove 1go to actiongen number of steps depends on LLM generation act in VE see a pan 2feedbackstove 1, on the env LLMcall  pan 2 from pick up stove 1 backfeedenv  … actionfinal  document QA Tokenized VideoLong Documen 23474:16; 91%; 18“Why did the woman bend down..? 0. walk away, 1. run after it, …”0. to jump.., 1. excercise, …” call LLM answer: 1. excercise<br>**----- End of picture text -----**<br>


Figure 1: **Prompt Sharing Features of Five Workloads.** _Green boxes represent shared prefixes. Grey boxes are non-shared prompts. White boxes are output generation. Red boxes contain statistics in average values: “promptlength:output-length; shared token percentage; number of requests sharing a sequence”._ 

## **2 Real-World Long and Shared Prompts** 

This section briefly presents our study results of five LLM use cases and an end-user LLM use trace: tool use Schick et al. (2023), embodied agent in a virtual environment Hao et al. (2023); Huang et al. (2022), software program generation Nijkamp et al. (2023), video QA Xiao et al. (2021), long document QA Li et al. (2023a), and the Azure LLM usage trace Patel et al. (2024) (which contains chat and code-generation usages). We pick the five workloads to study and evaluate, as they resemble long-and-shared-prompt use cases reported in production Anthropic (2024). Figure 8 demonstrates the prompt usages and overall features of these workloads. Appendix A presents our full study methodology and results. 

Overall, our study shows that prompts are significantly longer than output lengths in the five workloads and the Azure trace, ranging from 4 _×_ (Azure chat) to 2494 _×_ (video QA). Moreover, prompt sharing ranges from 85% ( _i.e._ , 85% prefix tokens in a prompt are shared with at least one more request) to 97% across the five 

3 

Published as a conference paper at ICLR 2024 

workloads, with embodied agents and program generation having the highest sharing amount. Additionally, a common sequence in a workload is shared by 8.6 to 126 requests on average, with different deviations across workloads. Finally, the Azure trace indicates high variations in total request loads manifested as different end-user request inter-arrival times that range from 2 microseconds to 217 seconds. 

Our study results indicate that LLM serving systems for such workloads should focus on optimizing prefill computation, and a viable way is to cache and share prefixes. However, the variations in prompt-sharing features and request arrival patterns make it challenging to build an efficient distributed serving system. 

## **3 Preble Design** 

We now present the E2 algorithm and the design of Preble, beginning with the overall system architecture of Preble, followed by its global scheduler and local scheduler designs. Preble significantly improves serving speed and reduces request latency for workloads with long and shared prompts. For workloads without any shared prompts, its behavior and performance are the same as SOTA LLM serving systems like vLLM Kwon et al. (2023). 

## **3.1 Overall System Architecture** 

Preble is a distributed GPU-based LLM serving system supporting both data parallelism and model parallelism. While its model parallelism support is standard ( _e.g._ , tensor parallelism), Preble’s scheduling of requests on data-parallel GPUs is designed specifically for long and shared prompts. 

**==> picture [216 x 137] intentionally omitted <==**

**----- Start of picture text -----**<br>
incoming request<br>Tokenizer global radix tree [GPU1:7req,GPU2:6req,<br>[GPU1:4req, GPU3:8req]<br>GPU2:5req] [GPU3:6req]<br>Global Scheduler:<br>E2 Scheduling [GPU1:1req] [GPU2:2req]<br>scheduled  [GPU2:1req]<br>requests req finish,<br>tree node evict<br>local tree<br>[7req] Local: Local  [6req] Local<br>Iteration,  Scheduler [5req] Scheduler<br>[4req] Priority  [2req] GPU4<br>[1req] Scheduling (model copy 3)<br>GPU GPU [1req]<br>wait req  GPU1 2 3 [8req]<br>priority  (model copy 1) (model parallelism<br>queues model copy 3) [6req]<br>**----- End of picture text -----**<br>


Figure 2: **Preble Architecture.** 

We propose a two-level scheduling system where a global scheduler performs _request-level_ scheduling decisions and orchestrates the overall load balancing across GPUs, while a per-model-instance local scheduler performs _iteration-level_ scheduling for requests assigned to the GPU, as shown in Figure 2. Depending on the GPU cluster topology, the global scheduler may be deployed on a separate server for a multi-server GPU cluster or on the same server as a single-server-multi-GPU cluster. The local scheduler manages one model instance (multiple GPUs when using model parallelism, single GPU when not) and runs on the CPU of the same server as the GPUs. Our current implementation of Preble scales 

to at least 70 to 391 GPUs. To offer larger, data-center-level scales, one can deploy several Preble clusters, each having one global scheduler. 

This design offers several benefits: 1) by having all requests in a cluster go through the global scheduler, we have a centralized place to maintain a global view of cluster load and prompt caching information, both being essential for E2; 2) by performing coarse-grained, request-level scheduling, a single global scheduler can scale to hundreds of GPUs, avoiding the complexity of maintaining multiple distributed global schedulers for a cluster; and 3) by performing fine-grained, iteration-level scheduling at each GPU, the local scheduler can quickly adapt to GPU resource and request availability changes and make precise decisions. 

## **3.2 E2 Global Scheduler** 

We now present our global scheduler design, which centers around the E2 distributed scheduling algorithm. 

4 

Published as a conference paper at ICLR 2024 

**Global scheduler data structures.** The global scheduler maintains several data structures to assist its prompt-aware request scheduling. The primary data structure is _global prefix trees_ , implemented as radix trees Wikipedia (2024). Each tree has a distinct root storing the shared prefix of all prompts under the tree. When inserting a new request to the tree, we match its tokens from the beginning ( _i.e._ , prefix matching) until no match exists, and we insert the remaining tokens as a new leaf node. If no match exists at all, we create a new tree with this request’s prompt as the root node. If an existing tree node only matches partially to the new request ( _i.e._ , the prefix of a node matches a sub-sequence of the new request), we split the node into the matched part and the remaining part. For each tree node, we record three pieces of information: the number of tokens in the tree node, the set of GPUs caching the tree node KVs, and the per-GPU number of requests sharing the tree node in a history window _H_ . When a tree node has no caching GPU and there is no request within the window _H_ in the whole system sharing it, we remove it from the tree. 

**Per-request scheduling policy.** To schedule a request, the global scheduler uses our proposed E2 scheduling algorithm, as illustrated in Algorithm 1. It first matches the request’s prompt in the global prefix trees. When the amount of recomputation saved (number of tokens in the matched prefix) is larger than the amount of new computation (number of tokens in the remaining prompt), we favor exploitation over exploration because the gain of saved GPU resources (computation for matched tokens) is higher than the load (computation for unique tokens) that can potentially be balanced in the GPU cluster. For such requests, E2 _exploits_ existing cache by assigning the request to the GPU that caches the tree node with the longest tokens in the matched prefix. If multiple such GPUs exist, E2 chooses the GPU with the lightest request load using the load calculation to be introduced next. 

If the matched prefix is shorter than the remaining tokens, E2 _explores_ the best GPU to run the request based on our proposed new prompt-aware “load cost” definition. Exploration gives E2 a chance to distribute load to different GPUs, which is the key to striking long-term cluster execution efficiency. E2 unifies three types of costs when calculating the per-GPU load: the computation cost aggregated across all requests within a time window, the recomputation cost needed for evicting memory to run the new request, and the computation cost of the new request. E2 calculates all three costs as GPU computation time and finds the GPU with the lowest sum. Instead of profiling the actual computation time, we only maintain token counts at the global scheduler, which largely reduces the system overhead. We leverage transformer-based LLMs’ properties that the computation amount of prefill and decoding are proportional to the number of prompt tokens and generated tokens (Figures 9 and 10). Below and in Algorithm 2, we detail the three calculations for scheduling a new request _Rk_ . 

The first cost is the _overall_ GPU computation load _Li_ for _GPUi_ . We capture a recent load history on _GPUi_ with a time window _H_ with a default value of 3 minutes (we test different _H_ lengths and find the results insensitive to it). We do not use _GPUi_ ’s load at the exact request scheduling time for two reasons: 1) a GPU’s load can change between the time of scheduling _Rk_ to the time of running it, and 2) the placement of a prefix has a longer-term effect than a single load in time because of other requests’ future exploitation of it. For each request _Rr_ in the history, we estimate its prefill time _PTr_ with a regression function using the number of tokens in _Rr_ that do not match any prefixes on _GPUi_ ; we estimate its decoding time _DTr_ with another regression function using the average request output length observed on _GPUi_ in window _H_ . We have the first category of load, _Li_ =[�] _r∈W_[(] _[PT][r]_[+] _[ DT][r]_[)][,][where] _[W]_[is][the][set][of][requests][in][the][window] _H_ . The regression functions used in this calculation are captured from offline profiling for each GPU type. Note that even though the number of output tokens is not known a priori, our workload study (Appendix A) shows that it is small and similar across a type of workload. Thus, we use the average output length in _H_ as the estimated decoding length for _DTr_ . 

The second cost is the potential cost to free GPU memory so that the new request, _Rk_ , can run. Given that GPUs run at full capacity with our and existing serving policies Agrawal et al. (2023), we expect this cost to always occur. Intuitively, the more tokens in a sequence that are shared and by more requests, the more costly it is to evict the sequence. Thus, to calculate the eviction cost, _Mi_ , the global scheduler first uses the 

5 

Published as a conference paper at ICLR 2024 

eviction algorithm to be discussed in Section 3.3 to find the tree nodes on _GPUi_ that would be evicted to run _Rk_ . For each such tree node _j_ , its eviction cost is the recomputation time of the evicted tokens multiplied by the hit rate of the node, _Nj_ . Thus, we have _Mi_ =[�] _j∈E[PT][j][×][ N][j]_[where] _[ E]_[is the eviction node set,] _[ PT][j]_ is the prefill time for the length of tree node _j_ , and _Nj_ is the the number of requests sharing tree node _j_ in history _H_ over the total number of requests on _GPUi_ in _H_ . Note that we do not include the decoding time, as a request’s decoding time is unaffected by prefix cache eviction, and decoding costs have already been counted in _Li_ . 

The third cost is the actual cost, _Pi_ , to run the new request _Rk_ on _GPUi_ , which is simply the prefill time of the missed tokens in request _Rk_ . We do not count its decoding time, as it is the same across GPUs, and our goal is to compare the per-GPU load across GPUs. 

The total cost of assigning the current request to _GPUi_ is _Li_ + _Mi_ + _Pi_ and we choose the GPU with the lowest total cost to assign the request to. 

**Post-assignment load adjustment.** With the above algorithm, after the global scheduler assigns a request to a GPU, its prefix lives there until its eviction. This greedy approach works well in cases where the load to a prefix is relatively stable but not otherwise. We propose two ways of managing post-assignment load changes. The first way shifts load between GPUs and is applicable when the load surge can be handled by a single GPU in the cluster. The global scheduler maintains a per-GPU load as discussed above. If the most heavily loaded GPU’s load is more than _Thbal_ times higher than the lightest GPU, it shifts load from the former to the latter until their difference is below _Thbal_ . _Thbal_ is configurable and can be deducted from profiling GPU and LLM types. To perform this load rebalancing, we direct future requests that are supposed to exploit the heavy GPU to the light GPU. 

The second way is to auto-scale a prefix by replicating it and splitting its subtree by load when we detect that a certain prefix’s request load is still too high (average queueing time doubles over _H_ ) even after the above load rebalancing. We calculate the subtree’s load using Algorithm 2. 

**Prefill-decoding balancing.** From our study results in Appendix A and reported by others, LLM prefill has a larger compute-to-memory ratio than decoding, causing inefficient GPU resource utilization and performance degradation. While various recent works tackle the prefill-decoding imbalance problem by chunking prefills Agrawal et al. (2023) and prefill-decoding disaggregation Zhong et al. (2024); Patel et al. (2024); Strati et al. (2024); Hu et al. (2024); Qin et al. (2024a), we propose a new way of solving the problem leveraging prompt sharing features at a cluster level. Our insight is that a request with its entire prompt shared and cached would only perform the decoding phase. Thus, it can be treated as a decoding-phase computing unit. Meanwhile, a request with a long prompt not cached and a short output length can be treated as a prefill-phase computing unit. A partially cached prompt can be treated as being between the prefilland decoding-phase units. Thus, we can balance prefill-decoding by combining requests with more or less prompt sharing instead of or in addition to existing balancing techniques. 

Specifically, when a request is about to be explored, the global scheduler first considers the prefill and decoding balancing for each GPU. If a GPU is heavily loaded with decoding-phase computing units, the global scheduler directs the current request to it, as a request to be explored will incur recomputation for prompt and is considered a prefill-phase unit. We prioritize this policy over the load-cost comparison (Algorithm 2) because a GPU with heavy decoding has unused computation capacity that we can almost freely use. The global scheduler performs the load-cost comparison if all GPUs have relatively balanced decoding-prefill loads. Apart from this prefill-decoding balancing performed at the global scheduler, our local scheduler also performs traditional chunked prefill for each GPU (Section 3.3). 

6 

Published as a conference paper at ICLR 2024 

## **3.3 Local Scheduler** 

**Local scheduler mechanism.** The local scheduler schedules the requests that the global scheduler assigns to its managed GPU(s). Similar to existing LLM serving systems Yu et al. (2022); Kwon et al. (2023); Zheng et al. (2023b); Aminabadi et al. (2022); Miao et al. (2024); Vaidya et al. (2023), we run one local scheduler per GPU and schedule requests at the iteration level. Each local scheduler maintains a request wait queue, a prefix (radix) tree, and the number of active requests sharing each prefix tree node. 

When a new request arrives, the local scheduler matches it to the local prefix tree and updates the tree accordingly. It also inserts the request into the waiting queue. After each model iteration, the local scheduler forms the next batch by selecting waiting requests using a priority-based algorithm to be discussed next. If a selected request has a long and non-shared prompt, we chunk the prompt similar to Sarathi Agrawal et al. (2023). If the GPU memory is not enough to run the batch, the local scheduler selects a tree node(s) or part of a tree node (if a part is enough) to evict based on the request accessing time (LRU) of tree nodes. The local scheduler then asynchronously informs the global scheduler about the eviction, and the latter processes it in the background. 

**Waiting queue request ordering.** Today’s LLM serving systems schedule requests in the waiting queue according to FCFS Kwon et al. (2023) or prefix sharing Zheng et al. (2023b) (serve the request with the highest sharing amount the first). The former ignores prompt sharing and results in more recomputation; the latter ignores fairness and could result in starvation Wu et al. (2023). We propose a priority-based wait queue scheduling policy that considers both prefix sharing and fairness. Specifically, we create _P_ (a configurable parameter) priority groups and assign a request to the priority group according to its cached token percentage. For example, if 63 out of 100 tokens in a request’s prompt are cached on the GPU and _P_ is 10, it will be assigned priority six. When selecting requests to form the next batch, the scheduler proportionally selects requests from each priority group, with the higher priority group getting more requests selected than lower priority ones. For example, if 55 requests are to be selected to form a batch, the scheduler picks ten from priority group 10, nine from priority 9, etc. 

## **4 Implementation and Evaluation Results** 

## **4.1 Implementation** 

We implemented Preble as a standalone layer to perform distributed LLM serving. As such, Preble can be added to any existing serving systems with no or minimal changes — we currently support vLLM Kwon et al. (2023) and SGLang Zheng et al. (2023b) as two backends. 

**Global scheduler scalability.** In implementing the global scheduler, we use a few techniques to improve its scalability. Incoming requests are first tokenized by a parallel tokenization layer. Afterward, the global scheduler spawns asynchronous request handlers to schedule requests. Access to the global radix tree during request handling is lock-free, as most operations are read-only. The only exceptions are updating a GPU to be assigned to a tree node and the increment of request count hitting the tree node, both of which can be expressed as atomic instructions. Additionally, the global scheduler maintains a current load count for each GPU by updating it every time a new request is assigned to it or when it evicts a tree node. Thus, our realization of the E2 algorithm is performance-efficient. Finally, to ensure foreground request performance, the global scheduler runs non-request-scheduling tasks such as rebalancing and eviction bookkeeping in the background with separate threads. 

## **4.2 Workloads and Environments** 

**Workloads.** We evaluate our results on five LLM use cases: LLM generation with tool demonstration and calling Schick et al. (2023), LLM interacting with virtual environments as an embodied agent Hao et al. 

7 

Published as a conference paper at ICLR 2024 

(2023); Huang et al. (2022), LLM for software program generation Nijkamp et al. (2023), answering questions about videos Xiao et al. (2021), and answer questions about long documents Li et al. (2023a). Their properties are presented in Appendix A. 

For each workload, we sample enough requests to fulfill the request-per-second (RPS) needs and GPU setup ( _e.g._ , a larger GPU or more GPUs can handle more). For experiments other than the ones using the Azure Inference Trace, we set the inter-arrival time using a Poisson distribution with a mean that corresponds to the RPS we test (X-axis in most figures). We then run the experiments until stable state is reached and lasts for a significant length. 

**LLMs and environments.** We test Preble and baselines using two popular open-source Large Language Models, the Mistral 7B model Jiang et al. (2023) and the Llama-3 70B model Meta (2024). We run our experiments in one of the two local testbed environments: a two-server cluster each containing two NVidia A600 GPUs and one eight NVidia-H100-GPU server. 

**Baseline.** Our baselines are serving systems that support single-GPU prefix sharing, including SGLang Zheng et al. (2023b) and vLLM (which recently added a beta feature for prefix sharing Moore & Li (2024)). To run SGLang and vLLM in a distributed fashion, we set up a load balancer that sends requests in a round-robin fashion to individual SGLang/vLLM instances ( _i.e._ , non-prompt-aware data parallelism). As round- robin distributes requests evenly, these baselines capture a distributed serving system that balances request loads and then performs prefix sharing within each parallel instance. 

**Metrics.** We use three key metrics: request per second, which measures serving capacity; average end-toend request latency (including scheduling time, queueing time, prefill, and decoding time); and p99 request latency. Note that our metrics differ slightly from some existing LLM serving works Kwon et al. (2023); Yu et al. (2022), as we do not use TPOT (time per output token) or TTFT (time to first token) as key metrics. This is because our target LLM use has short output lengths, rendering TPOT not as meaningful and TTFT close to the request latency. We consider p99 latency since it is important to control the tail latency in LLM serving as with all other user-facing services DeCandia et al. (2007); Ongaro et al. (2011). 

## **4.3 End-to-End Workload Performance** 

We first present the overall performance of Preble and the baselines. Below, we focus on the comparison with SGLang as it is specifically designed for (single-GPU) prefix sharing while being up-to-date on major LLM serving techniques. We provide Preble’s comparison to vLLM and to different SGLang versions in the Appendix C. 

**Single workload results.** We now present the average and p99 latency against increasing requests arriving per second (RPS) of Preble and SGLang on the five workloads, two LLMs, and two GPU environments, as shown in Figure 3. Overall, Preble significantly outperforms the data-parallel SGLang baseline for all settings, as can be seen from Preble’s lower average and p99 latency, especially under higher RPS (or the other way around, for the same latency target, Preble can serve higher RPS). Our improvements over SGLang range from 1.5 _×_ to 14.5 _×_ in terms of average latency and 2 _×_ to 10 _×_ in p99 latency. 

Comparing across workloads, we see bigger improvements of Preble over SGLang on the Toolbench, embodied agent, video QA, and LooGLE workloads than the programming workloads. The programming workload has the longest decoding length among all the workloads. As decoding time starts to dominate total request latency, and we do not improve decoding performance, the room for improvement for Preble is smaller. Nonetheless, Preble still achieves 1.56 _×_ to 1.8 _×_ improvement in average latency and 3 _×_ to 4 _×_ in p99 latency over SGLang in the programming workload. 

Comparing across the number of GPUs, Preble’s relative improvement over the baselines stays similar when going from two to four A6000 GPUs. Considering absolute values, we see Preble successfully maintain similar latency even as RPS doubles, showing its strong scalability. When changing from A6000 to eight 

8 

Published as a conference paper at ICLR 2024 

**==> picture [430 x 387] intentionally omitted <==**

**----- Start of picture text -----**<br>
Preble SGLang<br>Tool Use Embodied Agent Programming Video QA LooGLE<br>15 200 60<br>80<br>100 150<br>10 40 60<br>100<br>50 40<br>5 50 20 20<br>0 0<br>5 10 15 5 10 15 1 2 3 4 2 4 6 0.25 0.50 0.75 1.00<br>80 800<br>300<br>60 600 100 400<br>200<br>40 400<br>50 200<br>100 20 200<br>0 0 0 0 0<br>5 10 15 5 10 15 1 2 3 4 2 4 6 0.25 0.50 0.75 1.00<br>20<br>150 200 100<br>60<br>100 15 150 75<br>10 100 40 50<br>50 5 50 20 25<br>0 0<br>10 20 30 10 20 30 2 4 6 8 2.5 5.0 7.5 10.0 0.5 1.0 1.5 2.0<br>400 800 150 600<br>300 40 600<br>100 400<br>200 400<br>100 20 200 50 200<br>0 0 0 0 0<br>10 20 30 10 20 30 2 4 6 8 2.5 5.0 7.5 10.0 0.5 1.0 1.5 2.0<br>10.0 15 60 60 8<br>7.5 10 40 40 6<br>5.0<br>5 20 4<br>20<br>2.5<br>0<br>5 10 15 10 15 3 4 5 2 4 6 0.4 0.6 0.8<br>300 30<br>30 30 300<br>200<br>20 20 200 20<br>100<br>10 10 100 10<br>0<br>5 10 15 10 15 3 4 5 2 4 6 0.4 0.6 0.8<br>Rate (req/s) Rate (req/s) Rate (req/s) Rate (req/s) Rate (req/s)<br>Mistral 7B<br>2xA6000 Avg (s)<br>Mistral 7B<br>2xA6000 p99 (s)<br>Mistral 7B<br>4xA6000 Avg (s)<br>Mistral 7B<br>4xA6000 p99 (s)<br>Llama3 70B<br>8xH100 Avg (s)<br>Llama3 70B<br>8xH100 p99 (s)<br>**----- End of picture text -----**<br>


Figure 3: **End-to-end Workload Performance** _The top and middle two rows run on two and four A6000 GPUs with the Mistral 7B model. The bottom two rows run on eight H100 GPUs set up as 4-GPU tensor parallelism plus data parallelism with the Llama-3 70B model._ 

H100 and switching the Mistral 7B model to the Llama-3 70B model, we find relative improvements of Preble to increase. 

**Azure trace and mixed workloads.** Our experiments above use a Poisson request arrival distribution (which is the same as most existing LLM works’ experimental methodology Kwon et al. (2023); Li et al. (2023b)). To understand Preble’s performance under real-world request load, we run the tool use and video QA workloads using Azure’s LLM request arrival pattern (Appendix A.6) instead of Poisson distributions. Here, we mix the two workloads to mimic Azure’s mixed chat and code traces. As shown in Figure 4, Preble has significant improvements in average and p99 latencies and on average TTFT and TPOT. 

9 

Published as a conference paper at ICLR 2024 

**==> picture [430 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
400<br>p99<br>12 70 0.30 1.0 350300 avg<br>10 60 0.25 0.8 250<br>8 50 0.20 200<br>40 0.6 150<br>6 0.15 2.42x<br>30 0.4 100<br>42 6.02x 2010 4.61x 0.100.05 5.60x 0.2 500<br>0 0 0.00 0.0<br>SGLang Preble SGLang Preble SGLang Preble SGLang Preble<br>(baseline)SGLang E2 no re-adjustment+re-adjustment +Prefill/Decode +Priority-WQ(=ALL)<br>Latency (s)<br>Avg TTFT (s)<br>Avg Latency (s) p99 Latency (s) Avg TPOT (s/tok)<br>**----- End of picture text -----**<br>


Figure 4: **Mixed Workload With Azure Trace** _Running Tool and Video mixed workloads with Azure trace arrival patterns on 4 A6000 GPUs._ 

Figure 5: **Ablation Results** _Running ToolBench with Zipf-1.1 skew to different prompts running on four A6000 GPUs_ 

## **4.4 Deep Dive** 

We now provide a detailed analysis of Preble, including an ablation study and global scheduler scalability test. Because of H100 GPUs’ high cost and low availability, we run all experiments in this section with A6000 GPUs. 

**Ablation study.** To understand where the benefits of Preble come from, we evaluate Preble by incrementally adding features presented in Section 3. We chose the tool use workload with a Zipf-1.1 popularity distribution among the prompts in the dataset to represent real-life skewed tool popularity. Other workloads and distributions benefit from a different set of techniques. We start with using the SGLang round-robin baseline. We first add the per-request E2 policy (Section 3.2), which results in an improvement on both average and p99 request latency because of E2’s dynamic load partitioning. We then add the post-assignment global rebalancing and autoscaling, which successfully balances out load even more, resulting in further improvement, especially with p99. Further adding the prefill/decode-aware handling results in more improvement on both average and p99, since it considers the current batch composition and is able to better utilize the GPU resources. Finally, we add the local-scheduler priority-based wait-queue scheduling (§3.3), which, as expected, improves p99 but not average latency, as its goal is fairness. 

**Global scheduler performance and scalability.** We measure the maximum throughput of Preble’s global scheduler by sending a large number of requests ( _e.g._ , 50,000) at once to eliminate the effect of request arrival patterns and saturate the scheduler. Since the global prefix tree search is the most time-consuming task at the global scheduler, we test the Toolbench and VideoQA workloads, which have the most complex and simplest prefix tree structures in our five workloads. Preble’s global scheduler achieves a processing rate of 245 and 2931 requests per second for Toolbench and VideoQA. We also measure the network processing speed and find it not to be the bottleneck. With the peak GPU processing rate (30-150 tokens per second decoding speed with Mistral 7B on A100) and our workloads’ output length (Table 1), one Preble global scheduler can sustain at least 70 to 391 concurrent A100 GPUs. If accounting for prefill time or running bigger models, our scheduler would sustain even more GPUs. 

## **5 Related Works** 

LLMs’ usages are shifting to be more prompt-heavy. As a result, the problem of prefill and decoding having different compute-to-memory ratios is exacerbated. Several recent works have targeted LLM usages with long prompts and proposed solutions to solve the imbalance problem Agrawal et al. (2023; 2024); Zhong et al. (2024); Patel et al. (2024). The first approach, called _chunked prefill_ , chunks a prompt and runs each chunk with other decoding requests in a batch in a single iteration to reduce or avoid waiting Agrawal et al. (2023; 2024). The second approach is to separate prefill and decoding to different GPUs to avoid prefill- 

10 

Published as a conference paper at ICLR 2024 

decoding interference Patel et al. (2024); Zhong et al. (2024). These solutions target long prompts but do not consider prompt sharing. Preble consider prompt length and sharing, and we use a novel sharing-based approach on top of chunked prefill to solve the prefill-decoding imbalance problem. 

A recent work, SGLang Zheng et al. (2023b), proposes to share prefixes across requests using a prefix tree. More recently, vLLM also added the support for prompt prefix sharing Moore & Li (2024). Unlike Preble, SGLang and vLLM are both single-GPU solutions. To run them on a distributed GPU cluster, one would need to add a standard data or model parallelism layer and then run SGLang or vLLM on each GPU. As no parallelism or distributed serving systems today are prompt-aware, simply distributing requests or models and then performing prefix-sharing within a GPU ignores the cluster-level prefix-sharing opportunity. Apart from distributed support for long and shared prompts (Section 3.2), Preble also improves prefix-caching with fairness over SGLang and vLLM with a better waiting-request ordering policy (Section 3.3). Another recent work, Prompt Cache Gim et al. (2024), proposes sharing arbitrary user-defined sub-sequences in a prompt by allowing mismatched positional encodings and incomplete attention computation. As such, non-prefix sharing is a lossy process that could result in lower-quality generation. Moreover, like SGLang, Prompt Cache is also a single-GPU solution and shares SGLang’s limitations discussed above. Hydragen Juravsky et al. (2024) is another recent work that proposes an efficient implementation of the attention operation for shared prefixes, which is orthogonal to Preble, as Preble can support any underlying attention kernels. 

## **6 Conclusion** 

This paper identified the problem of distributed serving for long and sharing prompts. To solve this problem, we performed a study on five LLM workloads and one real LLM trace. We presented E2, a distributed LLM request scheduling algorithm targeting LLM usages with long and shared prompts. We built Preble, a distributed LLM serving system on top of the E2 algorithm and a hierarchical scheduling architecture. Our results show that Preble significantly improves LLM serving performance over SOTA serving systems while ensuring request fairness and controlling the tail latency. 

## **References** 

- Amey Agrawal, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, and Ramachandran Ramjee. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. _arXiv preprint arXiv:2308.16369_ , August 2023. 

- Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav S Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming throughput-latency tradeoff in llm inference with sarathi-serve. _Proceedings of 18th USENIX Symposium on Operating Systems Design and Implementation, 2024, Santa Clara_ , 2024. 

- Reza Yazdani Aminabadi, Samyam Rajbhandari, Ammar Ahmad Awan, Cheng Li, Du Li, Elton Zheng, Olatunji Ruwase, Shaden Smith, Minjia Zhang, Jeff Rasley, et al. Deepspeed-inference: enabling efficient inference of transformer models at unprecedented scale. In _SC22: International Conference for High Performance Computing, Networking, Storage and Analysis_ , Dallas, Texas, November 2022. IEEE. 

- Anthropic. Prompt caching with claude. https://www _._ anthropic _._ com/news/prompt-caching, 2024. 

- Giuseppe DeCandia, Deniz Hastorun, Madan Jampani, Gunavardhan Kakulapati, Avinash Lakshman, Alex Pilchin, Swaminathan Sivasubramanian, Peter Vosshall, and Werner Vogels. Dynamo: Amazon’s highly available key-value store. _ACM SIGOPS operating systems review_ , 41(6):205–220, oct 2007. 

- Patrick Esser, Robin Rombach, and Bj¨orn Ommer. Taming transformers for high-resolution image synthesis. In _2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , Los Alamitos, CA, June 2020. 

- In Gim, Guojun Chen, Seung seob Lee, Nikhil Sarda, Anurag Khandelwal, and Lin Zhong. Prompt cache: Modular attention reuse for low-latency inference. In _Proceedings of the 7th MLSys Conference_ , Santa Clara, CA, May 2024. 

11 

Published as a conference paper at ICLR 2024 

- Zhicheng Guo, Sijie Cheng, Hao Wang, Shihao Liang, Yujia Qin, Peng Li, Zhiyuan Liu, Maosong Sun, and Yang Liu. Stabletoolbench: Towards stable large-scale benchmarking on tool learning of large language models. _arXiv preprint arXiv:2403.07714_ , Mar 2024. 

- Shibo Hao, Tianyang Liu, Zhen Wang, and Zhiting Hu. Toolkengpt: Augmenting frozen language models with massive tools via tool embeddings. In _Advances in Neural Information Processing Systems 36_ , New Orleans, Louisiana, December 2023. URL https://proceedings _._ neurips _._ cc/paper ~~f~~ iles/paper/2023/file/8fd1a81c882cd45f64958da6284f4a3fPaper-Conference _._ pdf. 

- Dan Hendrycks, Steven Basart, Saurav Kadavath, Mantas Mazeika, Akul Arora, Ethan Guo, Collin Burns, Samir Puranik, Horace He, Dawn Song, and Jacob Steinhardt. Measuring coding challenge competence with apps. In _Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks_ , December 2021. 

- Cunchen Hu, Heyang Huang, Junhao Hu, Jiang Xu, Xusheng Chen, Tao Xie, Chenxi Wang, Sa Wang, Yungang Bao, Ninghui Sun, and Yizhou Shan. MemServe: Context Caching for Disaggregated LLM Serving with Elastic Memory Pool. _arXiv preprint arXiv:2406.17565_ , June 2024. URL https://arxiv _._ org/abs/2406 _._ 17565. 

- Wenlong Huang, Pieter Abbeel, Deepak Pathak, and Igor Mordatch. Language models as zero-shot planners: Extracting actionable knowledge for embodied agents. In Kamalika Chaudhuri, Stefanie Jegelka, Le Song, Csaba Szepesvari, Gang Niu, and Sivan Sabato (eds.), _Proceedings of the 39th International Conference on Machine Learning, Honolulu_ , volume 162 of _Proceedings of Machine Learning Research_ , pp. 9118–9147. PMLR, 17–23 Jul 2022. URL https://proceedings _._ mlr _._ press/v162/huang22a _._ html. 

- Sam Ade Jacobs, Masahiro Tanaka, Chengming Zhang, Minjia Zhang, Shuaiwen Leon Song, Samyam Rajbhandari, and Yuxiong He. Deepspeed ulysses: System optimizations for enabling training of extreme long sequence transformer models. _arXiv preprint arXiv:2309.14509_ , October 2023. 

- Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, L´elio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timoth´ee Lacroix, and William El Sayed. Mistral 7b. _arXiv preprint arXiv:2310.06825_ , October 2023. 

- Jordan Juravsky, Bradley Brown, Ryan Ehrlich, Daniel Y. Fu, Christopher R´e, and Azalia Mirhoseini. Hydragen: Highthroughput llm inference with shared prefixes. _arXiv preprint arXiv:2402.05099_ , May 2024. 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles_ , Koblenz, Germany, October 2023. 

- Jiaqi Li, Mengmeng Wang, Zilong Zheng, and Muhan Zhang. Loogle: Can long-context language models understand long contexts? _arXiv preprint arXiv:2311.04939_ , November 2023a. 

- Junkai Li, Siyu Wang, Meng Zhang, Weitao Li, Yunghwei Lai, Xinhui Kang, Weizhi Ma, and Yang Liu. Agent hospital: A simulacrum of hospital with evolvable medical agents. _arXiv preprint arXiv:2405.02957_ , May 2024a. 

- Junyou Li, Qin Zhang, Yangbin Yu, Qiang Fu, and Deheng Ye. More agents is all you need. _arXiv preprint arXiv:2402.05120_ , February 2024b. 

- Zhuohan Li, Lianmin Zheng, Yinmin Zhong, Vincent Liu, Ying Sheng, Xin Jin, Yanping Huang, Zhifeng Chen, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. AlpaServe: Statistical multiplexing with model parallelism for deep learning serving. In _17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23)_ , Boston, MA, July 2023b. 

- Hao Liu, Matei Zaharia, and Pieter Abbeel. Ringattention with blockwise transformers for near-infinite context. In _Proceedings of the 12th International Conference on Learning Representations, Vienna_ , 2024. URL https: //openreview _._ net/forum?id=WsRHpHH4s0. 

Meta. Meta llama 3. https://llama _._ meta _._ com/llama3/, 2024. 

12 

Published as a conference paper at ICLR 2024 

- Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems_ , April 2024. 

- Sage Moore and Zhouhan Li. Automatic prefix caching. https://github _._ com/vllm-project/vllm/issues/2614, March 2024. 

- Tsendsuren Munkhdalai, Manaal Faruqui, and Siddharth Gopal. Leave no context behind: Efficient infinite context transformers with infini-attention. _arXiv preprint arXiv:2404.07143_ , April 2024. 

- Erik Nijkamp, Hiroaki Hayashi, Caiming Xiong, Silvio Savarese, and Yingbo Zhou. Codegen2: Lessons for training llms on programming and natural languages. _arXiv preprint arXiv:2305.02309_ , July 2023. 

- Diego Ongaro, Stephen M. Rumble, Ryan Stutsman, John Ousterhout, and Mendel Rosenblum. Fast Crash Recovery in RAMCloud. In _Proceedings of the 23rd ACM Symposium on Operating Systems Principles (SOSP ’11)_ , Cascais, Portugal, October 2011. 

- OpenAI. Learning to reason with llms. https://openai _._ com/index/learning-to-reason-with-llms/, September 2024a. 

- OpenAI. Video generation models as world simulators. https://openai _._ com/index/video-generation-models-as-worldsimulators, 2024b. 

- Pratyush Patel, Esha Choukse, Chaojie Zhang, Aashaka Shah,[´] I˜nigo Goiri, Saeed Maleki, and Ricardo Bianchini. Splitwise: Efficient generative llm inference using phase splitting. In _2024 ACM/IEEE 51st Annual International Symposium on Computer Architecture (ISCA)_ , pp. 118–132. IEEE, 2024. 

- Ruoyu Qin, Zheming Li, Weiran He, Mingxing Zhang, Yongwei Wu, Weimin Zheng, and Xinran Xu. Mooncake: A kvcache-centric disaggregated architecture for llm serving. _arXiv preprint arXiv:2407.00079_ , Jul 2024a. URL https://arxiv _._ org/abs/2407 _._ 00079. 

- Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, dahai li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In _The Twelfth International Conference on Learning Representations, Vienna_ , 2024b. URL https://openreview _._ net/forum?id=dHng2O0Jjr. 

- Timo Schick, Jane Dwivedi-Yu, Roberto Dessi, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In _Thirtyseventh Conference on Neural Information Processing Systems, New Orleans_ , 2023. URL https://openreview _._ net/ forum?id=Yacmpz84TH. 

- Mohit Shridhar, Xingdi Yuan, Marc-Alexandre Cote, Yonatan Bisk, Adam Trischler, and Matthew Hausknecht. _{_ ALFW _}_ orld: Aligning text and embodied environments for interactive learning. In _International Conference on Learning Representations, Virtual_ , 2021. URL https://openreview _._ net/forum?id=0IOX0YcCdTn. 

- Foteini Strati, Sara Mcallister, Amar Phanishayee, Jakub Tarnawski, and Ana Klimovic. D´ej`avu: KV-cache streaming for fast, fault-tolerant generative LLM serving. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp (eds.), _Proceedings of the 41st International Conference on Machine Learning_ , volume 235 of _Proceedings of Machine Learning Research_ , pp. 46745–46771, Vancouver, Canada, 21–27 Jul 2024. PMLR. URL https://proceedings _._ mlr _._ press/v235/strati24a _._ html. 

FlashInfer Team. Flashinfer, October 2024a. URL https://github _._ com/flashinfer-ai/flashinfer. 

SGLang Team. SGLang, October 2024b. URL https://github _._ com/sgl-project/sglang. 

- Neal Vaidya, Nick Comly, Joe DeLaere, Ankit Patel, and Fred Oh. Nvidia tensorrt-llm supercharges large language model inference on nvidia h100 gpus. https://developer _._ nvidia _._ com/blog/nvidia-tensorrt-llm-supercharges-largelanguage-model-inference-on-nvidia-h100-gpus/, 2023. 

13 

Published as a conference paper at ICLR 2024 

- Xingyao Wang, Zihan Wang, Jiateng Liu, Yangyi Chen, Lifan Yuan, Hao Peng, and Heng Ji. Mint: Evaluating llms in multi-turn interaction with tools and language feedback. In _The Twelfth International Conference on Learning Representations_ , Vienna, Austria, May 2024. 

- Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In _The Eleventh International Conference on Learning Representations_ , 2023. URL https://openreview _._ net/forum?id=1PL1NIMMrw. 

- Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In _Proceedings of the 36th International Conference on Neural Information Processing Systems_ , New Orleans, LA, November 2024. 

Wikipedia. Radix tree. https://en _._ wikipedia _._ org/wiki/Radix ~~t~~ ree, 2024. [Online; accessed 1-October-2024]. 

- Bingyang Wu, Yinmin Zhong, Zili Zhang, Gang Huang, Xuanzhe Liu, and Xin Jin. Fast distributed inference serving for large language models. _arXiv preprint arXiv:2305.05920_ , May 2023. 

- Qingyun Wu, Gagan Bansal, Jieyu Zhang, Yiran Wu, Beibin Li, Erkang Zhu, Li Jiang, Xiaoyun Zhang, Shaokun Zhang, Jiale Liu, Ahmed Hassan Awadallah, Ryen W White, Doug Burger, and Chi Wang. Autogen: Enabling next-gen LLM applications via multi-agent conversation. In _ICLR 2024 Workshop on Large Language Model (LLM) Agents, Vienna_ , 2024. URL https://openreview _._ net/forum?id=uAjxFFing2. 

- Junbin Xiao, Xindi Shang, Angela Yao, and Tat-Seng Chua. Next-qa: Next phase of question-answering to explaining temporal actions. In _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)_ , pp. 9777–9786, June 2021. 

- Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. Tree of thoughts: Deliberate problem solving with large language models. In A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine (eds.), _Thirty-seventh Conference on Neural Information Processing Systems_ , volume 36, New Orleans, Louisiana, December 2023a. URL https://proceedings _._ neurips _._ cc/paper ~~f~~ iles/paper/2023/ file/271db9922b8d1f4dd7aaef84ed5ac703-Paper-Conference _._ pdf. 

- Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R Narasimhan, and Yuan Cao. React: Synergizing reasoning and acting in language models. In _The Eleventh International Conference on Learning Representations_ , Kigali, Rwanda, April 2023b. URL https://openreview _._ net/forum?id=WE ~~v~~ luYUL-X. 

- Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A Distributed Serving System for Transformer-Based Generative Models. In _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’22)_ , Carlsbad, CA, July 2022. 

- Shoubin Yu, Jaemin Cho, Prateek Yadav, and Mohit Bansal. Self-chained image-language model for video localization and question answering. In _Thirty-seventh Conference on Neural Information Processing Systems_ , New Orleans, Louisiana, December 2023. 

- Zhuosheng Zhang, Aston Zhang, Mu Li, and Alex Smola. Automatic chain of thought prompting in large language models. In _The Eleventh International Conference on Learning Representations_ , Kigali, Rwanda, May 2023. URL https://openreview _._ net/forum?id=5NTt8GFjUHkr. 

- Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-judge with MT-bench and chatbot arena. In _Thirty-seventh Conference on Neural Information Processing Systems Datasets and Benchmarks Track_ , New Orleans, Louisiana, December 2023a. URL https://openreview _._ net/forum?id=uccHPGDlao. 

- Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Jeff Huang, Chuyue Sun, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. Efficiently programming large language models using sglang. _arXiv preprint arXiv:2312.07104_ , December 2023b. 

- Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, and Hao Zhang. Distllm: Disaggregating prefill and decoding for goodput-optimized large language model serving. In _Proceedings of the 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI ’24)_ , Santa Clara, CA, July 2024. 

14 

Published as a conference paper at ICLR 2024 

## **7 Appendix** 

## **A Study on LLM Prompts** 

Today’s LLM usage goes beyond simple chatting. As LLM usage becomes more commercialized, LLM prompts become more structured and complex, outshadowing the text an LLM generates. This section presents our study results of five popular new LLM use cases: tool (or API, agent) use Schick et al. (2023), interacting with virtual environments as an embodied agent Hao et al. (2023); Huang et al. (2022), software program generation Nijkamp et al. (2023), answering questions about videos Xiao et al. (2021), and answer questions about long documents Li et al. (2023a). Figure 8 demonstrates the prompt usages of these workloads. We study each case with real public datasets and understand their prompt features from a systems perspective. For datasets that do not provide outputs, we use Llama-3 7B model as the LLM to generate outputs. For each dataset, we construct a prefix tree for all the requests in the dataset ( _i.e._ , assuming an infinite prefix cache). 

Table 1 and Figure 6 summarize our study results, including prompt and decoding (output) length, amount of sharing in a prompt, key portion size in a prompt, and number of requests sharing a key portion. We define the “key portion” of a request as the deepest node in a path that has more tokens than the sum of its predecessors. 

To understand real-world LLM user request features, we study a recently released public cloud LLM trace. This section ends with our summary insights. 

|**Workload**|**Prompt Len**|**Output Len**|**Shared Prefx**<br>**in Prompt**|**KeyPort.**<br>**in Prompt**|**Req Share KeyPort.**|
|---|---|---|---|---|---|
|**Toolbench**<br>**Embodied Agent**<br>**Programming**<br>**Video QA**<br>**LooGLE**|(1835, 742)<br>(2285, 471)<br>(3871, 1656)<br>(9865, 5976)<br>(23474,6105)|(43, 16)<br>(16, 13)<br>(190, 343)<br>(4, 1.5)<br>(16,9.9)|(85%, 13%)<br>(97%, 14%)<br>(97%, 7.4%)<br>(88%, 32%)<br>(91%,24%)|(76%, 16%)<br>(76%, 12%)<br>(78%, 13%)<br>(99%, 0.2%)<br>(94%,15%)|(39, 64)<br>(48, 8)<br>(126, 2157)<br>(8.6, 2)<br>(18,8.6)|



Table 1: **LLM Prompt Properties** _Each cell except for number of requests shows (mean, standard deviation). Length represented using number of tokens. “KeyPort.” stands for Key Portion._ 

## **A.1 Tool Use** 

Today, LLMs are often augmented by various tools such as calculators and web searches. To equip a model with the ability to invoke a tool, it must be given the correct syntax for querying the tool, along with examples (or “demonstrations”) of tool use. We evaluate the Toolbench Guo et al. (2024) dataset, which consists of more than 210k queries that call over 16k unique tools. Each query shares the same system prompt followed by tool-specific instructions. The final part of the query is the user’s specific question or task. These are all concatenated together to form the final prompt. We find that most of the sharing comes from queries that all share the same tool, and these instructions can be 43x longer than the output length. The Toolbench workload is also representative of other tasks that “prep” an LLM in a similar fashion. For example, instead of tool-calling, LLMs can have roles layered on top of the system prompt, which is popular in emerging systems that utilize the same LLM with multiple roles to create an ensemble Wu et al. (2024); Li et al. (2024b;a). 

## **A.2 Embodied Agents** 

LLMs are increasingly found in agents that can interact with environments, such as a player in a role-playing game or controlling a robot. In this scenario, the LLM receives feedback from the environment, forms an action, and then “performs” the action. This is conducted in a loop until the model has achieved the goal. 

15 

Published as a conference paper at ICLR 2024 

**==> picture [430 x 94] intentionally omitted <==**

**----- Start of picture text -----**<br>
Toolbench Embodied Agent Programming Video QA LooGLE<br>1.0 1.0 1.0 1.0<br>0.8 0.8 0.8 0.8<br>0.6 0.6 0.6 0.6<br>0.4 0.4 0.4 0.4<br>0.2 0.2 0.2 0.2<br>0.0 0.0 0.0 0.0<br>1 10 100 1000 10000 0.0 0.2 0.4 0.6 0.8 1.0 0.2 0.4 0.6 0.8 1.0 0 50 100 150 200 250<br>Prompt-to-Decode Ratio Shared Prefix in Prompt Key Portion in Prompt Req Share Key Portion<br>CDF CDF CDF CDF<br>**----- End of picture text -----**<br>


Figure 6: **CDF Plot of Key Metrics** _Showing CDF for all five workloads on prompt-to-decode ratio, shared prefix_ 

The workload we utilize is sourced from the ALFWorld Shridhar et al. (2021) dataset and has 7.5k requests. Prompts first describe the environment and the task, followed by a demonstration of steps to solve the task. The model then solves its given task by looping over a planning step followed by an action step. After each action, the text-based environment returns an observation that the model incorporates into its next planning step. Every new invocation to the LLM in this loop is treated as a new request, resulting in each step sharing the context of previous steps. Interestingly, the number of steps is determined by LLM generation, creating an unpredictable sharing pattern. Because steps are chained together, prompts are still 157x longer than output tokens. 

The embodied agent workload can represent a wide variety of other use cases, such as chain of thought Yao et al. (2023b); Wei et al. (2024), multi-turn tool usage Wang et al. (2024); Qin et al. (2024b), and chatbots Zheng et al. (2023a). Any dependency between the model and the outside environment can be considered an agent receiving feedback. 

## **A.3 Program Generation** 

One of the popular uses of LLMs is to generate software programs Nijkamp et al. (2023). We study the APPS competitive programming dataset Hendrycks et al. (2021), a dataset of programming problems. To generate better-quality programs, an approach taken by a recent paper Juravsky et al. (2024) is to add a demonstration of several generic code examples before the user problem to instruct an LLM. This added demonstration is the same across all problems and becomes the system prompt. Following the system prompt is the programming problem description. Afterward, this approach invokes the LLM several times in parallel to generate multiple candidate programs, out of which the best is chosen to return to the user. As generated code is relatively long (compared to outputs of other workloads we study), the prompt-to-output ratio (20x) is relatively low. Prompt sharing comes from two places: the system prompt of code demonstration is shared across all requests, and the programming problem is shared across all parallel generations. Depending on how complex the problem is, its description could be longer or shorter than the system prompt; a problem description can also be partially the same as another problem description. Such complexity results in competitive programming having diverse key-portion properties. Such example demonstration and parallel generation technique is common in recent prompt engineering, for example, with ReAct Yao et al. (2023b), Tree-of-Thoughts Yao et al. (2023a), and Self Consistency Wang et al. (2023). 

## **A.4 Video Question and Answer** 

The advent of video models like OpenAI Sora OpenAI (2024b) has created an explosion of interest in multimodal models. The use of LLMs, then, goes beyond natural language. A recent usage is to answer questions about videos by tokenizing a video segment and inputting it to an LLM Yu et al. (2023); Esser et al. (2020). To study this, we analyze the NExT-QA benchmark Xiao et al. (2021), which consists of 8.5K questions for 1000 video segments. Prompts to the LLM consist of a tokenized video followed by a multiple-choice question. Because of the multiple-choice nature, the outputs of this dataset only have six tokens. Long tokens for representing videos plus short outputs result in this dataset having the highest prompt-to-decoding token 

16 

Published as a conference paper at ICLR 2024 

**==> picture [258 x 93] intentionally omitted <==**

**----- Start of picture text -----**<br>
1.0 1.0<br>Conversation<br>0.8 0.8<br>Programming<br>0.6 0.6<br>0.4 0.4<br>0.2 0.2<br>0.0 0.0<br>0.01 1 100 10000 0.1 1 10 100 1000<br>Inter-Arrival Time (ms) Prompt-to-Decode Ratio<br>Figure 7: Azure LLM Trace Analysis Results.<br>CDF<br>**----- End of picture text -----**<br>


ratio of all workloads we explored, with nearly 2500 _×_ more prompt tokens. Apart from videos, images and audio can also be tokenized to have LLMs answer questions, and we expect them to have similar properties as video QA. 

## **A.5 Long Document Question and Answer** 

With newer models, the maximum context length has increased substantially Munkhdalai et al. (2024); Liu et al. (2024); Jacobs et al. (2023), with the latest development supporting 1M tokens Munkhdalai et al. (2024). Longer contexts enable new LLM applications such as asking questions about a long document or even a book. We evaluate this usage with the LooGLE dataset Li et al. (2023a), a collection of 776 long documents and over 6.4k questions. LooGLE has a small system prompt of 13 tokens followed by a long document and then a question about the document. As a common practice, a user or multiple users often ask multiple questions to the same document, resulting in large amounts of shared tokens. Meanwhile, the answers are usually short ( _e.g._ , a true or false). These features result in high prompt-to-decode ratio and high sharing ratio in LooGLE. 

## **A.6 LLM Usages in the Wild** 

To understand LLM usage in the wild, we analyze the recently released Azure LLM Inference Trace Patel et al. (2024). The trace includes two types of LLM usages: program generation and chat conversation. It provides request arrival time, prompt length, and decode length. As it does not provide actual request content, it is not feasible for us to evaluate prompt content or sharing. Figure 7 plot our analysis results in CDF. We find that the arrival rate is approximately 5 requests per second for chat conversation and 7 requests per second for programming. On average, chat requests arrive 118 ms apart while programming requests arrive 63 ms apart. The mean prompt-to-decode ratio for chat conversations is 4. Since we have no details about shared context from follow-up conversations, this number is expected to be much lower. For the longest 20% of all chat prompts, the mean prompt-to-decode ratio is 175, which is consistent with our observations on other workloads. For programming, the mean prompt-to-decode ratio is 92 for all prompts. This falls within the range of all the workloads we evaluated. 

## **A.7 Summary Insights** 

Our analysis of the five real-world LLM workloads and a real user LLM request trace reveals several key findings. 

**Insight 1:** Contrary to popular belief, prompts are significantly longer than output lengths because LLMs support longer context and new LLM usages keep emerging. We believe this trend will continue as LLMs are augmented with more capabilities. **Implication 1:** Optimizing prefill computation can largely improve overall application performance, and imbalanced prefill and decoding computation features should be considered in LLM serving. 

17 

Published as a conference paper at ICLR 2024 

**Insight 2:** Prompt sharing, or reuse, is common, and the sharing amount is high. Sharing can come from different user requests needing the same tools or instructions to solve a task. It can come from a user asking multiple questions about the same document or video. Context sharing can also happen within the same user task that is solved with a chain or a tree of steps. **Implication 2:** Reuse computation across shared prefixes can largely improve real workloads’ performance and should be efficiently supported by distributed LLM serving systems. 

**Insight 3:** Most requests have a portion of the prompt sequence that gets a different degree of sharing and is longer than its prefix, reflected as a key portion in prefix trees. Key portions account for the majority of prompts and are shared by a significant amount of requests. **Implication 3:** Identifying the key portion of prompts and optimizing the placement of requests according to their key portions is a viable way of reducing the complexity of scheduling while achieving good performance. 

**Insight 4:** Real-world LLM usages have varying load intensity, and different usages (programming vs. conversation) have different loads. Real-world prompts are also much longer than decoding length, but different usages have different prompt-to-decode ratios. Still, the longest prompts are significantly longer. **Implication 4:** An efficient LLM serving system should consider complex, mixed-usage scenarios and factor in both load and prompt sharing variations. 

## **B Prefill/Decoding Times** 

The prefill and decoding stages exhibit different computation behaviors, with the former being computationbound and the latter being memory-bandwidth bounded. To understand their behaviors and to acquire prefill/decoding computation time functions to be used by E2, we profile the prefill and decoding stage performance with Mistral 7B on the A6000 GPU. Figure 9 plots the prefill time and its breaking downs when prompt length increases. As seen, longer prompts increase prefill time, suggesting that the more savings we can get from prefix sharing, the lower prefill time will be. Moreover, since the linear layer dominates the model forwarding at the prefill stage, the prefill time is overall linear to the prompt length. Figure 10 shows the performance of a single request’s decoding performance with varying context lengths (the length of the prompt sequence plus the sequence generated thus far). We observe a similar linear relationship to context token length. Overall, these profiling results suggest that attention computation is regular. Thus, we could use the token length with a profile regression function to estimate computation time. 

18 

Published as a conference paper at ICLR 2024 

**==> picture [344 x 404] intentionally omitted <==**

**----- Start of picture text -----**<br>
tool use<br>call<br>System: You are AutoGPT,  Tool 1.  User: .. Could you provide me basic  LLM ans<br>you can use many tools… financial_statement.. daily data…? wer<br>You have access of the  {‘parameters’:<br>following tools: {‘type’: ‘object’, … Tool 2. .. User: …?<br>{‘parameters’:<br>Tool 3. ..<br>… User: …?<br>{‘parameters’: …<br>virtual environment<br>number of steps depends on LLM generation<br>“You are in the middle of a  call  gen env call<br>room with kitchen itemsYour task is to: … LLM go to action act in VE feedbackon the  LLM pick up  env  … actionfinal<br>Here are two examples… stove 1 stove 1,  pan 2 from  feed<br>see a pan 2 stove 1 back<br>program generation call<br>User  LLM 1st parallel code<br>“You are a computer science programmer … Programming  generation<br>Here is an example: Problem 1 call  2nd parallel<br>problem, --Input--, --outpu—, example code LLM code generation<br>Here is another example: … User Problem<br>2 …<br>video QA “What happened to the baby…?  call LLM answer: 0.<br>0. walk away, 1. run after it, …” walk away<br>Tokenized Video<br>“Why did the woman bend down..?  call LLM answer: 1.<br>0. to jump.., 1. excercise, …” excercise<br>document  call<br>QA Question: How many  LLM six<br>Long Document  people…? 1. seven, 2. six, ..<br>Please answer the  1<br>call<br>the long texts below.question based on  Long  Question: In which year did…? LLM 1982<br>Doc 2<br>**----- End of picture text -----**<br>


Figure 8: **Workload Demonstration.** _Green boxes represent shared prefixes. Grey boxes are non-shared prompts. White boxes are output generation. Yellow star represents key portions that always happen at fixed parts; pink stars at non-fixed parts. Blue clouds represent the parts that would be used for distributing prefixes if knowing the oracle._ 

**==> picture [139 x 107] intentionally omitted <==**

**----- Start of picture text -----**<br>
2.0 Linear Layers<br>Self-Attention<br>1.5 Total<br>1.0<br>0.5<br>0.0<br>0 5 10<br>Prompt Length (x1000 #token)<br>Execution time (s)<br>**----- End of picture text -----**<br>


Figure 9: **Prefill Time Decomposition** 

**==> picture [129 x 113] intentionally omitted <==**

**----- Start of picture text -----**<br>
Linear Layers<br>Self-Attention<br>40 Total<br>20<br>0<br>0 10 20 30<br>Context length (x1000 #tokens)<br>Figure 10: Decoding Time<br>Execution time (ms)<br>**----- End of picture text -----**<br>


19 

Published as a conference paper at ICLR 2024 

## **Algorithm 1** E2 Global Scheduling Algorithm 

**function** SCHEDULEREQUEST( _Rk_ ) Match _Rk_ to global radix tree _cached_ ~~_l_~~ _en ←_ sum of matched length _missed_ ~~_l_~~ _en ← prompt_ ~~_l_~~ _en − cached_ ~~_l_~~ _en_ **if** _missed_ ~~_l_~~ _en < cached_ ~~_l_~~ _en_ **then** _▷_ Exploit _Rk K ←_ GPUs with longest node in matched path **for each** GPU _i_ in _K_ **do** _Costi ←_ LOADCOST( _i, Rk_ ) **end for return** _i_ with lowest _Costi_ **else** _▷_ Explore _Rk_ **for each** GPU _i_ in all GPUs **do** _Ratioi ←_ DECODERATIO( _i_ ) **end for** _▷_ IMBALR: calc based on GPU type and LLM **if** highest _Ratiomax >_ IMBALR **then return** _max_ **end if for each** GPU _i_ in all GPUs **do** _Costi ←_ LOADCOST( _i, Rk_ ) **end for return** _i_ with lowest _Costi_ **end if end function** 

20 

Published as a conference paper at ICLR 2024 

**Algorithm 2** GPU Load Cost Calculation 

_▷_ Load cost calculation for GPU _i_ and request _Rk_ **function** LOADCOST( _i_ , _Rk_ ) _L ←_ 0; _M ←_ 0; _P ←_ 0; 

_▷_ Calculate total load on GPU _i_ **for each** _Rj_ in history _H_ **do** _missed_ ~~_l_~~ _en ←_ non-cached prompt length for _j L ← L_ + PREFILLTIME( _missed_ ~~_l_~~ _en_ ) _decode_ ~~_l_~~ _en ←_ average request output length in _H L ← L_ + DECODETIME( _decode_ ~~_l_~~ _en_ ) **end for** 

_▷_ Calculate eviction cost _E ←_ tree nodes to evict on GPU _i_ to run _Rk_ **for each** _j_ in _E_ **do** _Nj ←_ number of requests sharing _j_ in _H_ / number of total requests in _H M ← M_ + PREFILLTIME(length of _j_ ) _× Nj_ **end for** 

_▷_ Calculate cost to run _Rk missed_ ~~_l_~~ _en_ ~~_k_~~ _←_ non-cached prompt length for _Rk P ←_ PREFILLTIME( _missed_ ~~_l_~~ _en_ ~~_k_~~ ) 

**return** _L_ + _M_ + _P_ **end function** 

## **Algorithm 3** E2 Local Scheduling Algorithm 

**function** SCHEDULEREQUESTS **for all** requests in waiting queue **do** Determine prefix hit ratio Assign to priority group based on hit ratio **end for** Process groups in round-robin fashion with limits **end function** 

21 

Published as a conference paper at ICLR 2024 

**==> picture [344 x 141] intentionally omitted <==**

**----- Start of picture text -----**<br>
Preble SGLang<br>200 400<br>150 300<br>100 200<br>50 100<br>0 0<br>2 4 6 2 4 6<br>Rate (req/s) Rate (req/s)<br>P99<br>Avg Latency<br>**----- End of picture text -----**<br>


Figure 11: **vLLM Backend Performance** _Evaluated on the Video QA workload using the Mistral 7B model on 2 GPUs._ 

## **C Comparison with vLLM and Other SGLang Versions** 

To demonstrate Preble’s versatility with multiple LLM backends, we evaluate Preble on vLLM with the vanilla vLLM as the baseline. vLLM recently added support for prefix caching, which we include in the baseline. We use a slightly different version of the Mistral 7B model (v0.2) for this experiment, as vLLM only supports this version. Figure 11 plots the results of running the VideoQA workload on 2 GPUs and the Mistral 7B v0.2 model for both Preble and vLLM. Compared to SGLang as a backend, vLLM as a backend gives Preble less relative improvement for several reasons: 1) local-GPU prefix sharing is in beta version and not as performant as SGLang; 2) vLLM does not use the flash ~~i~~ nfer kernel which makes prefix sharing more efficient; and 3) vLLM does not support chunked prefill together with prefix caching. 4) vLLM has significant scheduling delay 

To demonstrate Preble’s versatility across multiple SGLang versions, we evaluate the latest SGLang version (v0.3) Team (2024b) and the latest FlashInfer kernel (v1.6) Team (2024a) in addition to the v0.1.12 we used in Section 4. SGLang made two major changes between these two versions: 1) the new FlashInfer kernel improved the performance of sharing 32K context length in the same batch, and 2) SGLang reduced its scheduling overhead and applied other engine optimizations. Figure 12 plots the results of running LooGLE and Toolbench with SGLang v0.3 and Preble. Overall, Preble’s improvements over SGLang persist across SGLang versions. 

22 

Published as a conference paper at ICLR 2024 

**==> picture [301 x 270] intentionally omitted <==**

**----- Start of picture text -----**<br>
Preble SGLang<br>Tool Use LooGLE<br>60<br>30<br>40<br>20<br>10 20<br>6 8 10 12 14 0.4 0.6 0.8<br>80<br>200<br>60<br>150<br>40<br>100<br>20<br>50<br>6 8 10 12 14 0.4 0.6 0.8<br>Rate (req/s) Rate (req/s)<br>SGLang v0.3<br>Mistral 7B<br>2xA6000 Avg (s)<br>Mistral 7B<br>2xA6000 p99 (s)<br>**----- End of picture text -----**<br>


Figure 12: **Latest SGLang v0.3 and Flashinfer 0.1.6** 

23 

