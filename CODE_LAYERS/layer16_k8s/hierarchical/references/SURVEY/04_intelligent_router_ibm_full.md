# **INTELLIGENT ROUTER FOR LLM WORKLOADS: IMPROVING PERFORMANCE THROUGH WORKLOAD-AWARE LOAD BALANCING** 

**Kunal Jain**[1] **Anjaly Parayil**[1] **Ankur Mallick**[1] **Esha Choukse**[1] **Xiaoting Qin**[1] **Jue Zhang**[1] **Íñigo Goiri**[1] **Rujia Wang**[1] **Chetan Bansal**[1] **Victor Rühle**[1] **Anoop Kulkarni**[1] **Steve Kofsky**[1] **Saravan Rajmohan**[1] 

## **ABSTRACT** 

Large Language Model (LLM) workloads have distinct prefill and decode phases with different compute and memory requirements which should ideally be accounted for when scheduling input queries across different LLM instances in a cluster. However existing scheduling algorithms treat LLM workloads as monolithic jobs without considering the distinct characteristics of the two phases in each workload. This leads to sub-optimal scheduling and increased response latency. In this work, we start by characterizing factors affecting the response latency during LLM inference serving. We establish that better load balancing of inference requests across the available LLM instances can improve the end-to-end latency to a larger extent than merely focusing on optimizing the instance-level scheduler. Motivated by our findings, we propose a heuristic-guided reinforcement learning-based intelligent router for data-driven and workload-aware scheduling. Our router schedules queries across LLM instances by leveraging a trainable response-length predictor, and a novel formulation for estimating the impact of mixing different workloads and achieves over 11% lower end-to-end latency than existing approaches on a mix of public datasets and 7.8% lower end-to-end latency on real workload data with diverse input and output trends from Cloud Provider X. Additionally, the proposed framework can also serve as a standard for benchmarking different LLM inference schedulers since it provides the best latency for a given model, hardware, and instance-level scheduler combination. 

## **1 INTRODUCTION** 

The emergence of large language models (LLMs) and their generative ability has led to an increase in their usage in conversational engines, search engines, and code assistants Adiwardana et al. (2020); Chen et al. (2021); Roller et al. (2020). The widespread usage of these large models, coupled with the significant GPU compute required for inference, has made LLM inference the dominant GPU workload. Optimizing LLM inference is thus critical for improving user experience, lowering the pressure on GPU resources, and reducing the environmental impact of running LLM workloads, and so there has been a flurry of recent work looking at various aspects of LLM inference optimization Li et al. (2024); Lin et al. (2024); Spector & Re (2023); Zhang et al. (2024). 

LLM inference is usually performed on the cloud by model instances hosted by commercial cloud providers (Google; Microsoft) or dedicated LLM inference platforms (HuggingFace; OpenAI) that serve inference requests from a variety of tenants. Owing to the widespread use of LLMs in chatbots, document summarization, and content creation, the requests vary in terms of their input and output characteristics. Each LLM instance that serves the inference request contains a scheduler, which is a batching system 

responsible for creating a batch of requests by retrieving requests from a queue and scheduling the execution engine. There exist multiple approaches in the literature that try to optimize the batching of these requests at a _single_ LLM instance (Agrawal et al., 2024; Patel et al., 2023; Yu et al., 2022; Zhong et al., 2024) with various goals like reducing queueing delay of requests, maximizing the utilization of the serving infrastructure, etc. For similar reasons, works like (Ding et al., 2024; Ong et al., 2024) have looked at routing requests across _multiple LLMs_ (route easy requests to a small model and hard requests to a big model). However neither of these lines of work have considered _routing requests across multiple instances of a single LLM_ . 

This is a significant gap since all cloud providers host multiple instances of each model and need to design policies for assigning requests to instances such that they can be served with low latency. The wide variety in the size of input queries and LLM responses across scenarios implies that sub-optimal request assignment can significantly increase inference latency. Figure 1a shows spikes in execution time of a request when new requests are added while the LLM instance is still processing an initial request. The execution time of a request during each iteration is significantly impacted by the addition of new requests. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

**==> picture [475 x 150] intentionally omitted <==**

**----- Start of picture text -----**<br>
20<br>50<br>With mixing Round Robin<br>Without mixing Baseline RL<br>0 . 3 15 40 Workload Aware RL<br>Workload Guided RL<br>30<br>0 . 2 10<br>20<br>0 . 1 5 10<br>0 0<br>0 250 500 750 1 , 000 Baseline Workload Workload 0 20 40 60 80 100<br>Iterations of first request RL Aware RL Guided RL Time since start<br>(a) Effects of mixing requests (b) E2E latency (c) Average TTFT of requests served<br>TTFT<br>Avg.<br>Execution Time (s)<br>Improvement over RR (s)<br>**----- End of picture text -----**<br>


_Figure 1._ Key Results: (a) The red curve indicates trend in the execution time when a LLM instance serves a single request, while the blue curve shows spikes in the total execution time of a request due to the addition of additional requests to that instance at fixed intervals. (b) Our RL-based approaches improve upon Round-robin (RR) routing in terms of overall latency with Workload Guided RL reducing average end-to-end latency by 19.18 seconds. (c) The average Time-To-First-Token is the lowest for the proposed approach. 

We conducted an empirical study (see Figure 2) analyzing the disparity between optimal and random assignment of 8 requests arriving at a rate of 1 per second, with varying input and output lengths assigned to two LLM instances using set partitioning. Through exhaustive search of all possible combinations, we found that the best achievable latency was 27.03 seconds, the worst was 32.34 seconds, and a random assignment yielded a latency of 29.81 seconds. On average, a random assignment yields 10% higher end-to-end latency than the optimal assignment. 

In this work, we start by analyzing the prompt and decode phases of inference requests. We estimate the time to complete a request for a given prompt and decode length. Next, we analyze the factors that affect the end-to-end latency of requests running in an LLM instance. These factors include co-serving requests in the prompt and decode phase (determined by the instance-level scheduler), and the diversity in the prompt and decode distribution. By classifying requests based on prompt and decode characteristics, we model the latency impact of mixing incoming and existing requests and propose a latency impact estimator. To help with the latency impact estimation, we develop a lightweight decode length predictor backed by insights from the empirical study. 

Given that requests are served by one of many instances of an LLM in reality, we analyze the interplay between routing strategies and instance-level scheduling algorithms and how they affect the end-to-end latency of the requests. We assert that poor assignment of requests at the instance-level scheduler cannot improve the end-to-end latency, and hence we need better routing strategies that consider characteristics of incoming requests as well as those of requests at each model instance. It is also necessary to treat routing as _distinct_ from instance-level scheduling so that we can consider the efficacy of routing strategies independent of instance level 

innovations such as prefill chunking and prompt caching (Agrawal et al., 2023; Gim et al., 2024). We propose to optimize the routing strategies for _any_ optimizations that exist at each LLM instance. 

Finally, we use the above components to propose an intelligent router for a given instance level scheduler, LLM, and hardware combination. By strategically delaying routing and selecting the best model instance based on the current state of LLM instances and incoming requests, we reduce queuing at model instances which improves latency by 11.43% on average for 2000 requests over 4 LLM instances. We present an extensive evaluation of the proposed framework’s scalability with an increase in the available LLM instances, adaptability to different LLM-hardware combinations, and performance on a real production trace. 

**Contributions:** 1) We assert that poor choices of requests at the instance-level scheduler cannot improve the end-to-end latency beyond a point and established the impact of concurrently serving inference requests with diverse characteristics (section 4). 2) We propose a novel formulation to model this impact (section 4). 3) We develop a lightweight model for predicting decode length, performing well across various prompt and decode characteristics (subsection 5.1). 4) We propose a heuristic-guided, workload-aware reinforcement learning router that encodes prior knowledge of workload mixing effects and routes requests to the best-suited LLM instance, thereby achieving improved end-to-end performance (subsection 5.3). Given that the intelligent router finds the best possible assignment for optimizing end-to-end latency for a given hardware, model, and instance-level scheduler, we present a standard for future benchmarking of inference schedulers. Additionally, the framework also offers flexibility to plug and play different optimizations, such as prefill chunking or prefix caching, to evaluate the best possible 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

improvements in such scenarios. 

## **2 PRELIMINARIES** 

## **2.1 LLM Inference** 

Large Language Models (LLMs) go through prompt/prefill and decode phases while serving a request. The prefill phase processes input tokens in parallel performing self attention computations at each layer and is thus typically compute bound. The decode phase generates subsequent output tokens sequentially based on the forward pass of the last output token and cached context ( **K** and **V** matrices) from previous tokens and is thus memory bound (Agrawal et al., 2023). The response generation ends either when the model produces an EOS token or if the request reaches its maximum token count. A single forward pass of the model is referred to as one iteration of the model (Yu et al., 2022). 

## **2.4 Problem Setup** 

We consider serving a stream of requests using multiple homogeneous LLM instances, each with a scheduler Yu et al. (2022) to iteratively batch requests using a First-Come-FirstServed policy. Requests vary in tasks like summarization, QnA, and translation, each with different prompt and decode characteristics. Requests queue centrally and are routed one at a time to an LLM instance with available capacity. Due to memory constraints, a request may be preempted midprocess if its response exceeds expected size. Our goal is to minimize end-to-end latency by assigning requests to LLM instances, assuming the request arrival rate maintains system equilibrium and given any optimization strategies present at the model-level scheduler. 

## **3 RELATED WORK** 

## **3.1 LLM Serving Systems** 

## **2.2 Scheduler block at LLM instance** 

Each LLM instance that serves the inference request contains a scheduler, which is a batching system responsible for creating a batch of requests by retrieving requests from a queue and scheduling the execution engine. The scheduler controls how many and which requests are processed in each iteration and may use techniques like iteration-level scheduling introduced in Yu et al. (2022) to reduce queueing delay. The highlighted blocks in Figure 3 show the scheduler at each LLM instance. Often, the First-Come-First-Served policy is used for scheduling requests as online requests are latency-sensitive. 

## **2.3 LLM Output Length Prediction** 

Prior works such as _S_[3] (Jin et al., 2023) also focus on optimizing the throughput of the LLM instance by predicting the output sequence length given an input prompt using a lightweight Distilbert model and batching inputs based on the predicted output length. The prediction is treated as a classification problem by dividing output length into 10 uniform buckets and training the predictor to pick the correct bucket for an input. This lightweight approach predicts output length with 98.6% accuracy on a QnA dataset and we build on it in this work. State-of-the-art approaches for predicting output length often rely on expensive models or on relative ranking and probability distributions, as noted by (Fu et al., 2024; Nie et al., 2024; Shahout & Mitzenmacher, 2024; Shahout et al., 2024; Zheng et al., 2024). However, approaches such relative ranking may not effectively capture the performance degradation that occurs when serving requests with different characteristics together 

Recent advancements in inference serving systems for LLMs have focused on optimizing throughput, latency, and resource management. ORCA Yu et al. (2022), Sarathi Agrawal et al. (2023), FlashAttention Dao et al. (2022), and vAttention Prabhu et al. (2024) are examples of systems that have achieved significant improvements in performance through techniques such as iteration-level scheduling, innovative batching, and IO-aware algorithms. 

## **3.2 LLM Serving Algorithms** 

This space has also seen several algorithmic innovations. QLM (Patke et al. (2024)) utilizes Bayesian statistics and stochastic programming to manage non-deterministic execution times inherent in LLMs. Similarly, Qiu et al. (2024) advocates for speculative shortest-job-first scheduling, and Wu et al. (2023) employs preemptive scheduling to improve performance. DistServe and Splitwise (Patel et al. (2023); Zhong et al. (2024)) optimize LLM serving performance by separating prefill and decoding computation for throughput enhancement while maintaining low latency. In addressing system load and request patterns, Jha et al. (2024) and Mendoza et al. (2024) utilize deep reinforcement learning to dynamically adjust service quality, increasing hardware utilization for cost-efficient serving. Additionally, Liu et al. (2024) optimize Quality-of-Experience (QoE) for LLM serving systems, focusing on user-centric metrics to enhance individual experiences. Patke et al. (2024); Sun et al. (2024) proposes a multi-model queue management framework for LLM serving and orchestrate the actions such as model swapping, request eviction, GPU-CPU state swapping, load balancing, and warm model start. While these works optimize request scheduling at the instance level, they ignore the diversity in the prompt and decode characteristics across requests. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

|(a) Experimental Set Up|||
|---|---|---|
||E2E Latency|# of interference spikes|
||27.03<br>28.92<br>29.47<br>30.27<br>31.13<br>32.34|18<br>23<br>22<br>30<br>34<br>40|
||(b) E2E latency and spikes||



_Figure 2._ (a) We assigned 8 requests, arriving at a rate of 1 per second, with input and output lengths varying from 10 to 100, to two LLM instances using set partitioning. Through exhaustive search of all possible combinations, we found that the best achievable latency was 27.03 seconds, the worst was 32.34 seconds, and random assignment resulted in an expected latency of 29.81 seconds. The ideal case is approximately 10% better than the average. (b) The figure shows the number of spikes and their corresponding E2E latency during the simulation. The ideal scenario has a maximum of seven spikes, but we observe a significantly higher number of spikes, suggesting request preemption 

**==> picture [341 x 123] intentionally omitted <==**

_Figure 3._ Our intelligent router optimizes request routing for end-to-end latency by using the output length predictor and workload impact estimator to route incoming requests to the appropriate model instance based on request characteristics and instance state. In contrast, current approaches focus on instance-level scheduling, as shown by highlighted regions around each model instance. Our router achieves optimal improvements regardless of the LLM instance’s optimization strategy. 

## **3.3 Hybrid LLM Inference** 

Recent works (Ding et al., 2022; 2024; Kag et al., 2022; Ong et al., 2024) have introduced a hybrid inference paradigm which uses two _different models_ instead of a single model for inference. The key idea is to save inference cost without compromising on response quality by routing easy queries to the smaller and less capable model (e.g. Mixtral (Jiang et al., 2024)) while the difficult queries are routed to the larger and more capable model (e.g. GPT-4 (OpenAI, 2023)). The routing is typically achieved by training a query-difficulty classifier and is thus different from our reinforcement learning based router which seeks to find the optimal assignment of requests across different instances of the _same model._ 

## **3.4 Reinforcement Learning for routing jobs** 

Reinforcement Learning (RL) has been a natural choice for routing jobs in multi-server queues owing to the challenges in deriving exact policies. While previous works (Jali et al., 

2024; Staffolani et al., 2023) have looked at general jobs, in this work we leverage the specific characteristics of LLM requests and insights from our workload-study to design novel workload aware RL approaches for routing inference requests across LLM instances. 

## **4 OBSERVATION** 

## **4.1 Dataset** 

In general, LLM queries come from different tasks and differ in terms of their prompt and decode distribution. We simulate the prompt and decode distribution using data from five different tasks: sentiment analysis, entity recognition, in-context QnA, general QnA, and translation (prompt details in Appendix A.4). Table 1 summarizes the distribution of input (prompt) and generated output (decode) tokens for the requests from these tasks. We see a clear distinction in the average length of prompt and decode tokens, and in the percentage of requests with heavy decode, across tasks. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

**==> picture [475 x 154] intentionally omitted <==**

**----- Start of picture text -----**<br>
· 10 [−] [2]<br>0 . 8<br>3 . 7 30 With mixing<br>Without mixing<br>0 . 6<br>3 . 6<br>20<br>0 . 4<br>3 . 5<br>10<br>0 . 2<br>3 . 4<br>0 0<br>0 500Prompt Length1 , 500 2 , 500 2 , 000 Total Tokens4 , 000 6 , 000 0 Iteration of first request250 500 750 1 , 000<br>(a) Effect of prompt length (b) Effect of decode length (c) Effect of request mixing<br>Execution Time (s) Execution Time (s)<br>Total Time Elapsed (s)<br>**----- End of picture text -----**<br>


_Figure 4._ **Effects of prompt and decode tokens on batch execution time.** (a) Execution time of batch with a request in prefill phase grows fast and linearly with the number of prefill tokens. (b) Execution time of batch with only decode tokens is affected to a much lesser degree with the number of tokens. (c) Increase in execution time on mixing requests with the arrival pattern of Figure 1a 

|**Source**|**Task**|**Samples**|**Average**<br>**Prompt**|**Tokens**<br>**Decode**|**Heavy Decode**|**Accuracy**<br>**SOTA**<br>**Ours**|**Accuracy**<br>**SOTA**<br>**Ours**|
|---|---|---|---|---|---|---|---|
|BooksTiedemann(2012)|Translation|7351|29.09|61.76|9.18%|4.47%|93.10%|
|Eli5 (Reddit subset)Fan et al.(2019)|QnA|6988|29.83|334.40|58.18%|5.91%|70.36%|
|IMDbMaas et al.(2011)|Sentiment Analysis|6564|211.54|142.53|41.01%|6.81%|79.92%|
|SQuADRajpurkar et al.(2016)|In-context QnA|7122|125.16|220.02|47.95%|6.22%|65.27%|
|WNUTDerczynski et al. (2017)|EntityRecognition|3304|26.41|64.10|8.71%|2.76%|95.06%|
|**Total**|-|31329|89.03|175.71|35.54%|5.5%|79.15%|



_Table 1._ **Dataset and Performance of Output Length Predictor.** Average prompt and decode tokens varies across data sources. The second last column indicates the percentage of requests with heavy decodes ( _≥_ 5 seconds estimated completion time) and the last column indicates the accuracy of our decode length predictor model described in subsection 5.1 for each source. 

## **4.2 Prompt-Decode characteristics of requests and their execution time** 

In this section, we start by characterizing the processing time of a request in terms of their prompt and decode token count. Figures 4a and 4b show that batch execution time increases linearly with the number of prompt tokens due to it’s compute bound nature, and the growth is much slower during the decode phase. Thus, we estimate the processing time for a request with _p_ prompt and _d_ decode tokens as _p ×_ (time per prompt token) + _d ×_ (average decode batch time). Similarly, the earliest time any model instance will become available is (iterations left) _×_ (average batch time). It is to be noted that Figures 4a and 4b correspond to the profiles for Llama-2-7b models on V100. A similar profiling approach can be followed for the processing time of a different LLM and hardware combination. 

**Consistency Across Hardware and Model Combinations** : For consistency across different LLM and hardware combinations, we classify prompt and decode phases as either heavy or light based on their processing time for that LLM and hardware. Requests that take 0.5 seconds or more to complete their prompt phases are defined as heavy prompts, 

while requests that take 5 seconds or more in the decode phase are defined as heavy decodes. These values are hyperparameters that can be tuned to the provider’s needs. It should be noted that input prompts with 1024 tokens may be heavy for a V100, but they may not be classified as heavy for an H100 due to the latter’s better processing capability. We then divide all incoming requests into four categories: light prompt-light decode (LL), light prompt-heavy decode (LH), heavy prompt-light decode (HL), and heavy promptheavy decode (HH). In the following section, we quantify the factors affecting the latency and end-to-end performance of LLM inference. 

## **4.3 Effects of Mixing Requests** 

**Effect of co-serving requests in the prompt and decode phase on a single LLM instance.** To analyze the effect of processing requests in their prompt and decode phase on a single machine, we served a single request on a LLM instance and added requests at fixed intervals while the first request is still in its decode phase. As shown in Figure 1a, the execution time of the first request experienced spikes when new requests were added while the LLM instance was still processing an initial request. The orange curve in 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

Figure 4c shows the ideal latency for the first request, which is 17 seconds for a request with _p_ = 1000 and _d_ = 1000. However, the end-to-end latency increases to 31 seconds due to the arrival of new requests of _p_ = 500 and _d_ = 500 at every 50[th] iteration. The decode phase of the first request experienced increase in execution time due to the latency spikes caused by the prompt phase of each of the incoming requests (Hu et al., 2024). 

**Effect of serving requests with distinct prompt and decode characteristics on a single LLM instance.** Recall that: (i) the latency of the incoming request during the prompt phase increases rapidly and linearly with an increase in the number of prompt tokens, and (ii) the decode phase has a much lower impact, and the mean iteration time varies slowly with an increase in total tokens. As we can see from Figure 4 (Llama-2-7b models profiled on V100 GPUs), the gradient for our configuration can be calculated as 3 _._ 2 _×_ 10 _[−]_[4] and 3 _._ 3 _×_ 10 _[−]_[5] . Thus mixing requests with different prompt and decode characteristics in a batch at a model instance can impact overall latency due to mismatch and interference between the prompt and decode phase of the requests. We see a roughly linear increase in batch execution with an increase in token count. The same approach can be followed for other model and hardware combinations. 

## **4.4 Factors affecting E2E Latency** 

Given that requests are served by multiple LLM instances in reality, we analyze the interplay between routing strategies and model-level scheduling algorithms and how they affect the end-to-end latency of the requests. We take two LLM instances and consider the following arrival patterns of 3000 requests: 1) LH and HL requests arriving in a random fashion, 2) Requests from all four classes arriving in a random fashion, 3) LH requests arriving first followed by HL requests, and 4) HL requests arriving first followed by LH requests. We study the end-to-end latency in each scenario with different combinations of batching and routing algorithms. Details of these algorithms are added to Appendix A.1. We can see from Table 2 that it is the combination of routing and batching algorithms that affects the overall end-to-end latency of processing the requests. 

For example, the bin-packing scheduler finishes Scenario I six seconds faster when using round-robin routing compared to decode balancer routing but the same combination is four seconds slower when benchmarked on Scenario II. We also observe that sub-optimal routing strategies, such as having dedicated servers for small and large requests, can have a severely adverse effect on the overall performance of our system. This highlights the importance of optimizing the routing strategy and shows that _scheduling algorithms can only provide results as good as the choices provided to it by the router_ . Another interesting insight is that, for 

Scenario III and IV, all the batching algorithms show the exact same end-to-end latency, and it is the routing algorithm that improves the end-to-end latency which hints at the need to adapt scheduling policies based on the characteristics of LLM requests and their arrival sequence. 

## **4.5 Insights** 

Quantitative analysis highlights the importance of considering request characteristics and avoiding serving requests with diverse characteristics concurrently at a single LLM instance, especially when multiple LLM instances are available. Further, the scheduling algorithms and optimizations at the LLM instance can only provide results as good as the choice provided to it by the router. Anchoring on our thesis that a batching algorithm can only provide results as good as the choices provided to it, and that different optimizations can exist in the instance-level scheduler, in the next section, we propose to find optimal routing strategies for a given model-level scheduler, hardware, and model combination. The next section will discuss the overall design of the router and the individual building blocks required. 

## **5 INTELLIGENT ROUTER: DESIGN** 

Figure 3 shows the overall design of the intelligent router. Based on the insights from section 4, an intelligent router should: a) classify requests based on prompt-decode characteristics and be able to estimate decode length, b) estimate the adverse effect of mixing diverse requests at a single model instance on end-to-end latency, c) leverage prior knowledge of these adverse effects for decision making, and d) possess lightweight modules for efficient processing. To achieve this, we develop an output length predictor and workload impact estimator for intelligent routing. Additionally, we propose a reinforcement learning framework to utilize accumulated context and prior knowledge, improving end-to-end latency. 

## **5.1 Output length predictor** 

Similar to (Jin et al., 2023), we generate responses for each request in the dataset discussed in section 4 and categorize each request into a bucket based on the number of output tokens in its response. We use these buckets as labels and input prompts as inputs to fine-tune a DistillBERT model for predicting the range of output tokens for new requests. However, instead of using buckets of equal size, we define the bucket ranges based on the estimated completion time for the request. Following the heavy-light decode logic described in section 4, we predict buckets with ranges 0 _−_ 0 _._ 5 seconds, 0 _._ 5 _−_ 0 _._ 5 _×_ 4 seconds, 0 _._ 5 _×_ 4 _−_ 0 _._ 5 _×_ 8 seconds and so forth. Our experimental set up produces roughly 500 decode tokens per second on average, to bucket bucket sizes of 0 _−_ 250, 250 _−_ 1000, 1000 _−_ 4000 and so on. Our 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

|**Batching Algorithm**<br>**Routing Algorithm**|**Total End to End Latency (seconds)**<br>**(LH, HL random)**<br>**(Random)**<br>**(LH, then HL)**<br>**(HL, then LH)**|
|---|---|
|Bin PackingJin et al.(2023)<br>Dedicated Small-Large<br>Round Robin<br>Decode Balancer|704.5<br>644.75<br>566.25<br>588.15<br>**581.5**<br>559.3<br>424.8<br>**440.68**<br>595.82<br>555.4<br>424.82<br>440.81|
|Least Work Left<br>Dedicated Small-Large<br>Round Robin<br>Decode Balancer|704.5<br>641.81<br>566.25<br>588.15<br>585.14<br>**554.00**<br>**424.64**<br>440.82<br>596.95<br>559.97<br>424.66<br>440.81|
|FCFSYu et al.(2022)<br>Dedicated Small-Large<br>Round Robin<br>Decode Balancer|704.5<br>648.66<br>566.25<br>588.15<br>607.45<br>572.16<br>424.80<br>440.82<br>605.65<br>573.17<br>424.82<br>440.81|



_Table 2._ **Performance of batching and routing algorithm combinations.** We simulate arrival of requests with distinct characteristics using the request classification discussed in section 4 and test the combined affect of routing and batching strategies. Good routing algorithm on an average shows greater end-to-end latency improvement compared to the batching algorithm on all the scenarios with distinct characteristics and arrival sequence. 

choice of (unequal) bucket sizes helps us better distribute the requests among the buckets. This approach provides our routing strategy with actionable information by aligning the bucket ranges with the expected completion times. 

We observe that the approach by Jin et al. (2023) does not directly generalize well to our dataset, which comprises requests from distinct prompt/decode distributions. The model achieves an accuracy of only 5.5% in predicting unequal-sized buckets and 9.3% in predicting equal-sized buckets of 250 tokens. Leveraging our insight from section 4 that input and output characteristics depend on the task type, we enhance the model’s performance by appending the task type as a hint to the prompt provided to the DistillBERT model. Employing this technique, we achieve an accuracy of 79.15% in predicting unequally sized buckets and 68.23% in predicting equal-sized buckets. We can predict task type from the input prompt with an accuracy of 93.79% (c.f. subsection A.7). 

## **5.2 Workload impact estimator** 

Next, we use the profiling approach from Section 4 to obtain the analytic expression for the processing times of the prompt and decode phase. Let there be _n_ requests within model instance _m_ , and _p[m] j_[and] _[ d][m] j_[indicate the number of] prompt and decode tokens processed by the _j_ -th existing request at model _m_ . As the impact on the prompt phase is directly proportional to the number of prompt tokens in the request and the total number of tokens already running in the decode phase, we can model the impact on the prompt phase (time to process _pi_ , _Tp[m] i_[)][ of an incoming request with] _pi_ tokens when added to the model instance with n requests, and corresponding penalty as: 

**==> picture [185 x 19] intentionally omitted <==**

**==> picture [186 x 31] intentionally omitted <==**

Here, we introduce a penalty if the latency impact exceeds _ϵ_ . Similarly, the impact on existing requests beyond the prompt phase is directly proportional to the total number of requests in the model. We can model the penalty due to the impact of an incoming request with _pi_ prompt tokens and _di_ decode tokens on the decode phase of already existing _n_ requests as: 

**==> picture [214 x 30] intentionally omitted <==**

With our selection of grad1 as 3 _._ 2 _×_ 10 _[−]_[4] and grad2 as 3 _._ 3 _×_ 10 _[−]_[5] , we expect the values _rd[m]_[and] _[ T][ m] pi_[to be in the] ranges of [ _−_ 1 _,_ 1] and [ _−_ 1 _,_ 0] respectively when there are no requests waiting at the model instance. We combine Equation 1 and Equation 2 to get the final penalty of mixing requests: _r_ mixing( _st, st_ +1) = _αrp[m] i_[+ (1] _[ −][α]_[)] _[r] d[m]_[where] _[ m]_ is the action taken for the state transition _st → st_ +1. Here, parameter _α ∈_ (0 _,_ 1) balance priority over the prompt and decode phases. 

## **5.3 RL based router** 

We formulate the problem of routing incoming requests to the _m_ model instances as discrete-time Markov Decision Process (MDP) and propose a reinforcement learning-based solution. The discounted MDP is denoted as a tuple _M_ = ( _S, A, P, r, γ_ ), where _S_ is the state space, _A_ is the action space, _P_ ( _s[′] |s, a_ ) is the transition dynamics, _r_ ( _s, a_ ) is the reward function, and _γ ∈_ [0 _,_ 1) is the discount factor. The goal is to find the policy _π_ ( _s|a_ ), a distribution of actions over states, which maximizes the discounted return _Gt_ = � _∞k_ =0 _[γ][k][r][t]_[+] _[k]_[+1][ in expectation.][We assume an arrival rate] of _λ_ for the requests and the ideal estimated time to complete request _i_ as _T_[ˆ] _i_ . Let _ojt_ denote the total number of output 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

tokens produced until time _t_ by request _j_ , and _d_[ˆ] _jt_ denote the estimated decode tokens for the _j_ -th request. Therefore, we denote the fraction of request _j_ completed at time _t_ by _fjt_ := _d[o]_ ˆ _[j] jt[t]_[.][We assume state transition at every][ ∆] _[t]_[, where] ∆ _t_ is the average time to generate a decode token. 

**State Space** : At time _t_ , the state of the system, which comprises _m_ model instances and requests waiting in the queue, can be captured by the following: 1) The number of requests in the queue at time _t_ , denoted by _wqt_ ; 2) The exact number of prompt tokens, denoted by _pt ∈_ R, and the estimated bucket for the decode tokens, denoted by _dt ∈{_ 0 _, . . . , nd}_ , corresponding to the next request in the queue. Here, the estimated bucket varies from zero to _nd_ ; 3) Matrices, **P** _t ∈_ R _[mn][p]_ and **D** _t ∈_ R _[mn][d]_ , capturing the prompt and decode distribution of requests at the model instances. We represent the prompt (decode) distribution by _np_ ( _nd_ ) buckets. ( **P** _t_ ) _i,j_ (( **D** _t_ ) _i,j_ ) denotes the number of requests in prompt (decode) phase at the _i_ -th model instance that are present in the _j_ -th bucket i.e. have _j_ prompt (decode) tokens. We represent the prompt and decode distribution across the model instances as a matrix, which maintains the finite dimensionality of the state space. 4) The capacity available at the model instances at time t is denoted by **C** _t ∈_ R _[m]_ , as a function _g_ (batch size _,_ **P** _t,_ **D** _t_ ); and 5) The estimated completion time for the earliest request in model _j_ , denoted by _T_[ˆ] _ct_ . 

**Action Space** : At any given point in time, the agent must decide whether to schedule incoming request _i_ to any of the _m_ model instances or choose to take no action. Therefore, _a ∈{_ 0 _, . . . , m}._ Here, index _m_ refers to no action being taken by the router. 

**Reward Design** : Based on the insights from section 4, we include the following elements in the reward formulation: a) a negative penalty for requests in the queue, decreasing as requests are processed, to account for the autoregressive nature of requests, b) a positive reward for each completed request, and c) a workload impact estimator-based penalty, which encodes the adverse effect of routing specific requests to a model instance with existing requests, and prevents requests from being queued at each model instance due to a lack of memory. Note that adding the workload impact estimator-based penalty directly to the reward function might introduce bias. Therefore, we propose to augment the prior knowledge using a heuristic-guided formulation (Cheng et al., 2021), and the reward at time t is given by: 

**==> picture [185 x 81] intentionally omitted <==**

**==> picture [235 x 89] intentionally omitted <==**

**----- Start of picture text -----**<br>
0 . 5 Round Robin 0 . 5 1 . 2<br>0 . 4 Workload Aware RLBaseline RL 0 . 4 1<br>Workload Guided RL 0 . 8<br>0 . 3 0 . 3<br>0 . 6<br>0 . 2 0 . 2 0 . 4<br>0 . 1 0 . 1 0 . 2<br>0 0<br>0 20 40 60 80 100 Baseline Workload Workload Baseline Workload Workload<br>Time since start RL Aware RL Guided RL RL Aware RL Guided RL<br>(a) Average TBT of(b) Average TBT dis-(c) Queue length at<br>requests served tribution model instance<br>TBT (s) TBT (s)<br>waiting requests<br>Avg. Avg.<br>Avg.<br>**----- End of picture text -----**<br>


_Figure 5._ **Experimental Results.** We simulate the arrival of 2000 requests, each with distinct characteristics, at an arrival rate of 20 requests per second and average results over 20 episodes RoundRobin performs better initially in terms of average TBT, but the value increases over time as more requests with different characteristics accumulate. Workload Guided RL minimizes the variance in average number of waiting requests and TBT values, with fewer requests in the waiting queue compared to other methods. 

where 

**==> picture [214 x 31] intentionally omitted <==**

Here, _J_ includes the set of scheduled and unscheduled requests, and **w** _mit ∈{_ 0 _,_ 1 _}_ indicates whether the _i[th]_ at the _m[th]_ model completed at time _t_ . _rw ∈_ Z[+] is the positive reward for completing a request. The function _h_ : **S** _×_ **S** _→_ R represents the difference in penalty due to assigning the incoming request to a model other than the one for which the impact is minimum (a function of equations Equation 1 and Equation 2). The term "guidance discount" is given by _γ_ ˜ _k_ = _λkγ_ , where the subscript _k_ denotes the _k_ -th episode. Here, _λk ∈_ [0 _,_ 1] is the mixing coefficient and settles to zero with an increase in episodes (Cheng et al., 2021). The discount factor in the MDP is set to _γ_ ˜ _e_ during training. The function _h_ () returns zero when the request is assigned to the model with the least workload mixing impact. Intuitively, the formulation introduces horizon-based regularization, and its strength diminishes as the mixing coefficient increases, which modifies the orig-˜ inal MDP, _M_ = ( _S, A, P, r, γ_ ), to _M_ = ( _S, A, P,_ ˜ _r,_ ˜ _γ_ ). Over the course of training, the agent interacts with the environment, and the effects of the heuristic in the MDP decrease, and the agent eventually optimizes for the original MDP. Guarantees on the boundedness of the reshaped MDP’s value function directly translate from (Cheng et al., 2021). 

## **6 EXPERIMENTS** 

We conducted an extensive evaluation of the proposed framework to evaluate the following: 

1. What is the performance improvement of the intelligent 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

router compared to different heuristics on the dataset presented in Section 4? 

2. How does the performance improvement change with an increase in the number of model instances available for serving inference requests and for different hardware and LLM combinations? 

3. How does the intelligent router perform and adapt in the presence of different optimizations available at the model-instance level? 

4. How does the performance of an intelligent router vary with different datasets and when some information, such as prompt content, is missing? 

**Evaluation metrics** : For all the experiments, we report the end-to-end latency, Time-To-First-Token (TTFT), which is the time taken for the user to see the initial response, and Time-Between-Tokens (TBT), which is the average token streaming latency (Patel et al., 2023). Additionally, we report the throughput achieved by different approaches. We included three variants of the RL formulation, including the baseline RL formulation with a reward function consisting of only the first and second terms from Equation 3, workload-augmented RL which simply adds the penalty from _r_ mixing to the baseline RL (workload knowledge augmented), and workload-guided RL that uses the heuristicguided formulation 3. 

**Setup** We route requests between four instances of LLama2-7b-hf model (Touvron et al., 2023) on a cluster of four V100 GPUs using vLLM (Kwon et al., 2023) with its default First-Come-First-Served (FCFS) scheduler for iterationlevel scheduling. We assume an average request arrival rate of _λ_ = 20 _/s_ , with requests uniformly sampled at random from the dataset in Section 4. The routing of requests to model instances is asynchronous, and we take actions every 0.02 seconds, which is the minimum decode batch execution time. 

˜ For baseline RL, we set _γ − γe_ = 0 in the reward function from Equation 3. For workload-aware RL, we directly augment the penalty for mixing requests to the reward function. ˜ Therefore, we set _γ − γe_ = 1. For all experiments, we give equal weight to the impact on the prompt and decode phase. Therefore, _α_ for equations Equation 1 and Equation 2 is set to 0.5. For workload-guided RL, we use the guidance mechanism from Equation 3. We set _λk_ = _e[−][β][d][k]_ (exponential decay over each episode) with _βd_ = 0 _._ 5, and the guided discount factor for training _γ_ as ˆ _γ_ = (1 _− e[−][β][d][k]_ ) _γ_ . Additional details on the model training are added to Appendix A. 

## **6.1 Performance evaluation of Intelligent router** 

Here we compare the performance improvements with respect to various heuristics. 

**End-to-end latency** : We evaluate RL based approaches over 20 episodes, each comprising 2,000 requests with distinct characteristics. As shown in Figure 1b, our methods outperformed Round-Robin in terms of end-to-end latency for servicing all requests. Baseline RL surpassed Round Robin by an average of 7.53 seconds (4.35%). Incorporating the workload-aware penalty into the reward function enhanced this advantage to 13.50 seconds (7.79%), and utilizing the penalty as heuristic guidance for the RL agent improved the advantage to 19.18 seconds (11.43%). This is intuitive as heuristics should only be employed as a warm start and should be reduced as the agent collects more information about the environment. 

Classical heuristics such as Join Shortest Queue, Maximum Capacity Usage, and Min-Min Algorithm (Chen et al., 2013) only marginally outperformed Round Robin by 0.46%, 2.60%, and 1.50%, respectively, in terms of end-to-end latency. These results are intuitive as classical heuristics do not translate well for LLM workloads due to their unique nature. Due to this, we only provide further results in comparison to Round Robin. We provide further details on these algorithms in the appendix subsection A.2. 

**Improvements in TTFT and TBT** : RL-based approaches outperformed the Round-robin router in terms of average TTFT (Figure 1c), with significant improvement as the number of accumulated requests increased over time. Baseline RL halved the TTFT for late-arriving requests by finding a better assignment than Round-robin. Workload-aware penalty further enhanced these decisions, but not optimally, as it diluted the urgency to complete requests promptly and introduced constant bias. Workload-guided RL performed the best by selecting more optimal model instances and mitigating spikes in TBT of existing requests (Figure 5a). Although Round-robin performed better initially in terms of average TBT, the value increased over time as more requests with different characteristics accumulated. Workload awareness effectively reduced the number of outliers and the variance of the distribution (Figure 5b). 

**Queuing at Router and Model Instance** : Figure 5c illustrates the average length of the waiting queue at the model instances. While Baseline RL exhibited a shorter average waiting time of 0.59 seconds at the router, the requests got preempted at the model instances and accumulated substantial delays. This approach was suboptimal since postponing the routing decision could have resulted in a better model instance getting assigned and resulted in faster processing of the request. In contrast, Workload-aware RL, with an average router wait time of 4.41 seconds, addressed this issue by incorporating a penalty based on the workload. Workload Guided RL further refined this strategy by utilizing the penalty as a guidance mechanism, resulting in an average router wait time of 2.05 seconds and improved overall performance. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

|RoutingAlgorithm<br>Round Robin|Prefll Chunking<br>No|Avg. E2E Latency (s)<br>248.41|Improvement<br>-|
|---|---|---|---|
|Baseline RL|No|240.58|3.15%|
|Workload Aware RL<br>Workload Guided RL|No<br>No|231.66<br>221.80|6.74%<br>10.71%|
|Round Robin<br>Baseline RL<br>Workload Aware RL|Yes<br>Yes<br>Yes|247.30<br>240.68<br>231.12|0.45%<br>3.11%<br>6.96%|
|Workload Guided RL|Yes|220.93|11.06%|



_Table 3._ Intelligent router was able to generalize the approach across different model and hardware combinations, outperforms heuristics, and shows additional improvements even with chunked prefills. 

**==> picture [235 x 90] intentionally omitted <==**

**----- Start of picture text -----**<br>
1000 894Prompt 920 Generation923 935 1210 RoundBaselineWorkloadRobinRLGuided RL 0.080.07<br>800 8 Workload Aware RL 0.060.05<br>715 713 719 716 6 0.04<br>4 0.03<br>600 2 0.02<br>0 0.01<br>RoundRobin BaselineRL AwareWorkloadRL GuidedWorkloadRL 0 20 Time40 since60start 80 100 RoundRobin BaselineRL AwareWorkloadRLGuidedWorkloadRL<br>(a) Throughput on (b) Average TTFT(c) TBT distribution<br>A100s with Llama 3.1 with chunking with chunking<br>(s)<br>(tokens/s) TTFT TBT<br>Avg. Avg.<br>Throughput<br>**----- End of picture text -----**<br>


_Figure 6._ Model and hardware generalizability: Experiments on A100s with Llama 3.1 8B shows that intelligent router maintains prompt and generation throughput similar to Round Robin. Intelligent router still outperform Round Robin in the presence of optimizations such as chunked prefills. 

## **6.2 Different LLM and Hardware combination** 

We conducted experiments on different LLM and hardware combinations, specifically testing on A100 with Llama-3.18B. Due to better processing capabilities, we increased the arrival rate to 80 RPS, benchmarked the gradients again for the hardware/model combination, and retrained our agent with the same remaining hyperparameters. With more requests coming in, the router had many more decisions to make. Even then, our strategies were able to outperform Round Robin by similar margins (10.81%), as shown by the first four rows of Table 3. We observe in Figure 6a that our methods maintain prompt and generation throughput similar to Round Robin. Round Robin exhibits similar throughput over a longer period of time, indicating that it generates more tokens to service the same number of requests, highlighting the impact of request preemption. Additional experiments were conducted to validate the scalability of the proposed approach, and the results are presented in subsection A.11. The intelligent router outperformed Round Robin by 11.62% when evaluated on a setting with eight LLM instances. 

et al., 2023). The aim is to assess the performance improvements achieved by the intelligent router in the context of optimizations at the instance-level scheduler. 

For Round-Robin, chunked prefill tokens only improves performance by 0.45%, which could be due to experimental noise. Chunking is not primarily intended to improve E2E latency but rather to enhance user experience by reducing TBT/decode throughput at the expense of TTFT. However, we observe that our method is able to adapt well to this new setting and maintain its lead over Round Robin. Figure 6b shows that the intelligent router still improves TTFT with chunking, despite the fact that chunking is supposed to harm TTFT. Figure 6c shows that TBT has much less variance now, and the average TBT across methods is the same. The performance gains are intuitive, as the intelligent router prevents preemptions of requests and selects the best suitable LLM instance for each request based on the request characteristics and other requests currently being served by each instance. Additional experiments that validate performance improvements on a different dataset, which is the real production trace from Cloud provider X, have been added to subsection A.12. 

## **7 LIMITATIONS AND CONCLUSION** 

We propose a heuristic-guided Reinforcement Learning (RL) based router for efficiently scheduling requests across homogeneous LLM instances. Our approach introduces and models the novel notion of performance impact resulting from serving workloads with distinct characteristics concurrently. By incorporating prior knowledge on mixing workloads with distinct characteristics and their related adverse effects into the router, we are able to improve overall end-to-end latency over current approaches. Our formulation is sufficiently generalized to improve request-level metrics such as Time-To-First-Token (TTFT) and Time-Between-TwoTokens (TBT), and can be extended to include additional requirements such as serving throughput. 

Our extensive experimental evaluations demonstrate the superior performance of our proposed approach over other baselines, different model-hardware combinations, and with respect to different datasets. Although the RL based approach imposes additional overhead on the LLM inference serving compared to heuristics, our framework enables the identification of optimal load balancing strategies for a given model-hardware combination and for a given LLM instance level scheduler. As such, we hope that it will serve as a standard for future benchmarking of inference schedulers for the community. 

## **6.3 Performance in the presence of SOTA Optimizations** 

Next, we will evaluate the performance of the intelligent router in the presence of chunked prefill tokens (Agrawal 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

## **REFERENCES** 

- Adiwardana, D., Luong, M.-T., So, D. R., Hall, J., Fiedel, N., Thoppilan, R., Yang, Z., Kulshreshtha, A., Nemade, G., Lu, Y., et al. Towards a human-like open-domain chatbot. _arXiv preprint arXiv:2001.09977_ , 2020. 

- Agrawal, A., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., and Ramjee, R. Sarathi: Efficient llm inference by piggybacking decodes with chunked prefills. _arXiv preprint arXiv:2308.16369_ , 2023. 

- Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B. S., Tumanov, A., and Ramjee, R. Taming throughput-latency tradeoff in llm inference with sarathiserve, 2024. 

- Chen, H., Wang, F., Helian, N., and Akanmu, G. User-priority guided min-min scheduling algorithm for load balancing in cloud computing. In _2013 National Conference on Parallel Computing Technologies (PARCOMPTECH)_ , pp. 1–8, 2013. doi: 10.1109/ ParCompTech.2013.6621389. 

- Chen, M., Tworek, J., Jun, H., Yuan, Q., Pinto, H. P. d. O., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., et al. Evaluating large language models trained on code. _arXiv preprint arXiv:2107.03374_ , 2021. 

- Cheng, C.-A., Kolobov, A., and Swaminathan, A. Heuristicguided reinforcement learning. _Advances in Neural Information Processing Systems_ , 34:13550–13563, 2021. 

- Dao, T., Fu, D., Ermon, S., Rudra, A., and Ré, C. Flashattention: Fast and memory-efficient exact attention with io-awareness. _Advances in Neural Information Processing Systems_ , 35:16344–16359, 2022. 

- Derczynski, L., Nichols, E., van Erp, M., and Limsopatham, N. Results of the WNUT2017 shared task on novel and emerging entity recognition. In _Proceedings of the 3rd Workshop on Noisy User-generated Text_ , pp. 140– 147, Copenhagen, Denmark, September 2017. Association for Computational Linguistics. doi: 10.18653/ v1/W17-4418. URL https://www.aclweb.org/ anthology/W17-4418. 

- Ding, D., Amer-Yahia, S., and Lakshmanan, L. V. On efficient approximate queries over machine learning models. _arXiv preprint arXiv:2206.02845_ , 2022. 

- Ding, D., Mallick, A., Wang, C., Sim, R., Mukherjee, S., Ruhle, V., Lakshmanan, L. V., and Awadallah, A. H. Hybrid llm: Cost-efficient and quality-aware query routing. _arXiv preprint arXiv:2404.14618_ , 2024. 

- Fan, A., Jernite, Y., Perez, E., Grangier, D., Weston, J., and Auli, M. ELI5: long form question answering. In 

Korhonen, A., Traum, D. R., and Màrquez, L. (eds.), _Proceedings of the 57th Conference of the Association for Computational Linguistics, ACL 2019, Florence, Italy, July 28- August 2, 2019, Volume 1: Long Papers_ , pp. 3558–3567. Association for Computational Linguistics, 2019. doi: 10.18653/v1/p19-1346. URL https:// doi.org/10.18653/v1/p19-1346. 

- Fu, Y., Zhu, S., Su, R., Qiao, A., Stoica, I., and Zhang, H. Efficient llm scheduling by learning to rank. _arXiv preprint arXiv:2408.15792_ , 2024. 

- Gim, I., Chen, G., Lee, S.-s., Sarda, N., Khandelwal, A., and Zhong, L. Prompt cache: Modular attention reuse for low-latency inference. _Proceedings of Machine Learning and Systems_ , 6:325–338, 2024. 

- Google. Vertex ai. https://cloud.google.com/ vertex-ai. 

- Hu, C., Huang, H., Xu, L., Chen, X., Xu, J., Chen, S., Feng, H., Wang, C., Wang, S., Bao, Y., et al. Inference without interference: Disaggregate llm inference for mixed downstream workloads. _arXiv preprint arXiv:2401.11181_ , 2024. 

- HuggingFace. Hugging face inference api. https:// huggingface.co/inference-api. 

- Jali, N., Qu, G., Wang, W., and Joshi, G. Efficient reinforcement learning for routing jobs in heterogeneous queueing systems. _arXiv preprint arXiv:2402.01147_ , 2024. 

- Jha, S., Hooper, C., Liu, X., Kim, S., and Keutzer, K. Learned best-effort llm serving. _arXiv preprint arXiv:2401.07886_ , 2024. 

- Jiang, A. Q., Sablayrolles, A., Roux, A., Mensch, A., Savary, B., Bamford, C., Chaplot, D. S., Casas, D. d. l., Hanna, E. B., Bressand, F., et al. Mixtral of experts. _arXiv preprint arXiv:2401.04088_ , 2024. 

- Jin, Y., Wu, C.-F., Brooks, D., and Wei, G.-Y. $s^3$: Increasing GPU utilization during generative inference for higher throughput. In _Thirty-seventh Conference on Neural Information Processing Systems_ , 2023. URL https: //openreview.net/forum?id=zUYfbdNl1m. 

- Kag, A., Fedorov, I., Gangrade, A., Whatmough, P., and Saligrama, V. Efficient edge inference by selective query. In _The Eleventh International Conference on Learning Representations_ , 2022. 

- Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J. E., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles_ , 2023. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

- Li, J., Xu, L., Xu, H., and Akella, A. Blockllm: Multi-tenant finer-grained serving for large language models. _arXiv preprint arXiv:2404.18322_ , 2024. 

- Lin, B., Peng, T., Zhang, C., Sun, M., Li, L., Zhao, H., Xiao, W., Xu, Q., Qiu, X., Li, S., et al. Infinite-llm: Efficient llm service for long context with distattention and distributed kvcache. _arXiv preprint arXiv:2401.02669_ , 2024. 

- Liu, J., Wu, Z., Chung, J.-W., Lai, F., Lee, M., and Chowdhury, M. Andes: Defining and enhancing quality-ofexperience in llm-based text streaming services. _arXiv preprint arXiv:2404.16283_ , 2024. 

- Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., and Potts, C. Learning word vectors for sentiment analysis. In _Proceedings of the 49th Annual Meeting of the Association for Computational Linguistics: Human Language Technologies_ , pp. 142–150, Portland, Oregon, USA, June 2011. Association for Computational Linguistics. URL http://www.aclweb. org/anthology/P11-1015. 

- Mendoza, D., Romero, F., and Trippel, C. Model selection for latency-critical inference serving. In _Proceedings of the Nineteenth European Conference on Computer Systems_ , pp. 1016–1038, 2024. 

- Microsoft. Azure ai studio. https://ai.azure. com/. 

- Nie, C., Fonseca, R., and Liu, Z. Aladdin: Joint placement and scaling for slo-aware llm serving. _arXiv preprint arXiv:2405.06856_ , 2024. 

- Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M. W., and Stoica, I. Routellm: Learning to route llms with preference data. _arXiv preprint arXiv:2406.18665_ , 2024. 

- OpenAI. Openai platform. https://platform. openai.com/overview. 

- OpenAI. Gpt-4 technical report, 2023. 

- Patel, P., Choukse, E., Zhang, C., Íñigo Goiri, Shah, A., Maleki, S., and Bianchini, R. Splitwise: Efficient generative llm inference using phase splitting, 2023. 

- Patke, A., Reddy, D., Jha, S., Qiu, H., Pinto, C., Cui, S., Narayanaswami, C., Kalbarczyk, Z., and Iyer, R. One queue is all you need: Resolving head-of-line blocking in large language model serving, 2024. URL https: //arxiv.org/abs/2407.00047. 

- Prabhu, R., Nayak, A., Mohan, J., Ramjee, R., and Panwar, A. vattention: Dynamic memory management for serving llms without pagedattention. _arXiv preprint arXiv:2405.04437_ , 2024. 

- Qiu, H., Mao, W., Patke, A., Cui, S., Jha, S., Wang, C., Franke, H., Kalbarczyk, Z. T., Ba¸sar, T., and Iyer, R. K. Efficient interactive llm serving with proxy model-based sequence length prediction. _arXiv preprint arXiv:2404.08509_ , 2024. 

- Rajpurkar, P., Zhang, J., Lopyrev, K., and Liang, P. SQuAD: 100,000+ questions for machine comprehension of text. In Su, J., Duh, K., and Carreras, X. (eds.), _Proceedings of the 2016 Conference on Empirical Methods in Natural Language Processing_ , pp. 2383–2392, Austin, Texas, November 2016. Association for Computational Linguistics. doi: 10.18653/v1/D16-1264. URL https: //aclanthology.org/D16-1264. 

- Roller, S., Dinan, E., Goyal, N., Ju, D., Williamson, M., Liu, Y., Xu, J., Ott, M., Shuster, K., Smith, E. M., et al. Recipes for building an open-domain chatbot. _arXiv preprint arXiv:2004.13637_ , 2020. 

- Shahout, R. and Mitzenmacher, M. Skippredict: When to invest in predictions for scheduling. _arXiv preprint arXiv:2402.03564_ , 2024. 

- Shahout, R., Malach, E., Liu, C., Jiang, W., Yu, M., and Mitzenmacher, M. Don’t stop me now: Embedding based scheduling for llms. _arXiv preprint arXiv:2410.01035_ , 2024. 

- Spector, B. and Re, C. Accelerating llm inference with staged speculative decoding. _arXiv preprint arXiv:2308.04623_ , 2023. 

- Staffolani, A., Darvariu, V.-A., Bellavista, P., and Musolesi, M. Rlq: Workload allocation with reinforcement learning in distributed queues. _IEEE Transactions on Parallel and Distributed Systems_ , 34(3):856–868, 2023. 

- Sun, B., Huang, Z., Zhao, H., Xiao, W., Zhang, X., Li, Y., and Lin, W. Llumnix: Dynamic scheduling for large language model serving. _arXiv preprint arXiv:2406.03243_ , 2024. 

- Tiedemann, J. Parallel data, tools and interfaces in OPUS. In Calzolari, N., Choukri, K., Declerck, T., Do˘gan, M. U., Maegaard, B., Mariani, J., Moreno, A., Odijk, J., and Piperidis, S. (eds.), _Proceedings of the Eighth International Conference on Language Resources and Evaluation (LREC’12)_ , pp. 2214–2218, Istanbul, Turkey, May 2012. European Language Resources Association (ELRA). URL http://www.lrec-conf.org/proceedings/ lrec2012/pdf/463_Paper.pdf. 

- Touvron, H., Martin, L., Stone, K., Albert, P., Almahairi, A., Babaei, Y., Bashlykov, N., Batra, S., Bhargava, P., Bhosale, S., Bikel, D., Blecher, L., Ferrer, C. C., Chen, 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

- M., Cucurull, G., Esiobu, D., Fernandes, J., Fu, J., Fu, W., Fuller, B., Gao, C., Goswami, V., Goyal, N., Hartshorn, A., Hosseini, S., Hou, R., Inan, H., Kardas, M., Kerkez, V., Khabsa, M., Kloumann, I., Korenev, A., Koura, P. S., Lachaux, M.-A., Lavril, T., Lee, J., Liskovich, D., Lu, Y., Mao, Y., Martinet, X., Mihaylov, T., Mishra, P., Molybog, I., Nie, Y., Poulton, A., Reizenstein, J., Rungta, R., Saladi, K., Schelten, A., Silva, R., Smith, E. M., Subramanian, R., Tan, X. E., Tang, B., Taylor, R., Williams, A., Kuan, J. X., Xu, P., Yan, Z., Zarov, I., Zhang, Y., Fan, A., Kambadur, M., Narang, S., Rodriguez, A., Stojnic, R., Edunov, S., and Scialom, T. Llama 2: Open foundation and fine-tuned chat models, 2023. 

- Wu, B., Zhong, Y., Zhang, Z., Huang, G., Liu, X., and Jin, X. Fast distributed inference serving for large language models. _arXiv preprint arXiv:2305.05920_ , 2023. 

- Yu, G.-I., Jeong, J. S., Kim, G.-W., Kim, S., and Chun, B.G. Orca: A distributed serving system for TransformerBased generative models. In _16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)_ , pp. 521–538, Carlsbad, CA, July 2022. USENIX Association. ISBN 978-1-939133-28-1. URL https://www.usenix.org/conference/ osdi22/presentation/yu. 

- Zhang, Z., Sheng, Y., Zhou, T., Chen, T., Zheng, L., Cai, R., Song, Z., Tian, Y., Ré, C., Barrett, C., et al. H2o: Heavy-hitter oracle for efficient generative inference of large language models. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Zheng, Z., Ren, X., Xue, F., Luo, Y., Jiang, X., and You, Y. Response length perception and sequence scheduling: An llm-empowered llm inference pipeline. _Advances in Neural Information Processing Systems_ , 36, 2024. 

- Zhong, Y., Liu, S., Chen, J., Hu, J., Zhu, Y., Liu, X., Jin, X., and Zhang, H. Distserve: Disaggregating prefill and decoding for goodput-optimized large language model serving. _arXiv preprint arXiv:2401.09670_ , 2024. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

## **A APPENDIX / SUPPLEMENTAL MATERIAL** 

## **A.1 Batching and Routing Algorithms** 

All the batching algorithms are non-preemptive in nature, meaning that once the processing of a request has started, it is prioritized over requests which have not. Next, we discuss different batching and routing algorithms defined in section 4. 

## _A.1.1 Batching: Bin Packing Algorithm_ 

When a new request’s processing can be started, we select the largest request that can fit into the memory available. Ties are broken by FCFS. 

## _A.1.2 Batching: Least Work Left_ 

Among the requests available, we select the request with the smallest number of decode tokens. 

## _A.1.3 Batching: FCFS_ 

The request which arrives first is processed first. 

## _A.2.3 Min-Min Algorithm_ 

We implemented the classical Min-min algorithm, using the number of prompt tokens and the upper bund of the predicted decode token buckets to calculate the time for finishing each job. Since we have homogenous model instances, this strategy becomes similar to shortest job first. 

## **A.3 Overhead of the Router** 

For each decision, the router has to perform two additional steps in our approach: (i) inference from DistillBERT for output length bucket and (ii) inference from the neural network being used. The approach can parallelize these modules when the number of requests in the queue is large (and the request being routed has already been processed by the length predictor). (i) takes us 0.01 (on GPU) and 0.8 (on CPU) seconds per batch of size 64 and (ii) takes _<_ 10[6] operations (within miliseconds) to process. 

## **A.4 Details of Dataset** 

From each dataset, we take the subset of prompts that have a maximum prompt length of 1000 tokens. 

## _A.1.4 Routing: Dedicated Small-Large_ 

For the two LLM model instances, we dedicate one instance for servicing only the heavy-decode requests while the other model instance services only the light-decode requests. 

## _A.1.5 Routing: Round Robin_ 

Each of the two model is user alternatively by the router to send requests to. 

## _A.1.6 Routing: Decode Balancer_ 

We assume that the total number of output tokens is known beforehand for the request and we balance the sum of decode tokens on both the model instances. 

## _A.4.1 Prompts_ 

For each task, the prompt is created in the following manner: 

**Sentiment Analysis (IMDb dataset)** For each review in the dataset, we randomly select one of the following sentences and add it to the review: 

1. "Based on this review, judge if the user liked this movie or not?" 

2. "Please identify if the review is positive or negative?" 

3. "Based on this review, should we recommend this movie to other users with similar tastes?" 

## **A.2 Additional Baselines** 

We implemented three baselines other than Round Robin and the Light-weight Heuristic: 

## _A.2.1 Join Shortest Queue_ 

Each arriving request is routed to the model with the least number of prompt and decode tokens yet to be processed. 

## _A.2.2 Maximum Capacity Usage_ 

Request at the front of the queue is routed to the model with the maximum capacity available, given that it can process this particular request, at intervals of one second. 

We add these tasks either at the beginning or at the end of the prompts, again randomly. 

**QnA (Eli5 Reddit subset)** We pick the question in the title as it is and provide it as the prompt to the LLM. 

**Entity Recognition (WNUT dataset)** We add the suffix "Can you identify the <entity> mentioned in the above sentence?" where <entity> is selected ranomdly from "person", "place", and "object". 

**In context QnA (SQuAD dataset)** We add the question as well as the four options of the answer to the prompt and ask the LLM to select the correct option and provide reasoning with it as well. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

**==> picture [118 x 98] intentionally omitted <==**

_Figure 7._ Prompt decode distribution for our dataset with responses generated from Llama 2 7B model. 

**==> picture [118 x 98] intentionally omitted <==**

**==> picture [120 x 131] intentionally omitted <==**

**----- Start of picture text -----**<br>
(a) Distribution of entire trace<br>(b) Distribution of certain appli-<br>cations<br>**----- End of picture text -----**<br>


_Figure 8._ Prompt-decode distribution of the production traffic from Cloud provider X. 

**Translation (Books dataset)** We provide the text and add the phrase "Please translate this text into <language>" either at the start or at the end of the text. <language> is selected from the ones provided in the dataset itself. 

## _A.4.2 Task Hints_ 

For providing a hint of the task to the model, we add the phrase "This is a <task> task" at the end of each prompt before providing it to the classifier. 

## **A.5 Prompt-Decode Distribution** 

Figure 7 shows the distribution of prompts and decode tokens across the different datasets we mixed. We can clearly see the different distributions each dataset has. Prompts from Eli-5 Reddit subset are shorter in length and have longer responses than the rest of the dataset, while the IMDb 

distribution on the other hand has longer prompt lengths and shorter responses. Such a varied distriubtion contributes to the low accuracy of the current SOTA model by Jin et al. (2023). 

## **A.6 Training details of the output length predictor** 

We had a total of 31329 samples in our mixed dataset, from which we had an 80:20 train-test split. We had a train time accuracy of 81% after performing 6 epochs of fine-tuning with the entire training set. 

## **A.7 Task Predictability** 

We predict the task of a prompt sampled from our dataset described in section 4 using DistillBERT, the same methodology we use to predict their output length bucket as discussed in subsection 5.1. We observe an accuracy of 93.79%. 

This allows us to proceed safely with the assumption that we can provide task type as part of the prompt to the output length predictor. 

## **A.8 Licenses** 

1. WNUT Dataset: CC-by-4.0 

2. SQuAD dataset: CC-by-SA-4.0 

3. vLLM: Apache-2.0 

## **A.9 Details of RL training** 

For our experiments, we use 4 LLM model instances to route among. This results in our state space having 27 dimensions (6 for each model instance and 4 for the request queue at the router). In order to bound our state space, we round the estimated capacity available at each model instance and the estimated completion time for the earliest request to two decimal places. We also upper bound the waiting queue length that we provide to the DQN to 4 _×_ (max batch size) = 4 _×_ 128 = 512. We provide the DQN with 3 buckets: 0-256, 256-2048, _≥_ 2048. 

## _A.9.1 Q-Learning_ 

Q-Learning yields poor performance for our task due to the size of the state space. If we upper bound the total number of requests that can be present at a model instance to 150 (even though there can be infinitely many) and the prompt and decode length to 4096 (maximum content window of LLama-2 7B), each model instance can be in 150 _×_ 150 _×_ 100 _×_ (4096 _×_ 150 _×_ grad2 _×_ 100) = 3 _._ 0405 _×_ 10[9] different states. This would result in a total of (3 _._ 0405 _×_ 10[9] )[4] _×_ 512 _×_ 4096 _×_ 3 _≈_ 5 _×_ 10[44] . Even though we will never visit most of these states, the possible states that be visited are large enough to make Q-Learning infeasible. 

**Intelligent Router for LLM Workloads: Improving Performance Through Workload-Aware Load Balancing** 

**==> picture [71 x 51] intentionally omitted <==**

_Figure 9._ Training reward for workload guided RL 

**==> picture [230 x 80] intentionally omitted <==**

**----- Start of picture text -----**<br>
(b) Workload Aware (c) Workload Guided<br>(a) Baseline RL RL RL<br>**----- End of picture text -----**<br>


_Figure 10._ Rewards collected during testing for each strategy 

## _A.9.2 Training Rewards_ 

Figure 9 Shows the rewards collected during training of the RL model. We see that the guidance heuristic helps the agent converge. After episode 20, we no longer use explore with random actions and exploit this knowledge. 

## _A.9.3 Double-DQN_ 

We take a double DQN approach for our RL agent. We set the request completion reward to 60 and train our DQN with a batch size of 512. We use a neural network with layer sizes (27 _,_ 64) _,_ (64 _,_ 64) _,_ (64 _,_ 5) and ReLU activation function for layers 1 and 2. 

Figure 10 shows the rewards collected by each strategy during testing. Requests stop arriving at iteration number 4000, after which, we see the rewards tend to positive values due to the high request completion reward. 

## **A.10 Overhead of the router** 

For each decision, the router has to perform two additional steps in our approach: (i) inference from DistillBERT for output length bucket and (ii) inference from the neural network being used. The approach can parallelize these modules when the number of requests in the queue is large (and the request being routed has already been processed by the length predictor). (i) takes us 0.01 (on GPU) and 0.8 (on CPU) seconds per batch of size 64 and (ii) takes _<_ 10[6] operations (within milliseconds) to process. 

## **A.11 Additional experiments to validate the scalability of proposed framework** 

To scale our approach, we needed to increase the parameters in our neural network. Our methods outperformed the Round-Robin approach in this setup as well. On average, Baseline RL, Workload Aware RL, and Workload Guided RL outperformed Round Robin by 5.84%, 6.64%, and 11.62%, respectively. 

## **A.12 Experiments on Real Production Trace from Cloud Provider X** 

Next, we validate our approach using one hour production trace from Cloud provider X. We use 4000 requests for our experiments, with average prompt length of 5526.64 tokens and average decode length of 112.69 tokens. We do our experiments at 80 requests per second, again using Llama3.1-8B model. We enable chunking for this experiment with maximum number of batched tokens set to 1024. Round robin takes 1005.31 seconds on average (across 20 random iterations). We see that the advantages of our algorithms are less pronounced when the prompt length becomes much longer than the decode length, with advantages of baseline RL, workload aware RL and workload guided RL reducing to 2.28% (982.38 seconds), 4.39% (961.17 seconds) and 7.84% (926.49 seconds) respectively. This can also be attributed to the lesser number of preemptions happening as the decode length has gotten shorter. 

To reduce the overhead of output length prediction, we assume the unavailability of prompt content and only assume the availability of prompt token count. Therefore, for the bucket prediction module, we train a Random Forest which takes the prompt length of the request along with the application name associated with the request. Using the same bucket sizes as before, this module is able to achieve 79% accuracy (while 68.44% of the requests were in bucket 0) due to the predictable nature of production traffic. Figure 8 shows the prompt and decode distribution from the production trace. The prompt and decode distribution of applications from the production trace show distinct trend as shown in Figure 8b which makes the decode length predictable with prompt length and application type. 

## **A.13 Additional Proofs** 

Reshaping the MDP ( _M_ ) with heuristic guided RL preserves the value bounds and linearity of the original MDP: 1) If _h_ ( _s_ ) _∈_ [0 _,_ 1 _−_ 1 _γ_[]][, then value function corresponding to] the policy, _π_ , _V_[˜] _[π]_ ( _s_ ) _∈_ [0 _,_ 1 _−_ 1 _γ_[]][ for all] _[ π]_[ and] _[ s][ ∈S]_[. 2) If] _[ M]_ is a linear MDP with feature vector _ϕ_ ( _s, a_ ) (i.e. _r_ ( _s, a_ ) and E _s′|s,a_ [ _g_ ( _s[′]_ )] for any _g_ that can be linearly parameterized in _ϕ_ ( _s, a_ ) (Cheng et al., 2021). 

To further test the scalability of our approach, we tested our methods with eight model instances. We increased the number of processed requests to 4,000 and the request arrival rate to 40/s to remain consistent with previous experiments. 

