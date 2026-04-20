# Speculative Decoding and Beyond: An In-Depth Survey of Techniques

**Yunhai Hu**¹\*, **Zining Liu**²\*, **Zhenyuan Dong**¹\*, **Tianfan Peng**¹,³\*, **Bradley McDanel**⁴, **Sai Qian Zhang**¹†

¹ New York University  
² University of Pennsylvania  
³ Shenzhen Institute of Information Technology  
⁴ Franklin and Marshall College  
{yunhai.hu, zd2362, sai.zhang}@nyu.edu · zliu0@seas.upenn.edu · tianfanpeng@gmail.com · bmcdanel@fandm.edu

\* Equal contributions. † Corresponding author.

---

**Abstract** — Sequential dependencies present a fundamental bottleneck in deploying large-scale autoregressive models, particularly for real-time applications. While traditional optimization approaches like pruning and quantization often compromise model quality, recent advances in generation-refinement frameworks demonstrate that this trade-off can be significantly mitigated.

This survey presents a comprehensive taxonomy of generation-refinement frameworks, analyzing methods across autoregressive sequence tasks. We categorize methods based on their generation strategies (from simple n-gram prediction to sophisticated draft models) and refinement mechanisms (including single-pass verification and iterative approaches). Through systematic analysis of both algorithmic innovations and system-level implementations, we examine deployment strategies across computing environments and explore applications spanning text, images, and speech generation. This systematic examination of both theoretical frameworks and practical implementations provides a foundation for future research in efficient autoregressive decoding.

**Index Terms** — Large Language Model, Speculative Decoding, Computer System, Distributed System.

---

## I. Introduction

Large Models (LMs) have demonstrated remarkable capabilities across diverse domains, from text generation [1], [2], [3] and translation [4], [5], [6] to image synthesis [7], [8], [9] and video generation [10], [11], [12]. However, these models face a critical challenge: their inherently sequential nature creates significant latency bottlenecks, particularly for real-time applications. While traditional optimization approaches like quantization and pruning often compromise model quality for speed, recent research has focused on maintaining output quality while breaking sequential dependencies through novel algorithmic and system-level innovations.

Generation-refinement frameworks have emerged as a promising family of solutions that directly address these sequential bottlenecks. These approaches encompass a range of methods, from speculative decoding with draft models to iterative refinement techniques inspired by numerical optimization. The common thread among these approaches is their division of the generation process into two phases: an initial generation step that produces draft tokens in parallel, followed by a refinement step that ensures output quality.

The implementation of these frameworks presents unique system-level challenges across different deployment scenarios. Edge devices require careful optimization of memory usage and computation patterns [13], [14], while distributed systems must manage complex communication patterns and load balancing. These system-level considerations have driven innovations in areas like kernel design, hardware acceleration, and batch processing optimization, significantly influencing both algorithmic choices and practical performance.

This survey synthesizes research across these approaches, examining both algorithmic innovations and their system implementations. We present a systematic taxonomy of generation-refinement methods, analyze deployment strategies across computing environments, and explore applications spanning text, images [15], [16], and speech [17], [18]. Our primary contributions include comprehensive analysis of system-level implementations and optimizations, detailed examination of applications across modalities, and identification of key research challenges in efficient neural sequence generation.

---

## II. The Sequential Bottleneck in Large Model Inference

### A. Understanding Sequential Dependencies

Modern LLMs, such as the Llama series [19], [20], [21] and the GPT series [22], [1], are built on transformer architectures consisting of stacked decoder blocks. As shown in Figure 1(a), each decoder block contains two fundamental components: a Self-Attention (SA) block and a feed-forward network (FFN). During execution, the input of the SA block is first multiplied with three weight matrices *W_Q*, *W_K*, and *W_V*, yielding the outputs termed query (*q*), key (*k*), and value (*v*), respectively.

The computation flow, detailed in Figure 1(b), shows how query and key vectors compute attention scores through matrix multiplication. After softmax normalization, these scores weight the value vectors, producing the SA output through a weighted sum and residual connection. This SA output feeds into the FFN, typically implemented as either a standard MLP [23], [22] or gated MLP [24], [19], [20], with multiple fully connected layers and activation functions like GeLU [25] or SiLU [26].

The core challenge emerges during inference, which consists of two main phases: prefill and decoding. While the prefill phase can process input sequences in parallel, the decoding phase introduces a critical bottleneck. As shown in Figure 1(c), the model must predict each token sequentially, using both current and previous token information through their Key and Value (KV) vectors. These KV vectors are cached for subsequent predictions, leading to significant memory access latency as the sequence length grows.

**Figure 1:** (a) The Llama architecture consists of stacked transformer decoder blocks. (b) Each decoder block contains a self-attention (SA) block and feedforward (FFN) block. (c) During the decoding stage, tokens are generated auto-regressively.

### B. Breaking Sequential Dependencies

Traditional approaches to accelerating LM inference have focused on reducing computational costs through model compression, knowledge distillation, and architectural optimizations. However, these methods primarily address individual computation costs rather than the fundamental sequential dependency that requires each token to wait for all previous tokens.

Speculative decoding (SD) [27] has emerged as a promising solution that directly targets this sequential bottleneck. As illustrated in Figure 2, this approach introduces a two-phase process where a smaller, faster *draft model* first predicts multiple tokens in parallel, followed by verification using the target model. The draft model enables parallel token generation, breaking away from traditional token-by-token generation, while the target model's verification step maintains output quality through accept/reject decisions.

This strategy has proven particularly valuable for real-time applications like interactive dialogue systems, where response latency directly impacts user experience. The verification mechanism provides a crucial balance between generation speed and output quality, accepting correct predictions to maintain throughput while falling back to sequential generation when necessary to preserve accuracy.

While SD represents one successful approach to breaking sequential dependencies in autoregressive (AR) models, it belongs to a broader family of *generation-refinement* methods. The following sections present a systematic taxonomy of these approaches, examining how different techniques balance the trade-offs between generation parallelism and output quality.

**Figure 2:** Illustration of speculative decoding workflow.

---

## III. A Taxonomy for Generation and Refinement Frameworks

To systematically analyze approaches for breaking sequential dependencies in large models, we propose a unified taxonomy that categorizes methods based on their generation and refinement strategies. As shown in Figure 3, our taxonomy decomposes these frameworks into two fundamental phases: *Sequence Generation* and *Sequence Refinement*. This decomposition not only encompasses traditional SD approaches but also captures a broader range of emerging methods that trade off between generation parallelism and output quality.

The sequence generation phase focuses on different strategies for producing draft tokens more efficiently than conventional auto-regressive decoding using a single larger model. These strategies range from simple approaches like random token sampling (used in conjunction with iterative decoding) to more sophisticated methods like retrieval-based generation and draft model prediction. Each generation method offers tradeoffs in terms of computational cost and prediction quality. The sequence refinement phase then determines how these candidates are processed — either accepting them directly (with possible poorer quality), verifying a subset of tokens in a single pass, or refining the draft tokens through multiple iterations until convergence.

**Figure 3:** A taxonomy of generation-refinement frameworks, showing two phases: (1) Generation of draft tokens through various methods and (2) Refinement through verification strategies.

**Figure 4:** Taxonomy of Speculative Decoding Algorithms. Symbols indicate implementation approach: ▲ Direct application (no training required), ∙ Full model training from scratch, ■ Model fine-tuning, 🟊 Parameter-efficient fine-tuning (PEFT), ◆ Knowledge distillation from target model.

The full taxonomy (Figure 4) maps methods across all categories:

| Category | Methods |
|---|---|
| Predefined Fill Tokens (§IV-A) | ▲ Jacobi [28], ▲ LOOKAHEAD [29], ■ CLLMs [30] |
| Retrieval-based (§IV-B) | ▲ LLMA [31], ▲ REST [32], ■ Speculative RAG [33] |
| N-gram-based (§IV-C) | ▲ ANPD [34], ▲ The N-Grammys [35], ▲ ADED [36] |
| Independent Drafter (§IV-D1) | ■◆ SpecDec [37], ∙ SpecDec++ [38], ■◆ BiLD [39], ▲ ON-THE-FLY [40], ◆ OSD [41], ◆ DistillSpec [42], ◆ FastDraft [43], ∙ Judge [44], ◆∙■ [45] |
| Dependent Drafter — Layer-Skipping (§IV-D2) | ■ SPEED [46], 🟊◆ FREE [47], ▲ Draft&Verify [48], ∙ LayerSkip [49], ■ Kangaroo [50], ◆ EESD [51], ▲ SWIFT [52], ■ Speculative Streaming [53], ▲ Draft on the Fly [54] |
| Dependent Drafter — FFN Heads (§IV-D2) | ▲ EAGLE [55], ◆ Falcon [56], ◆ HASS [57], ◆ Hydra [58], ◆ Mixture of Attentions [59] |
| Multi-Token Generation (§IV-E) | ■◆ Blockwise [27], [60], ◆🟊 Medusa [61], ∙ [62], ■ Amphista [63], ◆ CTC-based Drafting [64] |
| Linear Verification (§V-A1) | ■◆ SpecDec [37], ▲ Draft&Verify [48], ▲ Fast Inference [65], ∙ [66], ▲ Block verification [67], ▲ MTAD [68], [69] |
| Tree-based Verification (§V-A2) | ▲ SpecTr [70], ■ SpecInfer [71], ■ Staged SD [72], ▲ Sequoia [73], ◆🟊 Medusa [61], ▲ EAGLE [55], ▲ EAGLE-2 [74], ■ ProPD [75], ▲ OPT-Tree [76], ▲ DSBD [77], ▲ GSD [78], ▲ RSD [79], ◆ ReDrafter [80], 🟊 Speculative Streaming [53], ▲ ADED [36], ▲ DySpec [81], ▲ SpecHub [82], ▲ Multi-Draft SD [83], ▲ [84] |
| Parallel SD (§VI-A) | ■ SPEED [46], ▲ CS Drafting [85], 🟊◆ FREE [47], ■ PPD [86], ■ PASS [87], ■ Faster Cascades [88], ▲ PEARL [89], ▲ Ouroboros [90], ∙ ParallelSpec [91], ■ SPACE [92] |
| Distributed SD (§VI-B) | ▲ SpecExec [13], ▲ EdgeLLM [14], ◆ Dovetail [93] |
| Compiler/Hardware (§VI-C) | ▲ SpecPIM [94], ▲ MagicDec [95], ▲ BASS [96], ▲ SEED [97], ▲ PipeInfer [98], ◆ [99], ◆ SKD [100], ▲ [101], ▲ [69] |
| Vision (§VII-A) | ▲ [15], ▲ LANTERN [16], ▲ SJD [102] |
| Multimodal (§VII-B) | ■🟊 VADUSA [17], ■ [18], ∙ [103], ■ IbED [104] |
| Recommendation Systems (§VII-C) | ▲ DARE [105], 🟊 AtSpeed [106] |

---

## IV. Sequence Generation Methods

### A. Predefined Fill Tokens

The simplest approach uses random initialization or predefined tokens (e.g., PAD). While computationally free, these methods provide poor initialization points, requiring multiple refinement iterations as discussed in Section V-B.

### B. Retrieval-based Methods

LLMA [31] first proposed exploiting overlaps between LLM outputs and reference documents to accelerate inference through parallel token verification while maintaining identical generation results. In retrieval-based approaches, REST [32] replaces smaller language models with exact suffix matching from a datastore to generate draft tokens. It builds a Trie (prefix tree) from retrieved continuations, where node weights reflect token sequence frequencies. Speculative RAG [33] uses a fine-tuned specialist LM to generate complete answer drafts with supporting rationales. It clusters retrieved documents by similarity, generates diverse drafts from different document subsets, and employs self-consistency and self-reflection scores for draft evaluation instead of token-level verification.

### C. N-gram-based Methods

Several approaches leverage n-gram patterns for efficient token generation. ANPD [34] replaces traditional draft models with an adaptive N-gram system that updates predictions based on context. LOOKAHEAD [29] uses n-gram verification by collecting and utilizing n-grams from previous iterations as draft tokens. The N-Grammys [35] further develops this idea by creating a dedicated n-gram based prediction system that can operate without requiring a separate draft model.

### D. Auto-regressive Generation

Most sequence generation methods employ auto-regressive drafting, where a smaller model generates draft tokens that are verified by a larger target model. This drafting paradigm has spawned numerous techniques that vary in how the draft model interacts with the target model.

#### D.1 Independent Drafters

Auto-regressive independent drafters are techniques in which smaller model(s) generate tokens one at a time while a separate larger target model subsequently verifies the draft tokens in parallel. SpecDec [37] pioneered this approach with an independent draft model using distinct attention queries for masked positions. SpecDec++ [38] improves SpecDec [37] by training a prediction head on top of the draft model that estimates the probability of token acceptance by the target model. Based on these predictions, it dynamically determines when to stop generating tokens and trigger verification.

Recent works focus on dynamic adaptation and confidence monitoring. BiLD [39] triggers target model verification when draft confidence falls below a threshold, while ON-THE-FLY [40] dynamically adjusts window sizes based on prediction accuracy. OSD [41] enables online adaptation through knowledge distillation during inference, and DistillSpec [42] extends this by accessing target model logits for improved alignment. [45] introduces special tokens for draft models to autonomously determine target model consultation, eliminating separate verification at some performance cost. For mathematical applications, Judge [44] adds a learned verification layer atop the target model's embeddings, using contextual correctness assessment to reduce strict output alignment requirements.

#### D.2 Dependent Drafters

The main drawbacks of independent drafting approaches are that (1) the computation required to generate the draft tokens is fixed per token, meaning that computation is over-provisioned for many "easy" tokens and (2) the target model cannot reuse the features of the drafting process, increasing the amount of compute required. Self-speculative decoding approaches generate draft tokens by relying directly on a subset (**Layer Skipping**) or extension (**Dependent Heads**) of the target model.

**Layer Skipping.** Draft&Verify [48], SWIFT [52], and Draft on the Fly [54] achieve fast draft token generation by selectively skipping some intermediate layers in the draft process, and then verify these drafts using the full LLM. In order to achieve good draft accuracy, they also designed an intermediate layer selection algorithm based on Bayesian optimization. LayerSkip [49] uses an early exiting [107] approach to dynamically output tokens at different depths of the target model. Kangaroo [50] also applied early exit by adopting a shallow sub-network to generate drafts and using a lightweight adapter module to bridge the performance gap with the full model, achieving efficient and accurate decoding. EESD [51] uses a Thompson Sampling Control [108] mechanism to adaptively determine how many draft tokens will be generated. SPEED [46] combines speculative execution with parameter sharing, using early predictions to process multiple tokens in parallel through shared decoder layers, rather than waiting for each token to complete sequentially.

**Dependent Heads.** Dependent head-based drafting eliminates the need for a separate draft model by adding lightweight feed-forward prediction heads using the hidden states of the target model. The main idea is that the first token in a sequence generation block uses the target model as usual, but the features at the end of the model are fed into additional heads to predict subsequent tokens without passing back through the entire target model.

EAGLE [55] uses a trained head that takes in hidden states from the target model and generates subsequent draft tokens in an AR manner. Hydra [58] uses multiple decoding heads, one for each draft token position.

EAGLE extensions have focused on improving parallel token generation and attention mechanisms. Falcon [56] introduces a semi-autoregressive framework combining LSTM layers and relaxed causal-masked self-attention to generate k tokens per forward pass, while HASS [57] enhances knowledge distillation by prioritizing high-probability tokens during training. Mixture of Attentions [59] incorporates multiple attention types (LSA, SA, and CA) for improved token prediction, and DeepSeek-V3 [109] adapts [62]'s multi-token approach (discussed in Section IV-E) while maintaining complete causal attention during inference.

### E. Multi-token Prediction

[27] proposes adding multiple decoding heads on top of a model to predict *k* future tokens in parallel, requiring training the entire model from scratch. Medusa [61] introduces a parameter-efficient approach, where lightweight decoding heads are fine-tuned on top of pre-trained language models. Each head is trained to predict a specific future position in the sequence without modifying the target model. [62] proposes a multi-token prediction paradigm where a shared backbone is optimized jointly with multiple prediction heads that enable propagation of information related to sequential tokens during training, and can be discarded at inference to enable parallel generation (similar to Medusa).

Recent improvements enhance Medusa's independent draft heads by modeling inter-token relationships. Amphista [63] uses bi-directional self-attention to consider both past and future predictions, while CTC Drafting [64] employs Connectionist Temporal Classification (CTC) with blank tokens and repetition, followed by duplicate removal to generate draft sequences.

---

## V. Sequence Refinement Methods

### A. Single-pass Verification

Single-pass verification represents the most common refinement strategy in draft-and-verify approaches, where drafted tokens are verified exactly once by the target model.

#### A.1 Linear Verification

Linear verification sequentially validates draft tokens against the target model's logit distributions, with early works like SpecDec [37] and Draft&Verify [48] comparing drafted tokens against the target model's predictions. When a token fails verification (i.e., when the draft output doesn't match the target model's distribution), the system falls back to standard AR generation from that point.

Fast Inference [65] and [66] introduced speculative sampling to improve acceptance rates while approximately maintaining the target distribution. Their method accepts a token if the target model assigns equal or higher probability; otherwise, it accepts with probability p(x)/q(x) or resamples from an adjusted distribution.

Block Verification [67] and MTAD [68] improve upon linear verification by examining the joint probability distribution of draft tokens as a chain of conditional probabilities. This block-based evaluation approach typically results in higher acceptance rates compared to token-by-token verification for similar quality.

#### A.2 Tree-based Verification

Tree-based verification extends the single-pass paradigm by enabling parallel exploration of multiple completion paths. Unlike linear verification that processes a single sequence, tree-based methods construct and verify a tree of possible completions simultaneously, making more efficient use of parallel compute resources.

SpecInfer [71] pioneered this approach by developing an efficient tree-based attention masking scheme that enables parallel verification while maintaining proper token dependencies. This innovation maintains generation quality while significantly increasing the number of tokens that can be verified in parallel.

Recent works have focused on optimizing tree structure and size to maximize computational efficiency. Sequoia [73] introduces a hardware-aware tree optimizer that maximizes inference performance by selecting appropriate tree dimensions based on available computing resources. OPT-Tree [76] searches for optimal tree structures to maximize expected acceptance length per decoding step. DSBD [77] uses a small model to generate multiple candidate sequences via beam search, then the large model verifies these sequences layer by layer while dynamically adjusting the beam width based on acceptance probabilities to balance efficiency and quality. DySpec [81] enables dynamic tree expansion during runtime based on prediction confidence, while EAGLE-2 [74] incorporates context-aware tree construction to improve acceptance rates. DDD [110] optimizes EAGLE-2 [74]'s tree drafting method by making the depth dynamic based on draft model confidence.

Several works have explored hybrid approaches that combine tree-based verification with other techniques. ProPD [75] integrates progressive refinement into the tree structure, while RSD [79] employs recursive verification strategies. GSD [78] and ADED [36] extend tree-based methods to handle more complex dependency structures through graph-based representations and adaptive depth adjustment.

In terms of verifying multiple candidate draft tokens in parallel (Multi-Draft Speculative Decoding, MDSD), [84] propose a hybrid sampling strategy that combines deterministic selection of high-probability tokens with random sampling of the final token, improving acceptance rates in certain scenarios. [83] introduce a two-phase verification method that uses importance sampling to select a draft token before applying single-draft verification, optimizing the process for parallel draft generation.

### B. Iterative Decoding

Iterative decoding methods extend the single-pass verification paradigm by allowing multiple refinement iterations on draft tokens until convergence. These approaches draw inspiration from classical numerical methods for solving systems of nonlinear equations, particularly the Jacobi and Gauss-Seidel iteration methods.

In [28], the authors reframe AR text generation as an iterative optimization problem. Their approach expresses token generation as a system where each position must output the most likely token given the current state of all other positions. Starting with a randomly initialized sequence, they adapt the Jacobi method to update all positions in parallel during each iteration until convergence. The authors prove that this process produces identical output to traditional AR decoding under greedy sampling. [29] builds upon this framework with LOOKAHEAD decoding, which combines Jacobi iterations with n-gram verification to accelerate convergence by leveraging predictions from earlier steps.

CLLMs [30] leverages consistency training to accelerate convergence by enabling better multi-token prediction in early iterations.

---

## VI. System-Level Optimizations and Implementation Strategies

### A. Parallel Speculative Decoding

Traditional SD processes tokens sequentially, with the draft model generating tokens followed by target model verification, creating inherent bottlenecks. As shown in Figure 5, parallel approaches overcome this limitation by enabling simultaneous operation — while the target model verifies earlier tokens, the draft model generates subsequent ones, enabling continuous overlapped execution.

Recent methods build upon this paradigm: CS Drafting [85] employs vertical and horizontal cascade structures for 81% speedup, PaSS [87] uses lookahead embeddings for 30% speedup, and Faster Cascades [88] incorporates deferral rules for improved cost-quality tradeoffs. PEARL [89] further advances this through pre-verify and post-verify strategies with adaptive draft lengths, achieving 4.43× speedup over AR decoding and 1.50× over standard SD. AMUSD [111] presents an asynchronous multi-device approach to SD, decoupling the draft and verify phases into continuous, asynchronous operations.

**Figure 5:** Comparison of speculative decoding approaches: (a) Sequential processing where draft generates tokens (0–3) before target verification. (b) Parallel processing where draft generates new tokens while target simultaneously verifies previous ones.

### B. Distributed Speculative Decoding

Edge computing environments impose stringent constraints on memory, compute power, and latency, necessitating specialized SD approaches to deploy LLMs effectively in resource-constrained settings. SpecExec [13] is designed to harness the parallel processing power of consumer GPUs to accelerate LLM inference. By generating multiple tokens per target model iteration and constructing a "cache" tree of probable continuations, SpecExec efficiently validates these continuations with the target model in a single pass. EdgeLLM [14] further optimizes on-device LLM inference through novel techniques for resource allocation and error correction, achieving great token generation speeds and significantly outperforming existing engines. Dovetail [93] represents a significant advancement in heterogeneous computing for LLM inference. By deploying the draft model on the GPU and the target model on the CPU, Dovetail reduces the granularity of data transfer and enhances the overall inference process. The introduction of Dynamic Gating Fusion (DGF) and optimizations for low-end hardware further improve the balance between latency and performance.

**Figure 6:** Asynchronous and heterogeneous schedules: (a) Asynchronous Schedule — the draft stage is followed by the verify stage, with optional stop signals determining further processing. (b) Heterogeneous Schedule — both CPU and GPU devices are utilized for different stages of the decoding process, enabling parallel processing and optimizing performance through resource allocation.

### C. Compiler and Hardware Optimization for Speculative Decoding

Efficient implementation of SD requires careful optimization of both hardware resources and compiler strategies to maximize throughput and minimize latency. SpecPIM [94] presents a novel approach to accelerate speculative inference on a Processing-in-Memory (PIM) system through co-exploration of architecture and dataflow. This method constructs a design space that comprehensively considers algorithmic and architectural heterogeneity, enabling optimal hardware resource allocation for different models and computational patterns. [101] investigates improvements in speculative sampling on GPUs, achieving significant speed gains by parallelizing computations and using sigmoid approximations for softmax, though this comes with a minor reduction in accuracy.

Recent studies have focused on enhancing throughput using SD by optimizing batch processing and scheduling strategies. Using Markov chain theory, [69] establishes SD's optimality among unbiased algorithms while highlighting the tradeoff between inference speed and output quality. Their analysis reveals that batch processing benefits are limited by the distribution gap between small and large models. MagicDec [95] identifies the shift from compute-bound to memory-bound bottlenecks as batch size and sequence length increase, using sparse KV caches in draft models to optimize throughput. BASS [96] extends SD to a batched setting with customized CUDA kernels for ragged tensors in attention calculations and dynamically adjusts draft lengths for better GPU utilization. SEED [97] accelerates reasoning tree construction through scheduled speculative execution, using a rounds-scheduled strategy for conflict-free parallel processing. PipeInfer [98] addresses single-request latency through pipelined speculative acceleration, reducing inter-token latency via asynchronous speculation and early cancellation. TRIFORCE [112] introduces a hierarchical SD mechanism with a dynamic sparse KV cache to achieve lossless acceleration of long sequence generation, significantly improving generation speed and efficiency while maintaining quality. [113] proposes QSPEC, a novel framework that combines weight-shared quantization schemes with SD, achieving up to 1.55× acceleration without quality loss. [99] introduces a hardware-aware SD algorithm that accelerates the inference speed of Mamba and hybrid models. Inspired by SD, SKD [100] represents a novel adaptive approach to knowledge distillation — by dynamically generating tokens and using the teacher model to filter or replace low-quality samples, it bridges the gap between supervised KD's reliance on static data and on-policy KD's susceptibility to low-quality outputs.

---

## VII. Multimodal Models and Applications

### A. Speculative Decoding for Visual Output Generation

Researchers are now using SD to improve the efficiency of AR image generation [114], [115], [116]. As shown in Figure 7, this method greatly speeds up the process by reducing the inference steps needed for generating visual tokens. [15] proposes a novel continuous SD method that designs a new acceptance criterion for the diffusion distributions, significantly improving the efficiency of AR image generation. Similarly, LANTERN [16] presents a relaxed acceptance condition for the SD strategy to substantially speed up the inference process in visual AR models. Additionally, Speculative Jacobi Decoding (SJD) [102] offers a training-free speculative Jacobi decoding technique that effectively accelerates text-to-image generation tasks.

**Figure 7:** Flow of AR image generation with SD.

### B. Speculative Decoding for Multimodal Output Generation

Recent advancements in SD have substantially improved the efficiency and quality of AR generation across various modalities. In the domain of speech synthesis, VADUSA [17] leverages SD to accelerate the inference process in AR text-to-speech (TTS) systems, enhancing the quality of speech synthesis as well. Inspired by SD, [18] introduces a multi-token prediction mechanism, offering substantial improvements in inference efficiency for speech generation.

In the context of multimodal large language models, [103] investigates the integration of SD into the LLaVA 7B model to optimize inference efficiency. Their findings indicate that employing a lightweight, language-only draft model facilitates a memory-constrained acceleration of up to 2.37×. IbED [104] proposes the "In-batch Ensemble Drafting" method to further enhance the robustness and efficiency of SD. It adopts ensemble techniques during batch-level inference, requires no additional model parameters, and significantly increases the validation probability of draft tokens, thereby improving performance and robustness across diverse input scenarios.

### C. Recommendation Systems

LLM-based recommendation systems have shown great potential in enhancing personalized recommendations, but their high inference latency poses a significant challenge for real-world deployment. To address this, recent research has focused on optimizing decoding efficiency to accelerate recommendation generation. DARE [105] integrates retrieval-based SD to accelerate recommendation knowledge generation, thereby improving the deployment efficiency of LLM-based recommender systems in industrial settings. AtSpeed [106] combines strict top-K alignment (AtSpeed-S) and relaxed sampling verification (AtSpeed-R), to significantly accelerate LLM-based generative recommendation with speedup from 2× to 2.5×, addressing inference latency challenges in top-K sequence generation.

---

## VIII. Conclusion

This survey analyzes generation-refinement frameworks for mitigating sequential dependencies in autoregressive models, highlighting how these approaches are fundamentally changing efficient neural sequence generation across text, speech, and visual domains. Through examining both algorithmic innovations and system-level implementations, we have demonstrated their broad applicability while providing crucial deployment insights for practitioners. Moving forward, significant challenges persist in constructing solid theoretical foundations to grasp the balance between parallelism and quality, as well as in developing comprehensive approaches that span different modalities — efforts that could narrow the divide between the capabilities of large models and their actual implementation. Additionally, it remains crucial to examine the scalability of the speculative decoding system as the quantity of draft and target models increases.

---

## References

- [1] T. Brown et al., "Language models are few-shot learners," *Advances in Neural Information Processing Systems*, vol. 33, pp. 1877–1901, 2020.

- [2] Y. Zhuang et al., "ToolQA: A dataset for LLM question answering with external tools," *arXiv:2306.13304*, 2023.

- [3] H. Touvron et al., "LLaMA: Open and efficient foundation language models," *arXiv:2302.13971*, 2023.

- [4] W. Zhu et al., "Multilingual machine translation with large language models: Empirical results and analysis," *arXiv:2304.04675*, 2023.

- [5] M. U. Hadi et al., "A survey on large language models: Applications, challenges, limitations, and practical usage," *TechRxiv*, 2023.

- [6] H. Huang et al., "Towards making the most of LLM for translation quality estimation," in *CCF Int. Conf. on NLP and Chinese Computing*, Springer, 2023, pp. 375–386.

- [7] J. Ho, A. Jain, and P. Abbeel, "Denoising diffusion probabilistic models," *NeurIPS*, vol. 33, pp. 6840–6851, 2020.

- [8] L. Yang et al., "Diffusion models: A comprehensive survey of methods and applications," *ACM Computing Surveys*, vol. 56, no. 4, pp. 1–39, 2023.

- [9] K. Tian et al., "Visual autoregressive modeling: Scalable image generation via next-scale prediction," *arXiv:2404.02905*, 2024.

- [10] N. Ding et al., "Sparse low-rank adaptation of pre-trained language models," *arXiv:2311.11696*, 2023.

- [11] J. Z. Wu et al., "Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation," in *ICCV 2023*, pp. 7623–7633.

- [12] "Open-Sora Report v1.1," https://github.com/hpcaitech/Open-Sora, 2024.

- [13] R. Svirschevski et al., "SpecExec: Massively parallel speculative decoding for interactive LLM inference on consumer devices," *arXiv:2406.02532*, 2024.

- [14] D. Xu et al., "EdgeLLM: Fast on-device LLM inference with speculative decoding," *IEEE Transactions on Mobile Computing*, 2024.

- [15] Z. Wang et al., "Continuous speculative decoding for autoregressive image generation," *arXiv:2411.11925*, 2024.

- [16] D. Jang et al., "LANTERN: Accelerating visual autoregressive models with relaxed speculative decoding," *arXiv:2410.03355*, 2024.

- [17] B. Li et al., "Fast and high-quality auto-regressive speech synthesis via speculative decoding," *arXiv:2410.21951*, 2024.

- [18] D. Raj et al., "Faster speech-LLaMA inference with multi-token prediction," *arXiv:2409.08148*, 2024.

- [19] H. Touvron et al., "LLaMA: Open and efficient foundation language models," *arXiv:2302.13971*, 2023.

- [20] H. Touvron et al., "LLaMA 2: Open foundation and fine-tuned chat models," *arXiv:2307.09288*, 2023.

- [21] A. Dubey et al., "The LLaMA 3 herd of models," *arXiv:2407.21783*, 2024.

- [22] A. Radford et al., "Language models are unsupervised multitask learners," *OpenAI Blog*, vol. 1, no. 8, p. 9, 2019.

- [23] A. Radford, "Improving language understanding by generative pre-training," 2018.

- [24] H. Liu et al., "Pay attention to MLPs," *NeurIPS*, vol. 34, pp. 9204–9215, 2021.

- [25] D. Hendrycks and K. Gimpel, "Gaussian error linear units (GELUs)," *arXiv:1606.08415*, 2016.

- [26] S. Elfwing, E. Uchibe, and K. Doya, "Sigmoid-weighted linear units for neural network function approximation in reinforcement learning," *Neural Networks*, vol. 107, pp. 3–11, 2018.

- [27] M. Stern, N. Shazeer, and J. Uszkoreit, "Blockwise parallel decoding for deep autoregressive models," *NeurIPS*, vol. 31, 2018.

- [28] A. Santilli et al., "Accelerating transformer inference for translation via parallel decoding," *arXiv:2305.10427*, 2023.

- [29] Y. Fu et al., "Break the sequential dependency of LLM inference using lookahead decoding," *arXiv:2402.02057*, 2024.

- [30] S. Kou et al., "CLLMs: Consistency large language models," *arXiv:2403.00835*, 2024.

- [31] N. Yang et al., "Inference with reference: Lossless acceleration of large language models," *arXiv:2304.04487*, 2023.

- [32] Z. He et al., "REST: Retrieval-based speculative decoding," *arXiv:2311.08252*, 2023.

- [33] Z. Wang et al., "Speculative RAG: Enhancing retrieval augmented generation through drafting," *arXiv:2407.08223*, 2024.

- [34] J. Ou, Y. Chen, and W. Tian, "Lossless acceleration of large language model via adaptive N-gram parallel decoding," *arXiv:2404.08698*, 2024.

- [35] L. Stewart et al., "The N-Grammys: Accelerating autoregressive inference with learning-free batched speculation," *arXiv:2411.03786*, 2024.

- [36] X. Liu et al., "Adaptive draft-verification for efficient large language model decoding," *arXiv:2407.12021*, 2024.

- [37] H. Xia et al., "Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation," in *Findings of ACL: EMNLP 2023*, pp. 3909–3925.

- [38] K. Huang, X. Guo, and M. Wang, "SpecDec++: Boosting speculative decoding via adaptive candidate lengths," *arXiv:2405.19715*, 2024.

- [39] S. Kim et al., "Speculative decoding with big little decoder," *NeurIPS*, vol. 36, 2024.

- [40] J. Liu, B. Park, and X. Shen, "A drop-in solution for on-the-fly adaptation of speculative decoding in large language models," *OpenReview*, 2025.

- [41] X. Liu et al., "Online speculative decoding," *arXiv:2310.07177*, 2023.

- [42] Y. Zhou et al., "DistillSpec: Improving speculative decoding via knowledge distillation," *arXiv:2310.08461*, 2023.

- [43] O. Zafrir et al., "FastDraft: How to train your draft," *arXiv:2411.11055*, 2024.

- [44] G. Bachmann et al., "Judge decoding: Faster speculative sampling requires going beyond model alignment," *ICLR 2025*.

- [45] G. Liu et al., "Knowledge distillation with training wheels," Amazon Science, 2025.

- [46] C. Hooper et al., "SPEED: Speculative pipelined execution for efficient decoding," *arXiv:2310.12072*, 2023.

- [47] S. Bae et al., "Fast and robust early-exiting framework for autoregressive language models with synchronized parallel decoding," *arXiv:2310.05424*, 2023.

- [48] J. Zhang et al., "Draft & Verify: Lossless large language model acceleration via self-speculative decoding," *arXiv:2309.08168*, 2023.

- [49] M. Elhoushi et al., "LayerSkip: Enabling early exit inference and self-speculative decoding," *arXiv:2404.16710*, 2024.

- [50] F. Liu et al., "Kangaroo: Lossless self-speculative decoding via double early exiting," *arXiv:2404.18911*, 2024.

- [51] J. Liu et al., "Speculative decoding via early-exiting for faster LLM inference with Thompson sampling control mechanism," *arXiv:2406.03853*, 2024.

- [52] H. Xia et al., "SWIFT: On-the-fly self-speculative decoding for LLM inference acceleration," *arXiv:2410.06916*, 2024.

- [53] N. Bhendawade et al., "Speculative streaming: Fast LLM inference without auxiliary models," *arXiv:2402.11131*, 2024.

- [54] M. R. Metel et al., "Draft on the fly: Adaptive self-speculative decoding using cosine similarity," *arXiv:2410.01028*, 2024.

- [55] Y. Li et al., "EAGLE: Speculative sampling requires rethinking feature uncertainty," *arXiv:2401.15077*, 2024.

- [56] X. Gao et al., "Falcon: Faster and parallel inference of large language models through enhanced semi-autoregressive drafting and custom-designed decoding tree," *arXiv:2412.12639*, 2024.

- [57] L. Zhang et al., "Learning harmonized representations for speculative sampling," *arXiv:2408.15766*, 2024.

- [58] Z. Ankner et al., "Hydra: Sequentially-dependent draft heads for Medusa decoding," *arXiv:2402.05109*, 2024.

- [59] M. Zimmer et al., "Mixture of attentions for speculative decoding," *arXiv:2410.03804*, 2024.

- [60] T. Kim et al., "Accelerating blockwise parallel language models with draft refinement," *NeurIPS 2024*.

- [61] T. Cai et al., "Medusa: Simple LLM inference acceleration framework with multiple decoding heads," *arXiv:2401.10774*, 2024.

- [62] F. Gloeckle et al., "Better & faster large language models via multi-token prediction," *arXiv:2404.19737*, 2024.

- [63] Z. Li et al., "Amphista: Accelerate LLM inference with bi-directional multiple drafting heads in a non-autoregressive style," *arXiv:2406.13170*, 2024.

- [64] Z. Wen, S. Gui, and Y. Feng, "Speculative decoding with CTC-based draft model for LLM inference acceleration," *arXiv:2412.00061*, 2024.

- [65] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in *ICML 2023*, pp. 19274–19286.

- [66] C. Chen et al., "Accelerating large language model decoding with speculative sampling," *arXiv:2302.01318*, 2023.

- [67] Z. Sun et al., "Block verification accelerates speculative decoding," *ICLR 2025*.

- [68] Z. Qin et al., "Optimized multi-token joint decoding with auxiliary model for LLM inference," *arXiv:2407.09722*, 2024.

- [69] M. Yin et al., "A theoretical perspective for speculative decoding algorithm," *arXiv:2411.00841*, 2024.

- [70] Z. Sun et al., "SpecTr: Fast speculative decoding via optimal transport," *NeurIPS*, vol. 36, 2024.

- [71] X. Miao et al., "SpecInfer: Accelerating generative large language model serving with tree-based speculative inference and verification," *arXiv:2305.09781*, 2023.

- [72] B. Spector and C. Re, "Accelerating LLM inference with staged speculative decoding," *arXiv:2308.04623*, 2023.

- [73] Z. Chen et al., "Sequoia: Scalable, robust, and hardware-aware speculative decoding," *arXiv:2402.12374*, 2024.

- [74] Y. Li et al., "EAGLE-2: Faster inference of language models with dynamic draft trees," *arXiv:2406.16858*, 2024.

- [75] S. Zhong et al., "ProPD: Dynamic token tree pruning and generation for LLM parallel decoding," *arXiv:2402.13485*, 2024.

- [76] J. Wang et al., "OPT-Tree: Speculative decoding with adaptive draft tree structure," *arXiv:2406.17276*, 2024.

- [77] Z. Qin et al., "Dynamic-width speculative beam decoding for efficient LLM inference," *arXiv:2409.16560*, 2024.

- [78] Z. Gong et al., "Graph-structured speculative decoding," *arXiv:2407.16207*, 2024.

- [79] W. Jeon et al., "Recursive speculative decoding: Accelerating LLM inference via sampling without replacement," *arXiv:2402.14160*, 2024.

- [80] Y. Cheng et al., "Recurrent drafter for fast speculative decoding in large language models," *arXiv:2403.09919*, 2024.

- [81] Y. Xiong et al., "DySpec: Faster speculative decoding with dynamic token tree structure," *arXiv:2410.11744*, 2024.

- [82] R. Sun et al., "SpecHub: Provable acceleration to multi-draft speculative decoding," *arXiv:2411.05289*, 2024.

- [83] A. Khisti et al., "Multi-draft speculative sampling: Canonical architectures and theoretical limits," *arXiv:2410.18234*, 2024.

- [84] Z. Hu et al., "Towards optimal multi-draft speculative decoding," *ICLR 2025*.

- [85] Z. Chen et al., "Cascade speculative drafting for even faster LLM inference," *arXiv:2312.11462*, 2023.

- [86] S. Yang et al., "Predictive pipelined decoding: A compute-latency trade-off for exact LLM decoding," *arXiv:2307.05908*, 2023.

- [87] G. Monea, A. Joulin, and E. Grave, "PASS: Parallel speculative sampling," *arXiv:2311.13581*, 2023.

- [88] H. Narasimhan et al., "Faster cascades via speculative decoding," *arXiv:2405.19261*, 2024.

- [89] T. Liu et al., "Parallel speculative decoding with adaptive draft length," *arXiv:2408.11850*, 2024.

- [90] W. Zhao et al., "Ouroboros: Generating longer drafts phrase by phrase for faster speculative decoding," in *EMNLP 2024*, pp. 13378–13393.

- [91] Z. Xiao et al., "ParallelSpec: Parallel drafter for efficient speculative decoding," *arXiv:2410.05589*, 2024.

- [92] H. Yi et al., "Generation meets verification: Accelerating large language model inference with smart parallel auto-correct decoding," in *Findings of ACL 2024*, pp. 5285–5299.

- [93] L. Zhang et al., "Dovetail: A CPU/GPU heterogeneous speculative decoding for LLM inference," *arXiv:2412.18934*, 2024.

- [94] C. Li et al., "SpecPIM: Accelerating speculative inference on PIM-enabled system via architecture-dataflow co-exploration," in *ASPLOS 2024*, pp. 950–965.

- [95] J. Chen et al., "MagicDec: Breaking the latency-throughput tradeoff for long context generation with speculative decoding," *arXiv:2408.11049*, 2024.

- [96] H. Qian et al., "BASS: Batched attention-optimized speculative sampling," *arXiv:2404.15778*, 2024.

- [97] Z. Wang et al., "SEED: Accelerating reasoning tree construction via scheduled speculative decoding," *arXiv:2406.18200*, 2024.

- [98] B. Butler et al., "PipeInfer: Accelerating LLM inference using asynchronous pipelined speculation," in *SC24*, IEEE, 2024, pp. 1–19.

- [99] J. Wang et al., "The Mamba in the Llama: Distilling and accelerating hybrid models," *arXiv:2408.15237*, 2024.

- [100] W. Xu et al., "Speculative knowledge distillation: Bridging the teacher-student gap through interleaved sampling," *arXiv:2410.11325*, 2024.

- [101] D. Wagner et al., "Optimized speculative sampling for GPU hardware accelerators," *arXiv:2406.11016*, 2024.

- [102] Y. Teng et al., "Accelerating auto-regressive text-to-image generation with training-free speculative Jacobi decoding," *arXiv:2410.01699*, 2024.

- [103] M. Gagrani et al., "On speculative decoding for multimodal large language models," *arXiv:2404.08856*, 2024.

- [104] M. Lee et al., "In-batch ensemble drafting: Toward fast and robust speculative decoding for multimodal language models."

- [105] Y. Xi et al., "A decoding acceleration framework for industrial deployable LLM-based recommender systems," *arXiv:2408.05676*, 2024.

- [106] X. Lin et al., "Efficient inference for large language model-based generative recommendation," *arXiv:2410.05165*, 2024.

- [107] S. Teerapittayanon, B. McDanel, and H. Kung, "BranchyNet: Fast inference via early exiting from deep neural networks," in *ICPR 2016*, IEEE, pp. 2464–2469.

- [108] A. Slivkins et al., "Introduction to multi-armed bandits," *Foundations and Trends® in Machine Learning*, vol. 12, no. 1–2, pp. 1–286, 2019.

- [109] A. Liu et al., "DeepSeek-V3 technical report," *arXiv:2412.19437*, 2024.

- [110] O. Brown et al., "Dynamic depth decoding: Faster speculative decoding for LLMs," *arXiv:2409.00142*, 2024.

- [111] B. McDanel, "AMUSD: Asynchronous multi-device speculative decoding for LLM acceleration," *arXiv:2410.17375*, 2024.

- [112] H. Sun et al., "TRIFORCE: Lossless acceleration of long sequence generation with hierarchical speculative decoding," *arXiv:2404.11912*, 2024.

- [113] J. Zhao et al., "QSPEC: Speculative decoding with complementary quantization schemes," *arXiv:2410.11305*, 2024.

- [114] M. Ding et al., "CogView: Mastering text-to-image generation via transformers," *NeurIPS*, vol. 34, pp. 19822–19835, 2021.

- [115] J. Yu et al., "Scaling autoregressive models for content-rich text-to-image generation," *arXiv:2206.10789*, 2022.

- [116] T. Li et al., "Autoregressive image generation without vector quantization," *arXiv:2406.11838*, 2024.
