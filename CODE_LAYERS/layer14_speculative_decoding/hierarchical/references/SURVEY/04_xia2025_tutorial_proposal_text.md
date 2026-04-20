# **Tutorial Proposal: Speculative Decoding for Efficient LLM Inference** 

**Heming Xia** _[†]_ **, Cunxiao Du** _[‡]_ **, Yongqi Li** _[†]_ **, Qian Liu** _[‡]_ **, Wenjie Li** _[†] †_ The Hong Kong Polytechnic University, _‡_ Sea AI Lab {he-ming.xia}@connect.polyu.hk; {cnsdunm}@gmail.com 

## **Abstract** 

This tutorial presents a comprehensive introduction to Speculative Decoding (SD), an advanced technique for LLM inference acceleration that has garnered significant research interest in recent years. SD is introduced as an innovative decoding paradigm to mitigate the high inference latency stemming from autoregressive decoding in LLMs. At each decoding step, SD efficiently drafts several future tokens and then verifies them in parallel. This approach, unlike traditional autoregressive decoding, facilitates the simultaneous decoding of multiple tokens per step, thereby achieving promising 2 _×∼_ 4 _×_ speedups in LLM inference while maintaining original distributions. This tutorial delves into the latest techniques in SD, including draft model architectures and verification strategies. Additionally, it explores the acceleration potential and future research directions in this promising field. We aim for this tutorial to elucidate the current research landscape and offer insights for researchers interested in Speculative Decoding, ultimately contributing to more efficient LLM inference. 

## **1 Introduction** 

Large Language Models (LLMs) have achieved significant progress across various domains, e.g., machine translation, fact verification, and conversational systems (OpenAI, 2023; Touvron et al., 2023a,b). However, the token-by-token generation nature necessitated by autoregressive decoding falls short in efficiency, especially as model sizes increase (Pope et al., 2023). The inference latency primarily stems from the memory-bound computation of LLMs (Patterson, 2004; Shazeer, 2019). Each forward pass requires transferring all LLM parameters from High-Bandwidth Memory (HBM) to the on-chip cache of modern accelerators like GPUs. This process, which produces only one token per step, leads to the underutilization of these accelerators and results in inefficiencies. 

To mitigate the high inference latency, Speculative Decoding (SD) has been introduced as an innovative decoding paradigm (Stern et al., 2018; Chen et al., 2023; Xia et al., 2023; Leviathan et al., 2023). The key idea of SD is to exploit the tokenlevel computation parallelism of LLMs by integrating an efficient “draft” model into the inference process. At each decoding step, SD first utilizes this draft model to efficiently predict multiple future tokens and then validates all these tokens with the target LLM in parallel (Xia et al., 2024). Since the draft model requires much fewer computation resources and the cost of parallel verification is nearly equivalent to decoding a single token, SD achieves considerable speedups by substantially reducing overall decoding steps, which diminishes the need for frequent memory operations of LLM parameters (Cai et al., 2024; Li et al., 2024b; Fu et al., 2024; Du et al., 2023). Furthermore, SD has been theoretically proven to maintain identical distributions to the target LLM (Leviathan et al., 2023; Chen et al., 2023), which broadens its application in both academic and industrial communities. 

The primary factors for the acceleration effect of SD are threefold: **1)** the inference efficiency of the introduced draft model, **2)** the acceptance rate of the drafted tokens, and **3)** the average token acceptance length throughout the inference process. To fully exploit the inference parallelism and obtain high speedups, the structure of the draft model, the drafting mechanism, as well as the verification strategy of LLMs play a vital role in SD. In this tutorial, we will present a comprehensive introduction to this innovative decoding paradigm. We will start by providing preliminaries covering the foundational concepts of LLMs (e.g., decoder-only LLMs and autoregressive decoding), and illustrating the memory bottleneck in LLM inference. We will then focus on recent advancements in _drafting and verification strategies_ , _leading algorithms_ , and _applications_ of Speculative Decoding. 

**A taxonomy of methods** We introduce a taxonomy of SD methods based on a variety of dimensions. SD approaches can be categorized by _draft model architectures_ into two main types: 1) independent drafting methods such as leveraging smaller LMs from the same model series (Leviathan et al., 2023; Chen et al., 2023; Miao et al., 2024; Zhou et al., 2024; He et al., 2024) and 2) self-drafting approaches that integrate lightweight draft modules into the target LLM (Cai et al., 2024; Li et al., 2024b; Du et al., 2023; Fu et al., 2024). At the same time, SD methods can be categorized based on _the supported verification strategy_ , such as the greedy decoding (Stern et al., 2018; Xia et al., 2023; Zhang et al., 2023a), speculative sampling (Leviathan et al., 2023; Chen et al., 2023), and token tree verification (Miao et al., 2024; He et al., 2024; Li et al., 2024b,a), an effective approach to increase the token acceptance rate. 

**Cutting-edge algorithms** Then, we offer a thorough discussion and analysis of cutting-edge SD approaches. Specifically, we delve into an indepth analysis of the state-of-the-art methods, Eagle (Li et al., 2024b) and Eagle-2(Li et al., 2024a), which achieve an average 4 _×_ speedup across various generation tasks. This analysis covers their drafting model architectures, drafting mechanisms, and innovative verification strategies. Furthermore, we conduct discussions on other leading methods such as Medusa (Cai et al., 2024), GliDe with a CaPE (Du et al., 2023), and Lookahead Decoding (Fu et al., 2024), offering insights for researchers interested in this promising research area. 

**Evaluations and applications** After discussing the fundamental components and prominent variants of Speculative Decoding, we present a fair comparison of various open-source SD methods using a unified testing environment and evaluation platform. This comparison aims to raise awareness among academics about the practical speedups expected from SD. Beyond general generation tasks, we further introduce SD techniques in downstream applications, such as retrieval-augmented SD (Zhang et al., 2023b; Wang et al., 2024), longcontext SD (Sun et al., 2024), and multimodal SD (Gagrani et al., 2024). Additionally, we discuss SD methods tailored for specific input contexts (Sun et al., 2021; Yang et al., 2023; Saxena, 2023) and advanced SD approaches for mobile phone applications (Xu et al., 2023). 

Finally, we will demonstrate the effectiveness 

of Speculative Decoding through a practical exercise. We conclude this tutorial by summarizing the strengths and challenges of SD and discussing several important future directions. These include: 1) how we can further improve the speedups of SD methods by optimizing the trade-offs between speculation accuracy and drafting efficiency, 2) how to apply Speculative Decoding in batched inference scenarios, and 3) how to integrate Speculative Decoding with other leading techniques. We hope this tutorial serves as an essential guide for newcomers to this field and motivates further research. 

## **2 Target Audience** 

This tutorial will be accessible to anyone with a basic knowledge of machine learning and natural language processing. We believe the topic will be of interest to both NLP researchers and students in academia, as well as NLP practitioners in the industry, particularly those interested in LLM efficiency, sparsity, and computational parallelism. By providing a systematic overview of recent promising approaches for this rapidly evolving field, this tutorial hopefully reveals new research opportunities to the audience. 

## **3 Outline** 

**1. Introduction** (15 minutes) 

   - An overview of the tutorial 

   - The benefits of Speculative Decoding 

**2. Preliminaries** (15 minutes) 

   - Auto-regressive decoding 

   - Memory bottleneck in LLM inference 

**3. Speculative Decoding: A taxonomy of methods** (40 minutes) 

- Definition, formulation, and illustrated algorithms of Speculative Decoding 

- Introduction to various draft model architectures and verification strategies 

## **4. Speculative Decoding: Cutting-edge algorithms** (40 minutes) 

- Discussions and analysis of the state-of-theart algorithms 

- Advanced techniques in Speculative Decoding: adaptive token tree verification, draft model alignment, and etc 

## **5. Speculative Decoding: Metrics and evaluations** (20 minutes) 

- Evaluation metrics of Speculative Decoding 

- Fair comparisons and discussions of leading methods across diverse scenarios 

## **6. Speculative Decoding: Downstream adapta-** 

**tions** (20 minutes) 

- Retrieval-augmented Speculative Decoding 

- Long-context and multimodal Speculative Decoding 

## **7. Demonstration: An exercise to show how Speculative Decoding works** (10 minutes) 

**8. Conclusions and future directions** (10 minutes) 

## **4 Diversity Considerations** 

The speakers are from two academic institutions with an affiliation and an academic research group, including both a professor and Ph.D. students. The methods presented in our tutorials are languageagnostic and can be extended to non-English contexts and we also offer a brief overview of several papers focusing on multilingual and expertdomain extensions of the core frameworks. We will reach out to both academic and industry communities such as CAMEL-AI[1] and DeepSeek[2] , to encourage diverse audience participation in our tutorial. Given that speculative decoding effectively accelerates LLM inference while maintaining unchanged distributions, we anticipate this tutorial will be particularly beneficial for researchers with modest computational resources who may not have access to extensive hardware infrastructure. 

## **5 Other Information** 

**Type of the tutorial** Cutting-edge. 

**Length** This is a 4-hour tutorial. 

**Breath** We estimate that approximately 20% of the tutorial will center around work done by the presenters. The papers we will cover are from both academia and industry. 

**An estimate of the audience size** Given that Speculative Decoding is now applied in a various range of NLP platforms such as vLLM (Kwon et al., 

- 1https://www.camel-ai.org/ 

> 2https://www.deepseek.com/ 

2023), Transformers (Wolf et al., 2020), and PyTorch (Paszke et al., 2019) to accelerate LLM inference, we estimate that the number of audiences will be around 100+. 

**Technical equipment.** We would like to have Internet access to show online demos. 

**Open access** We plan to make all teaching material available online and agree to allow the publication of slides and video recordings in the COLING anthology. 

**Pedagogical material** We plan to do some short hands-on exercises to let the audience try different Speculative Decoding techniques to accelerate LLM inference using Colab. 

## **6 Presenters** 

**Heming Xia** Heming Xia is a Ph.D. student at the Natural Language Processing Group of The Hong Kong Polytechnic University, advised by Prof. Wenjie Li. His research interest lies in natural language processing and machine learning. His recent research focuses on speculative decoding, tool learning, vision-language understanding, and representation learning of LMs. He has served as the reviewer for top-tier conferences like ACL, EMNLP, and Neurips. He was recognized as a Merit Student at Peking University in 2021. 

**Cunxiao Du** Cunxiao Du is a Research Scientist in the Sea AI Lab. He received a PhD degree from The Singapore Management University and a Bachelor’s degree from Shandong University. His research interests include LLM, especially, the efficiency of LLM and parallel decoding of LLM. He has also served as the reviewer for top-tier conferences like ICML, ICLR, and Neurips. 

**Yongqi Li** Yongqi Li is a Postdoctoral Fellow in the Department of Computing at The Hong Kong Polytechnic University. He received a PhD degree from The Hong Kong Polytechnic University and a Bachelor’s degree from Shandong University. His research interests include natural language processing, information retrieval, and multimedia computing. He has published about 20 papers in leading conferences and journals, such as ACL, CVPR, SIGIR, KDD, AAAI, CIKM, TOIS, and IPM. He has also served as an area chair for COLING 2025, a guest editor for Frontiers in Big Data, and a reviewer for TKDE, TOIS, TMM, and TCVST journals. 

**Qian Liu** Qian Liu is a Research Scientist at Sea AI Lab, Singapore. Before he joined Sea AI Lab, he was a joint Ph.D. candidate at Beihang University and Microsoft Research Asia. His research interests include code generation and reasoning. He has published more than 20 papers in top-tier conferences, and his representative works include TAPEX, LoraHub, and StarCoder. He was nominated for the Baidu Scholarship 2020 and the KAUST AI Rising Star 2024. 

**Wenjie Li** Wenjie Li is a Professor in the Department of Computing at the Hong Kong Polytechnic University, Hong Kong. She received a Ph.D. degree in systems engineering and engineering management from the Chinese University of Hong Kong, Hong Kong, in 1997. Her main research interests include natural language understanding and generation, machine conversation, summarization, and question answering. Wenjie served as the program chair for ACL 2021 and (senior) area chair for many *ACL conferences. 

## **7 Reading List** 

- Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding (Xia et al., 2024) 

- Blockwise Parallel Decoding for Deep Autoregressive Models (Stern et al., 2018) 

- Speculative Decoding: Exploiting Speculative Execution for Accelerating Seq2seq Generation (Xia et al., 2023) 

- Fast Inference from Transformers via Speculative Decoding (Leviathan et al., 2023) 

- Accelerating Large Language Model Decoding with Speculative Sampling (Chen et al., 2023) 

- Speculative Decoding with Big Little Decoder (Kim et al., 2023) 

- Inference with Reference: Lossless Acceleration of Large Language Models (Yang et al., 2023) 

- Accelerating Transformer Inference for Translation via Parallel Decoding (Santilli et al., 2023) 

- SpecInfer: Accelerating Generative Large Language Model Serving with Speculative Inference and Token Tree Verification (Miao et al., 2024) 

- Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding (Zhang et al., 2023a) 

- REST: Retrieval-Based Speculative Decoding (He et al., 2024) 

- Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads (Cai et al., 2024) 

- EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (Li et al., 2024b) 

- GliDe with a CaPE: A Low-Hassle Method to Accelerate Speculative Decoding (Du et al., 2023) 

- Break the Sequential Dependency of LLM Inference Using Lookahead Decoding (Fu et al., 2024) 

## **Ethics Statement** 

Speculative Decoding (SD) is an effective LLM inference acceleration technique that maintains unchanged distributions of LLMs. Most advanced SD techniques do not require full-parameter retraining, which makes it more energy efficient and can reduce carbon footprints. Previous research also shows that SD can be integrated with other decoding strategies to improve generation quality further. However, we note that SD may copy relevant text spans from historical contexts or retrieve raw data from a corpus, potentially leaking privacysensitive information, especially when built on a private corpus. Though most SD methods do not alter the output distributions of LLMs, privacy information may leak through other statistical information, such as overall speedups. We acknowledge this to caution those who manage to apply SD techniques in privacy-sensitive domains. 

## **References** 

- Tianle Cai, Yuhong Li, Zhengyang Geng, Hongwu Peng, Jason D. Lee, Deming Chen, and Tri Dao. 2024. Medusa: Simple LLM inference acceleration framework with multiple decoding heads. _CoRR_ , abs/2401.10774. 

- Charlie Chen, Sebastian Borgeaud, Geoffrey Irving, Jean-Baptiste Lespiau, Laurent Sifre, and John Jumper. 2023. Accelerating large language model decoding with speculative sampling. _CoRR_ , abs/2302.01318. 

- Cunxiao Du, Jing Jiang, Yuanchen Xu, Jiawei Wu, Sicheng Yu, Yongqi Li, Shenggui Li, Kai Xu, Liqiang Nie, Zhaopeng Tu, and Yang You. 2023. Glide with a cape: A low-hassle method to accelerate speculative decoding. In _Proceedings of International Conference on Machine Learning, ICML_ , Proceedings of Machine Learning Research. PMLR. 

- Yichao Fu, Peter Bailis, Ion Stoica, and Hao Zhang. 2024. Break the sequential dependency of llm inference using lookahead decoding. _Preprint_ , arXiv:2402.02057. 

- Mukul Gagrani, Raghavv Goel, Wonseok Jeon, Junyoung Park, Mingu Lee, and Christopher Lott. 2024. On speculative decoding for multimodal large language models. _Preprint_ , arXiv:2404.08856. 

- Zhenyu He, Zexuan Zhong, Tianle Cai, Jason Lee, and Di He. 2024. REST: Retrieval-based speculative decoding. In _Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)_ , pages 1582–1595, Mexico City, Mexico. Association for Computational Linguistics. 

- Sehoon Kim, Karttikeya Mangalam, Suhong Moon, Jitendra Malik, Michael W. Mahoney, Amir Gholami, and Kurt Keutzer. 2023. Speculative decoding with big little decoder. In _Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023_ . 

- Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph Gonzalez, Hao Zhang, and Ion Stoica. 2023. Efficient memory management for large language model serving with pagedattention. In _Proceedings of the 29th Symposium on Operating Systems Principles, SOSP 2023, Koblenz, Germany, October 23-26, 2023_ , pages 611– 626. ACM. 

- Yaniv Leviathan, Matan Kalman, and Yossi Matias. 2023. Fast inference from transformers via speculative decoding. In _International Conference on Machine Learning, ICML 2023, 23-29 July 2023, Honolulu, Hawaii, USA_ , volume 202 of _Proceedings of Machine Learning Research_ , pages 19274–19286. PMLR. 

- Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024a. Eagle-2: Faster inference of language models with dynamic draft trees. _Preprint_ , arXiv:2406.16858. 

- Yuhui Li, Fangyun Wei, Chao Zhang, and Hongyang Zhang. 2024b. Eagle: Speculative sampling requires rethinking feature uncertainty. _Preprint_ , arXiv:2401.15077. 

- Xupeng Miao, Gabriele Oliaro, Zhihao Zhang, Xinhao Cheng, Zeyu Wang, Zhengxin Zhang, Rae Ying Yee Wong, Alan Zhu, Lijie Yang, Xiaoxiang Shi, Chunan Shi, Zhuoming Chen, Daiyaan Arfeen, Reyna Abhyankar, and Zhihao Jia. 2024. Specinfer: Accelerating large language model serving with tree-based speculative inference and verification. In _Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3_ , ASPLOS ’24, page 

   - 932–949, New York, NY, USA. Association for Computing Machinery. 

- OpenAI. 2023. GPT-4 technical report. _CoRR_ , abs/2303.08774. 

- Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zachary DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, Junjie Bai, and Soumith Chintala. 2019. Pytorch: An imperative style, high-performance deep learning library. In _Advances in Neural Information Processing Systems 32: Annual Conference on Neural Information Processing Systems 2019, NeurIPS 2019, December 8-14, 2019, Vancouver, BC, Canada_ , pages 8024–8035. 

- David A. Patterson. 2004. Latency lags bandwith. _Commun. ACM_ , 47(10):71–75. 

- Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jeff Dean. 2023. Efficiently scaling transformer inference. In _Proceedings of the Sixth Conference on Machine Learning and Systems, MLSys 2023, Miami, FL, USA, June 4-8, 2023_ . mlsys.org. 

- Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodolà. 2023. Accelerating transformer inference for translation via parallel decoding. In _Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2023, Toronto, Canada, July 9-14, 2023_ , pages 12336–12355. Association for Computational Linguistics. 

Apoorv Saxena. 2023. Prompt lookup decoding. 

- Noam Shazeer. 2019. Fast transformer decoding: One write-head is all you need. _CoRR_ , abs/1911.02150. 

- Mitchell Stern, Noam Shazeer, and Jakob Uszkoreit. 2018. Blockwise parallel decoding for deep autoregressive models. In _Advances in Neural Information Processing Systems 31: Annual Conference on Neural Information Processing Systems 2018, NeurIPS 2018, December 3-8, 2018, Montréal, Canada_ , pages 10107–10116. 

- Hanshi Sun, Zhuoming Chen, Xinyu Yang, Yuandong Tian, and Beidi Chen. 2024. Triforce: Lossless acceleration of long sequence generation with hierarchical speculative decoding. _Preprint_ , arXiv:2404.11912. 

- Xin Sun, Tao Ge, Furu Wei, and Houfeng Wang. 2021. Instantaneous grammatical error correction with shallow aggressive decoding. In _Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, ACL/IJCNLP_ 

_2021, (Volume 1: Long Papers), Virtual Event, August 1-6, 2021_ , pages 5937–5947. Association for Computational Linguistics. 

- Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023a. Llama: Open and efficient foundation language models. _CoRR_ , abs/2302.13971. 

- Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, Dan Bikel, Lukas Blecher, Cristian CantonFerrer, Moya Chen, Guillem Cucurull, David Esiobu, Jude Fernandes, Jeremy Fu, Wenyin Fu, Brian Fuller, Cynthia Gao, Vedanuj Goswami, Naman Goyal, Anthony Hartshorn, Saghar Hosseini, Rui Hou, Hakan Inan, Marcin Kardas, Viktor Kerkez, Madian Khabsa, Isabel Kloumann, Artem Korenev, Punit Singh Koura, Marie-Anne Lachaux, Thibaut Lavril, Jenya Lee, Diana Liskovich, Yinghai Lu, Yuning Mao, Xavier Martinet, Todor Mihaylov, Pushkar Mishra, Igor Molybog, Yixin Nie, Andrew Poulton, Jeremy Reizenstein, Rashi Rungta, Kalyan Saladi, Alan Schelten, Ruan Silva, Eric Michael Smith, Ranjan Subramanian, Xiaoqing Ellen Tan, Binh Tang, Ross Taylor, Adina Williams, Jian Xiang Kuan, Puxin Xu, Zheng Yan, Iliyan Zarov, Yuchen Zhang, Angela Fan, Melanie Kambadur, Sharan Narang, Aurélien Rodriguez, Robert Stojnic, Sergey Edunov, and Thomas Scialom. 2023b. Llama 2: Open foundation and fine-tuned chat models. _CoRR_ , abs/2307.09288. 

   - Daliang Xu, Wangsong Yin, Xin Jin, Ying Zhang, Shiyun Wei, Mengwei Xu, and Xuanzhe Liu. 2023. Llmcad: Fast and scalable on-device large language model inference. _CoRR_ , abs/2309.04255. 

   - Nan Yang, Tao Ge, Liang Wang, Binxing Jiao, Daxin Jiang, Linjun Yang, Rangan Majumder, and Furu Wei. 2023. Inference with reference: Lossless acceleration of large language models. _arXiv preprint arXiv:2304.04487_ . 

   - Jun Zhang, Jue Wang, Huan Li, Lidan Shou, Ke Chen, Gang Chen, and Sharad Mehrotra. 2023a. Draft & verify: Lossless large language model acceleration via self-speculative decoding. _CoRR_ , abs/2309.08168. 

   - Zhihao Zhang, Alan Zhu, Lijie Yang, Yihua Xu, Lanting Li, Phitchaya Mangpo Phothilimthana, and Zhihao Jia. 2023b. Accelerating retrieval-augmented language model serving with speculation. In _Submitted to The Twelfth International Conference on Learning Representations_ . Under review. 

   - Yongchao Zhou, Kaifeng Lyu, Ankit Singh Rawat, Aditya Krishna Menon, Afshin Rostamizadeh, Sanjiv Kumar, Jean-François Kagy, and Rishabh Agarwal. 2024. Distillspec: Improving speculative decoding via knowledge distillation. In _The Twelfth International Conference on Learning Representations_ . 

- Zilong Wang, Zifeng Wang, Long Le, Huaixiu Steven Zheng, Swaroop Mishra, Vincent Perot, Yuwei Zhang, Anush Mattapalli, Ankur Taly, Jingbo Shang, Chen-Yu Lee, and Tomas Pfister. 2024. Speculative rag: Enhancing retrieval augmented generation through drafting. _Preprint_ , arXiv:2407.08223. 

- Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, Joe Davison, Sam Shleifer, Patrick von Platen, Clara Ma, Yacine Jernite, Julien Plu, Canwen Xu, Teven Le Scao, Sylvain Gugger, Mariama Drame, Quentin Lhoest, and Alexander M. Rush. 2020. Huggingface’s transformers: State-of-the-art natural language processing. _Preprint_ , arXiv:1910.03771. 

- Heming Xia, Tao Ge, Peiyi Wang, Si-Qing Chen, Furu Wei, and Zhifang Sui. 2023. Speculative decoding: Exploiting speculative execution for accelerating seq2seq generation. In _Findings of the Association for Computational Linguistics: EMNLP 2023, Singapore, December 6-10, 2023_ , pages 3909–3925. Association for Computational Linguistics. 

- Heming Xia, Zhe Yang, Qingxiu Dong, Peiyi Wang, Yongqi Li, Tao Ge, Tianyu Liu, Wenjie Li, and Zhifang Sui. 2024. Unlocking efficiency in large language model inference: A comprehensive survey of speculative decoding. _Preprint_ , arXiv:2401.07851. 

