# L1 References — Multi-LoRA Inference: Introductions and Blog Posts

Three practitioner-facing articles that build intuition before the papers.

## Reading order

1. **`01_lorax_serve_1000.md`** — 3-min Docker quickstart: load 1 base model, serve 1000 adapters, zero restarts. Best first read.
2. **`02_sagemaker_multi_lora.md`** — 15-min end-to-end tutorial: build, deploy, and benchmark LoRAX on AWS SageMaker with 50 adapters. Includes the SGMV / Punica explanation and a live benchmark showing same throughput for 1 vs 50 adapters.
3. **`03_lorax_towardsai.md`** — 13-min deep dive into LoRAX architecture: the Dedicated Model Problem, three pillars (dynamic loading, tiered caching, continuous multi-adapter batching), SGMV kernel math, structured generation, Lookahead LoRA.
