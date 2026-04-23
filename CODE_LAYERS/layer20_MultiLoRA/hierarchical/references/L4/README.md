# L4 References — Multi-LoRA Inference: Research Systems

Six research papers advancing beyond S-LoRA and Punica, each addressing a specific production bottleneck.

## Reading order (by theme)

**Dynamic orchestration:**
1. **`01_dlora_osdi24.md`** — dLoRA (OSDI 2024): dynamic merge/unmerge of adapters + request-adapter co-migration between worker replicas. 57.9× throughput over vLLM.

**CPU-assisted loading:**
2. **`02_caraserve_2024.md`** — CaraServe (2024): CPU-assisted prefilling during adapter cold-start loading, rank-aware scheduling for SLO attainment.

**Unified fine-tuning + serving:**
3. **`03_loquetier_neurips25.md`** — Loquetier (NeurIPS 2025): virtualized module design unifying LoRA fine-tuning and inference serving in one runtime.

**Serverless deployment:**
4. **`04_serverlesslora_2025.md`** — ServerlessLoRA (2025): backbone sharing + pre-loading for serverless LoRA inference; 86% TTFT reduction, 89% cost reduction.
5. **`05_predictive_lora_2025.md`** — Predictive-LoRA (2025): LSTM traffic predictor for proactive adapter prefetching; page-based memory management.

**Disaggregated serving:**
6. **`06_infinilora_2026.md`** — InfiniLoRA (2026): decouples LoRA execution from base-model inference via shared LoRA Server; handles MoE's high LoRA memory cost.
