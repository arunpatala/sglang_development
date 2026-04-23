# L3 References — Multi-LoRA Inference: Foundational Papers

Three foundational papers that underpin all production multi-LoRA serving systems.

## Reading order

1. **`01_lora_original.md`** — Hu et al. (2021). The LoRA paper itself: rank decomposition math, scaling `lora_alpha/r`, weight initialization, why `q_proj` and `v_proj` are typically targeted. Essential reading before everything else.
2. **`02_punica_mlsys24.md`** — Chen et al. (MLSys 2024). Invented the SGMV kernel for batched multi-LoRA computation. Defines the 12× throughput gain benchmark. Powers LoRAX, SGLang, and vLLM's LoRA backends.
3. **`03_slora_mlsys24.md`** — Sheng et al. (MLSys 2024). Unified Paging memory pool for thousands of concurrent adapters. Tensor parallelism strategy for distributed LoRA. Foundation for SGLang's `--max-loras-per-batch` and `lora_eviction_policy`.
