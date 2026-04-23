# L2 References — Multi-LoRA Inference: Production Framework Documentation

Three official framework guides for LoRA serving in production.

## Reading order

1. **`01_sglang_lora_docs.md`** — SGLang's complete LoRA serving guide: server arguments, single/multi-adapter usage, dynamic load/unload API, OpenAI-compatible syntax. Most relevant to the Layer 20 implementation.
2. **`02_vllm_lora_docs.md`** — vLLM's multi-LoRA documentation: offline `LoRARequest` API, serving with `--lora-modules`, dynamic loading via REST endpoints, LoRAResolver plugins, in-place adapter reloading.
3. **`03_hf_peft_lora.md`** — HuggingFace PEFT developer guide for LoRA: configuration parameters, supported layer types, adapter merging/unmerging, the checkpoint format used in `verify_lora.py`.
