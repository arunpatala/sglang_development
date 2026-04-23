# LLM Inference Performance Estimator — Interactive Roofline Calculator

**Source:** https://joursbleu.github.io/llm-perf-model/
**Author:** joursbleu (GitHub)
**Type:** Interactive web tool (no article to read; run it in a browser)
**Level:** L1 (interactive tool) — Roofline analysis; prefill/decode estimator; multi-GPU scaling
**Why here:** The most concrete way to internalise the compute-bound vs memory-bound distinction. Enter your exact model and GPU configuration and see prefill latency, decode throughput, and a roofline plot showing where each phase sits on the arithmetic intensity axis. Makes the motivation for disaggregation immediately concrete: you can see the 5× arithmetic intensity drop between prefill and decode for any model you care about, with exact numbers for your hardware.

**Usage:** Open https://joursbleu.github.io/llm-perf-model/ in a browser. No login or installation needed.

---

## What the Tool Does

The estimator computes prefill latency, decode throughput, TTFT, and memory usage for LLMs on various GPUs using **roofline analysis** — the same theoretical framework used in DistServe, Splitwise, and the LLM Inference Unveiled survey (L3/04).

---

## Inputs

### Model Configuration
- **Model family**: LLaMA, Mistral, DeepSeek, Qwen, and others
- **Model size**: 7B, 13B, 34B, 70B, 405B, etc.

### Device Configuration
- **GPU**: H100 SXM, H100 PCIe, A100 SXM, A100 PCIe, A6000, RTX 4090, L40S, H200, B200, etc.
- **FLOPS utilisation %**: adjustable to account for real-world overhead
- **Memory utilisation %**: adjustable for realistic BW utilisation
- **Network BW utilisation %**: for multi-GPU configurations

### Quantization
- **Model weights**: FP16/BF16, INT8/FP8, INT4/GPTQ/AWQ, 3-bit (GGUF Q3), 2-bit (GGUF Q2)
- **KV Cache precision**: FP16, INT8/FP8, INT4, INT2

### Runtime Configuration
- **Prompt Length (prefill)**: number of input tokens (default 512)
- **Output Length (decode)**: number of output tokens (default 256)
- **Batch Size**: number of concurrent requests
- **Tensor Parallel (GPUs)**: TP degree (1, 2, 4, 8)
- **FlashAttention**: toggle IO-aware tiling on/off

---

## Outputs

| Output | What it shows |
|---|---|
| **Prefill Latency** | TTFT estimate in milliseconds |
| **Decode Speed** | Tokens/second at the given batch size |
| **Total Time** | Prefill + Decode combined |
| **Model Memory** | Weights + KV Cache per request in GB |

---

## Key Displays

### Roofline Plot
Shows the arithmetic intensity axis (FLOPs/byte) with:
- **Memory bandwidth ceiling**: diagonal line from origin (slope = peak memory bandwidth)
- **Compute ceiling**: horizontal line at peak TFLOPS
- **Prefill operation point**: high on the arithmetic intensity axis (compute-bound)
- **Decode operation point**: low on the axis, below the memory bandwidth ceiling (memory-bound)

**This is the most direct visual demonstration of why disaggregation makes sense**: prefill and decode sit in fundamentally different regions of the roofline, requiring fundamentally different hardware profiles.

### Formulas Used

```
Prefill (Compute-Bound):
FLOPs ≈ 2 × Params × SeqLen + Attention O(n²)
Prefill Time = (Linear FLOPs + Attn FLOPs) / (Effective TFLOPS × 10¹²)
Note: For MoE models, only active parameters participate.

Decode (Memory-Bandwidth-Bound):
Each token reads all weights + KV cache from VRAM.
Time = max(compute, memory)
Decode Time/Token = Model Size (bytes) / (Memory BW × TP × BW Utilization)

KV Cache Size:
2 × layers × kv_heads × head_dim × seq_len × bytes_per_element
Note: GQA/MLA significantly reduces KV Cache vs MHA.

Arithmetic Intensity:
Prefill AI ≈ SeqLen (increases with prompt length → more compute-bound for longer prompts)
Decode AI ≈ 1 FLOP/byte (constant — always memory-bound regardless of sequence length)

FlashAttention Effect:
Without FlashAttention: O(N²) HBM traffic + lower utilisation (~40%)
With FlashAttention: IO-aware tiling keeps N×N scores in SRAM → higher effective utilisation
```

### Additional Views

- **Per-Op Layer Breakdown**: detailed breakdown of FLOPs and memory per layer (link to separate page)
- **Multi-Device Comparison**: compares your current model+settings across all supported GPUs that have enough VRAM
- **Multi-GPU Scaling**: tensor parallel performance scaling with interconnect-aware communication modelling

---

## Recommended Usage for Layer 19

**Before reading DistServe (L3/01):**
1. Open the tool.
2. Select LLaMA-3.1-70B + H100 SXM.
3. Set prompt length = 4096, batch size = 1, TP = 1.
4. Observe: prefill latency (high, compute-bound) vs decode speed (low, memory-bound).
5. Look at the roofline plot — see where prefill and decode fall.
6. Change batch size from 1 to 64 → decode throughput increases significantly (amortises memory reads), prefill latency increases proportionally.

**To understand KV transfer sizing:**
1. Read "Model Memory" output (KV Cache per request).
2. This is exactly the number that must transfer over the network between prefill and decode workers.
3. Compare with the network bandwidth table in L1/02 to see if your interconnect is fast enough.

**To evaluate hardware choices:**
1. Use "Multi-Device Comparison" to compare H100 SXM vs A100 SXM vs H200 for your specific model.
2. Notice that H200 (larger HBM) shows higher decode throughput — confirms the hardware heterogeneity insight from Splitwise (L3/02).

---

## Key Takeaways for Layer 19

- `Decode AI ≈ 1 FLOP/byte` is the fundamental formula explaining why decode is always memory-bound and why FLOPS don't predict decode speed.
- `Prefill AI ≈ SeqLen` explains why longer prompts are more compute-bound — and why very long prompts benefit more from a dedicated prefill pool.
- The roofline plot makes the two-phase asymmetry visual in a way that text descriptions cannot — run this tool before reading DistServe.
- The KV Cache formula (`2 × layers × kv_heads × head_dim × seq_len × bytes`) in the tool's methodology section is the same formula used in L1/01 and L1/02 — seeing it implemented in a live calculator builds intuition.
- FlashAttention's effect (switching from 40% to higher GPU utilisation during prefill) is visible in the tool — toggle it on/off to see the impact.
