# Inference Optimizations for Large Language Models: Effects, Challenges, and Practical Considerations

**Source:** https://arxiv.org/abs/2408.03130
**Paper PDF:** https://arxiv.org/pdf/2408.03130
**Venue:** arXiv August 2024
**Authors:** Leo Donisch, Sigurd Schacht, Carsten Lanquillon (Karlsruhe University of Applied Sciences)
**Level:** L4 — Broad survey; context for KV quantization within the larger inference optimization landscape
**Why here:** This survey places KV cache quantization in the context of all LLM inference optimizations — quantization, pruning, knowledge distillation, and architectural changes. Its key contribution for Layer 18 is the **distinction between weight quantization (static) and activation quantization (dynamic)** — KV cache quantization is a special case of activation quantization, which is fundamentally harder than weight quantization. This distinction explains why SGLang's approach (calibrated per-tensor scales from checkpoint) is a reasonable compromise, and why vLLM's per-token-head dynamic scales are more principled.

**BibTeX:**
```bibtex
@article{donisch2024llmoptsurvey,
  title={Inference Optimizations for Large Language Models: Effects, Challenges, and Practical Considerations},
  author={Leo Donisch and Sigurd Schacht and Carsten Lanquillon},
  journal={arXiv preprint arXiv:2408.03130},
  year={2024},
  url={https://arxiv.org/abs/2408.03130}
}
```

---

## Taxonomy of LLM Inference Optimizations

```
LLM Inference Optimizations
├── Quantization
│   ├── Weight quantization (static: calibrate once, never changes)
│   │   ├── Post-training (GPTQ, AWQ, GGUF)
│   │   └── Quantization-aware training (QAT)
│   └── Activation quantization (dynamic: changes per input)
│       ├── KV cache quantization ← THIS LAYER
│       ├── Q/K/V projection outputs (SageAttention2)
│       └── Attention weight (softmax output)
│
├── Pruning
│   ├── Structured (remove heads, layers, neurons)
│   └── Unstructured (sparse weights)
│
├── Knowledge Distillation
│   └── Train smaller model to mimic larger
│
└── Architectural Optimizations
    ├── GQA / MQA (reduce KV heads)
    ├── Sparse attention (local, sliding window)
    └── Linear attention (replace softmax)
```

---

## Why KV Cache Quantization Is Harder Than Weight Quantization

This is the most important conceptual point from this survey for Layer 18:

### Weight quantization: static activations

```
Weights W are fixed after training.
→ Calibration: run model once on a representative dataset
→ Compute per-layer scale that minimizes quantization error on calibration data
→ Scale is fixed at inference time — never changes
→ Works well because W has stable statistics
```

### Activation (KV) quantization: dynamic activations

```
KV cache values K, V depend on the input tokens.
→ Different inputs → different K, V distributions
→ A single calibrated scale may not fit all possible inputs
→ Especially problematic for:
   - Out-of-distribution inputs
   - Very long contexts (statistics shift over tokens)
   - Unusual prompts (code, math, foreign languages have different distributions)
```

**Concrete example:** A scale calibrated on English text prompts may be too small for code inputs where K channels have larger dynamic ranges. The model still works but loses precision in K for code — possibly causing incorrect attention to code tokens.

### The three solutions to the dynamic activation problem

| Approach | Who uses it | Tradeoff |
|---|---|---|
| **Calibrated per-tensor scale** (static) | SGLang default | Simple but distribution-dependent; fail on OOD inputs |
| **Dynamic per-token-head scale** | vLLM fp8_per_token_head | More robust; slight memory overhead for scale tensors |
| **Per-channel/per-vector** (calibrated) | KIVI, KVQuant | Best accuracy; complex implementation; calibration dataset needed |

---

## Weight Quantization Methods (context for scale generation)

The survey covers weight quantization methods that are **upstream** of KV quantization:

### GPTQ (Optimal Brain Quantization)
- Layer-by-layer INT4 weight quantization using second-order information
- The `--quantization gptq` flag in SGLang/vLLM
- GPTQ-quantized models can be combined with FP8 KV cache: GPTQ weights + FP8 KV

### AWQ (Activation-Aware Weight Quantization)
- Identifies and protects important weight channels (those with large activation magnitudes)
- AWQ pipelines also collect **KV activation statistics** as a byproduct — this is how the `quantization_param_path` JSON files are generated
- AWQ + FP8 KV is the recommended joint quantization recipe for production

### GGUF / llama.cpp formats
- 2–8-bit weight quantization for CPU inference
- Separate from KV cache quantization (GGUF is for weights; CPU inference has its own KV handling)
- Not directly relevant to SGLang/vLLM GPU serving

---

## The Accuracy-Efficiency Frontier

The survey characterizes quantization methods by two dimensions:

```
Accuracy
  ↑
  │  BF16 weights + BF16 KV ●  (100% accuracy, 100% memory)
  │
  │  BF16 weights + FP8 KV  ●  (~99% accuracy, 50% KV memory)
  │
  │  GPTQ INT4 weights + FP8 KV ●  (~97% accuracy, 25% weight memory + 50% KV memory)
  │
  │  KIVI-2bit KV + INT4 weights ●  (~96% accuracy, 6.25% KV memory)
  │
  └──────────────────────────────────→ Memory efficiency
```

Each quantization decision moves along this frontier. The ideal is the rightmost point with acceptable accuracy for your use case.

---

## Practical Recommendations from the Survey

### When to use FP8 KV cache (Layer 18)

Use when:
- VRAM is the bottleneck (can't fit desired batch size or context length)
- Accuracy difference between FP8 and BF16 is acceptable (typically < 1% on standard benchmarks)
- You have a calibrated checkpoint (models from HuggingFace fp8 repos) or can tolerate scale=1.0

Avoid when:
- Model outputs are extremely sensitive to numerical precision (scientific, medical)
- Input distribution is very different from calibration data (specialized domains)
- Sub-0.1% accuracy is required (use BF16)

### When to go sub-4-bit (KIVI, KVQuant)

Use when:
- 2× memory saving from FP8 isn't enough
- Running very long contexts (1M+ tokens) where BF16 is physically impossible
- Using custom CUDA kernels is acceptable
- Calibration dataset is available

### When to use dynamic per-token-head (vLLM fp8_per_token_head)

Use when:
- Input distribution is diverse (multiple languages, domains, code + text)
- No calibration dataset is available or practical
- vLLM is the engine (SGLang doesn't support this yet)

---

## Key Takeaways for Layer 18

- **KV cache quantization is activation quantization** — harder than weight quantization because activations change per input, weights don't.
- The **three scale strategies** (per-tensor calibrated, per-token-head dynamic, per-channel/vector) represent different accuracy/complexity tradeoffs on the activation quantization hardness spectrum.
- Weight quantization (GPTQ, AWQ) and KV quantization are **independent and composable** — the best production recipe uses both.
- **FP8 is the sweet spot for 2026**: 2× memory reduction with < 1% accuracy impact on calibrated models, production-ready in both SGLang and vLLM.
- Sub-4-bit KV quantization is the research frontier — expect production support in 2026–2027 as hardware and kernel support matures.
