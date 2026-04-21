# A Survey on Large Language Model Acceleration based on KV Cache Management

**Source:** https://arxiv.org/abs/2412.19442
**Paper PDF:** https://arxiv.org/pdf/2412.19442
**Venue:** arXiv December 2024; revised July 2025
**Authors:** Haoyang Li et al. (TreeAI Lab)
**GitHub (curated list):** https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management
**Level:** L4 — Comprehensive survey; landscape orientation
**Why here:** The definitive 200+ paper survey on KV cache management. The **token-level quantization** section is the most comprehensive summary of KV quantization research available in one place, covering methods from FP8 down to 1-bit. Reading this alongside the specific papers (KIVI, KVQuant, ZipCache) gives the full context: what other approaches exist, how they compare, and what open problems remain.

**BibTeX:**
```bibtex
@article{li2024kvcachesurvey,
  title={A Survey on Large Language Model Acceleration based on KV Cache Management},
  author={Haoyang Li and others},
  journal={arXiv preprint arXiv:2412.19442},
  year={2024},
  url={https://arxiv.org/abs/2412.19442}
}
```

---

## Survey Taxonomy

The survey organizes KV cache management into three levels:

```
KV Cache Management
├── Token-level Optimizations
│   ├── KV cache selection (token dropping: H2O, StreamingLLM, SnapKV)
│   ├── Budget allocation (adaptive budgets: PyramidKV, RazorAttention)
│   ├── KV merging (merge similar KV: KVMerger, CLA)
│   ├── KV quantization ← Layer 18's focus
│   └── Low-rank decomposition (factorize K/V matrices)
│
├── Model-level Optimizations
│   ├── MQA / GQA (shared K/V heads → smaller KV cache)
│   ├── MLA (Multi-head Latent Attention: DeepSeek-style)
│   └── Architectural changes (sliding window, sparse attention)
│
└── System-level Optimizations
    ├── Memory management (PagedAttention, RadixAttention)
    ├── Scheduling (PD disaggregation)
    └── Hardware-aware designs (HiCache, LMCache) ← Layer 17's focus
```

**Layer 18** covers token-level quantization. **Layer 17** covers system-level multi-tier storage. The two combine orthogonally: FP8 KV (Layer 18) × HiCache (Layer 17) = the full production stack.

---

## Token-Level KV Quantization: Survey Section

The survey categorizes KV quantization methods by precision and technique:

### FP8 quantization (8-bit)

| Method | Granularity | Notes |
|---|---|---|
| SGLang `--kv-cache-dtype fp8_e4m3` | Per-tensor | Calibrated scales from checkpoint |
| vLLM `fp8_per_token_head` | Per-(token, head) | Dynamic scales, no calibration |
| SageAttention2 | Per-thread | Compute path (Q×K INT4, P×V FP8) |

**Assessment**: FP8 is production-ready in 2025. The main remaining question is whether to use per-tensor (simple) or per-token-head (more accurate) scales.

### Sub-8-bit quantization (4-bit and below)

| Method | Bits | K strategy | V strategy |
|---|---|---|---|
| KIVI | 2 | Per-channel | Per-token |
| KVQuant | 3 | Per-channel, pre-RoPE | Per-vector, non-uniform |
| ZipCache | Mixed | Channel-separable tokenwise | Salient=FP16, rest=INT2 |
| WKVQuant | 4 | Windowed quantization | Per-token |
| QJL | 4 | Random JL transform | Random JL transform |
| Gear | 4 | Low-rank + sparse outlier | Low-rank + sparse outlier |

**Assessment**: 4-bit is approaching production-readiness for select models. 2–3-bit requires model-specific calibration and custom kernels.

### 1-bit / ultra-low-bit

Research-stage only. Methods: KVSharer (share K/V across layers), MiKV (progressive quantization). Quality degradation significant; not ready for production.

---

## Model-Level Techniques: How They Reduce KV Cache Size

The survey also covers model-architecture changes that directly reduce KV cache requirements:

### Multi-Query Attention (MQA) and Grouped Query Attention (GQA)

Standard MHA: `num_kv_heads = num_query_heads`
MQA: `num_kv_heads = 1` (all heads share one K/V)
GQA: `num_kv_heads = num_query_heads / group_size`

**Memory impact:**
- MHA: full KV per head
- GQA (Llama-3 style, group=8): 8× smaller KV cache
- MQA (Falcon, GPT-NeoX): num_heads× smaller KV cache

GQA already makes FP8 quantization even more impactful: a Llama-3 model with GQA-8 + FP8 KV uses **16× less KV memory** than the equivalent MHA model in BF16.

### Multi-head Latent Attention (MLA)

DeepSeek's MLA compresses KV into a **low-rank latent vector** before storing:
```
Standard: store K, V separately (full head_dim each)
MLA: store compressed latent C (much smaller) + rope_K separately
     at attention time: expand C → K, V using learned projection
```

The latent vector has dimension `kv_lora_rank << head_dim × num_heads`, so the KV cache is **much smaller**. Quantizing the MLA KV cache (SGLang's `MLATokenToKVPoolHost`) is different from MHA: the tensor to quantize is the compressed latent, not the full K/V.

---

## The Interaction Between Quantization and Other Optimizations

### Quantization + token dropping (mixed precision)

ZipCache shows this is possible. The survey identifies this as an **active research area**: should non-salient tokens be quantized (rather than dropped) to enable recovery if needed? This combines the "don't drop anything permanently" property of quantization with the "focus bits on important tokens" property of saliency-based methods.

### Quantization + tiered storage (HiCache)

When KV caches are offloaded to CPU DRAM or disk (Layer 17), quantization reduces:
- **VRAM usage** (fits more in GPU)
- **PCIe bandwidth** (FP8 transfers are 2× smaller than BF16)
- **CPU DRAM usage** (fits 2× more in CPU tier)
- **Storage I/O** (2× smaller writes to and reads from L3)

The κ_crit threshold (from the PCIe bottleneck paper in Layer 17 references) effectively doubles with FP8: for the same PCIe bandwidth, you can offload 2× more tokens before becoming memory-bound. FP8 KV + HiCache is a multiplicative combination.

### Quantization + model architecture (MLA, GQA)

Modern models (DeepSeek-V3, Llama-3, Qwen2.5) already use MLA or GQA to reduce KV size architecturally. Applying FP8 quantization on top gives:

| Model | Architecture | KV reduction vs Llama-2 MHA BF16 |
|---|---|---|
| Llama-3.1-70B | GQA-8, BF16 | 8× smaller |
| Llama-3.1-70B | GQA-8, FP8 | **16× smaller** |
| DeepSeek-V3-671B | MLA, BF16 | ~60× smaller (vs hypothetical MHA) |
| DeepSeek-V3-671B | MLA, FP8 | **~120× smaller** |

This is why DeepSeek-V3 can be served efficiently on much less hardware than its parameter count suggests.

---

## Open Problems Identified by the Survey

1. **Unified framework** combining quantization + token dropping + model changes
2. **Hardware-efficient sub-4-bit kernels** — custom CUDA for INT2/INT3 dequant at attention speeds
3. **Pre-RoPE quantization** in production (KVQuant shows it works, but engines don't implement it)
4. **Quantization-aware training (QAT)** for KV cache — all current methods are PTQ
5. **Multimodal KV quantization** — vision tokens have different distributions than text tokens

---

## Key Takeaways for Layer 18

- KV cache quantization is a **token-level** optimization — it changes how individual token KV values are encoded, not where they are stored.
- The survey places FP8 (production-ready) at the beginning of the quantization spectrum; 2–4-bit is approaching production for select models.
- **The right combination for production**: MLA/GQA architecture + FP8 KV quantization + HiCache tiering = three multiplicative improvements in KV memory efficiency.
- The curated list at [github.com/TreeAI-Lab/Awesome-KV-Cache-Management](https://github.com/TreeAI-Lab/Awesome-KV-Cache-Management) is the best reference for new papers as the field evolves rapidly.
