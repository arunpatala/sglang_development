# Loquetier: A Virtualized Multi-LoRA Framework for Unified LLM Fine-tuning and Serving

**Source:** https://arxiv.org/abs/2511.00101
**Paper PDF:** https://arxiv.org/pdf/2511.00101
**Code:** https://github.com/NJUDeepEngine/Loquetier
**Authors:** Yuchen Zhang, Hanyue Du, Chun Cao, Jingwei Xu (Nanjing University)
**Submitted:** October 30, 2025
**Venue:** NeurIPS 2025 (inferred from submission date and acceptance timeline)
**Level:** L4 — Advanced research system; unified fine-tuning and serving with virtualized multi-LoRA
**Why here:** Loquetier addresses a gap that all prior work ignored: production ML systems need to simultaneously *train* new adapters (fine-tuning on new data) and *serve* existing adapters (inference on live traffic). Running both on the same GPU cluster creates resource contention. Loquetier's Virtualized Module design isolates training and serving paths while sharing the base model, and its fused kernel design enables 3× serving throughput over the prior state-of-the-art co-serving system.

**BibTeX:**
```bibtex
@inproceedings{zhang2025loquetier,
  title  = {Loquetier: A Virtualized Multi-{LoRA} Framework for Unified
            {LLM} Fine-tuning and Serving},
  author = {Yuchen Zhang and Hanyue Du and Chun Cao and Jingwei Xu},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year   = {2025},
  url    = {https://arxiv.org/abs/2511.00101}
}
```

---

## Abstract

Low-Rank Adaptation (LoRA) has become a widely adopted PEFT technique for adapting large language models to downstream tasks. While prior work has explored strategies for integrating LLM training and serving, there still remains a **gap in unifying fine-tuning and inference for LoRA-based models**. We present Loquetier, a virtualized multi-LoRA framework that seamlessly integrates LoRA fine-tuning and serving within a single runtime.

Loquetier introduces two key components:
1. A **Virtualized Module** that isolates PEFT-based modifications and supports multiple adapters on a shared base model
2. An **optimized computation flow with a kernel design** that merges fine-tuning and inference paths in forward propagation, enabling efficient batching and minimizing kernel invocation overhead

Results: Loquetier achieves up to **3.0× the throughput** of the state-of-the-art co-serving system on inference-only tasks and **46.4× higher SLO attainment** than PEFT on unified fine-tuning and inference tasks.

---

## The Gap: Fine-Tuning + Serving Simultaneously

Prior multi-LoRA work (Punica, S-LoRA, dLoRA) focused exclusively on inference serving:
- Base model fixed in memory
- Multiple adapters loaded/evicted dynamically
- Requests processed in inference mode only

Real production deployments also need to:
- Fine-tune new adapters on fresh user data continuously
- Serve existing adapters to live users without interruption
- Run both workloads on the same GPU cluster to avoid over-provisioning

Running these simultaneously creates **resource contention**:

```
GPU Resources:
┌─────────────────────────────────────────────────┐
│                                                 │
│  Training: forward + backward + optimizer step  │  ← large memory
│  Serving: prefill + decode × multiple adapters  │  ← time-sensitive
│                                                 │
└─────────────────────────────────────────────────┘
         ↑ both compete for VRAM and compute
```

---

## The Virtualized Module

Loquetier introduces a **Virtualized Module** (VM) abstraction that wraps each LoRA-targeted base layer:

```python
class VirtualizedModule:
    def __init__(self, base_layer):
        self.base_layer = base_layer      # shared, frozen
        self.adapters = {}                # adapter_id → (A, B, mode)
        # mode: "train" or "serve"

    def forward(self, x, adapter_id=None, mode="serve"):
        base_out = self.base_layer(x)
        if adapter_id is None:
            return base_out
        A, B, _ = self.adapters[adapter_id]
        if mode == "serve":
            return base_out + self._serve_delta(x, A, B)
        else:  # mode == "train"
            return base_out + self._train_delta(x, A, B)  # includes gradient tracking
```

Key design decisions:
- Base layer is **never duplicated** regardless of how many adapters are active
- Each adapter has its own `(A, B)` pair with independent gradient state
- Training and serving adapters can be in the pool simultaneously
- The VM separates the PEFT bookkeeping from the underlying base layer computation

---

## Fused Kernel for Training + Serving

The key performance contribution is a single kernel that handles both fine-tuning and inference batches:

### Naive approach (prior systems)

```
Training batch  → separate kernel → loss + gradients
Serving batch   → separate kernel → output logits
```

Two kernel launches per forward pass, even when both are operating on the same base model layer.

### Loquetier's fused approach

```
Mixed batch [train_tokens, serve_tokens]
              ↓
     Single fused kernel:
     - Computes base model output for all tokens
     - For train_tokens: computes LoRA delta + gradient checkpoints
     - For serve_tokens: computes LoRA delta (no gradient tracking)
     - Applies appropriate masking per token type
```

This is conceptually similar to Layer 20's float mask, but extended to two types of masks: training and serving.

---

## Evaluation

Three task settings evaluated on A100 80GB:

| Setting | Baseline | Loquetier | Improvement |
|---|---|---|---|
| Inference only | SOTA co-serving | Loquetier | **3.0× throughput** |
| Training only | PEFT | Loquetier | ~1.5× throughput |
| Unified (training + serving) | PEFT | Loquetier | **46.4× SLO attainment** |

The 46.4× SLO attainment improvement on unified tasks is the headline result: PEFT's single-adapter sequential processing fails massively when mixed with serving workloads, while Loquetier's virtualized batching handles both efficiently.

---

## Comparison to Layer 20

| Feature | Layer 20 | Loquetier |
|---|---|---|
| Mode | Serving only | Unified training + serving |
| Adapters | 1 static | N dynamic |
| Training support | None | Full backward pass |
| Kernel | Float mask (CPU construct) | Fused GPU kernel |
| VM abstraction | Not used | Core design |

Loquetier represents the direction Layer 20 would evolve toward if used in an active learning pipeline — where new adapters are trained continuously as users interact with the system, while existing adapters serve live traffic.
