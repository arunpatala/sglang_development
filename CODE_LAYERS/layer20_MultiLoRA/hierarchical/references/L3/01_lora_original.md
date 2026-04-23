# LoRA: Low-Rank Adaptation of Large Language Models

**Source:** https://arxiv.org/abs/2106.09685
**Paper PDF:** https://arxiv.org/pdf/2106.09685
**Code:** https://github.com/microsoft/LoRA
**Authors:** Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen (Microsoft)
**Submitted:** June 17, 2021 (v1), revised October 16, 2021 (v2)
**Venue:** ICLR 2022
**Level:** L3 — Foundational paper; the mathematical basis for everything in Layer 20
**Why here:** LoRA is the mathematical foundation of all of Layer 20. The `LoRAAdapter` class in `lora.py`, the masked forward pass in `attention.py` and `mlp.py`, and the `verify_lora.py` comparison against PEFT are all implementations of the LoRA algorithm defined in this paper.

**BibTeX:**
```bibtex
@inproceedings{hu2022lora,
  title  = {LoRA: Low-Rank Adaptation of Large Language Models},
  author = {Edward J. Hu and Yelong Shen and Phillip Wallis and Zeyuan Allen-Zhu
            and Yuanzhi Li and Shean Wang and Lu Wang and Weizhu Chen},
  booktitle = {International Conference on Learning Representations},
  year   = {2022},
  url    = {https://arxiv.org/abs/2106.09685}
}
```

---

## Abstract

An important paradigm of natural language processing consists of large-scale pre-training on general domain data and adaptation to particular tasks or domains. As we pre-train larger models, full fine-tuning, which retrains all model parameters, becomes less feasible. Using GPT-3 175B as an example — deploying independent instances of fine-tuned models, each with 175B parameters, is prohibitively expensive. We propose Low-Rank Adaptation, or LoRA, which freezes the pre-trained model weights and injects trainable rank decomposition matrices into each layer of the Transformer architecture, greatly reducing the number of trainable parameters for downstream tasks. Compared to GPT-3 175B fine-tuned with Adam, LoRA can reduce the number of trainable parameters by 10,000 times and the GPU memory requirement by 3 times. LoRA performs on-par or better than fine-tuning in model quality on RoBERTa, DeBERTa, GPT-2, and GPT-3, despite having fewer trainable parameters, a higher training throughput, and, unlike adapters, no additional inference latency.

---

## Core Idea: The LoRA Hypothesis

Full fine-tuning updates all model weights: `W = W₀ + ΔW`. The trainable delta `ΔW` lives in a high-dimensional space.

**LoRA hypothesis:** The intrinsic dimensionality of the adaptation `ΔW` is very low. Specifically, there exists a low-rank matrix `BA` where `B ∈ ℝ^{d×r}`, `A ∈ ℝ^{r×k}`, and `r ≪ min(d,k)` such that `ΔW ≈ BA`.

Instead of updating `W` directly, LoRA freezes `W₀` and trains only `A` and `B`.

---

## The LoRA Forward Pass

For a linear layer with weight `W₀ ∈ ℝ^{d×k}` and input `x ∈ ℝ^{k}`:

```
h = W₀x + ΔWx
  = W₀x + BAx · (α/r)
```

Where:
- `W₀`: frozen pre-trained weight matrix
- `A ∈ ℝ^{r×k}`: "down-projection" — maps input to rank-r space
- `B ∈ ℝ^{d×r}`: "up-projection" — maps rank-r space back to output
- `α`: hyperparameter for scaling (analogous to a learning rate)
- `r`: rank, typically 1–64; smaller = fewer parameters

**In transposed notation (as used in PyTorch with `F.linear(x, W)` convention):**

```python
h = x @ W₀.T + (x @ A.T) @ B.T * (alpha / r)
```

This is exactly what `LoRAAdapter.apply()` in Layer 20 computes.

---

## Weight Initialization

- **A initialized with Kaiming-uniform (random)** — ensures gradients flow from the first step
- **B initialized to zeros** — so `BA = 0` at initialization, meaning the pre-trained model behavior is preserved at the start of training

This initialization means that LoRA adds *no perturbation at initialization* but the model immediately starts adapting.

---

## Scaling: Why `alpha/r`?

The scaling factor `alpha/r` is introduced to make the adaptation magnitude comparable across different ranks:

- If `r` doubles (double the parameters), the scale is halved to compensate
- `alpha` is kept fixed as a hyperparameter (often set to `r` for simplicity, or `2r` for stronger adaptation)
- When `alpha = r`, the effective scaling is 1.0

**Practical note for Layer 20:** The `phh/Qwen3-0.6B-TLDR-Lora` adapter uses:
- `r = 8`, `lora_alpha = 32`
- Effective scaling = `32/8 = 4.0`
- Layer 20 stores `self.scaling = lora_alpha / r` and applies it in `apply()`

---

## Which Layers to Target?

LoRA can be applied to any weight matrix in a Transformer. The paper explores targeting:

| Target | Parameters | Typical Use |
|---|---|---|
| `q_proj` only | Minimal | Baseline |
| `q_proj`, `v_proj` | Balanced | Most common (paper recommendation) |
| All attention projections (`q`, `k`, `v`, `o`) | More params | Better quality |
| All attention + MLP (`gate`, `up`, `down`) | Full coverage | Best quality, more memory |

The paper finds that targeting more matrices with a smaller rank `r` outperforms targeting fewer matrices with larger `r`, given the same parameter budget. This is why modern adapters like `phh/Qwen3-0.6B-TLDR-Lora` target all 7 projection modules.

---

## No Inference Latency

Unlike adapter layers (which insert new modules between existing layers), LoRA can be merged into the base weights for inference:

```
W = W₀ + BA · (alpha/r)
```

After merging, the forward pass is exactly `W₀x + ΔWx = Wx` — the same as the base model, with zero additional overhead. However, this prevents serving multiple adapters simultaneously, since merging "bakes in" one adapter.

**Layer 20 deliberately avoids merging** to enable mixed-batch serving (base + LoRA in same batch). This introduces the masked delta computation overhead, which is the cost of multi-adapter capability.

---

## Empirical Results from the Paper

On GPT-3 175B:

| Method | Trainable Params | Memory Reduction | Performance vs Full FT |
|---|---|---|---|
| Full fine-tuning | 175B | — | baseline |
| Adapter layers | ~1.1M | 3× | comparable |
| LoRA (r=4) | ~4.7M | **3×** | **on-par or better** |
| LoRA (r=8) | ~9.4M | **3×** | **on-par or better** |

On RoBERTa-large:
- LoRA matches full fine-tuning on GLUE with 0.3% of trainable parameters

---

## Rank-Deficiency Investigation

The paper includes an empirical study showing that optimal fine-tuning subspaces are genuinely low-rank:

- Pre-training subspace and fine-tuning subspace have high overlap
- The adaptation direction is well-captured by rank-1 to rank-4 matrices for most tasks
- This explains why small `r` values work well in practice

This also motivates the research into adaptive rank allocation (e.g., EVA in PEFT, dLoRA in OSDI 2024).

---

## Relationship to Layer 20

Layer 20 is a direct implementation of the LoRA paper's serving scenario:

```
paper:   h = W₀x + BAx · (alpha/r)
layer20: output = base_output + delta * lora_mask
         where delta = (x @ A.T) @ B.T * scaling
               lora_mask ∈ {0.0, 1.0} per token
```

The `lora_mask` extension is Layer 20's contribution beyond the paper: it enables a mixed batch of base-model tokens (mask=0) and LoRA tokens (mask=1) in a single forward pass without running two separate forward passes.
