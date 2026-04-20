# SGLang Speculative Decoding Docs

**Source:** https://docs.sglang.ai/advanced_features/speculative_decoding.html
**Level:** L4 — The canonical production reference for SGLang speculative decoding; all CLI parameters, all algorithms, full launch examples
**Why here:** This is the primary production engine for Layer 14. Every CLI flag in the doc maps to a constructor argument or config value in `spec_runner.py`. The SGLang docs also define which EAGLE variant to prefer and why — directly informing `lesson/08_eagle.md`. The performance table (baseline 158 → EAGLE-2 244 → EAGLE-3 373 tokens/s on H100) is the canonical benchmark cited throughout the book.

---

## Available algorithms — decision tree

| Goal | Algorithm | CLI flag |
|------|-----------|----------|
| Best speed/quality | EAGLE-3 | `--speculative-algorithm EAGLE3` |
| Broad compatibility | EAGLE-2 | `--speculative-algorithm EAGLE` |
| Reduce lm_head overhead | EAGLE-2 + FR-Spec | `--speculative-token-map` |
| Model has built-in MTP heads | MTP | `--speculative-algorithm EAGLE` (small steps/topk) |
| Have a smaller draft LLM | STANDALONE | `--speculative-algorithm STANDALONE` |
| No extra model available | NGRAM | `--speculative-algorithm NGRAM` |
| Experimental overlap scheduling | SpecV2 | `SGLANG_ENABLE_SPEC_V2=True` |

---

## Performance highlights (LLaMA-3.1-8B, MT bench, 1× H100)

| Method | Throughput |
|--------|-----------|
| SGLang baseline (no speculative) | 158.34 tokens/s |
| SGLang + EAGLE-2 | 244.10 tokens/s (+54%) |
| SGLang + EAGLE-3 | 373.25 tokens/s (+136%) |

This is the reference benchmark for Layer 14. Any discussion of speculative decoding speedup in the book should cite these numbers or explain why your setup would differ (different model, hardware, batch size).

---

## Method comparison table

| Method | Draft source | Separate model? | Notes |
|--------|-------------|-----------------|-------|
| EAGLE-2 | EAGLE draft model (feature drafting + tree) | Typically yes | Tune `--speculative-num-steps`, `--speculative-eagle-topk`, `--speculative-num-draft-tokens` |
| EAGLE-2 + torch.compile | Same | Typically yes | Benefit varies by hardware; benchmark to verify |
| EAGLE-2 + FR-Spec | Same + token subset | Typically yes | Reduces `lm_head` overhead with high-frequency token vocab |
| EAGLE-3 | EAGLE3 draft model | Yes | Best throughput in benchmarks |
| MTP | Built-in multi-token heads (model-specific) | Often no | Uses speculative workflow; draft may be auto-handled |
| STANDALONE | Smaller draft LLM (token-level) | Yes | No `--enable-dp-attention` support |
| SpecV2 (experimental) | V2 workers + overlap scheduler | N/A | Only supports `topk=1` |
| NGRAM | Ngram cache from previous tokens | No | CUDA-only; no `--enable-dp-attention` |

---

## EAGLE-2 Decoding

Complete launch example:

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Llama-2-7b-chat-hf \
    --speculative-algorithm EAGLE \
    --speculative-draft-model-path lmsys/sglang-EAGLE-llama2-chat-7B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

Key parameters for EAGLE/EAGLE-3:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-draft-model-path` | Draft model path/weights | `None` (required for EAGLE) |
| `--speculative-num-steps` | Depth of autoregressive drafting | Auto: 5 for Llama/Grok, 3 for others |
| `--speculative-eagle-topk` | Branching factor per step | Auto: 4 for Llama/Grok, 1 for others |
| `--speculative-num-draft-tokens` | Max parallel verification capacity | Auto: 8 for Llama/Grok, 4 for others |
| `--speculative-accept-threshold-single` | Acceptance threshold for single-token | 1.0 |
| `--speculative-accept-threshold-acc` | Accumulated acceptance threshold | 1.0 |
| `--speculative-attention-mode` | `prefill` or `decode` | `"prefill"` |
| `--speculative-draft-model-quantization` | Quantization for draft model | Same as target |

> Leave `--speculative-num-steps`, `--speculative-eagle-topk`, and `--speculative-num-draft-tokens` **all unset** to use auto-tuning, or set **all three explicitly** when tuning.

## EAGLE-3 Decoding

```bash
python3 -m sglang.launch_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --speculative-algorithm EAGLE3 \
    --speculative-draft-model-path jamesliu1/sglang-EAGLE3-Llama-3.1-Instruct-8B \
    --speculative-num-steps 3 \
    --speculative-eagle-topk 4 \
    --speculative-num-draft-tokens 16 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --dtype float16 \
    --log-level warning
```

Note: `--speculative-token-map` (FR-Spec) is **ignored** for EAGLE-3 models.

---

## STANDALONE Decoding (classic two-model)

This is the mode Layer 14 implements. A separate smaller draft model proposes tokens, the target verifies.

```bash
python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 2 \
    --speculative-num-draft-tokens 7 \
    --mem-fraction-static 0.7 \
    --cuda-graph-max-bs 8 \
    --log-level warning
```

STANDALONE constraints:
- Does **not** support `--enable-dp-attention`
- `--speculative-draft-model-quantization "unquant"` to force no quantization on draft even when target is quantized

---

## MTP (Multi-Token Prediction)

For models with built-in MTP heads (e.g., DeepSeek, MiMo). Use small `num_steps`/`topk`:

```bash
python3 -m sglang.launch_server \
    --model XiaomiMiMo/MiMo-7B-RL \
    --speculative-algorithm EAGLE \
    --speculative-num-steps 1 \
    --speculative-eagle-topk 1 \
    --speculative-num-draft-tokens 2 \
    ...
```

---

## NGRAM Decoding

No separate model — retrieves draft tokens from an ngram cache built from previously generated tokens.

```bash
python3 -m sglang.launch_server \
    --model <model> \
    --speculative-algorithm NGRAM \
    --speculative-num-draft-tokens 12
```

NGRAM-specific parameters:

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--speculative-ngram-min-bfs-breadth` | Min BFS breadth | 1 |
| `--speculative-ngram-max-bfs-breadth` | Max BFS breadth | 10 |
| `--speculative-ngram-match-type` | `"BFS"` or `"PROB"` | `"BFS"` |
| `--speculative-ngram-max-trie-depth` | Max suffix length | 18 |
| `--speculative-ngram-capacity` | Cache capacity | 10,000,000 |

NGRAM constraints: CUDA-only; no `--enable-dp-attention`; disables overlap scheduler & mixed chunked prefill.

---

## SpecV2 (experimental overlap scheduler)

```bash
SGLANG_ENABLE_SPEC_V2=True python3 -m sglang.launch_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --speculative-algorithm STANDALONE \
    --speculative-draft-model-path Qwen/Qwen2.5-1.5B-Instruct \
    --speculative-num-steps 4 \
    --speculative-eagle-topk 1 \    # MUST be 1 for SpecV2
    --speculative-num-draft-tokens 5 \
    ...
```

SpecV2 constraint: **only supports `--speculative-eagle-topk 1`**. Setting topk > 1 will error.

---

## Full parameter reference (summary)

| Stat tracked | Internal metric | What it measures |
|--------------|-----------------|------------------|
| `spec_verify_stats` | `acceptance_rate` | Fraction of draft tokens accepted per step |
| `spec_verify_stats` | `tokens_per_step` | Average tokens generated per speculative step |

These map directly to `_total_accepted / _total_proposed` in Layer 14's `lesson/07_statistics.md`.

---

## How this maps to Layer 14

| SGLang CLI param | Layer 14 code |
|-----------------|---------------|
| `--speculative-algorithm STANDALONE` | `SpecRunner` with two `ModelRunner` instances |
| `--speculative-draft-model-path` | `DraftModelRunner` constructor arg |
| `--speculative-num-steps` | `num_spec_tokens` (how many draft steps) |
| `--speculative-eagle-topk` | Not used in STANDALONE (chain decoding, not tree) |
| `--speculative-num-draft-tokens` | Verification batch size (K + 1 tokens) |
| `--mem-fraction-static` / KV split | `kv_memory_fraction` between draft and target pools |
| `--speculative-algorithm EAGLE3` | Advanced: `lesson/08_eagle.md` extension |

The STANDALONE mode in the SGLang docs is the direct production equivalent of the two-`ModelRunner` architecture taught in Layer 14. Understanding the difference between STANDALONE and EAGLE modes is the key L4 insight.

---

## OOM troubleshooting (from docs)

If you get OOM errors with speculative decoding:
1. **Reduce `--mem-fraction-static`** — both draft and target need VRAM headroom for KV caches
2. **Reduce `--speculative-num-draft-tokens`** — fewer tokens in the verification batch
3. **Reduce `--speculative-eagle-topk`** — narrower tree uses less memory
4. **Use `--speculative-draft-model-quantization`** — quantize the draft model to save VRAM
5. **Use `"unquant"` for the draft** — if the target is quantized but you want draft in full precision for better acceptance rates

The `kv_memory_fraction` tension (how much VRAM to allocate to the draft model's KV pool vs the target model's) is a central production concern not visible from the outside — it's configured via `--mem-fraction-static` in combination with draft model size.

---

## FR-Spec optimization (EAGLE-2 only)

Frequency-Ranked Speculative Sampling reduces `lm_head` overhead by using only a high-frequency subset of the vocabulary in the draft model's head. The draft still proposes tokens from the target's full vocabulary, but the lm_head computation is cheaper because it operates on a smaller token subset.

```bash
--speculative-token-map thunlp/LLaMA3-Instruct-8B-FR-Spec/freq_32768.pt
```

Not applicable to EAGLE-3 models (ignored if set).
