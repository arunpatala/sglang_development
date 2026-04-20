# L4 References: Speculative Decoding

**Level:** L4 — Production + engines (real stacks, gotchas, tradeoffs)

**Reader profile:** Has read Layer 14 lessons end-to-end. Comfortable with the accept/reject loop, KV pool management, and the two-`ModelRunner` architecture. Now wants to understand how this translates to real deployment decisions: which engine, which algorithm variant, what parameters, what fails in production.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_sglang_speculative_decoding_docs.md` | SGLang official docs | **Primary production reference.** All CLI flags, all algorithms (EAGLE-2/EAGLE-3/MTP/STANDALONE/NGRAM), performance table (158 → 244 → 373 tok/s on H100). Maps directly to Layer 14 code. |
| 02 | `02_vllm_speculative_decoding_docs.md` | vLLM v0.8.5 official docs | Parallel production engine. Best for: losslessness guarantees, floating-point caveats, draft TP constraint, n-gram and EAGLE modes. Comparison to SGLang reveals where the same algorithm is configured differently. |
| 03 | `03_specdecode_bench_performance_or_illusion.md` | SpecDecode-Bench / arXiv 2601.11580 | The honest production benchmark. Debunks the 3–4× numbers (batch=1 only). Shows speedups shrink with batch size, verification dominates runtime, chain > tree at batch > 32, n-gram wins code editing. Oracle 4.9× gap shows what's possible. |
| 04 | `04_redhat_speculators_production.md` | Red Hat Developer Blog | Ecosystem standardization. Speculators v0.2.0 defines a unified HuggingFace format; released EAGLE-3 models for Llama, Qwen3, Llama-4-Maverick. One-liner vLLM deployment. |
| 05 | `05_baseten_speculative_decoding_engine_builder.md` | Baseten Blog | Production config tutorial. Best for: complete YAML config file, GPU memory allocation (`kv_cache_free_gpu_mem_fraction`), when NOT to use speculative decoding (high load, lightweight models). |
| 06 | `06_eagle3_overview_e2e_networks.md` | E2E Networks Blog | Deepest treatment of EAGLE → EAGLE-2 → EAGLE-3 progression. Multi-layer feature fusion, training-time testing, tree attention masks, SGLang config tuning. |
| 07 | `07_magicdec_long_context_speculative_decoding.md` | arXiv 2408.11049 (CMU) | Long-context speculative decoding. Sparse KV self-speculation. Critical sequence length S_inflection. 2.51× speedup at batch=32–256 for 32K+ context. |
| 08 | `08_quantspec_quantized_kv_self_speculative.md` | arXiv 2502.10424 / ICML | Quantized KV self-speculation. Hierarchical INT4/INT8 sharing eliminates draft KV pool. Double full-precision buffer solves KV rollback cleanly. >90% acceptance rate. 2.49× speedup at 128K context. |

---

## Recommended reading order

**Immediately after Layer 14 (deployment path, 2–3 hours):**
01 → 02 → 03 → 05

- **01 first:** The production manual for the engine the book uses (SGLang). Every lesson in Layer 14 maps to a CLI flag here.
- **02 next:** The parallel engine (vLLM). The losslessness discussion and floating-point caveat are uniquely clear here.
- **03 for calibration:** Read before claiming any speedup numbers. Batch size kills most of the gain; this quantifies exactly how much.
- **05 for deployment:** The config template and the "when NOT to use" guidance close the loop on practical deployment.

**For the EAGLE extension lesson (lesson/08_eagle.md):**
06 → 04

- **06 first:** The mechanism of EAGLE-3 — multi-layer fusion, training-time testing, tree attention.
- **04 then:** The production models already available (RedHatAI collection on HuggingFace).

**For long-context and memory architecture (advanced):**
07 → 08

- **07:** Sparse KV approach; critical sequence length; when SD helps at large batches.
- **08:** Quantized KV approach; hierarchical INT4/INT8; the elegant KV rollback solution.

---

## How these map to Layer 14 lessons

| Layer 14 lesson | Most relevant L4 reference |
|-----------------|---------------------------|
| `01_from_one_to_n_plus_one.md` (why?) | 03 (calibrated speedup numbers), 07 (KV bottleneck at long context) |
| `02_spec_runner_architecture.md` | 01 (STANDALONE = two-ModelRunner), 02 (losslessness guarantee) |
| `03_draft_kv_mirroring.md` | 05 (kv_cache_free_gpu_mem_fraction), 08 (hierarchical KV as alternative) |
| `04_draft_phase.md` | 01 (STANDALONE num-steps), 06 (EAGLE tree building vs chain) |
| `05_verify_extend.md` | 06 (tree attention mask), 02 (batch expansion doc) |
| `06_accept_reject_rewind.md` | 08 (double FP buffer for clean KV rollback), 02 (rejection sampler tests) |
| `07_statistics.md` | 03 (acceptance varies by position/request/dataset), 01 (acceptance_rate CLI metric) |
| `08_eagle.md` (extension) | 06 (EAGLE-3 mechanism), 04 (Speculators models), 01 (EAGLE3 CLI config) |

---

## What L4 adds that L3 didn't cover

| Topic | L3 framing | L4 reveals |
|-------|-----------|-----------|
| "3–4× speedup" | Common benchmark claim | Only at batch=1; at batch=32, speedup is 1.2–1.5× (SpecDecode-Bench) |
| Verification cost | "The target verifies in one pass" | Verification is 42–95% of total runtime — it's the bottleneck, not the drafting |
| `num_speculative_tokens` | "Choose k=5 for a good default" | The optimal k also depends on engine, model, and batch size; all three flags must be tuned together |
| EAGLE vs STANDALONE | "EAGLE is an extension" | EAGLE-3 is 2.3× faster than STANDALONE at batch=1, due to multi-layer features and training-time testing |
| KV memory | "Draft has its own KV pool" | At long context (32K+), the draft KV pool is the memory bottleneck — sparse KV (MagicDec) or quantized KV (QuantSpec) eliminate it |
| Losslessness in practice | "Speculative decoding is lossless" | True up to floating-point precision; vLLM doesn't guarantee logprob stability across runs |
| When NOT to use | Not discussed at L3 | High GPU load, already-lightweight models, short sequences at high batch: all three cases make SD counterproductive |

---

## Common L4 limits to name for readers

These articles do not explain:
- How to **train** your own EAGLE-3 speculator (only how to use existing ones) — see L5/TorchSpec
- The internals of SGLang's `RadixAttention` and how the tree attention mask is generated at the CUDA level
- How continuous batching interacts with speculative decoding when requests have different sequence lengths (heterogeneous batch problem)
- The HASS algorithm (Harmonized Speculative Sampling) available in the Speculators library but not widely documented yet
