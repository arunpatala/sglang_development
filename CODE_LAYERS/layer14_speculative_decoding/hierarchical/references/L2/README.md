# L2 References: Speculative Decoding

**Level:** L2 — Definitions + motivation (terms re-anchored, small examples, minimal code)

**Reader profile:** Knows AI terminology, comfortable with Python, wants the mechanism explained clearly before any deep code. Satisfied when key terms are locked in, intuition is solid, and they can explain speculative decoding to a colleague.

---

## Files in this directory

| # | File | Source | Best for |
|---|------|--------|----------|
| 01 | `01_mlmastery_practitioner_guide.md` | ML Mastery | Locking in α, γ, τ with formulas + worked numbers. Full HF Gemma implementation. |
| 02 | `02_jarvislabs_speculative_decoding_vllm.md` | Jarvis Labs | Five-technique comparison with real benchmark tables (8B vs 70B, chat vs code). |
| 03 | `03_huggingface_tgi_speculation_conceptual.md` | HuggingFace TGI docs | Quick vocabulary anchor (Medusa, N-gram, draft model) before going deeper. |
| 04 | `04_huggingface_dynamic_speculation.md` | HuggingFace blog (Intel + HF) | Why static N is wrong; dynamic N tied directly to Layer 14's acceptance_rate metric. |
| 05 | `05_huggingface_layerskip_self_speculative.md` | HuggingFace blog (Meta) | One-model contrast; motivates why Layer 14 chose two-ModelRunner over self-speculation. |
| 06 | `06_adaptive_ml_speculative_decoding_visualized.md` | Adaptive ML (Jan 2026) | Best visual treatment of the probability geometry (green/red acceptance regions). Distribution preservation proof. |
| 07 | `07_pierce_freeman_how_speculative_decoding_works.md` | Pierce Freeman | Most concise explainer; best/worst/average case analysis; "you only need the draft to be right *some* of the time." |
| 08 | `08_brenndoerfer_interactive_guide.md` | Michael Brenndoerfer (Jan 2026) | Most complete single article: quantified memory bottleneck, full acceptance criterion code, speedup calculator, causal masking explanation. |

---

## Recommended reading order

**Fast path (30 min total):** 07 → 06 → 01
- 07 for the cleanest intuition and best/worst/average framing.
- 06 for the visual probability geometry (why distribution is exactly preserved).
- 01 to lock in α, γ, τ with formulas and a working HuggingFace implementation.

**Thorough path (90 min total):** 07 → 06 → 08 → 09 → 03 → 02 → 04 → 05
- Add 08 for the full interactive guide with quantified speedup analysis.
- Add 09 for the rigorous distribution proof + panoramic view of all SD variants.
- 03 for vocabulary disambiguation across Medusa/N-gram/draft model.
- 02 for real vLLM benchmark data across five techniques.
- 04 for dynamic scheduling and the tie to Layer 14's statistics.
- 05 for the LayerSkip contrast (why two-model was chosen).

---

## How these map to Layer 14 lessons

| Layer 14 lesson | Most relevant L2 reference |
|-----------------|---------------------------|
| `01_from_one_to_n_plus_one.md` (why?) | 01 (memory-bound bottleneck), 03 (vocabulary) |
| `02_spec_runner_architecture.md` | 05 (two-model vs. self-speculation contrast) |
| `03_draft_kv_mirroring.md` | 05 (LayerSkip shared KV cache concept) |
| `04_draft_phase.md` | 02 (draft workflow; "Guess, Check, Keep") |
| `05_verify_extend.md` | 09 (SpecJudge proof), 01 (α/γ/τ formulas), 06 (probability geometry) |
| `06_accept_reject_rewind.md` | 09 (SpecNormalize / residual), 01 (formulas), 02 (worked token example) |
| `07_statistics.md` | 04 (dynamic speculation; why static N is suboptimal) |

---

## Common L2 limits to name for readers

These articles **do not explain**:
- How two KV pools are allocated and managed simultaneously (`DraftReq` + `Req`).
- The N+1 window trick in `verify_extend` (why N draft tokens + 1 target token fit in one forward pass with correct causal masking).
- Production gotchas: P99 latency, prefill dominance, VRAM fragmentation.
- How to choose `kv_memory_fraction` to split VRAM between draft and target.

Those live in L3 (lesson files) and L4 (SGLang docs, production references).
