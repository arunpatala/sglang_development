# Reader personas — five levels with examples

The five levels are **reading intents**, not permanent identities. The same person may be Level 2 on a commute and Level 5 on a weekend. Design **tracks, callouts, and signposts** — not "this entire chapter is only for Level 3 readers."

These levels map to **zoom depth** in the hierarchical model (`HIERARCHICAL.md`): L1–L2 cruise high; L3 is the mandatory spine; L4–L5 are drill-ins.

See also: `HIERARCHICAL.md` (zoom and navigation model), `OVERALL_GUIDELINES.md` (strategy), `GENERAL_GUIDELINES.md` (voice and prose craft).

---

## How levels connect to the zoom model

Each chapter has one **spine** (L3, always mandatory) and **typed attachments** (optional magnifications):

```
Topic: 08-paged-attention
├── 08-paged-attention-primer        ← L1 attachment
├── 08-paged-attention-glossary      ← L2 attachment
├── 08-paged-attention               ← L3 spine (mandatory)
├── 08-paged-attention-production    ← L4 attachment
└── 08-paged-attention-lab           ← L5 attachment
```

Readers at different levels open different paths into the **same topic node**.

---

## Level 1 — Curious non-builder ("Why should I care?")

### Who they are

Non-technical or semi-technical. Reads AI news, uses LLM products, curious about how they work. No expectation of implementing anything.

**Typical background:** product manager, writer, student exploring the field, informed non-engineer.

### What they want

Orient and demystify. One clean analogy. An honest limit on that analogy. A sense of what the cost or tradeoff is.

### What success looks like

Can explain in plain language: *what problem* a technique solves, *what cost* it involves (latency, memory, quality), *why it is hard without it* — without pretending to know GPU details.

### Media and format

Short blog post, newsletter section, one-pager, optional short animated figure. **500–1,000 words max.** No math, no code.

### Zoom position

Highest zoom: **topic overview only**. No drill-ins. This is the "what you see from 10,000 feet" node.

### Example: speculative decoding at L1

> Your phone autocomplete suggests the next few words before you finish typing — it is fast because it is small and often wrong. Speculative decoding works like this: a tiny fast model guesses the next several tokens, and a large accurate model checks all of them at once. If the guess was right, you get multiple tokens for the cost of one verification step. If it was wrong, you throw out the mistake and the large model corrects it. The GPU is always doing something useful — the draft keeps it busy, and the checker never stalls.

**Why this metaphor breaks:** phone autocomplete is visible to you; speculative decoding's guesses are never shown to the user if rejected. Name the limit.

### Anti-goals

- Fake precision (exact throughput numbers, specific GPU names) without scaffolding.
- Implying "full understanding" from one post.
- Being wrong to be simple — metaphors must be honest about their limits.

---

## Level 2 — Term-literate reader ("Refresh, then motivate")

### Who they are

Has heard the terms (transformer, KV cache, batching, quantization) but definitions slide around. Reads ML blogs, maybe took a course. Not shipping an engine but wants *correct intuition*.

**Typical background:** data scientist, ML practitioner, grad student adjacent to but not in inference systems.

### What they want

Re-anchor vocabulary with tight definitions. Then see a motivating example or diagram that makes the *new idea* feel necessary, not arbitrary.

### What success looks like

Can narrate a small concrete scenario and spot common misconceptions in headlines. Can draw a rough picture on a whiteboard.

### Media and format

Blog series, illustrated tutorial, guided tour page. May include small diagrams. **Toy code snippets are fine** (illustrative, not contractual). Still not architecture completeness.

### Zoom position

**Cluster-level gloss + local motivation.** Appears at the entrance to a chapter cluster (e.g. before the KV cache chapters) as a glossary + "here is why this cluster matters" page — not repeated inline every section.

### Example: KV cache at L2

> Every transformer layer computes keys and values from all prior tokens. Without caching, generating token #100 repeats work for tokens #1–99 every step. The KV cache stores those keys and values in GPU memory so each new token only needs to attend to already-computed data.
>
> The cost: GPU memory. One token's KV across all layers and heads for a 7B model is roughly a few KB; a 4,096-token sequence is several hundred MB. That is the tradeoff you will see in every inference system: speed (no recompute) for memory (always resident).

No code yet. One diagram of "compute vs cache" covers the concept. The next section (L3 spine) will formalize page layout and indexing.

### Anti-goals

- Long code listings that read as documentation.
- Un-signposted prior-chapter dumps ("In Chapter 3 we established…" without any re-anchoring).
- Architecture completeness — save that for L3.

---

## Level 3 — Book spine reader ("Throwbacks + invariants + composition")

### Who they are

Follows the book narrative chapter by chapter (or has equivalent background). Wants to **understand the system as a map**: where each piece fits, what changes with each new technique.

**Typical background:** software engineer curious about ML systems, ML engineer moving into inference, graduate student studying serving stacks.

### What they want

A **throwback** to what was true before this chapter, then the **one thing that changes**, then the **invariant that restores correctness**. Bounded pseudocode or real code as confirmation — not the opening move.

### What success looks like

Can explain the mechanism as **dataflow + invariants**. Can do a **hand trace** on small inputs (5 tokens, 2 pages, N=3 draft tokens) without needing the codebase open.

### Media and format

Online book chapters. Structure: stakes → throwback → new idea → invariant → pseudocode → thin real snippet → sanity check. **This is the mandatory spine.** All other levels are optional relative to L3.

### Zoom position

**Main chapter node** in the topic tree. L1/L2 are linked from the header; L4/L5 are linked at the footer or in sidebar callouts.

### Chapter header template

```
## Chapter N — [Title]

**Assumes:** Chapters X, Y (or: "L2 glossary on KV" for new readers)
**Delivers:** After this chapter you can [one sentence outcome].
**Optional:** [L4 Production notes] | [L5 Lab: code_layers/N]
```

### Example: speculative decoding at L3

**Throwback:** In Layer 13, every decode step commits exactly one token per target forward pass. The bottleneck is not how fast the target runs — it is how many times we call it.

**New idea:** Run a small draft model N times (cheap); then run the target once over all N+1 positions in parallel (one extend, same causal mask as autoregressive decode). Accept the longest correct prefix.

**Invariant:** After each spec step, target KV and draft KV both cover exactly `kv_committed_len` positions. Rejected positions are freed before the next step begins. If this invariant breaks, the next draft phase starts from a corrupt prefix.

**Pseudocode (10 lines):**

```python
for step in range(max_steps):
    draft_tokens = draft_model.decode_n_steps(context, N)
    logits = target_model.extend([context[-1]] + draft_tokens)
    accept_len = first_mismatch(logits.argmax(), draft_tokens)
    bonus = logits[accept_len].argmax()
    commit(draft_tokens[:accept_len] + [bonus])
    rewind_kv(target_pool, accept_len + 1)
    rewind_kv(draft_pool, accept_len)
```

**Sanity check:** If `accept_len = 0` every step, you still emit `bonus` each time — one token per target call, same throughput as Layer 13, no regression.

### Anti-goals

- Production gotchas inlined in every paragraph (those belong in L4 callouts).
- Full labs without a map (L5 is separate).
- Code-first layout where listings carry all the explanatory weight.

---

## Level 4 — Engine reader ("Real systems, tradeoffs, gotchas")

### Who they are

Solid at L3. Wants to read SGLang, vLLM, or Triton code with confident questions. Interested in failure modes, correctness edges, memory fragmentation, profiling.

**Typical background:** inference engineer, ML systems researcher, engineer evaluating or deploying serving stacks.

### What they want

Map the L3 mental model to **real component names, real knobs, and real failure modes**. Not a tutorial — a guided read of how real systems make the same tradeoffs.

### What success looks like

Can open a design doc or PR and ask: "What invariant does this maintain? What breaks if this assumption fails? What benchmark proves this is better?"

### Media and format

Deep-dive appendix section, "Production notes" callout box inside the L3 chapter, annotated traces, engine-specific reading guides. **Real code, guided as a reading exercise** — not a lab.

### Zoom position

**Optional drill-in** attached to the L3 node. Declared in the L3 chapter footer: `Production notes: 08-paged-attention-production`.

### Example: speculative decoding at L4

> **SGLang anchor:** `spec_runner.py → SpecRunner._rewind_target_kv`. The rewind logic must handle partial pages: if `accept_len + 1 = 13` and `page_size = 16`, page 0 is kept (holds positions 0–15) but positions 13–15 within that page are stale. FlashInfer uses `kv_last_page_lens` to know how many tokens are valid in the last page — the stale positions are never attended to, but the page is not freed until overwritten on the next step.
>
> **Gotcha:** if `page_size` differs between draft and target `ModelRunner`, `compute_write_info` silently produces misaligned indices. No assertion fires; outputs are wrong. Always assert `draft.page_size == target.page_size` at `SpecRunner.__init__`.
>
> **Production knob:** `N` (draft steps per spec step). At `N=1` you get close to standard decode with extra draft overhead. At `N=10` with a weak draft model, expected `accept_len` < 1 and you are slower than baseline. Acceptance rate below `~0.5` is a signal to reduce `N` or switch draft models. `SpecRunner.stats_str()` exposes both metrics at inference time.

### Anti-goals

- Lab-style exercises (L5 covers those).
- Full file listings without guided structure ("here is 200 lines, read it").

---

## Level 5 — Builder ("Implement, extend, contribute")

### Who they are

Wants **muscle memory**: write code, run tests, benchmark, open PRs. Uses `CODE_LAYERS` as a scaffold. May be a student in a systems course, an engineer onboarding to an inference team, or a contributor to an open-source engine.

**Typical background:** anyone who learns by doing — the L3 map is the prerequisite, the lab is the work.

### What they want

A **runnable artifact** with clear success criteria. Environment set up. Tests that fail when the invariant is broken. Extension tasks with increasing scope.

### What success looks like

Ships a small change. Passes targeted tests. Can justify design choices against invariants from the L3 spine. Can open a PR with a coherent description.

### Media and format

Repo (`CODE_LAYERS`) + lab workbook + automated tests. May include a short debugging walkthrough. **Full real code; tests are first-class.**

### Zoom position

**Optional drill-in** attached to the L3 node, called out in the chapter header as `[L5 Lab: code_layers/N]`. Explicitly marked optional in the preface.

### Example: speculative decoding at L5

```
Lab: CODE_LAYERS/layer14_speculative_decoding

Starter tasks (verify your L3 understanding):
  1. Run tests/test_speculative.py — all should pass.
  2. Add an assertion in SpecRunner.__init__ that page_size matches
     between draft and target. Confirm the gotcha from L4 is now caught.
  3. Print acceptance_rate and tokens_per_step after 50 steps on
     a short prompt. Does the number match the L3 formula (1 + N × p)?

Extension tasks (go beyond the book):
  4. Implement a dynamic N: if acceptance_rate < 0.5 for 10 steps,
     halve N; if > 0.8, increase N up to the configured max.
  5. Batch the draft phase: run all requests' N draft steps together
     in one draft.decode_step call instead of sequentially.
     What invariants does _rewind_draft_kv need to handle now?

Contribution path:
  - Pick extension 4 or 5 and open a PR against the layer repo.
  - Checklist: does your change preserve the mirroring invariant?
    Does tokens_per_step improve or stay stable?
```

### Anti-goals

- Hidden environment setup cost ("just run the code").
- Conflating "it compiled" with "I understand the mechanism" — always tie the lab back to L3 invariants.
- Exercises with no pass/fail signal — every task should be verifiable.

---

## One-page matrix

| Level | Intent | Best media | Code | Zoom position | Primary risk |
|------:|--------|------------|------|---------------|--------------|
| 1 | Orientation | Blog / card | None | Topic overview | Wrong or patronizing metaphors |
| 2 | Repair terms + motivate | Tutorial / illustrated | Toy snippets | Cluster gloss node | Endless recap, no new learning |
| 3 | Composable map | Book chapter (spine) | Pseudocode + thin real | Main chapter node | Becomes internal spec |
| 4 | Production truth | Deep dive / callout | Real, guided read | Optional drill-in | Exhausting without the L3 map |
| 5 | Build track | Repo + lab workbook | Full, with tests | Optional drill-in | Labs without conceptual spine |

---

## Chapter envelope (use at the top of every L3 chapter)

```markdown
**Assumes:** [prior chapters or concepts]
**Delivers:** After this chapter you can [one concrete outcome].
**Optional deeper reading:**
- [L4 Production notes →]
- [L5 Lab: CODE_LAYERS/layerN →]
```

This header, placed before the first section, does more anti-overwhelm work than shortening the truth anywhere else in the chapter.

---

## Key rules (from `HIERARCHICAL.md`, applied to personas)

1. **Levels are zoom policies, not separate books.** One topic tree; L1–L5 are typed attachments per node.
2. **L3 spine must stay true if all other levels are skipped.** If an L4 gotcha is load-bearing for correctness, promote it to L3.
3. **L1 must name the limits of its metaphor.** Simplicity is not permission to be wrong.
4. **L2 belongs at cluster entrances, not inline in every section.**
5. **L4 and L5 share code; their success criteria differ.** L4: read and question. L5: change and prove.
