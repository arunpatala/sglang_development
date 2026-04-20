# Overall guidelines — book and teaching strategy

This file is the **top-level compass** for *LLM Inference from Scratch* (and similar code-forward technical books). It answers: **who the book is for**, **what job each layer of content does**, and **how the pieces in this folder fit together**.

Chapter-level voice, empathy, and section craft live in `GENERAL_GUIDELINES.md`. Lesson-style lint checks tied to `CODE_LAYERS` patterns live in `SPECIFIC_GUIDELINES.md`. External references live in `RESOURCES.md`. The hierarchical / zoom-level content model lives in `HIERARCHICAL.md`.

---

## How this writing guide fits together

| Document | Role |
|----------|------|
| **OVERALL_GUIDELINES.md** (this file) | Strategy: audience, content modes, scope, validation, relationship between book and repo. |
| **GENERAL_GUIDELINES.md** | Pedagogy and prose: stakes-first, invariants, code budget, checkpoints, sanity tests. |
| **SPECIFIC_GUIDELINES.md** | Concrete editorial checks anchored in real lessons (e.g. Layer 14 speculation). |
| **RESOURCES.md** | External books and frameworks (Diátaxis, teaching, cognitive science, publishing). |
| **PERSONAS.md** | Reader levels (1–5): intents, media fit, depth, anti-goals; maps spine vs satellite content. |
| **HIERARCHICAL.md** | Zoom-level / progressive disclosure model: spine vs drill-ins, tree navigation, metadata, build order. |

---

## 1. One primary reader; declare it on page one

A book that tries to be for “everyone” becomes documentation-shaped: cautious, exhaustive, low personality. Pick **one primary persona** (e.g. strong Python, weak GPU; or ML engineer, weak systems). Secondary readers are fine; they get **signposts** (“skip this box on first read”), not equal weight in every sentence.

---

## 2. One clear promise per volume (or per part)

Finish this sentence and put it where you see the manuscript drifting: *After this book/part, the reader can ___, and they can explain why ___ without the code in front of them.*

If a chapter does not advance that promise, it is a candidate for cut, merge, or appendix.

---

## 3. Use four content modes on purpose (Diátaxis spine)

Readers get tired when **learning**, **doing**, **looking up**, and **understanding** are blended in one paragraph. Across the book (and companion site), separate:

- **Tutorial** — guided path to a working artifact or a traced run.
- **How-to** — task recipes (“add speculative decoding,” “debug KV length”).
- **Reference** — APIs, structs, invariants table, file anchors.
- **Explanation** — mental models, tradeoffs, correctness arguments.

Code-heavy “from scratch” work skews **tutorial + reference**; reserve **explanation** for the pages that teach *why* before *where in the repo*. See [Diátaxis](https://diataxis.fr/) in `RESOURCES.md`.

---

## 4. The repo is allowed to be complete; the chapter is not

The **repository** can be the full truth (long files, edge cases). The **chapter body** should carry **bounded** code and one spine example; defer the rest to “repo track,” appendix, or linked source.

Code-driven learning means the reader *can* go deep in code—not that every reading pass must parse every line.

---

## 5. Validate like a product, not like a monologue

Ship material in **small slices**: blog posts, workshop sections, early-access chapters, exercises with solutions. Watch where people stall (timing, notation, missing diagram). Revise **backwards**: Chapter 4 often rewrites Chapter 3.

Teaching references: [Teaching Tech Together](https://teachtogether.tech/en/) in `RESOURCES.md`.

---

## 6. Manage cognitive load at the book scale

At the **book** level, cognitive load is: number of new concepts per chapter, unresolved notations carried forward, and parallel storylines (e.g. math + CUDA + Python packaging in one breath). Rules of thumb:

- **One new core idea per major section** where possible.
- **Resolve notation collisions** (indexing, “length vs position”) with a figure or table once, then reuse.
- **Spiral**: revisit the same invariant in new contexts rather than front-loading every variant.

Multimedia and segmenting principles (Mayer, Clark) summarized in `RESOURCES.md` apply to figures, callouts, and chapter length—not only to video.

---

## 7. Fight the curse of knowledge explicitly

Assume the author (or AI assist) will **skip** the step that felt obvious after the tenth implementation. Overall fix: **wrong-intuition callouts**, **reconciliation tables** when two definitions almost match, and **external novice readers** on a schedule—not only at the end.

---

## 8. Personality without noise

Personality is **direct address**, **honest scope**, **rhythm** (questions, callbacks), and **taste** (which ugly corner of the system you choose to show). It is not mandatory jokes or anecdotes unless they serve the model. The mind behind the page should be visible; the chat log behind the draft should not be.

Details: `GENERAL_GUIDELINES.md` §6 and §10.

---

## 9. Measurement and “production truth” get explicit lanes

When you introduce metrics (throughput, acceptance rate, memory fractions), treat them like API surfaces: **definition**, **what they do not mean**, **common misread**. Inference systems are full of footguns; the book should train skepticism, not only optimism.

Pattern: `SPECIFIC_GUIDELINES.md` §9; habit: `GENERAL_GUIDELINES.md` cross-cutting “Metrics need footguns.”

---

## 10. Keep the manuscript maintainable

Inference stacks change. Prefer **stable concepts** (invariants, complexity shapes, dataflow) over volatile names where possible; when you bind to a project (e.g. SGLang), say **version or era** and keep **deep anchors** in a reference strip or repo.

---

## Quick alignment checklist (before you ship a chapter)

1. **Persona** — Who is drained if I add one more prerequisite sentence? Who is bored if I remove it?
2. **Mode** — Is this chunk tutorial, how-to, reference, or explanation? Is the heading honest about that?
3. **Promise** — What can the reader *do* or *predict* after this chapter that they could not before?
4. **Load** — What is deferred to “second read,” and is that called out?
5. **Proof** — Is there at least one **trace** (paper or debugger) and one **sanity test**, not only a summary?

---

## See also

- `GENERAL_GUIDELINES.md` — ten chapter-level pedagogy rules + cross-cutting habits.
- `SPECIFIC_GUIDELINES.md` — ten editorial checks with Layer 14 examples.
- `RESOURCES.md` — external frameworks and books.
