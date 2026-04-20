# Hierarchical / zoom-level book model

**One map, multiple magnifications.**

The book has one topic tree (the inference story). At each node, readers choose how deep to go. This is not literal zoom — it is information architecture and authoring discipline.

See also: `PERSONAS.md` (L1–L5 reader intents), `OVERALL_GUIDELINES.md` (strategy), `GENERAL_GUIDELINES.md` (spine prose and code budget).

---

## Core idea

The book has a **mandatory spine** (short, always true: problem → invariant → consequence) and **optional drill-ins** (derivations, long code, production notes, labs) attached at specific anchors.

> **Spine is mandatory truth; everything else is optional magnification.**

Readers choose a default zoom: cruise high, or punctuate with deep dives on topics that matter to them.

---

## Three navigation questions at any node

At every part, chapter, section, or named topic anchor, the reader should be able to answer:

| Direction | Question |
|-----------|----------|
| **Up (parent)** | What container am I in, and why does it exist? |
| **Left / right (siblings)** | What did I just read? What comes next at this level? |
| **Down (children)** | What can I open deeper here? |
| **Across (requires)** | What must I already know, even if it lives elsewhere? |

Each node should provide short context blocks for all four: a parent sentence, prev/next glue lines, a "in this chapter" child outline, and "you should already …" prerequisite links.

**Rule of thumb:** avoid more than ~three practical zoom depths (overview → section body → appendix / lab). Deeper nesting is hard to navigate and maintain.

---

## Two hierarchies — do not collapse them

### A) Outline tree (reading order)

**Part → Chapter → Section → (optional subsection)**

Drives narrative order and manuscript zoom depth.

### B) Depth attachments (same topic, different job)

For a canonical topic id (e.g. `08-paged-attention`):

| Artifact suffix | Level | Job |
|-----------------|-------|-----|
| `-primer` | L1 | One-screen orientation, no code |
| `-glossary` | L2 | Definitions + motivation + toy examples |
| *(main chapter)* | L3 | Spine: throwbacks, invariants, bounded pseudocode |
| `-production` | L4 | Engines, gotchas, traces, tuning |
| `-lab` | L5 | Exercises, `CODE_LAYERS`, tests |

Outline children are **subsections of the chapter**.
Depth attachments are **siblings in depth**, linked from the chapter header — not infinite nesting in prose.

---

## The one rule that keeps it honest

> If skipping a drill-in makes the **next spine paragraph false**, the box was never optional — merge it into the spine.

Additional design rules:

1. Every drill-in **declares its parent anchor** ("Unpacks §3.2 *Paged indices*") so readers can return.
2. One drill-in box, **one job** — derivation *or* long code *or* production story *or* exercise. Mixed boxes read like documentation soup.
3. Repo may be complete; **chapter body stays bounded** — long truth lives in appendix or repo, not inline.
4. Use **stable ids** for anchors (e.g. `08-paged-attention`). Titles can change; links should not break.

---

## Reader levels map to zoom (not five separate books)

Personas (L1–L5) are **different default zoom policies** on the same content graph:

| Reading style | Typical zoom |
|---------------|--------------|
| High zoom | L1–L2 flavor + compressed L3 spine (still must stay true) |
| Drill in on one topic | Full L3 → optional L4 → optional L5 |

---

## Metadata to manage the graph

For each stable unit, maintain a small record (YAML front matter or equivalent):

- **`id`** — stable slug; titles can change, this should not.
- **`parent_id`** — outline parent.
- **`prev_id` / `next_id`** — narrative siblings at this level.
- **`children_ids`** — finer outline pieces.
- **`requires` / `see_also`** — prerequisite and lateral links.
- **`artifacts`** — map of level → path/url: `{ L1, L2, L3, L4, L5 }`.

Generate breadcrumbs, sidebars, and prev/next from this **once**. Do not hand-maintain in fifty places.

---

## Build order

1. **L3 topic tree first** — establish dependency order (what must precede paging, speculative decoding, etc.).
2. **Per node, add the envelope:** `Assumes / Delivers / Optional` (L4/L5 pointers) — see `PERSONAS.md`.
3. **L2/L1 satellites** — cluster-level primers, linked from chapter headers, not repeated as ELI5 walls inside every section.
4. **L4** — after L3 stabilizes for that topic (engine versions churn; don't chase them early).
5. **L5** — aligned to topic ids (`CODE_LAYERS` labs keyed to the same slug as the chapter).

---

## What this model is not

- Not "Part I = easy, Part II = hard" as the only axis — that still traps readers who need local depth.
- Not five unrelated linear books — one map, multiple magnifications at anchors.
- Not literal browser zoom — authoring discipline and information architecture.

---

## Quick authoring checklist (per section)

- [ ] **Spine true without boxes?** Can a reader skip all drill-ins and still follow what comes next?
- [ ] **Drill-in declared?** Each box labels its parent anchor and has one job.
- [ ] **Navigation defined?** `parent / prev / next / requires` noted (or consciously N/A).
- [ ] **Envelope present?** Chapter header states `Assumes / Delivers / Optional`.
