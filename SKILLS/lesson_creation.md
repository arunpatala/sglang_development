---
name: lesson-creation
description: >-
  Creates lesson materials for a new CODE_LAYERS layer. Use when adding a new
  layer folder (layer3_*, layer4_*, etc.) and needing to create 00_outline.md,
  summary.md, and the numbered section files. Captures the prose style, code
  anchoring, and before/after narrative conventions established in layers 0–2.
---

# Lesson Creation

Each layer in `CODE_LAYERS/` has a `lesson/` folder with three kinds of file:
the outline, the summary, and the individual section files. All three follow
specific conventions. Read this before creating any of them.

---

## 1. The Outline (`00_outline.md`)

The outline is the planning document. Write it first. It defines the section
list that all other files follow, so changes here propagate everywhere.

### Structure

```
# Layer N — Lesson Outline

## What This Lesson Covers
[One short paragraph connecting to the previous layer. State what changes,
what stays the same, and why the change matters.]

---

## Sections

### 01 — Title (`01_title.md`)
- bullet: what this section covers
- bullet: key concept introduced
- bullet: code anchor (file and rough line)

### 02 — Title (`02_title.md`)
...

---

## Supporting Files

- `summary.md` — blog-post-style summary covering all sections
- `sglang_reference.md` — maps layer concepts to SGLang source files

---

## Key Code Anchors

| Concept | Location |
|---|---|
| [concept] | `model.py` line N: `code snippet` |
```

### Rules
- Section numbers are zero-padded two digits: `01`, `02`, ..., `08`.
- Each section entry has 4–6 bullets describing what that file covers. Bullets
  in the outline can be concise; the section files are where detail lives.
- The "Key Code Anchors" table is specific: include the actual line and a short
  code snippet. Update it after writing `model.py` so line numbers are accurate.
- Sections are ordered to follow the code top to bottom (not concept-first).
  The natural reading order of `model.py` or the main changed file should
  dictate section order.

---

## 2. The Summary (`summary.md`)

The summary is a blog-post-style document that a reader could hand to someone
who has not read any of the section files. It is more detailed than a typical
blog post but shorter and less exhaustive than the individual sections.

### Structure

```
# Layer N — Summary

[One-sentence statement of what Layer N changes relative to Layer N-1.
One-sentence statement of what is unchanged (API, server.py, benchmark.py).]

---

## From [Previous Layer] to [This Layer]

[Show the key code from Layer N-1 — just enough to show what the problem was
or what was hidden. Then show the equivalent code from Layer N immediately
below. Explain the structural difference in one paragraph.]

---

## [Section Title matching 01_*.md]

[Show the relevant code block from model.py.]

[Explain what the code does and why, in prose. No bullet points.]

---

## [Section Title matching 02_*.md]
...

---

## What Comes Next

[One paragraph only. What does Layer N+1 change? What file does it touch?
What metric improves? What stays the same? End with the repeating pattern:
one concept changes, one file changes, the benchmark measures exactly that.]
```

### Rules

- **Opening section must show "before" code and "after" code** side by side (or
  stacked with labels). This is the structural anchor of the whole summary.
  Use a comment label: `# Layer N-1 — ...` and `# Layer N — ...`.
- **No bullet points anywhere in the body.** Every explanation is prose.
- **Each section maps to exactly one section file** and covers the same
  concepts, just less exhaustively.
- **Code blocks are mandatory** in every section except "What Comes Next".
  Show the actual snippet from the real file, not pseudocode, unless
  illustrating a future layer.
- **Section order in the summary must match the section file order** (01, 02,
  03, ...). Do not reorganise for narrative flow at the expense of alignment.
- **Tone**: clear, direct, technical. No hedging. No "it is important to note".
  State facts and explain consequences.

---

## 3. Section Files (`01_title.md` … `0N_title.md`)

Each section file is a textbook chapter: one self-contained topic, explained
thoroughly in prose with embedded code snippets. It is more detailed than the
summary section that covers the same topic.

### File template

```
# NN — Title

## [Subsection: the "before" state or the code anchor]

[For the first section file: open by showing the Layer N-1 code (the "before")
and the Layer N code (the "after"). Explain the structural difference.]

[For all other section files: open by quoting the specific lines from model.py
(or kv_cache.py, sampling.py, etc.) that this section explains. The code comes
first, the explanation follows.]

---

## [Subsection: explanation]

[Prose explanation. No bullet points. Explain what the code does, why it was
written that way, and what would break if it were different.]

---

## [Subsection: consequence or next connection]

[End with what this means for the next section or the next layer. Forward
references are allowed here — it is the natural landing point.]
```

### Mandatory conventions

**Open with the "before" code in section 01.**
Section 01 is always the decode loop or the main generate method. It must show
Layer N-1's version of the loop first (abbreviated to the key lines), then
Layer N's new structure. This is the anchor the reader returns to mentally
while reading every subsequent section.

**Open with a code anchor in sections 02 onwards.**
Every section after 01 opens by quoting the specific lines it explains. Do not
open with abstract explanation and then show code — show the code and explain
it. The pattern is: quote the code block, then explain it in prose below.

**Tie each section to the full feature in one orientation paragraph.**
Before the code anchor (or immediately after the title), add a short paragraph
— two to four sentences — that names where this piece lives in the full system
loop and why the system needs it. The reader should know, at a glance, which
step of the scheduler or generate loop this section explains and what breaks if
this piece is missing. This paragraph is not the detailed explanation; it is the
hook that prevents the section from reading as an isolated component description.

Example (for a section explaining `PerReqKVCache`):
> The decode step needs a single rectangular `[B, heads, kv_len, dim]` tensor
> for `F.sdpa`, but every request in `_running` has accumulated a different
> amount of KV history because they arrived at different times. `PerReqKVCache`
> stores each request's history independently; `BatchedKVCache` temporarily
> pads and stacks those histories into the rectangular shape the forward pass
> requires.

Example opening for section 04 (explaining `past_key_values`):
```markdown
The decode loop in section 01 contains a line that appears after every model call:

\`\`\`python
past_kv = out.past_key_values
\`\`\`

This line is easy to read past, but it is where the cache contract between our
code and HuggingFace's model internals is expressed. ...
```

**No bullet points.**
Write in paragraphs. If listing things (e.g., three vectors in attention),
introduce each with a bold term and follow with a sentence or two: `**The
query** is what a token is asking for. ...`. This reads as prose while still
being scannable.

**Concrete code snippets with shape comments.**
When showing tensors, always note the shape:
```python
out.logits        # shape: [batch, seq_len, vocab_size]
out.logits[0, -1, :]   # shape: [vocab_size]
```

**Cross-reference earlier sections.**
If a concept was established in a prior section, say so: "As section 02
established, the causal mask ensures..." Do not re-explain concepts that have
already been covered in the same layer.

**Forward references only at the end of a section.**
Do not mention Layer N+1 concepts mid-section. If a forward reference is
needed, put it in the final paragraph of the section as a "this is what
Layer N+1 changes" note.

---

## Narrative Arc (All Three Layers Follow This Pattern)

Every layer lesson follows the same arc:

1. **Start from the previous layer.** Show what the previous layer's code looked
   like. State what was hidden or what the problem was.

2. **Show the change.** Show the new code. State that the change is small (which
   file changed, how many lines).

3. **Explain each piece in code order.** Follow the model file top to bottom.
   Each section explains one part of the generate method (or the supporting
   file it delegates to).

4. **Revisit and tie together.** After all the pieces have been explained
   individually, add a "The Full Loop" section that traces the complete call
   end to end, referencing every concept by name in execution order. This is
   the section where the reader sees how all the pieces connect.

5. **Connect to the next layer.** The final section always points at the
   remaining inefficiency and what Layer N+1 addresses.

This arc appears at three levels: in the opening of `01_*.md`, in the opening
section of `summary.md`, and implicitly in the "What Comes Next" section. The
"Full Loop" tying section appears both as a section file (`0N_the_full_loop.md`)
and as "The Full Loop" in `summary.md`.

---

## The Full Loop Section

Every layer must have a "The Full Loop" section that traces one complete call
through the generate method from first line to last. It appears:

- As `0N_the_full_loop.md` — a dedicated section file placed after the last
  concept section and before "What Comes Next".
- As "## The Full Loop" in `summary.md` — placed after all concept sections
  and before "## What Comes Next".

### What it covers

The section traces the call in execution order, using short subsection headers
that map to the code's natural steps (e.g. "Step 1 — Tokenize",
"Step 2 — Prefill", "Step 3 — The Decode Loop", "Step 4 — Decode and Return").

Each step:
- Shows the relevant code block (the actual lines, not pseudocode).
- References the concept sections that explain it by name: "as section 03
  established...", "the `cumsum` fix from section 03...", etc.
- Explains how this step hands off to the next one.

The reader should finish this section with a clear mental model of the full
call and how all the independently explained pieces fit together.

### What it does not do

It does not re-explain concepts in detail — those belong in the concept
sections. It does not introduce new ideas. It is purely a connecting narrative
that uses every concept the reader has just learned and shows them working
together as a coherent system.

### Example structure

```markdown
# 0N — The Full Loop

[One-sentence framing: now that all parts are explained, trace a call end to end.]

---

## Step 1 — [First phase, e.g. Tokenize]

\`\`\`python
[relevant code block]
\`\`\`

[One paragraph: what this step produces and what the next step needs from it.]

---

## Step 2 — [Second phase, e.g. Prefill]

\`\`\`python
[relevant code block]
\`\`\`

[One paragraph connecting to the concepts explained in sections 02–0N.]

---

## Step 3 — [Main loop]

\`\`\`python
[relevant code block]
\`\`\`

[One paragraph naming each mechanism (by its section) and explaining the
handoff between steps inside the loop.]

---

## Step 4 — [Output]

\`\`\`python
[relevant code block]
\`\`\`

[One paragraph on what is returned, how metrics are computed, and what
server.py receives.]
```

---

## File Naming

```
lesson/
├── 00_outline.md
├── 01_the_decode_loop.md      # always: the full generate method, layer comparison
├── 02_[concept].md            # the new mechanism introduced by this layer
├── 03_[concept].md
├── ...
├── 0N_the_full_loop.md        # always second-to-last concept: tying everything together
├── 0N+1_whats_next.md         # always last: what the next layer changes
├── summary.md
└── sglang_reference.md
```

`01_the_decode_loop.md`, `0N_the_full_loop.md`, and `0N+1_whats_next.md` are
fixed. The middle sections vary by layer.

---

## Quick Checklist

Before marking a layer's lesson complete:

- [ ] `00_outline.md` has a Key Code Anchors table with accurate line numbers
- [ ] `summary.md` opens by showing Layer N-1 code then Layer N code
- [ ] Every summary section has at least one code block
- [ ] No bullet points in summary or section files
- [ ] Section 01 opens with the "before" (Layer N-1) then "after" (Layer N) loop
- [ ] Sections 02+ open by quoting the specific code they explain
- [ ] Sections 02+ have a short orientation paragraph tying the section to the full system loop
- [ ] `0N_the_full_loop.md` exists and traces the full call in step-by-step subsections
- [ ] "The Full Loop" section exists in `summary.md` between the last concept section and "What Comes Next"
- [ ] Section order matches outline order
- [ ] "What Comes Next" (last section file and last summary section) names the
      specific file and metric that Layer N+1 changes
