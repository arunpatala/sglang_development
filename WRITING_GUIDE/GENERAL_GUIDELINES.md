# General writing guidelines — explanatory, empathetic, personality-forward

These apply across chapters (inference engines, KV, scheduling, speculation, and so on). They are **style and pedagogy** rules, not correctness rules for the code.

---

## 1. Lead with the human problem, not the API

Open each chapter (and each hard section) with a tension the reader already feels: latency, memory pressure, fear of subtle bugs, “why do we need a rewind?”, and so on. Introduce mechanisms and names **after** stakes are clear.

Documentation starts at function boundaries; explanatory books start at **why this moment hurts** without the idea you are about to teach.

---

## 2. Teach invariants before implementations

State one to three crisp truths the reader can use to **predict** system behavior (“after this step, these lengths or pointers must agree…”). Code should **prove** the invariant, not **discover** it for the first time.

If the reader cannot say the invariant in one sentence, the section is still organized like a spec.

---

## 3. Stage wrong intuitions on purpose

For each non-obvious idea, name the **tempting mistake** (“I can keep the extra KV”), give **one sentence** on what breaks, then show the fix. Empathy is often normalizing confusion **before** resolving it—not only listing correct facts.

---

## 4. Use a bounded code diet in the chapter body

Code-driven learning is not the same as code-first layout. Prefer short pseudocode or small excerpts (often on the order of **5–15 lines**) that carry the core idea; put long listings in an appendix, a “repo track,” or a linked file.

Readers are rarely afraid of code; they are afraid of **unbounded code**—every line feeling equally examinable on first pass.

---

## 5. Give explicit “first read / second read” guidance

Tell readers what to skim or defer (kernel details, packing arithmetic, vendor-specific APIs) **without** implying those parts are unimportant. This reduces overwhelm more reliably than shortening the truth.

---

## 6. Write in a directed voice (“you”), with checkpoints

Use questions and small pauses: “Pause: what should the committed length be after this step?” That turns passive documentation into a **conversation**. Personality shows up less in jokes and more in **rhythm**: callbacks, honest limits, small celebrations of insight.

---

## 7. One worked example beats three paragraphs of generality

Walk a **tiny concrete trace** (tokens, lengths, pool pages, indices) on paper before generalizing. Concrete numbers are often more empathetic than abstract `N`.

---

## 8. Separate “why it is true” from “why we implemented it this way”

Keep correctness and math in one lane; engineering tradeoffs (paging, batching, kernels, memory layout) in another. Mixing them reads like an internal spec and exhausts newcomers even when every sentence is accurate.

---

## 9. End sections with a sanity test, not only a summary

Summaries read like documentation. **Sanity tests** build confidence: “If you misunderstood X, you would observe Y” or “After reading this, you should be able to fill in this table without the code.”

---

## 10. Let authorial point of view show—within clear constraints

It is fine to prefer one mental model, warn about production pitfalls, or admit what you deferred. Readers forgive limits if they sense a **mind** behind the page, not a neutral merger of threads.

---

## Cross-cutting habits (short)

- **Headline before hair**: the conceptual headline belongs in the first screen of a section; implementation detail supports it.
- **Side effects are outcomes**: when a step has a crucial side effect (e.g. optimistic KV writes), state it as a first-class outcome, not only as a postscript to machinery.
- **“Actually …” is a rewrite signal**: when you feel the urge to correct mid-paragraph, two mental models collided—reconcile them explicitly (two-column “reader thinks / code means / therefore”).
- **Metrics need footguns**: for every metric, say what it measures, what it does **not** measure, and a common misinterpretation.

These general rules pair with `SPECIFIC_GUIDELINES.md`, which grounds the same ideas in concrete lesson patterns (Layer 14 speculative decoding).
