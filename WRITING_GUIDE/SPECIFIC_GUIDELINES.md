# Specific writing guidelines — grounded in Layer 14 lessons

These guidelines are **actionable editorial checks** tied to patterns in:

`CODE_LAYERS/layer14_speculative_decoding/lesson/`

They implement the ideas in `GENERAL_GUIDELINES.md` against real chapter shapes (mirroring, verify extend, accept/reject, rewind, statistics). Use them as a **lint list** when revising any lesson that resembles Layer 14: dual runners, speculative steps, KV bookkeeping.

---

## 1. Turn definitions into “predictable behavior” first

**Pattern:** `03_draft_kv_mirroring.md` opens by defining *KV mirroring* as a requirement (draft must see the same committed context as the target). That is correct; a learner still needs **what goes wrong** if mirroring fails.

**Do:** Immediately after a definition like mirroring, add:

- one sentence **failure mode** (e.g. draft prefix diverges from target → proposals drift → acceptance collapses), and  
- one sentence **observable symptom** (e.g. high reject rate not explained by model mismatch alone).

**Why:** Definitions read like docs; *failure → symptom → invariant* reads like teaching.

---

## 2. Promote invariants to a “contract box,” separate from structs

**Pattern:** `03_draft_kv_mirroring.md` has a strong **Mirroring Invariant** section (e.g. length equalities and what each pool must cover). That is the spine. The `DraftReq` struct is how you **encode** the contract.

**Do:** Put the invariant in a scannable box **before** the class / field listing. Keep code as “encoding,” not as the first encounter with the idea.

---

## 3. When you write “Actually …,” reconcile models explicitly

**Pattern:** `05_verify_extend.md` explains `kv_committed_len`, then corrects with language like **“Actually `kv_committed_len = len(req.output_ids) - 1` …”**. The word *Actually* usually signals two colliding mental models mid-paragraph.

**Do:** Replace with a tiny reconciliation table:

| Reader might think | In this codebase it means | Therefore … |

**Why:** Learners learn fastest at the **collision site**; hiding it inside prose feels like the author moving the goalposts.

---

## 4. Make indexing conventions impossible to misread (one canonical map)

**Pattern:** `05_verify_extend.md` — **Logit Alignment**: `logits[b, j, :]` vs verifying `d_{j+1}`, plus conditions involving `j` and `j-1`. Easy to transpose on first read.

**Do:** Add one table for the whole chapter convention, for example:

| Index `j` | Last token “seen” for that logit row | Compared to | Row used in code |

Keep any code snippets **row-aligned** to that table.

**Why:** Offset bugs are where readers quit; tables are cheaper than rereading three paragraphs.

---

## 5. Lead “parallel verify” with the reader’s doubt, then discharge it

**Pattern:** `05_verify_extend.md` — **Why This Is Safe to Batch**: causal masking shows packed extend matches autoregressive semantics. This is strong content but can read like a spec clause if it arrives cold.

**Do:** Open with an explicit reader question, e.g. “Isn’t batched verify cheating compared to one-token decode?” Then the causal-mask argument lands as **relief**, not as homework.

---

## 6. Separate “math intuition” from “math precision” with clear signage

**Pattern:** `01_from_one_to_n_plus_one.md` gives a quick `1 + N·p` style story, then contrasts with a more exact expectation when acceptance is position-dependent. Both are valuable; readers often think they misunderstood when numbers disagree.

**Do:** Label the approximation as a **model** (“back-of-envelope throughput story”) and the refinement as an **audit tool** (“what changes when errors compound along the draft chain”).

---

## 7. Teach side effects as first-class outcomes (not postscripts)

**Pattern:** `05_verify_extend.md` — **K/V Written into the Target Pool**: verify writes **all** N+1 KV positions, including branches that may be rejected; rewind exists because of that optimism.

**Do:** Use a bold **claim → consequence** sequence, e.g. “We materialize a branchy future in KV; acceptance chooses a prefix; rewind deletes alternate branches.” Then point to the rewind chapter.

**Why:** Readers remember **causal stories** about state better than a list of calls in execution order.

---

## 8. Make “bonus token needs draft KV” a before/after story

**Pattern:** `03_draft_kv_mirroring.md` — **What Happens to Draft KV During Acceptance**: the bonus was not produced by the draft, so the draft must run an extra decode to restore mirroring. This is one of the most teachable moments in the layer.

**Do:** Show two snapshots: **KV / sequence coverage before bonus repair** vs **after** (`decode_step_single` or equivalent). Function names are secondary to the **state transition**.

---

## 9. Flag measurement footguns the moment you introduce metrics

**Pattern:** `07_statistics.md` — counters are **lifetime** aggregates across requests; that changes how `acceptance_rate` and `tokens_per_step` should be interpreted.

**Do:** For every metric block, add three lines:

1. what it measures,  
2. what it does **not** measure,  
3. a common misread (“not per-request fairness,” “not a latency histogram,” etc.).

---

## 10. End the hardest sections with a paper trace, not a summary

**Pattern:** `06_accept_reject_rewind.md` interleaves accept logic, commit, target rewind, draft rewind, stale bytes inside pages, and stats—dense and faithful.

**Do:** Close with a **short worked trace** using your own symbols (`N`, `accept_len`, which pages freed, final `output_ids`). Summaries feel like documentation; traces feel like a teacher checking understanding.

---

## Optional: one “style reference” rewrite target

Pick one dense subsection (for example **Logit Alignment** plus the `kv_committed_len` clarification in `05_verify_extend.md`) and rewrite it as the canonical example of:

- doubt-first opening,  
- reconciliation table,  
- minimal code.

Point other chapters at that example: “write it like the reference patch for 05.”
