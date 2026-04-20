# L1 References — Orientation level

Five articles for readers who want to understand **what problem speculative decoding solves and why it matters** — without code, math, or engine internals.

**Persona:** Curious non-builder (product managers, adjacent engineers, developers who use LLM tools but don't build serving stacks).

**Success:** Reader can state what problem speculative decoding solves, what cost it trades, and where it is already used in production — in plain language.

---

## Files in this folder

| File | Source | Best for |
|------|--------|----------|
| `01_chris_thomas_speculative_decoding.md` | christhomas.co.uk | Best entry point. "Junior/senior developer" analogy. Practical VS Code demo with Qwen Coder. |
| `02_nvidia_speculative_decoding_intro.md` | NVIDIA Developer Blog | Authoritative GPU framing. "Chief scientist / lab assistant" analogy. Also covers EAGLE-3 and MTP for readers who want to go further. |
| `03_google_research_looking_back.md` | Google Research Blog | Written by the original authors. Covers the "why we built this" story, the speculative sampling theory, and confirmed production use in Google Search AI Overviews. |
| `04_nebius_moe_speculative_decoding.md` | Nebius Blog | Best "products fail even when benchmarks look fine" article. P99 tail latency, cascaded systems, why throughput averages mislead. L1–L2 level. |
| `05_google_cloud_five_techniques.md` | Google Cloud Blog | Landscape view: where speculative decoding fits alongside routing, disaggregation, quantization, and prefix caching. Best for readers who want the "whole picture" quickly. |

---

## Reading order suggestion

1. **Start with `01`** (Chris Thomas) — the analogy is the most natural.
2. **Then `03`** (Google Research) — the original authors explain *why the math works*, in plain language.
3. **Then `05`** (Google Cloud five techniques) — understand how speculative decoding fits alongside other inference optimizations.
4. **`04`** (Nebius) if the reader will work with production systems or long-context workloads.
5. **`02`** (NVIDIA) if the reader wants to understand EAGLE-3 or MTP before reading further.

---

## Common limit to name for L1 readers

All five articles treat speculative decoding as a black box from the user's perspective. The "draft tokens are rejected" step is where implementation complexity lives — none of these articles explain KV rewind, page management, or the mirroring invariant. That is L3 territory (`lesson/04`–`06`).

The key metaphor limit to note: accepted or rejected draft tokens are **never visible to the user** — unlike a junior developer's edits, which you can see and choose to accept. The acceptance decision is internal and automatic.

---

## See also

- `../REFERENCES.md` — complete reference list with level labels.
- `../references/L2/` — definition repair + motivation level articles (9 files available).
- `../../lesson/01_from_one_to_n_plus_one.md` — L3 spine: the formal "one token vs N+1" argument.
