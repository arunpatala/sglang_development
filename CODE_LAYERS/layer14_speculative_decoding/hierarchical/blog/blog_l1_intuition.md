# How Your AI Gets Faster Without Getting Worse

*For curious builders, product people, and anyone who's wondered why "better AI" sometimes feels slower.*

---

There's a frustrating pattern in AI-powered products. You upgrade to a bigger, smarter model and the answers get better — but the responses slow down. Users notice. Completion lags. The product starts to feel less snappy, even though it's technically more capable.

The obvious fix is better hardware. But even on the fastest GPUs available today, large language models have a fundamental problem that more compute alone won't solve. Understanding that problem is the first step to understanding why speculative decoding matters.

---

## The encyclopedia problem

Every time an LLM generates a word, it reads its entire set of parameters from memory. Not the relevant section — all of them. For a large model, that's hundreds of gigabytes to a terabyte of data, loaded for every single word.

It gets worse: because each word depends on the one before it, the model can't generate word two until word one is done. Everything is sequential.

Modern GPUs are extremely good at math — they can perform hundreds of trillions of arithmetic operations per second. But reading data from memory is much slower. The model's arithmetic circuits sit mostly idle, waiting for memory to deliver the next round of weights. The bottleneck isn't how smart the GPU is. It's how fast memory can keep up.

This is why powerful hardware alone doesn't fix slow generation. You can have more compute, but if you're bottlenecked by memory reads, most of that compute is wasted.

---

## The insight that changes everything

The Google Research team that invented speculative decoding made two observations that open a new path.

**First:** not every word is equally hard to generate.

> *"What is the square root of 7? The square root of 7 is 2.646."*

Generating "7" for the second time is trivial — it's just copying from the context. Generating "2.646" requires the model to recall or compute a fact. A small, less powerful model can handle the easy cases. It only falls short on the hard ones.

**Second:** because memory is the bottleneck, spare compute is sitting there unused. Running more arithmetic doesn't slow down memory reads. So you can run extra work in parallel, for free, as long as you do it cleverly.

Combine the two: use a small, fast model to handle the easy words in parallel, and let the large model focus only on verification and correction. The large model's quality is preserved. The small model's speed is harvested.

That's the core of speculative decoding.

---

## The junior developer and the senior developer

The clearest way to think about it: two developers working together.

A **junior developer** types quickly. They draft code suggestions rapidly — not always perfect, but fast and usually reasonable. A **senior developer** knows exactly what should be written. They're slower to type but never wrong.

In a naive world, the senior writes every line of code themselves. That's baseline LLM inference: one giant model, doing everything, one token at a time.

Speculative decoding changes the workflow. The junior drafts a block of several lines quickly. The senior reviews it: for each line, they ask "would I have written exactly this?" If yes, it's accepted and they move on. If no, they fix it from the first mistake and continue from there.

The junior's output is **never shown to the user directly**. The user always sees the senior's authoritative final version. But the senior's time is used much more efficiently — reviewing a block of draft text rather than composing each line from scratch.

In practice, two models replace the two developers:
- A **smaller, faster model** (the junior) proposes several candidate tokens ahead.
- A **larger, more capable model** (the senior) verifies them in one pass and accepts what matches.

The result: **2–4× faster output with identical quality** to the large model working alone.

---

## What "identical quality" actually means

This is worth being precise about, because it sounds like magic.

The large model remains the **final authority** on every token. The verification step isn't editorial — it's mathematical. The large model runs its own prediction at every position and checks whether the draft matches what it would have produced. If yes, it's accepted. If no, the large model's own prediction is used instead.

Nothing is approximated. Nothing is skipped. The output you receive is exactly what you would have gotten if the large model had generated every token itself — just produced much faster.

This was confirmed in production: **Google AI Overviews in Google Search** uses speculative decoding today. The original authors describe *"remarkable speed-ups in inference, while maintaining the same quality of responses."* It scales to real traffic. The quality doesn't degrade.

---

## It's already in your hands

Speculative decoding isn't a research curiosity. It's shipped.

- **Google Search AI Overviews** uses it in production at scale.
- **LM Studio** added it in a recent release — local users can turn it on today.
- Tools like `llama.cpp` support running a draft and target model together: a 3B model drafting, a 14B model verifying, on a mid-range machine, with 2–4× faster code completions in your editor.

What used to require a tradeoff — *do you want the fast model or the accurate one?* — now isn't a tradeoff at all.

---

## One important correction to the analogy

The junior/senior framing is useful, but it breaks in one specific way worth naming.

In real code review, you can see the junior's suggestions. You choose which to accept. The process is editorial.

In speculative decoding, **the user never sees the draft**. The acceptance or rejection of draft tokens happens internally, automatically, before any output reaches you. It isn't a judgment about quality — it's a check of whether the draft token is exactly what the large model would have produced.

This means rejected draft tokens don't represent "worse answers." They're discarded silently. The model doesn't degrade or compromise when it rejects a draft — it just uses its own output instead.

The user experience: responses come faster, in short bursts rather than word by word. Quality is unchanged. The mechanism is invisible.

---

## The shape of what's to come

Speculative decoding as described here — one draft model, one target model, linear verification — is the baseline. The field has kept going.

More sophisticated variants exist: systems that use the target model's own internal states to self-draft (no separate model needed), tree-shaped candidate sets that verify many possible continuations in one pass, and prediction heads baked into the model architecture itself. The mechanism gets more elaborate, but the core idea stays the same: **draft cheap, verify in parallel, commit exactly what the target would have produced.**

For readers who want to go deeper on the mechanism — the step-by-step algorithm, what acceptance rates mean, how to configure it in practice — that's the L2 blog. For the full algorithm, hardware analysis, and production architecture: L3.

> *"The key insight isn't just about making things faster — it's about making better use of the computing resources we already have."* — Chris Thomas

---

*Sources: [Chris Thomas (2025)](https://christhomas.co.uk/blog/2025/02/16/speculative-decoding-using-llms-efficiently/), [Google Research (Dec 2024)](https://research.google/blog/looking-back-at-speculative-decoding/), [NVIDIA Developer Blog (Sep 2025)](https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/), [Nebius Blog](https://nebius.com/blog/posts/moe-spec-decoding)*
