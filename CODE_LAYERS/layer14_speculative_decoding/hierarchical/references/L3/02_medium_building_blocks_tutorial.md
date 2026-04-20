# Speculative Decoding Tutorial (Building Blocks / Medium)

**Source:** https://medium.com/@buildingblocks/speculative-decoding-tutorial-007936be2bbb
**Author:** Building Blocks (Medium)
**Level:** L3 — Step-by-step greedy implementation in PyTorch + HuggingFace; bridges L2 intuition to code
**Access note:** Member-only article on Medium. A free account can read a limited preview.
**Why here:** Unique in the L3 tier for showing speculative decoding as *runnable user-land code* — Hugging Face `transformers` models, PyTorch tensors, and explicit accept/reject loops — without the complexity of an inference engine. This is "what lesson/04–06 looks like when you do it yourself from scratch with GPT-2."

---

## What this article covers (from the preview and abstract)

The article implements speculative decoding in the **greedy decoding setting** using:
- PyTorch
- Hugging Face Transformers library

### Scope

**Greedy speculative decoding:** guarantees the output is identical to what the target model would produce alone. Not sampling-based (that requires the full rejection sampling math from the Chen et al. paper; the article links to it for readers who want the sampling extension).

**Draft model:** a smaller model from HuggingFace (e.g. GPT-2 as draft for GPT-2-medium or GPT-2-large as target)
**Target model:** a larger model from the same family

### The implementation shows

1. **Draft loop:** run the draft model autoregressively to produce K tokens
2. **Batch verification:** pass the prompt + K draft tokens through the target model in one forward pass
3. **Accept/reject loop:** compare greedy argmax of target vs. draft token at each position; accept matches, reject on first mismatch
4. **Fallback token:** use the target model's prediction at the first rejected position
5. **KV cache handling:** how to manage KV caches for both models (the main source of bugs in real implementations)

---

## Why the greedy setting is the right L3 starting point

The greedy setting eliminates the probabilistic complexity of the acceptance criterion (the `min(1, p/q)` formula). In greedy:

```
Accept token t_i if:  argmax(target_logits[i]) == draft_token[i]
Reject if:            argmax differs → use argmax(target_logits[i]) as replacement
```

This is a deterministic rule. The correctness guarantee is exact: the output is **bit-for-bit identical** to greedy target-only decoding. No probability theory required.

Once the greedy case is clear, the sampling extension (Chen et al.) makes intuitive sense — it's the same loop but with `min(1, p/q)` replacing the equality check.

---

## What the code structure looks like (reconstructed from article description)

```python
def speculative_decode_greedy(draft_model, target_model, input_ids, K=4, max_new_tokens=50):
    """
    Greedy speculative decoding.
    Guarantees identical output to greedy target-only decoding.
    """
    generated = input_ids
    
    while len(generated[0]) < max_new_tokens:
        # Step 1: Draft phase — run small model K times autoregressively
        draft_tokens = []
        draft_input = generated
        for _ in range(K):
            with torch.no_grad():
                draft_out = draft_model(draft_input)
            next_tok = draft_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            draft_tokens.append(next_tok)
            draft_input = torch.cat([draft_input, next_tok], dim=1)
        
        draft_sequence = torch.cat(draft_tokens, dim=1)  # shape [batch, K]
        
        # Step 2: Verify phase — run target model on prompt + K draft tokens
        verify_input = torch.cat([generated, draft_sequence], dim=1)
        with torch.no_grad():
            target_out = target_model(verify_input)
        
        # target logits at positions: [last_generated, draft_0, ..., draft_{K-1}]
        # target_out.logits[:, len(generated)-1 : len(generated)+K, :]
        target_preds = target_out.logits[:, len(generated[0])-1 : -1, :].argmax(dim=-1)
        # target_preds[i] = what target would have predicted at position i
        
        # Step 3: Accept/reject loop
        accepted = 0
        for i in range(K):
            if target_preds[:, i] == draft_sequence[:, i]:
                accepted += 1
            else:
                break  # first mismatch — stop
        
        # Step 4: Append accepted tokens + fallback token
        new_tokens = draft_sequence[:, :accepted]
        fallback = target_preds[:, accepted:accepted+1]
        
        generated = torch.cat([generated, new_tokens, fallback], dim=1)
        
        # If all K tokens accepted, also append target's prediction for position K
        # (the "bonus token" from the K+1 position — this is what gives us K+1 tokens
        #  when acceptance is perfect)
    
    return generated
```

This matches the structure in Layer 14's `_draft_extend()` → `_verify_extend()` → `_accept_reject()` pipeline exactly.

---

## Layer 14 mapping

| Tutorial concept | Layer 14 equivalent |
|-----------------|---------------------|
| Draft loop (K autoregressions) | `ModelRunner.forward_draft_extend()` in `spec_runner.py` |
| Batch verify on target | `ModelRunner.forward_verify_extend()` |
| Accept/reject greedy check | `_accept_reject()` with exact-match criterion |
| Fallback / replacement token | The K+1th token from target logits (bonus token) |
| KV cache for draft tokens | `DraftReq.draft_kv_pool` pages |
| Discarding rejected KV pages | `_kv_rewind()` call in `accept_reject_rewind` |

The tutorial intentionally omits KV rewind complexity (it's user-land, not an inference engine). Layer 14 adds that complexity because SGLang must manage GPU memory explicitly. Reading the tutorial first, then Layer 14, shows exactly what problem KV rewind solves.

---

## Reading strategy for L3 readers

1. **Read this article first** — implement the code yourself with GPT-2 from HuggingFace. ~30 min.
2. **Then read Layer 14 `lesson/04_draft_extend.md`** — now you understand the structure; the lesson explains why SGLang's version is more complex (batching, KV pool management, request scheduling).
3. **Then read Layer 14 `lesson/06_accept_reject_rewind.md`** — after coding the greedy case, the sampling criterion and KV rewind make sense as extensions.

---

## Limits of this article (for book context)

- Greedy only — does not implement the probabilistic rejection sampling required for nucleus/top-p sampling. For that, see the L2 articles on the Chen et al. criterion.
- No KV cache rewind — tutorial recomputes from scratch on each step; production engines must reuse/rewind KV pages.
- Single sequence, small models — no batching logic, no continuous batching, no multi-request scheduling.
- No benchmark numbers — the article focuses on correctness of the algorithm, not throughput measurements.
