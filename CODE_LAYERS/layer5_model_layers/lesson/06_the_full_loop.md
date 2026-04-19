# 06 — The Full Loop

Sections 01 through 05 explained each component individually. This section traces one complete `generate_batch` call from server startup to the returned result dicts, naming every concept in the order it executes.

---

## Step 0 — Model Loading

Before any request arrives, `BatchedModel.__init__` calls:

```python
self.model = Qwen3ForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
```

`from_pretrained` resolves the model path, reads `config.json` into our `Qwen3Config` dataclass, builds the full architecture on CPU, casts to `bfloat16`, and streams weights from `model.safetensors` one tensor at a time into each parameter. The `lm_head` and `embed_tokens` weights share memory via the tie. These steps are covered in Layer 4; they carry over to Layer 5 unchanged. Every subsequent forward call goes through code we own.

---

## Step 1 — Tokenise

```python
input_ids, attention_mask, prompt_lens_list = self.tokenizer.prepare_batch(batch_messages)
prompt_lens = torch.tensor(prompt_lens_list, dtype=torch.long, device="cuda")
prefill_pos = (attention_mask.long().cumsum(-1) - 1).clamp(min=0)
# input_ids:      [B, max_prompt_len]
# attention_mask: [B, max_prompt_len]  1=real, 0=pad
# prefill_pos:    [B, max_prompt_len]  per-token position indices
```

`prepare_batch` formats each conversation with `apply_chat_template` and tokenises the batch with left-padding. The `cumsum` formula converts `attention_mask` into per-token position IDs: each real token gets the rank it would have in an independent B=1 run (0 for the first real token, 1 for the second, and so on), while padding positions receive 0. This corrects the RoPE encoding that would otherwise be wrong for shorter, left-padded prompts — the same fix carried from Layer 3.

---

## Step 2 — Prefill

```python
kv = KVCache()
logits = self.model(input_ids, attention_mask=attention_mask,
                    kv_cache=kv, position_ids=prefill_pos)
# logits: [B, max_prompt_len, vocab_size]
ttft_ms = round((time.perf_counter() - t_prefill) * 1000, 1)
```

`Qwen3ForCausalLM.forward` delegates immediately to `Qwen3Model.forward`. Inside:

```python
hidden    = self.embed_tokens(input_ids)              # [B, max_prompt_len, 1024]
cos, sin  = self.rotary_emb(hidden, prefill_pos)      # each [B, max_prompt_len, 128]
additive_mask = _build_additive_mask(
    attention_mask, q_len, past_len + q_len, ...
)                                                     # [B, 1, max_prompt_len, max_prompt_len]

for layer in self.layers:                             # 28 ×
    hidden = layer(hidden, cos, sin, additive_mask, kv)
```

`embed_tokens` maps each token ID to a 1024-dimensional vector. `RotaryEmbedding` (section 03) computes `cos` and `sin` once from `prefill_pos` — the outer product of `inv_freq [64]` with the position indices gives `freqs [B, max_prompt_len, 64]`, doubled to `[B, max_prompt_len, 128]`, then `.cos()` and `.sin()`. `_build_additive_mask` (section 04) constructs the `[B, 1, max_prompt_len, max_prompt_len]` mask combining upper-triangular `NEG_INF` for future positions and `NEG_INF` for padding positions.

Each of the 28 `Qwen3DecoderLayer` blocks (section 02) applies pre-norm with `input_layernorm`, then calls `Qwen3Attention` (section 05). Inside attention: `q_proj/k_proj/v_proj` project to `[B, max_prompt_len, 16, 128]` and `[B, max_prompt_len, 8, 128]` respectively; `q_norm` and `k_norm` (per-head RMSNorm, section 02) normalise each head's Q and K; after transposing to `[B, heads, max_prompt_len, 128]`, `apply_rotary_pos_emb` (section 03) encodes position via `q_rot = q*cos + rotate_half(q)*sin`; `kv.update(layer_idx, k, v)` appends K and V to the cache and returns the full accumulated tensors; `repeat_kv` expands K and V from 8 heads to 16 without copying; `F.scaled_dot_product_attention` computes `softmax(Q·Kᵀ/√128 + mask)·V`, dispatching to FlashAttention on CUDA. The decoder block then adds the attention residual, normalises with `post_attention_layernorm`, runs the SwiGLU MLP (`gate_proj` + `up_proj` gated through `silu`, projected down with `down_proj`, section 02), and adds the MLP residual.

After all 28 layers, `model.norm` (final RMSNorm) normalises `hidden [B, max_prompt_len, 1024]`, and `lm_head` projects to `logits [B, max_prompt_len, vocab_size]`. Back in `model_runner.py`:

```python
next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]
```

`logits[:, -1, :]` extracts the last-position logit for each row `[B, vocab_size]`. `sample_batch` returns a `[B]` tensor of first-generated token IDs. TTFT is recorded.

---

## Step 3 — The Decode Loop

```python
finished  = next_tokens == self.eos_id   # [B] bool
decode_step = 0

for _ in range(max_new_tokens - 1):
    if finished.all():
        break

    current = torch.where(finished.unsqueeze(1),
                          torch.full_like(..., self.pad_id),
                          next_tokens.unsqueeze(1))   # [B, 1]

    attention_mask = torch.cat(
        [attention_mask, torch.ones(B, 1, ...)], dim=1
    )   # [B, kv_len + 1]

    decode_pos = (prompt_lens + decode_step).unsqueeze(1)   # [B, 1]
    decode_step += 1

    logits = self.model(current, attention_mask=attention_mask,
                        kv_cache=kv, position_ids=decode_pos)
    next_tokens = sample_batch(logits[:, -1, :], temperature)   # [B]

    finished = finished | (next_tokens == self.eos_id)
```

On each step, finished requests have their input token replaced with `pad_id` via `torch.where` — their row stays in the batch at shape `[B, 1]` but produces no meaningful output. `attention_mask` grows by one column per step, keeping the mask shape consistent with the growing KV cache. `decode_pos [B, 1]` gives each request its own absolute position (`prompt_lens[i] + decode_step`) rather than a shared offset, ensuring correct RoPE encoding for sequences of different lengths (section 04).

The forward call is identical to the prefill call but with `q_len = 1`. Inside `Qwen3Model.forward`, `cos` and `sin` are `[B, 1, 128]`; `_build_additive_mask` produces a `[B, 1, 1, kv_len]` mask where the causal part is all-zeros (no future positions to mask for a single query token — section 04); `kv.update(layer_idx, k, v)` appends one new K/V row per layer to the cache; `repeat_kv` and SDPA run as before. `logits[:, -1, :]` is `logits[:, 0, :]` since `q_len = 1`, and `sample_batch` draws the next `[B]` tokens. `finished` accumulates with `|=` and the loop exits early when all requests have emitted EOS.

---

## Step 4 — Results

```python
texts   = self.tokenizer.decode_batch(generated)
tpot_ms = round((sum(step_times) / len(step_times)) * 1000, 1)

results = [
    {"text": texts[i], "prompt_tokens": prompt_lens_list[i],
     "completion_tokens": len(generated[i]),
     "latency_ms": latency_ms, "ttft_ms": ttft_ms, "tpot_ms": tpot_ms}
    for i in range(B)
]
```

`generated` is a list of B token-ID lists accumulated during the loop, excluding tokens produced after EOS. `decode_batch` converts each list to a string with the tokenizer's batch decoder. TPOT is the mean over `step_times`, covering only decode forward passes — prefill cost lives exclusively in TTFT. Each result dict uses individual `prompt_tokens` and `completion_tokens` counts but shares the `ttft_ms` and `tpot_ms` values, which are properties of the batched computation as a whole. `server.py` wraps each dict into a `GenerateResponse` Pydantic model without modification — the external API is unchanged from Layer 3.
