# 05 — The Decode Loop

## The Complete `NaiveModel.generate`

Everything from the previous three sections comes together in one method. Here is the full generate loop from `model.py`, unabridged:

```python
def generate(
    self,
    messages: list[dict],
    max_new_tokens: int = 64,
    temperature: float = 1.0,
) -> dict:
    t0 = time.perf_counter()

    # --- Step 1: apply chat template and tokenize ---
    formatted = self.tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
    prompt_tokens = ids.shape[1]

    # --- Step 2: manual decode loop ---
    generated_ids: list[int] = []
    step_times: list[float] = []

    for step in range(max_new_tokens):
        t_step = time.perf_counter()

        with torch.no_grad():
            out = self.model(input_ids=ids, use_cache=False)

        next_token_logits = out.logits[0, -1, :]
        next_token_id = self._sample_next_token(next_token_logits, temperature)

        step_times.append(time.perf_counter() - t_step)

        if next_token_id == self.eos_id:
            break

        generated_ids.append(next_token_id)
        ids = torch.cat([ids, torch.tensor([[next_token_id]],
                         dtype=torch.long, device=self.device)], dim=1)

    # --- Step 3: decode ---
    text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

    latency_ms = round((time.perf_counter() - t0) * 1000, 1)
    ttft_ms = round(step_times[0] * 1000, 1) if step_times else 0.0
    decode_steps = step_times[1:]
    tpot_ms = round((sum(decode_steps) / len(decode_steps)) * 1000, 1) \
               if decode_steps else ttft_ms

    return {
        "text": text,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": len(generated_ids),
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "tpot_ms": tpot_ms,
    }
```

This is the complete, working inference engine for Layer 1. Everything that follows is an explanation of each part.

## Line-by-Line Walkthrough

### Step 1 — Format and Tokenize

```python
formatted = self.tokenizer.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
)
ids = self.tokenizer(formatted, return_tensors="pt").input_ids.to(self.device)
prompt_tokens = ids.shape[1]
```

This was covered in the Layer 0 lesson. The messages list is serialised into the formatted string the model was trained on, then tokenized into a `LongTensor` of shape `[1, prompt_len]`. `prompt_tokens` is saved so we know how many tokens came from the user versus how many were generated. The variable is named `ids` rather than `input_ids` because it will grow as we append tokens — it is not just the input any more, it is the living sequence.

### Initialising the Accumulators

```python
generated_ids: list[int] = []
step_times: list[float] = []
```

`generated_ids` collects the token IDs produced during generation. We keep these separate from `ids` — the growing tensor — because at the end we only want to decode the tokens the model generated, not the prompt tokens that were already in the sequence. `step_times` records how long each forward pass took, which is used to compute TTFT and TPOT.

### The Loop

```python
for step in range(max_new_tokens):
```

`max_new_tokens` is the upper bound on how many tokens the model is allowed to generate. The loop will exit early if the model produces an end-of-sequence token before reaching this limit. This guards against runaway generation: without it, a model that never emits EOS would loop forever.

### The Forward Pass

```python
with torch.no_grad():
    out = self.model(input_ids=ids, use_cache=False)
```

On step 0 this is the prefill: `ids` contains only the prompt tokens and the model processes all of them in parallel. On step 1 onwards this is a decode step: `ids` contains the prompt plus all tokens generated so far, and the model reprocesses everything from scratch. `use_cache=False` prevents HuggingFace from quietly maintaining its own internal cache, keeping the computation honest and visible.

### Extracting the Next Token Logits

```python
next_token_logits = out.logits[0, -1, :]
```

`out.logits` has shape `[1, current_seq_len, vocab_size]`. We slice `[0, -1, :]` to get a single vector of shape `[vocab_size]` — the scores at the last position, which is the position the model is predicting a next token for. All other positions' logits are discarded.

### Sampling

```python
next_token_id = self._sample_next_token(next_token_logits, temperature)
```

This calls the sampling function from section 04. It receives the `[vocab_size]` logit vector and returns a single integer — the token ID the model has selected as its next output. Depending on `temperature`, this is either a greedy argmax or a multinomial draw from the softmax distribution.

### Timing the Step

```python
step_times.append(time.perf_counter() - t_step)
```

The step timer started just before the forward pass and stops here, after sampling. This captures the full cost of one decode step: GPU forward pass plus CPU sampling. `step_times[0]` will be the prefill cost; `step_times[1:]` will be the decode costs.

### The EOS Check

```python
if next_token_id == self.eos_id:
    break
```

`self.eos_id` is set in `__init__` from `self.tokenizer.eos_token_id`. When the model generates the end-of-sequence token it is signalling that it considers the response complete. We break immediately rather than appending the EOS token to `generated_ids` — the caller should receive clean text, not a sequence that ends with a stop marker. This check happens before the append so EOS is timed but never included in the output.

### Growing the Sequence

```python
generated_ids.append(next_token_id)
ids = torch.cat([ids, torch.tensor([[next_token_id]],
                 dtype=torch.long, device=self.device)], dim=1)
```

Two things happen here. First, the token ID is appended to `generated_ids`, the Python list that accumulates the output. Second, `ids` — the tensor being passed to the model — is extended by one column. `torch.tensor([[next_token_id]])` creates a `[1, 1]` tensor (batch size 1, sequence length 1) and `torch.cat(..., dim=1)` concatenates it along the sequence dimension, extending `ids` from shape `[1, N]` to `[1, N+1]`. On the next iteration the model will receive this longer sequence.

### Step 3 — Decoding the Output

```python
text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
```

Once the loop exits — either by hitting EOS or reaching `max_new_tokens` — `generated_ids` contains the integer IDs of every token the model produced. `tokenizer.decode` converts them back to a UTF-8 string. `skip_special_tokens=True` strips any residual structural tokens like `<|im_end|>` so the caller receives clean text.

Notice we decode `generated_ids`, not `ids`. `ids` is the full sequence including the original prompt tokens; `generated_ids` is only the tokens the model generated. Decoding the prompt would give back the formatted conversation header which the caller does not want.

### Computing the Metrics

```python
latency_ms = round((time.perf_counter() - t0) * 1000, 1)
ttft_ms = round(step_times[0] * 1000, 1) if step_times else 0.0
decode_steps = step_times[1:]
tpot_ms = round((sum(decode_steps) / len(decode_steps)) * 1000, 1) \
           if decode_steps else ttft_ms
```

`latency_ms` is the total wall time from the start of `generate` to here, covering tokenization, all forward passes, sampling, and decoding. `ttft_ms` is just `step_times[0]` converted to milliseconds — the cost of the first step, which is the prefill. `tpot_ms` is the average of all subsequent step times — the cost of a pure decode step, excluding the prefill overhead. These two metrics are discussed in detail in the next section.

The method returns all of this as a plain dictionary, which `server.py` wraps directly into a `GenerateResponse` Pydantic model.
