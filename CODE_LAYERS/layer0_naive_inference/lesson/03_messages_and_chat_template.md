# 03 — Messages, Roles, and the Chat Template

## The Problem With Raw Text

A base language model — one trained purely to predict the next token from a raw text corpus — will complete any prompt you give it. Feed it the string "The capital of France is" and it will continue naturally: "Paris, which has been the seat of government since...". That works well for text completion tasks. But if you feed it a question directly:

```python
output = model.generate(tokenizer("What is the capital of France?",
                                  return_tensors="pt").input_ids)
```

the model treats it as a document to continue, not a question to answer. It might generate more questions, or continue as if writing a geography quiz. It has no concept of being "asked" something — it just sees a sequence of tokens to extend.

This is the difference between **text completion** and **chat completion**. Text completion is the raw capability: given a prefix, extend it. Chat completion is a behaviour layered on top: given a conversation, reply as an assistant. That behaviour is taught through fine-tuning on conversations that are formatted with explicit speaker markers, so the model learns to recognise when it is the assistant's turn to speak and what register to use. The specific string format used during fine-tuning is the model's **chat template**, and you must use the same format at inference time or the model's behaviour is undefined.

## The Messages Schema

To represent a conversation in a structured way, the standard format is a list of message objects, each with a `role` and `content`. There are three roles. `"user"` is the human sending a message. `"assistant"` is the model replying. `"system"` is an optional instruction that sets the overall behaviour of the assistant — it is written by the developer, not the end user, and typically appears once at the start of the conversation to say things like "you are a concise assistant" or "you only answer questions about cooking".

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user",   "content": "What is the capital of France?"},
]
```

Here the developer has told the model how to behave via the system message, and the user has asked a question. The model will generate the assistant's reply. This is the standard format that LLM serving APIs use, and the Layer 0 server accepts exactly this format from its clients.

A single-turn conversation omits the system message entirely and contains just one user message:

```python
messages = [{"role": "user", "content": "What is 2 + 2?"}]
```

A multi-turn conversation includes prior assistant responses so the model has the full context of the exchange. This is worth thinking about carefully: the model has no persistent memory between calls. Every time you send a request, you send the entire conversation history as a flat list of messages. The server does not store any state on your behalf. If the user asks a follow-up question, the client is responsible for appending the assistant's previous reply and the new user message before sending the next request. The model then reads the whole history from scratch and generates the next reply in context.

```python
messages = [
    {"role": "user",      "content": "Name three planets."},
    {"role": "assistant", "content": "Mars, Venus, and Jupiter."},
    {"role": "user",      "content": "Which of those is the largest?"},
]
```

In this example the model receives all three turns at once. It can see that it already said "Mars, Venus, and Jupiter" and can now answer "Which is the largest?" accordingly. Without the assistant turn included in the list, the follow-up question would arrive without context and the model would have no way to know what "those" refers to.

## Applying the Chat Template

The messages list is a structured Python object. Before it can be processed by the model it must be serialised into a single string according to the model's chat template. The tokenizer provides `apply_chat_template` for exactly this purpose:

```python
formatted = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

`tokenize=False` tells the method to return a plain string rather than token IDs, so we can inspect it before tokenizing. `add_generation_prompt=True` appends the opening marker for the assistant's turn — for Qwen3 this is `<|im_start|>assistant\n` — which signals to the model that it should now generate a reply. Without this marker the model would not know its turn has begun and might generate user-side text instead.

`enable_thinking=False` is specific to Qwen3. Qwen3 was trained with a "thinking" mode in which the model first emits a `<think>...</think>` block containing its chain-of-thought reasoning before giving its final answer. That mode is valuable for hard reasoning tasks but adds latency and verbosity. Setting `enable_thinking=False` suppresses it, making Qwen3 behave like a standard assistant model.

The resulting `formatted` string for the two-message example above looks approximately like this:

```
<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
```

The model sees this entire string as its context. The `<|im_start|>` and `<|im_end|>` tokens are special tokens in Qwen3's vocabulary; the model was trained to understand that text between them belongs to the indicated role, and that text after the final `<|im_start|>assistant\n` is what it should generate. When the model finishes its reply, it emits `<|im_end|>` to close its own turn — this is the assistant's end-of-turn token, and it is what the generate loop detects to know the response is complete, in addition to the EOS token. When you call `tokenizer.decode(..., skip_special_tokens=True)`, both of these structural markers are stripped so the caller receives clean text.

Once formatted, the string is tokenized in the usual way:

```python
input_ids = tokenizer(formatted, return_tensors="pt").input_ids.to("cuda")
```

This produces a `torch.LongTensor` of shape `[1, sequence_length]` ready to be passed to `model.generate`. The `return_tensors="pt"` argument requests a PyTorch tensor rather than a plain Python list, and `.to("cuda")` moves it to the same device as the model weights.

## Why the Template Matters

Using the wrong template — or no template at all, passing the raw messages as plain text — produces subtly wrong behaviour. The model may not recognise the role boundaries, may not know when to stop generating, or may generate in the wrong register. Every model family (Llama, Gemma, Mistral, Qwen, Phi) has a different chat template baked into its tokenizer. The `apply_chat_template` method abstracts away these differences: the same calling code works correctly for any model as long as you load the tokenizer from the same checkpoint as the model.

This is the same call SGLang makes internally in `serving_chat.py` when it processes an incoming OpenAI-compatible `/v1/chat/completions` request. The client speaks the OpenAI schema; the server applies the model's chat template; the model sees a correctly formatted string.
