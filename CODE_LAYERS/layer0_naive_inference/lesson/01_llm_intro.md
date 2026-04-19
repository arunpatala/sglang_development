# 01 — What Large Language Models Are and What They Do

## The Capability

A large language model is a neural network trained on an enormous corpus of text — books, code, web pages, scientific papers — with a single objective: given a sequence of tokens, predict the next token. That is the whole training task. Yet from this deceptively simple objective, something remarkable emerges. A model trained long enough and large enough on diverse-enough data learns to reason through multi-step problems, summarise long documents, translate between languages, write working code, answer factual questions, and hold extended conversations. These capabilities are not hand-coded rules — they fall out of the training process itself.

What you can do with a well-trained LLM today spans a wide range: ask it to explain a concept at several levels of depth, have it debug a function, use it to draft and then revise an email, or use it as the reasoning core of an autonomous agent. A single set of weights handles all of these tasks, switching between them based purely on what the input text says.

## What This Chapter Covers

This chapter builds the simplest possible version of an LLM inference system, end to end. By the end of it you will have done the following things.

You will load a real transformer model — Qwen3-0.6B — from disk into GPU memory, along with its tokenizer. You will understand what that loading process actually does: which files it reads, how much memory the weights consume, and why the model stays loaded in memory for the lifetime of the server rather than being loaded fresh for each request.

You will format a conversation using the model's chat template and tokenize it into the integer sequence the model actually processes. You will run the model's generate loop, watch it produce tokens one at a time, and decode the result back into text. Every line of code in that pipeline will be explained.

You will then wrap that pipeline in an HTTP server using FastAPI, exposing an endpoint that accepts conversations in the OpenAI messages format and returns generated text. You will see how the server handles the model loading, routing, request validation, and response serialisation.

Finally, you will measure the server's performance against a real-world dataset using a benchmark script, and you will have concrete numbers — output tokens per second, average latency, request rate — that establish the baseline every subsequent chapter improves upon.

## The Transformer Architecture (Just Enough)

The model used throughout this curriculum is Qwen3-0.6B, a transformer-based decoder-only language model. A full treatment of the transformer is not required here, but two ideas are worth having before you see the code.

The first is that for the purposes of this chapter, the model is a black box with a simple contract: text goes in, text comes out. You send it a conversation and it replies. The internal machinery — layers, attention, probability distributions — will be opened up progressively as the curriculum builds. For now, the interface is all that matters.

The second is a simple memory fact. Qwen3-0.6B has 600 million parameters. Stored at `bfloat16` precision (2 bytes per value), that is roughly 1.2 GB of GPU VRAM just for the weights. This is why model loading takes several seconds at startup, and why loading the model once and keeping it in memory for the lifetime of the server — rather than reloading it for each request — is the only practical approach.
