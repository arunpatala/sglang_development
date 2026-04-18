"""
Test SGLang changes using the offline Engine (no HTTP server needed).
Run directly: .conda/bin/python scripts/engine_test.py

The Engine runs the model in-process — no restart needed between edits
to scheduler/model-runner code. For http_server.py changes, use serve_qwen3.sh.
"""

import os

# Use cached model, no network
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Add conda bin to PATH so ninja is found during FlashInfer JIT
CONDA = os.path.expanduser(
    "/home/arun/PROJECTS/sglang_development/.conda"
)
os.environ["PATH"] = f"{CONDA}/bin:{os.environ.get('PATH', '')}"

MODEL = os.path.expanduser(
    "~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots"
    "/c1899de289a04d12100db370d81485cdf75e47ca"
)

import sglang as sgl


def apply_chat_template(tokenizer, user_message: str, thinking: bool = False) -> str:
    """Format a user message using the model's chat template.
    Qwen3 is a thinking model — set thinking=True to allow <think> blocks,
    False to get direct answers (faster for testing).
    """
    messages = [{"role": "user", "content": user_message}]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )


def main():
    from transformers import AutoTokenizer

    print(f"Loading model: {MODEL}")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    llm = sgl.Engine(
        model_path=MODEL,
        attention_backend="flashinfer",
        log_level="info",   # show all logger.info() lines
    )

    questions = [
        "What is 2+2?",
        "What is the capital of France?",
        "Explain KV cache in one sentence.",
    ]

    # Apply chat template — thinking=False for fast direct answers
    prompts = [apply_chat_template(tokenizer, q, thinking=False) for q in questions]
    sampling_params = {"temperature": 0.0, "max_new_tokens": 256}

    print("\nRunning inference...")
    print("=" * 60)

    outputs = llm.generate(prompts, sampling_params)

    for question, out in zip(questions, outputs):
        print(f"\nQ: {question}")
        print(f"A: {out['text'].strip()}")
        print(f"   ({out['meta_info']['completion_tokens']} tokens)")

    print("\n" + "=" * 60)
    print("Done. Shutting down engine.")
    llm.shutdown()


# IMPORTANT: Engine uses multiprocessing spawn — must guard with __main__
if __name__ == "__main__":
    main()
