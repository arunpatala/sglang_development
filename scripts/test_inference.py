"""
Test SGLang inference via the OpenAI-compatible API.
Run after serve_qwen3.sh has started and printed "ready to roll".

Usage:
    python scripts/test_inference.py
    python scripts/test_inference.py --prompt "Explain transformers in one sentence"
    python scripts/test_inference.py --stream
"""

import argparse

from openai import OpenAI


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--prompt", default="What is 2+2? Answer briefly.")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--stream", action="store_true")
    args = parser.parse_args()

    client = OpenAI(
        base_url=f"http://{args.host}:{args.port}/v1",
        api_key="none",
    )

    print(f"Model  : {args.model}")
    print(f"Prompt : {args.prompt}")
    print(f"Stream : {args.stream}")
    print("-" * 40)

    if args.stream:
        stream = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
            stream=True,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                print(delta, end="", flush=True)
        print()
    else:
        response = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "user", "content": args.prompt}],
            max_tokens=args.max_tokens,
        )
        import json
        print(response.choices[0].message.content)
        print()
        print("Usage:")
        print(json.dumps(response.usage.model_dump(), indent=2))
        print(f"Finish: {response.choices[0].finish_reason}")


if __name__ == "__main__":
    main()
