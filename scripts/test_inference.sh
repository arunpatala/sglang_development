#!/usr/bin/env bash
# Send a test chat completion request to the running SGLang server
# Usage: ./test_inference.sh [prompt] [port]

PROMPT=${1:-"What is 2+2? Answer briefly."}
PORT=${2:-30000}
MODEL=${3:-Qwen/Qwen3-0.6B-Instruct}

echo "Sending request to localhost:$PORT"
echo "  prompt: $PROMPT"
echo ""

curl -s http://localhost:$PORT/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "$(cat <<EOF
{
  "model": "$MODEL",
  "messages": [{"role": "user", "content": "$PROMPT"}],
  "max_tokens": 256
}
EOF
)" | python3 -c "
import sys, json
data = json.load(sys.stdin)
if 'choices' in data:
    print(data['choices'][0]['message']['content'])
else:
    print(json.dumps(data, indent=2))
"
