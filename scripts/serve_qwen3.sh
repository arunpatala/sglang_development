#!/usr/bin/env bash
# Launch SGLang server with Qwen3-0.6B
# Model loads from local HuggingFace cache (no network needed).
# Wait for: "The server is fired up and ready to roll!"
#
# Usage: ./scripts/serve_qwen3.sh [model_path] [host] [port] [attention_backend]

set -e

CONDA=/home/arun/PROJECTS/sglang_development/.conda
PYTHON=$CONDA/bin/python

# Default to locally cached model snapshot (avoids HF network lookup)
DEFAULT_MODEL=$(ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/*/ 2>/dev/null | head -1)
MODEL=${1:-$DEFAULT_MODEL}
HOST=${2:-0.0.0.0}
PORT=${3:-30000}
ATTENTION_BACKEND=${4:-flashinfer}

# Add conda bin to PATH so ninja/cmake are found during FlashInfer JIT
export PATH="$CONDA/bin:$PATH"
# Use cached model without reaching out to HuggingFace
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "Starting SGLang server"
echo "  model             : $MODEL"
echo "  host              : $HOST"
echo "  port              : $PORT"
echo "  attention backend : $ATTENTION_BACKEND"
echo ""

$PYTHON -m sglang.launch_server \
  --model-path "$MODEL" \
  --host "$HOST" \
  --port "$PORT" \
  --attention-backend "$ATTENTION_BACKEND" \
#  --enable-response-cache
