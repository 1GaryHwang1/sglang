#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../python"

MODEL_PATH="${1:-/models/Qwen3-0.6B}"
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --schedule-policy fcfs \
  --disable-overlap-schedule \
  --max-running-requests 1 \
  --prefill-max-requests 1 \
  --enable-metrics \
  --load-snapshot-publish-interval 1
