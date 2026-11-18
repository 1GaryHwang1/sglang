#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../python"

MODEL_PATH="${1:-/models/Qwen3-0.6B}"

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --schedule-policy fcfs \
  --disable-overlap-schedule \
  --enable-metrics \
  --load-snapshot-publish-interval 1