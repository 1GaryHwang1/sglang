#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../../../python"

MODEL_PATH="${1:-/models/Qwen3-0.6B}"
MODE="${2:-normal}"
export SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND=python
export SGLANG_TEST_RETRACT=0
if [[ "$MODE" == "retract" ]]; then
  export SGLANG_TEST_RETRACT=1
  export SGLANG_TEST_RETRACT_INTERVAL=16
fi

exec python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 0.0.0.0 \
  --port 30000 \
  --schedule-policy fcfs \
  --disable-overlap-schedule \
  --disable-radix-cache \
  --chunked-prefill-size 128 \
  --page-size 1 \
  --enable-mixed-chunk \
  --max-running-requests 2 \
  --max-queued-requests 2 \
  --prefill-max-requests 1 \
  --enable-priority-scheduling \
  --schedule-low-priority-values-first \
  --priority-scheduling-preemption-threshold 0 \
  --enable-metrics \
  --load-snapshot-publish-interval 1 \
  "${@:3}"
