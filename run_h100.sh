#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python "$SCRIPT_DIR/train_predictor.py" \
  --data-dir "$SCRIPT_DIR/data" \
  --log-dir "$SCRIPT_DIR/runs" \
  --device auto \
  --precision bf16 \
  --batch-size 4 \
  "$@"
