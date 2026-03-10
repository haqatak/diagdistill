#!/bin/bash
set -euo pipefail

# Always run from repo root so relative paths in configs/code are valid.
cd "$(dirname "$0")"

CONFIG=configs/diadistill_train_init.yaml
RUN_TS="$(date +%Y%m%d_%H%M%S)"
LOGDIR="outputs/${RUN_TS}"
WANDB_SAVE_DIR="$LOGDIR"
export WANDB_MODE=disabled
mkdir -p "$LOGDIR"
echo "CONFIG=$CONFIG"
echo "RUN_DIR=$LOGDIR"

torchrun \
  --nproc_per_node=8 \
  train.py \
  --config_path "$CONFIG" \
  --logdir "$LOGDIR" \
  --wandb-save-dir "$WANDB_SAVE_DIR" \
  --disable-wandb
