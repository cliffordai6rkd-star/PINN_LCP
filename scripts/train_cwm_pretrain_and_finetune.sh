#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
PRETRAIN_CONFIG="${PRETRAIN_CONFIG:-config/train_cfg/cwm_pretrain_four_tasks.yaml}"
PRETRAIN_OUTPUT_DIR="${PRETRAIN_OUTPUT_DIR:-outputs/cwm_pretrain_four_tasks}"
LOG_DIR="${LOG_DIR:-$PRETRAIN_OUTPUT_DIR/logs}"
mkdir -p "$LOG_DIR"
if [[ -d "$PRETRAIN_OUTPUT_DIR/checkpoints" ]] && compgen -G "$PRETRAIN_OUTPUT_DIR/checkpoints/*.pt" >/dev/null; then
  echo "Refusing to overwrite existing pretraining checkpoints: $PRETRAIN_OUTPUT_DIR" >&2
  exit 2
fi
echo "[pretrain] config=$PRETRAIN_CONFIG"
"$PYTHON_BIN" -m train.trainer.contact_world_model_train -c "$PRETRAIN_CONFIG" 2>&1 | tee "$LOG_DIR/pretrain.log"
CHECKPOINT="$($PYTHON_BIN - "$PRETRAIN_OUTPUT_DIR/checkpoints/latest.pt" <<'PY'
import sys, torch
from pathlib import Path
p = Path(sys.argv[1])
if not p.is_file(): raise SystemExit(f"missing pretraining checkpoint: {p}")
c = torch.load(p, map_location="cpu", weights_only=False)
if not isinstance(c, dict) or not c.get("model"): raise SystemExit("pretraining did not produce a valid model checkpoint")
print(p.resolve())
PY
)"
echo "[pretrain] checkpoint=$CHECKPOINT"
for task in insert_usb push_button cucumber_peeling wipe_board; do
  cfg="config/train_cfg/cwm_finetune_${task}.yaml"
  echo "[$task] config=$cfg checkpoint=$CHECKPOINT"
  "$PYTHON_BIN" -m train.trainer.contact_world_model_finetune_train -c "$cfg" \
    --pretrained-checkpoint "$CHECKPOINT" 2>&1 | tee "$LOG_DIR/finetune_${task}.log"
done
