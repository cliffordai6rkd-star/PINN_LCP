#!/usr/bin/env bash
# Train an independent CaRS-WM Flow Student from an existing Teacher checkpoint.
# Usage:
#   TEACHER_CHECKPOINT_PATH=/abs/path/epoch_000100.pt STUDENT_STEPS=8 \
#     bash scripts/train_contact_wm_distill.sh
# Set STUDENT_STEPS=4 and STUDENT_OUTPUT_DIR=... for S4 (still supervised by T32).

set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
STUDENT_STEPS="${STUDENT_STEPS:-8}"
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-outputs/contact_world_model_distill_s${STUDENT_STEPS}}"
DISTILL_TEMPLATE="${DISTILL_TEMPLATE:-config/train_cfg/contact_world_model_distill.yaml}"
MAX_STEPS="${MAX_STEPS:-}"
STUDENT_INIT_CHECKPOINT_PATH="${STUDENT_INIT_CHECKPOINT_PATH:-}"

if [[ -z "$TEACHER_CHECKPOINT_PATH" ]]; then
  echo "Set TEACHER_CHECKPOINT_PATH=/path/to/teacher.pt" >&2
  exit 2
fi
if [[ ! -f "$TEACHER_CHECKPOINT_PATH" ]]; then
  echo "Teacher checkpoint not found: $TEACHER_CHECKPOINT_PATH" >&2
  exit 2
fi
if [[ "$STUDENT_STEPS" != "4" && "$STUDENT_STEPS" != "8" ]]; then
  echo "STUDENT_STEPS must be 4 or 8 (got $STUDENT_STEPS)" >&2
  exit 2
fi

mkdir -p "$STUDENT_OUTPUT_DIR"
RENDERED_CONFIG="$STUDENT_OUTPUT_DIR/config.yaml"

"$PYTHON_BIN" - "$TEACHER_CHECKPOINT_PATH" "$DISTILL_TEMPLATE" \
  "$RENDERED_CONFIG" "$STUDENT_OUTPUT_DIR" "$STUDENT_STEPS" "$MAX_STEPS" "$STUDENT_INIT_CHECKPOINT_PATH" <<'PY'
from pathlib import Path
import copy, sys, torch, yaml

checkpoint_path, template_path, output_path, output_dir, student_steps, max_steps, student_init = sys.argv[1:]
checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
teacher_config = checkpoint.get("config")
if not isinstance(teacher_config, dict):
    raise ValueError("Teacher checkpoint does not contain its training config")
with Path(template_path).open("r", encoding="utf-8") as f:
    overlay = yaml.safe_load(f) or {}
config = copy.deepcopy(teacher_config)

def merge(dst, src):
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)

merge(config, overlay)
model = config.setdefault("model", {})
model["flow_inference_steps"] = int(student_steps)
distill = config.setdefault("distillation", {})
distill["enabled"] = True
distill["teacher_checkpoint_path"] = str(Path(checkpoint_path).expanduser().resolve())
distill["teacher_steps"] = int(distill.get("teacher_steps", 32))
distill["student_steps"] = int(student_steps)
if student_init:
    distill["student_init_checkpoint_path"] = str(Path(student_init).expanduser().resolve())
train = config.setdefault("train", {})
train["output_dir"] = str(Path(output_dir).expanduser().resolve())
if max_steps:
    train["max_optimizer_steps"] = int(max_steps)
Path(output_path).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY

PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" train/trainer/contact_world_model_distill_train.py \
  --config "$RENDERED_CONFIG"
