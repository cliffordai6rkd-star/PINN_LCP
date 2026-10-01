#!/usr/bin/env bash
# Optional launcher. Direct usage: PYTHONPATH=. python train/trainer/contact_world_model_distill_train.py -c config/train_cfg/contact_world_model_distill_s4.yaml
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
PYTHON_BIN="${PYTHON_BIN:-$(command -v python)}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:?Set TEACHER_CHECKPOINT_PATH to original teacher EMA checkpoint}"
STUDENT_STEPS="${STUDENT_STEPS:-4}"
if [[ "$STUDENT_STEPS" != "2" && "$STUDENT_STEPS" != "4" && "$STUDENT_STEPS" != "8" ]]; then
  echo "STUDENT_STEPS must be 2, 4, or 8" >&2
  exit 2
fi
STUDENT_OUTPUT_DIR="${STUDENT_OUTPUT_DIR:-outputs/contact_world_model_distill_s${STUDENT_STEPS}}"
DISTILL_TEMPLATE="${DISTILL_TEMPLATE:-config/train_cfg/contact_world_model_distill_s${STUDENT_STEPS}.yaml}"
STUDENT_INIT_CHECKPOINT_PATH="${STUDENT_INIT_CHECKPOINT_PATH:-}"
MAX_STEPS="${MAX_STEPS:-}"
mkdir -p "$STUDENT_OUTPUT_DIR"
RENDERED_CONFIG="$STUDENT_OUTPUT_DIR/config.yaml"
"$PYTHON_BIN" - "$TEACHER_CHECKPOINT_PATH" "$DISTILL_TEMPLATE" "$RENDERED_CONFIG" "$STUDENT_OUTPUT_DIR" "$STUDENT_STEPS" "$MAX_STEPS" "$STUDENT_INIT_CHECKPOINT_PATH" <<'PY'
from pathlib import Path
import sys, yaml
teacher, template, output, output_dir, steps, max_steps, student_init = sys.argv[1:]
with Path(template).open(encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
config["distillation"]["teacher_checkpoint_path"] = str(Path(teacher).expanduser().resolve())
config["distillation"]["student_steps"] = int(steps)
if student_init:
    config["distillation"]["student_init_checkpoint_path"] = str(Path(student_init).expanduser().resolve())
config["train"]["output_dir"] = str(Path(output_dir).expanduser().resolve())
if max_steps:
    config["train"]["max_optimizer_steps"] = int(max_steps)
Path(output).write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
PY
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" train/trainer/contact_world_model_distill_train.py -c "$RENDERED_CONFIG"
