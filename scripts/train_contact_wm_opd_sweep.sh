#!/usr/bin/env bash

# Train a Contact WM Teacher, then distill every retained Teacher checkpoint
# into an independent two-step OPD Student.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-$ROOT_DIR/.conda-env/bin/python}"
TEACHER_CONFIG="${TEACHER_CONFIG:-config/train_cfg/contact_world_model.yaml}"
OPD_CONFIG="${OPD_CONFIG:-config/train_cfg/contact_world_model_opd.yaml}"
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/contact_world_model_opd_sweep/${RUN_TAG}}"
TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:-$RUN_ROOT/teacher}"
STUDENT_OUTPUT_ROOT="${STUDENT_OUTPUT_ROOT:-$RUN_ROOT/students}"

# Optional overrides. Empty values preserve the YAML template values.
TEACHER_MAX_STEPS="${TEACHER_MAX_STEPS:-}"
TEACHER_CHECKPOINT_EVERY="${TEACHER_CHECKPOINT_EVERY:-}"
TEACHER_TOP_K="${TEACHER_TOP_K:-}"
STUDENT_MAX_STEPS="${STUDENT_MAX_STEPS:-}"
STUDENT_CHECKPOINT_EVERY="${STUDENT_CHECKPOINT_EVERY:-}"
STUDENT_TOP_K="${STUDENT_TOP_K:-}"
TEACHER_STEPS="${TEACHER_STEPS:-}"
STUDENT_STEPS="2"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable does not exist or is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ ! -f "$TEACHER_CONFIG" ]]; then
  echo "Teacher config does not exist: $TEACHER_CONFIG" >&2
  exit 1
fi
if [[ ! -f "$OPD_CONFIG" ]]; then
  echo "OPD config does not exist: $OPD_CONFIG" >&2
  exit 1
fi

mkdir -p "$RUN_ROOT" "$STUDENT_OUTPUT_ROOT"
TEACHER_RENDERED_CONFIG="$RUN_ROOT/teacher_config.yaml"
TEACHER_LOG="$RUN_ROOT/teacher.log"

render_teacher_config() {
  "$PYTHON_BIN" - "$TEACHER_CONFIG" "$TEACHER_RENDERED_CONFIG" \
    "$TEACHER_OUTPUT_DIR" "$TEACHER_MAX_STEPS" \
    "$TEACHER_CHECKPOINT_EVERY" "$TEACHER_TOP_K" "$TEACHER_STEPS" <<'PY'
import sys
from pathlib import Path

import yaml

source, destination, output_dir, max_steps, interval, top_k, teacher_steps = sys.argv[1:]
with Path(source).open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise TypeError(f"config must be a mapping: {source}")

train = config.setdefault("train", {})
train["output_dir"] = output_dir
if max_steps:
    train["max_train_steps"] = int(max_steps)
if interval:
    train["checkpoint_every_steps"] = int(interval)
if top_k:
    train["top_k"] = int(top_k)
if teacher_steps:
    config.setdefault("model", {})["flow_inference_steps"] = int(teacher_steps)

Path(destination).parent.mkdir(parents=True, exist_ok=True)
with Path(destination).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
}

render_student_config() {
  local teacher_checkpoint="$1"
  local student_output_dir="$2"
  local destination="$3"
  "$PYTHON_BIN" - "$OPD_CONFIG" "$destination" "$teacher_checkpoint" \
    "$student_output_dir" "$STUDENT_MAX_STEPS" \
    "$STUDENT_CHECKPOINT_EVERY" "$STUDENT_TOP_K" \
    "$TEACHER_STEPS" "$STUDENT_STEPS" <<'PY'
import sys
from pathlib import Path

import yaml

(
    source,
    destination,
    teacher_checkpoint,
    output_dir,
    max_steps,
    interval,
    top_k,
    teacher_steps,
    student_steps,
) = sys.argv[1:]
with Path(source).open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise TypeError(f"config must be a mapping: {source}")

model = config.setdefault("model", {})
model["flow_inference_steps"] = int(student_steps)

distillation = config.setdefault("distillation", {})
distillation["enabled"] = True
distillation["teacher_checkpoint_path"] = str(Path(teacher_checkpoint).resolve())
distillation["student_steps"] = int(student_steps)
if teacher_steps:
    distillation["teacher_steps"] = int(teacher_steps)

train = config.setdefault("train", {})
train["output_dir"] = output_dir
if max_steps:
    train["max_train_steps"] = int(max_steps)
if interval:
    train["checkpoint_every_steps"] = int(interval)
if top_k:
    train["top_k"] = int(top_k)

Path(destination).parent.mkdir(parents=True, exist_ok=True)
with Path(destination).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
}

echo "[1/2] Training Contact WM Teacher"
render_teacher_config
PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" \
  train/trainer/contact_world_model_train.py \
  --config "$TEACHER_RENDERED_CONFIG" 2>&1 | tee "$TEACHER_LOG"

TEACHER_CHECKPOINT_DIR="$TEACHER_OUTPUT_DIR/checkpoints"
if [[ ! -d "$TEACHER_CHECKPOINT_DIR" ]]; then
  echo "Teacher checkpoint directory was not created: $TEACHER_CHECKPOINT_DIR" >&2
  exit 1
fi

mapfile -t TEACHER_CHECKPOINTS < <(
  find "$TEACHER_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step_*.pt' -printf '%f\n' \
    | sort -V
)
if [[ "${#TEACHER_CHECKPOINTS[@]}" -eq 0 && -f "$TEACHER_CHECKPOINT_DIR/latest.pt" ]]; then
  TEACHER_CHECKPOINTS=("latest.pt")
fi
if [[ "${#TEACHER_CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No Teacher step checkpoints found under: $TEACHER_CHECKPOINT_DIR" >&2
  exit 1
fi

echo "[2/2] Distilling ${#TEACHER_CHECKPOINTS[@]} Teacher checkpoints with Student=${STUDENT_STEPS} steps"
for checkpoint_name in "${TEACHER_CHECKPOINTS[@]}"; do
  checkpoint_path="$TEACHER_CHECKPOINT_DIR/$checkpoint_name"
  checkpoint_id="${checkpoint_name%.pt}"
  student_output_dir="$STUDENT_OUTPUT_ROOT/$checkpoint_id"
  student_config="$student_output_dir/config.yaml"
  student_log="$student_output_dir/train.log"

  echo "Distilling Teacher checkpoint: $checkpoint_path"
  render_student_config "$checkpoint_path" "$student_output_dir" "$student_config"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" \
    train/trainer/torque_world_model_opd_train.py \
    --config "$student_config" 2>&1 | tee "$student_log"
done

echo "Sweep complete. Outputs: $RUN_ROOT"
