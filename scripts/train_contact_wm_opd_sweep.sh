#!/usr/bin/env bash

# Train a Contact WM Teacher, then distill every retained Teacher checkpoint
# into an independent two-step OPD Student.
#
# To continue an interrupted run, point RUN_ROOT at the original directory:
#   RESUME=1 RUN_ROOT=outputs/contact_world_model_opd_sweep/<run> \
#     bash scripts/train_contact_wm_opd_sweep.sh
#
# To distill one existing Teacher checkpoint without retraining the Teacher:
#   OPD_ONLY=1 TEACHER_CHECKPOINT_PATH=/abs/path/epoch_0001270.pt \
#     bash scripts/train_contact_wm_opd_sweep.sh

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# Prefer an explicitly supplied interpreter, then the currently activated
# Conda environment, and finally the project-local environment.
if [[ -n "${PYTHON_BIN:-}" ]]; then
  PYTHON_BIN="$PYTHON_BIN"
elif [[ -n "${CONDA_PREFIX:-}" && -x "$CONDA_PREFIX/bin/python" ]]; then
  PYTHON_BIN="$CONDA_PREFIX/bin/python"
elif [[ -x "$ROOT_DIR/.conda-env/bin/python" ]]; then
  PYTHON_BIN="$ROOT_DIR/.conda-env/bin/python"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  echo "Python was not found. Activate the pinn environment or set PYTHON_BIN=/path/to/python." >&2
  exit 1
fi
TEACHER_CONFIG="${TEACHER_CONFIG:-config/train_cfg/contact_world_model.yaml}"
OPD_CONFIG="${OPD_CONFIG:-config/train_cfg/contact_world_model_opd.yaml}"
TEACHER_CHECKPOINT_PATH="${TEACHER_CHECKPOINT_PATH:-}"
OPD_ONLY="${OPD_ONLY:-0}"
RESUME_TRAINING="${RESUME_TRAINING:-${RESUME:-0}}"
RESUME_RUN_ROOT="${RESUME_RUN_ROOT:-}"
if [[ -n "${RESUME_FROM:-}" && "$RESUME_TRAINING" == "0" ]]; then
  # Supplying a checkpoint is an unambiguous request to resume.
  RESUME_TRAINING=1
fi
RUN_ROOT_WAS_SET=0
if [[ -n "${RUN_ROOT:-}" ]]; then
  RUN_ROOT_WAS_SET=1
fi
if [[ "$RESUME_TRAINING" == "1" && "$RUN_ROOT_WAS_SET" -eq 0 && -z "$RESUME_RUN_ROOT" ]]; then
  echo "Resume requested, but no existing run was selected. Set RUN_ROOT=/path/to/run (or RESUME_RUN_ROOT=/path/to/run)." >&2
  exit 2
fi
if [[ -n "$RESUME_RUN_ROOT" && -z "${RUN_ROOT:-}" ]]; then
  RUN_ROOT="$RESUME_RUN_ROOT"
fi
RUN_TAG="${RUN_TAG:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-outputs/contact_world_model_opd_sweep/${RUN_TAG}}"
TEACHER_OUTPUT_DIR="${TEACHER_OUTPUT_DIR:-$RUN_ROOT/teacher}"
STUDENT_OUTPUT_ROOT="${STUDENT_OUTPUT_ROOT:-$RUN_ROOT/students}"
# Keep a resumable copy at every scheduled checkpoint.  Set this to 0 to
# preserve the template's historical behavior, at the cost of losing the
# latest pointer after an interrupted run.
SAVE_LATEST_CHECKPOINT="${SAVE_LATEST_CHECKPOINT:-1}"
TEACHER_RESUME_FROM="${TEACHER_RESUME_FROM:-${RESUME_FROM:-}}"
if [[ "$RESUME_TRAINING" == "1" && -z "$TEACHER_RESUME_FROM" ]]; then
  TEACHER_RESUME_FROM="$TEACHER_OUTPUT_DIR"
fi
# Optional machine-specific tau-free checkpoint override. When empty, the
# path from the OPD YAML is preserved.
TAU_FREE_CHECKPOINT_PATH="${TAU_FREE_CHECKPOINT_PATH:-}"

# Optional overrides. Empty values preserve the YAML template values.
TEACHER_MAX_STEPS="${TEACHER_MAX_STEPS:-}"
TEACHER_CHECKPOINT_EVERY="${TEACHER_CHECKPOINT_EVERY:-}"
TEACHER_TOP_K="${TEACHER_TOP_K:-}"
TEACHER_NUM_EPOCHS="${TEACHER_NUM_EPOCHS:-}"
TEACHER_CHECKPOINT_EVERY_EPOCHS="${TEACHER_CHECKPOINT_EVERY_EPOCHS:-}"
STUDENT_MAX_STEPS="${STUDENT_MAX_STEPS:-}"
STUDENT_CHECKPOINT_EVERY="${STUDENT_CHECKPOINT_EVERY:-}"
STUDENT_TOP_K="${STUDENT_TOP_K:-}"
STUDENT_NUM_EPOCHS="${STUDENT_NUM_EPOCHS:-}"
STUDENT_CHECKPOINT_EVERY_EPOCHS="${STUDENT_CHECKPOINT_EVERY_EPOCHS:-}"
TEACHER_STEPS="${TEACHER_STEPS:-}"
STUDENT_STEPS="${STUDENT_STEPS:-2}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python executable does not exist or is not executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ "$RESUME_TRAINING" != "0" && "$RESUME_TRAINING" != "1" ]]; then
  echo "RESUME_TRAINING/RESUME must be 0 or 1, got: $RESUME_TRAINING" >&2
  exit 1
fi
if [[ "$SAVE_LATEST_CHECKPOINT" != "0" && "$SAVE_LATEST_CHECKPOINT" != "1" ]]; then
  echo "SAVE_LATEST_CHECKPOINT must be 0 or 1, got: $SAVE_LATEST_CHECKPOINT" >&2
  exit 1
fi
if [[ "$OPD_ONLY" != "0" && "$OPD_ONLY" != "1" ]]; then
  echo "OPD_ONLY must be 0 or 1, got: $OPD_ONLY" >&2
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

has_checkpoint() {
  local source="$1"
  if [[ -f "$source" ]]; then
    return 0
  fi
  local directory
  for directory in "$source" "$source/checkpoints"; do
    [[ -d "$directory" ]] || continue
    if compgen -G "$directory/*.pt" > /dev/null; then
      return 0
    fi
  done
  return 1
}

# Canonicalize paths through Python so exported values such as ``~/runs/foo``
# are expanded consistently; shell tilde expansion does not happen inside a
# quoted variable.
canonicalize_path() {
  "$PYTHON_BIN" - "$1" <<'PY'
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve(strict=False))
PY
}

# Store absolute paths in rendered configs so they remain valid if the
# launcher is invoked through a wrapper with a different working directory.
RUN_ROOT="$(canonicalize_path "$RUN_ROOT")"
TEACHER_OUTPUT_DIR="$(canonicalize_path "$TEACHER_OUTPUT_DIR")"
STUDENT_OUTPUT_ROOT="$(canonicalize_path "$STUDENT_OUTPUT_ROOT")"
if [[ -n "$TEACHER_RESUME_FROM" ]]; then
  TEACHER_RESUME_FROM="$(canonicalize_path "$TEACHER_RESUME_FROM")"
fi
if [[ -n "$TEACHER_CHECKPOINT_PATH" ]]; then
  TEACHER_CHECKPOINT_PATH="$(canonicalize_path "$TEACHER_CHECKPOINT_PATH")"
fi
if [[ "$OPD_ONLY" == "1" ]]; then
  if [[ -z "$TEACHER_CHECKPOINT_PATH" ]]; then
    echo "OPD_ONLY=1 requires TEACHER_CHECKPOINT_PATH=/path/to/teacher.pt (or a checkpoint directory)." >&2
    exit 2
  fi
  if ! has_checkpoint "$TEACHER_CHECKPOINT_PATH"; then
    echo "No usable Teacher checkpoint found at: $TEACHER_CHECKPOINT_PATH" >&2
    exit 2
  fi
fi
if [[ "$RESUME_TRAINING" == "1" ]] && ! has_checkpoint "$TEACHER_RESUME_FROM"; then
  echo "No usable Teacher checkpoint found at: $TEACHER_RESUME_FROM" >&2
  echo "The interrupted run cannot be resumed because it did not save a checkpoint; start a new RUN_ROOT or choose another checkpoint." >&2
  exit 2
fi

mkdir -p "$RUN_ROOT" "$STUDENT_OUTPUT_ROOT"
TEACHER_RENDERED_CONFIG="$RUN_ROOT/teacher_config.yaml"
TEACHER_LOG="$RUN_ROOT/teacher.log"

TEE_APPEND=()
if [[ "$RESUME_TRAINING" == "1" ]]; then
  TEE_APPEND=(-a)
fi

render_teacher_config() {
  "$PYTHON_BIN" - "$TEACHER_CONFIG" "$TEACHER_RENDERED_CONFIG" \
    "$TEACHER_OUTPUT_DIR" "$TEACHER_MAX_STEPS" \
    "$TEACHER_CHECKPOINT_EVERY" "$TEACHER_TOP_K" "$TEACHER_STEPS" \
    "$TEACHER_NUM_EPOCHS" "$TEACHER_CHECKPOINT_EVERY_EPOCHS" \
    "$TEACHER_RESUME_FROM" "$SAVE_LATEST_CHECKPOINT" <<'PY'
import sys
from pathlib import Path

import yaml

(
    source, destination, output_dir, max_steps, interval, top_k, teacher_steps,
    num_epochs, checkpoint_every_epochs, resume_from, save_latest,
) = sys.argv[1:]
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
if num_epochs:
    train["num_epochs"] = int(num_epochs)
if checkpoint_every_epochs:
    train["checkpoint_every_epochs"] = int(checkpoint_every_epochs)
if resume_from:
    train["resume_from"] = str(Path(resume_from).expanduser())
else:
    # A rendered config may itself be used as the next source template.
    # Never inherit a stale resume switch when starting a fresh run.
    for key in ("resume_from", "resume_checkpoint", "resume"):
        train.pop(key, None)
train["save_latest_checkpoint"] = save_latest == "1"

Path(destination).parent.mkdir(parents=True, exist_ok=True)
with Path(destination).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
}

render_student_config() {
  local teacher_checkpoint="$1"
  local student_output_dir="$2"
  local destination="$3"
  local resume_from=""
  if [[ "$RESUME_TRAINING" == "1" && -d "$student_output_dir" ]]; then
    if ! has_checkpoint "$student_output_dir"; then
      echo "Resume requested but no Student checkpoint exists at: $student_output_dir" >&2
      return 2
    fi
    resume_from="$student_output_dir"
  fi
  "$PYTHON_BIN" - "$OPD_CONFIG" "$destination" "$teacher_checkpoint" \
    "$student_output_dir" "$STUDENT_MAX_STEPS" \
    "$STUDENT_CHECKPOINT_EVERY" "$STUDENT_TOP_K" \
    "$TEACHER_STEPS" "$STUDENT_STEPS" \
    "$STUDENT_NUM_EPOCHS" "$STUDENT_CHECKPOINT_EVERY_EPOCHS" \
    "$TAU_FREE_CHECKPOINT_PATH" "$resume_from" "$SAVE_LATEST_CHECKPOINT" <<'PY'
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
    num_epochs,
    checkpoint_every_epochs,
    tau_free_checkpoint_path,
    resume_from,
    save_latest,
) = sys.argv[1:]
with Path(source).open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream)
if not isinstance(config, dict):
    raise TypeError(f"config must be a mapping: {source}")

model = config.setdefault("model", {})
model["flow_inference_steps"] = int(student_steps)

distillation = config.setdefault("distillation", {})
distillation["enabled"] = True
distillation["teacher_checkpoint_path"] = str(Path(teacher_checkpoint).expanduser())
distillation["student_steps"] = int(student_steps)
if teacher_steps:
    distillation["teacher_steps"] = int(teacher_steps)
if tau_free_checkpoint_path:
    rollout_contact = distillation.setdefault("rollout_contact", {})
    rollout_contact["tau_free_checkpoint_path"] = str(
        Path(tau_free_checkpoint_path).expanduser()
    )

train = config.setdefault("train", {})
train["output_dir"] = output_dir
wandb = train.get("wandb")
if isinstance(wandb, dict) and bool(wandb.get("enabled", False)):
    wandb["name"] = f"{wandb.get('name', 'contact-wm-opd-student')}-{Path(teacher_checkpoint).stem}"
if max_steps:
    train["max_train_steps"] = int(max_steps)
if interval:
    train["checkpoint_every_steps"] = int(interval)
if top_k:
    train["top_k"] = int(top_k)
if num_epochs:
    train["num_epochs"] = int(num_epochs)
if checkpoint_every_epochs:
    train["checkpoint_every_epochs"] = int(checkpoint_every_epochs)
if resume_from:
    train["resume_from"] = str(Path(resume_from).expanduser())
else:
    for key in ("resume_from", "resume_checkpoint", "resume"):
        train.pop(key, None)
train["save_latest_checkpoint"] = save_latest == "1"

Path(destination).parent.mkdir(parents=True, exist_ok=True)
with Path(destination).open("w", encoding="utf-8") as stream:
    yaml.safe_dump(config, stream, sort_keys=False)
PY
}

if [[ "$OPD_ONLY" == "1" ]]; then
  TEACHER_CHECKPOINTS=("$TEACHER_CHECKPOINT_PATH")
  echo "[1/1] Skipping Teacher; distilling checkpoint: $TEACHER_CHECKPOINT_PATH"
else
  echo "[1/2] Training Contact WM Teacher"
  render_teacher_config
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" \
    train/trainer/contact_world_model_train.py \
    --config "$TEACHER_RENDERED_CONFIG" 2>&1 | tee "${TEE_APPEND[@]}" "$TEACHER_LOG"

  TEACHER_CHECKPOINT_DIR="$TEACHER_OUTPUT_DIR/checkpoints"
  if [[ ! -d "$TEACHER_CHECKPOINT_DIR" ]]; then
    echo "Teacher checkpoint directory was not created: $TEACHER_CHECKPOINT_DIR" >&2
    exit 1
  fi

  # The scheduled epoch saver retains the newest ``train.top_k`` checkpoints.
  # Read the effective value from the rendered Teacher config so this sweep
  # cannot accidentally process stale files if an output directory is reused.
  if [[ -n "$TEACHER_TOP_K" ]]; then
    TEACHER_TOP_K_COUNT="$TEACHER_TOP_K"
  else
    TEACHER_TOP_K_COUNT="$($PYTHON_BIN - "$TEACHER_RENDERED_CONFIG" <<'PY'
import sys
from pathlib import Path

import yaml

with Path(sys.argv[1]).open("r", encoding="utf-8") as stream:
    config = yaml.safe_load(stream) or {}
value = (config.get("train") or {}).get("top_k", 3)
print(3 if value is None else int(value))
PY
  )"
  fi
  if ! [[ "$TEACHER_TOP_K_COUNT" =~ ^[1-9][0-9]*$ ]]; then
    echo "Teacher top_k must be a positive integer, got: $TEACHER_TOP_K_COUNT" >&2
    exit 1
  fi

  mapfile -t TEACHER_CHECKPOINTS < <(
    find "$TEACHER_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'epoch_*.pt' -printf '%f\n' \
      | sort -V -r \
      | head -n "$TEACHER_TOP_K_COUNT"
  )
  if [[ "${#TEACHER_CHECKPOINTS[@]}" -eq 0 ]]; then
    mapfile -t TEACHER_CHECKPOINTS < <(
      find "$TEACHER_CHECKPOINT_DIR" -maxdepth 1 -type f -name 'step_*.pt' -printf '%f\n' \
        | sort -V -r \
        | head -n "$TEACHER_TOP_K_COUNT"
    )
  fi
  if [[ "${#TEACHER_CHECKPOINTS[@]}" -eq 0 && -f "$TEACHER_CHECKPOINT_DIR/latest.pt" ]]; then
    TEACHER_CHECKPOINTS=("latest.pt")
  fi
  if [[ "${#TEACHER_CHECKPOINTS[@]}" -eq 0 ]]; then
    echo "No Teacher epoch/step checkpoints found under: $TEACHER_CHECKPOINT_DIR" >&2
    exit 1
  fi
fi

if [[ "$OPD_ONLY" == "1" ]]; then
  echo "[1/1] Distilling one explicitly selected Teacher checkpoint with Student=${STUDENT_STEPS} steps"
else
  echo "[2/2] Distilling ${#TEACHER_CHECKPOINTS[@]} of top_k=${TEACHER_TOP_K_COUNT} Teacher checkpoints from newest to oldest with Student=${STUDENT_STEPS} steps"
fi
for checkpoint_name in "${TEACHER_CHECKPOINTS[@]}"; do
  if [[ "$OPD_ONLY" == "1" ]]; then
    checkpoint_path="$checkpoint_name"
    checkpoint_file_name="$(basename "$checkpoint_name")"
  else
    checkpoint_path="$TEACHER_CHECKPOINT_DIR/$checkpoint_name"
    checkpoint_file_name="$checkpoint_name"
  fi
  checkpoint_id="${checkpoint_file_name%.pt}"
  student_output_dir="$STUDENT_OUTPUT_ROOT/$checkpoint_id"
  student_config="$student_output_dir/config.yaml"
  student_log="$student_output_dir/train.log"

  echo "Distilling Teacher checkpoint: $checkpoint_path"
  render_student_config "$checkpoint_path" "$student_output_dir" "$student_config"
  PYTHONPATH="$ROOT_DIR" "$PYTHON_BIN" \
    train/trainer/torque_world_model_opd_train.py \
    --config "$student_config" 2>&1 | tee "${TEE_APPEND[@]}" "$student_log"
done

echo "Sweep complete. Outputs: $RUN_ROOT"
