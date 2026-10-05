#!/usr/bin/env bash
# Train each WM once, then evaluate its final weights with 32/16/8 ODE steps.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
task_python="${PYTHON:-$task_repo_root/.conda-env/bin/python}"
task_run_root="${WM_RUN_ROOT:-outputs/xarm_wm_flow_sweep/$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$task_run_root"
task_run_root="$(cd -- "$task_run_root" && pwd)"
exec 9>"$task_run_root/launcher.lock"
flock -n 9 || { echo "A launcher already holds $task_run_root/launcher.lock" >&2; exit 1; }
echo "$$" > "$task_run_root/launcher.pid"
export TZ="${TZ:-Asia/Shanghai}"
export MPLBACKEND=Agg
task_helper=scripts/xarm_wm_flow_sweep.py

"$task_python" "$task_helper" prepare "$task_run_root" \
  --contact-config "${CONTACT_CONFIG:-config/train_cfg/pretrain/xarm/cwm_erase_board_100hz_40step.yaml}" \
  --latent-config "${LATENT_CONFIG:-config/train_cfg/latent_cwm_erase_board_100hz_40step.yaml}"
if [[ "${WM_PREPARE_ONLY:-0}" == 1 ]]; then
  echo "Prepared configs: $task_run_root"
  exit 0
fi

if [[ "${WM_PARALLEL:-0}" == 1 ]]; then
  task_pids=()
  for task_family in contact_wm latent_wm; do
    echo "$(date --iso-8601=seconds) training $task_family in parallel"
    "$task_python" -u "$task_helper" run "$task_run_root" --family "$task_family" \
      --defer-evaluation >> "$task_run_root/$task_family/train.log" 2>&1 &
    task_pids+=("$!")
  done
  task_failed=0
  for task_pid in "${task_pids[@]}"; do
    wait "$task_pid" || task_failed=1
  done
  if [[ "$task_failed" == 1 ]]; then
    "$task_python" "$task_helper" report "$task_run_root"
    echo "A parallel training job failed; inspect the model logs." >&2
    exit 1
  fi
  export WM_EVALUATION_CONTEXT=isolated_after_parallel_training
fi

for task_family in contact_wm latent_wm; do
  echo "$(date --iso-8601=seconds) starting $task_family; log: $task_run_root/$task_family/train.log"
  if "$task_python" -u "$task_helper" run "$task_run_root" --family "$task_family" \
      >> "$task_run_root/$task_family/train.log" 2>&1; then
    echo "$(date --iso-8601=seconds) completed $task_family"
  else
    task_exit=$?
    "$task_python" "$task_helper" report "$task_run_root"
    echo "Stopped: $task_family exited $task_exit; inspect its train.log" >&2
    exit "$task_exit"
  fi
done
"$task_python" "$task_helper" report "$task_run_root"
