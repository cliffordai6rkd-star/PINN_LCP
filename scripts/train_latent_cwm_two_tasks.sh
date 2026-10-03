#!/usr/bin/env bash
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
task_python="${PYTHON:-$task_repo_root/.conda-env/bin/python}"
task_run_root="${LATENT_RUN_ROOT:-outputs/latent_carswm_lstm_grid}"
mkdir -p "$task_run_root"
# flock prevents two invocations from training the same sequential run.
exec 9>"$task_run_root/two_tasks.lock"
flock -n 9 || { echo 'Another latent two-task launcher holds the lock.' >&2; exit 1; }

for task_name in insert_usb peel_cucumber; do
    if [[ "$task_name" == insert_usb ]]; then
        task_config=config/train_cfg/latent_cwm_insert_usb_100hz_40step.yaml
    else
        task_config=config/train_cfg/latent_cwm_peel_cucumber_40step.yaml
    fi
    task_output="$task_run_root/$task_name"
    mkdir -p "$task_output"
    task_args=(--config "$task_config" --output-dir "$task_output")
    if [[ -f "$task_output/checkpoints/latest.pt" ]]; then
        task_args+=(--resume "$task_output")
    fi
    echo "$(date --iso-8601=seconds) starting $task_name" | tee -a "$task_run_root/sequence.log"
    "$task_python" -u -m train.trainer.latent_contact_world_model_train "${task_args[@]}" \
        >>"$task_output/train.log" 2>&1
    # A signal/interruption must never authorize starting peel prematurely.
    "$task_python" - "$task_output" <<'PY'
import json,sys
from pathlib import Path
import torch
root=Path(sys.argv[1])
status=json.loads((root/'status.json').read_text())
if status['status'] != 'complete' or status['flow_step'] != 250000 or not status['codec_ready']:
    raise SystemExit(f"Task is not complete: {status}")
for step in (50000,100000,150000,200000,250000):
    path=root/'checkpoints'/f'step_{step:08d}.pt'
    payload=torch.load(path,map_location='cpu',weights_only=False)
    if payload['global_step'] != step or payload['latent_training']['stage'] != 'flow':
        raise SystemExit(f"Incomplete scheduled checkpoint: {path}")
print(f"Verified {root}: Flow 250000, finalized codec, five scheduled checkpoints")
PY
    echo "$(date --iso-8601=seconds) completed $task_name" | tee -a "$task_run_root/sequence.log"
done
