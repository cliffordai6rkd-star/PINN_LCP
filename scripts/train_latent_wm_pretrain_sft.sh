#!/usr/bin/env bash
# Nero codec/Flow pretraining, followed by concurrent or sequential xArm SFT jobs.
set -euo pipefail

task_repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$task_repo_root"
task_python="${PYTHON:-$task_repo_root/.conda-env/bin/python}"
task_run_root="${LATENT_RUN_ROOT:-outputs/latent_wm_v2/nero_pretrain_xarm_sft}"
task_pretrain_device="${PRETRAIN_DEVICE:-cuda:0}"
task_peel_device="${PEEL_DEVICE:-cuda:0}"
task_erase_device="${ERASE_DEVICE:-cuda:0}"
task_sft_parallel="${SFT_PARALLEL:-1}"
task_dry_run=false
case "${1:-}" in
    --dry-run) task_dry_run=true ;;
    --help|-h)
        cat <<'HELP'
Usage: bash scripts/train_latent_wm_pretrain_sft.sh [--dry-run]
Defaults: Nero codec 20k steps, batch 256, lr 3e-4; Flow 250k, batch 256, lr 1e-4.
          xArm peel/erase SFT each 50k steps, batch 128, lr 3e-5; Flow frozen.
Environment: PYTHON, LATENT_RUN_ROOT, PRETRAIN_DEVICE, PEEL_DEVICE, ERASE_DEVICE,
             PRETRAIN_BATCH_SIZE, SFT_BATCH_SIZE, CODEC_STEPS, PRETRAIN_STEPS,
             SFT_STEPS, SFT_PARALLEL (1=concurrent, 0=sequential),
             WANDB_MODE (online/offline/disabled).
Completed tasks are verified and skipped. Interrupted tasks resume automatically.
After verified pretraining, both SFT jobs start concurrently from one sealed checkpoint.
On interruption or failure, active children are stopped so they can save recovery state.
HELP
        exit 0 ;;
    "") ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
esac
if (( $# > 1 )); then echo "Expected at most one argument" >&2; exit 2; fi
if [[ "$task_sft_parallel" != 0 && "$task_sft_parallel" != 1 ]]; then
    echo 'SFT_PARALLEL must be 0 (sequential) or 1 (concurrent).' >&2
    exit 2
fi
if (( BASH_VERSINFO[0] < 5 || (BASH_VERSINFO[0] == 5 && BASH_VERSINFO[1] < 1) )); then
    echo 'This launcher requires Bash 5.1+ for concurrent child status handling.' >&2
    exit 2
fi
mkdir -p "$task_run_root"
task_run_root="$(cd "$task_run_root" && pwd)"
exec 9>"$task_run_root/pipeline.lock"
flock -n 9 || { echo "Another launcher holds $task_run_root/pipeline.lock" >&2; exit 1; }

task_cleanup() {
    local task_exit_code=$?
    trap - EXIT INT TERM
    local task_pid
    while read -r task_pid; do
        kill -TERM "$task_pid" 2>/dev/null || true
    done < <(jobs -pr)
    wait || true
    exit "$task_exit_code"
}
trap task_cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

task_prepare_args=(prepare --run-root "$task_run_root"
    --codec-steps "${CODEC_STEPS:-20000}" --pretrain-steps "${PRETRAIN_STEPS:-250000}"
    --sft-steps "${SFT_STEPS:-50000}" --pretrain-batch "${PRETRAIN_BATCH_SIZE:-256}"
    --sft-batch "${SFT_BATCH_SIZE:-128}" --pretrain-device "$task_pretrain_device"
    --peel-device "$task_peel_device" --erase-device "$task_erase_device"
    --sft-parallel "$task_sft_parallel")
if [[ -n "${WANDB_MODE:-}" ]]; then task_prepare_args+=(--wandb-mode "$WANDB_MODE"); fi
"$task_python" -m scripts.latent_wm_pipeline "${task_prepare_args[@]}" | tee "$task_run_root/plan.log"
if $task_dry_run; then
    echo "Dry run complete. Resolved configurations: $task_run_root/configs"
    exit 0
fi

task_launch() {
    local task_name=$1 task_config=$2
    local task_output="$task_run_root/$task_name"
    mkdir -p "$task_output"
    local task_args=(--config "$task_run_root/configs/$task_config")
    if [[ -f "$task_output/checkpoints/latest.pt" ]] || compgen -G "$task_output/checkpoints/step_*.pt" >/dev/null; then
        task_args+=(--resume "$task_output")
        echo "Resuming $task_name; log: $task_output/train.log"
    else
        echo "Starting $task_name; log: $task_output/train.log"
    fi
    "$task_python" -u -m train.trainer.latent_contact_world_model_train "${task_args[@]}" \
        >>"$task_output/train.log" 2>&1 &
    task_last_pid=$!
    echo "$task_last_pid" > "$task_output/launcher.pid"
}

task_wait_for() {
    local task_name=$1 task_pid=$2 task_exit_code=0
    wait "$task_pid" || task_exit_code=$?
    if (( task_exit_code != 0 )); then
        echo "$task_name failed (exit $task_exit_code); see $task_run_root/$task_name/train.log" >&2
        tail -n 25 "$task_run_root/$task_name/train.log" >&2 || true
    fi
    return "$task_exit_code"
}

task_completed() {
    local task_name=$1 task_check_code
    if "$task_python" -m scripts.latent_wm_pipeline check --run-root "$task_run_root" --task "$task_name"; then
        return 0
    else
        task_check_code=$?
        if (( task_check_code != 1 )); then exit "$task_check_code"; fi
        return 1
    fi
}

if ! task_completed nero_all; then
    task_launch nero_all nero_pretrain_all.yaml
    task_wait_for nero_all "$task_last_pid"
fi
# A normal child exit alone is insufficient (the trainer exits normally on a signal).
"$task_python" -m scripts.latent_wm_pipeline check --run-root "$task_run_root" --task nero_all
"$task_python" -m scripts.latent_wm_pipeline seal --run-root "$task_run_root"

task_sft_pids=()
declare -A task_pid_names=()
for task_name in xarm_peel_cucumber xarm_erase_board; do
    if task_completed "$task_name"; then
        continue
    fi
    if [[ "$task_name" == xarm_peel_cucumber ]]; then
        task_config=xarm_peel_cucumber_sft.yaml
    else
        task_config=xarm_erase_board_sft.yaml
    fi
    task_launch "$task_name" "$task_config"
    if [[ "$task_sft_parallel" == 0 ]]; then
        task_wait_for "$task_name" "$task_last_pid"
        "$task_python" -m scripts.latent_wm_pipeline check --run-root "$task_run_root" --task "$task_name"
    else
        task_sft_pids+=("$task_last_pid")
        task_pid_names["$task_last_pid"]=$task_name
    fi
done
while (( ${#task_sft_pids[@]} )); do
    task_finished_pid=''
    task_wait_code=0
    # A resumed job can finish before wait -n sees it. Reap completed PIDs
    # explicitly, and retry if both exit during the running-job check.
    for task_pid in "${task_sft_pids[@]}"; do
        if ! kill -0 "$task_pid" 2>/dev/null; then
            task_finished_pid=$task_pid
            wait "$task_pid" || task_wait_code=$?
            break
        fi
    done
    if [[ -z "$task_finished_pid" ]]; then
        wait -n -p task_finished_pid "${task_sft_pids[@]}" || task_wait_code=$?
        if (( task_wait_code == 127 )) && [[ -z "${task_finished_pid:-}" ]]; then continue; fi
    fi
    if (( task_wait_code == 0 )); then
        "$task_python" -m scripts.latent_wm_pipeline check --run-root "$task_run_root" --task "${task_pid_names[$task_finished_pid]}"
        task_remaining_pids=()
        for task_pid in "${task_sft_pids[@]}"; do
            if [[ "$task_pid" != "$task_finished_pid" ]]; then task_remaining_pids+=("$task_pid"); fi
        done
        task_sft_pids=("${task_remaining_pids[@]}")
    else
        task_exit_code=$task_wait_code
        echo "An SFT job failed (exit $task_exit_code); stopping the other active job." >&2
        exit "$task_exit_code"
    fi
done
for task_name in xarm_peel_cucumber xarm_erase_board; do
    "$task_python" -m scripts.latent_wm_pipeline check --run-root "$task_run_root" --task "$task_name"
done
echo "Complete: Nero pretraining and both xArm SFT jobs. Results: $task_run_root"
