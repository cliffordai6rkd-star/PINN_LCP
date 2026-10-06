"""Contact WM position parity, stage learning rates and three-job orchestration."""
import copy
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest
import torch
import yaml

from test_contact_world_model_dataset import make_dataset
from test_latent_contact_world_model import batch, threads
from test_latent_contact_world_model_sft import conditional_config, finalized, save_model
from test_latent_contact_world_model_training import training_case
from data_process.contact_world_model_dataset import ContactWorldModelDataset
from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from model.pinn_model.contact_world_model import ContactWorldModel
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel, load_latent_checkpoint
from scripts.latent_cwm_nero_offline import infer_physical
from scripts.latent_wm_pipeline import resolve_configs, completed_checkpoint, seal_pretrained, TASK_CONFIGS
from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer

ROOT = Path(__file__).resolve().parents[1]


def learned_config():
    cfg = conditional_config()
    cfg["model"].update(inputs=["q", "dq", "delta_q", "tau"], outputs=["q", "tau"])
    cfg["model"]["temporal_position_encoding"] = "contact_wm_learned_index"
    return cfg


def test_positions_use_same_embedding_indices_as_contact_wm_and_need_no_grid():
    cfg = learned_config()
    latent, contact = finalized(cfg).eval(), ContactWorldModel(cfg).eval()
    values = {key: value for key, value in batch(cfg).items() if "grid_positions" not in key}
    seen = {name: [] for name in ("latent_history", "contact_history", "latent_action", "contact_action", "latent_future", "contact_future")}
    hooks = []
    for prefix, model in (("latent", latent), ("contact", contact)):
        for stream in ("history", "action", "future"):
            key = prefix+"_"+stream
            hooks.append(getattr(model, stream+"_pos_embedding").register_forward_pre_hook(
                lambda _, args, key=key: seen[key].append(args[0].clone())))
    encoded = latent.encode_conditions(values)
    other = contact.encode_conditions(values)
    contact.flow_velocity(torch.zeros(3, contact.future_horizon, contact.flow_dim), torch.full((3, 1), .5), other)
    for hook in hooks:
        hook.remove()
    for stream in ("history", "action", "future"):
        assert seen["latent_"+stream] and seen["contact_"+stream]
        for indices in seen["contact_"+stream]:
            torch.testing.assert_close(indices, seen["latent_"+stream][0])
    assert seen["latent_history"][0].tolist() == [5, 4, 3, 2, 1, 0]
    assert seen["latent_action"][0].tolist() == [0, 1, 2]
    assert seen["latent_future"][0].tolist() == [0, 1, 2, 3]
    assert encoded["future_pe"].shape == (3, 4, 8)
    latent.set_stage("sft")
    assert all(not p.requires_grad for p in latent.future_pos_embedding.parameters())
    assert all(p.requires_grad for p in latent.history_pos_embedding.parameters())
    assert all(p.requires_grad for p in latent.action_pos_embedding.parameters())


def test_learned_dataset_retains_contact_wm_native_windows_with_camera_jitter(make_dataset):
    original, cfg = make_dataset(offset=1, future=4)
    original.source_dataset.hf_dataset[:]["timing.action_anchor_timestamp_ns"][40:44] += 8_000_000
    cfg["model"].update(family="latent_carswm_lstm_v2", temporal_position_encoding="contact_wm_learned_index")
    contact = ContactWorldModelDataset(cfg)
    latent = LatentContactWorldModelDataset(cfg)
    assert latent.valid_indices == contact.valid_indices
    assert latent.grid_audit["invalid_windows"] == 0
    for key in ("q", "action", "action_chunk_timestamp_ns"):
        torch.testing.assert_close(latent[52][key], contact[52][key])
    assert "history_grid_positions" not in latent[52]


def test_learned_codec_flow_resume_and_stage_learning_rate(make_dataset, tmp_path):
    cfg = training_case(make_dataset, tmp_path/"full")
    cfg["model"].update(family="latent_carswm_lstm_v2", temporal_position_encoding="contact_wm_learned_index")
    cfg["codec"]["lr"] = .003
    cfg["train"]["lr"] = .0001
    full = LatentContactWorldModelTrainer(cfg)
    full.train()
    cfg["train"]["output_dir"] = str(tmp_path/"partial")
    partial = LatentContactWorldModelTrainer(cfg)
    partial.setup()
    assert partial.lr == .003 and all(group["lr"] == .003 for group in partial.optimizer.param_groups)
    partial.run_stage(stop_after_updates=1)
    cfg["train"]["resume_from"] = str(partial.output_dir)
    resumed = LatentContactWorldModelTrainer(cfg)
    resumed.train()
    assert resumed.lr == .0001 and resumed.global_step == 6
    for key, value in full.model.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(value, resumed.model.state_dict()[key], rtol=0, atol=0)
    restored, _ = load_latent_checkpoint(resumed.ckpt_dir/"latest.pt")
    assert restored.learned_positions


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mixed-device validation requires CUDA")
def test_cuda_validation_and_codec_boundary_resume_keep_cpu_metadata(make_dataset, tmp_path):
    cfg = training_case(make_dataset, tmp_path/"cuda_resume")
    cfg["model"].update(family="latent_carswm_lstm_v2", temporal_position_encoding="contact_wm_learned_index",
                        flow_inference_steps=2)
    cfg["train"].update(device="cuda:0", gradient_every=1, amp={"enabled": True, "dtype": "bfloat16"})
    cfg["train"]["probabilistic_validation"].update(per_task_max_batches=1, latent_ablation=True)
    partial = LatentContactWorldModelTrainer(cfg)
    partial.setup()
    partial.run_stage()
    assert partial.codec_step == cfg["codec"]["max_optimizer_steps"]
    cfg["train"]["resume_from"] = str(partial.output_dir)
    resumed = LatentContactWorldModelTrainer(cfg)
    resumed.setup()
    assert resumed.codec_step == partial.codec_step and resumed.global_step == 0
    values = next(iter(resumed.val_loader))
    prepared = resumed.model.prepare_batch(resumed.batch_to_device(values))
    assert prepared["q"].is_cuda and prepared["task_index"].is_cuda
    assert prepared["history_indices"].device.type == "cpu"
    assert prepared["action_chunk_timestamp_ns"].device.type == "cpu"
    # This is the exact boundary that failed after 20k real Nero updates.
    resumed.finalize_codec()
    assert resumed.codec_validation["q_physical_mse"] >= 0
    assert resumed.model.codec_ready and (resumed.output_dir/"codec.pt").exists()
    b = values["q"].shape[0]
    mixed_tasks = torch.cat((torch.zeros(b//2, dtype=torch.long), torch.ones(b-b//2, dtype=torch.long)))
    resumed.val_loader = [dict(values, task_index=torch.zeros(b, dtype=torch.long)),
                          dict(values, task_index=mixed_tasks)]
    for stage in ("flow", "sft"):
        resumed._configure_stage(stage)
        result = resumed.evaluate()
        assert result["task_0_evaluated_windows"] == b
        assert result["task_1_evaluated_windows"] == b-b//2
        assert result["latent_zero_q_physical_mse"] >= 0
        assert result["val_loss"] >= 0


def test_learned_inference_accepts_physical_conditions_without_timestamp_pe():
    cfg = learned_config()
    model = finalized(cfg).eval()
    model.flow_inference_steps = 2
    values = batch(cfg)
    physical = {key: values[key] for key in ("q", "dq", "delta_q", "tau", "action", "action_mask")}
    out = infer_physical(model, physical, num_samples=2)
    assert out["q"].shape == (3, 2, 4, 2)
    assert out["future_grid_positions"].tolist() == [[1, 2, 3, 4]]*3


def test_pipeline_recipe_is_three_jobs_with_matching_positions(tmp_path):
    configs = resolve_configs(tmp_path)
    assert len(configs) == 3
    pre = configs["nero_all"]
    assert pre["codec"]["lr"] == 3e-4 and pre["codec"]["max_optimizer_steps"] == 20000
    assert pre["train"]["batch_size"] == 256 and pre["train"]["lr"] == 1e-4
    assert pre["train"]["max_optimizer_steps"] == 250000
    for task, cfg in configs.items():
        assert cfg["model"]["temporal_position_encoding"] == "contact_wm_learned_index"
        assert cfg["dataloader"]["state_history_horizon"] == 50
        assert cfg["dataloader"]["prediction_horizon"] == 32 and cfg["dataloader"]["action_condition_horizon"] == 8
        assert cfg["train"]["gradient_every"] == 1 and cfg["train"]["contact_sampling"]["phase_weights"] == [1, 5, 5]
        assert cfg["contact_gate"]["precontact_duration_s"] == 1
        if task != "nero_all":
            assert cfg["train"]["batch_size"] == 128 and cfg["train"]["lr"] == 3e-5
            assert cfg["train"]["max_optimizer_steps"] == 50000 and cfg["sft"]["flow_mode"] == "frozen"
            assert cfg["sft"]["pretrained_checkpoint"] == str(tmp_path/"shared/pretrained_for_sft.pt")


def test_only_completed_final_checkpoint_is_sealed_and_later_changes_rejected(tmp_path):
    cfg = learned_config()
    cfg["train"].update(max_optimizer_steps=4, stage="all", output_dir=str(tmp_path/"nero_all"))
    cfg["codec"] = {"max_optimizer_steps": 2}
    (tmp_path/"configs").mkdir()
    (tmp_path/"configs"/TASK_CONFIGS["nero_all"]).write_text(yaml.safe_dump(cfg))
    output = tmp_path/"nero_all"
    (output/"checkpoints").mkdir(parents=True)
    status = {"status": "interrupted", "stage": "flow", "flow_step": 4, "codec_ready": True}
    (output/"status.json").write_text(json.dumps(status))
    assert completed_checkpoint(tmp_path, "nero_all") is None
    with pytest.raises(RuntimeError, match="incomplete"):
        seal_pretrained(tmp_path)
    model = finalized(cfg)
    checkpoint = output/"checkpoints/step_00000004.pt"
    save_model(model, checkpoint)
    payload = torch.load(checkpoint, weights_only=False)
    payload.update(global_step=4, latent_training={"stage": "flow", "codec_step": 2, "final_validation": {"val_loss": .1}})
    torch.save(payload, checkpoint)
    status["status"] = "complete"
    (output/"status.json").write_text(json.dumps(status))
    seal_pretrained(tmp_path)
    assert (tmp_path/"shared/pretrained_for_sft.pt").read_bytes() == checkpoint.read_bytes()
    seal_pretrained(tmp_path)
    payload["latent_training"]["final_validation"]["val_loss"] = .2
    torch.save(payload, checkpoint)
    with pytest.raises(ValueError, match="changed after sealing"):
        seal_pretrained(tmp_path)


def test_real_learned_pretraining_and_two_sft_checkpoints_pass_pipeline_checks(make_dataset, tmp_path):
    configs_dir = tmp_path/"configs"
    configs_dir.mkdir()
    cfg = training_case(make_dataset, tmp_path/"nero_all")
    cfg["model"].update(family="latent_carswm_lstm_v2", temporal_position_encoding="contact_wm_learned_index")
    cfg["codec"]["lr"] = 3e-4
    cfg["train"]["max_optimizer_steps"] = 2
    (configs_dir/TASK_CONFIGS["nero_all"]).write_text(yaml.safe_dump(cfg))
    source = LatentContactWorldModelTrainer(cfg)
    source.train()
    seal_pretrained(tmp_path)
    for task in ("xarm_peel_cucumber", "xarm_erase_board"):
        target_cfg = copy.deepcopy(cfg)
        target_cfg["train"].update(stage="sft", output_dir=str(tmp_path/task), lr=3e-5)
        target_cfg["sft"] = {"pretrained_checkpoint": str(tmp_path/"shared/pretrained_for_sft.pt"), "flow_mode": "frozen"}
        (configs_dir/TASK_CONFIGS[task]).write_text(yaml.safe_dump(target_cfg))
        target = LatentContactWorldModelTrainer(target_cfg)
        target.train()
        assert completed_checkpoint(tmp_path, task).is_file()
        for key, value in source.ema.model.future_pos_embedding.state_dict().items():
            torch.testing.assert_close(value, target.model.future_pos_embedding.state_dict()[key], rtol=0, atol=0)


def fake_python(tmp_path):
    """Exercise Bash process scheduling without a long GPU run or real datasets."""
    path = tmp_path/"fake_python"
    code = r'''
import json,os,signal,sys,time
from pathlib import Path
a=sys.argv[1:];module=a[a.index('-m')+1]
def arg(name): return a[a.index(name)+1]
root=Path(os.environ['LATENT_RUN_ROOT']);root.mkdir(exist_ok=True)
def event(name):
    with (root/'events.jsonl').open('a') as f:f.write(json.dumps({'event':name,'t':time.time()})+'\n')
if module=='scripts.latent_wm_pipeline':
    action=a[a.index(module)+1]
    if action=='prepare': (root/'configs').mkdir(exist_ok=True)
    if action=='check': sys.exit(0 if (root/(arg('--task')+'.complete')).exists() else 1)
    if action=='seal':
        assert (root/'nero_all.complete').exists();event('sealed')
    sys.exit(0)
config=Path(arg('--config')).name
name={'nero_pretrain_all.yaml':'nero_all','xarm_peel_cucumber_sft.yaml':'xarm_peel_cucumber','xarm_erase_board_sft.yaml':'xarm_erase_board'}[config]
event(name+'.start')
def stop(*_): event(name+'.stopped');sys.exit(143)
signal.signal(signal.SIGTERM,stop)
mode=os.environ.get('FAKE_MODE','ok')
if name=='nero_all':
    time.sleep(.05)
    if mode=='incomplete': event(name+'.incomplete');sys.exit(0)
else:
    if mode=='fail' and name=='xarm_peel_cucumber': time.sleep(.2);sys.exit(7)
    time.sleep(2 if mode=='fail' else (0 if mode=='instant' else .3))
(root/(name+'.complete')).write_text('done');event(name+'.end')
'''
    path.write_text("#!"+sys.executable+"\n"+textwrap.dedent(code))
    path.chmod(0o755)
    return path


def launch_fake(tmp_path, mode="ok", sft_parallel="1"):
    environment = {**os.environ, "PYTHON": str(fake_python(tmp_path)), "LATENT_RUN_ROOT": str(tmp_path/"run"),
                   "FAKE_MODE": mode, "SFT_PARALLEL": sft_parallel}
    result = subprocess.run(["bash", str(ROOT/"scripts/train_latent_wm_pretrain_sft.sh")], env=environment,
                            capture_output=True, text=True, timeout=15)
    events = [json.loads(line) for line in (tmp_path/"run/events.jsonl").read_text().splitlines()]
    return result, events, environment


def test_bash_starts_two_sft_jobs_after_pretrain_and_skips_completed_runs(tmp_path):
    result, events, environment = launch_fake(tmp_path)
    assert result.returncode == 0, result.stderr
    time_for = {event["event"]: event["t"] for event in events}
    assert time_for["nero_all.end"] <= time_for["sealed"]
    assert max(time_for["xarm_peel_cucumber.start"], time_for["xarm_erase_board.start"]) < min(
        time_for["xarm_peel_cucumber.end"], time_for["xarm_erase_board.end"])
    assert time_for["sealed"] < min(time_for["xarm_peel_cucumber.start"], time_for["xarm_erase_board.start"])
    again = subprocess.run(["bash", str(ROOT/"scripts/train_latent_wm_pretrain_sft.sh")], env=environment,
                           capture_output=True, text=True, timeout=10)
    assert again.returncode == 0
    starts = [json.loads(line)["event"] for line in (tmp_path/"run/events.jsonl").read_text().splitlines() if '.start' in line]
    assert len(starts) == 3


def test_bash_runs_all_three_jobs_in_sequence_when_requested(tmp_path):
    result, events, _ = launch_fake(tmp_path, sft_parallel="0")
    assert result.returncode == 0, result.stderr
    time_for = {event["event"]: event["t"] for event in events}
    assert time_for["nero_all.end"] <= time_for["xarm_peel_cucumber.start"]
    assert time_for["xarm_peel_cucumber.end"] <= time_for["xarm_erase_board.start"]


def test_bash_sequential_failure_does_not_start_later_task(tmp_path):
    result, events, _ = launch_fake(tmp_path, "fail", sft_parallel="0")
    assert result.returncode == 7
    assert "xarm_peel_cucumber failed (exit 7)" in result.stderr
    assert not any(event["event"].startswith("xarm_erase_board.") for event in events)


def test_bash_does_not_start_sft_after_incomplete_normal_pretrain_exit(tmp_path):
    result, events, _ = launch_fake(tmp_path, "incomplete")
    assert result.returncode != 0
    assert not any(event["event"].startswith("xarm_") for event in events)


def test_bash_stops_other_sft_process_after_failure(tmp_path):
    result, events, _ = launch_fake(tmp_path, "fail")
    assert result.returncode == 7, result.stderr
    assert any(event["event"] == "xarm_erase_board.stopped" for event in events)
    assert not any(event["event"] == "xarm_erase_board.end" for event in events)


def test_bash_handles_jobs_that_finish_before_wait_n(tmp_path):
    result, events, _ = launch_fake(tmp_path, "instant")
    assert result.returncode == 0, result.stderr
    assert sum(event["event"].endswith(".end") for event in events) == 3
