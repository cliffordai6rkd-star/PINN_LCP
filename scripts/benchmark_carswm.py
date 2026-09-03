#!/usr/bin/env python3
"""Reproducible synthetic CARS-WM training/inference stage benchmark."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import torch
import torch.nn.functional as F
import yaml


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pinn_model.contact_world_model import ContactWorldModel
from train.carswm_metrics import distribution_metrics
from train.contact_world_model_loss import ContactWorldModelLoss


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rollout-depth", type=int, default=4)
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def load_config(path):
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    config["model"]["dropout"] = 0.0
    config["model"]["runtime_checks"] = False
    config["model"]["return_attention_weights"] = False
    config["model"]["emit_contact_probabilities"] = True
    config["dataloader"]["normalize_mode"] = None
    return config


def synthetic_batch(config, batch_size, device="cpu"):
    data = config["dataloader"]
    model = config["model"]
    history = int(data["state_history_horizon"])
    future = int(data["prediction_horizon"])
    action_horizon = int(data["action_condition_horizon"])
    joint_dim = int(model["joint_dim"])
    action_dim = int(model["action_dim"])
    result = {
        key: torch.randn(batch_size, history, joint_dim, device=device)
        for key in model["inputs"]
    }
    result.update(
        {
            f"{key}_future": torch.randn(batch_size, future, joint_dim, device=device)
            for key in model["inputs"]
        }
    )
    result["action"] = torch.randn(batch_size, action_horizon, action_dim, device=device)
    result["action_mask"] = torch.ones(
        batch_size, action_horizon, dtype=torch.bool, device=device
    )
    result["contact_future"] = torch.randint(
        0, 3, (batch_size, future, 1), device=device
    ).float()
    result["importance_weight"] = torch.ones(batch_size, device=device)
    return result


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, device, warmup, iterations):
    for _ in range(warmup):
        function()
    sync(device)
    started = perf_counter()
    for _ in range(iterations):
        function()
    sync(device)
    return 1.0e3 * (perf_counter() - started) / iterations


def shift_history(batch, output):
    return {
        **batch,
        **{
            key: torch.cat((batch[key][:, 1:], output[f"{key}_pred"][:, :1].detach()), dim=1)
            for key in ("q", "dq", "delta_q", "tau")
        },
    }


def main():
    args = parse_args()
    if args.batch_size < 1 or args.iterations < 1 or args.rollout_depth < 1:
        raise ValueError("batch-size, iterations, and rollout-depth must be positive")
    torch.manual_seed(42)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)
        torch.cuda.reset_peak_memory_stats(device)
    teacher_config = load_config(ROOT / "config/train_cfg/contact_world_model.yaml")
    student_config = load_config(ROOT / "config/train_cfg/contact_world_model_opd.yaml")
    teacher = ContactWorldModel(teacher_config).to(device).eval()
    student = ContactWorldModel(student_config).to(device).eval()
    teacher_loss = ContactWorldModelLoss(teacher_config)
    cpu_batch = synthetic_batch(teacher_config, args.batch_size)
    batch = {key: value.to(device) for key, value in cpu_batch.items()}
    source = torch.randn(
        args.batch_size, teacher.future_horizon, teacher.flow_dim, device=device
    )

    encoded = teacher.encode_conditions(batch)
    trajectory = source.clone()
    flow_time = torch.full((args.batch_size, 1), 0.5, device=device)
    generated = teacher.integrate_flow(source, encoded, steps=teacher.flow_inference_steps)

    stages = {}
    loader = torch.utils.data.DataLoader([cpu_batch] * 4, batch_size=None)
    stages["dataloader"] = measure(lambda: next(iter(loader)), device, 0, args.iterations)
    stages["normalizer_cpu_to_device"] = measure(
        lambda: {
            key: (value - 0.1).div(1.1).to(device, non_blocking=False)
            for key, value in cpu_batch.items()
            if value.is_floating_point()
        },
        device,
        args.warmup,
        args.iterations,
    )
    stages["state_action_encoders"] = measure(
        lambda: teacher.encode_conditions(batch), device, args.warmup, args.iterations
    )
    stages["one_flow_velocity"] = measure(
        lambda: teacher.flow_velocity(trajectory, flow_time, encoded),
        device,
        args.warmup,
        args.iterations,
    )
    stages["teacher_flow_integration"] = measure(
        lambda: teacher.integrate_flow(source, encoded, steps=teacher.flow_inference_steps),
        device,
        args.warmup,
        args.iterations,
    )
    student_source = source[:, :, : student.flow_dim]
    student_encoded = student.encode_conditions(batch)
    stages["student_flow_integration"] = measure(
        lambda: student.integrate_flow(
            student_source, student_encoded, steps=student.flow_inference_steps
        ),
        device,
        args.warmup,
        args.iterations,
    )
    stages["contact_head"] = measure(
        lambda: teacher.contact_logits(generated, encoded),
        device,
        args.warmup,
        args.iterations,
    )
    prediction = teacher.predict(batch, source_noise=source)
    stages["rollout_write_back"] = measure(
        lambda: shift_history(batch, prediction), device, args.warmup, args.iterations
    )

    def teacher_train_step():
        teacher.train()
        teacher.zero_grad(set_to_none=True)
        output = teacher(batch)
        loss, _ = teacher_loss(output, batch)
        loss.backward()
        teacher.eval()

    stages["teacher_batch_train"] = measure(
        teacher_train_step, device, 1, max(1, args.iterations // 2)
    )

    def direct_distillation():
        student.train()
        student.zero_grad(set_to_none=True)
        teacher_output = teacher.predict(batch, source_noise=source)
        student_output = student.predict_differentiable(
            batch, source_noise=student_source, steps=student.flow_inference_steps
        )
        loss = sum(
            F.mse_loss(student_output[f"{key}_pred"], teacher_output[f"{key}_pred"])
            for key in student.predicted_state_streams
        )
        loss = loss + F.kl_div(
            F.log_softmax(student_output["contact_logits"] / 2.0, dim=-1),
            F.softmax(teacher_output["contact_logits"] / 2.0, dim=-1),
            reduction="batchmean",
        ) * 4.0 / student.future_horizon
        loss.backward()
        student.eval()

    stages["student_direct_distillation"] = measure(
        direct_distillation, device, 1, max(1, args.iterations // 2)
    )

    def legacy_opd():
        current = batch
        for _ in range(args.rollout_depth):
            noise = torch.randn_like(source)
            teacher.predict(current, source_noise=noise)
            output = student.predict(current, source_noise=noise, steps=student.flow_inference_steps)
            current = shift_history(current, output)

    def optimized_opd():
        current = batch
        for _ in range(args.rollout_depth):
            output = student.predict(current, steps=student.flow_inference_steps)
            current = shift_history(current, output)
        noise = torch.randn_like(source)
        teacher.predict(current, source_noise=noise)
        student.predict(current, source_noise=noise, steps=student.flow_inference_steps)

    stages["legacy_opd_rollout"] = measure(legacy_opd, device, 1, args.iterations)
    stages["optimized_opd_rollout"] = measure(optimized_opd, device, 1, args.iterations)

    def validation():
        output = student.sample(
            batch,
            num_samples=args.num_samples,
            steps=student.flow_inference_steps,
        )
        distribution_metrics(
            {key: output[f"{key}_pred"] for key in student.predicted_state_streams},
            {key: batch[f"{key}_future"] for key in student.predicted_state_streams},
            output["contact_probability"],
            batch["contact_future"],
        )

    stages["validation_k_sample"] = measure(validation, device, 1, args.iterations)
    single_latency = measure(
        lambda: student.predict(batch, steps=student.flow_inference_steps),
        device,
        args.warmup,
        args.iterations,
    )
    total_profiled = sum(
        stages[key]
        for key in (
            "dataloader",
            "normalizer_cpu_to_device",
            "state_action_encoders",
            "teacher_flow_integration",
            "student_flow_integration",
            "contact_head",
            "rollout_write_back",
        )
    )
    result = {
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "seed": 42,
        "batch_size": args.batch_size,
        "history_horizon": teacher.history_horizon,
        "future_horizon": teacher.future_horizon,
        "action_horizon": teacher.action_condition_horizon,
        "iterations": args.iterations,
        "stage_ms": stages,
        "stage_percent": {
            key: 100.0 * value / total_profiled for key, value in stages.items()
        },
        "teacher_parameters": sum(parameter.numel() for parameter in teacher.parameters()),
        "student_parameters": sum(parameter.numel() for parameter in student.parameters()),
        "teacher_nfe": 2 * teacher.flow_inference_steps,
        "student_nfe": 2 * student.flow_inference_steps,
        "single_sample_inference_ms": single_latency,
        "k_sample_inference_ms": stages["validation_k_sample"],
        "opd_speedup": stages["legacy_opd_rollout"] / stages["optimized_opd_rollout"],
        "peak_gpu_memory_mb": (
            torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0)
            if device.type == "cuda"
            else None
        ),
        "tau_free_contact_model_ms": None,
        "tau_free_note": "not benchmarked: no compatible tau-free checkpoint is required for synthetic benchmark",
    }
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
