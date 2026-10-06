"""Codec/Flow pretraining and interface SFT with optimizer-update budgets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import signal
import time

import torch
from torch.utils.data import DataLoader
import yaml

from data_process.latent_contact_world_model_dataset import LatentContactWorldModelDataset
from model.pinn_model.latent_contact_world_model import LatentContactWorldModel
from model.pinn_model.latent_pretrained import normalizer_envelope, convert_scale
from train.base_trainer import BaseTrainer, ModelEMA
from train.carswm_metrics import contact_confusion_matrix, distribution_metrics
from train.latent_contact_world_model_loss import LatentContactWorldModelLoss
from train.nomalizer import Normalizer
from train.trainer.deterministic_world_model_train import DeterministicWorldModelTrainer

log = logging.getLogger(__name__)


def equal_contract(a, b):
    if torch.is_tensor(a) or torch.is_tensor(b):
        return torch.is_tensor(a) and torch.is_tensor(b) and torch.equal(a.cpu(), b.cpu())
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(equal_contract(a[k], b[k]) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(equal_contract(x,y) for x,y in zip(a,b))
    return a == b


class PlannedBatchSampler:
    """Main process acknowledges consumption; worker prefetch cannot advance resume."""
    def __init__(self, order, cursor, batch_size):
        self.order, self.cursor, self.batch_size = order, cursor, batch_size

    def __iter__(self):
        for i in range(self.cursor, len(self.order), self.batch_size):
            yield self.order[i:i+self.batch_size].tolist()

    def __len__(self):
        return math.ceil((len(self.order)-self.cursor)/self.batch_size)


class LatentContactWorldModelTrainer(BaseTrainer):
    # These utilities operate only on the existing dataset/phase sampler;
    # the network, losses, stage lifecycle and CLI are independent.
    _sample_indices = staticmethod(DeterministicWorldModelTrainer._sample_indices)
    fit_dataset_normalizer = DeterministicWorldModelTrainer.fit_dataset_normalizer
    _fit_contact_weight = DeterministicWorldModelTrainer._fit_contact_weight
    build_train_sampler = DeterministicWorldModelTrainer.build_train_sampler

    def __init__(self, config):
        config = copy.deepcopy(config)
        train = config.setdefault("train", {})
        for key,value in (("output_dir", "outputs/latent_carswm_lstm_grid/default"),
                          ("max_optimizer_steps", 250000), ("checkpoint_every_steps", 50000), ("top_k", 5)):
            train.setdefault(key, value)
        data = config.setdefault("dataloader", {})
        data.setdefault("normalize_mode", "gaussian")
        data.setdefault("normalize_lowdim_keys", ["q", "dq", "delta_q", "tau", "action"])
        super().__init__(config)
        self.loss_calculator = LatentContactWorldModelLoss(self.config)
        self.requested_stage = self.train_config.get("stage", "all")
        if self.requested_stage not in {"all", "codec", "flow", "sft"}:
            raise ValueError("train.stage must be codec, flow, all or sft")
        self.sft_config = self.config.get("sft") or {}
        if self.split_mode != "episode":
            raise ValueError("latent v1 trainer requires episode split")
        if self.early_stopping_enabled:
            raise ValueError("latent step budgets require early_stopping.enabled=false")
        rollout = self.train_config.get("rollout_validation") or {}
        if rollout.get("enabled"):
            raise ValueError("use evaluate_feedback() for measured reconditioning; legacy free-running rollout is unsupported")
        self.codec_config = self.config.get("codec") or {}
        self.flow_lr = self.lr
        self.codec_lr = float(self.codec_config.get("lr", self.flow_lr))
        if not math.isfinite(self.codec_lr) or self.codec_lr <= 0:
            raise ValueError("codec.lr must be finite and positive")
        self.codec_max_steps = int(self.codec_config.get("max_optimizer_steps", 10000))
        self.flow_max_steps = int(self.train_config.get("max_optimizer_steps", 250000))
        if self.codec_max_steps < 1 or self.flow_max_steps < 1:
            raise ValueError("codec and Flow budgets must be positive")
        self.max_optimizer_steps = self.flow_max_steps
        self.codec_step = 0
        self.stage = "codec"
        self.order = None
        self.batch_cursor = 0
        self.stage_epoch = 0
        self.stop_requested = False
        self.data_contract = None
        self.codec_validation = {}
        self.final_validation = {}
        self.latent_statistics_report = {}
        self.recovery_every = int(self.train_config.get("recovery_every_steps", 1000))
        if self.recovery_every < 1:
            raise ValueError("recovery_every_steps must be positive")
        if self.device_batch_keys is not None:
            self.device_batch_keys.update({*LatentContactWorldModel.CONDITION_KEYS, *LatentContactWorldModel.TARGET_KEYS,
                "contact", "history_valid_mask", "importance_weight", "task_index"})

    def build_dataset(self):
        return LatentContactWorldModelDataset(self.config, compute_normalizer=False)

    def build_model(self):
        return LatentContactWorldModel(self.config)

    def _write_json(self, name, value):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / name
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
        os.replace(temporary, path)

    def setup(self):
        self.set_seed()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(self.config, sort_keys=False))
        self.dataset = self.build_dataset()
        self._write_json("grid_audit.json", self.dataset.grid_audit)
        self.train_dataset, self.val_dataset = self.split_dataset_by_episode()
        if not len(self.train_dataset) or not len(self.val_dataset):
            raise ValueError("both train and validation episodes must contain compatible windows")
        self.fit_dataset_normalizer(self.train_dataset)
        self.train_sampler = self.build_train_sampler(self.train_dataset)
        self.data_contract = {
            "sources": (self.config.get("train_data") or {}).get("sources"),
            "split_mode": self.split_mode, "seed": self.seed,
            "train_indices_sha256": hashlib.sha256(torch.tensor(self.train_dataset.indices).numpy().tobytes()).hexdigest(),
            "val_indices_sha256": hashlib.sha256(torch.tensor(self.val_dataset.indices).numpy().tobytes()).hexdigest(),
            "raw_valid_indices_sha256": hashlib.sha256(torch.tensor(self.dataset.valid_indices).numpy().tobytes()).hexdigest(),
            "train_windows": len(self.train_dataset), "val_windows": len(self.val_dataset),
            "grid_audit": {k:v for k,v in self.dataset.grid_audit.items() if k != "episodes"},
            "tau_label_contract": (self.dataset.tau_label_report or {}).get("label_contract_sha256"),
        }
        if self.dataset_on_device:
            self.dataset.to_device(self.device)
            self.num_workers = self.val_num_workers = 0
            self.pin_memory = self.val_pin_memory = False
            self.persistent_workers = self.val_persistent_workers = False
        self.val_loader = DataLoader(self.val_dataset, **self._dataloader_kwargs(shuffle=False,
            num_workers=self.val_num_workers, prefetch_factor=self.val_prefetch_factor,
            pin_memory=self.val_pin_memory, persistent_workers=self.val_persistent_workers))
        self.model = self.build_model().to(self.device)
        self.model.wm_normalizer = normalizer_envelope(self.dataset.normalizer, self.config)
        resume_payload = None
        if self.resume_from is not None:
            path = self.resolve_resume_checkpoint(self.resume_from)
            resume_payload = torch.load(path, map_location="cpu", weights_only=False)
            state = resume_payload.get("latent_training") or {}
            if state.get("data_contract") != self.data_contract:
                raise ValueError("resume data/grid/split contract mismatch")
            if not equal_contract(resume_payload.get("normalizer"), self.model.wm_normalizer):
                raise ValueError("resume train-split normalizer mismatch")
            self.stage = state["stage"]
            if (self.stage == "sft") != (self.requested_stage == "sft"):
                raise ValueError("resume stage mismatch: SFT must resume with train.stage=sft")
            if self.stage == "sft" and state.get("sft_objective") != self._sft_objective():
                raise ValueError("resume SFT objective mismatch")
            self.codec_step = int(state["codec_step"])
            self.stage_epoch = int(state["stage_epoch"])
            self.order = state.get("sample_order")
            self.batch_cursor = int(state.get("batch_cursor", 0))
            self.codec_validation = state.get("codec_validation", {})
            self.latent_statistics_report = state.get("latent_statistics_report", {})
            self.loss_calculator.set_contact_class_weights(state["contact_class_weights"])
            # No initialization paths are opened during resume.
        else:
            if self.requested_stage == "sft":
                path = self.sft_config.get("pretrained_checkpoint")
                if not path:
                    raise ValueError("train.stage=sft requires sft.pretrained_checkpoint or a full SFT resume")
                if self.codec_config.get("checkpoint_path"):
                    raise ValueError("SFT imports the pretrained model's codec; codec.checkpoint_path must be null")
                payload = self.model.initialize_sft(self.resolve_resume_checkpoint(path),
                                                  use_ema=bool(self.sft_config.get("use_ema", True)))
                self.codec_step = int((payload.get("latent_training") or {}).get("codec_step", 0))
                self.latent_statistics_report = (payload.get("latent_training") or {}).get("latent_statistics_report", {})
                self.stage = "sft"
            else:
                source_preprocessing = self._source_preprocessing()
                self.model.initialize_pretrained_motion(wm_filters=self.dataset.filter_config,
                                                       source_preprocessing=source_preprocessing)
            if self.requested_stage != "sft" and self.codec_config.get("checkpoint_path"):
                self.load_codec(self.codec_config["checkpoint_path"])
                self.stage = "flow"
            elif self.requested_stage == "flow":
                raise ValueError("train.stage=flow requires a codec checkpoint or a full resume checkpoint")
        self._configure_stage(self.stage)
        if resume_payload is None and self.stage == "sft":
            self.codec_validation = self.evaluate(codec=True)
            self._write_json("sft_initial_reconstruction.json", self.codec_validation)
        if resume_payload is not None:
            super()._load_resume_checkpoint()
            self.model.set_stage(self.stage)
            if self.ema is not None:
                self.ema.model.requires_grad_(False).eval()
            # BaseTrainer's epoch records are not sample consumption cursors.
            self.current_epoch = self.stage_epoch
        self.setup_wandb()
        if self.model.transfer_provenance is not None:
            self._write_json("transfer.json", self.model.transfer_provenance)
        self._write_status("ready")

    def _sft_objective(self):
        return {"lambda_fm": self.loss_calculator.sft_fm_weight,
                "lambda_reconstruction": self.loss_calculator.sft_reconstruction_weight,
                "lambda_free": self.loss_calculator.lambda_free,
                "codec_contact_weight": self.loss_calculator.codec_contact_weight}

    def _source_preprocessing(self):
        result = {}
        sources = (self.config.get("train_data") or {}).get("sources") or []
        for source in sources:
            path = Path(source["root"]) / "meta/world_model_timeline.json"
            if not path.exists():
                continue
            fields = json.loads(path.read_text()).get("feature_filters", {})
            for key in ("q", "dq", "delta_q"):
                entry = fields.get(self.dataset.high_keys[key])
                if entry:
                    if key in result and result[key] != entry:
                        raise ValueError(f"sources disagree on preprocessing for {key}")
                    result[key] = entry
        return result

    def _configure_stage(self, stage):
        self.stage = stage
        self.model.set_stage(stage)
        self.lr = self.codec_lr if stage == "codec" else self.flow_lr
        self.optimizer = torch.optim.AdamW(self.model.parameter_groups(), lr=self.lr, weight_decay=self.weight_decay)
        self.max_optimizer_steps = self.codec_max_steps if stage == "codec" else self.flow_max_steps
        self.scheduler = self.build_scheduler()
        self.max_optimizer_steps = self.flow_max_steps
        # The codec snapshot is final raw; EMA begins only after freezing it.
        self.ema = ModelEMA(self.model, self.ema_decay, self.ema_update_after_step, self.ema_update_every) if stage != "codec" and self.ema_enabled else None

    def compute_loss(self, batch):
        out = self.model(batch)
        loss, out["loss_dict"] = self.loss_calculator(out, batch)
        return loss, out

    def _model_checkpoint_metadata(self):
        return {**super()._model_checkpoint_metadata(), "latent_training": {
            "stage": self.stage, "codec_step": self.codec_step, "flow_step": self.global_step,
            "stage_epoch": self.stage_epoch, "sample_order": self.order, "batch_cursor": self.batch_cursor,
            "data_contract": self.data_contract, "codec_validation": self.codec_validation,
            "latent_statistics_report": self.latent_statistics_report,
            "contact_class_weights": self.loss_calculator.contact_class_weights,
            "sft_objective": self._sft_objective() if self.stage == "sft" else None,
            "codec_snapshot_policy": "final_raw", "frozen_modules": [name for name, module in self.model.named_children()
                if not any(p.requires_grad for p in module.parameters())],
            "final_validation": self.final_validation}}

    def _resume_payload(self):
        return {"checkpoint_type":"latent_recovery", "model_version":self.model.MODEL_VERSION,
            "epoch":self.stage_epoch, "global_step":self.global_step,
            "model":self.ema.model.state_dict() if self.ema is not None else self.model.state_dict(),
            "model_raw":self.model.state_dict() if self.ema is not None else None,
            "ema":self._ema_checkpoint_metadata(), "optimizer":self.optimizer.state_dict(),
            "scheduler":self.scheduler.state_dict() if self.scheduler is not None else None,
            "config":self.config, "normalizer":normalizer_envelope(self.dataset.normalizer, self.config),
            "dataloader_filters":self.dataset.filter_config, "sample_rate_hz":self.dataset.sample_rate_hz,
            **self._model_checkpoint_metadata(), **self._checkpoint_runtime_state(resume_epoch=self.stage_epoch)}

    def save_recovery(self):
        self._save_checkpoint_atomic(self._resume_payload(), self.ckpt_dir / "latest.pt")

    def _write_status(self, status, **extra):
        self._write_json("status.json", {"status":status, "pid":os.getpid(), "stage":self.stage,
            "codec_step":self.codec_step, "flow_step":self.global_step, "flow_budget":self.flow_max_steps,
            "sft_step":self.global_step if self.stage == "sft" else 0,
            "codec_budget":self.codec_max_steps, "output_dir":str(self.output_dir),
            "codec_ready":bool(self.model.codec_ready) if self.model is not None else False,
            "updated_at_unix":time.time(), **extra})

    def _next_loader(self):
        n = len(self.train_dataset)
        if self.order is None or self.batch_cursor >= len(self.order):
            if self.order is not None:
                self.stage_epoch += 1
            generator = torch.Generator().manual_seed(self.seed + self.stage_epoch + (1_000_000 if self.stage != "codec" else 0))
            if self.train_sampler is None:
                self.order = torch.randperm(n, generator=generator)
            else:
                self.order = torch.multinomial(self.train_sampler.weights, self.train_sampler.num_samples,
                    self.train_sampler.replacement, generator=generator)
            self.batch_cursor = 0
        sampler = PlannedBatchSampler(self.order, self.batch_cursor, self.batch_size)
        kwargs = self._dataloader_kwargs(shuffle=False)
        kwargs.pop("shuffle", None)
        kwargs.pop("batch_size", None)
        kwargs.pop("drop_last", None)
        # Loader construction must not consume the source-noise RNG on resume.
        kwargs["generator"] = torch.Generator().manual_seed(self.seed + self.stage_epoch)
        self.loader = DataLoader(self.train_dataset, batch_sampler=sampler, **kwargs)
        return self.loader

    def run_stage(self, *, stop_after_updates=None):
        budget = self.codec_max_steps if self.stage == "codec" else self.flow_max_steps
        counter = lambda: self.codec_step if self.stage == "codec" else self.global_step
        started_at, initial_step = time.monotonic(), counter()
        while counter() < budget and not self.stop_requested:
            loader = self._next_loader()
            self.model.train()
            self.optimizer.zero_grad(set_to_none=True)
            accumulated = 0
            metric_sums, metric_counts = {}, {}
            for number, batch in enumerate(loader):
                batch = self.batch_to_device(batch)
                with self.autocast_context():
                    loss, out = self.compute_loss(batch)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"nonfinite {self.stage} loss at codec={self.codec_step} flow={self.global_step}")
                scaled = loss / self.gradient_every
                if self.amp_scaler is None:
                    scaled.backward()
                else:
                    self.amp_scaler.scale(scaled).backward()
                accumulated += 1
                self.batch_cursor += self._batch_size(batch)
                self._accumulate_scalar_metrics(metric_sums, metric_counts, out["loss_dict"], self._batch_size(batch),
                                               defer_device_sync=self.defer_metric_sync)
                update = accumulated == self.gradient_every or number+1 == len(loader) or self.stop_requested
                if not update:
                    continue
                if accumulated != self.gradient_every:
                    for parameter in self.model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.mul_(self.gradient_every/accumulated)
                if self.amp_scaler is not None:
                    self.amp_scaler.unscale_(self.optimizer)
                if self.gradient_clip_norm:
                    torch.nn.utils.clip_grad_norm_([p for p in self.model.parameters() if p.requires_grad], self.gradient_clip_norm)
                successful = True
                if self.amp_scaler is None:
                    self.optimizer.step()
                else:
                    scale = self.amp_scaler.get_scale()
                    self.amp_scaler.step(self.optimizer)
                    self.amp_scaler.update()
                    successful = self.amp_scaler.get_scale() >= scale
                self.optimizer.zero_grad(set_to_none=True)
                accumulated = 0
                if successful:
                    if self.stage == "codec":
                        self.codec_step += 1
                    else:
                        self.global_step += 1
                    if self.ema is not None:
                        self.ema.update(self.model, self.global_step)
                    if self.scheduler is not None and self.scheduler_config.get("name") != "reduce_on_plateau":
                        if self.scheduler_step_per_optimizer_step or self.step_based_training:
                            self.scheduler.step()
                step = counter()
                if successful and step % self.wandb_log_every_steps == 0:
                    metrics = self._average_scalar_metrics(metric_sums, metric_counts)
                    self._append_metrics({"stage":self.stage, "codec_step":self.codec_step, "flow_step":self.global_step,
                        "optimizer_updates_per_s":(step-initial_step)/max(time.monotonic()-started_at, 1e-6), **metrics})
                    self.log_wandb({f"{self.stage}/{k}":v for k,v in metrics.items()}, step=self.global_step)
                    self._write_status("running")
                    metric_sums, metric_counts = {}, {}
                if successful and self.stage != "codec" and step < budget and self.checkpoint_every_steps and step % self.checkpoint_every_steps == 0:
                    # Reuse the original newest-step top_k retention semantics.
                    validation = self.evaluate()
                    super().save_step_checkpoint(self.stage_epoch, {"avg_loss":float(loss.detach()), **validation})
                    self.render_checkpoint(step)
                elif successful and step % self.recovery_every == 0:
                    self.save_recovery()
                if counter() >= budget or self.stop_requested or (stop_after_updates and counter()-initial_step >= stop_after_updates):
                    self.save_recovery()
                    return
            # Preserve the task's validation interval as an epoch interval.
            if (self.stage_epoch+1) % self.val_every == 0 and counter() < budget:
                validation = self.evaluate()
                if self.scheduler is not None and self.scheduler_config.get("name") == "reduce_on_plateau":
                    self.scheduler.step(validation["val_loss"])
                elif self.scheduler is not None and not self.scheduler_step_per_optimizer_step and not self.step_based_training:
                    self.scheduler.step()

    def _append_metrics(self, record):
        record["time_unix"] = time.time()
        with (self.output_dir / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")

    @torch.no_grad()
    def finalize_codec(self):
        # The final RAW snapshot and every decoder are fixed before statistics.
        self.model.snapshot_target_encoder()
        self.model.set_stage("flow")
        self.model.eval()
        self.codec_validation = self.evaluate(codec=True)
        stats_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, num_workers=0)
        count = 0
        mean = torch.zeros(self.model.latent_dim, dtype=torch.float64, device=self.device)
        m2 = torch.zeros_like(mean)
        for batch in stats_loader:
            batch = self.batch_to_device(batch)
            # Welford merge, not moving statistics during codec training.
            latent = self.model.encode_future(batch, target=True).reshape(-1, self.model.latent_dim).double()
            chunk_count, chunk_mean = latent.shape[0], latent.mean(0)
            delta = chunk_mean - mean
            total = count + chunk_count
            m2 += (latent-chunk_mean).square().sum(0) + delta.square()*count*chunk_count/total
            mean += delta*chunk_count/total
            count = total
        if count == 0:
            raise ValueError("cannot fit latent stats on empty training split")
        std = (m2/count).sqrt()
        floor = float(self.codec_config.get("latent_std_floor", 1e-4))
        if not math.isfinite(floor) or floor <= 0:
            raise ValueError("latent_std_floor must be finite and positive")
        self.model.latent_mean.copy_(mean.float())
        self.model.latent_std.copy_(std.clamp_min(floor).float())
        self.model.codec_ready.fill_(True)
        digest = hashlib.sha256()
        for name in self.model.CODEC_MODULES:
            for key,value in getattr(self.model, name).state_dict().items():
                digest.update((name+key).encode())
                digest.update(value.cpu().numpy().tobytes())
        digest.update(self.model.latent_mean.cpu().numpy().tobytes())
        digest.update(self.model.latent_std.cpu().numpy().tobytes())
        self.model.codec_snapshot = "final_raw_sha256:" + digest.hexdigest()
        self.latent_statistics_report = {"training_future_frames":count, "std_floor":floor,
            "near_zero_channels":torch.nonzero(std < floor).flatten().tolist(),
            "unclamped_min_std":float(std.min()), "snapshot":self.model.codec_snapshot}
        log.info("frozen codec: validation=%s latent_statistics=%s", self.codec_validation, self.latent_statistics_report)
        payload = {"model_version":self.model.codec_contract()["schema"], "codec_contract":self.model.codec_contract(),
            "modules":{name:getattr(self.model,name).state_dict() for name in self.model.CODEC_MODULES},
            "latent_mean":self.model.latent_mean, "latent_std":self.model.latent_std,
            "codec_snapshot":self.model.codec_snapshot, "codec_step":self.codec_step,
            "normalizer":self.model.wm_normalizer, "codec_normalizer":self.model.codec_normalizer,
            "statistics":self.latent_statistics_report,
            "validation":self.codec_validation, "data_contract":self.data_contract}
        self._save_checkpoint_atomic(payload, self.output_dir / "codec.pt")
        self._write_json("codec_validation.json", {**self.codec_validation, "latent_statistics":self.latent_statistics_report})
        self.order, self.batch_cursor, self.stage_epoch = None, 0, 0
        self._configure_stage("flow")
        self.save_recovery()

    def load_codec(self, path):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("model_version") != self.model.codec_contract()["schema"] or payload.get("codec_contract") != self.model.codec_contract():
            raise ValueError("codec checkpoint architecture/preprocessing contract mismatch")
        if not equal_contract(payload.get("normalizer"), self.model.wm_normalizer):
            raise ValueError("codec input normalizer mismatch; frozen heads require their training scales")
        if payload.get("data_contract") != self.data_contract:
            raise ValueError("codec dataset/split contract mismatch")
        for name in self.model.CODEC_MODULES:
            getattr(self.model,name).load_state_dict(payload["modules"][name], strict=True)
        mean, std = payload["latent_mean"], payload["latent_std"]
        if mean.shape != self.model.latent_mean.shape or std.shape != mean.shape or not torch.isfinite(mean).all() or not torch.isfinite(std).all() or (std <= 0).any():
            raise ValueError("invalid saved latent statistics")
        self.model.latent_mean.copy_(mean)
        self.model.latent_std.copy_(std)
        self.model.codec_ready.fill_(True)
        self.model.codec_snapshot = payload["codec_snapshot"]
        self.model.codec_normalizer = copy.deepcopy(payload.get("codec_normalizer"))
        if self.model.conditional_codec and not equal_contract(self.model.codec_normalizer, self.model.wm_normalizer):
            raise ValueError("conditional codec target normalizer mismatch")
        self.codec_step = int(payload["codec_step"])
        self.codec_validation = payload["validation"]
        self.latent_statistics_report = payload["statistics"]

    @staticmethod
    def _class_metrics(confusion):
        tp = confusion.diag().float()
        precision = tp/confusion.sum(0).clamp_min(1)
        recall = tp/confusion.sum(1).clamp_min(1)
        f1 = 2*precision*recall/(precision+recall).clamp_min(1e-8)
        result = {"contact_macro_f1":float(f1.mean())}
        for k in range(3):
            result.update({f"contact_{k}_precision":float(precision[k]), f"contact_{k}_recall":float(recall[k]),
                           f"contact_{k}_f1":float(f1[k]), f"contact_{k}_support":int(confusion[k].sum())})
        return result

    @torch.no_grad()
    def evaluate(self, *, codec=None):
        if codec is None:
            codec = self.stage == "codec"
        training_model = self.model
        model = self.ema.model if self.ema is not None and self.ema_use_for_validation and not codec else self.model
        previous_mode = model.training
        model.eval()
        sums, counts = {}, {}
        confusion = torch.zeros(3,3, dtype=torch.long, device=self.device)
        task_confusions, task_batches, task_windows = {}, {}, {}
        probability_cfg = self.train_config.get("probabilistic_validation") or {}
        samples = int(probability_cfg.get("num_samples", 8))
        limit = int(probability_cfg.get("max_batches", 8))
        per_task_limit = int(probability_cfg.get("per_task_max_batches", 0))
        if per_task_limit < 0 or limit < 0 or samples < 1:
            raise ValueError("invalid probabilistic validation limits or sample count")
        ablations = bool(probability_cfg.get("latent_ablation", False)) and model.conditional_codec
        generator = torch.Generator(device=self.device).manual_seed(int((self.train_config.get("rollout_validation") or {}).get("source_seed", 1234)))
        # Validation cannot advance the training RNG.
        rng = self._capture_rng_state()
        try:
            for i, batch in enumerate(self.val_loader):
                batch = self.batch_to_device(batch)
                prepared = model.prepare_batch(batch)
                b = self._batch_size(batch)
                with self.autocast_context():
                    if codec:
                        out = model.codec_forward(batch)
                        _, metrics = self.loss_calculator.codec_loss(out, batch)
                    else:
                        noise = torch.randn(b,model.future_horizon,model.latent_dim,device=self.device,generator=generator)
                        out = model(batch, flow_time=self.train_config.get("validation_flow_time", 0.5), source_noise=noise)
                        _, metrics = self.loss_calculator(out, batch)
                self._accumulate_scalar_metrics(sums, counts, metrics, b, defer_device_sync=True)
                tasks = prepared.get("task_index", torch.zeros(b, dtype=torch.long, device=self.device))
                if not codec:
                    fm_errors = (out["flow_velocity_pred"].float()-out["flow_velocity_target"].float()).square().mean((1,2))
                    for task in tasks.unique().tolist():
                        rows = tasks == task
                        self._accumulate_scalar_metrics(sums, counts, {f"task_{task}_latent_fm_mse": fm_errors[rows].mean()},
                                                        int(rows.sum()), defer_device_sync=True)
                selected = torch.ones(b, dtype=torch.bool, device=tasks.device) if codec else torch.zeros(b, dtype=torch.bool, device=tasks.device)
                if not codec and probability_cfg.get("enabled", True):
                    if per_task_limit:
                        for task in tasks.unique().tolist():
                            if task_batches.get(task, 0) < per_task_limit:
                                selected |= tasks == task
                                task_batches[task] = task_batches.get(task, 0)+1
                    elif limit == 0 or i < limit:
                        selected.fill_(True)
                if selected.any():
                    if codec:
                        sampled = {key:value[:,None] for key,value in out.items() if key in
                                   ("q_pred", "tau_pred", "contact_logits", "contact_probability")}
                    else:
                        noise = torch.randn(b,samples,model.future_horizon,model.latent_dim,device=self.device,generator=generator)
                        with self.autocast_context():
                            sampled = model.sample(batch, num_samples=samples, source_noise=noise)
                    # Device filtering keeps timestamps/indices on CPU while
                    # predictions and task IDs live on the training device.
                    sampled = {key: value[selected.to(value.device)] for key, value in sampled.items() if torch.is_tensor(value) and value.shape[0] == b}
                    prepared = {key: value[selected.to(value.device)] if torch.is_tensor(value) and value.ndim and value.shape[0] == b else value
                                for key, value in prepared.items()}
                    tasks = tasks[selected]
                    b = int(selected.sum())
                    if not codec:
                        metrics = distribution_metrics({k:sampled[k+"_pred"].float() for k in ("q","tau")},
                            {k:prepared[k+"_future"].float() for k in ("q","tau")},
                            sampled["contact_probability"].float(), prepared["contact_future"])
                        self._accumulate_scalar_metrics(sums, counts, {k:v.mean() for k,v in metrics.items()}, b, defer_device_sync=True)
                    physical, per_window = {}, {}
                    for key in ("q","tau"):
                        prediction = convert_scale(key, sampled[key+"_pred"].float(), model.wm_normalizer, inverse=True)
                        truth = convert_scale(key, prepared[key+"_future"].float(), model.wm_normalizer, inverse=True)
                        error = prediction-truth[:,None]
                        per_window[key+"_physical_mae"] = error.abs().mean((1,2,3))
                        per_window[key+"_physical_mse"] = error.square().mean((1,2,3))
                    probability = sampled["contact_probability"].float().mean(1).clamp_min(1e-8)
                    labels = prepared["contact_future"][...,0].long()
                    per_window["contact_ce"] = -probability.gather(-1, labels[...,None]).log().mean((1,2))
                    physical.update({key: value.mean() for key, value in per_window.items()})
                    self._accumulate_scalar_metrics(sums, counts, physical, b, defer_device_sync=True)
                    confusion += contact_confusion_matrix(sampled["contact_probability"], prepared["contact_future"])
                    for task in tasks.unique().tolist():
                        rows = tasks == task
                        windows = int(rows.sum())
                        task_windows[task] = task_windows.get(task, 0)+windows
                        self._accumulate_scalar_metrics(sums, counts,
                            {f"task_{task}_{key}": value[rows].mean() for key, value in per_window.items()}, windows, defer_device_sync=True)
                        matrix = contact_confusion_matrix(sampled["contact_probability"][rows], prepared["contact_future"][rows])
                        task_confusions[task] = task_confusions.get(task, torch.zeros_like(matrix))+matrix
                    if not codec and ablations:
                        raw = sampled["raw_latent"]
                        variants = {"zero": model.latent_mean.expand_as(raw)}
                        if b > 1:
                            variants["shuffled"] = raw.roll(1, 0)
                        for name, latent in variants.items():
                            with self.autocast_context():
                                decoded = model.decode(latent, prepared)
                            ablation_metrics = {}
                            for key in ("q", "tau"):
                                prediction = convert_scale(key, decoded[key+"_pred"].float(), model.wm_normalizer, inverse=True)
                                truth = convert_scale(key, prepared[key+"_future"].float(), model.wm_normalizer, inverse=True)
                                ablation_metrics[f"latent_{name}_{key}_physical_mse"] = (prediction-truth[:,None]).square().mean()
                            probability = decoded["contact_probability"].float().mean(1).clamp_min(1e-8)
                            ablation_metrics[f"latent_{name}_contact_ce"] = -probability.gather(-1, labels[...,None]).log().mean()
                            self._accumulate_scalar_metrics(sums, counts, ablation_metrics, b, defer_device_sync=True)
        finally:
            model.train(previous_mode)
            self.model = training_model
            self._restore_rng_state(rng)
        result = self._average_scalar_metrics(sums, counts)
        for key in list(result):
            if key.endswith("_physical_mse"):
                result[key.removesuffix("mse")+"rmse"] = math.sqrt(result[key])
        result.update(self._class_metrics(confusion))
        for task, matrix in task_confusions.items():
            result.update({f"task_{task}_{key}": value for key, value in self._class_metrics(matrix).items()})
            result[f"task_{task}_evaluated_windows"] = task_windows[task]
        result["val_loss"] = (result["energy_score"] if not codec and probability_cfg.get("replace_val_loss", False)
                              and "energy_score" in result else result["total_loss"])
        result["validation_ode_steps"] = 0 if codec else model.flow_inference_steps
        result["validation_nfe"] = 0 if codec else model.flow_inference_steps*(2 if model.flow_solver == "heun" else 1)
        self._append_metrics({"stage":"codec_validation" if codec else self.stage+"_validation",
                              "codec_step":self.codec_step, "flow_step":self.global_step, **result})
        log.info("%s validation codec_step=%d flow_step=%d %s", self.stage, self.codec_step, self.global_step, result)
        return result

    @torch.no_grad()
    def evaluate_feedback(self, batches, *, source_seed=1234, num_samples=8):
        """Caller supplies consecutive measured windows, each with re-anchored actions.

        No generated tau/q is ever written back into the measurement history.
        Nero should keep the original request anchor when consuming a prefix.
        """
        previous = self.model.training
        self.model.eval()
        generator = torch.Generator(device=self.device).manual_seed(source_seed)
        results = []
        try:
            for measured in batches:
                measured = self.batch_to_device(measured)
                b = self._batch_size(measured)
                noise = torch.randn(b,num_samples,self.model.future_horizon,self.model.latent_dim,
                                    device=self.device,generator=generator)
                results.append(self.model.sample(measured, num_samples=num_samples, source_noise=noise))
        finally:
            self.model.train(previous)
        return results

    @torch.no_grad()
    def render_checkpoint(self, step):
        cfg = self.train_config.get("checkpoint_visualization") or {}
        if not cfg.get("enabled", False) or not cfg.get("every_saved_checkpoint", True):
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        model = self.ema.model if self.ema is not None and cfg.get("use_ema", True) else self.model
        mode = model.training
        model.eval()
        indices = cfg.get("fixed_validation_indices") or {}
        manifest = self.output_dir / "visualization_anchors.json"
        if not indices and manifest.exists():
            indices = json.loads(manifest.read_text())
        if not indices:
            for index in self.val_dataset.indices:
                raw = self.dataset.valid_indices[index]
                phase = str(self.dataset.future_contact_phase(raw))
                indices.setdefault(phase, index)
                if len(indices) == 3:
                    break
            self._write_json("visualization_anchors.json", indices)
        try:
            for phase,index in indices.items():
                if int(index) not in self.val_dataset.indices:
                    raise ValueError("visualization index is outside validation split")
                batch = self.batch_to_device(torch.utils.data.default_collate([self.dataset[int(index)]]))
                generator = torch.Generator(device=self.device).manual_seed(int(cfg.get("seed",2027)))
                n = int(cfg.get("num_samples",32))
                noise = torch.randn(1,n,model.future_horizon,model.latent_dim,device=self.device,generator=generator)
                with self.autocast_context():
                    out = model.sample(batch, num_samples=n, source_noise=noise)
                fig, axes = plt.subplots(3,1, figsize=(9,8))
                joint = int(cfg.get("wrist_joint_index",5))
                for ax,key in zip(axes[:2], ("q","tau")):
                    prediction = out[key+"_pred"][0].float()
                    truth = model.prepare_batch(batch)[key+"_future"][0].float()
                    if cfg.get("denormalize_for_plot", True):
                        prediction = convert_scale(key, prediction, model.wm_normalizer, inverse=True)
                        truth = convert_scale(key, truth, model.wm_normalizer, inverse=True)
                    ax.plot(prediction[:,:,joint].cpu().T, alpha=0.2, color="tab:blue")
                    ax.plot(truth[:,joint].cpu(), color="black", label="measured")
                    ax.set_ylabel(key)
                axes[2].plot(out["contact_probability"][0].float().mean(0).cpu())
                axes[2].set_ylabel("contact probability")
                fig.suptitle(f"Flow {step}, phase {phase}; shared latent sample")
                directory = self.output_dir / "visualizations"
                directory.mkdir(exist_ok=True)
                fig.savefig(directory/f"step_{step:08d}_phase_{phase}.png", bbox_inches="tight")
                plt.close(fig)
        finally:
            model.train(mode)

    def train(self):
        old_handlers = {}
        for signum in (signal.SIGTERM, signal.SIGINT):
            old_handlers[signum] = signal.signal(signum, lambda *_: setattr(self, "stop_requested", True))
        try:
            self.setup()
            if self.stage == "codec":
                self.run_stage()
                if self.stop_requested:
                    self._write_status("interrupted")
                    return
                self.finalize_codec()
            if self.requested_stage == "codec":
                self._write_status("codec_complete")
                return
            self.run_stage()
            if self.stop_requested:
                self._write_status("interrupted")
                return
            self.final_validation = self.evaluate()
            # Refresh final scheduled checkpoint with the completed validation.
            if self.global_step == self.flow_max_steps:
                self.best_checkpoints = [r for r in self.best_checkpoints if r["global_step"] != self.global_step]
                super().save_step_checkpoint(self.stage_epoch, self.final_validation)
                self.render_checkpoint(self.global_step)
            self._write_json("final_validation.json", self.final_validation)
            self._write_status("complete", final_validation=self.final_validation,
                scheduled_checkpoints=[str(p) for p in sorted(self.ckpt_dir.glob("step_*.pt"))])
        except BaseException as exc:
            # Ordinary failures preserve the last committed optimizer state;
            # do not overwrite it with a half-completed accumulation.
            self._write_status("failed", error=f"{type(exc).__name__}: {exc}")
            log.exception("latent WM training failed")
            raise
        finally:
            self.finish_wandb(exit_code=1 if self.stop_requested else 0)
            for signum, handler in old_handlers.items():
                signal.signal(signum, handler)


def main():
    parser = argparse.ArgumentParser(description="Train latent WM codec/Flow or adapt pretrained v2 interfaces with SFT")
    parser.add_argument("--config", "-c", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stage", choices=("codec","flow","all","sft"))
    parser.add_argument("--codec-checkpoint", type=Path)
    parser.add_argument("--pretrained-taufree", type=Path)
    parser.add_argument("--pretrained-checkpoint", type=Path, help="v2 Flow checkpoint for SFT initialization")
    parser.add_argument("--flow-adaptation", choices=("frozen", "adapter"))
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    config = yaml.safe_load(args.config.read_text())
    for key,value in (("resume_from",args.resume),("device",args.device),("output_dir",args.output_dir),("stage",args.stage)):
        if value is not None:
            config.setdefault("train",{})[key] = str(value)
    if args.codec_checkpoint:
        config.setdefault("codec",{})["checkpoint_path"] = str(args.codec_checkpoint)
    if args.pretrained_taufree:
        config.setdefault("model",{})["pretrained_taufree_path"] = str(args.pretrained_taufree)
    if args.pretrained_checkpoint:
        config.setdefault("sft",{})["pretrained_checkpoint"] = str(args.pretrained_checkpoint)
    if args.flow_adaptation:
        config.setdefault("sft",{})["flow_mode"] = args.flow_adaptation
    trainer = LatentContactWorldModelTrainer(config)
    if args.audit_only:
        trainer.setup()
        trainer.finish_wandb()
    else:
        trainer.train()


if __name__ == "__main__":
    main()
