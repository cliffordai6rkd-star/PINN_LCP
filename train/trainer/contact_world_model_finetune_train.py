"""Fine tune CaRS-WM from a completed Contact World Model checkpoint."""
from __future__ import annotations

import argparse
import copy
import logging
import sys
from pathlib import Path
from collections.abc import Mapping

import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from train.nomalizer import Normalizer
from train.trainer.contact_world_model_train import ContactWorldModelTrainer

log = logging.getLogger(__name__)


def _resolve(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    for candidate in (Path.cwd() / p, REPO_ROOT / p):
        if candidate.exists():
            return candidate.resolve()
    return p.resolve()


def _merge(dst, src):
    for key, value in src.items():
        if isinstance(value, Mapping) and isinstance(dst.get(key), Mapping):
            _merge(dst[key], value)
        else:
            dst[key] = copy.deepcopy(value)


def _load_checkpoint(path):
    path = _resolve(path)
    if not path.is_file():
        raise FileNotFoundError(f"pretrained checkpoint not found: {path}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping):
        raise ValueError(f"checkpoint must contain a mapping: {path}")
    return path, payload


class ContactWorldModelFinetuneTrainer(ContactWorldModelTrainer):
    def __init__(self, config, pretrained_checkpoint=None):
        self.pretrained_checkpoint_path = None
        self.pretrained_checkpoint = None
        self.checkpoint_metadata = {}
        if pretrained_checkpoint and not (config.get("train") or {}).get("resume_from"):
            self.pretrained_checkpoint_path, self.pretrained_checkpoint = _load_checkpoint(pretrained_checkpoint)
        super().__init__(config)

    def build_dataset(self):
        # The normalizer is restored from the source checkpoint in
        # fit_dataset_normalizer; target episodes are still split afresh.
        return super().build_dataset()

    def fit_dataset_normalizer(self, train_dataset):
        if self.pretrained_checkpoint is None:
            return super().fit_dataset_normalizer(train_dataset)
        payload = self.pretrained_checkpoint.get("normalizer")
        if not isinstance(payload, Mapping) or not payload.get("stats"):
            raise ValueError("pretrained checkpoint has no Normalizer stats")
        normalizer = Normalizer(copy.deepcopy(payload["stats"]), eps=float(payload.get("eps", 1e-6)))
        self.dataset.set_normalizer(normalizer)
        self.loss_calculator.set_normalizer(normalizer)
        self._fit_contact_weight(self._sample_indices(train_dataset))

    def initialize_model_weights(self):
        if self.pretrained_checkpoint is None:
            return
        checkpoint = self.pretrained_checkpoint
        self.model.validate_checkpoint(checkpoint)
        state = checkpoint.get("model")
        if not isinstance(state, Mapping):
            raise KeyError("pretrained checkpoint has no model weights")
        # The checkpoint contract and strict loading make structural changes
        # fail loudly. EMA is initialized from these loaded weights by BaseTrainer.
        self.model.load_state_dict(state, strict=True)
        ema = checkpoint.get("ema") or {}
        self.checkpoint_metadata = {
            "finetune": {
                "initialized_from": str(self.pretrained_checkpoint_path),
                "source_weight_key": "model",
                "source_ema_enabled": bool(ema.get("enabled", False)),
                "task": (self.config.get("train_data") or {}).get("sources"),
            }
        }
        log.info("initialized fine tuning from %s using checkpoint model weights (source EMA enabled=%s)", self.pretrained_checkpoint_path, ema.get("enabled", False))


def parse_args():
    parser = argparse.ArgumentParser(description="Fine tune CaRS-WM on one task")
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--pretrained-checkpoint", type=Path)
    return parser.parse_args()


def prepare_config(config_path, cli_checkpoint):
    overlay = yaml.safe_load(_resolve(config_path).read_text(encoding="utf-8")) or {}
    train_overlay = overlay.get("train") or {}
    # A resume checkpoint is itself the complete source of configuration;
    # this allows continuation after the original pretraining file is gone.
    checkpoint_ref = train_overlay.get("resume_from") or cli_checkpoint or train_overlay.get("pretrained_checkpoint")
    if not checkpoint_ref:
        raise ValueError("fine tune config requires train.pretrained_checkpoint or --pretrained-checkpoint")
    source_path, checkpoint = _load_checkpoint(checkpoint_ref)
    source_config = checkpoint.get("config")
    if not isinstance(source_config, Mapping):
        raise ValueError("pretrained checkpoint is missing its complete config")
    config = copy.deepcopy(dict(source_config))
    source_downsample = (source_config.get("train") or {}).get("downsample")
    if "downsample" in train_overlay and train_overlay["downsample"] != source_downsample:
        raise ValueError("fine tuning cannot change train.downsample from the pretrained checkpoint")
    _merge(config, overlay)
    config.setdefault("train", {}).pop("pretrained_checkpoint", None)
    output_dir = config["train"].get("output_dir")
    if output_dir and not Path(str(output_dir)).is_absolute():
        config["train"]["output_dir"] = str((REPO_ROOT / str(output_dir)).resolve())
    sources = (config.get("train_data") or {}).get("sources")
    if isinstance(sources, list):
        for source in sources:
            if isinstance(source, dict) and source.get("root"):
                root = Path(str(source["root"])).expanduser()
                if not root.is_absolute():
                    source["root"] = str((REPO_ROOT / root).resolve())
    config["train"]["finetune_source_checkpoint"] = str(source_path)
    return config, source_path


def main():
    args = parse_args()
    config, source_path = prepare_config(args.config, args.pretrained_checkpoint)
    log.info("Fine tune config: %s", config)
    trainer = ContactWorldModelFinetuneTrainer(config, source_path)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
