import argparse
import logging
from pathlib import Path

import yaml

from train.trainer.tau_f_sequence_train import TauFTrainer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a NEXT-style free-space inverse-dynamics model from joint "
            "position history."
        )
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/tau_free_sequence_v2.yaml"),
        help="Path to the V2 training config.",
    )
    return parser.parse_args()


class TauFreeSequenceTrainerV2(TauFTrainer):
    """Train stateless proprioceptive windows against free-space torque."""

    def __init__(self, config):
        self._validate_v2_contract(config)
        super().__init__(config)

    def build_dataset(self):
        dataset = super().build_dataset()
        available_columns = set(dataset.stats_dataset.column_names)
        data_config = self.config.get("dataloader") or {}
        model_config = self.config.get("model") or {}
        lowdim_keys = data_config.get("lowdim_keys") or {}
        required_keys = list(
            dict.fromkeys(
                list(model_config.get("inputs") or [])
                + [str(model_config.get("target_key", "tau"))]
                + list(data_config.get("normalize_lowdim_keys") or [])
            )
        )
        missing = {
            key: lowdim_keys.get(key)
            for key in required_keys
            if lowdim_keys.get(key) not in available_columns
        }
        if missing:
            details = ", ".join(
                f"{key} -> {column}" for key, column in missing.items()
            )
            hint = ""
            if "delta_q" in missing:
                hint = (
                    " Current data has no commanded joint position from which to "
                    "construct delta_q. Remove delta_q from model.inputs, or "
                    "reconvert data containing a meaningful q_cmd."
                )
            raise ValueError(
                f"tau-free V2 dataset is missing required columns: {details}."
                f"{hint}"
            )
        return dataset

    @staticmethod
    def _validate_v2_contract(config):
        data_config = config.get("dataloader") or {}
        model_config = config.get("model") or {}
        inputs = list(model_config.get("inputs") or [])
        target_key = str(model_config.get("target_key", ""))
        architecture = str(model_config.get("architecture", "")).lower()
        horizon = int(data_config.get("horizon", 0))
        pad_history = bool(data_config.get("pad_history", False))
        contact_free = data_config.get("contact_free")

        allowed_inputs = {"q", "dq", "delta_q"}
        unknown_inputs = sorted(set(inputs) - allowed_inputs)
        if not inputs or inputs[0] != "q" or unknown_inputs:
            raise ValueError(
                "tau-free V2 model.inputs must start with q and contain only "
                "q, dq, and delta_q; measured torque and ddq must not be inputs."
            )
        if len(set(inputs)) != len(inputs):
            raise ValueError("tau-free V2 model.inputs must not contain duplicates.")
        if target_key != "tau":
            raise ValueError(
                "tau-free V2 requires model.target_key=tau so the label is the "
                "measured motor torque from contact-free motion."
            )
        if architecture != "lstm":
            raise ValueError(
                "tau-free V2 requires model.architecture=lstm to match the "
                "selected NEXT architecture."
            )
        if horizon != 50:
            raise ValueError("tau-free V2 requires dataloader.horizon=50.")
        if pad_history:
            raise ValueError(
                "tau-free V2 requires dataloader.pad_history=false so every "
                "supervised target has 50 real history frames."
            )
        if contact_free is not True:
            raise ValueError(
                "tau-free V2 requires dataloader.contact_free=true; measured "
                "torque is a valid tau_free label only on contact-free data."
            )


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    log.info("tau-free sequence V2 train config: %s", config)
    trainer = TauFreeSequenceTrainerV2(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
