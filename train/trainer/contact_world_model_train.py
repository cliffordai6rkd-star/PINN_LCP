"""Training entry point for the q/tau/action -> contact WM."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from train.trainer.torque_world_model_train import TorqueWorldModelTrainer


logging.basicConfig(level=logging.INFO)
log = logging.getLogger("contact_world_model_train")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the q/tau/action-conditioned three-phase contact WM."
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=Path("config/train_cfg/contact_world_model.yaml"),
    )
    return parser.parse_args()


class ContactWorldModelTrainer(TorqueWorldModelTrainer):
    """Named wrapper keeping Contact WM checkpoints separate from torque WM."""

    pass


def main():
    args = parse_args()
    with args.config.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if (config.get("model") or {}).get("state_contract") != "q_tau_contact":
        raise ValueError(
            "contact_world_model_train requires model.state_contract=q_tau_contact"
        )
    log.info("contact WM config: %s", config)
    trainer = ContactWorldModelTrainer(config)
    summary = trainer.train()
    log.info("\n%s", trainer.format_summary(summary))


if __name__ == "__main__":
    main()
