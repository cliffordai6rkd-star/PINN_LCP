"""Short CPU training checks on two real, automatically labeled WM episodes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import yaml

from train.trainer.contact_world_model_train import ContactWorldModelTrainer
from train.trainer.latent_contact_world_model_train import LatentContactWorldModelTrainer


class TwoEpisodeSmoke:
    def build_dataset(self):
        dataset = super().build_dataset()
        dataset.episodes = dataset.episodes[:2]
        if len(dataset.episodes) != 2:
            raise ValueError('Smoke training needs at least two episodes')
        selected = []
        for episode in dataset.episodes:
            anchors = [i for i in dataset.valid_indices if episode['dataset_from_index'] <= i < episode['dataset_to_index']]
            stride = max(1, len(anchors)//128)
            selected.extend(anchors[::stride][:128])
        dataset.valid_indices = selected
        return dataset


class ContactSmoke(TwoEpisodeSmoke, ContactWorldModelTrainer):
    pass


class LatentSmoke(TwoEpisodeSmoke, LatentContactWorldModelTrainer):
    pass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/wm_tau_label_integration/smoke')
    parser.add_argument('--steps', type=int, default=2)
    parser.add_argument('--cwm-config', type=Path, default=ROOT/'config/train_cfg/pretrain/xarm/cwm_peel_cucumber_100hz_40step.yaml')
    parser.add_argument('--latent-config', type=Path, default=ROOT/'config/train_cfg/latent_cwm_xarm_peel_cucumber_100hz_40step.yaml')
    args = parser.parse_args()
    torch.set_num_threads(4)
    summary = {}
    for family, config_path, trainer_type in (
        ('contact', args.cwm_config, ContactSmoke), ('latent', args.latent_config, LatentSmoke)):
        cfg = yaml.safe_load(config_path.read_text())
        cfg['train'].update(device='cpu', output_dir=str(args.output/family), batch_size=8,
            num_workers=0, val_num_workers=0, pin_memory=False, persistent_workers=False,
            val_episode_indices=[1], max_optimizer_steps=args.steps, num_epochs=10,
            checkpoint_every_steps=args.steps, val_every=100, gradient_every=1,
            amp={'enabled': False}, wandb={'enabled': False},
            scheduler={'name': 'cosine', 'warmup_steps': 0, 'eta_min': 1e-6},
            probabilistic_validation={'enabled': False}, checkpoint_visualization={'enabled': False})
        cfg.setdefault('codec', {})['max_optimizer_steps'] = args.steps
        trainer = trainer_type(cfg)
        trainer.train()
        summary[family] = dict(optimizer_steps=trainer.global_step,
            codec_steps=getattr(trainer, 'codec_step', None), windows=len(trainer.dataset),
            label_cache_hits=trainer.dataset.tau_label_report['cache_hits'],
            label_contract=trainer.dataset.tau_label_report['label_contract_sha256'])
        print('SMOKE', family, json.dumps(summary[family]), flush=True)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')


if __name__ == '__main__':
    main()
