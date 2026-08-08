import unittest
from types import SimpleNamespace

import torch

from model.tau_f_sequence import TauFSequenceRegressor
from train.base_trainer import BaseTrainer, ModelEMA
from train.nomalizer import Normalizer
from train.trainer.tau_f_sequence_train import SampleIndexDataset, TauFTrainer


def make_config(architecture="lstm", inputs=None):
    return {
        "model": {
            "architecture": architecture,
            "inputs": inputs or ["q", "dq", "ddq", "tau"],
            "input_dims": {"q": 7, "dq": 7, "ddq": 7, "tau": 7},
            "output_dim": 7,
            "target_key": "tau_f",
            "hidden_dim": 16,
            "num_layers": 2,
            "head_hidden_dim": 24,
            "head_num_layers": 2,
            "dropout": 0.0,
        },
        "loss": {"joint_weights": None},
        "train": {},
    }


def make_batch(batch_size=4, horizon=8):
    return {
        key: torch.randn(batch_size, horizon, 7)
        for key in ("q", "dq", "ddq", "tau", "tau_f")
    }


class TauFSequenceRegressorTest(unittest.TestCase):
    def test_model_ema_smooths_parameters_and_copies_buffers(self):
        model = torch.nn.BatchNorm1d(2)
        ema = ModelEMA(model, decay=0.5)

        with torch.no_grad():
            model.weight.fill_(3.0)
            model.running_mean.fill_(4.0)
        ema.update(model, step=1)

        torch.testing.assert_close(
            ema.model.weight,
            torch.full_like(ema.model.weight, 2.0),
        )
        torch.testing.assert_close(
            ema.model.running_mean,
            model.running_mean,
        )
        self.assertFalse(any(p.requires_grad for p in ema.model.parameters()))

    def test_lstm_and_gru_forward_backward(self):
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauFSequenceRegressor(make_config(architecture))
                out = model(make_batch())

                self.assertEqual(out["tau_f_pred"].shape, (4, 7))
                self.assertEqual(out["tau_f_target"].shape, (4, 7))
                out["tau_f_pred"].square().mean().backward()
                self.assertTrue(
                    any(parameter.grad is not None for parameter in model.parameters())
                )

    def test_configurable_input_subset_changes_recurrent_input_size(self):
        model = TauFSequenceRegressor(make_config(inputs=["q", "tau"]))
        batch = make_batch()
        batch.pop("dq")
        batch.pop("ddq")

        out = model(batch)

        self.assertEqual(model.recurrent.input_size, 14)
        self.assertEqual(out["tau_f_pred"].shape, (4, 7))

    def test_target_is_last_window_frame(self):
        model = TauFSequenceRegressor(make_config())
        batch = make_batch(batch_size=2, horizon=5)
        batch["tau_f"] = torch.arange(70, dtype=torch.float32).reshape(2, 5, 7)

        out = model(batch)

        torch.testing.assert_close(out["tau_f_target"], batch["tau_f"][:, -1])

    def test_trainer_computes_normalized_and_physical_mae(self):
        trainer = TauFTrainer(make_config())
        trainer.model = trainer.build_model()

        loss, out = trainer.compute_loss(make_batch())

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertEqual(
            set(out["loss_dict"]),
            {
                "mse",
                "mae",
                "mae_nm",
                "mae_nm_j1",
                "mae_nm_j2",
                "mae_nm_j3",
                "mae_nm_j4",
                "mae_nm_j5",
                "mae_nm_j6",
                "mae_nm_j7",
            },
        )

    def test_physical_mae_uses_target_denormalization(self):
        trainer = TauFTrainer(make_config())
        std = torch.arange(1, 8, dtype=torch.float32)
        trainer.dataset = SimpleNamespace(
            normalize_mode="gaussian",
            normalize_lowdim_keys=["tau_f"],
            normalizer=Normalizer(
                {
                    "tau_f": {
                        "mean": torch.arange(7, dtype=torch.float32),
                        "std": std,
                    }
                }
            ),
        )
        target = torch.zeros(2, 7)
        prediction = torch.ones(2, 7)

        metrics = trainer._physical_mae_metrics(prediction, target)

        expected_joint_mae = std + trainer.dataset.normalizer.eps
        for joint_index, expected in enumerate(expected_joint_mae, start=1):
            torch.testing.assert_close(
                metrics[f"mae_nm_j{joint_index}"],
                expected,
            )
        torch.testing.assert_close(metrics["mae_nm"], expected_joint_mae.mean())

    def test_validation_mae_is_weighted_by_sample_count(self):
        class FixedPredictionModel(torch.nn.Module):
            def forward(self, batch):
                return {
                    "tau_f_pred": batch["prediction"],
                    "tau_f_target": batch["target"],
                }

        samples = [
            {
                "prediction": torch.full((7,), error),
                "target": torch.zeros(7),
            }
            for error in (1.0, 1.0, 3.0)
        ]
        trainer = TauFTrainer(make_config())
        trainer.device = "cpu"
        trainer.dataset = SimpleNamespace(
            normalize_mode=None,
            normalize_lowdim_keys=[],
        )
        trainer.model = FixedPredictionModel()
        trainer.val_loader = torch.utils.data.DataLoader(samples, batch_size=2)

        val_loss = trainer.validate_one_epoch(epoch=0)

        self.assertAlmostEqual(val_loss, 11.0 / 3.0)
        self.assertAlmostEqual(
            trainer.last_val_epoch_metrics["mae_nm"],
            5.0 / 3.0,
        )
        for joint_index in range(1, 8):
            self.assertAlmostEqual(
                trainer.last_val_epoch_metrics[f"mae_nm_j{joint_index}"],
                5.0 / 3.0,
            )

    def test_train_eval_disables_training_mode_noise(self):
        class ModeAwareModel(torch.nn.Module):
            def forward(self, batch):
                prediction = (
                    torch.ones_like(batch["target"])
                    if self.training
                    else torch.zeros_like(batch["target"])
                )
                return {
                    "tau_f_pred": prediction,
                    "tau_f_target": batch["target"],
                }

        trainer = TauFTrainer(make_config())
        trainer.device = "cpu"
        trainer.train_eval_enabled = True
        trainer.dataset = SimpleNamespace(
            normalize_mode=None,
            normalize_lowdim_keys=[],
        )
        trainer.model = ModeAwareModel().train()
        trainer.loader = torch.utils.data.DataLoader(
            [{"target": torch.zeros(7)} for _ in range(3)],
            batch_size=2,
        )

        train_eval_loss = trainer.evaluate_train_one_epoch(epoch=0)

        self.assertEqual(train_eval_loss, 0.0)
        self.assertEqual(trainer.last_train_eval_epoch_metrics["mae_nm"], 0.0)

    def test_invalid_architecture_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            TauFSequenceRegressor(make_config(architecture="transformer"))

    def test_each_window_starts_from_a_fresh_recurrent_state(self):
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauFSequenceRegressor(make_config(architecture))
                model.eval()
                batch = make_batch(batch_size=3, horizon=6)

                first = model(batch)["tau_f_pred"]
                model(make_batch(batch_size=3, horizon=6))
                second = model(batch)["tau_f_pred"]

                torch.testing.assert_close(first, second)

    def test_model_does_not_expose_a_streaming_hidden_state_api(self):
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauFSequenceRegressor(make_config(architecture))
                out = model(make_batch())

                self.assertFalse(hasattr(model, "forward_step"))
                self.assertFalse(hasattr(model, "init_recurrent_state"))
                self.assertNotIn("recurrent_state", out)
                self.assertNotIn("sequence_features", out)

    def test_next_topology_uses_two_recurrent_and_two_mlp_layers(self):
        config = make_config("lstm")
        config["model"].update(
            hidden_dim=128,
            num_layers=2,
            head_hidden_dim=256,
            head_num_layers=2,
        )

        model = TauFSequenceRegressor(config)

        self.assertIsInstance(model.recurrent, torch.nn.LSTM)
        self.assertEqual(model.recurrent.num_layers, 2)
        self.assertEqual(model.recurrent.hidden_size, 128)
        self.assertEqual(model.head[0].in_features, 128)
        self.assertEqual(model.head[0].out_features, 256)
        self.assertEqual(model.head[-1].in_features, 256)
        self.assertEqual(model.head[-1].out_features, 7)

    def test_invalid_history_mode_fails_fast(self):
        config = make_config()
        config["model"]["history_mode"] = "stateful_stream"

        with self.assertRaisesRegex(ValueError, "stateless_sliding_window"):
            TauFSequenceRegressor(config)


class FakeWindowDataset(torch.utils.data.Dataset):
    def __init__(self, episode_length=30, horizon=5):
        self.horizon = horizon
        self.valid_indices = list(range(horizon - 1, episode_length))
        self.raw_idx_to_episode_start = {idx: 0 for idx in self.valid_indices}
        self.raw_idx_to_episode_end = {
            idx: episode_length for idx in self.valid_indices
        }
        self.dataset = SimpleNamespace(
            meta=SimpleNamespace(
                episodes=[
                    {
                        "dataset_from_index": 0,
                        "dataset_to_index": episode_length,
                    }
                ]
            )
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        return self.valid_indices[index]


class PurgedTemporalSplitTest(unittest.TestCase):
    def test_train_and_validation_windows_share_no_raw_frames(self):
        trainer = BaseTrainer(
            {"train": {"val_ratio": 0.2, "split_mode": "purged_temporal"}}
        )
        trainer.dataset = FakeWindowDataset(episode_length=30, horizon=5)

        train_dataset, val_dataset = trainer.split_dataset_purged_temporal()

        def covered_frames(subset):
            frames = set()
            for sample_idx in subset.indices:
                raw_idx = trainer.dataset.valid_indices[sample_idx]
                frames.update(range(raw_idx - trainer.dataset.horizon + 1, raw_idx + 1))
            return frames

        train_frames = covered_frames(train_dataset)
        val_frames = covered_frames(val_dataset)
        self.assertTrue(train_frames)
        self.assertTrue(val_frames)
        self.assertTrue(train_frames.isdisjoint(val_frames))
        self.assertEqual(max(train_frames) + 1, min(val_frames))


class DeviceCacheTest(unittest.TestCase):
    def test_index_dataset_returns_only_sample_indices(self):
        dataset = SampleIndexDataset([4, 9])

        self.assertEqual(dataset[0], {"sample_idx": 4})
        self.assertEqual(dataset[1], {"sample_idx": 9})

    def test_cached_batch_gathers_windows_and_current_target(self):
        config = make_config(inputs=["q", "tau"])
        config["dataloader"] = {"horizon": 3}
        trainer = TauFTrainer(config)
        trainer.dataset = SimpleNamespace(horizon=3)

        values = torch.arange(70, dtype=torch.float32).reshape(10, 7)
        trainer.tensor_cache = {
            "q": values,
            "tau": values + 100,
            "tau_f": values + 200,
        }
        trainer.valid_raw_indices_device = torch.tensor([2, 3, 4])
        trainer.episode_starts_device = torch.tensor([0, 0, 0])

        batch = trainer._cached_batch(torch.tensor([0, 2]))

        torch.testing.assert_close(batch["q"][0], values[0:3])
        torch.testing.assert_close(batch["q"][1], values[2:5])
        torch.testing.assert_close(
            batch["tau_f"],
            torch.stack([values[2] + 200, values[4] + 200]),
        )

    def test_cached_batch_left_pads_early_history_with_zeros(self):
        config = make_config(inputs=["q"])
        config["dataloader"] = {"horizon": 3}
        trainer = TauFTrainer(config)
        trainer.dataset = SimpleNamespace(horizon=3)
        values = torch.arange(28, dtype=torch.float32).reshape(4, 7)
        trainer.tensor_cache = {"q": values, "tau_f": values + 100}
        trainer.valid_raw_indices_device = torch.tensor([0, 1])
        trainer.episode_starts_device = torch.tensor([0, 0])

        batch = trainer._cached_batch(torch.tensor([0, 1]))

        torch.testing.assert_close(
            batch["q"][0],
            torch.stack([torch.zeros(7), torch.zeros(7), values[0]]),
        )
        torch.testing.assert_close(
            batch["q"][1],
            torch.stack([torch.zeros(7), values[0], values[1]]),
        )
        torch.testing.assert_close(
            batch["tau_f"],
            torch.stack([values[0] + 100, values[1] + 100]),
        )

    def test_normalizer_stats_use_training_frames_only(self):
        trainer = TauFTrainer(make_config(inputs=["q"]))
        trainer.dataset = SimpleNamespace(
            normalize_mode="gaussian",
            normalize_lowdim_keys=["q"],
        )
        values = torch.tensor([[0.0], [2.0], [100.0], [200.0]])

        stats = trainer._normalizer_stats({"q": values}, training_frames=[0, 1])

        torch.testing.assert_close(stats["q"]["mean"], torch.tensor([1.0]))


if __name__ == "__main__":
    unittest.main()
