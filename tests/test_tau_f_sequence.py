import unittest
from types import SimpleNamespace

import torch

from model.tau_f_sequence import TauFSequenceRegressor
from train.base_trainer import BaseTrainer
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

    def test_trainer_computes_mse_and_mae(self):
        trainer = TauFTrainer(make_config())
        trainer.model = trainer.build_model()

        loss, out = trainer.compute_loss(make_batch())

        self.assertEqual(loss.ndim, 0)
        self.assertGreaterEqual(loss.item(), 0.0)
        self.assertEqual(set(out["loss_dict"]), {"mse", "mae"})

    def test_invalid_architecture_fails_fast(self):
        with self.assertRaisesRegex(ValueError, "architecture"):
            TauFSequenceRegressor(make_config(architecture="transformer"))

    def test_streaming_steps_match_full_sequence(self):
        batch = make_batch(batch_size=3, horizon=6)

        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauFSequenceRegressor(make_config(architecture))
                model.eval()

                full_out = model(batch)
                state = model.init_recurrent_state(batch_size=3)
                step_features = []
                for step in range(6):
                    step_batch = {
                        key: batch[key][:, step]
                        for key in model.active_inputs
                    }
                    step_out = model.forward_step(step_batch, state)
                    state = step_out["recurrent_state"]
                    step_features.append(step_out["sequence_features"])

                torch.testing.assert_close(
                    torch.cat(step_features, dim=1),
                    full_out["sequence_features"],
                )
                torch.testing.assert_close(
                    step_out["tau_f_pred"],
                    full_out["tau_f_pred"],
                )

    def test_recurrent_state_shape_and_detach(self):
        for architecture in ("lstm", "gru"):
            with self.subTest(architecture=architecture):
                model = TauFSequenceRegressor(make_config(architecture))
                state = model.init_recurrent_state(batch_size=4)
                states = state if isinstance(state, tuple) else (state,)

                self.assertEqual(len(states), 2 if architecture == "lstm" else 1)
                for tensor in states:
                    self.assertEqual(tensor.shape, (2, 4, 16))

                step_batch = {
                    key: value[:, 0]
                    for key, value in make_batch().items()
                    if key in model.active_inputs
                }
                next_state = model.forward_step(step_batch, state)["recurrent_state"]
                detached = model.detach_recurrent_state(next_state)
                detached_states = detached if isinstance(detached, tuple) else (detached,)
                self.assertTrue(all(not tensor.requires_grad for tensor in detached_states))


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
