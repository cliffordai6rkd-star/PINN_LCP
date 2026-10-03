"""Independent adapter: the original WM dataset retains all default semantics."""
import logging

import torch

from data_process.contact_world_model_dataset import ContactWorldModelDataset
from data_process.relative_time_grid import RelativeTimeGrid, GridMetadataError

log = logging.getLogger(__name__)


class LatentContactWorldModelDataset(ContactWorldModelDataset):
    def __init__(self, config, normalizer=None, compute_normalizer=False):
        if (config.get("dataloader") or {}).get("action_augmentation", {}).get("temporal_jitter", 0):
            raise ValueError("temporal action augmentation requires matching per-token grid metadata")
        if (config.get("dataloader") or {}).get("pad_future", False):
            raise ValueError("latent codec requires real future targets, pad_future must be false")
        super().__init__(config, normalizer=normalizer, compute_normalizer=False)
        self.sample_rate_hz = float(self.high_fps)
        self.grid = RelativeTimeGrid.from_config(config)
        policy = self.data_config.get("grid_invalid_window_policy", "error")
        if policy not in {"error", "drop"}:
            raise ValueError("grid_invalid_window_policy must be error or drop")
        original = len(self.valid_indices)
        retained = []
        report = []
        # Check a vectorized episode-local window table once, before splitting
        # or fitting normalizers. Never infer phase from concatenated row IDs.
        for episode in self.episodes:
            start, end = int(episode["dataset_from_index"]), int(episode["dataset_to_index"])
            anchors = [i for i in self.valid_indices if start <= i < end]
            table = self._action_tables.get(id(episode))
            if table is None:
                raise GridMetadataError("latent V3 adapter requires native unique action tables and timing metadata")
            invalid = 0
            for offset in range(0, len(anchors), 4096):
                rows = torch.tensor(anchors[offset:offset+4096], dtype=torch.long)
                hi = rows[:, None] + torch.arange(1-self.history_horizon, 1)
                history_real = hi >= start
                hi = hi.clamp_min(start)
                fi = rows[:, None] + torch.arange(1, self.future_horizon+1)
                current = torch.searchsorted(table["indices"], self.action_indices[rows])
                first = current + self.action_start_offset
                if self.inference_delay_ns:
                    first = torch.searchsorted(table["times"], table["times"][first] + self.inference_delay_ns)
                ai = first[:, None] + torch.arange(self.action_condition_horizon)
                good = self.grid.valid_windows(self.high_timestamps[hi], table["times"][ai],
                    self.high_timestamps[fi], action_start_offset=self.action_start_offset,
                    history_valid=history_real, action_indices=table["indices"][ai])
                invalid += int((~good).sum())
                retained.extend(rows[good].tolist())
            report.append({"episode_start": start, "windows": len(anchors), "invalid_windows": invalid})
        self.grid_audit = {"policy": policy, "original_windows": original,
                           "retained_windows": len(retained), "invalid_windows": original-len(retained),
                           "episodes": report, "contract": self.grid.contract()}
        if len(retained) != original:
            log.warning("relative-grid cadence audit: %s", {k:v for k,v in self.grid_audit.items() if k != "episodes"})
            if policy == "error":
                raise GridMetadataError(f"{original-len(retained)}/{original} windows violate relative-grid cadence; "
                                        "inspect data or explicitly select grid_invalid_window_policy=drop")
        if not retained:
            raise GridMetadataError("no cadence-compatible windows remain")
        self.valid_indices = retained
        if compute_normalizer:
            self.fit_normalizer(range(len(self)))

    def _build_sample(self, high_idx):
        sample = super()._build_sample(high_idx)
        # History validity in the original dataset also encodes contact-label
        # availability; grid padding validity is only about actual rows.
        episode = self._episode_for_index(high_idx)
        real = (torch.arange(high_idx-self.history_horizon+1, high_idx+1,
                             device=sample["q"].device) >= int(episode["dataset_from_index"]))
        positions = self.grid.positions(history_ns=sample["history_timestamp_ns"][None],
            action_ns=sample["action_chunk_timestamp_ns"][None], future_horizon=self.future_horizon,
            action_start_offset=self.action_start_offset, history_valid=real[None],
            action_indices=sample["action_chunk_index"][None])
        sample.update({key: value[0] for key, value in positions.items()})
        return sample
