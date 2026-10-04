"""Explicit approximate pilot adapter: 25 Hz VLA states -> 100 Hz WM windows.

Windows are anchored ONLY at observed camera rows. Thus every interpolation
bracket used by a history ends at or before its anchor. Future action tokens
remain the original 25 Hz sequence. Missing contact labels are never invented.
"""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from data_process.causal_data_filter import filter_episode_values, normalize_dataloader_filters


def interpolate(timestamps, values, queries):
    return np.stack([np.interp(queries.reshape(-1), timestamps, values[:, i])
                     for i in range(values.shape[-1])], -1).reshape(*queries.shape, values.shape[-1]).astype(np.float32)


class InterpolatedContactDataset(torch.utils.data.Dataset):
    batch_collate = staticmethod(torch.utils.data.default_collate)

    def __init__(self, config, root):
        root = Path(root)
        info = json.loads((root/"meta/info.json").read_text())
        if abs(float(info["fps"])-25) > 1e-6:
            raise ValueError("this explicit pilot adapter requires a 25 Hz VLA export")
        import pyarrow.parquet as pq
        names = ["episode_index", "frame_index", "timing.master_timestamp_ns", "observation.joint",
                 "observation.velocity", "observation.torque", "action.joint", "action.ee_pose"]
        paths = sorted((root/"data").rglob("*.parquet"))
        if not paths:
            raise ValueError("no VLA parquet data found")
        columns = {key: [] for key in names}
        digest = hashlib.sha256()
        digest.update((root/"meta/info.json").read_bytes())
        for path in paths:
            digest.update(str(path.relative_to(root)).encode())
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(8*1024*1024), b""): digest.update(chunk)
            table = pq.read_table(path, columns=names)
            for key in names: columns[key].extend(table[key].to_pylist())
        arrays = {key: np.asarray(value) for key, value in columns.items()}
        episodes = []
        for identifier in sorted(np.unique(arrays["episode_index"]).tolist()):
            indices = np.flatnonzero(arrays["episode_index"] == identifier)
            indices = indices[np.argsort(arrays["frame_index"][indices])]
            episodes.append({key: value[indices] for key, value in arrays.items()})
        self._build(config, episodes)
        self.provenance.update(root=str(root.resolve()), input_sha256=digest.hexdigest(),
                               source_fps=info["fps"], source_frames=sum(len(e["episode_index"]) for e in episodes))

    @classmethod
    def from_episodes(cls, config, episodes):
        result = cls.__new__(cls)
        result._build(config, episodes)
        return result

    def _build(self, config, episodes):
        data = config["dataloader"]
        rate = float(data["high_fps"])
        if rate != 100 or float(data.get("expert_fps", 25)) != 25:
            raise ValueError("pilot requires the teacher's 100 Hz states and 25 Hz actions")
        if config["model"].get("inputs") != ["q", "dq", "delta_q", "tau"]:
            raise ValueError("pilot supports the four original GRU state inputs")
        if int(config["model"]["joint_dim"]) != 7 or int(config["model"]["action_dim"]) != 7:
            raise ValueError("pilot expects seven joints and xyz/xyzw action poses")
        if data.get("action_augmentation", {}).get("enabled", False):
            raise ValueError("pilot requires disabled action augmentation")
        if float(config.get("loss", {}).get("free_dynamics_weight", 0)) > 0:
            raise ValueError("pilot has no contact labels for free-dynamics masking")
        h, f, a = (int(data[k]) for k in ("state_history_horizon", "prediction_horizon", "action_condition_horizon"))
        offset = int(data.get("action_start_offset", 1))
        if offset < 1 or not data.get("pad_history", True):
            raise ValueError("pilot requires future action offset >=1 and padded early histories")
        filters = normalize_dataloader_filters(data)
        self.normalizer = None
        self.normalize_mode = data.get("normalize_mode")
        self.normalize_keys = data.get("normalize_lowdim_keys", [])
        pieces, meta = {}, []
        self.valid_indices, self.raw_idx_to_episode_start = [], {}
        raw_start, delta_max, source_periods = 0, 0., []
        # One-second filter warmup before each history, initialized by repeating
        # the first available observation. Full configured filters are applied:
        # VLA source preprocessing cannot be verified from its metadata.
        warmup = 100
        history_offsets = np.arange(-(h+warmup-1), 1, dtype=np.float64)/rate
        future_offsets = np.arange(1, f+1, dtype=np.float64)/rate
        for ep in episodes:
            identifier = int(ep["episode_index"][0])
            ns = np.asarray(ep["timing.master_timestamp_ns"], dtype=np.int64)
            time = (ns-ns[0]).astype(np.float64)*1e-9
            if len(time) < a+offset+1 or np.any(np.diff(time) <= 0):
                raise ValueError(f"episode {identifier}: too short or nonmonotonic timestamps")
            source_periods.extend(np.diff(time).tolist())
            q = np.asarray(ep["observation.joint"], dtype=np.float32)
            command = np.asarray(ep["action.joint"], dtype=np.float32)
            delta_max = max(delta_max, float(np.max(np.abs(command-q))))
            indices = np.arange(len(time))
            valid = indices[(indices+offset+a <= len(time)) & (time+f/rate <= time[-1]+1e-9)]
            if not len(valid):
                raise ValueError(f"episode {identifier}: no full future windows")
            anchor = time[valid]
            history_queries = anchor[:, None]+history_offsets[None]
            future_queries = anchor[:, None]+future_offsets[None]
            queries = np.concatenate((history_queries, future_queries), axis=1)
            query_offsets = np.concatenate((history_offsets, future_offsets))
            raw = {"q": q, "dq": np.asarray(ep["observation.velocity"], dtype=np.float32),
                   "tau": np.asarray(ep["observation.torque"], dtype=np.float32)}
            for key, values in raw.items():
                if values.shape != (len(time), 7) or not np.isfinite(values).all():
                    raise ValueError(f"episode {identifier}: invalid {key}")
                series = interpolate(time, values, queries)
                spec = filters.get(key, {})
                if spec.get("enabled", False):
                    series = filter_episode_values(query_offsets, series.transpose(1, 0, 2),
                                                   spec["operations"]).transpose(1, 0, 2)
                pieces.setdefault(key, []).append(torch.from_numpy(series[:, warmup:warmup+h].copy()))
                if key in {"q", "tau"}:
                    pieces.setdefault(key+"_future", []).append(torch.from_numpy(series[:, -f:].copy()))
            # Proxy measured tracking error: recorded joint action - q.
            # This export has action.joint == observation.joint at all source
            # rows. Consequently its proxy is zero there; true q_cmd is absent.
            delta = interpolate(time, command-q, history_queries)
            spec = filters.get("delta_q", {})
            if spec.get("enabled", False):
                delta = filter_episode_values(history_offsets, delta.transpose(1, 0, 2),
                                              spec["operations"]).transpose(1, 0, 2)
            pieces.setdefault("delta_q", []).append(torch.from_numpy(delta[:, -h:].copy()))
            actions = np.asarray(ep["action.ee_pose"], dtype=np.float32).copy()
            if actions.shape != (len(time), 7) or not np.isfinite(actions).all():
                raise ValueError(f"episode {identifier}: invalid actions")
            quat_norm = np.linalg.norm(actions[:, 3:], axis=1)
            if not np.allclose(quat_norm, 1, atol=1e-3):
                raise ValueError(f"episode {identifier}: invalid xyz/xyzw quaternion")
            actions[actions[:, 6] < 0, 3:] *= -1
            action_indices = valid[:, None]+offset+np.arange(a)[None]
            pieces.setdefault("action", []).append(torch.from_numpy(actions[action_indices]))
            pieces.setdefault("action_mask", []).append(torch.ones(len(valid), a, dtype=torch.bool))
            pieces.setdefault("history_valid_mask", []).append(torch.from_numpy(history_queries[:, -h:] >= 0))
            pieces.setdefault("episode_index", []).append(torch.full((len(valid),), identifier, dtype=torch.int64))
            pieces.setdefault("anchor_timestamp_ns", []).append(torch.from_numpy(ns[valid].copy()))
            meta.append({"episode_index": identifier, "dataset_from_index": raw_start})
            self.valid_indices.extend((raw_start+valid).tolist())
            self.raw_idx_to_episode_start.update({raw_start+int(i): raw_start for i in valid})
            raw_start += len(time)
        if not meta:
            raise ValueError("empty episode set")
        self.values = {key: torch.cat(values) for key, values in pieces.items()}
        self.dataset = SimpleNamespace(meta=SimpleNamespace(episodes=meta))
        self.provenance = {"type": "approximate_25hz_vla_interpolation_pilot", "target_state_fps": rate,
                           "anchors": "original recorded rows only; no future observation used by history",
                           "interpolation": "linear q/dq/tau and recorded joint-action-minus-q; original 25hz future EE actions",
                           "delta_q": "interpolated recorded action.joint minus observation.joint; zero in supplied export; not verified q_cmd tracking error",
                           "recorded_action_joint_minus_q_abs_max": delta_max,
                           "contact_labels_available": False, "filter_policy": "full teacher causal filters on history/targets with 1s warmup; source preprocessing unverified",
                           "future_targets": "filtered interpolated q/tau; approximate, not native 100hz measurements",
                           "source_period_median_s": float(np.median(source_periods)),
                           "episodes": len(meta), "windows": len(self.valid_indices)}

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, index):
        sample = {key: values[index] for key, values in self.values.items()}
        if self.normalizer and self.normalize_mode:
            for key in self.normalize_keys:
                for field in (key, key+"_future"):
                    if field in sample:
                        sample[field] = getattr(self.normalizer, self.normalize_mode+"_normalize")(key, sample[field])
        return sample
