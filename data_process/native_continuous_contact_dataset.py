"""Native WM q/tau supervision without pretending missing contact labels exist.

Only the dataset's label construction is disabled. The model/normalizer/contact
contract is untouched. This path supports direct-state GRU Flow, not a contact-
conditioned future latent encoder, which requires genuine contact targets.
"""
import copy
import hashlib
import json

from data_process.contact_world_model_dataset import ContactWorldModelDataset


class NativeContinuousContactDataset(ContactWorldModelDataset):
    def __init__(self, config, normalizer=None, compute_normalizer=False):
        loader = copy.deepcopy(config)
        loader.setdefault("contact_gate", {})["enabled"] = False
        data = loader["dataloader"]
        data.setdefault("tau_ext_generation", {})["enabled"] = False
        data.setdefault("high_keys", {})["tau_ext"] = None
        fields = (loader.get("train_data") or {}).get("v3_fields")
        if fields is not None:
            fields["tau_ext"] = None
        super().__init__(loader, normalizer=normalizer, compute_normalizer=compute_normalizer)
        digest = hashlib.sha256()
        for spec in self.lerobot_source_specs:
            root = spec["root"]
            digest.update((root/"meta/info.json").read_bytes())
            for path in sorted((root/"data").rglob("*.parquet")):
                digest.update(str(path.relative_to(root)).encode())
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8*1024*1024), b""): digest.update(chunk)
            timeline_path = root/"meta/world_model_timeline.json"
            if not timeline_path.exists():
                raise ValueError("native unlabeled adapter requires WM timeline metadata")
            timeline = json.loads(timeline_path.read_text())
            if float(timeline["nominal_lerobot_fps"]) != self.high_fps:
                raise ValueError("native data rate differs from declared WM rate")
            digest.update(timeline_path.read_bytes())
        self.provenance = {"type": "native_wm_continuous_targets_without_contact_labels",
                           "state_fps": self.high_fps, "input_sha256": digest.hexdigest(),
                           "interpolation": False, "delta_q": "recorded observation.delta_q",
                           "contact_labels_available": False,
                           "future_targets": "native q/tau, teacher preprocessing and normalization",
                           "label_policy": "no tau_ext/free/alignment/contact labels synthesized; GRU Flow only",
                           "original_label_validity_exclusions_replayed": False,
                           "windows": len(self), "episodes": len(self.dataset.meta.episodes)}

    def _build_sample(self, high_idx):
        sample = super()._build_sample(high_idx)
        for key in ("contact", "contact_future", "future_phase", "free_dynamics_mask"):
            sample.pop(key, None)
        return sample
