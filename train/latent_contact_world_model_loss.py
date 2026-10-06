"""Separate deterministic codec reconstruction and latent velocity objectives."""
import math
import torch
import torch.nn.functional as F

from train.contact_world_model_loss import ContactWorldModelLoss


class LatentContactWorldModelLoss:
    contact_state_count = 3

    def __init__(self, config):
        loss = config.get("loss") or {}
        self.lambda_free = float(loss.get("lambda_free", 0.1))
        self.codec_contact_weight = float((config.get("codec") or {}).get("contact_weight", 1.0))
        self.contact_class_weights = (config.get("contact_gate") or {}).get("class_weights", [1.,1.,1.])
        self.contact_class_weights_is_auto = self.contact_class_weights == "auto"
        sft = config.get("sft") or {}
        self.sft_fm_weight = float(sft.get("lambda_fm", 1.0))
        self.sft_reconstruction_weight = float(sft.get("lambda_reconstruction", 1.0))
        if any(not math.isfinite(v) or v <= 0 for v in (self.sft_fm_weight, self.sft_reconstruction_weight)):
            raise ValueError("SFT FM and reconstruction weights must be finite and positive")
        if any(not math.isfinite(v) or v < 0 for v in (self.lambda_free, self.codec_contact_weight)):
            raise ValueError("loss weights must be finite and nonnegative")

    def set_contact_class_weights(self, weights):
        if len(weights) != 3 or any(not math.isfinite(float(v)) or v <= 0 for v in weights):
            raise ValueError("three finite positive class weights required")
        self.contact_class_weights = list(weights)
        self.contact_class_weights_is_auto = False

    @staticmethod
    def weighted_mean(values, batch):
        weights = batch.get("importance_weight", torch.ones_like(values)).to(values.device).float()
        if weights.shape != values.shape or not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("importance_weight must be finite nonnegative [B]")
        return (values*weights).mean()

    free_dynamics_loss = ContactWorldModelLoss.free_dynamics_loss

    def codec_loss(self, out, batch):
        batch = out.get("_prepared_batch", batch)
        losses = {key:(out[key+"_pred"].float()-batch[key+"_future"].float()).square().mean((1,2))
                  for key in ("q", "tau")}
        labels = batch["contact_future"][...,0].long()
        ce = F.cross_entropy(out["contact_logits"].float().transpose(1,2), labels, reduction="none")
        class_weights = ce.new_tensor(self.contact_class_weights or [1.,1.,1.])
        contact = self.weighted_mean((ce*class_weights[labels]).mean(1), batch)
        q, tau = (self.weighted_mean(losses[k], batch) for k in ("q", "tau"))
        total = q + tau + self.codec_contact_weight*contact
        return total, {"total_loss":total.detach(), "codec_q_mse":q.detach(), "codec_tau_mse":tau.detach(),
                       "codec_contact_ce":ce.mean().detach(), "codec_contact_loss":contact.detach()}

    def flow_loss(self, out, batch):
        batch = out.get("_prepared_batch", batch)
        target = out["flow_velocity_target"].detach()
        fm = self.weighted_mean((out["flow_velocity_pred"].float()-target.float()).square().mean((1,2)), batch)
        free, count = (self.free_dynamics_loss(out, batch) if "free_tau_pred" in out and self.lambda_free > 0
                       else (fm.new_zeros(()), fm.new_zeros(())))
        total = fm + self.lambda_free*free
        return total, {"total_loss":total.detach(), "latent_fm_loss":fm.detach(),
                       "free_auxiliary_loss":free.detach(), "free_auxiliary_count":count.detach(),
                       "free_auxiliary_contribution":(self.lambda_free*free).detach()}

    def __call__(self, out, batch):
        if "reconstruction" in out:
            return self.sft_loss(out, batch)
        return self.flow_loss(out, batch) if "flow_velocity_pred" in out else self.codec_loss(out, batch)

    def sft_loss(self, out, batch):
        flow, metrics = self.flow_loss(out, batch)
        reconstruction, rec_metrics = self.codec_loss(out["reconstruction"], out.get("_prepared_batch", batch))
        total = self.sft_fm_weight*flow+self.sft_reconstruction_weight*reconstruction
        metrics.update({"sft_reconstruction_loss": reconstruction.detach(),
                        "sft_reconstruction_contribution": (self.sft_reconstruction_weight*reconstruction).detach(),
                        **{key.replace("codec_", "sft_"): value for key, value in rec_metrics.items() if key != "total_loss"},
                        "total_loss": total.detach()})
        return total, metrics
