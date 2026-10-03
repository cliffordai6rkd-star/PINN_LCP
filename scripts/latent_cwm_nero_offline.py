"""Offline Nero integration example, using physical measured history/action.

The input .pt is a dict of batched physical q/dq/delta_q/tau, absolute EE
action, action_mask, history_timestamp_ns and action_chunk_timestamp_ns.
The action chunk must already be selected by native held action index + offset.
Alternatively provide explicit grid metadata under `explicit_grid`.
"""
import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.pinn_model.latent_contact_world_model import load_latent_checkpoint
from model.pinn_model.latent_pretrained import convert_scale


@torch.no_grad()
def infer_physical(model, physical, *, num_samples=8, seed=1234, integration_fn=None):
    if model.runtime_normalizer is None and model.wm_normalizer is not None:
        model.prepare_runtime_normalizers(physical=True)
    physical = {k:v.to(model.latent_mean.device) if torch.is_tensor(v) else v for k,v in physical.items()}
    positions = model.grid.positions(history_ns=physical.get("history_timestamp_ns"),
        action_ns=physical.get("action_chunk_timestamp_ns"), future_horizon=model.external_future_horizon,
        history_horizon=model.external_history_horizon, action_start_offset=model.action_start_offset,
        history_valid=physical.get("history_real_mask"), action_indices=physical.get("action_chunk_index"),
        explicit={k:v.to(model.latent_mean.device) for k,v in physical["explicit_grid"].items()}
                 if "explicit_grid" in physical else None)
    convert = model.runtime_normalizer.convert if model.runtime_normalizer is not None else (
        lambda key, value, inverse=False: convert_scale(key, value, model.wm_normalizer, inverse=inverse))
    batch = {key:convert(key, physical[key])
             for key in ("q","dq","delta_q","tau","action")}
    batch.update(positions)
    if "action_mask" in physical:
        batch["action_mask"] = physical["action_mask"]
    generator = torch.Generator(device=model.latent_mean.device).manual_seed(seed)
    noise = torch.randn(physical["q"].shape[0],num_samples,model.future_horizon,model.latent_dim,
                        device=model.latent_mean.device,generator=generator)
    output = model.sample(batch, num_samples=num_samples, source_noise=noise, integration_fn=integration_fn)
    output.update({key:convert(key,output[key+"_pred"].float(),inverse=True)
                   for key in ("q","tau")})
    output["future_grid_positions"] = model.prepare_batch(batch)["future_grid_positions"]
    # Keep the request anchor; inference completion does not redefine time zero.
    output["request_anchor_ns"] = physical["history_timestamp_ns"][:,-1] if "history_timestamp_ns" in physical else None
    output["request_anchor_grid_position"] = (physical["explicit_grid"]["anchor_grid_position"].to(model.latent_mean.device)
                                              if "explicit_grid" in physical else None)
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint",type=Path,required=True)
    parser.add_argument("--condition",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--device",default="cpu")
    parser.add_argument("--num-samples",type=int,default=8)
    parser.add_argument("--seed",type=int,default=1234)
    args=parser.parse_args()
    model,_=load_latent_checkpoint(args.checkpoint,device=args.device)
    condition=torch.load(args.condition,map_location="cpu",weights_only=False)
    result=infer_physical(model,condition,num_samples=args.num_samples,seed=args.seed)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save({k:v.cpu() if torch.is_tensor(v) else v for k,v in result.items()},args.output)
    print({k:list(result[k].shape) for k in ("q","tau","contact_probability")},"NFE",result["nfe"])


if __name__=="__main__":
    main()
