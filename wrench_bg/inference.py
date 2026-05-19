
import torch
import yaml

ckpt_path = "outputs/wrench_background_1/checkpoints/epoch_238_val_loss_0.0477.pt"

ckpt = torch.load(ckpt_path, map_location="cuda:0")

print("ckpt keys:", ckpt.keys())
print("epoch:", ckpt.get("epoch"))
print("global_step:", ckpt.get("global_step"))
print("monitor_key:", ckpt.get("monitor_key"))
print("monitor_score:", ckpt.get("monitor_score"))

print("\nconfig:")
print(yaml.safe_dump(ckpt["config"], sort_keys=False, allow_unicode=True))
