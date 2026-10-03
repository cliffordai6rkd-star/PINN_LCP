# Latent CARS-WM: inverse Gaussian source and few-step inference

This post-training path starts from a completed latent CARS-WM Heun checkpoint. It fits a condition-dependent diagonal Gaussian over the entire future latent trajectory to inverse sources, then distills the flow into a shorter Heun sampler. The source network starts at exactly N(0, I), so attaching it does not change the original sampler before training. The condition concatenates pooled history and masked pooled action tokens. The deployed checkpoint uses the existing `load_latent_checkpoint` and `sample` APIs.

The inverse is a fixed-point inversion of the **discrete Heun sampler** at the chosen student step count. The script checks its forward cycle RMSE on every training batch and stops when the configured limit is exceeded. During joint training, the original checkpoint supplies endpoint and velocity targets for student trajectories. The training set, episode split and normalizer are reconstructed from the base checkpoint's LeRobot v3 configuration and verified with the saved training-index hash. The script does not retrain the codec or modify the base checkpoint.

```bash
python scripts/posttrain_inverse_gaussian_latent.py \
  --base-checkpoint /path/to/latent_best.pt \
  --output /path/to/latent_inverse_gaussian_4step.pt \
  --steps 4 \
  --source-updates 1000 \
  --joint-updates 1000 \
  --device cuda:0
```

Use the same LeRobot v3 data paths as the base training configuration. If inversion stops on the cycle-error limit, increase `--inverse-iterations` or `--steps`; inspect the reported error before changing `--maximum-cycle-rmse`. A 4-step Heun sampler calls the velocity model 8 times versus 32 calls for the branch's default 16-step sampler. This count does not include the conditional source network, condition encoders or codec. Wall-clock latency, prediction quality and task success still need evaluation on the target hardware and held-out episodes; the training command alone does not establish them.

For inference, load the new checkpoint with `load_latent_checkpoint` and call `sample` as before. `source_noise`, if supplied, is the base standard-normal epsilon in both source modes, enabling paired comparisons. The output includes `nfe`.

To reduce the network depth as well as the integration count, add `--flow-layers 2` to the command. The condition LSTMs, latent dimension and codec stay the same; only the Flow decoder is shortened. Its first two blocks initialize from the teacher. This initialization is **not equivalent** to the full-depth teacher and requires positive `--joint-updates`; source-only fitting cannot validate a shortened network. Use the original trained teacher as the base, then evaluate held-out q/tau errors, contact behavior, stochastic coverage and task success before deployment. Layer counts and the transfer policy are saved in the exported checkpoint.
