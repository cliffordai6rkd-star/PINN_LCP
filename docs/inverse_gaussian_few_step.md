# Latent CARS-WM: independent source and distillation experiments

The two methods are independent experiments. `scripts/posttrain_inverse_gaussian_latent.py` defaults to **`source-only`**, which updates only the conditional Gaussian source and freezes every original Flow parameter. **`distill-only`** retains the standard `N(0,I)` source and trains only the Flow sampler. Combining both requires explicit `--mode joint`. A command that supplies positive source and Flow updates without choosing joint is rejected.

All modes start from the same completed, standard-Gaussian latent CARS-WM Heun checkpoint and write a separate checkpoint. Condition encoders, codec, latent statistics and normalizers stay frozen. The LeRobot v3 episode split and saved training-index hash are verified. Dropout stays disabled during post-training while gradients remain enabled for the selected parameters. DataLoader shuffle and distillation noise use separate seeded generators, independent of source-network initialization.

## A. Conditional Gaussian only

```bash
python scripts/posttrain_inverse_gaussian_latent.py \
  --mode source-only \
  --base-checkpoint /path/to/latent_best.pt \
  --output /path/to/latent_source_only_4step.pt \
  --steps 4 --source-updates 1000 --device cuda:0
```

The source is a condition-dependent diagonal Gaussian over the entire future latent trajectory. It is supervised by fixed-point inversion of the **frozen deployment-step Heun map**, with NLL and a KL penalty. Here the inverse uses 4 steps; a 32-step teacher is not involved. Its zero-initialized output starts at exactly `N(0,I)`, preserving the baseline sampler before source fitting at the same step count.

The forward cycle RMSE of the inverse is checked on every training batch. If it exceeds the limit, inspect the reported error and increase `--inverse-iterations` or the deployment step count before relaxing the limit. The method changes the source distribution, not the Flow weights. A source fitted for 4 steps is intended for the 4-step map; do not treat its 32-step output as an unchanged teacher. To compare source fitting at 32 steps independently, fit another source for that step count.

## B. Flow distillation only

```bash
python scripts/posttrain_inverse_gaussian_latent.py \
  --mode distill-only \
  --base-checkpoint /path/to/latent_best.pt \
  --output /path/to/latent_distill_only_4step.pt \
  --steps 4 --teacher-steps 32 \
  --flow-updates 1000 --device cuda:0
```

This mode creates no conditional source parameters and does not run inversion, source NLL or source KL. Student and teacher receive identical conditions and the **same standard-Gaussian epsilon**. Endpoint and velocity supervision train the shorter sampler without replacing stochastic futures by their conditional mean. Update budgets are optimizer steps, not epochs. The checkpoint records the source mode, teacher/source pairing, solver, NFE, seeds, trainable modules and samples seen.

Keep the original Flow depth for this first experiment. `--flow-layers 2` is supported in distill-only/joint but changes a second factor and should be tested separately after evaluating step-count distillation. Smaller students initialize from the teacher's prefix blocks; this is not equivalent to the full teacher and requires training. A smoother 32-step teacher is not automatically the best success-rate teacher: compare the distilled candidate with both original 4-step and 32-step baselines.

## Evaluation protocol

| Group | Source | Flow weights | Purpose |
|---|---|---|---|
| B4 | Standard Gaussian | Original | Fast, unmodified baseline |
| B32 | Standard Gaussian | Original | Higher-step baseline |
| G4 | Learned conditional Gaussian | Original, frozen | Source-only contribution versus B4 |
| D4 | Standard Gaussian | Independently distilled | Distillation contribution versus B4/B32 |

These commands use Heun: **4 ODE steps = 8 NFE; 32 steps = 64 NFE**. Confirm whether historical experiment labels meant ODE steps or network evaluations, and keep the solver fixed. The branch default is 16 Heun steps (32 NFE), which is a different label from 32 Heun steps.

Keep control/replanning cadence, executed horizon, input time alignment, sample selection, noise policy, temperature, precision and joint limits identical for the controlled comparison. Test natural measured inference latency and asynchronous takeover separately. Changes to temporal noise reuse, overlap blending or smoothing are additional experiments; they are not enabled by these training modes. Setting noise or temperature to zero is a diagnostic, not the predictive mean and not a multi-modal deployment evaluation.

The reported problem is a jump **between replans**, so prioritize old/new prediction differences at the same physical future timestamp and the actual command takeover boundary. `scripts/evaluate_trajectory_jitter.py` evaluates those from recorded physical-unit logs; see [log protocol](trajectory_jitter_evaluation.md). Success, completion time, joint/torque errors, contact-transition timing and probabilistic coverage must also be measured. Internal chunk smoothness alone cannot establish continuity between requests.

For model latency use `scripts/benchmark_latent_cwm.py` with a real checkpoint, normalized recorded input and a `32,4` step sweep. It measures model latency, not the full control loop or closed-loop success. Load exported models using `load_latent_checkpoint`; `source_noise` always denotes the base standard-normal epsilon in both source modes.

## Explicit combined experiment

Only after the independent comparisons, `--mode joint --source-updates 1000 --flow-updates 1000` runs source fitting followed by joint source/Flow optimization. Its teacher starts from the student's transformed source, recorded as `shared_student_source`; that differs from the standard-Gaussian pairing in distill-only. Joint results do not isolate either method's contribution.
