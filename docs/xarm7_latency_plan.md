# xArm7 execution and latent world-model latency

## Hardware boundary

The inspected xarm_ws checkout is main `d48d451`; the `feat/mit-to-position-admittance` implementation is already included there. Public xArm SDK joint torque commands are unsupported. The adapter declares `joint_torque_command=False` and `mit_impedance=False`; its optional impedance fallback transmits only joint position. SDK `set_report_tau_or_i` selects telemetry and does not command torque. See the [official SDK](https://github.com/xArm-Developer/xArm-Python-SDK/blob/master/xarm/wrapper/xarm_api.py) and [servo guide](https://docs.supportarticle.ufactory.cc/support_articles/developer/ufactory-servo-mode-guide.html).

The existing xarm π0/WM execution path consumes predicted q and sends bounded joint positions. Predicted tau can remain a supervision target and a signal for future load/risk evaluation. Direct motor-torque tracking cannot be assumed for this hardware. The GELLO MIT-to-position helper integrates a virtual dynamics model into q commands; it receives neither measured external force nor world-model tau and is not an external-force feedback controller. An external-force-to-position admittance controller would need a separate calibrated external torque estimate, gains, joint/velocity limits and hardware validation. Total predicted motor torque must not be substituted for external torque.

The xarm config runs position execution at 100 Hz and normally consumes 20 steps per world-model update, corresponding to approximately 5 Hz model refresh. Achieving 100 Hz complete re-prediction requires the whole request to finish within 10 ms, including sensor snapshot, preprocessing/transfers, model inference and delivery. The current xarm deployment loaders support the older GRU/deterministic models; the new latent model also needs an adapter preserving its relative-time-grid and normalization contracts before hardware deployment.

## Runtime optimizations with existing weights

Latent `sample()` in eval mode now projects each block's immutable history/action K/V once per request, before expanding samples. It also batches the fixed integration-time embeddings. These caches rebuild on every request and do not carry LSTM hidden state across windows. Training retains the original path; explicit cache use in training mode is rejected. Set `cache_condition_kv=False, cache_time_embeddings=False` for the original inference path. No checkpoint architecture or parameter names change.

`load_latent_checkpoint` prepares the frozen NEXT motion conversion constants once. Before entering a physical-input control loop, call `model.eval().prepare_runtime_normalizers()` to additionally validate and prepare all physical q/dq/delta_q/tau/action constants. The offline physical adapter also prepares these on its first call. This removes repeated validation and small host-to-device statistics copies from subsequent requests. Moving devices/dtypes, loading state or entering training clears those snapshots; prepare them again after setup. If normalization metadata is intentionally modified, prepare the snapshots again.

The pure `integrate_latent` core can be passed through `torch.compile` while keeping history validation and LSTMs outside the compiled graph:

```python
integrate = torch.compile(model.integrate_latent, mode="reduce-overhead", fullgraph=True)
result = model.sample(batch, integration_fn=integrate)
```

Warm up the intended shape/steps/precision before operation. GPU Inductor/CUDA graph speed and numerical agreement require measurement on the target RTX 5090. A CPU full-graph capture test only checks the interface.

## Structure compression and evaluation

The post-training script separates `--mode source-only` (conditional Gaussian, original Flow frozen) from `--mode distill-only` (standard Gaussian, Flow distillation). First test these methods independently at the original depth. A later distill-only experiment can add `--flow-layers 2 --steps 4`, keeping the condition LSTMs and codec while distilling a smaller Flow decoder from the original teacher. Reducing steps or depth changes predictions and requires held-out data/task evaluation. The Gaussian source alone does not establish unchanged success rate. See [post-training instructions](inverse_gaussian_few_step.md).

`scripts/benchmark_latent_cwm.py` compares original/cached sampling using identical conditions and noise, reports full-model p50/p95/p99 and missed 10 ms deadlines, and separates condition/velocity/codec costs. It can load a real checkpoint and normalized recorded condition, or initialize a synthetic model for timing only. Its model latency excludes acquisition, external preprocessing/transfers and control I/O; measure those separately for a system deadline. Keep FP32 as the reference before evaluating BF16, compilation, reduced steps and trained shallower students.

```bash
python scripts/benchmark_latent_cwm.py \
  --checkpoint /path/to/model.pt --condition /path/to/normalized_condition.pt \
  --device cuda:0 --precision fp32 --steps 16,4,2 \
  --warmup 10 --repeats 100 --output benchmark_fp32.json
```

Add `--compile --compile-mode reduce-overhead` for the warmed compiled core, then repeat with `--precision bf16` for a separate precision comparison. `--condition` is optional for timing but is required for meaningful recorded-input differences; it contains batch-1 normalized q/dq/delta_q/tau/action and true relative grid positions. A real checkpoint alone still uses synthetic inputs when no condition is supplied.

For performance retention, compare the trained candidate against the original teacher using the same held-out episodes and action contract. Check physical q/tau errors, contact-transition timing, prediction spread/coverage and real task success, with special attention to short execution windows and contact phases. Tail latency matters: an average below 10 ms does not establish 100 Hz operation.
