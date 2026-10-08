# S1a training: historical observation and EMA

Use `fiber_trace_3d/configs/train_s1a_nml_ema_16x8.json`. It retains the modern
S1a NML data, GroupNorm, stride-4 decoder, augmentation and LR 0.01. Global batch
is 128, physical maximum 16/GPU, limit 2,000,000 optimizer updates. Test and
checkpoint intervals are 100 updates; retained snapshots are every 10,000.
Runs and TensorBoard events go under `/ephemeral/hendrik/fiber3d_train2/runs`.

Started run: `s1a_nml_128_sd1_16x8_ema_20260912_111221` (implementation
`c4ef954c4`). Automatic calibration finished and the same run continued
adaptively; verified through update 400 / 51,200 samples. The conservative
measured allocation selects physical batch 8/GPU with two microbatches, retaining
global batch 128. Warm managed test passes cost about 5.1 s per 100 updates.

```bash
source /home/hendrik/.venv_las/bin/activate
python -m vesuvius.neural_tracing.fiber_trace_3d.train /home/hendrik/villa/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/train_s1a_nml_ema_16x8.json
```

The normal entrypoint automatically uses the GPU manager. Missing calibration
is measured using the real initial test and real updates under exclusivity,
then continues adaptively. EMA allocations and evaluation backups participate
in this config-specific memory measurement. No manager protocol change.

## Metrics

Each prefix has `loss_total` and `angle_mean_deg` scalars. The step axis is
global training samples, not optimizer updates.

| Prefix | Weights | Test definition |
| --- | --- | --- |
| `test` | raw | unchanged current dense/arclength test |
| `test_ema_0p99`, `test_ema_0p999` | EMA | current test |
| `test_historical` | raw | historical CP test |
| `test_historical_ema_0p99`, `test_historical_ema_0p999` | EMA | historical CP test |

`historical_test_datasets` is an optional separate dataset list. This config
uses `er_20260615T190505820_000003.json`, CP-only supervision, deterministic
control-point sampling, logical batch 8, no augmentation, trilinearly upsampled
predictions and full-resolution targets. Presence BCE uses the historical
central patch region (margin from `presence_negative_edge_margin_voxels`, or
ceil(max augmentation shift)). Batch-row means retain historical weighting.

**Compatibility limitation:** current native sampling no longer transports the
historical missing-source validity mask. This restores historical sampling,
resolution and central presence-loss region, not the old backend's hole
handling or bit-identical old values. Startup and TensorBoard configuration
explicitly record that distinction. No masks are restored to training.

Historical metrics are observational only. Raw `best.pt` continues to use
current `test/loss_total`; each EMA best uses its own current dense loss.

## EMA and snapshots

`training.ema_decays: [0.99, 0.999]` creates two FP32 shadows initialized from
the initial raw model. After each successful optimizer update:
`average = decay * average + (1 - decay) * raw`. Fixed decay, no hidden warmup
or bias correction. Nonfloating buffers are copied. BF16 model compute remains
unchanged. FP16 overflow/skipped updates do not advance EMA counters.

Raw plus both EMA models evaluate each already-loaded test batch. Temporary
weight copies preserve Parameter identity and restore raw state before every
manager boundary, including exception cleanup. The existing shared CPU queue
serves both test sources; historical source ownership is separate.

- `current.pt`, `best.pt`, `step_XXXXXXXX.pt`: ordinary resumable snapshots,
  including all EMA shadows, counters and best metrics.
- `current_ema_0p99.pt`, `best_ema_0p99.pt`, `step_XXXXXXXX_ema_0p99.pt`:
  independent inference snapshots; equivalent files for `0p999`.
- EMA exports carry `snapshot_kind: ema` and decay/update/start metadata.
  They deliberately contain no mismatched raw optimizer. Training rejects
  resuming from these exports; resume from the raw snapshot instead.
- Legacy raw checkpoints without EMA initialize new tracks at the resumed
  update with an explicit message. They do not invent earlier EMA history.
- Managed RAM handoff and replica verification include EMA; pause/resize
  therefore does not reset averages or best metrics.

## Validation and measured cost (2026-09-12)

Approved launcher: `vesuvius/scripts/run_fiber3d_cooperative_tests`, configured
by `fiber3d_cooperative_tests.json`; suites `ema`, `cuda-ema`, `cuda-cohort`.
GPU suites require an exclusive manager grant, enforce 20 GiB/GPU and leave
other manager tasks protected. No package rebuild or precision change.

One H100, production U-Net, synthetic 128-cube batches of 4, BF16, AdamW:
26 alternating updates, first 6 warmup, 10 measured per case.

| Case | Mean ms | Median ms | Min–max ms | Samples/s, one GPU |
| --- | ---: | ---: | ---: | ---: |
| Raw | 98.947 | 98.959 | 98.843–99.011 | 40.426 |
| Two EMA tracks | 99.586 | 99.556 | 99.526–99.792 | 40.166 |

EMA-only CUDA events average 0.617 ms/update; paired full-step overhead 0.65%.
Both shadows total 419.74 MiB/GPU, plus a temporary ~209.87 MiB test backup.
These are model-only timing results, not end-to-end training throughput.
Two-GPU → one-GPU tests, interrupted during training and testing, pass while
preserving model/optimizer/EMA and exactly-once metrics (45.85 s total).

Full real-data smoke: three updates, 8 H100, all six evaluations and snapshots,
exit 0 in 85.42 s including startup/shutdown. Warm test passes cost
2.817/2.982/2.906 s; initial pass 14.300 s. Evaluation overhead is separate from
EMA update overhead and amortizes over the 100-update production interval.
Artifacts: `fiber3d_train2/reproduction/ema_smoke-20260912_110704` and
`cooperative-tests/20260912_110553_431829`.

## Approved experiment controller

`vesuvius/scripts/fiber3d_experiment.json` has `production.config` pointing at
the normal config. The fixed approved launcher supports `production-start`,
`production-status`, and `production-stop`. Start is detached, with config,
input hashes, command, source revision and log in `<run_path>/.launches/`.
It refuses a duplicate live production launch. The trainer itself owns its
manager registration and bounded compute shutdown; no outer exclusive lease
prevents adaptive use. Stop sends TERM to the recorded identity and waits for
the supervisor; it errors rather than claiming success if that root survives.
