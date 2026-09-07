# Fiber training acceleration baseline

Goal: held-out test/loss_total below 1.0 within one hour of fresh training
startup, retaining the reference evaluation definition. Options and next
measurement protocol are in ../planning/task.md and ../planning/task_plan.md.

## Reference and provenance

Run: s1a_128_1_single_8x8_20260729_075013.
Input: S1a NML fibers, PHercParis4 20260411134726 at OME group 1.
Patch 128³; batch 8 per GPU, reference launch described as 8 ranks (global 64).
The saved config itself does not encode world size.
U-Net: base width 16, depth 5, pixelshuffle, batch normalization, 7 outputs.
Saved precision: BF16 autocast enabled, no gradient scaler.
Saved LR: **0.01**, weight decay 0.0. The mutable external sd1 config currently
uses 0.001; it must not be used as the exact historical reference.

Saved config and numerical summary: training_baseline_s1a_20260729.json.
The checkpoint config independently agrees with TensorBoard config.
Reference code revision and historical environment are not established by these
artifacts; current checkout is b218ace879d3239a63047e1e2039f327929d283e.
Current training already contains later distributed-test/preloading changes;
historical timings are not measurements of the current implementation.

## Measured historical quality

One event file, 193 test evaluations from step 0 through 19,200.
Elapsed time is relative to first TensorBoard event, not process launch.

- Best within first hour: 1.886512 at step 1,200, 58.14 minutes.
- Best overall: 1.017569 at step 17,700, 13.97 hours.
- Last: 1.153580 at step 19,200, 15.39 hours.
- No observed loss below 1.0. Do not invent a crossing time.
- Full audit retains direction/presence and angular-error curves as well.

## Historical pipeline measurements

191 logged timing samples, steps >100, normally one sample per 100 steps.
These are sampled rank-0 timings, not all-step or all-rank aggregates.

| Stage (ms) | Mean | Median | p95 | Min | Max |
| --- | ---: | ---: | ---: | ---: | ---: |
| Loader wait | 459.41 | 0.23 | 237.44 | 0.13 | 18714.84 |
| H2D/batch transfer | 44.50 | 44.60 | 57.22 | 25.32 | 62.72 |
| Target submission | 4.55 | 4.44 | 5.39 | 3.34 | 14.42 |
| Forward/backward/optimizer/scalar sync | 2103.00 | 502.36 | 10030.58 | 444.36 | 17573.94 |

Paired train-to-test scalar timestamp gaps: 192 samples, mean 58.11 seconds,
median 53.32, p95 56.62, min 51.36, max 553.59. Summed gaps are 3.10 hours,
about 20.1% of observed wall time. This includes logging/scheduling and any
visualization before the test scalar; it is not an isolated evaluation timer.

The very large compute and loader outliers warrant investigation. Sparse
rank-0 logs cannot attribute them to DDP, I/O, host contention or model work.
The ordinary median loader wait is tiny, so consistently slow loading is not
established. Removing evaluation overhead alone cannot yield a 15x speedup.
BF16 is already enabled; merely enabling AMP cannot provide the target.

## Reproduce the audit

From /home/hendrik/villa (no GPU or downloads required):

```bash
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.analyze_training /ephemeral/hendrik/fiber3d_train/runs/s1a_128_1_single_8x8_20260729_075013
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_analyze_training.py
```

The first command prints JSON with saved config, all test curves, threshold
crossing and timing distributions. Redirect stdout to a JSON file if desired.
It intentionally rejects multiple event files so resumed sessions are not
silently combined. It uses original simple-value scalar events from this
trainer; arbitrary third-party tensor-encoded scalar logs are not supported.

## Current measurement status and next run

Update: the authorized two-GPU tooling/compile comparison is complete; see
[training_profile_2gpu.md](training_profile_2gpu.md). The following records the
initial eight-GPU baseline status; that full-node measurement is still pending.

Observed hardware: 8 H100 80GB GPUs. GPUs 2-7 were already busy at inspection;
only 0-1 were free. No competing eight-GPU baseline was launched.

The existing --benchmark is eval/no_grad, forward-only and rejects DDP.
It cannot measure training backward or synchronization costs.
The normal trainer's current timers aggregate backward, optimizer and scalar
all-reduce; target timing normally measures enqueue time. New opt-in tracing
must use the normal trainer and identify those stages rather than duplicate
training logic. This instrumentation and live eight-GPU run remain pending.

Next: pin reference data file hashes and test coverage, recover saved config
into an isolated 300-step run, and measure a free eight-GPU node with retained
tests/visualization. Record process launch time, exact environment, cache state,
all-rank profiler windows and unprofiled wall throughput. The 300-step run is
a throughput baseline, not a one-hour convergence test. Do not change model,
sampling, loss, batch, or precision during this baseline.
