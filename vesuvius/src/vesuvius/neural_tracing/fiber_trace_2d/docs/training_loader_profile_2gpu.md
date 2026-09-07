# Loader-worker profile, 2026-09-07

Measurement only: no normalization, augmentation, ordering or worker-count
changes. Reference S1a NML, OME level 1, patch 128³, batch 8 per rank,
GPUs 0,1, BF16, eager training, 8 loader processes/rank and prefetch factor 2.
This global batch 16 run is not the historical global batch 64 convergence test.

Run from the villa checkout:

```bash
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode off --trace-steps 0 --profile-loader
```

Artifacts: `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_loader_20260907_083218_068643`.
Its `launch.json` records versions, command, source hashes and launch duration;
`config.json` records exact input/cache paths. Both `timings_rank*.jsonl` files
are under `s1a_128_sd1_2x8_20260907_083232`. All 120 steps completed, exit 0,
251.4 seconds including startup, testing and shutdown. Python 3.14.6,
torch 2.11.0+cu128; no compiled model in this run.

## Findings

Steady selection: steps 31–120 excluding test step 100, 89 batches per rank.
Times below are seconds per batch of 8 patches, mean / median / p95.
Nested sampling times must not be added to their parent sampling stage.

| Stage | Rank 0 | Rank 1 |
| --- | --- | --- |
| Worker batch service | 13.147 / 12.990 / 17.281 | 13.454 / 13.299 / 17.552 |
| Value augmentation (including input normalization) | 8.366 / 8.275 / 12.486 | 8.618 / 8.646 / 12.418 |
| Volume sampling, including coordinate handling | 3.140 / 3.145 / 3.309 | 3.170 / 3.168 / 3.382 |
| VC3D sampling alone (nested) | 2.617 / 2.613 / 2.748 | 2.617 / 2.605 / 2.781 |
| Geometry | 1.579 / 1.569 / 2.354 | 1.604 / 1.514 / 2.373 |
| Target specification | 0.019 / 0.019 / 0.022 | 0.018 / 0.017 / 0.021 |
| Batch stacking | 0.036 / 0.043 / 0.046 | 0.036 / 0.043 / 0.047 |
| Training loader wait | 0.858 / 0.00017 / 7.752 | 0.657 / 0.00017 / 6.272 |
| Training iteration | 1.632 / 0.472 / 8.568 | 1.632 / 0.472 / 8.568 |

Value augmentation accounts for about 64% of worker service time. Its current
timer includes normalization, intensity transforms, noise and both blur paths;
it does **not** isolate which operation is expensive. Input normalization here
is distinct from the model BatchNorm hotspot measured in the previous report.
Do not attribute this entire time to normalization or blur without finer timing.

Using the existing `_worker_overlap_summary`, maximum active workers is 8 on
each rank; mean active workers is 6.76 / 7.00 across their measured spans.
Worker process CPU-time totals divided by those spans are 8.13 / 8.38.
The loader is parallel, but a batch takes much longer to prepare than a normal
~0.44-second compute step. More prefetch alone does not remove that service cost.

Ordered delivery also contributes: rank 0 step 33 waited 10.19 seconds for
worker 0, while batches 34,35,36,38,40 had already finished worker computation
before that wait began. Step 41 waited 10.27 seconds while batches 43,46,48
had already finished. This demonstrates computed-ahead work, not proof that
its IPC payload was already available. Preserve deterministic order when
designing any scheduling improvement.

Worker-end to coordinator-receive mean lag is 7.18 / 7.61 seconds. This combines
transport, ordering and intentional prefetch residence; it is not an IPC-only
measurement. Statistics cover received batches, not unconsumed prefetched work.
Worker construction and first 30 steps are excluded. This is one shared-host
run with additional CPU timers; timing overhead is not independently isolated.

## Comparison and validation

Previous eager run: steady mean/median/p95 iteration 1.595/0.483/8.554 seconds;
this run: 1.632/0.472/8.568 seconds. No speedup is claimed. Initial test loss
2.138115 and step-100 test loss 2.276050 (angle 32.97°) match the previous eager
run's printed results; first training loss matches exactly at 2.49044132232666.
19 focused tests passed, including exact batch-value/sample-order preservation
and worker-to-coordinator timing propagation.

Next measurement: split value augmentation into normalization, intensity/noise
and blur timings, then optimize the measured operation while preserving sample
semantics. Model BatchNorm optimization remains separate and unimplemented.
No eight-GPU or one-hour convergence result is claimed.

## Throughput budget

One sample below means one 128³ patch. The historical eight-GPU run processed
64 samples per optimizer step. From test step 0 through test step 19200 it
processed 1,228,800 samples in 55,315 seconds: 22.21 samples/s globally, or
2.78 samples/s/GPU, including tests and the normal between-test bookkeeping.
The intervals from completion of one test to the next training scalar (100
steps later) sustain a mean 27.85 samples/s globally, or 3.48/GPU. Those
intervals can include checkpoint/bookkeeping after the preceding test, so they
are an observed training-cycle rate rather than pure model timing.

For the current two-GPU run, eight loader workers on each rank each construct
a complete local batch of eight samples. Dividing 64 worker-samples by mean
worker batch service gives 4.87 samples/s on rank 0 and 4.76 on rank 1, or
about 9.6 samples/s globally. Observed steady training is 16 / 1.632 = 9.80
samples/s globally (4.90/GPU); finite buffering explains the small difference.

When a batch is ready, current median forward/backward/optimizer/scalar-sync is
0.441 seconds: 18.14 samples/s/GPU, or 36.28 globally on two GPUs. Median full
no-wait iteration wall time is 0.472 seconds: 16.95 samples/s/GPU, or 33.90
globally. The historical median ready-batch compute time was 0.502 seconds:
15.92 samples/s/GPU, or 127.40 globally across its eight GPUs. These are
empirical ready-batch ceilings, not a controlled repeated-batch benchmark.

Thus the current loader supplies only about 27% of the model's measured
compute-only appetite per GPU. Linear extrapolation to eight ranks would be
roughly 38 samples/s globally, but that is only a capacity estimate: 64 active
training workers, idle persistent test pools, storage/cache contention and DDP
balance may prevent linear scaling. A controlled repeated-batch DDP benchmark
is still required before treating the ready-batch ceiling as a standalone raw
U-Net benchmark.
