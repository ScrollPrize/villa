# Two-GPU Fiber training profile, 2026-09-07

All three normal-training experiments completed with exit code 0.
Compilation remains off by default. This is a throughput/tooling experiment,
not validation of the one-hour convergence goal.

## Setup and reproduction

Physical GPUs 0,1: H100 80GB. Other GPUs were occupied; host CPU, RAM and
storage resources were shared. One run per mode, sequentially: eager, default,
reduce-overhead. Existing VC3D disk cache reused; page-cache warmth was not
controlled. No separate prefetch was run.

Interpreter: /home/hendrik/.venv_las/bin/python, Python 3.14.6,
torch 2.11.0+cu128, NumPy 2.4.6, TensorBoard 2.21.0, TensorStore 0.1.84,
volume-cartographer 3.0.3. Checkout base b218ace879d3239a63047e1e2039f327929d283e
plus this task's profiling changes. Cleanup protection and compile-wrapper
unwrapping were added after eager launch; neither changes eager training math.

Saved historical S1a config: 128³ patches, 8 patches/rank, BF16, LR0.01.
Two ranks make global batch16, versus the reference run's global batch64.
Model initialization is explicitly seeded0 for all paired experiments.
Normal test at steps0/100, visualization and checkpoint rules retained.
Only max_steps=120, logging every step, output location, explicit initialization
seed and experimental profiling/compile controls differ from saved settings.

From /home/hendrik/villa:

```bash
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode off
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode default
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode reduce-overhead
```

Run sequentially. The launcher pins CUDA_VISIBLE_DEVICES=0,1 and native thread
limits before torchrun; outputs, compiler caches, generated configs and logs
go under the new root. Input data/cache paths continue to reference the
existing dataset. Default --steps120 and --trace-steps5 are explicit CLI options.

Compilation wraps DDP, following the
[PyTorch DDP guidance](https://docs.pytorch.org/docs/main/notes/ddp.html).
Raw models are retained for evaluation and checkpoint serialization.

## Results

120 steps/run. The steady table uses rank0 steps31–120 excluding step100:
89 untraced iterations, keeping loader stalls. Steps21–25 are traced;
steps<=30 also exclude trace-export skew. Rank1 iteration totals agree.
These distributions are per iteration, not repeated whole-run statistics.

| Mode | Total run s | Iteration mean ms | Median ms | p95 ms | Min/max ms | Median compute+sync ms | Peak allocated GiB/rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Eager | 264.8 | 1595.3 | 483.3 | 8554.2 | 476.6 / 10072.1 | 449.2 | 13.83 |
| Compile default | 295.0 | 1560.1 | 531.2 | 5872.0 | 520.7 / 11522.6 | 494.9 | 14.36 |
| Compile reduce-overhead | 284.6 | 1596.8 | 536.0 | 8835.7 | 525.0 / 13104.8 | 502.8 | 18.12 |

Steady effective throughput including stalls: about10.03, 10.26 and10.02
global patches/s respectively. The small mean-rate difference is not evidence
of a compile speedup given shared-host/cache variability and one run per mode.
Typical compiled iterations were about10–11% slower.

Rank0 first forward/backward/optimizer/sync region: eager0.85s,
default44.69s, reduce-overhead29.68s. These include first-use computation and
waiting, not isolated compiler timers. Reduce-overhead reused the compiler
cache from the preceding default run; startup costs are not cold-cache peers.

## Bottleneck evidence

- Eager rank0 loader wait: mean1114ms, median0.16ms, p958076ms.
  Rank1's mean wait was87ms, but its forward/DDP mean rose to1238ms.
  Matching iteration totals show the other rank waiting for the delayed input,
  not independent proof that its GPU executes the network more slowly.
- Eager trace rank0 contains9.99s in DataLoader iteration over five traced
  steps. Ordinary steps are fast between these stalls. Worker internals are
  not yet profiled: this does not identify I/O versus augmentation versus
  worker scheduling as the source.
- Raw rank0 kernel events (cat=kernel, without double-counting CPU parent
  operator rows): batch-normalization1197.89ms, NCCL31.10ms,
  other kernels872.77ms over five steps. Batch norm is57.0% of summed kernel
  duration; NCCL1.5%. These are kernel sums, not wall-time percentages.
- Transfers are roughly24ms median. Current target wall timers mostly capture
  submission; use GPU events in the trace for device attribution.
- Current complete test routine: initial12.4–13.1s, step1006.1–6.3s.
  Historical roughly53s paired timestamp gaps are not equivalent measurements.
- Both compiled modes warned that Dynamo reached recompile_limit=8 for a
  reused convolution frame. These are partially compiled models with PyTorch's
  own graph-break/eager fallback, not successful full-graph compilation.

## Numerical and format checks

All modes had identical initial test loss2.138115. First training loss:
eager2.4904413223 versus compiled2.4906229973 (absolute difference0.000181675).
At step100 test losses were2.276050, 2.148344 and2.311375; angular errors
32.97°,37.24°,33.35°. Full direction/presence values are in the companion JSON.
These short trajectories are not a convergence ranking or numerical-equivalence
claim. BF16 compilation can change arithmetic and accumulated updates.

Historical and current eager model checkpoints have the same714 keys and
matching tensor shapes. Eager/compiled step100 snapshots keep those714 raw
keys without module. or _orig_mod. prefixes.

## Artifacts and verification

Each experiment contains run.log, config.json, launch.json and its training
subdirectory with TensorBoard events, snapshots, timings_rank0/1.jsonl,
trace_rank0/1.json and operators_rank0/1.txt. Compiler-run summaries are also
written to summary.json. Companion training_profile_2gpu_20260907.json records
the exact directories and complete timing distributions.

```bash
AGENTS_AGENT_MODE=1 CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONPATH=/home/hendrik/villa/vesuvius /home/hendrik/.venv_las/bin/python -m pytest -q vesuvius/tests/neural_tracing/test_training_profile.py vesuvius/tests/neural_tracing/test_analyze_training.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'training_profile or analyze_training or distributed or snapshot or mixed_precision'
```

21 passed,193 deselected; one informational PyTorch profiler warning.
Both ranks exported CPU/CUDA traces in all three real runs.
Default-disabled instrumentation and CPU tracing are tested; macOS/other GPU
hardware have not been exercised.

## Next measurements and explicit limits

Keep eager as default. Profile loader worker stages using the existing loader
profiling hooks, then test worker count/prefetch capacity under the same sampling.
Investigate the batch-normalization kernel/layout cost before further compile
tuning; changing normalization semantics requires a separate quality experiment.

The current trace labels forward/loss, backward, optimizer and scalar sync.
It does not separately scope visualization/checkpoint stages or worker internals.
Those remain follow-up work. Test/visualization/checkpoint costs are retained in
whole-run timing. No full unprofiled repeat, eight-GPU run or one-hour convergence
experiment was performed in this two-GPU tooling check.

## Replayed-batch compute ceiling and matched real training, 2026-09-08

The current eager trainer was measured again on physical H100 GPUs 0 and 1.
Both runs used the saved S1a model/config, batch 8 per rank (global 16), BF16,
AdamW, no compilation, and 120 optimizer steps. Steps 1--30 warmed the model;
the reported throughput is a synchronized contiguous window over steps 31--120
using the slowest rank's elapsed time. Test datasets, visualization, snapshots,
TensorBoard, and profiler tracing were disabled for the benchmark window.

Replay mode loads and prepares one real batch per rank onto its GPU, shuts down
the DataLoader workers, and reuses that batch. It retains the production model,
loss, backward, optimizer, loss all-reduce, DDP, and advancing BatchNorm/model
state. It therefore measures a compute ceiling, not learning quality or input
diversity.

| Path | Window | Slowest-rank wall | Global samples/s | Samples/s/GPU | Rank-0 iteration mean / median / p95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPU-resident batch replay | 90 steps / 1,440 samples | 39.560 s | 36.400 | 18.200 | 439.34 / 438.79 / 440.29 ms |
| Real warm-cache batches | 90 steps / 1,440 samples | 42.831 s | 33.620 | 16.810 | 475.51 / 475.87 / 480.44 ms |

The real loader reached 92.36% of the replay ceiling in this run. Rank-0 loader
wait was mean 3.16 ms, median 3.17 ms, p95 3.46 ms; no old multi-second loader
stall occurred. This is a warm-cache result on a shared host and must be repeated
before treating the small remaining gap as stable. The new standalone shared
sample-worker pipeline is not connected to the trainer yet, so neither result
measures that pipeline inside training.

For historical scale only, the original eight-GPU run completed 19,200 steps
from the first event through 55,389.446 s at global batch 64: 22.185 global
samples/s, 2.773 samples/s/GPU, and 0.3466 steps/s. That event-time rate includes
periodic evaluation/visualization/checkpoint/logging gaps and excludes process
startup. It cannot be compared directly as a global two-versus-eight-GPU speedup.

Commands from the repository root:

```bash
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode off --trace-steps 0 --replay-batch
AGENTS_AGENT_MODE=1 /home/hendrik/.venv_las/bin/python -m vesuvius.neural_tracing.fiber_trace_3d.profile_experiment --reference vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/training_baseline_s1a_20260729.json --output /ephemeral/hendrik/fiber3d_train2 --compile-mode off --trace-steps 0 --throughput-only
```

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_replay_20260908_111539_550175`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_20260908_112510_129671`

Each training run writes `throughput_rank0.json` and `throughput_rank1.json`
with the synchronized window, world size, batch accounting, and rate.

## Normalization compute ablation, 2026-09-08

Three sequential replay trials per mode used the same two H100s, global batch
16, BF16 eager training, 30 warmup steps, and 90 measured steps. DDP buffer
broadcast was disabled for every mode. These measurements compare speed only;
they do not establish equivalent convergence or quality.

| Normalization | Modules in model | Global samples/s, three trials | Mean global samples/s |
| --- | ---: | --- | ---: |
| Local BatchNorm | 57 BatchNorm3d | 31.744, 31.750, 31.752 | 31.749 |
| SyncBatchNorm | 57 SyncBatchNorm | 36.535, 36.610, 36.659 | 36.601 |
| GroupNorm | 57 GroupNorm | 55.278, 55.334, 55.292 | 55.301 |

GroupNorm is retained for the current speed experiments because it was 51.1%
faster than SyncBatchNorm in this matched replay. The ordinary production model
default remains BatchNorm for checkpoint compatibility; the profile launcher
defaults to GroupNorm so follow-on architecture tests retain the fastest tested
configuration unless explicitly overridden.

## Native quarter-resolution output, 2026-09-08

The Fiber3D U-Net can now stop its decoder at the quarter-resolution skip via
`model_3d.output_stride: 4`. Targets are materialized directly on that aligned
lattice, rather than constructing full-resolution targets and reducing them.
Stride 1 remains the compatibility default. The shared whole-volume and
Trace2CP inference paths compose the checkpoint's native stride with only the
remaining requested output downscale, preventing a second factor-four
reduction.

Matched GroupNorm replay trials:

| Native output stride | Output shape | Parameters | Peak allocated/rank | Global samples/s, three trials | Mean |
| ---: | ---: | ---: | ---: | --- | ---: |
| 1 | 128x128x128 | 28,650,951 | 27.83 GiB | 55.278, 55.334, 55.292 | 55.301 |
| 4 | 32x32x32 | 28,523,527 | 14.91 GiB | 95.708, 95.766, 95.668 | 95.714 |

Native stride 4 improved the replayed compute ceiling by 73.1%, reduced peak
allocated GPU memory by 46.4%, and removed 0.44% of parameters. A fresh paired
run after adding source-only hashes produced 55.289 versus 95.699 global
samples/s. Both runs recorded the same source hash
`9d4d0bb3c7d8186f49a80d9eae89ae179028b1f427994de0181c067b065f3139`,
while their target-inclusive hashes differed as expected. Rank-0 iteration
mean/median/p95 were 289.13/289.10/289.68 ms at stride 1 and
166.98/166.98/167.36 ms at stride 4.

The matched real-sample stride-4 run reached 59.154 global samples/s versus
50.231 for the earlier stride-1 GroupNorm run, a 17.8% end-to-end improvement.
Its rank-0 iteration mean/median/p95 were 270.94/197.66/551.74 ms; loader wait
was highly skewed (58.54 ms mean, 0.25 ms median, 356.77 ms p95), so input
delivery now hides much of the compute gain. This is still a speed-only result:
no convergence or prediction-quality equivalence has been established.

The fixed reusable caller reads one argument per line from
`vesuvius/scripts/fiber3d_profile_benchmark.args` without shell evaluation:

```bash
vesuvius/scripts/run_fiber3d_profile_benchmark
```

Fresh paired artifacts:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_replay_group_os1_20260908_150959_126046`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_replay_group_os4_20260908_151616_525550`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_20260908_143002_480159`
