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

## Quarter-resolution input-path attribution, 2026-09-08

Fresh-process GroupNorm stride-4 runs on the same two H100s separated four
paths. All used global batch 16, BF16 eager training, 30 warmup steps and a
90-step slowest-rank throughput window.

| Path | Global samples/s | Role |
| --- | ---: | --- |
| prepared GPU batch replay | 95.708 | model/DDP ceiling |
| augmented CPU batch replay | 88.224 | repeats H2D and target construction |
| old whole-batch DataLoader | 60.115 | original real producer |
| shared per-sample trainer, trial 1 | 84.028 | integrated replacement |
| shared per-sample trainer, trial 2 | 83.595 | integrated replacement |
| shared trainer with loader timings | 84.853 | instrumentation control |

The old trainer assigns a complete eight-sample batch to each of eight
DataLoader workers per rank. A worker took about 2.08s per batch. Stalls recur
every eight optimizer steps: one rank waits for the next producer wave while
the other blocks at the following DDP reduction. Stride 1 hid this limitation
because its model step was slower; the matched stride-1 real run reached 49.771
samples/s with approximately 3.2ms median loader wait.

The CPU replay retained normal transfer and target work and reached 88.224
samples/s, showing that those stages explain only a small part of the gap. The
integrated per-sample pipeline divides one global 128-worker, 512-slot budget
across DDP ranks and preserves the exact canonical sample stream. Two
uninstrumented runs averaged 83.812 samples/s, 39.4% faster than the old real
path and 95.0% of the CPU-replay ceiling. Median producer wait is below 1ms;
remaining work is primarily H2D/target preparation and small host scheduling
jitter.

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_loader_group_os4_20260908_155518_542157`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_loader_group_os1_20260908_155834_757852`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_20260908_160021_141679`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_replay_group_os4_20260908_160148_611880`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_cpu_replay_group_os4_20260908_160615_494193`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_20260908_162049_091414`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_loader_group_os4_sample_pipeline_20260908_162210_991101`

## Asynchronous batch preparation, 2026-09-08

The shared trainer can keep two future batches prepared using
`training.sample_pipeline_cuda_prefetch_depth: 2`. Ordered collation and pinned
host staging run on one background coordinator; nonblocking H2D and native
quarter-resolution target construction run on a dedicated CUDA stream. Taking
batch N admits N+2. Current plus two future batches bound live staging to three
batches per rank, and evaluation pauses and drains preparation.

Two fresh uninstrumented trials used the same two H100s, GroupNorm, native
output stride 4, global batch 16, BF16 eager training, 30 warmup steps, and 90
measured steps:

| Path | Global samples/s | Iteration mean / median / p95 (rank 0) |
| --- | ---: | ---: |
| synchronous shared pipeline, mean of 2 | 83.812 | 191.14 / 187.40 / 211.49 ms (representative trial) |
| asynchronous preparation, trial 1 | 91.792 | 173.99 / 173.41 / 178.11 ms |
| asynchronous preparation, trial 2 | 91.646 | 174.26 / 173.23 / 181.60 ms |
| prepared GPU replay | 95.708 | 166.98 / 166.98 / 167.36 ms |

The asynchronous mean is 91.719 samples/s: 9.4% above the synchronous shared
pipeline and 95.8% of the GPU replay ceiling. Exposed preparation wait was
0.17/0.035/0.61ms mean/median/p95 in trial 1 and 0.32/0.32/0.61ms in trial 2.
Peak allocated memory was 14.99GiB per rank, peak reserved memory 20.31GiB, and
the calculated maximum live pinned-host storage was 240.0MiB per rank. Steps
98--100 produced exactly the same reported losses and metrics as the
synchronous deterministic run.

The remaining approximately 4% versus replay is not CPU sample starvation:
replay omits transfer and target work. Concurrent preparation slightly raises
the forward/backward/step region from roughly 166--167ms to roughly 170ms
because its copy and target kernels still consume GPU resources.

Command:

```bash
vesuvius/scripts/run_fiber3d_profile_benchmark
```

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_20260908_165937_591417`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_20260908_170107_746537`

## Raw loading and CUDA value-augmentation suffix, 2026-09-08

This section records an intermediate prototype. The later independent
sequential-blur implementation below removes both `final-mask`/`blur-mask`
modes and all final validity masking.

The standalone warm-cache S1a pipeline was measured without value augmentation,
while retaining deterministic sample selection, coordinate augmentation, VC3D
dependency resolution/decoding, and final coordinate sampling. With 128
single-threaded workers, 128 warmup samples, and 4,096 measured samples it
reached **324.11 samples/s**. Coordinate generation sustained 24.294 MP/s
(11.584 samples/s per service core) and coordinate sampling 21.728 MP/s
(10.361 samples/s per service core). The corresponding full CPU pipeline reached
222.09 samples/s. Its mean value-augmentation service was 177.59ms/sample:
47.37ms native normalization/intensity, 21.49ms noise, 63.69ms isotropic blur,
43.01ms directional blur, and 1.96ms final masking.

The CUDA implementation preserves the existing operation order using one
contiguous CPU-prefix/CUDA-suffix cut after a single H2D transfer. Two suffixes
were measured on two H100s with GroupNorm, native output stride 4, global batch
16, BF16 model training, 30 warmup steps, and 90 measured steps:

| CUDA suffix | Global samples/s | Rank-0 iteration mean / median / p95 | Rank-0 CUDA suffix mean |
| --- | ---: | ---: | ---: |
| none (previous asynchronous baseline) | 91.719 | 174.12 / ~173.3 / <181.6 ms | 0 ms |
| blur + final mask | 86.588 | 184.50 / 183.85 / 188.43 ms | 45.98 ms/batch |
| final mask only | 91.767 | 174.01 / 172.93 / 178.74 ms | 2.97 ms/batch |

`blur-mask` is a measured negative result: although preparation remained ahead
of training, its CUDA kernels competed with the U-Net and raised the model/DDP
step to about 181.5ms. It remains an explicit experimental mode and is not the
recommended/default path. `final-mask` is the safe first port and was within
noise of the established baseline; exposed wait remained below 1ms and peak
allocated memory was about 15.01GiB/rank. CPU/CUDA comparison uses tolerance
equivalence; repeated execution on the same CUDA/software/hardware backend is
exactly reproducible in the focused test.

Reusable commands (arguments live in the adjacent `.args` files):

```bash
vesuvius/scripts/run_fiber3d_pipeline_benchmark
vesuvius/scripts/run_fiber3d_profile_benchmark
```

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_173232_814065`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_20260908_175210_882168`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_20260908_175613_297576`

## Independent sequential Gaussian blur, 2026-09-08

The production experiment now samples isotropic, directional-along, and
directional-across sigmas independently and uniformly from zero through the
configured maxima. Isotropic blur is three separable 1D passes. It is followed
by one direct rectangular directional Gaussian. Directional kernels are packed
with a deterministic greedy 1.25x padding limit and all groups run serially on
the existing CUDA preparation stream. Dense validity/loss masks are no longer
collated or transferred: GT lines define positives and all other output voxels
are presence negatives.

The actual S1a distribution (10,000 samples) was 50.34% directional; rectangular
boxes had mean/median/p95 volumes of 302.6/245/637. Greedy packing inflated
dense work by 1.055x. On one H100, 32 local batches of eight took 4.134ms/batch
for the full sequential blur (1,935 samples/s); one common max box was slower at
1,318 samples/s.

The matched two-H100 training run used GroupNorm, native output stride 4,
BF16 eager training, local batch 8, 30 warmup steps, and 90 measured steps:

| Path | Global samples/s | Rank-0 iteration mean / median / p95 | Blur CUDA event mean / median / p95 |
| --- | ---: | ---: | ---: |
| model-only replay ceiling | 95.708 | 166.98 / 166.98 / 167.36 ms | n/a |
| prior async pipeline, no GPU blur | 91.719 | 174.12 / ~173.3 / <181.6 ms | 0 ms |
| independent sequential GPU blur | **88.487** | **180.82 / 180.58 / 188.49 ms** | **30.73 / 31.69 / 40.96 ms** |

The CUDA event is longer than the isolated 4.134ms because preparation-stream
blur overlaps and competes with model execution. The implementation is faster
than the earlier dense combined-kernel experiment (83.229 samples/s), but it
does not yet reach the no-blur pipeline or model-only ceiling.

The CPU fallback, measured on the same warm S1a input with 112 single-threaded
workers, 128 warmup samples, and 4,096 measured samples, produced 115.76
samples/s (111.17 including startup/drain). A 128-worker standalone attempt
failed during VC3D worker initialization with `Resource temporarily unavailable`,
so the successful worker count is recorded explicitly rather than extrapolated.

Commands:

```bash
vesuvius/scripts/run_fiber3d_blur_benchmark
vesuvius/scripts/run_fiber3d_pipeline_benchmark
vesuvius/scripts/run_fiber3d_profile_benchmark
```

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/blur_benchmark_20260908_202909.json`
- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_205516_907901/result.json`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_gpu_blur_20260908_205620_262669`

## Full post-sampling CUDA augmentation, 2026-09-08

The production `all` suffix leaves only coordinate generation and native VC3D
volume sampling in the CPU pool. Workers transport the sampled patch as raw
`uint8`; each rank's asynchronous CUDA preparation stream converts it to FP32
and applies normalization, contrast, brightness, gamma, stream-index-seeded
noise, isotropic blur, and directional blur before target materialization.

On the standalone warm-cache S1a producer, 128 single-threaded workers with 120
load-affinity workers, 128 warmup samples, and 4,096 measured samples sustained
**339.80 samples/s production** and **330.04 samples/s end-to-end** while using
**122.0 CPU cores** on average. Production throughput was therefore **2.783
samples/s per occupied CPU core**. The previous exact CPU-prefix boundary was
262.75 samples/s on 104.8 CPU cores. A 124-affinity attempt failed in the native
sampler with `Resource temporarily unavailable`, so it is not a supported point.

The matched two-H100 training run used GroupNorm, output stride 4, BF16 eager
training, global batch 16, 30 warmup steps, and 90 measured steps. It reached
**88.020 samples/s globally**, or **44.010 samples/s/GPU**. Mean iteration time
was 181.44ms, mean H2D was 0.855ms/batch, and the overlapped full CUDA value
augmentation event was 35.22ms/batch. Peak allocated/reserved device memory was
16.53/22.13GiB per rank. A final repeat reached **88.492 samples/s globally**,
or **44.246 samples/s/GPU**, with rank-0 iteration mean/median/p95 of
180.45/180.00/189.57ms, 1.12ms mean H2D, and 35.40ms mean CUDA value
augmentation.

Coordinate generation intentionally remains CPU-side: VC3D currently consumes
the dense coordinate field on CPU, so CUDA generation alone would require a
roughly 24MiB D2H transfer for every 128-cubed sample. Removing that boundary
requires a fused native coordinate-generation/sampling implementation or a GPU
sampler.

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/sample_pipeline_20260908_221924_248510/result.json`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_sample_pipeline_async_batch_gpu_all_20260908_221959_719439`

## Per-GPU batch-size sweep, 2026-09-09

The real-sample end-to-end path was swept on two H100 GPUs with GroupNorm,
output stride 4, BF16 eager training, full post-sampling CUDA augmentation, 30
warmup steps, and 90 measured steps. One sample is one 128-cubed CP patch.
Throughput uses slowest-rank elapsed time; iteration statistics use the
per-step maximum across both ranks.

| Batch/GPU | Global batch | Global samples/s | Samples/s/GPU | Critical-path iteration mean/median/p95 | Peak reserved/GPU |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 16 | 88.492 | 44.246 | ~180.5/~180.0/~189.6ms | 20.61GiB |
| 12 | 24 | 96.419 | 48.209 | 248.63/248.39/258.45ms | 30.46GiB |
| 16 | 32 | 100.948 | 50.474 | 316.66/316.02/325.75ms | 40.42GiB |
| 24 | 48 | 105.330; 105.666 | 52.665; 52.833 | 455.30/454.78/464.33ms; 453.91/453.49/464.47ms | 60.45GiB |
| 28 | 56 | **107.972; 108.088** | **53.986; 54.044** | 518.36/518.76/530.56ms; 517.79/517.42/530.51ms | 70.48GiB |

Batch 28 is the fastest benchmark-safe point and improved aggregate throughput
by about 22.1% over batch 8. It leaves only roughly 11--12% device-memory
headroom, so it is not yet production-safe for evaluation/visualization, and
the larger global optimizer batch still needs convergence validation. The next
point was not run because projected reserved memory falls below the declared
10% headroom threshold.

Reusable command:

```bash
vesuvius/scripts/run_fiber3d_profile_benchmark
```

The adjacent confirmation artifacts are:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b24_sample_pipeline_async_batch_gpu_all_20260909_061011_852473`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b28_sample_pipeline_async_batch_gpu_all_20260909_060627_834494`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b28_sample_pipeline_async_batch_gpu_all_20260909_060819_313464`

## On-demand scalar metric synchronization, 2026-09-09

Training previously packed, globally reduced, and converted every loss metric
to Python floats on every step. The loop now performs that synchronization only
on the rank-consistent union of console and TensorBoard due steps; default
first-100-step and periodic logging remain unchanged. Non-due profiling rows
omit `loss_total` rather than forcing a local CUDA synchronization.

A benchmark-only forced-reduction control suppressed output while retaining the
every-step collective, isolating metric synchronization from terminal I/O. At
batch 28/GPU it reached **107.672 samples/s globally** (**53.836/GPU**). Two
on-demand repeats reached **108.949** and **108.808 samples/s globally**
(**54.474** and **54.404/GPU**), an average improvement of **1.12%**. Aggregate
throughput uses the synchronized slowest-rank window; per-step wall timings are
not directly comparable because non-due steps no longer end at a forced CUDA
synchronization point.

Artifacts:

- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b28_sample_pipeline_async_batch_gpu_all_scalar_sync_control_20260909_062357_589452`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b28_sample_pipeline_async_batch_gpu_all_20260909_062615_301653`
- `/ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b28_sample_pipeline_async_batch_gpu_all_20260909_062808_605523`
