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
