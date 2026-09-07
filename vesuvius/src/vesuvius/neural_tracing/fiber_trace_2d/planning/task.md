# Fiber 3D training acceleration

Reference: s1a_128_1_single_8x8_20260729_075013. S1a NML at input OME
level 1, 8 GPUs, 8 patches per GPU.

Goal: test/loss_total < 1.0 within one hour from fresh training startup,
including testing, visualization and checkpointing. Record cache preparation
separately. Preserve the held-out data, preprocessing, loss and aggregation.
Track direction/presence loss and angular error too. No resumed initialization
may count as from-scratch acceleration.

Options (not yet approved changes):
1. Pipeline utilization: loading, targets, transfers, forward/backward,
   optimizer, DDP, evaluation, visualization and snapshots.
2. Verify AMP; benchmark compilation/convolution settings.
3. Informative sampling: hard positives/negatives and annotation coverage.
4. Learning rate, warmup/schedule and effective batch size together.
5. Loss gradient balance and incomplete-label treatment.
6. U-Net width/depth/block allocation.
7. Difficulty/context curriculum.

Current work: measurement first. Recover the exact saved reference settings;
do not substitute mutable configs. The one-hour target is not a promised result.

Authorized next experiment: two-GPU tooling/iteration/compile comparison on
physical GPUs 0,1, using /home/hendrik/.venv_las and writing all run artifacts
under /ephemeral/hendrik/fiber3d_train2. Keep batch 8 per rank (global 16),
so this is not the eight-GPU convergence baseline.

Next: profile loader workers to explain stalls, reusing existing stage timers.
Keep current task-log updates minimal. Loader and normalization improvement
remain the optimization priorities; current change is measurement only.

Authorized implementation: introduce a central, reusable two-stage per-sample
pipeline and benchmark it before connecting it to training. Stage one loads raw
samples; stage two performs value augmentation. Both queues are bounded, work
is claimed by the next free worker, final delivery remains deterministic, and
the benchmark reports throughput, queue pressure and stalls. Target loader
capacity is greater than 221 samples/s globally on the intended eight-GPU host.

Current measurement refinement: make VC3D coordinate interpolation concurrency
an explicit loader/benchmark setting, run with source read, chunk decode, and
coordinate sampling each limited to one worker, and profile coordinate-map
generation, actual chunk decoding, and coordinate interpolation separately.
Report decoder chunks/s and MiB/s plus coordinate MP/s; normalize every stage
to equivalent 128^3 training samples/s. Preserve the existing transformed
per-coordinate trilinear sampling algorithm and its outputs.

Authorized optimization: profile requested-level coordinate dependency
discovery independently on the real S1a 128^3 workload, then eliminate its
redundant per-corner hash probes with a previous-coordinate chunk-footprint
fast path. Restore equivalent cheap locality in pinned chunk resolution and
inspect the existing render sampler for other exact, portable fast paths.
Target coordinate generation and requested-level sampling capacity is at least
20 samples/s each so their serial composition can sustain 10 samples/s. Do not
use GPUs while they remain allocated to other work.

Authorized next optimization: replace the PyTorch coordinate generator with a
custom C++ implementation that directly builds the final base-coordinate and
validity arrays. Each output coordinate must be evaluated independently from
the affine metadata and low-resolution smooth controls; do not implement the
transform as full-resolution PyTorch layers. Approximate coordinate agreement
with the existing generator is acceptable. Consider precomposing the smooth
transforms on a low-resolution 3D lattice when it is measurably faster overall.
Keep all work CPU-only while GPUs remain allocated elsewhere.
