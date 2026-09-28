# Axial fiber follower

The training launcher uses `axial_fiber_memory_v2`, the model from
`output/axial_memory_seq_run1`. The model, memory writer, observation construction
and losses were restored from revision `28f69ae6a`. Its saved configuration has
4,803,129 parameters and its checkpoint loads strictly, without renamed or missing
weights. A fresh launch initializes random weights; it does not load that checkpoint.

## Model

The current 120 × 101 × 101 CT/presence crop at spacing .5 passes through the axial
encoder once. Visible observed history and the seed condition the spatial tokens.
One curve decoder predicts 16 forward points. Smaller local heads apply two bounded
lateral corrections and predict prefix confidence. Supplied candidate curves use
that confidence head without writing hypothetical observations to memory.

Historical CT uses a separate small encoder: two stride-two 3-D convolutions over
17 × 17 × 17 CT/presence patches, pooled to eight 128-channel observation tokens.
A gated recurrent writer updates 16 learned memory slots. Eight seed tokens remain
immutable for the trace. Relative motion and seed pose enter the writer, and curve
queries read the slots and seed through attention.

Training reconstructs up to 64 historical observations, plus the current head.
Reconstructed observations are spaced four trace voxels apart. Recorded replay
tracks retain their actual observation positions. The newest 32 historical writes
and the current head backpropagate; earlier writes build the state without gradients.
An auxiliary per-write head predicts departure from, and offset to, the original
fiber. Its targets never enter model inputs. The memory read initially contributes
zero; the auxiliary head trains the writer from the first update.

Each sampled state rebuilds memory with current weights. During tracing, each trace
carries its own slots and immutable seed, and subsequent decisions read only the new
head patch. Replay stores observed geometry and supervision rather than learned
features. Crop, seed and historical patch footprints are checked against holdouts.

Geometry, prefix-confidence, candidate, visible-reference identity and memory-probe
losses match the original run. CT features outside the current crop reach the model
through recurrent patch memory. The explicit identity loss uses visible references.

## Training

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_memory.sh
tail -F output/logs/axial_memory_seq_run2.log
```

The launcher reproduces the original recipe: effective batch 16, microbatch 4,
12 loader workers, learning rate .0003, 500 warmup updates, 100,000 total updates,
BF16 CUDA autocast, compilation, and no activation checkpointing. It uses the same
data paths, split, neighbor bank, sampling, recovery monitoring and online replay
settings. Existing shared sampler optimizations and interval timing logs are retained.
The live neighbor bank can grow, so this is the same recipe, not an identical data
snapshot or a promise of bitwise reproduction.

The default destination is `output/axial_memory_seq_run2`. The launcher refuses an
existing run directory or log. Set `RUN_NAME` for another fresh run:

```bash
RUN_NAME=axial_memory_seq_run3 bash scripts/launch_memory.sh
```

Stop a named run and its workers with `bash scripts/stop.sh NAME` before launching
another. `scripts/launch_regression.sh NAME [options]` exposes the trainer directly;
its memory defaults are also 16 slots and 64 historical observations. Resuming uses
`--resume` and the original matching options. Neither `--resume` nor `--init-tracer`
is used by the fresh-run recipe.

## Validation

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q \
  -o cache_dir=/tmp/memory-pytest
```

Tests cover writer gradients, burn-in boundaries, immutable seeds, padded inputs,
coordinate transforms, streaming/reconstruction parity, active-trace ownership,
holdouts, per-write supervision, microbatch weighting and checkpoint round trips.
The full-size training launch also exercises real-data loading, compiled CUDA
forward/backward, optimizer updates and finite-loss checks.

Evaluation uses `scripts/evaluate_regression.py` and
`scripts/evaluate_regression_recovery.py` with the frozen calibration/final protocol.
Implementation checks do not establish trained tracing quality.
