# Continuous trajectory memory (v4)

`--memory-version 4 --no-correction` selects `TrajectoryMemoryFollower`.
One transformer decoder predicts all future points jointly. A query for each
forward distance samples initial image features on the forward axis, exchanges
information with the other queries through self-attention, and cross-attends to
image/history tokens, persistent identity slots, immutable seed tokens, and the
latest observed patch. The forward-axis query locations do not constrain the
output to that line: a linear head predicts two continuous lateral coordinates
per query, bounded by the crop. The first point also respects the existing
maximum connection distance. Forward planes remain fixed.

There is no lattice, dynamic-programming route search, correction head, or second
decoder pass. Prefix confidence inspects the predicted curve; it is not multiplied
by lattice probability mass. Diagnostics expose a single proposal. The shared
confidence and identity heads still inspect local features; those evaluations
do not update coordinates.

The v3 `IdentityMemory` writer is retained, including its pre-write admission
gate, immutable seed, FP32 writes, and separate latest-observation tokens. Seed
pose is expressed in the actual query crop frame, including its roll. Each
decoder layer reads this identity memory directly, so coordinate loss reaches
the persistent writer and seed encoder without passing through discrete route
selection. This does not establish that a trained model will use memory well;
matched-input identity tests and held-out tracing are still needed.

## Supervision and replay

The existing dense coordinate, prefix-confidence, candidate, visible-reference
identity and memory-probe losses are reused. Coordinate loss directly supervises
the sole proposal. Recoverable drift and on-policy replay use the same annotation
and holdout rules as the existing trainer. Confirmed departures teach rejection
and memory probes, but do not receive trajectory geometry loss. Unobservable or
censored labels remain masked. V4 has no route-localization loss.

Matched decisions retain the v3 policy: identical local crops and histories, with
different causal earlier observations and seeds outside the crop. This provides
examples where local image/history information alone cannot identify the target.
`--trajectory-sequence-weight 0.5` adds geometry/confidence supervision at an
earlier causal decision on eligible observed tracks. It reconstructs that prefix
with current weights; it does not use later observations or train through future
commits. Set the weight to zero to disable this extra decision. The original
`--route-sequence-weight` continues to apply only to v3.

## Launch and checkpoints

From `fiber_follow`, using the existing project environment:

```bash
bash scripts/launch_trajectory_memory.sh --direction-inputs
```

The launcher uses the existing drift/replay recipe, 16 future points, 8 committed
points, 16 memory slots, up to 64 historical observations, and gradients through
the newest 32 plus the current observation. It refuses an existing run name.
It starts fresh weights unless `--init-tracer` is supplied. It launches background
training; do not invoke it just to inspect arguments. To see trainer options:

```bash
bash scripts/launch_regression.sh --help
```

A separate run can initialize from a v2 or v3 checkpoint:

```bash
RUN_NAME=axial_trajectory_memory_v4_run1 bash scripts/launch_trajectory_memory.sh \
  --init-tracer output/axial_spatial_memory_v3_run2/ckpt_007000.pt \
  --direction-inputs --batch 8 --microbatch 4 --workers 8
```

Compatible encoder, decoder, coordinate, confidence and memory weights are
retained; lattice and correction heads are omitted. V3's coordinate head was
unused during v3 training, so its weights are inherited rather than trained by
that run. Direct identity attention changes decoder behavior even with shared
weights. This migration starts a new optimizer, schedule and run; it is not a
behavior-preserving resume. Memory dimensions must match when reusing weights.

V4 checkpoints use `axial_fiber_memory_v4` and load through the existing trainer,
collector and tracer. True `--resume` is supported within a v4 run with matching
options. V2/v3 checkpoint loading and model behavior remain supported; a running
v3 trainer is not restarted or migrated by adding this implementation.

## Validation

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  PYTHONPATH=../../.. python -m pytest \
  tests/test_trajectory_memory.py tests/test_spatial_memory_v3.py \
  tests/test_learned_memory.py tests/test_memory_sequences.py \
  tests/test_identity_decisions.py tests/test_regression.py -q
```

Tests check direct geometry gradients into seed and earlier observations, memory
sensitivity with fixed local inputs, bidirectional influence between future
points (including point 2 and point 8), continuous coordinates and connection
bounds, censoring/departure masks, matched identity sampling, causal replay,
streaming/reconstruction parity, burn-in, tracing state ownership, checkpoint and
optimizer restoration, migration, direction expansion, and full-graph capture.
Implementation checks do not establish tracing quality or CUDA performance.

Validation on 2026-09-29: the six-file suite above passed 108 tests, skipped five
CUDA tests, and deselected one test under the repository's default marker filter.
The subsequently added point-2/point-8 interaction test also passed. Full-graph
capture used `torch.compile(backend='eager', fullgraph=True)`, not CUDA Inductor.
Loading `ckpt_007000.pt` from the v3 run and migrating in memory retained every
shared tensor exactly (4,931,226 v3 parameters; 4,530,711 v4 parameters). No trained
v4 quality comparison or throughput measurement has been performed.

The existing training environment lacked pytest. Validation used an already
cached pytest distribution, without installing packages:

```bash
AGENTS_AGENT_MODE=1 PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES='' \
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=/home/sean/Documents/villa4/vesuvius/src:/home/sean/.cache/uv/archive-v0/c6JuZFIQxH2QbGds/lib/python3.14/site-packages \
/home/sean/Documents/villa4/vesuvius/.venv/bin/python -m pytest \
  tests/test_trajectory_memory.py tests/test_spatial_memory_v3.py \
  tests/test_learned_memory.py tests/test_memory_sequences.py \
  tests/test_identity_decisions.py tests/test_regression.py \
  -q -o cache_dir=/tmp/fiber-trajectory-pytest
```

The same environment was used for the final targeted check with pytest arguments
`tests/test_trajectory_memory.py::test_future_points_influence_each_other_in_both_directions
-q -o cache_dir=/tmp/fiber-trajectory-pytest`.
