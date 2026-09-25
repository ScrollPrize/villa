# fiber_follow

The active follower is `single_path_flow_v11`: one jointly denoised future curve,
trained from scratch. Historical v10 results and the pre-migration source remain
in `output/single_path_v11_baseline/`; old follower checkpoints are rejected by
training/inference. The independent beam model keeps its checkpoint names.

## Model and tracing

- CT level 1 (`ct_grid_scale=8`), presence, and rendered observed history.
- One 176 × 96 × 96 crop at spacing 1: 128 voxels behind, 47 ahead,
  ±47.5 laterally. All 128 one-voxel history observations condition the model.
- Shared encoder-decoder widths (24, 64, 128), group norm, geometric-history
  FiLM. History tokens include coordinates, age, validity and local features;
  a separate support flag distinguishes out-of-crop observations.
- Six pre-norm blocks, width 256, eight heads, FFN width 1024. Bidirectional
  self-attention joins history and future tokens. Cross-attention reads spatial
  tokens pooled another factor of two from the deepest features of this crop.
- One 16-point future curve starts at normalized lateral residual zero. Four
  midpoint steps require eight velocity evaluations. A ninth evaluation at
  the final coordinates and `t=1` supplies confidence. Local patches refresh
  during refinement; static image/history features and image keys/values cache
  within a decision.
- The public output is `points [B,16,3]`, `confidence_logits [B,16]`, and
  monotone `confidence [B,16]`. Inference is deterministic for fixed inputs and
  execution settings. No forecast is carried into the next decision.
- Commit at most four points, respecting the six-voxel first connection limit,
  bounds, loops, stop patience, and bounded collection exploration. History
  coordinates and previous commits remain fixed.

Flow matching retains 64 stratified time/noise draws per state. Residual scales
are fitted from 2,048 masked training states. Confidence labels describe the
actually generated curve at tolerance 1.5; generated coordinates and labels are
detached, while the final evaluation trains the shared features and denoiser.
Confidence loss is half masked BCE over prefixes 1–4 and half over all 16.
Its coefficient ramps from zero to one over 2,000 optimizer updates. Unknown
annotation endings are censored; confirmed departures have confidence negatives
and no localization loss. GT history is diagnostic data only.

## Data and first run

The mixture is 50% fresh augmentation, 25% permanent recovery bank and 25%
recent replay. Each replay source reserves 10% of draws for confirmed departures;
other draws balance the drift bands <1, 1–1.5, 1.5–2, 2–3.5 and fibers within each
band. Missing strata fall back to fresh states. Logs contain realized source
fractions and stratum counts. A smooth accumulated displacement over a uniformly
sampled 16–64 history voxels augments existing drift, heading, wobble, missing
and truncated histories.

Replay v5 stores observed states and original-fiber correspondence, independently
of prediction shapes. The one-time importer discards predictions, relabels drift,
checks the larger crop/holdout footprint, preserves source provenance, deduplicates
states, and samples up to 20,000 unique eligible states. The fixed bank never
rotates out with the latest four completed caches. Every training replay draw is
relabeled and checked again before recropping. The original 48-voxel holdout
position guard and complete crop/history/target exclusion remain in force.

From this directory, using an existing environment with the project dependencies:

```bash
export PYTHONPATH=../../..
python scripts/prepare_single_path.py
python scripts/import_replay.py output/flow_v10_small_b8_d64/dagger/decisions_*.npz \
  output/replay_bootstrap/*.npz \
  --out output/single_path_v11_preparation/fixed_recovery.npz
bash scripts/launch_single_path.sh single_path_v11_run1
```

Preparation refuses to overwrite a bank and reuses a matching frozen seed manifest.
These preparation artifacts already exist in this workspace. The launcher benchmarks
before starting training. If microbatch 2 exhausts GPU memory, rerun with
`MICROBATCH=1`; effective batch remains eight and the crop remains full sized.
The benchmark reports GPU peak allocated/reserved memory, training throughput and
complete tracing latency, including data preparation and final confidence.

Training defaults: 50,000 optimizer updates; microbatch 2 with four accumulation
steps; AdamW, peak LR 1e-3, weight decay 1e-4, 1,000-update warmup/cosine decay,
gradient clipping at 1, EMA decay .999, CUDA BF16 and channels-last convolutions.
A detached curve pass establishes masked-loss denominators across the effective
batch; each microbatch then re-encodes for flow/confidence gradients. Censoring
and departures therefore do not change the objective when switching microbatch
size. Inference still encodes once per decision.
Checkpoints save every 1,000 updates and at completion. One background collector
refreshes replay every 1,000 updates when idle, using EMA, 64 training seeds,
a 6,000-voxel cap, threshold .7, and eight exploration calls. Diagnostics use the
original monitor fibers at thresholds .5 and .85. Images show observed history,
GT, the final curve, and successive denoising updates.

## Evaluation

The frozen manifest retains the original 32 monitor fibers and 48 assessment
fibers. The remaining 44 validation fibers form the final split; monitor fibers
are excluded as well as assessment fibers. Current seed counts are 32 monitor,
96 calibration and 176 final. Geometry hashes and source identities are stored.

```bash
python scripts/evaluate_recovery.py output/RUN/last.pt \
  --fixtures output/single_path_v11_preparation/calibration_recovery.npz \
  --out output/RUN/recovery.json
python scripts/evaluate_single_path.py calibrate \
  --manifest output/single_path_v11_preparation/seeds.json --out output/RUN/calibration \
  --checkpoints output/RUN/ckpt_*.pt
python scripts/evaluate_single_path.py final \
  --manifest output/single_path_v11_preparation/seeds.json --out output/RUN/final \
  --selection output/RUN/calibration/selection.json
```

Calibration selects the greatest coverage among checkpoint/threshold pairs
reaching 95% scored precision, then locks the checkpoint hash and threshold.
Final evaluation enforces that choice and uses actual 6,000-voxel rollouts.
Reports include drift-band counts, four-plane correctness, false stops,
departure continuations, coverage, divergence, total wrong length and its
per-trace distribution. Fixed-state evaluation also measures subsequent recovery
toward the original fiber from identical observed histories. Existing rollout
scoring semantics (3-voxel tolerance, sustained departure) are preserved; dense
prefix confidence labels use 1.5 voxels. Unknown length receives no verified
continuation credit.

Pass paired final baseline rows via `--baseline-rows` to compute fiber bootstrap
intervals. Baseline and new rows must have identical seed identities. The target
is ≥20% relative coverage improvement at matched 95% precision with no worse
sustained wrong continuations; no improvement is assumed from confidence shifts.

`evaluate_recovery.py --baseline-archive output/single_path_v11_baseline/source_before_v11.tar.gz`
loads the archived v10 implementation only for baseline evaluation. Use
`collect_baseline_fixtures.py --help` to preserve its actual observed decisions.
The archived deterministic rollouts have a 1,200-voxel cap and do not substitute
for the locked final comparison. `evaluate_single_path.py` also accepts
`--baseline-archive` for baseline calibration and locked final rollouts; calibrate
the baseline separately, using the same frozen seeds. Both final reports are
needed before making a matched-precision improvement claim.

## Verification and current status

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONPATH=../../.. \
  PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests -q \
  -o cache_dir=/tmp/single-path-pytest
```

Tests cover attention across futures and full history, masked and unsupported
observations, GT isolation, endpoint censoring, departed states, confidence gradient
routes, recovery limits, coordinate transforms, deterministic inference, midpoint
integration, replay import/rotation, checkpoint round trips, beam compatibility,
accumulation, collection and synthetic parallel-fiber identity recovery.

The full-crop RTX 5090 benchmark passed: microbatch 2, effective batch 8,
9.20 GiB peak allocated / 10.02 GiB reserved, 8.85 samples/second (cached-batch
training, including accumulation and optimizer updates), and 82.7 ms per complete
tracing decision over a 64-voxel rollout. There are three measured updates after
warmup. `output/single_path_v11_preparation/benchmark_b2.json` holds raw timings.
GPU access requires execution outside this session's filesystem sandbox.

All 34 tests pass with CUDA access. A real-data two-update training/checkpoint
smoke and two-seed replay publication also passed. Full-crop prediction and
denoising images are in the preparation directory. `single_path_v11_run1` is the
fresh 50,000-update run; watch `output/logs/single_path_v11_run1.log`. Training and
experimental acceptance are still in progress. CT level 1 losing fine neighboring
fiber detail remains an experimental risk.

## Beam re-ranker (`beam/`): learning to choose the VC3D tracer's paths

A second training method. The volume-cartographer line-annotation window
traces control point to control point with a C++ beam search over the fiber
prediction volume (81-direction cone, width 8, lookahead 2, hand loss from
presence and direction agreement plus Lasagna-normal smoothness). `beam/`
keeps that tracer as is and trains a model to re-rank its candidate pools.

The C++ tracer is driven through new nanobind bindings, `vc.fiber_trace`
(`volume-cartographer/python/vc/fiber_trace.cpp`). The tracer gained a
per-request *beam hook*: at prune time it hands the best `pool_size` (32)
frontier candidates, each as a full path from the trace start with its
cumulative hand loss, to a callback that may return replacement losses and a
stop flag. The standard width-8 diversity prune then runs inside the pool.
With no hook, or a hook that returns nothing, the search is bit-identical to
the plain tracer (the width-8 selection is a prefix of the pool selection), so
observed pools are exactly what the annotation window would see.

**Units.** Everything in `beam/` is in fiber_follow trace-grid voxels (8 base
voxels). VC trace voxels are `prediction_to_base / 2**scaledown_power` = 2 base
voxels on this dataset, so one VC step (4 trace voxels) is one grid voxel. The
conversion factor is read from the opened field (`NativeBeam.grid_to_trace`),
never hard-coded. VC's default smoothness needs the Lasagna normal dataset;
training and evaluation take `--normal-manifest` (default in the launcher:
`las_008_s1_full/las_008.lasagna.json`).

**States and labels** (`beam/states.py`). Each pool is anchored at the end of
the candidates' common trunk. The trunk is the history (CT crop channel and
conditioning), and every candidate becomes `k_back` trunk points, the anchor
and `k_fwd` new points (defaults 16 + 1 + 8 at one-voxel spacing) in the
anchor's local frame, with its hand loss relative to the pool's best as an
extra feature. Only the new points are labeled against the dense annotated
curve: a point fails when it is more than `--tolerance` from the curve or
regresses along it; failure persists along the prefix; continuation past an
untagged annotation end is censored; anchors more than 3.5 voxels from the
curve supply negatives only, with no tube supervision. Per candidate the
model learns a listwise rank (soft target from prefix quality), an on-fiber
logit for the last new point, and per-point prefix logits; a dense CT tube
head (`tube_loss`) is an auxiliary target unless `--no-tube`.

**Data** (`beam/data.py`). Every loader worker opens the prediction field
and normal sampler itself (their readers keep process-global state, so the
DataLoader uses the forkserver context) and runs the beam over a random span
of a training fiber, optionally from a laterally/angularly perturbed start.
Every vertex between a fiber's first and last control point is user verified,
so span endpoints are arbitrary vertices of that trimmed line (12 to
`--max-span` grid voxels, log-uniform length); the annotated control points
are only special for mining and for the evaluation metric. An observing hook
records every pool (`--hook-every-rounds 4` rounds = 8 grid voxels of new
path per decision); up to `--states-per-trace` pools are kept per trace with
hard pools (hand tracer's choice off-fiber) oversampled by `--hard-prob`.
Because the hand beam succeeds on about 95% of spans, `beam/mine.py` first
runs VC's restart metric over the training fibers (cached under
`output/hard_spans_<digest>.json`, `--hard-spans none` to skip) and the
dataset traces one of the failing spans with `--hard-span-prob`. Held-out
band filtering also covers every candidate path.

**Hook at inference** (`beam/hook.py`). `additive` (default) sets
`loss = hand_loss + w * (-log p_onfiber)`; `replace` uses `-rank` directly.
Open-ended traces stop when no candidate reaches the `--confidence`
probability.

```bash
# Train (CT level 0, forkserver workers, span diagnostics every 500 steps).
bash src/vesuvius/neural_tracing/fiber_follow/scripts/launch_beam.sh beam_v2 --steps 20000 --batch 32

# VC's restart metric on held-out fibers, hand beam and model in the loop.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans hand --tag hand
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans output/beam_v2/last.pt --tag beam_v2

# Open-ended seed evaluation through the existing harness (hand beam or model).
.venv/bin/python "$FF/scripts/eval_ckpt.py" hand --tag hand_beam
.venv/bin/python "$FF/scripts/eval_ckpt.py" output/beam_v2/last.pt --tag beam_v2 --params '{"confidence": 0.5}'

# Re-trace the spans of an annotated fiber with the model, or trace from seeds.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.infer output/beam_v2/last.pt --fiber-json fiber.json --out /tmp/retraced

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest python -m pytest -q \
  tests/neural_tracing/test_fiber_follow_beam.py ../volume-cartographer/python/tests/test_fiber_trace.py
```

The bindings must be present in the environment: rebuild and reinstall
volume-cartographer (`uv pip install --python .venv/bin/python --no-deps --reinstall
--config-settings build-dir=../volume-cartographer/build-py -e ../volume-cartographer`).
For development against a CMake build tree configured with `VC_BUILD_PYTHON=ON`,
set `VC_PYTHON_BUILD_DIR=<build tree>` when running the tests; the tests are
skipped when `vc.fiber_trace` cannot be imported.

Beam checkpoints carry `architecture = beam_rerank_v2`, the `BeamSpec`
(manifests, trace config overrides, hook cadence and pool size) and the state
configuration; `runloop.py` holds the run-directory, logging, schedule and
checkpoint helpers shared with `train.py`.
