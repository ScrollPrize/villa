# fiber_follow

## Current experiment: direct history-anchored paths (v7)

`direct_paths_v7` predicts continuation coordinates directly from CT, fiber
presence, and the actual trace history. The follower has no Gaussian prediction
head, Gaussian targets or loss, per-plane heatmaps, or peak decoder. The smooth
rendering of observed history remains an **input channel**.

The production preset in `scripts/launch_ct_paths.sh` uses:

| Setting | Value |
| --- | --- |
| Inputs | Native level-0 CT, fiber presence, observed history |
| Crop | 208 × 128 × 128 at 0.5 trace-voxel spacing |
| Visual support | 32 voxels behind, 71.5 ahead, ±31.75 laterally |
| Forecast | 6 candidate paths, each 64 points at one forward voxel spacing |
| Clean history | Current point + previous 32 points |
| Geometry history | Up to 128 previous points |
| Encoder widths / hidden size | 24, 64, 128 / 128 |
| Train / collector / diagnostic batch | 2 / 1 / 1 |
| Training | 50,000 steps; 100,000 sampled states |
| Diagnostics | Every 500 steps, 32 seeds |

CT and presence are independently sampled in the same oriented frame. Both are
scaled by 255. Presence is a learned input cue; annotated fibers supply labels.
The crop aligns its forward axis with the current heading so that prediction
planes have a consistent meaning despite changes in global fiber direction.

### Start and watch

From this directory, with the existing project environment:

```bash
bash scripts/launch_ct_paths.sh ct0_presence_paths_v7
tail -F output/logs/ct0_presence_paths_v7.log
```

The launcher starts a background process. Run names must be new. Checkpoints,
`config.json`, `log.jsonl`, images and replay archives live in
`output/ct0_presence_paths_v7/`. Additional training options go after the name.
This architecture requires fresh training; no old-checkpoint adapters or old
Gaussian launcher are retained.

## Architecture and supervision

1. A 3D encoder-decoder produces spatial features from the three input channels
   and local coordinates. The cleaner predicts corrections to the current point
   and supplied history, capped at four voxels per point. Its loss uses only
   points that are both annotated and actually supplied.
2. An ordered history encoder combines observed and cleaned history. Every
   candidate decoder starts at the cleaned current point, with an initial
   tangent fitted to six recent corrected points and a learned mode embedding.
3. A recurrent coordinate decoder advances across all 64 forward planes. At
   each plane it samples a 3×3 neighborhood of image features around its expected
   location (radius two voxels), updates its state, and predicts a lateral step.
   Its state is conditioned on the history throughout the continuation. There
   is no ground-truth input or teacher forcing in this decoder.
4. A bidirectional scorer examines each proposed continuation together with
   the history. Ranking selects a path; prefix confidence controls how far to
   commit. Candidate coordinates are detached at the scorer input so that
   classification cannot improve its own labels by moving a path. Coordinate
   supervision trains the decoder; both objectives train shared features.

The lateral displacement between adjacent points is capped at two voxels per
forward voxel. The first point is connected to the corrected anchor by the same
bound. With the preset, its distance from the actual current point cannot exceed
`sqrt((4 + 2)^2 + 1) ≈ 6.08` voxels. This prevents the old 10–20 voxel first-point
jumps, but does not guarantee the correct fiber. Cleaning error and local path
accuracy still need to improve through training.

**Coordinate objective:** Smooth-L1 against dense annotated crossings, choosing
one best candidate over the whole known trajectory, plus 0.5 times the first-four-
point loss across **every** candidate, plus 0.25 times the winning candidate's
step-vector loss. This teaches a common near-term continuation while allowing
later alternatives. Candidate modes start with small learned directional
variations; no repulsion forces them apart on unambiguous fibers.

Teacher paths and perturbed paths bootstrap the scorer, but cannot win the
coordinate assignment. Unknown suffixes are censored. Already departed states
supply continuation negatives, without coordinate supervision. Known annotated
points remain coordinate targets even if they leave the crop.

Total loss adds the coordinate objective, ranking, prefix confidence, and
0.5 times the clean-history loss. `proposal` in the log now means the coordinate
objective. Useful measurements include:

- `coordinate_full`, `coordinate_trunk`, `coordinate_steps`: direct losses.
- `first_plane_error`, `clean_current_error`: placement and anchor accuracy.
- `candidate_first_step_max`, `first_step_length_max`: worst first-point jumps.
- `candidate_endpoint_spread`: whether candidates remain distinct.
- `target_step_limit_fraction`: annotated steps outside the decoder's allowed
  lateral displacement; persistent values indicate a representational limit.
- `target_crop_oob`, `target_crop_edge`: lack of visual support, not endpoints.
- `oracle_error`, `selected_error`, `oracle_recall`: candidate quality and ranking.
- Gate false stops/continues at thresholds 0.3, 0.5 and 0.7.

At batch two, per-batch oracle recall often takes values 0, 0.5 or 1. Compare
averages over many batches and held-out rollouts. The old threshold sweep is
recorded in `EXPERIMENTS.md`; it does not calibrate this new model.

## Direct path decoder trunk loss (`direct_paths_v7`)

The direct decoder's coordinate loss assigns one whole-path winner among the
proposed modes. On the first four planes (the commit window) it uses relaxed
winner-takes-all: the winner has weight 1 and every other mode
`--trunk-loser-weight` (default 0.1). Losing modes are kept alive and near the
fiber without being pulled onto the average of two plausible branches; a
weight of 1 recovers the earlier mean over modes, and 0 is pure WTA. The log
reports `trunk_spread`, the mean lateral spread of the modes over those
planes: near zero everywhere means the candidates hedge together, while
opening on some states is the intended behaviour at forks and parallel
neighbours. Tests: `tests/test_direct_paths.py`.

## Ground truth

**Every line point between the first and last control points of every fiber is
valid ground truth.** Line points and control points have the same authority.
The `reviewed` tag has no meaning for loading, sampling, weighting, losses, or
DAgger. Interpolation provenance does not change supervision either.

The loader preserves every interior line vertex and subdivides longer segments
to at most one trace-grid voxel. It excludes exterior tails and fibers with
fewer than two controls. All dense geometry, including low-presence regions,
trains point placement. Presence selects strong starting seeds; it supplies
neither pseudo-labels nor stop targets.

An outer annotation boundary is unknown continuation, unless explicitly marked
`kollesis_termination`. Missing targets beyond unknown boundaries are censored.
Tagged physical endpoints supply negative continuation labels. Control/span
metadata and source hashes are retained for reproducibility, without confidence
weights. Geometry identities invalidate stale seed and replay caches.

Coordinates are trace-grid voxels: one voxel is eight base voxels on this dataset.
The spatial split uses the arclength-weighted centroid of the controlled curve.
After augmentation, every training state must exclude the held-out z band from
its crop read block, active history, and dense targets, with an additional
48-voxel position guard. The same filter applies to replay and collection.

## GPU execution

CUDA training and checkpoint loading use channels-last storage for 3D
convolutions. The tensor dimensions, weights, losses, batch size, and BF16
training precision are unchanged. Fixed-size training batches enable cuDNN
kernel tuning; diagnostic rollouts disable tuning because active batch sizes
change. Independent inference and collection do not enable tuning. CPU execution
keeps its regular tensor layout.

The selected kernels can introduce floating-point rounding differences and
change choices between nearly tied proposals. Fixed seeds do not imply a
bit-identical optimization trajectory across layouts or kernel choices. Run
configuration records `cuda_channels_last_3d` and `cudnn_benchmark_training`.
See `EXPERIMENTS.md` for timing and numerical checks.

## Continuous DAgger

Training keeps one optimizer running. Every `--dagger-every` steps (default 500),
if the collector is idle, it saves a policy snapshot and starts a background
collection process. It does not wait for collection. Completed caches are
published atomically; persistent loader workers refresh replay every eight
chunks, with normal loader prefetch delay. A busy collector skips that snapshot
opportunity and launches at a later interval.

Collection uses high-presence seeds, records the exact input frame and actual
trace history before each decision, and matches progress against the seed's
original fiber. It marks the preceding 48 voxels as hard near departure or a
would-stop decision, retains up to 24 voxels after departure, and optionally
explores eight additional calls after a confidence stop. Exploratory states are
explicitly marked. Unknown endpoint crossings and held-out space censor
collection. Ordinary decisions are thinned to 16-voxel spacing; each trace
contributes at most 192 states.

With both replay pools available, sampling targets 50% fresh perturbed states,
30% ordinary replay, and 20% hard replay. Missing pools fall back to fresh data.
The latest four completed collections are active by default; newer collections
receive linearly greater sampling weight. Each archive records its checkpoint,
training step, collection settings, crop, volume, and fiber identities. Logs
report actual sample fractions and cumulative replay samples consumed. Older
archives remain on disk for auditing or explicit reuse.

`--dagger-device` can place collection on another GPU or on CPU; its default is
the training device, where the two processes share compute and memory. Start
with a small `--dagger-batch` if memory is constrained. Asynchronous completion
changes the exact sample ordering across runs. Use recorded fixed caches with
`--dagger-every 0` for controlled replay experiments. Training shutdown stops an
unfinished collector and reports it; incomplete collections are never published.

## Evaluation and checks

From the `vesuvius/` project root:

```bash
FF=src/vesuvius/neural_tracing/fiber_follow
.venv/bin/python "$FF/scripts/eval_ckpt.py" field --seeds-only --rebuild-seeds
.venv/bin/python "$FF/scripts/eval_ckpt.py" \
  "$FF/output/ct0_presence_paths_v7/last.pt" --tag presence_paths_v7 \
  --history-audit --batch 1 --params '{"confidence":0.7,"n_commit":4}'
```

Evaluation conserves `correct + offtrack + unknown == length`. Read scored
length precision together with coverage, unknown length fraction and verified
length fraction. Unknown tails are not credited as verified continuation.

Focused tests in this directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_history_confidence.py tests/test_direct_paths.py
```

Use an environment with PyTorch and pytest and this checkout on `PYTHONPATH`.
The v7 validation includes coordinate gradients, bounded connectivity, synthetic
curve fitting, censored targets, teacher independence and checkpoint loading.
A full-size real-data BF16 GPU check completed two optimizer steps with finite
gradients (20.94 GiB peak allocated, 22.10 GiB reserved), and a separate collector
produced a replay archive. These checks establish execution, not trained quality.
See `EXPERIMENTS.md` for details.

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
