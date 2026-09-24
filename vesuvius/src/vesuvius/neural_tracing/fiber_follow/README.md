# fiber_follow

Supervised autoregressive tracing from a high-presence seed. A spatial network
proposes several continuations, scores them against image features and trace
history, and estimates how much of each continuation is safe to commit. It
traces both directions for inference and writes VC3D fiber JSON.

## Current experiment: longer visual history and continuation scoring (v6)

`spatial_candidates_v6` scores continuations directly against the observed
history and its predicted clean version. Each history token contains image
features at both positions, both coordinates, the correction vector, and a
validity mask. A GRU reads valid points oldest to newest; its final state is
combined with a masked mean of history tokens so older observations have a
direct route into the scorer. Masked history never advances recurrent state.

Each candidate is processed in both directions: a forward GRU initialized from
history, and a backward GRU over the proposed future. Both heads combine the
forward and backward states, a pooled future summary, and explicit history.
Ranking pools these joint features across the continuation. Prefix confidence
uses the same context at each point, so distant evidence can influence the
first decision. Candidate and batch states are independent; adding training
teacher candidates cannot change another candidate's scores. At rollout the
scorer uses proposed paths and visible image features, with no future ground
truth. Ground-truth and perturbed teacher candidates are training-only inputs.

The CT launcher uses **32 trace voxels of visual history and a 64-voxel
prediction horizon**. At native CT spacing 0.5 this is a **208×128×128** crop,
with 64 crop samples behind: physical support is -32 to +71.5 along the
heading and ±31.75 laterally. The extra 7.5 voxels ahead provide image context
past the last prediction. All 64 future planes stay one trace voxel apart;
the cleaner/scorer uses 32 past points plus current. Geometric history remains
128 points. Tracing still commits at most four future points per decision.

The crop was widened after the first v6 run showed substantial target clipping.
The heatmap now has 125 lateral samples at spacing 0.5 (±31 voxels), and the
heading-noise standard deviations are 2°, 5°, and 10°. A fixed geometry audit
reduced fresh crossing OOB from 12.9% to 2.1%; widening alone reduced sampled
on-track replay OOB from 23.8% to 7.2%. Curvature remains a source of clipping.

The wider crop has four times the voxels of the initial v6 crop. The preset
therefore uses training batch **2**, DAgger batch **1**, and diagnostic batch
**1**, with all 32 diagnostic seeds retained. These conservative GPU batch
sizes are provisional: the wide configuration passed a full-size CPU
forward/backward check, but GPU memory has not been measured while the narrow
run occupies the GPU. There is no gradient accumulation; 50k steps means 100k
examples. `--steps 200000` would match the initial v6 preset's 400k-example
budget. Compare examples and elapsed time, not just steps.

The decoder estimates its initial tangent with a weighted line fit over up to
six corrected points, including the current point. It accounts for the
corrected anchor's forward coordinate when extrapolating to the first plane.
`--clean-tangent-points 2` provides a two-point tangent ablation. Rollout still
commits only future points; corrections do not rewrite the stored trace.

The CT launcher now selects `--inputs ct+presence`. Input channels are **native
CT / 255, predicted fiber presence / 255, own-history tube**. Presence is read
on the fiber grid and trilinearly sampled at the same world positions as CT;
CT is not downsampled. No direction zarrs are opened in this mode. Presence is
an input feature, not a target, supervision weight, or stopping rule.

Start a fresh run from this directory:

```bash
bash scripts/launch_ct_tube.sh ct0_presence_h32_f64_w128_v6
# Same scorer, two-point tangent ablation:
bash scripts/launch_ct_tube.sh ct0_presence_h32_f64_w128_v6_tangent2 --clean-tangent-points 2
# CT-only input ablation:
bash scripts/launch_ct_tube.sh ct0_h32_f64_w128_v6 --inputs ct
```

These are separate experiments, not commands to run concurrently. V6 requires
a fresh checkpoint; no earlier-architecture weight adapters are provided.

Training logs compare observed and cleaned current/history position errors,
tangent errors in degrees, fractions improved, and correction size. Every new
metric includes its valid-state count. Improvement uses only points with both
observed history and annotated targets; absent history is not treated as a
successful reconstruction.

For held-out rollout auditing, from `vesuvius/`:

```bash
FF=src/vesuvius/neural_tracing/fiber_follow
.venv/bin/python "$FF/scripts/eval_ckpt.py" \
  "$FF/output/ct0_presence_h32_f64_w128_v6/ckpt_010000.pt" \
  --tag presence_h32_f64_w128_v6_c03 --history-audit --batch 1 \
  --params '{"confidence":0.3,"max_len":400}'
```

The evaluation JSON includes `history_audit.groups`: on-track, off-track,
false first-point stops/continues, recoverable drift, and states without a
correct first-point candidate. Decisions use DAgger's original-fiber,
progress-bounded matching. Unknown annotation ends and the end of the short
departure suffix censor the audit, without stopping the actual trace.
Already-departed states have correction-size statistics, not supervised clean
position/tangent errors. The audit measures agreement with the original fiber;
it does not identify which neighboring fiber a wrong correction selects.

Repeat with identical seeds at confidence 0.2, 0.3, 0.5, and 0.7, using distinct
tags. Compare coverage at matched `wrong_len_mean`, not only at one threshold.
The checkpoint-24k sweep in `EXPERIMENTS.md` motivates this evaluation, but does not calibrate
the newly trained V6 head. Full evaluation coverage uses the entire available
annotation; the short training diagnostic additionally caps that denominator
at 400 voxels, so their coverage numbers differ.

Tests for these changes (from this directory, with pytest available):

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ../../../../.venv/bin/python -m pytest -q \
  tests/test_history_confidence.py
```

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

## Architecture and supervision

`spatial_candidates_v3` uses a 3D encoder-decoder with spatial skip connections,
metric coordinate channels, and early conditioning on 128 voxels of trace
history. The default crop is uniformly sampled **64×64×64**, with one trace-grid
voxel spacing on every axis. It covers 16 voxels behind and 47 ahead, and ±31.5
laterally. The current point stays offset toward the back of the oriented cube.
There are 16 future planes, two forward voxels apart; each has a 61×61 heatmap
covering ±30 lateral voxels with one-voxel spacing. This leaves image context
around every proposed point. The full-prefix head and 0.7 threshold are retained.

1. Plane heatmaps learn the dense GT curve crossings directly. Peak decoding
   connects modes into four coherent candidate paths with a curvature penalty
   and path-diversity suppression. This is local proposal decoding; rollout is
   greedy and has no multi-step beam yet.
2. The scorer samples decoded spatial features along each candidate and its
   lateral neighborhood. It predicts candidate rank and confidence for every
   prefix. A GRU carries information from the beginning of each candidate into
   every later confidence prediction; its state resets for each candidate and
   decision. Training scores current proposals, GT/jittered proposals, and the
   original proposals saved in replay.
3. The ranking target is `softmax(quality * --rank-temperature)` (default 20);
`rank_target_entropy` in the log shows how far it is from uniform, and the
`gate*` metrics report false stops and false continues of the confidence gate
on the rank-selected proposal at 0.3/0.5/0.7.
   Confidence labels compare interpolated candidates with dense GT crossings
   every 0.5 forward voxel by default. A prefix is positive when every known
   crossing agrees within `--tolerance` (default 1.5 voxels). A known mistake
   makes that prefix and subsequent prefixes negative; unknown continuation is
   masked. Already departed states supply negative confidence labels and no
   position targets. Prefix checking begins at the first predicted point, so a
   perturbed state can learn to recover onto its fiber.

Inference selects the highest-ranked candidate whose first prefix clears the
confidence threshold (default **0.7**), then commits up to four safe points (2–8 forward voxels).
Confidence is made nonincreasing along each candidate. If no candidate clears
the threshold, it stops **before** committing. There is no stop trimming or
presence-based model-stop heuristic. Bounds, loops, and maximum length remain
geometric limits. Heading updates use committed points only.

The 0.7 default is shared by training diagnostics, DAgger collection, and
inference, and can be overridden with `--confidence`. On the CT run's
checkpoint 24,000 a threshold of 0.3 gave 2.6x the coverage at the same
length precision (see `EXPERIMENTS.md`, stop policy sweep). `TraceParams`
also offers `stop_patience` (consecutive would-stop calls before stopping,
committing one point meanwhile) and `commit_floor`; the defaults keep the
immediate stop. It was selected from
short held-out rollouts of the preceding model; recalibrate it after training
the new head.

The confidence threshold is an operating parameter, not a demonstrated
calibration guarantee. Compare coverage and wrong length on held-out rollouts.
Logs separate best-proposal error/recall from selected-proposal error to help
distinguish proposal failures from scoring failures. Error diagnostics clip
lateral errors at eight voxels and exclude GT/jitter/replay candidates.

`target_crop_oob`, `target_crop_edge`, and `target_heatmap_oob` report fractions
of known dense crossings outside the crop, within three voxels of its lateral
edge (including outside), and outside prediction support. Lower is better.
These measure sample geometry, not model accuracy; no GT is discarded from
confidence supervision because of these counters. Position heatmap loss only
applies where the target is representable.

Batch images mark the current point and prediction limits in cyan, supplied
history in red, GT in green, and the full proposed continuation in orange.
Magenta squares flag GT outside the crop. Plot bounds remain the actual crop;
orange shows the full proposal before confidence gating. Rollout images show
the model-generated path starting from a single seed.

## Native level-0 CT tube experiment

Use `scripts/launch_ct_tube.sh RUN_NAME` for a fresh CT + presence Gaussian-tube run.
The preset uses `/mnt/raid_nvme/volpkgs/s1_2um_ds2.volpkg/volumes/s1_ds2.zarr/0`.
One CT voxel in this dataset is four base voxels (`--ct-grid-scale 4`). All trace
geometry and tolerances remain in the original eight-base-voxel trace grid.
The oriented crop is 208×128×128 with `--crop-spacing 0.5`: one native CT voxel
per sample. It reads level-0 CT directly. Forward planes are one trace voxel
apart and lateral heatmap samples are half a trace voxel apart. The 64-plane
horizon is 64 trace voxels (128 CT voxels); visual history spans 32 trace voxels.

The channels are CT intensity / 255, predicted presence / 255, and the tracer's history tube.
Metric coordinate channels and geometric history conditioning remain. Presence
is used for initial seed selection/snapping and a local weighted-PCA heading;
CT and CT + presence modes never open the predicted nx/ny arrays. Inference also accepts an
explicit `--heading x,y,z` for each seed. After initialization, crop reads and
rollout images use CT. Presence-based break statistics in evaluation are only
additional diagnostics, not inputs or stopping rules.

`--heatmap-target tube --tube-sigma 0.35` supervises the full crop scalar field
with `exp(-distance_to_annotated_polyline**2 / (2*sigma**2))`, truncated at four
sigma. Sigma 0.35 trace voxels equals 0.7 native CT voxels. Distance is measured
to original line segments, retaining bends and multiple plane crossings. This
is different from the previous per-plane normalized Gaussian crossing targets.
Unknown annotation ends censor outward continuation; physical endpoints retain
background supervision. Already-departed states retain confidence negatives
but have no tube-position loss. The annotated tube is a target, never an input.
The model still receives only its own history at rollout time.

The tube uses sigmoid outputs and squared-error regression, averaging losses
in the three-sigma tube neighborhood and remaining known background separately
to keep empty voxels from dominating. Existing proposal decoding, candidate
ranking, prefix confidence and DAgger remain. The decoder samples the volume on
forward planes, so its forward-monotone path representation is unchanged even
though the supervised volume can represent arbitrary annotated bends.

`images/tube_STEP.png` shows image, history, target tube, predicted tube, and
known-region masks in two orthogonal projections. `batch_STEP.png` now shows
CT behind the proposed path. Tube checkpoints record target mode, sigma, CT
voxel scale and crop/heatmap spacing. Start fresh; old fiber-input checkpoints
and replay do not match this configuration. Run names remain unique.

```bash
bash src/vesuvius/neural_tracing/fiber_follow/scripts/launch_ct_tube.sh ct0_tube_64
# Override ordinary training settings after the name if desired:
# ... ct0_tube_64 --steps 10000 --batch 16

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py \
  tests/neural_tracing/test_fiber_follow_ct_tube.py
```

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

## Commands

From `vesuvius/`, use the existing environment. The codec may need:

```bash
export LD_LIBRARY_PATH="$(.venv/bin/python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
FF=src/vesuvius/neural_tracing/fiber_follow

# Fresh spatial model; collection and replay run automatically during training.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.train \
  --name spatial_dagger_v3 --steps 10000 --batch 32 \
  --fiber-zarrs /mnt/raid_nvme/spiral_dataset_working/fiber_zarrs \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers

# Fixed geometry-bound validation seeds. Rebuild after annotation geometry changes.
.venv/bin/python "$FF/scripts/eval_ckpt.py" field --seeds-only --rebuild-seeds
.venv/bin/python "$FF/scripts/eval_ckpt.py" \
  "$FF/output/spatial_dagger_v3/last.pt" --tag spatial_dagger_v3 \
  --params '{"confidence":0.7,"n_commit":4}'

# Independent collection is also available, e.g. for fixed replay ablations.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.collect \
  --checkpoint "$FF/output/spatial_dagger_v3/last.pt" \
  --fibers /mnt/raid_nvme/spiral_dataset_working/fibers \
  --max-seeds 64 --out /tmp/fiber_decisions.npz

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py
```

Add `--dagger-every 0` for fresh-only training. Add `--onpolicy CACHE...` (oldest
to newest) for existing replay. `--init` accepts only this exact architecture and
configuration and starts a new optimizer. `--allow-history-change` permits only
the history renderer and sigma to differ when initializing weights. The uniform-crop v3 architecture requires a fresh
run; earlier checkpoints cannot initialize it. Continuous DAgger itself never
restarts the optimizer. Run names must be new. There are no old-checkpoint
adapters, history-padding fallbacks, or cache migrations.

Each run contains `config.json`, `log.jsonl`, checkpoints, optional diagnostic
images, and `dagger/` with source snapshots, decision archives, memory-mapped
arrays, collection logs, and the atomic `replay.json` index. Replay format is v3;
the controlled-span evaluation format remains v2, with geometry-bound identity.

Evaluation conserves `correct + offtrack + unknown == length`. Read scored
`length_precision` together with coverage, `unknown_length_fraction`, and
`verified_length_fraction`; unknown tails are never credited as verified.
Historical experiments in `EXPERIMENTS.md` are not evidence for the quality of
this newly implemented architecture. It needs a full training/evaluation run.


### Smooth history for the CT tube experiment

`bash scripts/launch_ct_tube.sh RUN_NAME` now uses connected Gaussian history
segments (including the last segment to the current position), sigma 0.35 trace
voxels = 0.7 native CT voxels, and zero independent point jitter. Smooth drift
and wobble remain. Empty histories stay empty and masked gaps are not joined.
The renderer and width are saved in each checkpoint and used in rollout and
replay collection. Old checkpoints retain their original point rendering.

V5 starts fresh with a new run name. Stop an existing run with
`bash scripts/stop.sh OLD_NAME` before replacing it.
`stop.sh` checks the recorded training PID and terminates only that run's process
tree (including forkserver workers), rather than matching all Python workers.

History regressions can be checked with the existing environment:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 .venv/bin/python -m pytest -q \
  tests/neural_tracing/test_fiber_follow.py \
  tests/neural_tracing/test_fiber_follow_ct_tube.py \
  tests/neural_tracing/test_fiber_follow_history.py
```

Batch diagnostics label fresh versus replay samples, mark already-departed
examples as `OFF TRACK: reject continuation`, and show the selected proposal's
next-step confidence. In tube diagnostics these examples retain the reference
geometry for inspection but explicitly mark it as masked: only rejection
confidence is supervised, not tube position or ranking.

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
bash src/vesuvius/neural_tracing/fiber_follow/scripts/launch_beam.sh beam_v1 --steps 20000 --batch 32

# VC's restart metric on held-out fibers, hand beam and model in the loop.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans hand --tag hand
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.evaluate_spans output/beam_v1/last.pt --tag beam_v1

# Open-ended seed evaluation through the existing harness (hand beam or model).
.venv/bin/python "$FF/scripts/eval_ckpt.py" hand --tag hand_beam
.venv/bin/python "$FF/scripts/eval_ckpt.py" output/beam_v1/last.pt --tag beam_v1 --params '{"confidence": 0.5}'

# Re-trace the spans of an annotated fiber with the model, or trace from seeds.
.venv/bin/python -m vesuvius.neural_tracing.fiber_follow.beam.infer output/beam_v1/last.pt --fiber-json fiber.json --out /tmp/retraced

PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run --no-sync --with pytest python -m pytest -q \
  tests/neural_tracing/test_fiber_follow_beam.py ../volume-cartographer/python/tests/test_fiber_trace.py
```

The bindings must be present in the environment: rebuild and reinstall
volume-cartographer (`uv pip install --python .venv/bin/python --no-deps --reinstall
--config-settings build-dir=../volume-cartographer/build-py -e ../volume-cartographer`).
For development against a CMake build tree configured with `VC_BUILD_PYTHON=ON`,
set `VC_PYTHON_BUILD_DIR=<build tree>` when running the tests; the tests are
skipped when `vc.fiber_trace` cannot be imported.

Beam checkpoints carry `architecture = beam_rerank_v1`, the `BeamSpec`
(manifests, trace config overrides, hook cadence and pool size) and the state
configuration; `runloop.py` holds the run-directory, logging, schedule and
checkpoint helpers shared with `train.py`.
