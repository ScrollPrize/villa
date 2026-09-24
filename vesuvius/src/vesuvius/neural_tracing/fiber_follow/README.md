# fiber_follow

## Current experiment: normalized future flow with fixed observed history (v10)

`future_flow_v10` generates only the 64 future coordinates. The current point
and actual trace history are fixed conditioning inputs; no history or anchor
correction is predicted, averaged, or applied to the trace. CT, fiber presence,
and the smooth rendering of observed history remain the image inputs.

The production preset in `scripts/launch_ct_flow.sh` uses:

| Setting | Value |
| --- | --- |
| Inputs | Native level-0 CT, fiber presence, observed history |
| Crop | 208 × 128 × 128 at 0.5 trace-voxel spacing |
| Visual support | 32 voxels behind, 71.5 ahead, ±31.75 laterally |
| Forecast | 64 future planes at one forward voxel spacing |
| Fixed tokens | Current point + up to 32 history points + 64 prior-mean observations |
| Flow | 4 transformer blocks, 4 heads, hidden 128; 4 midpoint steps (8 evaluations) |
| Residual scales | Per-plane lateral standard deviations fitted from 2,048 training states; floor 1 voxel |
| Flow patches | 3 × 3 lateral features at 2 trace-voxel pitch, independent of crop spacing |
| Samples / candidates | All 16 samples scored; support radius 1.5 voxels RMS |
| Training draws | 32 stratified (time, noise) draws per state against shared image features |
| Geometry history | Up to 128 previous points |
| Encoder widths / hidden size | 24, 64, 128 / 128; group norm |
| Train / collector / diagnostic batch | 2 / 1 / 1 |
| Training | 50,000 steps; 100,000 sampled states; 1,000-step warmup |
| EMA | Decay ramps to 0.999; collection, diagnostics and inference use EMA |
| Diagnostics | Every 500 steps, 32 seeds |

CT and presence are independently sampled in the same oriented frame and
scaled by 255. Presence is an input cue; annotated fibers supply labels.
The crop's forward axis follows the actual trace heading.

### Start and watch

From this directory, with the existing project environment:

```bash
bash scripts/launch_ct_flow.sh ct0_presence_flow_v10
tail -F output/logs/ct0_presence_flow_v10.log
```

The launcher starts a background process. Run names must be new. Checkpoints,
`config.json`, `log.jsonl`, images and replay archives live in
`output/ct0_presence_flow_v10/`. Additional training options go after the name.
This architecture requires fresh training. Checkpoints store live weights in `model`,
EMA weights in `ema`, and fitted scales in `model_cfg.flow_sigma`.
`--init` restores both weight copies and the fitted scales with an exact configuration
match, then starts a new optimizer. Collection, evaluation and inference load EMA.
`--flow-steps` controls midpoint steps; each step evaluates the flow twice.
`--flow-calibration-states` controls the initial scale-fitting sample count.
`--flow-stencil-radius` specifies lateral patch pitch/radius in trace voxels.

## Architecture and supervision

1. A 3D encoder-decoder produces spatial features from CT, presence, observed
   history rendering and local coordinates, conditioned on geometric history.
2. **Future flow** (`PathFlow`). The transformer reads fixed current/history
   tokens, one always-valid observation token per future plane at the prior mean,
   and noisy future tokens. Each token carries a 3 × 3 lateral feature patch,
   voxel coordinates, token type, time and position embeddings, and history
   context. Static patches are sampled once and reused across training draws
   and midpoint stages. Only noisy future tokens emit lateral velocities;
   forward coordinates remain pinned to their planes. Missing history tokens
   are excluded from attention.
3. **Flow objective.** A weighted observed-history tangent sets the prior mean:
   forward distance times lateral slope, clamped inside the crop with space for
   the observation stencil. Unmeasured, backward and nearly perpendicular
   tangents fall back to straight ahead. Per-plane lateral residual standard
   deviations are fitted once from training-loader states, excluding unknown
   targets and departed states, and floored at one trace voxel. Training uses
   `y_1 = (x_1 - mu) / sigma`, `y_0 ~ N(0, I)`,
   `y_t = (1-t) y_0 + t y_1`, and MSE against `y_1 - y_0` over known lateral
   coordinates. Times are stratified as `(d + uniform()) / draws` per state.
   Integration stays in normalized coordinates; feature queries and returned
   paths use `x = mu + sigma * y`. Partial annotations retain their existing
   behavior: unknown future tokens are excluded from attention keys and loss;
   all observation tokens stay valid. Sampling generates the full horizon.
   Crop exits do not censor flow targets or dense scorer labels. Departed states
   provide no flow supervision but retain confidence negatives. GT history is
   used only for diagnostics.
4. **Candidate support.** Every generated sample is scored, including coincident
   paths. Support is the fraction of generated samples within `--support-radius`
   RMS lateral distance over the first four planes. Generated candidates exclude
   their own vote and use the remaining sample count as denominator. Teachers
   and replay candidates receive measured support against the generated sample
   set; they do not contribute votes. Support is an input feature for both
   ranking and confidence, not a correctness label.
5. **Scoring and tracing.** The bidirectional scorer uses the actual observed
   history and each future. Its first-step geometry is relative to the actual
   current point. Generated candidates receive no scorer gradients; ranking
   and prefix-confidence losses still train the shared spatial features.
   The tracer commits a confident prefix from its existing current point.

`--recent-history-points` controls the dense observed history seen by the flow and scorer. It does not create history targets
for learning. The first predicted point can recover from a displaced current
position. `--max-recovery-distance` bounds the full 3D connection from the actual
current point to the first prediction (default: 6 trace-grid voxels, independent
of crop spacing). Longer connections receive negative prefix labels and are
ineligible for tracing, including exploration and stop-patience overrides.
If no connection is eligible, the trace stops with `recovery_limit`. The limit
is saved in the model configuration. Within this recovery segment, departure
from GT is permitted; dense GT agreement starts at the first prediction.

Total loss is `--flow-weight` × flow + ranking + prefix confidence. Useful
measurements include:

- `flow`, `flow_known_fraction`: future velocity loss and annotated fraction.
- `observed_current_error`, `observed_history_error`,
  `observed_tangent_error_deg`: input drift diagnostics, not learned cleaning.
- `candidate_support`, `selected_support`: sample agreement; low agreement
  can reflect ambiguity, sampling error, or an undertrained generator.
- `first_plane_error`, `first_step_length_max`: recovery placement and jumps.
- `candidate_recovery_reject_fraction`, `recovery_blocked_fraction`,
  `target_recovery_reject_fraction`: rejected connections, states with no
  eligible connection, and annotated targets outside the recovery limit.
- `oracle_error`, `oracle_recall`: all generated candidates, excluding teachers
  and replay extras. Recall checks the commit horizon and recovery limit;
  error is mean lateral error at annotated future crossings, clipped at 8
  voxels as in ranking. `oracle_recall_known_fraction` reports the fraction of
  eligible states with a known oracle recall outcome.
- `selected_error`: the tracer's confidence-gated, recovery-eligible choice,
  including its diagnostic fallback when it stops. `accepted_selected_error`
  covers accepted choices only; `selected_accept_fraction` reports their share.
  Training metrics and EMA diagnostic plots/rollouts use `--confidence`.
- `target_crop_oob`, `target_crop_edge`: missing visual support, not endpoints.
- Gate false stops/continues at thresholds 0.3, 0.5 and 0.7.

Sampling is stochastic. Training uses the global seed; rollouts seed one
sampler per `trace()` call using `TraceParams.seed`. Reproduction requires the
same seeds and batch composition. Compare averaged held-out measurements and
rollout precision/coverage, not isolated batch-two recall values. Validation
thresholds from older architectures do not calibrate this one.

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
  "$FF/output/ct0_presence_flow_v10/last.pt" --tag presence_flow_v9 \
  --history-audit --batch 1 --params '{"confidence":0.7,"n_commit":4}'
```

Evaluation conserves `correct + offtrack + unknown == length`. Read scored
length precision together with coverage, unknown length fraction and verified
length fraction. Unknown tails are not credited as verified continuation.

Focused tests in this directory:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest -q \
  tests/test_history_confidence.py tests/test_future_flow.py
```

Use an environment with PyTorch and pytest and this checkout on `PYTHONPATH`.
CUDA-only recovery and metric regressions (skipped if CUDA is unavailable):

```bash
AGENTS_AGENT_MODE=1 python -m pytest -q -p no:cacheprovider tests/test_recovery_policy.py
```

The focused suite covers fixed observed history, future-plane pinning,
masked-token isolation, distinct candidate selection, gradient separation,
synthetic curve fitting, and training/checkpoint/tracing integration. See
`EXPERIMENTS.md` for checks performed on each architecture; historical GPU
measurements are not measurements of v9.

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
