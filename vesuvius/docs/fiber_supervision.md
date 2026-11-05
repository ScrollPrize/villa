# Fiber3D annotated supervision

These are the existing optional Fiber3D training features, extracted onto the
shared sample queue and cooperative trainer. No experiment datasets, run
recipes or changed scientific defaults are bundled.

## Data and sampling

Each dataset supplies a base volume, its scale, fiber files and coordinate
identity. `dense_supervision: true` supervises the selected fiber's interpolated
line, not other fibers in the crop. Eligible JSON spans are spline/trace spans
bounded by control points; Lasagna-interpolated spans and extrapolated tails
are excluded. `sampling_strategy: scroll_balanced_arclength` first selects a
scroll uniformly, then distance along that scroll's eligible fiber collection.
Train and deterministic test loaders use the same geometry rules and queue.

Reviewed pieces inherit H/V and parent identity from full original fibers.
Use [the collector/baker](../scripts/collect_reviewed_fibers.md) to preserve
normal provenance and bake corrected normals onto dense line points. Normal
axes and widths transform with geometric augmentation. A missing label is not
a negative example. Invalid normal samples are skipped rather than fabricated.

## Optional output heads and targets

`annotations.normal_weight`, `width_weight` and `hv_weight` enable the respective
heads; all default to zero. Normal uses the same sign-ambiguous six-channel
representation and magnitude-weighted loss as fiber direction. Width is full
fiber width in network-input voxels, transformed with geometric scaling; H/V is
binary. On Gaussian targets, direction, normal and width supervision cover the
tube with Gaussian falloff, not only its ridge. Missing annotation fields do
not acquire synthetic width/normal supervision.

`presence_target_mode: anisotropic_gaussian` selects the existing anisotropic profile.
The existing ridge/distribution/background terms retain separate normalization.
Optional sidebands have their own class means and annotation-aware detached
width/direction agreement gates. This does not gate ordinary ridge/background
supervision. `mixed_supervision` additionally supports coarse fibers: maxima in
reference-oriented ellipses, projected direction supervision, H/V where known,
and no invented width targets. Coarse width gating requires actual width labels.

All logical-batch denominators are computed from targets before forward work;
microbatch accumulation uses one backward per forward. Detached eligibility
weights do not introduce output-dependent cross-sample denominators. Individual
leaf metrics are unweighted; combined metrics contain their weighted inputs.

## Refinement, embeddings and checkpoints

`refinement.enabled` activates the existing model-feedback line optimizer.
Annotation geometry stays the baseline for anisotropic localization penalties.
Training updates persist in run-local refinement state; snapshot fiber archives
make that state inspectable and resumable. Explicit reset restores annotation
geometry/orientation and parent median widths without resetting model/optimizer.
Test crops stay fixed on their original deterministic selections; their local
refinement is independent, without training's border taper or outside anchors.
Raw/EMA evaluation shares that refined test geometry.

Optional width refinement maintains one scalar per parent; normal-plane motion,
shape, direction and width-staying terms use the existing configuration. Nothing
is refined when disabled. `update_enabled: false` retains restored refined
geometry without further updates.

`embedding` retains pairwise/local-contrastive and paired-patch supervision,
including optional short-/long-range channel splits. Negative selection uses
detached presence importance and excludes the normal-oriented reference tube.
Paired patches select the same fiber, share value augmentations, and have
independent geometry transforms. Embeddings remain disabled by default.

`ema` retains independent configured tracks, evaluation and snapshot selection.
Decoder migration is explicit: ordinary restore stays strict; supported stride
coarsening or pixelshuffle-to-trilinear transitions keep compatible named model,
optimizer and EMA state and report reinitialized tensors.

## Training trace evaluation

The optional `training.reviewed_trace_test` hook retains the existing locked
reviewed-failure benchmark. It requires `interval`, `template`, `split_root`
and `native_binary`; the selected reference cohort and inference template are
external inputs, not bundled datasets or local runner scripts. The original
angle gate, sample-axis logging, restart/retained-length accounting and snapshot
selection are retained. Raw snapshots retain both first/latest tied
best-by-restarts choices; EMA tracks retain their configured loss selection.

The supervisor calls `training_trace_benchmark` with resident inference workers
under its current manager grant. The native CPU replay process uses the shared
case queue/live bridge and writes canonical replay artifacts. It does not
register an overlapping GPU task or reload a Python training process.

Build `vc_fiber_trace_metric` and `vc_cache_root` in a configured native build.
The benchmark's native `replay-set` entrypoint preserves prediction-aligned
restart acquisition, committed-point presence filtering and separate
forward/backward settings. Those differ from the intentionally smaller public
`vc.fiber_evaluation` API (reference-point restarts and candidate-point presence
filtering); that API's defaults are unchanged. Both share core replay matching,
thresholds, consistency checks and diagnostics rather than duplicate the tracer.

## Run and validate

From an environment with this checkout's packages and native bindings available:

```sh
python -m vesuvius.neural_tracing.fiber_trace_3d.train /data/training/config.json
```

The normal entrypoint uses the GPU manager, automatic resource requests and the
existing startup calibration. `--unmanaged` remains an explicit debug escape,
not a production launch recommendation. CPU loaders/datasets remain resident
across GPU allocation changes. Checkpoints and metrics use sample-axis steps.

Representative CPU tests (from `vesuvius/`):

```sh
python -m pytest -q tests/neural_tracing/test_fiber_annotations.py tests/neural_tracing/test_normal_tube.py tests/neural_tracing/test_refinement.py tests/neural_tracing/test_local_competition.py tests/neural_tracing/test_loss_reporting.py -k 'not cuda'
```

Native annotation baking uses the shared VC3D display-field implementation.
CUDA validation must be performed under a manager grant; CPU tests alone do
not establish CUDA throughput, convergence or cross-device numerical parity.
