# Training-feedback fiber refinement

Disabled by default. Add a top-level object to a training JSON:

```json
"refinement": {
  "enabled": true,
  "iterations": 4,
  "spacing_base": 8.0,
  "samples_per_segment": 4,
  "normal_radius_base": 40.0,
  "tangent_radius_base": 80.0,
  "gt_weight": 1.0,
  "shape_weight": 0.01,
  "shape_normal_weight": 0.1,
  "shape_tangent_weight": 10.0,
  "direction_weight": 0.1,
  "orientation_enabled": true,
  "orientation_learning_rate": 0.05,
  "orientation_gt_weight": 0.01,
  "orientation_max_step_degrees": 1.0,
  "orientation_probe_radius_sigma": 2.0,
  "learning_rate": 0.01,
  "max_step_fraction": 0.125,
  "border_fraction": 0.125
}
```

Use dense supervision and `scroll_balanced_arclength` sampling, with a
`lasagna_manifest_path` on every train/test dataset. Start the normal managed
training entrypoint with `--resume` pointing to the desired pretrained snapshot.
No separate refinement model is loaded; raw current-model presence from the
existing forward pass supplies feedback. Predictions are detached, not the model
weights. EMA evaluations do not update geometry.

## Solver and loading

`fiber_trace_3d/refinement.py` performs four bounded semi-implicit iterations by
default. Segment midpoint quadrature supplies presence gradients through the
loader's affine and smooth coordinate mapping. A native banded solve combines
the immutable-original GT ellipse tether and weak bending of the resulting line,
not of its displacement. The penalty is `shape_weight * integral |d²x/ds²|² ds`,
when the optional normal/tangent weights are omitted. With those weights the
curvature penalty is `normal_weight*|nnT*kappa|² + tangent_weight*|(I-nnT)*kappa|²`:
normal0.1/tangent10 gives 10x/1000x the previous isotropic0.01. Unoriented normal
projectors, not signed normals, are interpolated. The full 3x3 block-pentadiagonal
system has scalar half-bandwidth8. It uses nonuniform current arclength coefficients
frozen within each solve. Material
arclength masses match the presence quadrature. There is no length penalty or
coupling across separate pieces, and test ends have natural free boundaries.
Updates are projected into each current tangent's perpendicular plane,
then capped to 1/8 ellipse. For training only, the cap ramps from zero at the patch
boundary to full at 1/8 patch width, and local section endpoints remain fixed. Every iteration
resamples by arclength while retaining original-reference material coordinates.
The cap applies per iteration; total drift is controlled by the original-GT
tether, not a hard cumulative distance cap.

The current follow-up config keeps normal0.1/tangent10 and lowers direction1->0.1.
Model-training loss weights and LR remain unchanged. Restart
from the original pre-refinement checkpoint for this comparison.

The full-resolution continuation uses
`train_5scroll_json_eligible_fullres_refinement_ema_16x8.json`. It resumes the
raw stride-1 run at update4100 (524,800 samples), retaining its model,
optimizer, and EMA tracks while initializing refinement geometry from the
original annotations. It keeps normal0.1/tangent10/direction0.1, LR0.01,
global batch128, and embeddings disabled.

### Anisotropic Gaussian presence targets

The current full-resolution follow-up changes that config to initialize model
weights from the refinement run's best EMA-0.999 snapshot while restoring its
saved fiber geometry and continuing online refinement. This is a weights-only
`training.initialize_from`, not an optimizer/step resume; the new optimizer,
EMA tracks, and update counter start fresh.

It opts into a soft line-distribution target:

```json
"presence_radius_voxels": 2,
"presence_target_mode": "anisotropic_gaussian",
"presence_gaussian_width_ratio": 4,
"presence_gaussian_truncate_sigma": 3
```

`presence_radius_voxels` is the surface-normal thickness sigma in this mode.
The in-sheet axis perpendicular to the fiber uses four times that sigma. For a
voxel's closest point on a finite line segment, the target is
`exp(-0.5*((d_normal/sigma_normal)^2 + (d_width/sigma_width)^2))`; finite end
caps use the width sigma, overlapping segments combine by maximum, and values
outside three sigma are exactly zero. Output-stride targets retain input-voxel
units. Direction supervision covers every nonzero tube voxel. Each voxel uses
the tangent of the segment that supplied its maximum probability; its direction
loss and angle metric are weighted by that probability and normalized by the
sum of Gaussian weights, preserving the direction-loss scale.

Normals come from the same native validity/encoding-aware Lasagna sampler as
refinement. The loader transforms them through each affine and smooth
augmentation as covectors (`J^-T n`), then forms the local in-sheet axis from
the transformed segment tangent. Invalid normals or singular transforms use
the existing deterministic same-scroll sample replacement rather than a fake
axis. Soft BCE is averaged over nonzero tube support per patch; pure-zero
background is averaged separately, preserving the established equal
positive/background patch weighting. BCE remains appropriate because its
pointwise optimum is the configured Gaussian probability. The target-only
macro-batch prepass uses the identical support/background group counts and
Gaussian direction-weight sum.

The half-sigma continuation resumes the preceding run's full raw checkpoint,
including optimizer, EMA, update counter, refined curves, and material state.
It changes thickness sigma from 2 to 1 input voxel and therefore in-sheet sigma
from 8 to 4 voxels. It runs on the managed physical-GPU subset 0-6.

When `orientation_enabled` is true, refinement may also rotate the Gaussian
cross-section about its local fiber tangent. The state is an unoriented axis,
so each material node stores `(cos(2 theta), sin(2 theta))`; interpolation and
resampling therefore remain continuous modulo pi. Theta is relative to the
native Lasagna normal sampled at the immutable baseline material location.
Target construction applies the persisted rotation in base space before the
existing exact covector augmentation transform.

Each refinement iteration estimates an angular derivative with central
finite differences. Four symmetric probes use the same physical radius on the
candidate normal and width axes; the score is mean width-axis presence minus
mean normal-axis presence. Unlike an equal-Mahalanobis ellipse contour, this is
uniquely maximized modulo pi by the correct axes when width exceeds thickness.
Output-voxel sigma offsets are
pulled back through the local full augmentation Jacobian, then evaluated with
the exact nonlinear base-to-patch transform. A segment contributes only when
both angular candidates have complete in-patch probe sets. The explicit update
is capped by `orientation_max_step_degrees` per iteration and uses the same
training border taper/endpoint pinning as position; independent test pieces use
the uniform cap. `orientation_gt_weight * sin(theta)^2` weakly tethers the axis
to the original native normal. Geometry and orientation use one optimistic
compare-and-swap, so a stale proposal changes neither.

Optional direction guidance (default weight0, current trial0.1) penalizes
`1-(unit_segment_tangent dot predicted_base_axis)^2` at the same segment quadrature.
Six encoded channels are interpolated before shared analytic decoding. The full
finite-difference base-to-patch Jacobian, including smooth deformation, pulls the
predicted tangent back into base ZYX. Axes/Jacobians are frozen within each step
and refreshed next iteration: this is a lagged-field solve, not a spatial gradient
through the direction field. The analytic endpoint force uses material quadrature
and the same lumped masses as presence. Unsupported/zero-axis samples contribute
no direction force; singular/nonfinite Jacobians fail explicitly.

Geometry math is batched native NumPy/SciPy on CPU after backward; detached
presence maps and, when enabled, six direction channels leave the GPU. No additional model forward or backward is needed.
Normals use the existing vc3d/C++ encoding-aware sampler, including validity.
Invalid normals skip that patch's refinement, not its ordinary supervision.
Only the connected in-patch section nearest the selected location is updated.

The existing fast queue carries immutable geometry observations. Rank zero
commits proposals in sample-index order after each macro-update and each test
wave, rejecting stale overlapping proposals. Later loads see the new geometry;
already-prefetched batches retain their old targets. Train/test stores are
separate. Raw and EMA evaluations within a wave use the same prepared targets.
Test crop selection and all image transforms are computed from immutable original
GT. Each deterministic crop stores its own local refined curve (`test_patch` in
archive metadata); overlapping crops do not update one another or a whole test
fiber. Other visits of the selected fiber stay present in targets as original GT;
the solver still updates only the nearest connected visit. Refined targets move,
but the image crop does not follow them. Test pieces use a uniform 1/8-ellipse cap:
no border taper and no endpoint pinning. Artificial joins to the original outside
curve are omitted from dense labels. Quadrature outside prediction support supplies
no presence force (no fabricated border extension); the original-GT tether remains.
New crop entries initialize lazily from original GT, including after resume.
Legacy whole-test refinements are unused; their `fiber.json` exports are marked
`immutable_source_only` and show baseline geometry. Training geometry is unchanged.
Raw and EMA TensorBoard tracks expose the same direction/presence/total, angle,
branch and selection metrics (plus embedding diagnostics where enabled).
Test targets intentionally evolve, so these metrics are not fixed-GT validation.
Sampling weights remain based on original lengths; selected material locations
map onto the current curves. Multiworker lookahead timing can affect which
geometry version is observed; bit-identical scheduling is not promised.

## Snapshots and lifecycle

Alongside model snapshots, `fibers_step_########.zip` contains every registered
train/test component's original `baseline.npy`, refined `current.npy`, refined
`orientation.npy`, source
`metadata.json`, and inspectable `fiber.json`. Arrays are base-volume ZYX plus
original arclength in the fourth column; JSON coordinates are XYZ. Original
control-point/segment provenance is retained for JSON fibers.
`orientation.npy` holds the doubled-angle vectors on the `current.npy` material
lattice. Geometry-only archives from before orientation refinement load as an
explicit zero rotation `(1,0)` and are upgraded in the live temporary store;
`fiber.json` remains a geometry-only inspection export.
Checkpoints reference the archive filename and SHA256. Keep the matching archive
beside a checkpoint when moving it. Resume checks its integrity and dataset
identity; changed source membership fails explicitly. Original annotations are
never overwritten. Shared geometry remains owned by the training supervisor
during GPU pause/reallocation, and is removed when that supervisor exits.

To train a fresh model on an already refined training corpus, set
`update_enabled: false`, `state_checkpoint` to the refinement checkpoint, and
`state_sha256` to its recorded archive digest. This is not model resume: the new
training has fresh weights, optimizer, EMA and update0. The source archive is
restored read-only, validated against every training fiber, and never copied into
new snapshots. No proposals, solves, commits, or refinement metrics run. Tests use
their immutable original GT rather than incompatible saved crop identities. A
later model resume still restores the same explicit geometry source.

Currently unsupported with refinement: unmanaged multi-GPU launches, conditioned
decoders, historical-test datasets, and test-output upsampling. These fail
explicitly. Standard managed training and native output strides are supported.
Direction guidance additionally requires a single direction branch; multiple
branches fail explicitly rather than arbitrarily choosing one.

## Validation

Run the approved `vesuvius/scripts/run_fiber3d_cooperative_tests` launcher with
`refinement` and `cuda-refinement` suites in its JSON control. CUDA checks acquire
an exclusive manager allocation. Tests cover solver/reference agreement,
projection/taper, normal-sign invariance, queue payloads, stale writes, archives,
worker-count changes and detached GPU feedback with backward. Orientation tests
cover modulo-pi interpolation, periodic tether/cap, invalid probes, anisotropic
alignment, atomic stale writes, legacy/new archives, and target rotation.

Measured 2026-09-15: one CPU thread, synthetic 24-node curves, affine + smooth
warps, four iterations, ten repeats: 16-patch solve min/median/max
34.43/34.52/34.65 ms (464 patches/s). Earlier per-patch Torch mapping measured
7.33 ms, native scalar mapping 2.98 ms; batching is 2.16 ms/patch. This excludes
normal loading, D2H, distributed commit and archive writes, and is not an
end-to-end training speed claim. The bending update measured 34.87/35.07/35.21 ms
on the same workload (ten repeats, one CPU thread), versus displacement smoothing
34.81/38.24/42.15 ms; host variation precludes a speedup claim. Default shape weight
0.01 contributed 0.067% of the presence-only update in the noisy-line synthetic
check; this is not a convergence-quality guarantee.

Console output reports mean weighted post-solve `loss_presence` (1 minus presence),
`loss_smoothing_normal`, `loss_smoothing_tangent`, `loss_direction`,
`loss_localization` (original-GT ellipse tether), and their sum,
per training macro-batch and test wave. Terms are normalized by each patch's
material arclength and averaged over committed patches only (`patches=N`); invalid
or stale proposals are excluded. Empty batches report unavailable, not zero losses.
TensorBoard `refinement_train/loss_*` reports each training batch; `refinement_test/loss_*`
reports the patch-count-weighted mean over the complete test pass, including across
managed pause/resume. Raw, EMA99 and EMA999 consume identical targets before raw
feedback is committed. Cumulative time/accepted/stale TB metrics were removed.
TB also retains `loss_smoothing` as the sum of its two components, not counted
again in total. These are post-solve objective diagnostics, not model training losses.

Anisotropic/direction version: same 16-patch synthetic workload, one CPU thread,
four iterations/ten repeats, guided solve plus post-solve loss reporting
107.12/107.39/111.66ms min/median/max. Isotropic solve-only on this version was
37.41/37.48/37.69ms; different work is included, so this is not a like-for-like
speed comparison. Tests include dense anisotropic reference, axis-sign invariance,
direction finite-difference gradient and nonlinear warp pullback.

Orientation version measured 2026-09-30 on the same 16-patch, 24-node,
four-iteration affine+smooth synthetic workload (ten repeats, one CPU thread):
111.62/111.75/115.99 ms min/median/max, or 143.2 patches/s. The geometry-only
batched solve in the same invocation measured 38.94/39.04/39.15 ms. This is
solver-only host timing, not an end-to-end training-throughput claim.

Managed production continuation was launched from
update11800. The first fixed-test-crop continuation incorrectly resumed refinement
update13300; it was replaced, at user request, with the original non-refinement
update11800 checkpoint and fresh train/test refinement state.
