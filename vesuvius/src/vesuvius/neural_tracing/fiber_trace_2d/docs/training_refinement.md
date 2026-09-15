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
the immutable-original GT ellipse tether and a small displacement-smoothness
penalty. Updates are projected into each current tangent's perpendicular plane,
then capped to 1/8 ellipse. For training only, the cap ramps from zero at the patch
boundary to full at 1/8 patch width, and local section endpoints remain fixed. Every iteration
resamples by arclength while retaining original-reference material coordinates.
The cap applies per iteration; total drift is controlled by the original-GT
tether, not a hard cumulative distance cap.

Geometry math is batched native NumPy/SciPy on CPU after backward; only detached
presence maps leave the GPU. No additional model forward or backward is needed.
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
train/test component's original `baseline.npy`, refined `current.npy`, source
`metadata.json`, and inspectable `fiber.json`. Arrays are base-volume ZYX plus
original arclength in the fourth column; JSON coordinates are XYZ. Original
control-point/segment provenance is retained for JSON fibers.
Checkpoints reference the archive filename and SHA256. Keep the matching archive
beside a checkpoint when moving it. Resume checks its integrity and dataset
identity; changed source membership fails explicitly. Original annotations are
never overwritten. Shared geometry remains owned by the training supervisor
during GPU pause/reallocation, and is removed when that supervisor exits.

Currently unsupported with refinement: unmanaged multi-GPU launches, conditioned
decoders, historical-test datasets, and test-output upsampling. These fail
explicitly. Standard managed training and native output strides are supported.

## Validation

Run the approved `vesuvius/scripts/run_fiber3d_cooperative_tests` launcher with
`refinement` and `cuda-refinement` suites in its JSON control. CUDA checks acquire
an exclusive manager allocation. Tests cover solver/reference agreement,
projection/taper, normal-sign invariance, queue payloads, stale writes, archives,
worker-count changes and detached GPU feedback with backward.

Measured 2026-09-15: one CPU thread, synthetic 24-node curves, affine + smooth
warps, four iterations, ten repeats: 16-patch solve min/median/max
34.43/34.52/34.65 ms (464 patches/s). Earlier per-patch Torch mapping measured
7.33 ms, native scalar mapping 2.98 ms; batching is 2.16 ms/patch. This excludes
normal loading, D2H, distributed commit and archive writes, and is not an
end-to-end training speed claim. TensorBoard exposes cumulative refinement time,
accepted and stale proposals. Managed production continuation was launched from
update11800. The first fixed-test-crop continuation incorrectly resumed refinement
update13300; it was replaced, at user request, with the original non-refinement
update11800 checkpoint and fresh train/test refinement state.
