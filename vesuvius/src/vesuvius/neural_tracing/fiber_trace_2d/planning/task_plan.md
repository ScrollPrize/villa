# 3D Fiber Follow-Up Plan

## Current State Check

Regular affine augmentation is already present in the 3D loader:

- `augment_shift_zyx` chooses the CP location inside the final patch.
- `augment_rotation_degrees` samples a 3D axis-angle rotation.
- `augment_scale_min/max` and `augment_flip_probability` are also wired.
- `_sample_volume_patch` samples the 3D source block with the same
  output-to-source affine transform.
- `_build_targets` currently transforms the source fiber line into output patch
  coordinates with the paired source-to-output affine matrix.
- Dataset-level fiber-coordinate affine transforms are also supported through
  `fiber_transform`, `transform`, `fiber_transform_json`, and
  `transform_invert`.

Shear/skew and ringing are intentionally out of scope. Non-zero 3D shear or
ringing config keys should continue to fail loudly.

## Goal

Extend the 3D CP-centered fiber path with the missing planned pieces while
preserving the simpler non-strip 3D loading model:

1. smooth displacement augmentation in 1D, 2D, and full 3D;
2. anisotropic directional 3D blur as a value augmentation;
3. Trace2CP evaluation/visualization wiring for real 3D checkpoints.

## Smooth Displacement Design

The 3D path must use the same geometric augmentation contract as the 2D strip
path. Every geometric transform is represented by paired concrete maps built at
augmentation construction time:

- `backward_map_zyx[D,H,W,3]`: output voxel -> source voxel, used for volume
  sampling;
- `forward_map_zyx[Dsrc,Hsrc,Wsrc,3]`: source voxel -> output voxel, used for
  fiber line/control-point lookup and label construction.

Consumers must not derive one direction from the other after the fact. That
means no dense nearest search, brute-force inversion, iterative solver, or
runtime formula re-evaluation in training, visualization, or Trace2CP bridge
paths. If a path needs source-to-output coordinates, it samples the prebuilt
`forward_map_zyx`, exactly like 2D samples `forward_map_xy`.

This is the correction to the earlier V0 note: labels must not be derived only
from the output-to-source map. The line/CP geometry is transformed through the
prebuilt source-to-output map, then labels are generated in output coordinates
from that transformed line.

### Smooth Modes

Add one shared map-building implementation with active component/domain masks:

- `none`: current affine-only behavior.
- `1d`: one smooth displacement component as a function of one coordinate.
  Example: move `z` as a smooth function of `x`; the paired maps apply the same
  function with opposite sign in the forward/backward construction.
- `2d`: one or more displacement components generated from a smooth 2D control
  lattice and extruded along the third coordinate, again only in forms where
  both map directions are generated directly.
- `3d`: paired smooth displacement constructed by an explicitly invertible
  composition, not by arbitrary dense 3D inverse estimation. The first supported
  3D mode should be a sequence of smooth coupling/triangular updates whose
  inverse is known by construction, for example component updates such as
  `z += f_z(y,x)`, `y += f_y(z,x)`, `x += f_x(z,y)` with the inverse applying
  the same controls in reverse order and subtracting the offsets. This gives
  all three axes smooth displacement while preserving a direct paired
  forward/backward map construction.

Suggested config keys:

- `augment_smooth_displacement_mode`: `none | 1d | 2d | 3d`;
- `augment_smooth_displacement_amplitude_zyx`: max absolute source-voxel
  displacement per component;
- `augment_smooth_displacement_control_spacing_zyx`: coarse lattice spacing;
- `augment_smooth_displacement_probability`: default `0.0` until profiled.

Implementation details:

- Generate deterministic coarse displacement controls from sample index and
  seed.
- Upsample/interpolate controls only during paired map construction.
- Compose affine and smooth stages into both concrete maps at construction time.
- Use `backward_map_zyx` for dense 3D `grid_sample`.
- Use direct trilinear gather against `forward_map_zyx` for source line and CP
  coordinates.
- Include max displacement amplitude in `_prefetch_source_bbox` and
  `_actual_source_bbox`.
- Keep labels ignored where transformed line/CP lookup is invalid or outside
  the output patch.

## Anisotropic 3D Blur

Keep blur as a value augmentation after volume sampling and after geometric
coordinate sampling. It must run on torch tensors, preferably GPU when training
uses CUDA.

Add config keys:

- `augment_anisotropic_blur_probability`;
- `augment_anisotropic_blur_sigma_along`;
- `augment_anisotropic_blur_sigma_across`;
- `augment_anisotropic_blur_orientation`: `fiber | random | axis`;
- optional `augment_anisotropic_blur_roll_degrees`.

Implementation plan:

1. Keep existing `augment_blur_sigma` as isotropic separable 3D Gaussian.
2. Add anisotropic blur as opt-in and default off.
3. For axis-aligned blur, use separable `conv3d`.
4. For arbitrary/fiber-aligned blur, use three sequential oriented 1D
   Gaussian samples with `grid_sample` along the principal directions, batched
   over patches where possible.
5. The default principal direction for `fiber` orientation is the transformed
   local CP tangent; the two perpendicular axes are built deterministically
   from that tangent plus optional roll.
6. Profile before enabling in checked-in configs.

## Trace2CP Metric Wiring

The current `fiber_trace_3d/projection.py` only provides a synthetic-tested
3D-to-2D direction projection helper. Full wiring should add a 3D model adapter
that can feed the existing 2D Trace2CP metric.

Plan:

1. Add a 3D Trace2CP adapter module, e.g.
   `fiber_trace_3d/trace2cp_bridge.py`.
2. Reuse the existing 2D loader only for Trace2CP geometry construction:
   CP pair, side-strip coordinates, local strip x/y axes, valid mask, and
   metric calculation.
3. For each test CP pair, build ordinary axis-aligned 3D inference blocks that
   cover the Trace2CP strip coordinates. Do not load strips for 3D training.
4. Run the 3D model densely on those blocks.
5. Interpolate the 3D model outputs at the 2D strip coordinates:
   - six Lasagna 3x2 direction channels;
   - presence channel.
6. Decode/project the 3D direction into the local 2D strip frame using the
   projection helper, but replace the current grid-search decoder if a faster
   analytic least-squares/direct decode is added.
7. Feed the projected 2D direction field and projected presence field into the
   existing Trace2CP bidirectional metric path.
8. Add optional 3D Trace2CP visualization:
   - side-strip source image from the existing 2D geometry;
   - projected 3D direction overlay;
   - projected 3D presence;
   - public `trace2cp_error=...` line and whole-fiber
     `trace2cp_error_mean=...` line.
9. Add 3D training config keys:
   - `training.test_trace2cp_enabled`;
   - `training.test_trace2cp_control_points`;
   - `training.test_trace2cp_step_px`;
   - `training.test_trace2cp_rf_margin_px`;
   - optional 2D metric-loader config path if the 3D config cannot safely
     contain every 2D strip geometry setting.
10. Best checkpoint selection should use `test/trace2cp_error` when this
    metric is enabled; otherwise keep current dense loss selection.

## Shear And Ringing

Do not implement these in this follow-up:

- keep rejecting non-zero `augment_shear_*`;
- keep rejecting non-zero `augment_ringing*`;
- document that no placeholder behavior exists.

## Spec Update

Update `planning/specs.md` to add:

- 3D affine augmentation support status: shift, rotation, scale, flips present;
- 3D geometric augmentation must follow the 2D paired-map contract:
  `backward_map_zyx` and `forward_map_zyx` are both concrete tensors built by
  construction;
- smooth displacement must be implemented only through paired map construction
  with direct forward/backward semantics;
- target generation transforms source line/control-point coordinates by direct
  lookup against `forward_map_zyx`, then generates output-space labels from the
  transformed line;
- anisotropic blur as opt-in value augmentation, not a geometric transform;
- full 3D Trace2CP bridge semantics and best-checkpoint behavior;
- shear/ringing explicitly out of scope and rejected.

## Docs Updates

Update `docs/code_structure.md` with:

- the 3D geometry augmentation object/map flow;
- where smooth displacement and anisotropic blur live;
- how 3D checkpoints are projected into the existing 2D Trace2CP metric.

## Tests

Add focused tests for:

- affine shift/rotation remains unchanged for identity smooth displacement;
- 1D/2D/3D smooth displacement source-coordinate maps are deterministic and
  finite;
- `backward_map_zyx` and `forward_map_zyx` are both constructed and no lookup
  path tries to invert one from the other;
- source line/control-point lookup through `forward_map_zyx` matches the old
  affine target path for affine-only samples;
- transformed-line output-space labels give expected direction targets for a
  known smooth bend;
- prefetch envelopes include configured displacement amplitude;
- anisotropic blur changes a synthetic impulse with expected directionality;
- shear/ringing non-zero keys still raise;
- synthetic 3D prediction projected onto a simple 2D strip produces the expected
  Trace2CP error;
- existing 2D and 3D focused tests continue to pass.

Regression command:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`

## Implementation Order

1. Refactor 3D affine sampling into a reusable paired coordinate-map builder
   while keeping outputs bitwise or numerically close for affine-only tests.
2. Move label generation to source-line/control-point lookup through
   `forward_map_zyx`, matching the 2D map contract.
3. Add smooth displacement modes and prefetch/source envelope inflation.
4. Add anisotropic blur behind opt-in config.
5. Add Trace2CP bridge and 3D CLI/train metric wiring.
6. Update specs/docs/changelog/task log and run tests.

## Risks

- Full 3D smooth displacement must not be an arbitrary dense field unless a
  direct paired inverse is also constructed. The first supported 3D mode should
  use explicitly invertible coupling/triangular stages; if amplitudes make
  transformed geometry invalid, those samples/voxels should be rejected or
  ignored rather than silently supervised.
- Arbitrary anisotropic blur can be expensive. It should stay off by default
  until benchmarked on the real training path.
- Trace2CP bridge inference may need tiling for long CP-pair strips; the first
  implementation should favor correctness and clear profiling over premature
  batching.
