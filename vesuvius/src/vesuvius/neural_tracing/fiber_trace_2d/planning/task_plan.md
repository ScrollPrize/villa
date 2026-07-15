# 3D Fiber CP Model Variant Plan

## Goal

Add a sibling 3D CP-centered fiber training path that mirrors the current 2D
fiber model workflow where that makes sense: deterministic CP sampling,
coordinate-space augmentation before volume sampling, GPU value augmentation,
TensorBoard/debug visualization, prefetch, and benchmark/profile support.

The 2D fiber path remains supported. Shared code should be extracted only for
fiber metadata, deterministic sample ordering, and Zarr/chunk I/O; 3D model
heads, labels, and training should be separate.

Unlike the 2D path, the 3D loader does not build fiber-aligned strips or slices.
It loads ordinary CP-centered 3D volume blocks and applies 3D augmentations
around the CP. That should make loading and inference simpler than the 2D strip
code: no side/top strip frame construction, no strip-z offset stack, and no
fiber-aligned coordinate mesh for the input image.

## Scope

In scope:

- New 3D loader/trainer/configs/tests.
- Shared Zarr chunk loading/prefetch utilities used by both 2D and 3D.
- CP-centered 3D patch loading from JSON and NML fibers with existing Lasagna
  manifest and affine transform handling.
- 3D fiber-direction and fiber-presence supervision.
- Effective batch size around 192 CP patches per optimizer step.
- Initial 3D checkpoint evaluation through 2D strip projection and the existing
  2D tracer/Trace2CP-style test path.

Out of scope for the first implementation:

- Replacing the current 2D tracer/Trace2CP path.
- Full 3D inference/tracing tools.
- Changing 2D label semantics.
- Reintroducing 3D contrastive embedding by default.

## Architecture

1. Add a new sibling package, tentatively
   `vesuvius.neural_tracing.fiber_trace_3d`, instead of overloading the current
   2D package.
2. Keep the older `vesuvius.neural_tracing.fiber_trace` package available, but
   do not build new 3D CP code by mutating its crop loader. Reuse its
   `DirectionConditionedFiberTraceModel`/`Vesuvius3dUnetModel` vocabulary where
   appropriate.
3. Extract only the shared pieces from `fiber_trace_2d` into a common module,
   for example `vesuvius.neural_tracing.fiber_trace_common`:
   - fiber JSON/NML loading and deterministic CP indexing;
   - dataset record resolution and affine transform application;
   - compact in-RAM fiber line geometry;
   - VC3D/Zarr cache request classification and Python prefetch download;
   - parallel chunk/block reads from the cached/remote Zarr volume.
4. Leave compatibility shims in the 2D modules so existing imports and tests
   continue to work.

## 3D Data Loading

1. Build one deterministic CP sample stream shared with prefetch and training:
   sample index -> record -> fiber -> control point -> augmentation seed.
2. For each selected CP, construct an axis-aligned final 3D output patch shape
   `patch_shape_zyx`, defaulting to a small enough cube/cuboid to make
   effective batch size 192 realistic.
3. Build an oversized axis-aligned 3D source block around the CP that covers
   the maximum configured geometric augmentation range. This is a plain volume
   block, not a fiber-aligned strip/slice.
4. Read the source block from Zarr using shared chunk/block loading:
   - optionally round source boxes to chunk boundaries;
   - read chunk-aligned blocks in parallel;
   - preserve exact selected-level voxel values;
   - keep prefetch using the same envelope calculation.
5. Convert the loaded source block to a torch tensor once, then sample final
   augmented output voxels from the source block with a 3D coordinate grid when
   geometric augmentation is enabled. With geometric augmentation disabled, the
   final patch can be a direct crop/view from the loaded block. In both cases
   this is not strip slicing; the model input remains a regular 3D volume patch.
   Labels and CP/line points use the same 3D transform parameters.
6. Reject samples whose transformed CP/fiber supervision cannot be placed
   inside the final valid patch.

## Model

1. Prefer the existing `Vesuvius3dUnetModel` backbone through the same config
   concepts already used by `fiber_trace.model`:
   `features_per_stage`, `unet_base_channels`, `unet_depth`, `strides`,
   `decoder_upsample_mode`, and input channels.
2. Add a 3D fiber head wrapper if needed:
   - direction head: 6 channels using Lasagna's 3x2 ambiguous channel encoding;
   - presence head: 1 sigmoid channel, default enabled;
   - optional embedding head remains supported but disabled by default.
3. Support `batch_size: 192` as the effective logical batch. Add
   `model_micro_batch_size` or equivalent gradient accumulation if dense 3D
   U-Net memory cannot fit all patches in one forward pass.

## 3D Direction Encoding

Use the Lasagna 3D direction layout and formulas, not raw normalized
`(dx, dy, dz)` output channels. Local references:

- `lasagna/tifxyz_labels.py::encode_direction_channels`
- `lasagna/train_unet_3d.py::_encode_dir`
- `lasagna/train_unet_3d.py::compute_targets_3d`

For a unit fiber tangent `(tx, ty, tz)`, emit three ambiguous 2D double-angle
pairs:

- `dir0_z, dir1_z = encode(tx, ty)` for the XY projection / Z slices;
- `dir0_y, dir1_y = encode(tx, tz)` for the XZ projection / Y slices;
- `dir0_x, dir1_x = encode(ty, tz)` for the YZ projection / X slices.

Each `encode(a, b)` uses the Lasagna double-angle formula:

- `cos2t = (a*a - b*b) / (a*a + b*b + eps)`
- `sin2t = 2*a*b / (a*a + b*b + eps)`
- `dir0 = 0.5 + 0.5*cos2t`
- `dir1 = 0.5 + 0.5*(cos2t - sin2t)/sqrt(2)`

The direction loss should compare these six channels directly, with optional
per-pair projection-magnitude weights as in `train_unet_3d.py` so a direction
nearly perpendicular to a projection plane does not dominate the loss.

## Supervision

1. Transform fiber line points and CP positions through the same 3D geometric
   augmentation parameters used for image sampling.
2. Direction targets are derived from the transformed 3D line tangent and
   encoded into the six Lasagna ambiguous direction channels described above.
3. Fiber-presence supervision:
   - positive near transformed fiber-line voxels around the selected CP;
   - negative outside the configured reachable CP/fiber region, balanced against
     positives similarly to the 2D presence loss;
   - ignore patch edges that cannot contain a CP due to configured shift
     margins, matching the 2D edge-handling principle.
4. Initial V0 trains direction + presence only. Embedding loss can remain
   config-supported but off.

## Initial 3D Evaluation Path

1. The first test case should run the 3D model over CP-centered 3D test patches.
2. To compare against the existing 2D tooling, project/interpolate the predicted
   3D direction field onto the same 2D fiber side strip used by the test fiber.
3. Decode/project the six Lasagna direction channels into the side-strip 2D
   tangent direction at each strip pixel, respecting the local strip frame.
4. Run the existing 2D strip tracer/Trace2CP metric on that projected direction
   field.
5. This evaluation bridge is for testing and metrics only. It must not imply
   that 3D training loads fiber-aligned strips.

## 3D Augmentation Matrix

All geometric augmentations are represented as coordinate transforms before
final volume sampling. Value augmentations happen after sampling as batched
torch tensor operations on the configured device.

| Augmentation | 3D handling |
| --- | --- |
| Shift / translation | Extend from 2D `shift_x/y` to `augment_shift_zyx` or `augment_shift_xyz`. Applied as output-space translation of the CP within the final 3D patch. The unreachable border region is ignored for negative supervision. |
| Rotation | Replace 2D in-plane rotation with 3D rotation around the CP/patch center. Support either random SO(3) or constrained axis-angle ranges via config. Direction vectors and line points are rotated with the same matrix. |
| Flips | Allow independent x/y/z reflections. Transform direction vectors with the reflection matrix, then apply sign-ambiguous direction loss. |
| Scale | Support isotropic scale first. Optional anisotropic scale can be added behind explicit config, but default should be conservative because anisotropic scale changes apparent fiber geometry. |
| Shear/skew | Do not enable by default for 3D. The TODO explicitly questions skew; V0 should reject non-zero 3D shear unless a later experiment defines useful bounds. |
| Smooth distortion, 1D | Implement as a smooth displacement along one configured volume axis as a function of an independent patch coordinate, with paired point/volume coordinate transforms and no search-based inversion. It is not a strip offset. |
| Smooth distortion, 2D | Add only if it can be represented by explicit paired coordinate maps over the regular 3D patch. Candidate: smooth 2-component displacement over a plane in patch coordinates. Keep disabled by default until validated. |
| Smooth distortion, full 3D | Treat as experimental. Use a coarse 3D displacement lattice trilinearly upsampled into the output-to-source sampling grid. If exact paired line/CP mapping is not available, do not use it for supervised training labels. No brute-force inverse/search. |
| Brightness | Same as 2D, batched torch operation after sampling. |
| Contrast | Same as 2D, batched torch operation after sampling, centered on valid patch values. |
| Gamma | Same as 2D, batched torch operation after sampling. |
| Noise | Same as 2D but 3D tensor-shaped; deterministic per training stream index. |
| Blur, isotropic | 3D separable Gaussian blur with one sigma for z/y/x, batched on GPU. |
| Blur, anisotropic directional | Add config for stronger blur along one rotated direction and weaker blur along the two perpendicular directions. Implement as a 3D oriented sampling/filter operation or grouped convolution where kernels are shared. Keep disabled by default until profiled. |
| Blur, arbitrary rotation | Represent the anisotropic blur covariance in volume coordinates. Use the current fiber tangent as the default principal direction, with optional random roll. |
| Ringing artifact | Leave as explicit future/experimental augmentation. Do not add a silent placeholder; reject `augment_ringing_*` keys until the artifact model is specified. |
| Chunk-boundary source rounding | Add `round_source_to_chunk_boundaries` for 3D source boxes. Prefetch and training must compute the same rounded source envelopes. |

## Prefetch

1. Add 3D prefetch mode to the new trainer entrypoint.
2. Prefetch computes chunk requests from the full source envelope, not from one
   concrete random augmentation draw.
3. `--prefetch-steps 0` means all deterministic CP samples, matching current
   2D semantics.
4. Progress should report sample progress and download progress separately,
   preserving the current deterministic shuffled sample order.

## Configs

1. Add a 3D example config near the new package, plus an S1A NML config if the
   first implementation targets the current S1A dataset.
2. Recommended first defaults:
   - `batch_size: 192` effective CP patches;
   - `model_micro_batch_size` configurable;
   - `patch_shape_zyx` small enough for smoke training;
   - `augment_enabled: true`;
   - `round_source_to_chunk_boundaries: true`;
   - six-channel Lasagna direction + one-channel presence heads enabled;
   - embedding disabled.
3. Keep 2D config keys valid for the 2D loader. Do not reinterpret 2D
   `patch_shape_hw` or strip-specific keys as 3D settings.

## Spec Update

Update `planning/specs.md` with a new 3D CP model section covering:

- separate 2D and 3D entrypoints;
- shared common fiber/Zarr I/O utilities;
- 3D coordinate-space augmentation semantics;
- the 3D augmentation matrix above;
- 3D model output channel layout: six Lasagna direction channels plus one
  presence channel by default;
- effective batch size versus micro-batch behavior;
- 3D prefetch envelope semantics and chunk-boundary rounding;
- the 2D strip-projection evaluation bridge for initial tests;
- compatibility guarantee that 2D behavior and configs remain unchanged.

## Docs Updates

Update `docs/code_structure.md` with:

- the new `fiber_trace_common` shared module;
- the new `fiber_trace_3d` loader/model/train/config modules;
- how ordinary 3D CP-centered source-block loading differs from 2D strip
  coordinate sampling;
- how 3D predictions are projected onto 2D test strips for the initial metric;
- the run and prefetch commands for 3D smoke/training.

## Changelog

When implemented, add one changelog entry for the new 3D CP fiber training path
and shared I/O extraction.

## Testing

1. Unit tests:
   - deterministic CP sample order matches prefetch/training;
   - 3D affine transforms move CP, line points, and direction targets together;
   - six-channel Lasagna direction targets match
     `lasagna/tifxyz_labels.py::encode_direction_channels` on known tangents;
   - flip/rotation direction targets are sign-ambiguous and finite;
   - chunk-boundary source envelope calculation is stable;
   - smooth 1D distortion uses paired direct transforms, no search/inversion;
   - Zarr fake-array source blocks load the expected values;
   - 3D-to-2D strip projection produces the expected strip tangent for a
     synthetic straight fiber;
   - 2D loader imports and existing tests still pass through compatibility
     shims.
2. Smoke tests:
   - one small synthetic 3D train step on CPU;
   - one synthetic 3D inference -> 2D strip projection -> 2D trace metric run;
   - one CUDA `--benchmark --load-only --profile` run when available;
   - one prefetch dry run on fake/local data.
3. Regression command:
   `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`
   plus new `test_fiber_trace_3d_*.py` tests.

## Open Risks

- Effective batch size 192 may require micro-batching for real 3D U-Net patch
  sizes. The loader should still produce 192 logical patches per step.
- Full 3D smooth displacement is only valid for supervised training if paired
  line/CP mapping is explicit. Otherwise it must remain disabled for training.
- Arbitrarily rotated anisotropic blur can become expensive; it needs profiling
  before enabling by default.
