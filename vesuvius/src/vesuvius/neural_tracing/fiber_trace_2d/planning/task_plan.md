# Task Plan: Use Augment-Vis Loading Path Everywhere

## Scope

Make the augment-vis path the single source of truth for 2D fiber-strip
coordinate generation, augmentation, volume sampling, and prefetch.

The current mismatch is that augment-vis uses the tested torch-vectorized
source-geometry path, while training and prefetch still use older
NumPy/Python strip-coordinate generation and a separate chunk-prefetch wrapper.
This task removes those mismatches.

In scope:

- Refactor loader internals so augment-vis, training batch loading, runner
  normal loading, and prefetch all use the same source-strip and final-coordinate
  generation code.
- Preserve the intended training difference: one selected control point expands
  to multiple strip-z offsets.
- For multiple strip-z offsets, build the CP-local source strip geometry once,
  then derive each offset strip from that source geometry by offsetting along
  the already-computed strip normal/frame direction.
- Ensure prefetch uses the same final augmented coordinates as actual loading.
- Make prefetch independent of specific random augmentation draws by covering
  the maximum configured augmentation envelope for each CP and strip-z offset.
- Fix prefetch/cache behavior so it does not use a cache-incompatible Python
  chunk-store path for VC3D remote volumes.
- Add tests that fail if augment-vis, training batch loading, and prefetch drift
  apart again.

Out of scope:

- Changing model architecture, direction targets, loss semantics, or batch
  composition.
- Changing augmentation parameter distributions.
- Adding multiprocessing or threaded training data loaders.
- Prefetching Lasagna manifest channels.
- Replacing VC3D coordinate sampling semantics.
- Changing cache storage format beyond using the proper VC3D cache-aware path.

## Current Mismatches To Remove

1. Training uses the old strip grid builder
   - Current training path:
     `build_sample -> build_side_strip_patch_grid_from_line_window`.
   - Intended shared path:
     `build_augmented_center_strip_source -> build_side_strip_patch_grid_from_line_window_torch`.

2. Prefetch uses the old strip grid builder and a single random augmentation draw
   - Current prefetch path:
     `chunk_requests_for_sample_index -> build_side_strip_patch_grid_from_line_window`.
   - Intended shared path:
     generate cache-covering coordinates through the same source object, but
     using the configured maximum augmentation envelope rather than one
     specific random training augmentation draw.

3. Augment-vis reuses source geometry, training/prefetch rebuild it
   - Keep source geometry reuse as the normal implementation.
   - Training should reuse the source per selected CP and derive all strip-z
     offsets from it.

4. Augment-vis only renders the center strip-z offset
   - Keep that for contact-sheet display.
   - Training/loading should use the same machinery for all configured
     strip-z offsets.

5. Prefetch uses a separate chunk-store fetch path
   - Current VC3D chunk requests use `_Vc3dChunkStore.__getitem__`, while actual
     loading uses `Volume.sample_coords(..., blocking=True)`.
   - This makes cache behavior differ from augment-vis/training sample loading.
   - Replace prefetch download/fetch behavior with a VC3D cache-aware method.

6. Prefetch cache detection is not VC3D-aware
   - Current prefetch checks `request.store._cache_dir`, which exists for the
     old Python Zarr store but not for VC3D `_Vc3dChunkStore`.
   - The prefetch path must ask the VC3D sampler/volume for cache-aware
     dependency status or perform blocking prefetch through VC3D itself.

## Design

### 1. Introduce A Shared Source Object For All Loading

Refactor the current augment-vis source object into the normal loader path.

Target API shape:

- `build_strip_source(sample_index, *, device, profile=None)`
  - selects deterministic record/control point;
  - computes augmentation source shape;
  - computes CP-local line window;
  - samples Lasagna normals for that line window;
  - builds the center-offset source strip with the torch-vectorized builder;
  - stores enough frame/normal data to derive any configured strip-z offset.

- `build_strip_patch_from_source(source, offset_index, params, *, device, load_image=True, profile=None)`
  - derives the requested strip-z offset coordinates from `source`;
  - applies coordinate-space geometric augmentation;
  - computes transformed line coordinates and transformed CP output coordinate;
  - optionally samples the base volume and applies torch value augmentation;
  - returns `FiberStripSample` plus image/valid data when requested.

Augment-vis should become a thin caller of this shared API for the center
offset and its visualization parameter rows.

Training batch loading should call the same shared API once per selected CP
source and then once per configured strip-z offset.

Prefetch should call the same shared API with `load_image=False` to get final
coordinates and validity masks.

### 2. Prefetch Uses The Configured Augmentation Envelope

Training should continue to use deterministic random augmentation params per
sample and strip-z offset. Prefetch should not.

For each selected CP and strip-z offset, prefetch should cover the full area
that any configured augmentation can read:

- derive the oversized source strip from `augmentation_padding(config.augment,
  patch_shape_hw)`;
- use the source coordinate valid mask/extent that bounds all configured random
  transforms;
- ask the sampler/cache layer for dependencies over that source/envelope
  coordinate set, not over only one sampled `random_combined_augmentation(...)`
  result.

The practical target is conservative cache coverage:

- it may prefetch slightly more chunks than one concrete training draw;
- it must avoid missing chunks for any later draw within the configured
  augmentation limits;
- it must remain CP-local and strip-z-local, not fetch a global volume region.

### 3. Derive Multiple Strip-Z Offsets From One Source

The shared source should avoid rebuilding line windows/normals/frames for every
offset.

Implementation approach:

- Build the center source strip once with `strip_z_offset=0` or the configured
  center offset.
- Preserve a per-pixel or per-column normal/offset direction from the torch
  grid/frame construction.
- For a requested offset:
  - convert offset from selected-scale units to base-coordinate units using
    `record.volume_spacing_base`;
  - add `offset_delta * normal_or_frame_axis_zyx` to source coordinates before
    coordinate augmentation.

If the existing torch grid does not expose the needed offset direction, extend
`FiberStripGrid` or add a closely scoped companion array from
`build_side_strip_patch_grid_from_line_window_torch`. Do not recompute the
full strip grid through the old NumPy builder.

### 4. Make Prefetch Use The Same VC3D Sampling/Cache Path

Avoid the current `_Vc3dChunkStore` cache mismatch.

Preferred implementation:

- Add a method to `CoordinateSampler`, for example:
  `prefetch_coords(coords_zyx_base, valid_mask) -> dict`.
- For `Vc3dCoordinateSampler`, implement it with VC3D volume APIs so the same
  remote cache used by `sample_coords(..., blocking=True)` is used.
  Candidate approaches, in order:
  - add/use a VC3D binding that prefetches collected coordinate dependencies
    through the volume's remote cache and can report cached/missing/downloaded;
  - or call an existing VC3D coordinate dependency prefetch API if one already
    exists;
  - only as a temporary fallback, call `sample_coords(..., blocking=True)` on
    final coords and discard image values, because that at least uses the same
    cache path as actual loading.
- For `NumpyZarrCoordinateSampler`, keep a test/local implementation based on
  explicit chunk requests.

Then make `FiberStrip2DLoader.prefetch(...)` call sampler-level prefetch on the
same shared-source coordinate envelope generated by the loader. Do not use
`_Vc3dChunkStore.__getitem__` for VC3D remote prefetch.

### 5. Add Progress For Both Discovery And Fetch

Since final coordinate generation can still take time over many samples:

- Print a progress row/bar for coordinate/dependency generation.
- Print a separate progress row/bar for actual cache misses/download work when
  the sampler can report it.
- Include enough counters to distinguish:
  - samples processed;
  - strip patches processed;
  - generated unique chunks or dependency keys;
  - already-cached chunks;
  - downloaded chunks;
  - errors;
  - MiB/s and ETA where available.

Do not hide a long CPU-bound discovery phase behind no output.

## Implementation Steps

1. Refactor shared source/patch API
   - Rename or generalize `build_augmented_center_strip_source`.
   - Rename or generalize `build_augmented_center_strip_patch`.
   - Keep compatibility wrappers for runner call sites if useful, but make them
     delegate to the shared implementation.
   - Ensure all geometry uses the torch-vectorized source builder.

2. Add offset derivation from source
   - Extend source/grid data to expose the strip-z offset direction required to
     derive all configured offsets.
   - Replace per-offset calls to the old NumPy builder in `build_sample`.
   - Remove the old per-offset grid-build usage from prefetch.

3. Unify training batch loading
   - Update `build_sample` / `load_batch` so training uses the shared source
     path.
   - Preserve returned shapes:
     - `images`: `[control_point_sample, strip_z_offset, 1, H, W]`;
     - `coords_zyx`: `[control_point_sample, strip_z_offset, H, W, 3]`;
     - `valid_mask`: `[control_point_sample, strip_z_offset, H, W]`;
     - `samples`: flattened per strip-z patch.
   - Preserve deterministic sample-index behavior.

4. Unify augment-vis
   - Keep contact-sheet behavior visually unchanged.
   - Make its center-strip rendering delegate to the same shared source and
     patch function used by training.
   - Keep timing output, raw stats, labels, and fixed-thickness line overlay.

5. Rewrite prefetch around final-coordinate sampler calls
   - Generate augmentation-envelope coords with the shared source path and
     `load_image=False`.
   - Do not use one sampled `random_combined_augmentation(...)` draw to decide
     prefetch coverage.
   - Move cache-aware prefetch into `CoordinateSampler`.
   - For VC3D, use the same volume/cache machinery as blocking sampling.
   - Keep base-volume-only scope.
   - Keep `train.py --prefetch --prefetch-steps N/0` semantics.

6. Remove or quarantine obsolete paths
   - Remove direct training/prefetch use of
     `build_side_strip_patch_grid_from_line_window`.
   - Keep the NumPy builder only for tests/comparison or small compatibility
     helpers if still needed.
   - Remove `_Vc3dChunkStore` from production prefetch if the sampler-level
     prefetch replaces it.

## Spec Update

Update `planning/specs.md` to state:

- The augment-vis source/patch path is the canonical loader path.
- Training, runner batch loading, augment-vis, and prefetch must share the same
  source-strip and final-coordinate generation implementation.
- Training's multiple strip-z offsets are derived from one CP-local source
  geometry by offsetting along strip normals/frame direction, not by a separate
  coordinate-generation path.
- Prefetch must use sampler-level cache-aware VC3D prefetch or blocking sample
  semantics, not a separate Python chunk-store path that cannot see VC3D cache
  state.
- Prefetch must use a conservative configured augmentation envelope per CP and
  strip-z offset, not one sampled random augmentation instance.
- A successful prefetch for a CP/offset should cover later deterministic random
  training augmentations whose parameters are within the configured limits.
- The old NumPy strip-coordinate builder may remain only as a test/reference
  path and must not be used by production training or prefetch.
- Prefetch reports progress for coordinate generation and fetch/cache phases.

Remove or revise any spec wording that implies each strip-z patch is built
independently through separate coordinate-generation machinery.

## Docs Updates

Update `docs/code_structure.md`:

- Describe the shared source/patch loader API and which entry points call it.
- Document the training multi-offset derivation from a single CP-local source.
- Document the sampler-level prefetch/cache path and why it matches
  augment-vis/training sampling.
- Remove wording that says training/prefetch have separate coordinate builders.

Update `planning/local_development.md`:

- Keep the existing augment-vis command.
- Keep the training and training-prefetch commands.
- Add a short note that augment-vis, training, and prefetch now use the same
  source/coordinate/sampler path, so cache/debug behavior should match.

## Testing Plan

1. Shared-coordinate parity tests
   - For a fake/local sampler and augmentation enabled, assert augment-vis-style
     center patch and training batch center-offset patch produce the same final
     coords, valid mask, line coords, and CP output coordinate for the same
     `sample_index` and augmentation params.
   - Assert `build_sample` uses the shared source path by monkeypatching or
     instrumenting the old NumPy grid builder and verifying production loading
     does not call it.

2. Multi-offset derivation tests
   - Use a simple straight fiber and constant Lasagna normal.
   - Build one source and derive multiple strip-z offsets.
   - Compare derived offset coordinates against the existing reference NumPy
     builder within the accepted tolerance.
   - Confirm returned batch shapes and deterministic sample selection remain
     unchanged.

3. Prefetch/load parity tests
   - Instrument sampler calls so prefetch records exactly the final coords and
     valid masks passed to sampler-level prefetch.
   - Load several deterministic random augmentations for the same CP/offset and
     assert their dependency keys are subsets of the envelope dependency keys.
   - Cover at least two strip-z offsets and nonzero geometric augmentation.
   - Include edge-case augmentation params at configured extrema where possible
     so the envelope test is not only checking typical random draws.

4. VC3D cache/prefetch behavior tests
   - Unit-test `CoordinateSampler.prefetch_coords` with the fake sampler.
   - For VC3D, add the strongest local smoke possible without network:
     verify the sampler-level method is called by loader prefetch and the old
     `_Vc3dChunkStore` path is not used.
   - If a remote smoke is run manually, repeat the same prefetch twice and
     confirm the second run reports already-cached work instead of redownloading
     all chunks.

5. Existing focused suite
   - Run:

     ```bash
     PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
     ```

6. Local smoke commands
   - Run augment-vis command and confirm output still renders.
   - Run `train.py --help`.
   - If local cache/network conditions allow, run:

     ```bash
     PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.train $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --prefetch --prefetch-steps 1
     ```

## Changelog Update

Add a dated changelog entry because this changes core loader/prefetch behavior:

- Unified training, runner loading, augment-vis, and prefetch around the
  augment-vis-style source/coordinate/sampler path.
- Prefetch now uses cache-aware sampler behavior consistent with actual
  blocking volume sampling.

## Review Checklist

- No production training or prefetch path calls the old NumPy strip-coordinate
  builder directly.
- Augment-vis remains visually unchanged except for any timing improvements.
- Training still returns all configured strip-z offsets.
- Multi-offset training derives offsets from one CP-local source geometry.
- Geometric augmentation remains coordinate-space only.
- Value augmentation remains torch/device-based after Zarr sampling.
- Prefetch uses the same final augmented coordinates as loading.
- Prefetch coverage is augmentation-draw-independent and conservative over the
  configured max augmentation envelope.
- Prefetch cache behavior is VC3D-aware and no longer treats every VC3D chunk
  as missing just because `_cache_dir` is absent.
- Lasagna manifest channels are still not prefetched.
- Deterministic sample-index behavior is unchanged.
