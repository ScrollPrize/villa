# Task Plan: Real Fused Geometric Augmentation Map Tensors

## Implementation

1. Redefine `StripAugmentTransform` around map tensors, not per-call formulas.
   - On construction, build and store:
     - `backward_map_xy`: output pixel -> source pixel, shape
       `[output_h, output_w, 2]`;
     - `forward_map_xy`: source pixel -> output pixel, shape
       `[source_h, source_w, 2]`, or an equivalent dense lookup tensor.
   - Bake all geometric stages into both maps during construction:
     shift, flips, scale, shear, rotation, and smooth offset.
   - Smooth offset is evaluated only while constructing the maps.
   - After construction, no caller should evaluate smooth interpolation or
     affine transform formulas for line/CP points.

2. Build the backward map directly.
   - Keep the current output-grid -> source-grid semantics, but compute it once
     into `backward_map_xy`.
   - Smooth must be baked into the source coordinates before the map is stored.
   - `output_to_source_grid()` returns the cached map tensor.
   - `output_to_source_points(...)` should either be removed from hot paths or
     use lookup against the cached map when needed.

3. Build the forward map as a real lookup map.
   - Produce `forward_map_xy` over source pixel locations.
   - Preferred construction:
     - generate a source pixel grid;
     - apply the exact inverse transform once to every source pixel;
     - store source pixel -> output pixel coordinates.
   - Because this work happens once per transform construction, formula use
     here is acceptable.
   - Smooth inverse is baked here as `source_y -= f(source_x)` during map
     construction, not during line/CP lookup.

4. Make line/CP mapping lookup-only.
   - `source_to_output_points(...)` must sample `forward_map_xy` at source
     point coordinates using bilinear lookup.
   - The function must not:
     - call smooth interpolation;
     - run affine formula stacks;
     - construct random generators;
     - iterate/solve;
     - search a dense output grid.
   - Line and CP stay batched in one lookup call.

5. Reuse the same map object in loader paths.
   - Coordinate augmentation uses `transform.backward_map_xy`.
   - Line/CP mapping uses `transform.forward_map_xy` lookup.
   - `build_strip_patch_from_source` keeps one `StripAugmentTransform` per
     source/params and passes it to both paths.
   - `build_sample` continues to cache map objects and line/CP results by
     params so shared strip-z offsets do not recompute.

6. Update runner/tracing helpers where relevant.
   - Any helper that needs geometric inverse/forward mapping should consume the
     map tensor object rather than rebuilding formula paths.
   - Avoid broad tracing redesign unless required by tests, but do not add a
     new formula-based duplicate path.

7. Profiling/result expectation.
   - The `line` stage should become mostly bilinear lookup + filtering +
     NumPy boundary cost.
   - If `line` remains high, split timing in a follow-up into:
     `line_lookup`, `line_filter`, and `line_numpy`.

## Spec Update

Update `planning/specs.md` to state unambiguously:

- “Fused geometric augmentation” means actual precomputed map tensors, not a
  shared function bundle.
- Smooth, affine, flips, scale, shear, and rotation are baked into the maps at
  construction time.
- Line/CP mapping is lookup/interpolation against `forward_map_xy` only.
- Image/coordinate augmentation uses `backward_map_xy` only.
- No smooth interpolation or affine formula evaluation is allowed in the hot
  line/CP mapping path.

## Docs Updates

Update `docs/code_structure.md` to describe the map-tensor based
`StripAugmentTransform` and loader data flow.

Update `planning/changelog.md`, `planning/status.md`, and
`planning/task_log.md` for this current task only.

## Tests

- Add a unit test that monkeypatches smooth interpolation/control evaluation
  after transform construction and verifies line/CP lookup still works.
- Add a unit test that monkeypatches affine formula helpers after transform
  construction and verifies `source_to_output_points` uses map lookup only.
- Add a test for `forward_map_xy`/`backward_map_xy` round-trip consistency on
  representative points.
- Keep existing tests for:
  - one transform object shared between coord augmentation and line/CP mapping;
  - line+CP batched mapping;
  - strip-z line/CP reuse.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Validation

- Run compile checks for touched Python files.
- Run focused pytest.
- If possible, run the GPU profile command and compare the `line` column before
  and after this map-tensor change.
