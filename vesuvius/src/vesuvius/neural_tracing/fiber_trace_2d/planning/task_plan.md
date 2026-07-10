# Task Plan: Fused Forward/Backward Augmentation Coordinate Maps

## Implementation

1. Introduce a shared geometric transform helper in `augmentation.py`.
   - Add a small class or pure-function bundle, e.g.
     `StripAugmentTransform`, built from `output_shape_hw`,
     `source_shape_hw`, `FiberStripAugmentParams`, and `device`.
   - It must expose:
     - `output_to_source_points(xy)`;
     - `source_to_output_points(xy)`;
     - `output_to_source_grid()`.
   - Keep tensors on the requested torch device.

2. Represent transforms as composable map functions.
   - Affine components should be represented by paired forward/backward matrix
     or vectorized functions.
   - Smooth offset should be represented as a vertical column-dependent map
     using the existing deterministic `smooth_offset_field` control generation.
   - Compose the functions in one place so the backward map used for image
     sampling and the forward map used for line/CP coordinates are paired.

3. Remove smooth nearest-search line/CP mapping.
   - Replace the smooth branches in `transformed_centerline_coords_torch` and
     `transformed_source_point_coords_torch`.
   - Delete or stop using `_nearest_output_pixels_for_source_points` for normal
     augmentation line/CP generation.
   - Source-to-output line mapping should be vectorized over the cached line
     points and should not build an `H x W` source grid unless the caller asks
     for the grid.

4. Adapt loader line generation.
   - `_line_and_cp_xy_for_params` should use cached `source_line_xy` and
     `source_control_point_xy`.
   - For `params is None`, return cached source tensors directly.
   - For augmented params, build the fused transform once and apply
     `source_to_output_points` to cached source line/CP tensors.

5. Preserve existing sampling behavior.
   - `source_coordinate_grid_for_output` should call the shared transform's
     `output_to_source_grid()` so coordinate augmentation and runner TTA still
     use the same backward map as before.
   - `_resample_coord_tensors_like_augmentation` should continue to sample once
     from the fused backward map.

6. Runner/TTA cleanup where in scope.
   - Update runner helpers that call `source_coordinate_grid_for_output` to keep
     working through the shared transform.
   - Do not redesign the full tracing API in this task, but avoid introducing a
     second transform implementation.

## Spec Update

Update `planning/specs.md` to state that geometric augmentation exposes paired
fused forward/backward maps, and smooth vertical offset line/CP mapping must use
the analytic map rather than nearest-pixel search.

## Docs Updates

Update `docs/code_structure.md` to document the shared transform helper and the
new line/CP path through cached source coordinates plus
`source_to_output_points`.

## Tests

- Add unit tests that compare affine forward/backward round trips.
- Add unit tests for smooth offset:
  - `output_to_source_points(source_to_output_points(points))` round trips
    within tolerance for representative line/CP points;
  - transformed line/CP generation does not call the nearest-search helper.
- Add a loader test confirming augmented `line_xy` is generated from cached
  source line coordinates through the shared transform.
- Keep existing augment-vis, TTA, direction-supervision, and loader tests
  passing.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add one 2026-07-10 changelog line for fused forward/backward geometric
augmentation maps and analytic smooth-offset line/CP mapping.
