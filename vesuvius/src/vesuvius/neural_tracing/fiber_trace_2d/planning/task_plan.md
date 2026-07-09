# Task Plan: Vectorized Strip And Line Coordinate Generation

## Task

Implement the first item from `planning/todo.md`:

- make strip coord and line coord a (as far as possible) vectorized pytorch coord generation
- then test if it works faster using cuda

## Current Problem

The augment visualization timing shows that, after the first cold-cache volume sample, runtime is dominated by geometry generation:

- `strip_coords`: about 460-480 ms per contact-sheet cell
- `line_coords`: about 60-80 ms per contact-sheet cell

The likely causes are repeated per-cell recomputation and Python/Numpy loops in the side-strip coordinate path and line-coordinate inversion path.

## Scope

In scope:

- Vectorize strip coordinate generation where it can preserve current semantics.
- Vectorize transformed line coordinate generation where it can preserve current semantics.
- Reuse CP-local geometry across augment-visualization cells for one selected sample.
- Keep coordinate-space geometric augmentation as the only geometric augmentation mechanism.
- Keep VC3D blocking coordinate sampling for image reads.
- Measure before/after timings using the existing augment-vis profiling output.
- Add focused regression tests for geometry equality/valid masks/line output shape and behavior.

Out of scope:

- Model/training changes.
- New label targets.
- Changing augmentation ranges or semantics.
- Replacing the VC3D volume sampler.
- Changing Lasagna normal decoding semantics.
- Changing generated images except for acceptable tiny floating-point differences from equivalent vectorized math.

## Design

### 1. Split CP-Local Geometry From Per-Augmentation Work

Add an internal CP-local geometry object or helper result that contains:

- selected record / record index / control point index;
- center strip-z offset;
- augmentation source shape;
- CP-local line window;
- sampled Lasagna normals for that window;
- frame data needed to build strip coordinates;
- unaugmented source strip coordinate grid;
- unaugmented source valid mask;
- unaugmented source centerline pixel coordinates.

For augment visualization, build this once for the selected sample and reuse it for every contact-sheet cell. Per cell, only apply that cell's geometric transform to the already-built oversized source coordinate grid, sample the volume, apply value augmentation, and compute/draw the transformed line coordinates.

### 2. Vectorize Strip Coordinate Generation

Add a torch implementation for the dense parts of `build_side_strip_patch_grid_from_line_window`:

- construct row/column grids with torch tensors on the selected augment device;
- compute arc coordinates for all output/source pixels as a tensor;
- compute segment indices with `torch.searchsorted`;
- evaluate cubic Hermite center positions in bulk;
- gather/interpolate frame normals in bulk;
- normalize interpolated normals in bulk;
- produce `coords_zyx` and `valid_mask` tensors.

Keep the frame construction itself conservative:

- frame transport and roll smoothing can remain in the existing Python/Numpy code initially because the local line window is small and its semantics are easier to preserve there;
- convert the resulting per-line-point frame arrays into torch tensors for dense coordinate generation.

This matches "as far as possible" without risking a behavioral rewrite of the VC3D/Lasagna frame logic.

### 3. Vectorize Transformed Line Coordinates

Replace the current nearest-pixel inversion loop in `transformed_centerline_coords` with a direct vectorized mapping when possible.

Approach:

- add an augmentation helper that can transform source pixel coordinates to output pixel coordinates using the inverse of `source_coordinate_grid_for_output`'s affine/smooth-offset mapping where feasible;
- for the line, start from source centerline coordinates as continuous `(x, y)` points;
- map all points in one torch operation;
- filter points outside the output patch in one tensor expression;
- remove adjacent rounded duplicates vectorially.

If smooth offset inversion is not analytically exact, keep a fallback that is correct but isolate it behind the same API. The first implementation should still remove the Python loop over target `x` positions for affine-only transforms, because affine-only cells are currently common in the contact sheet.

### 4. Keep Correctness Boundaries Explicit

The vectorized path must preserve:

- selected-scale pixel spacing semantics from `base_volume_scale`;
- cubic Hermite interpolation over arc length;
- CP-local line-window behavior;
- local-only Lasagna normal sampling;
- coordinate-space geometric augmentation;
- line overlay generated from coordinates, not transformed raster pixels;
- final VC3D blocking coordinate sampling for the actual image.

Where exact equality is not realistic because torch and numpy evaluate floating point in a different order, use a tight tolerance and verify no visible semantic drift:

- coordinate tolerance: small absolute tolerance in base-volume coordinates;
- valid-mask equality where branch conditions are mathematically identical;
- line coordinates within subpixel tolerance.

## Implementation Steps

1. Add small conversion helpers in `strip_geometry.py` or a new local module:
   - extract frame arrays from existing `FiberStripFrame` list;
   - compute arc lengths / derivatives as torch tensors;
   - bulk Hermite interpolation and normal interpolation.

2. Add a torch-backed strip-grid builder:
   - input: `FiberStripLineWindow`, source/patch shape, strip offset, sampled normals, pixel spacing, device;
   - output: tensors or numpy arrays compatible with existing `FiberStripGrid`;
   - default callers can request numpy output so current sampler APIs remain unchanged.

3. Update loader augment-vis path to build CP-local geometry once:
   - one line window;
   - one Lasagna normal sample set;
   - one oversized unaugmented strip coordinate grid;
   - reuse those for each augment cell.

4. Update normal `build_sample` only if it can share the same helper without expanding scope. Otherwise keep it unchanged for this task and limit optimization to the measured augment-vis path.

5. Replace/vectorize `transformed_centerline_coords` for affine-only transforms first, with a correctness fallback for any transform that cannot be represented exactly yet.

6. Keep profiling keys stable:
   - `strip_coords`;
   - `coord_augmentation`;
   - `line_coords`;
   - `volume_sample`;
   - `value_augmentation`.

7. Record findings and deviations in `planning/task_log.md`.

## Testing

Run focused tests:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

Add or update tests for:

- torch vectorized strip coordinates match the current numpy path on a fake/local fiber fixture;
- valid masks match the current path;
- vectorized line coordinates match the previous affine behavior within subpixel tolerance;
- smooth-offset line-coordinate fallback remains correct if exact vectorized inversion is not implemented yet;
- augment-vis still writes JPG outputs and includes fixed-thickness coordinate-drawn line overlays.

## Performance Validation

Use the existing local augment-vis command from `planning/local_development.md`:

```bash
PYTHONPATH=$SRC/volume-cartographer/build/python-bindings/python:$SRC/vesuvius/src:$SRC python -m vesuvius.neural_tracing.fiber_trace_2d.runner $SRC/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir /tmp/fiber_trace_aug_vectorized --sample-index 1 --augment-vis
```

Capture before/after timing rows for at least:

- `unaugmented`;
- `shift_x_min`;
- `shift_y_min`;
- one combined random augmentation cell.

Expected improvement:

- repeated contact-sheet cells should no longer spend about 460-480 ms rebuilding `strip_coords`;
- `line_coords` should drop substantially for affine-only transforms;
- first cell may still be slower because VC3D blocking sampling may fill the chunk cache.

## Spec Update

Update `planning/specs.md` after implementation to state:

- dense strip-coordinate generation may be torch-vectorized, but must preserve VC3D side-strip semantics;
- CP-local source geometry can be reused across augmentation variants for the same selected sample;
- transformed line coordinates should be computed as vectorized coordinate products where possible, with no raster line transform.

No spec change should relax:

- VC3D side-strip equivalence;
- Lasagna normal handling;
- coordinate-space-only geometric augmentation;
- VC3D blocking coordinate sampling for image reads.

## Docs Updates

Update docs after implementation:

- `planning/status.md`: task checklist and current state.
- `planning/task_log.md`: implementation notes, measured timings, and any fallback decisions.
- `docs/code_structure.md`: mention where vectorized strip/line coordinate generation lives.
- `planning/local_development.md`: only if the validation command changes.

## Review Checklist

- Does the plan implement only the TODO item?
- Does it keep current specs intact?
- Does it avoid image-space geometric transforms?
- Does it avoid changing volume sampling semantics?
- Does it include correctness tests and performance measurement?
