# Task Plan: Coordinate Tensor Boundary Cleanup

## Current Problem

The canonical strip path currently does avoid Python per-pixel loops, but it
still crosses NumPy/PyTorch boundaries too often:

- `build_side_strip_patch_grid_from_line_window_torch` computes dense strip
  coordinates on torch, then immediately converts `coords_xyz`, `coords_zyx`,
  `offset_axis_*`, and `valid_mask` back to NumPy.
- `build_strip_patch_from_source` then converts those NumPy arrays back to
  torch inside `_resample_coords_like_augmentation` for geometric augmentation,
  and converts the augmented coordinates back to NumPy for VC3D sampling.
- `transformed_centerline_coords` and `transformed_source_point_coords` compute
  with torch but return NumPy, forcing line/CP coordinate consumers to use CPU
  arrays even when the rest of augmentation is already on the configured device.
- Value augmentation must still accept NumPy image data from VC3D sampling, but
  that should be a single image-value boundary, not part of coordinate cleanup.

## Scope

Keep the existing VC3D/Lasagna geometry semantics, deterministic sample order,
augmentation behavior, prefetch behavior, and public runner/training outputs.
This task is a data-representation cleanup only: fewer conversions and clearer
ownership of tensor vs NumPy boundaries.

## Implementation Plan

1. Add an internal torch-native strip grid representation.
   - Introduce a small dataclass, for example `FiberStripGridTorch`, carrying
     `coords_xyz`, `coords_zyx`, `valid_mask`, `offset_axis_xyz`,
     `offset_axis_zyx` as torch tensors plus the existing frame metadata.
   - Keep the existing NumPy `FiberStripGrid` for public/backward-compatible
     callers and tests that expect NumPy.

2. Split torch grid construction from NumPy conversion.
   - Add a torch-returning builder for CP-local source grids.
   - Reimplement `build_side_strip_patch_grid_from_line_window_torch` as a thin
     compatibility wrapper that calls the torch-returning builder and converts
     once to `FiberStripGrid`.
   - Keep `build_side_strip_patch_grid_from_line_window` unchanged unless tests
     show shared helpers should be factored.

3. Keep source geometry and offset grids as torch inside `FiberStrip2DLoader`.
   - Change `_StripSource.grid` to hold the torch grid.
   - Change `_offset_grid_from_source` to add strip-z offsets using torch tensor
     operations on the configured device.
   - Avoid `astype(copy=True/False)` churn in the source/offset path.

4. Make geometric coordinate augmentation torch-in/torch-out.
   - Replace `_resample_coords_like_augmentation` with a torch-native helper
     that accepts torch coords and valid tensors and returns torch coords and
     valid tensors.
   - Keep exactly the same `grid_sample` settings, validity behavior, and
     source-coordinate mapping.

5. Make line/control-point coordinate helpers optionally torch-native.
   - Add torch-returning variants or a `return_numpy`/wrapper split for
     `transformed_centerline_coords` and `transformed_source_point_coords`.
   - Use torch tensors in loader internals until final sample assembly.
   - Preserve NumPy-returning public helpers for runner code that draws with PIL
     and for tests that currently inspect arrays.

6. Centralize final NumPy conversion at explicit boundaries.
   - Before `source.record.sampler.sample_coords(...)`, convert final
     `coords_zyx` and `valid_mask` to contiguous CPU NumPy exactly once.
   - Convert final `line_xy` and `control_point_xy` to NumPy when constructing
     `FiberStripSample`.
   - Keep `image_t.cpu().numpy()` after value augmentation because images come
     from VC3D as NumPy and runner/training batch assembly currently stores
     NumPy arrays.

7. Leave prefetch CPU-pinned but remove duplicate conversions there too.
   - Prefetch still builds coordinates on CPU by design.
   - Convert CPU torch coords to NumPy once before dependency discovery.
   - Do not change chunk-request semantics or path metadata handling.

8. Update profiling labels only if useful.
   - `strip_coords`, `coord_augmentation`, `line_coords`, and `volume_sample`
     should continue to mean the same stages.
   - If the change adds explicit conversion timing, make it narrow and only for
     debugging; do not add noisy permanent table columns unless needed.

## Spec Update

Add to `planning/specs.md`:

- Dense source-strip coordinates, strip-z offset coordinates, geometric
  coordinate augmentation, and transformed line/CP coordinates stay as torch
  tensors on `augment_device` until a consumer explicitly requires NumPy.
- VC3D coordinate sampling and runner/PIL visualization are explicit NumPy
  boundaries.
- Prefetch remains CPU-pinned, but still uses the same torch-native source-grid
  logic and converts once for dependency discovery.

No public config keys or output file formats should change.

## Docs Updates

Update `docs/code_structure.md` to describe the new internal tensor/NumPy
boundary:

- Loader source geometry and coord augmentation are torch-native.
- VC3D sampling is the point where final coordinates become CPU NumPy.
- Export/visualization converts line/CP/image data as needed for PIL/JPG output.

Update `planning/local_development.md` only if validation commands or expected
profile interpretation change.

## Tests

- Run syntax validation for modified modules.
- Run the focused loader tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Add or adjust tests if current coverage does not assert:
  - augmented coords/valid masks match the previous NumPy boundary behavior;
  - public NumPy wrappers still return NumPy arrays;
  - prefetch dependency generation still receives the same unique chunk set for
    a fixed sample/config.

## Changelog

Add a short `planning/changelog.md` entry because this changes internal loader
data flow and profiling/performance expectations, even though public behavior
should remain unchanged.

## Risks

- Torch tensors held longer on CUDA can increase temporary VRAM use. Mitigate by
  keeping only one CP-local source and one offset/augmentation result live at a
  time, as the current loader does.
- CPU prefetch should not accidentally use CUDA. Keep prefetch calls passing
  `device=torch.device("cpu")`.
- Tests that compare exact dtypes or array ownership may need updates, but
  numeric coordinate values should remain equivalent.
