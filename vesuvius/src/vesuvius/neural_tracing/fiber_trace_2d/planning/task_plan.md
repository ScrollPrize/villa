# Trace2CP Side Z-Axis Correction Plan

## Current Diagnosis

- `build_side_strip_patch_grid_tensor_from_line_window()` builds side-strip
  pixels as `center + mesh_normal * row_offset`.
- For side strips, `FiberStripGridTorch.offset_axis_xyz` is currently also the
  interpolated mesh-normal field. That is correct for the image row axis, but
  wrong for Trace2CP z-search.
- `_Trace2CpZPlaneCache.get()` samples z layers by calling
  `loader.sample_trace2cp_segment_source(..., strip_z_offset=...)`, which
  delegates to `_offset_grid_from_source()` and therefore shifts the side strip
  along the side image y axis.
- Regular stepwise z-search and the side DP backend both consume the same
  z-plane cache, so fixing the cache/layer construction fixes both. The
  side/top z experiment also samples local top patches from side-trace z values
  and must use the same out-of-plane interpretation.

## Implementation

1. Add an explicit side out-of-plane axis for Trace2CP side sources.
   - Compute it from the same side-strip frame used to build the strip:
     `side_axis = normalize(cross(mesh_normal, tangent))` with sign aligned to
     the existing frame side axis.
   - Keep the existing side-strip row axis as mesh-normal; do not reinterpret
     `grid.offset_axis_xyz` globally.
   - Store/expose the side out-of-plane axis on `_Trace2CpSegmentSource` or via
     a Trace2CP-only helper so every z-layer caller uses the same field.

2. Add a Trace2CP side-z coordinate helper in `loader.py`.
   - Build side z-layer coordinates as:
     `base_side_coords_xyz + side_z_axis_xyz * (layer_offset_voxels *
     volume_spacing_base)`.
   - Return the shifted `coords_xyz/coords_zyx` with the original side-strip
     validity mask.
   - Keep regular center side-strip sampling unchanged when no z offset is
     requested.

3. Route all Trace2CP side z-layer creation through that helper.
   - `_Trace2CpZPlaneCache.get()` should request a side-z layer offset, not the
     generic strip-row offset.
   - `trace2cp_segment_coords_xyz()` and `sample_trace2cp_segment_source()`
     should either take an explicit axis mode for Trace2CP or have dedicated
     side-z variants, so callers cannot silently choose the wrong axis.
   - Z-layer TIFF, OBJ, and z-corrected image assembly should continue to read
     from the z-plane cache, so they inherit the corrected geometry.

4. Update stepwise and DP Trace2CP semantics.
   - Regular `--trace2cp-combined --trace2cp-z-search` keeps its current
     candidate scoring logic, but layer `k` now means out-of-plane selected-scale
     voxels from the side strip.
   - `--trace2cp-dp` keeps its `(side_z_layer, y, prev_dy, prev_dz)` state, but
     side_z_layer now maps to the side out-of-plane axis.
   - Remove or update any debug label/text that implies z layers are side-row
     shifts.

5. Fix the side/top z experiment axis usage.
   - `_trace2cp_side_top_z_source_arrays()` should expose both the side-row
     mesh-normal axis and the side out-of-plane axis.
   - `_trace2cp_side_top_z_axes_at_point()` should:
     - sample the side-row mesh-normal for side-view direction lifting;
     - apply `z_offset_voxels` along the out-of-plane side axis;
     - construct local top patches using tangent and out-of-plane side axis.
   - The experiment should still re-run top inference at each accepted step; it
     should only change which 3D axis the stepwise z coordinate represents.

6. Clean up misleading top/presence debug handling.
   - Remove or relabel the old "z-pillar" presence panel if it is just a
     resampled side-y stack and no longer represents the corrected side-z axis.
   - If we keep a z-stack presence diagnostic, construct it from corrected
     side-z layers so rows correspond to out-of-plane z layers, not side-strip
     y.

## Spec Update

- Replace the current statement that side z layers use `grid.offset_axis_zyx`
  for side strips.
- Define side-strip axes explicitly:
  - side x: fiber tangent / arc direction;
  - side y: Lasagna mesh-normal row axis;
  - side z: out-of-plane side axis, approximately `mesh_normal x tangent`.
- State that Trace2CP z-search, side DP, z-layer TIFF/OBJ exports, and
  side/top z experiment must use side z for layer offsets.
- State that non-z side-strip loading and Lasagna normal sign alignment remain
  unchanged.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP section to explain the corrected
  side z-axis and which helpers own it.
- Add a changelog entry describing the z-axis correction.
- Replace the current task log with implementation notes and validation results
  for this task once implemented.

## Testing

- Add a geometry/unit test that builds a small side strip and verifies:
  - side-row displacement is aligned with the mesh-normal row axis;
  - side-z layer displacement is approximately orthogonal to that row axis;
  - side-z displacement is aligned with the frame side/out-of-plane axis.
- Add a Trace2CP z-layer test that compares layer `+1` and `-1` coordinates
  against the explicit side-z axis times `z_step_voxels * volume_spacing_base`.
- Add or update side/top z experiment tests so `z_offset_voxels` changes the
  sampled top patch center along the out-of-plane axis, not the side-row axis.
- Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```

## Review Notes

- The plan preserves the existing center side-strip coordinate generation.
- The plan intentionally does not make visualization-only corrections; it fixes
  the sampled z-layer geometry used by scoring and inference.
- The main implementation risk is ambiguous naming around `normal`, `side`, and
  `offset_axis`; the code should use explicit names like `side_row_axis_xyz`
  and `side_z_axis_xyz` at Trace2CP boundaries.
