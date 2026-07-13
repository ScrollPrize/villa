# Trace2CP Sub-Voxel Z-Step Interpolation Plan

## Implementation

1. Extend `_Trace2CpZPlaneCache` with separate inferred-layer storage.
   - Keep `z_step_voxels` as the DP/search state spacing.
   - When `z_step_voxels >= 1`, keep the current behavior.
   - When `z_step_voxels < 1`, infer side-z slices only at integer
     selected-scale voxel offsets.

2. Interpolate fields for fractional DP/search layers.
   - Map requested state layer `k` to `z_voxels = k * z_step_voxels`.
   - Infer bracketing integer side-z layers `floor(z_voxels)` and
     `ceil(z_voxels)`.
   - Align ambiguous direction signs per pixel before linear interpolation,
     then normalize the interpolated direction field.
   - Linearly interpolate presence when present.
   - Keep image/debug sampling on the nearest inferred slice; the requested
     prediction still carries the fractional `z_voxels`.

3. Route existing callers through the cache API.
   - Stepwise z-search and side DP already call `plane_cache.get(layer)`, so
     they should receive interpolated fields without call-site rewrites.
   - Presence z-pillars should also interpolate through the same cache.
   - TIFF/debug layer export should expose actual inferred slices when
     available, so it does not imply sub-voxel inferences were run.

## Spec Update

- Document that `--trace2cp-z-step-voxels < 1` affects DP/search state spacing,
  not inference spacing.
- Document that direction and presence are interpolated between integer
  side-z inference layers for sub-voxel states.
- Keep the no-image-resampling/no-image-interpolation statement for
  z-corrected image debug output.

## Docs Updates

- Update `docs/code_structure.md` z-search section with the inference/state
  spacing distinction.
- Add a changelog entry.
- Replace `task_log.md` with this task's implementation notes and validation.

## Testing

- Add a cache unit test proving `z_step_voxels=0.5` samples only integer side-z
  offsets while `get(1)` returns interpolated direction and presence at
  `z_voxels=0.5`.
- Keep existing side-z geometry and Trace2CP tests passing.
- Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py
```
