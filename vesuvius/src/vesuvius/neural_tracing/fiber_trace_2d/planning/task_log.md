# Trace2CP Short Z-Search Task Log

## Planning Notes

- Replaced the active task with the requested Trace2CP short z-search.
- Drafted a plan that limits the first implementation to an experimental
  `--trace2cp-z-search` extension of `--trace2cp-combined`.
- The plan keeps direction-only Trace2CP, training loss, best-checkpoint
  selection, and public `trace2cp_error` unchanged.
- The plan interprets z as the existing strip offset axis, measured in
  selected-scale voxels. The default layer step is planned as two selected-scale
  voxels.
- The plan applies `abs(dy) + abs(dz_voxels)` to the connection /
  closest-approach logic, not to the public metric.
- The z-corrected visualization reconstruction is planned as per-image-column
  nearest-layer copy from already inferred layer images. It must not re-sample
  the volume or interpolate image values between z layers.

## Validation

- Planning-only task so far; no code tests run.
