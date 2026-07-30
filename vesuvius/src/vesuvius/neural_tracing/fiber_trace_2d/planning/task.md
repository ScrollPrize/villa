# VC3D Fiber-Global Tracing Mode

## User Request

- Support extrapolation and interpolation with the normal-based Lasagna
  optimizer and with the trained fiber model.
- Add a global switch per fiber and persist it with that fiber.
- When trained-fiber interpolation fails, fall back to Lasagna optimization for
  that segment.
- A Lasagna-normal segment neighboring a trained-fiber segment must use the
  trained segment's control-point direction, derived from the adjacent dense
  line point, as its optimization continuation direction.
- Switching the fiber-global mode between Lasagna and trained fiber must run a
  full rebuild: the existing full optimization for Lasagna, and per-segment
  tracing for trained fiber.
- Add a spin box or equivalent control for extrapolation distance if one does
  not already exist.

## Scope

- VC3D line-annotation dialog, session, stored-fiber JSON, and optimization
  orchestration.
- Shared native tracer support required for bounded open-ended extrapolation.
- Shared Lasagna reinitializer use for per-segment fallback and continuation
  from neighboring native-traced spans.
- Focused persistence, orchestration, extrapolation, and UI tests.

## Correctness Constraints

- The mode is fiber-wide; per-segment `segment_to_next` remains the source of
  truth for which interpolated spans were successfully native traced.
- Native interpolation failure is local to one CP-to-CP span and must not undo
  successful native spans.
- Fallback must use the shared Lasagna optimizer. Native spans are passed as
  protected spans so their endpoint directions seed neighboring fallback spans.
- Native and Lasagna modes both produce open tails of the configured base-voxel
  distance. Native tail failure falls back to the Lasagna-generated tail.
- Mode changes and optimization results are transactional and run in the
  existing background-task/edit-blocking flow.
- Stored line/control geometry remains in base coordinates. Native inference
  remains in trace coordinates through the existing coordinate adapter.
- Existing version-1/version-2 fibers without a stored mode default to Lasagna.

## Out Of Scope

- Changing native trace scoring, fusion, multi-plane intersection semantics,
  or the 20-base-voxel interpolation acceptance threshold.
- Running PyTorch or model inference inside VC3D.
- Persisting the UI's extrapolation-distance preference in every fiber JSON;
  the resulting extrapolated geometry is persisted, while the control remains
  a normal VC3D setting.
