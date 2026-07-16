# Native 3D Trace2CP Render Brightness And Blocking Guard Plan

## Findings

- Native 3D Trace2CP strip panels used `_image_to_u8(...)` with per-panel
  `1..99` percentile scaling. That changes apparent brightness and is wrong
  for debugging raw volume rendering.
- 3D training model input uses `_normalize_image(...)` according to
  `image_normalization`; the current fast 3D config sets `zscore`.
- Trace2CP side/top strip image sampling delegates to the 2D
  `FiberStrip2DLoader.sample_trace2cp_*` methods.
- Those methods call the configured coordinate sampler, but they did not
  explicitly reject non-blocking samplers or VC3D fine-to-coarse fallback
  caused by chunk errors.
- The VC3D Python binding uses `sampleCoordsFineToCoarse` after blocking
  dependency prefetch. If fine chunks report errors, rendering can still
  produce an image from coarser data unless Python rejects the stats.

## Implementation

- Change native 3D Trace2CP `_image_to_u8(...)` to raw clipped display:
  valid finite values are rounded and clipped to `0..255`; invalid pixels stay
  black.
- Add a `FiberStrip2DLoader._sample_trace2cp_coords_blocking(...)` helper for
  Trace2CP rendering paths.
- Use that helper for side strip, top strip, traced top strip, and side-z strip
  sampling.
- The helper rejects samplers with `blocking=False` and raises on
  `error_chunks > 0` instead of allowing misleading mixed fine/coarse fallback
  images.

## Spec Update

- Document that native 3D Trace2CP rendered volume panels are raw clipped
  `0..255` displays, while training/model input normalization remains the
  configured `image_normalization`.
- Document that Trace2CP strip render sampling must fail on VC3D chunk errors
  instead of silently displaying fine-to-coarse fallback data.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section to describe raw
  clipped rendering and strict Trace2CP strip sampling.

## Testing Plan

- Add a native 3D test proving `_image_to_u8(...)` uses raw clipped values.
- Add 2D Trace2CP loader tests proving render sampling rejects non-blocking
  samplers and reported chunk errors.
- Run focused 3D and 2D loader tests.

## Changelog

- Add a 2026-07-16 entry for raw native Trace2CP rendering and strict
  Trace2CP strip-sampling guards.

## Deviations Or Deferrals

- Strip geometry/orientation correctness is intentionally deferred per the
  user note. This task only fixes basic brightness and blocking/error handling.
