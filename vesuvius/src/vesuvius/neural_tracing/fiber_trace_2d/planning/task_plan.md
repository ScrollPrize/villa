# Trace2CP Side-Z Presence Blur Plan

## Implementation

- Add constants for the side-z presence blur radii: z=11, x=5.
- Add a weighted separable Gaussian blur helper for presence stacks shaped
  `[z_layer, y, x]`; invalid pixels must not contribute to the weighted average.
- Add a cached `blurred_presence_for_layer()` API on `_Trace2CpZPlaneCache`.
  It should gather the requested layer's z-radius neighborhood, blur over z and
  x only, and return the requested layer's blurred presence map.
- Route z-search stepwise candidate scoring, image-z candidate scoring, z-DP
  presence stacks, z-corrected presence panels, z-pillar panels, and z-layer
  TIFF presence pages through the blurred cache presence.
- Leave non-z Trace2CP presence scoring unchanged because there is no side-z
  stack to blur.

## Spec Update

- Document that Trace2CP z-search presence is smoothed over side-z layers and
  strip x before use/display, preserving y-localization.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/runner notes with the new cache-level
  presence blur behavior.

## Testing

- Add focused unit tests for the weighted x/z blur helper and for
  `_Trace2CpZPlaneCache.blurred_presence_for_layer()`.
- Run the focused fiber_trace_2d loader test suite:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`.

## Changelog

- Add a short 2026-07-13 entry for side-z presence blurring.
