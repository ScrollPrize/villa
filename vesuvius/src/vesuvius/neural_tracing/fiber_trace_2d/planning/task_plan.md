# Trace2CP Direction-Aligned Side Presence Blur Plan

## Implementation

- Replace the axis-aligned x/z side-presence blur with a direction-aligned
  anisotropic blur over `[z_layer, y, x]` stacks.
- Keep radius 21 over side-z. Use radius 5 along the local predicted side
  direction in side-image x/y, and a small radius 1 across that direction.
- Use the local side direction field for the output pixel/layer, normalize it,
  and use a symmetric kernel so the Lasagna direction sign ambiguity does not
  affect the blur.
- Keep valid-mask weighted normalization so invalid pixels do not contribute.
- Implement the blur in batched PyTorch. The z pass can remain separable; the
  x/y pass should use vectorized `grid_sample` over layer chunks, not per-pixel
  Python loops.
- Add `--trace2cp-presence-blur`; keep it disabled by default. When disabled,
  z-search presence scoring/display must use raw per-layer presence.
- Preserve the existing `_Trace2CpZPlaneCache.blurred_presence_for_layer(s)`
  API and all existing z-search scoring/display call sites.
- Leave non-z Trace2CP presence scoring unchanged because there is no side-z
  stack to blur.

## Spec Update

- Update the Trace2CP z-search presence blur spec from axis-aligned x/z blur to
  opt-in direction-aligned anisotropic x/y plus side-z blur.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP/runner notes with the new cache-level
  direction-aligned presence blur behavior.

## Testing

- Add focused unit tests for the weighted direction-aligned blur helper and for
  `_Trace2CpZPlaneCache.blurred_presence_for_layer()`, including default-off
  raw-presence behavior.
- Run the focused fiber_trace_2d loader test suite:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`.

## Changelog

- Add a short 2026-07-14 entry for direction-aligned side-presence blurring.
