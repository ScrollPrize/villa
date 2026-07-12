# Trace2CP Top-Model Direction Debug

- Keep `--trace2cp-top-model-dir-vis` diagnostic-only.
- Given the fused top-model direction field, build a monotone-x path that
  connects the two CP columns on the top-strip center row.
- The path should minimize direction-alignment error under the ambiguous
  direction semantics, i.e. edge cost uses `1 - abs(dot(path_tangent, dir))`.
- Use longer pixel-aligned horizontal transitions, starting with fixed 8 px
  horizontal steps, and integrate direction-alignment error along each
  transition.
- This should avoid the current behavior where the path stays mostly
  horizontal and then takes one large vertical jump.
- Prefer valid fused top-direction pixels, but allow invalid/missing pixels with
  a penalty so the diagnostic path still connects the CPs when the field has
  gaps.
- Draw this path in the top-model direction debug panel alongside the existing
  forward/reverse local top traces for comparison.
