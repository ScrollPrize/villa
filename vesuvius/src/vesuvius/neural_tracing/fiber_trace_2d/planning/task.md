# Task: Median TTA Line Tracing

Implement the `planning/todo.md` item "median of TTA for tracing":

- Use test-time augmentations inside a single trace rather than only drawing
  separate flock traces.
- Trace in the unaugmented reference patch space.
- At each step, warp the current point into each TTA patch space, sample the
  direction there, inverse-warp the orientation back into reference space, and
  use the median direction in reference space.
- Handle the ambiguous direction sign by trying both signs and discarding the
  sign that points more than 90 degrees away from the previous step direction.
- Add a separate `--med-tta` flag. With `--line-trace-vis`, it should add a
  third visualization column after the reference-only trace and the TTA flock.
- Follow-up cleanup: remove hardcoded CPU coordinate generation from
  non-prefetch loader/runner paths. Prefetch may stay CPU-pinned for dependency
  generation.
