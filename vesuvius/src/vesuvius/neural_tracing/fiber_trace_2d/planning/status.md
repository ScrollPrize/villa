# Status

Task: construct route and connection geometry from the prediction rather than
from the skeleton's own voxels.

- [x] Capture the task (`task.md`).
- [x] Write the task plan (`task_plan.md`).
- [x] Measure what the prediction constrains: the fibre section is a ribbon,
      and presence places the fibre across the sheet only.
- [x] Sub-ports from the section (centre and extremes), each centred by a
      small local presence step.
- [x] Every line variant and region connection refined from a path-search
      init through the thresholded prediction, and kept.
- [x] Per-sub-port routing: each sub-port's cheapest variant and connection,
      the tracer walks sub-port to sub-port, nothing chosen ahead of tracing.
- [x] Measured: case31 one segment end to end, no restart, maximum 21.17 to
      8.69, p95 13.75 to 5.77, median 2.52 to 2.02; control reproduces it.
- [x] Native tests: section candidates, path search, crossing routes; 14
      cases green; replay suite 298 passed.
- [x] Re-render the benchmark strips.
- [x] Docs, spec and changelog.

Open, for a later task:

- One region on case31 exceeds the 14-item limit and is left unresolved.
- `explicit_choices` case 1 fails for an unrelated reason: the route leaves the
  fibre entirely, maximum 111.
- Whether the seam centre should be a routing target by default; it is off in
  the committed configs.
