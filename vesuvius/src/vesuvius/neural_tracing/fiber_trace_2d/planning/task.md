# Native 3D Whole-Fiber Continuous Strip Visualization

For native 3D Trace2CP `--fiber-json` whole-fiber mode, replace the current
per-segment column visualization with continuous long strips.

Requirements:

- Whole-fiber visualization should render the full traced fiber length as
  continuous long strip images, split only at failure/restart points.
- Segment traces should not be visualized as independent columns anymore.
- The rendered strip cross-width should be fixed at 64 px, not adaptive.
- If a traced path leaves that 64 px visualization strip, that is acceptable;
  tracing and 3D sampling are independent of the strip and the strip is only for
  visualization.
- Preserve the existing whole-fiber tracing semantics: continue from successful
  target-plane crossings and re-initialize only at failure/restart points.
- Preserve progressive output: the regular visualization output path should be
  overwritten as segments complete so long runs show partial progress.
