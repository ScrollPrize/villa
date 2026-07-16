# Native 3D Trace2CP Tool Plan

Plan a separate native 3D Trace2CP inspection tool for the 3D fiber tracer.

Requirements:

- Add a separate 3D Trace2CP tool path, distinct from the current 3D-to-2D
  projected Trace2CP bridge.
- Native tracing should operate in 3D volume coordinates, not in a precomputed
  side-strip 2D direction field.
- The tracer should step through candidate directions sampled in a cone centered
  on the inferred 3D direction.
- Candidate selection should maximize direction agreement and fiber presence,
  matching the spirit of the 2D combined Trace2CP direction/presence scorer.
- Stop tracing when the step crosses the target plane around the other CP, not
  by hitting a target image column.
- 3D model inference should be cached in an inferred-block structure:
  - model input patch size and trusted output/core size may differ;
  - crop away an output border from each inferred patch to avoid edge artifacts;
  - overlapping inferred blocks should route point/candidate lookups to the
    block whose trusted core contains the queried point.
- After tracing, create side and top strips from the resulting 3D trace line.
  Reuse existing strip geometry semantics where possible; do not invent a
  simplified planar replacement.
- Produce visualization and stdout metrics from the new native 3D tool.
- Update specs/docs/tests plan before implementation.
