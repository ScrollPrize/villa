# Whole-Fiber Trace2CP Visualization

User request:

- Add a Trace2CP variant that accepts `--fiber-json`.
- When `--fiber-json` is present, run Trace2CP over all CP pairs for that
  fiber.
- Run the metric pair by pair, then build the visualization as a second step
  by transforming each pair-local trace/result into one final long strip image.

Interpretation:

- `--fiber-json` identifies a fiber already present in the configured loader
  datasets, so the Lasagna manifest, volume scale, cache, and sampler context
  remain the same as normal Trace2CP.
- Whole-fiber mode uses adjacent CP pairs by default, matching the existing
  default `--trace2cp-target-offset 1`. Other non-zero offsets reuse the same
  argument and only include in-range pairs.
- Each CP pair uses the existing Trace2CP segment loader and tracing/scoring
  path. The long-strip JPG is a visualization-only composition: pair-local
  image pixels and traced points are translated into a shared arc-length x
  coordinate system for the selected fiber.
- Whole-fiber mode writes a separate JPG and text summary, leaving the
  single-pair `trace2cp_vis.jpg` behavior unchanged when `--fiber-json` is not
  passed.
