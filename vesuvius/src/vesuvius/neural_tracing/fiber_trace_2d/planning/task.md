# Whole-Fiber Trace2CP Visualization

User request:

- Add a Trace2CP variant that accepts `--fiber-json`.
- When `--fiber-json` is present, run Trace2CP over all CP pairs for that
  fiber.
- Run the metric pair by pair, then build the visualization as a second step
  by transforming each pair-local trace/result into one final long strip image.

Interpretation:

- `--fiber-json` should load exactly that fiber JSON instead of requiring it
  to already be part of the configured dataset glob/list. The runner should
  still reuse the configured dataset's Lasagna manifest, volume scale, cache,
  and sampler context.
- Whole-fiber mode uses adjacent CP pairs by default, matching the existing
  default `--trace2cp-target-offset 1`. Other non-zero offsets reuse the same
  argument and only include in-range pairs.
- Each CP pair uses the existing Trace2CP segment loader and tracing/scoring
  path. The long-strip JPG is a visualization-only composition: pair-local
  image pixels and traced points are mapped into a shared arc-length x
  coordinate system for the selected fiber. The mapping must use each pair's
  local start/target CP columns, so reversed local segment orientation flips
  the segment image data and overlays consistently.
- Whole-fiber visualization should use the same four-row structure as
  single-pair Trace2CP: full traces, partial closest traces, fused line, and
  optimized line. Each segment panel must be rendered by the same regular
  Trace2CP visualization path as single-pair mode; whole-fiber composition must
  not redraw partial traces or other rows with a separate custom renderer.
- Adjacent whole-fiber pair strips must keep a consistent vertical strip row
  direction. Lasagna normals are sign-ambiguous, so each pair-local segment
  may otherwise choose `normal` versus `-normal` independently and flip the
  sampled image in y. Whole-fiber mode should align a segment's actual strip
  row-axis vector at a shared CP against already accepted shared-CP row axes
  before image sampling.
- Whole-fiber mode should write a debug text file and print concise per-pair
  strip CP vectors: start/target CP strip coordinates, strip-space CP delta,
  start/target row axes, frame vectors, and 3D CP delta projected into the
  start frame.
- Whole-fiber mode should skip invalid CP-pair segments, for example zero
  Lasagna `grad_mag` samples in that pair's local line window, and list those
  skips in the summary. It should only fail if every requested pair is skipped.
- Whole-fiber mode writes a separate JPG and text summary, leaving the
  single-pair `trace2cp_vis.jpg` behavior unchanged when `--fiber-json` is not
  passed.
