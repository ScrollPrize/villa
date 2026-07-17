# Native 3D Whole-Fiber Continuous Strip Visualization Plan

## Current State

- `trace_native_3d_whole_fiber(...)` already produces segment results and a
  stitched trace while restarting only after failed segments.
- `_render_native_whole_fiber_segment_panels(...)` currently renders one
  four-panel column per CP segment.
- `_adaptive_trace2cp_cross_strip_height(...)` currently shrinks/expands the
  visual cross-strip height from projected overlays; this is now the wrong
  behavior for whole-fiber visualization.
- Single-pair native 3D rendering still uses the pair-local debug panel path and
  should be left alone unless a shared helper is needed.

## Implementation

- Add a small whole-fiber visualization model:
  - Convert `NativeWholeFiberResult.segments` into restart-delimited visual
    spans.
  - A span starts at the first CP after a restart and ends at the last successful
    target CP before the next restart.
  - Failed segment overlays are clipped before the next CP as they are today,
    but they do not define the next continuous span; the next span starts at the
    failed target CP.
  - Empty/degenerate spans are skipped with an explicit summary/debug reason.
- Add a fixed-width strip-source builder for a CP interval:
  - Build side/top strip geometry over a CP index interval instead of a single
    CP pair column.
  - Reuse the existing Lasagna/VC3D-style segment strip construction and compact
    geometry validity checks.
  - Force `cross_strip_height_px=64` for whole-fiber visualization.
  - Do not call `_adaptive_trace2cp_cross_strip_height(...)` in whole-fiber
    mode.
- Render each restart-delimited span as four full-length strip rows:
  - side volume, side 3D presence, top volume, top 3D presence.
  - Use the same blocking requested-scale volume sampling and native inference
    cache presence sampling as the current segment panels.
  - Tile very long strip coordinate arrays by x-column chunks if needed, so
    memory use stays bounded while the resulting image is still one continuous
    strip per span.
- Overlay traces in the span coordinate system:
  - Project the stitched traced path portions belonging to the span onto the
    span side/top strips.
  - Draw the reference centerline and CP markers as context.
  - Draw successful traced portions continuously.
  - Draw failed clipped portions in failure color up to the existing
    before-target cutoff, then leave a visual restart gap before the next span.
  - If an overlay leaves the 64 px strip, clip only the drawing; do not treat it
    as a tracing or rendering error.
- Compose the whole-fiber JPG:
  - For each span, produce a four-row block.
  - Stitch span blocks left-to-right with a small black restart separator.
  - Keep the row order `side volume`, `side presence`, `top volume`,
    `top presence`.
  - Continue overwriting `trace2cp_native_3d_vis.jpg` after each completed
    segment/span update.
- Keep metrics unchanged:
  - `native_trace2cp_fiber_restart_rate` remains restart count divided by
    segment count.
  - Per-segment status, reasons, and in-plane errors remain in the summary.

## Spec Update

- Replace the native 3D whole-fiber visualization spec that says each CP
  segment is rendered as the next column.
- State that whole-fiber native 3D visualization renders restart-delimited
  continuous long strips.
- State that whole-fiber strip cross-width is fixed at 64 px and is
  visualization-only.
- State that paths leaving the 64 px visualization strip are clipped in the
  overlay only and do not invalidate tracing, metric calculation, or 3D sampling.
- Keep the existing progressive-overwrite requirement.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP description to explain:
  - bare `--fiber-json` uses whole-fiber continuous-strip visualization;
  - restarts split the long strip into separate visual spans;
  - the four rows are side volume, side presence, top volume, top presence;
  - the strip width is fixed at 64 px.
- Add a changelog entry for the visualization rewrite.

## Tests

- Add focused unit coverage in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`:
  - restart-delimited span grouping from synthetic whole-fiber segment results;
  - fixed whole-fiber visualization strip width of 64 px;
  - adaptive-height helper is not used by whole-fiber rendering;
  - failed segment overlay clipping does not drop the following restart span.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Run `git diff --check` over touched files.

## Non-Goals

- Do not change native tracing/search/fusion scoring.
- Do not change single-pair native 3D visualization except shared utility
  extraction if needed.
- Do not make 64 px strip width adaptive through another config key in this
  task.
