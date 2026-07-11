# Whole-Fiber Trace2CP Visualization Plan

## Implementation Plan

- Add runner config narrowing for `--trace2cp-vis --fiber-json <path>` so the
  loader receives only that fiber JSON while reusing the configured dataset's
  volume/manifest/cache settings.
- Keep the loader helper that maps the loaded `fiber_json` path to the flat
  sample indices for that fiber's control points.
- Refactor Trace2CP pair evaluation in `runner.py` so the existing single-pair
  export and new whole-fiber export use the same pair loading, model
  inference, optional median-TTA, tracing, and scoring code.
- Add `--fiber-json` to the Trace2CP runner arguments.
- In `--trace2cp-vis --fiber-json <path>` mode:
  - construct the loader from the narrowed one-fiber config;
  - find all in-range CP pairs for the requested non-zero
    `--trace2cp-target-offset`;
  - evaluate each pair with `sample_mode="flat"` and explicit target CP;
  - carry signed line-window normal metadata and actual start/target strip
    row-axis vectors out of each segment sample;
  - before sampling each later segment, align its actual generated strip
    row-axis at a shared CP against an already accepted shared-CP row-axis
    reference when one is available; if the first generated grid disagrees,
    flip the whole local normal sequence and rebuild the grid before sampling;
  - print and write per-pair debug rows with start/target CP strip coordinates,
    strip-space CP deltas, row axes, frame vectors, and 3D CP deltas projected
    into the start frame;
  - skip individual invalid CP-pair segments and record skip reasons;
  - map pair-local centerlines, CP markers, traces, fused lines, and optimized
    lines into a shared fiber arc-length x coordinate system using each pair's
    local start/target CP columns;
  - compose pair-local image data with dense valid-mask rectangular averaging
    rather than sparse per-pixel splatting, so display composition does not
    introduce black scanline holes;
  - render the same four rows as single-pair Trace2CP: full traces, partial
    closest traces, fused line, and optimized line;
  - write `trace2cp_fiber_vis.jpg` and `trace2cp_fiber_summary.txt`;
  - print concise per-fiber summary stats.
- Keep existing single-pair output filenames and behavior unchanged when
  `--fiber-json` is omitted.

## Spec Update

- Document `--fiber-json` whole-fiber Trace2CP mode, one-fiber config
  narrowing, adjacent/in-range CP-pair selection, shared arc-length
  visualization coordinates, four-row long-strip layout, actual row-axis
  sign-alignment behavior, debug vector output, skipped-pair behavior, output
  filenames, and unchanged single-pair behavior.

## Docs Updates

- Update `docs/code_structure.md` with the new runner mode and output files.

## Tests

- Add unit coverage for:
  - mapping a configured fiber JSON path to flat CP sample indices;
  - whole-fiber CP-pair index selection;
  - long-strip Trace2CP composition produces a valid image wider than one pair.
  - whole-fiber export skips an invalid pair and keeps a summary for valid
    remaining pairs.
  - Trace2CP segment loading can align the ambiguous Lasagna row-axis sign to
    a reference row-axis and flips the sampled strip row direction consistently.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog Update

- Add a 2026-07-11 changelog line for whole-fiber Trace2CP visualization.
