# Whole-Fiber Trace2CP Visualization Plan

## Implementation Plan

- Add a loader helper that maps a configured `fiber_json` path to the flat
  sample indices for that fiber's control points.
- Refactor Trace2CP pair evaluation in `runner.py` so the existing single-pair
  export and new whole-fiber export use the same pair loading, model
  inference, optional median-TTA, tracing, and scoring code.
- Add `--fiber-json` to the Trace2CP runner arguments.
- In `--trace2cp-vis --fiber-json <path>` mode:
  - find all in-range CP pairs for the requested non-zero
    `--trace2cp-target-offset`;
  - evaluate each pair with `sample_mode="flat"` and explicit target CP;
  - translate pair-local images, centerlines, CP markers, traces, fused lines,
    and optimized lines into a shared fiber arc-length x coordinate system;
  - write `trace2cp_fiber_vis.jpg` and `trace2cp_fiber_summary.txt`;
  - print concise per-fiber summary stats.
- Keep existing single-pair output filenames and behavior unchanged when
  `--fiber-json` is omitted.

## Spec Update

- Document `--fiber-json` whole-fiber Trace2CP mode, adjacent/in-range CP-pair
  selection, shared arc-length visualization coordinates, output filenames, and
  unchanged single-pair behavior.

## Docs Updates

- Update `docs/code_structure.md` with the new runner mode and output files.

## Tests

- Add unit coverage for:
  - mapping a configured fiber JSON path to flat CP sample indices;
  - whole-fiber CP-pair index selection;
  - long-strip Trace2CP composition produces a valid image wider than one pair.
- Run:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog Update

- Add a 2026-07-11 changelog line for whole-fiber Trace2CP visualization.
