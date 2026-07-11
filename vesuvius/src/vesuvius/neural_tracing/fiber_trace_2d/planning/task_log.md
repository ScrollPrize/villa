# Whole-Fiber Trace2CP Normal-Alignment Task Log

## Notes

- The visual reversal was a y-axis flip from Lasagna normal sign ambiguity, not
  an x-order problem. Each pair-local Trace2CP segment built its side-strip row
  axis independently, so adjacent segments could choose opposite normal signs.
- Added signed line-window normal metadata to `FiberStripSegmentSample`.
- Added optional Trace2CP segment row-axis alignment: if a caller supplies a
  reference line index and row-axis vector, the segment first builds the grid,
  checks the actual generated row axis at that line index, and rebuilds with
  the local Lasagna normal sequence flipped when the row axis disagrees.
- Whole-fiber Trace2CP now carries accepted CP row axes by original line index.
  When a later pair shares a CP, it aligns its local segment to that stored row
  axis before sampling image data.
- Added `trace2cp_fiber_debug.txt` and matching stdout rows with per-pair
  start/target strip CP vectors, start/target row axes, frame vectors, and 3D
  CP deltas projected into the start frame.
- Removed the sparse per-pixel whole-fiber image splat compositor that caused
  display holes. The compositor now uses dense valid-mask rectangular averaging
  over already sampled pair images.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - passed: `138 passed in 6.68s`.
- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - passed.
- `git diff --check -- <task-touched files>`
  - passed.
- Real command rerun with expanded local paths:
  `PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.runner /home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/configs/loader_example.json --export-dir ./ --checkpoint /home/hendrik/business/aiconsulting/vesuviuschallenge/data/fiber_trace_2d_runs/cps_only_20260711_132923/snapshots/best.pt --trace2cp-vis --fiber-json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_2026-06-16/test/er_20260615T190505820_000003.json`
  - passed in `/tmp/trace2cp_debug_current`.
  - produced `trace2cp_fiber_vis.jpg`, `trace2cp_fiber_summary.txt`, and
    `trace2cp_fiber_debug.txt`.
  - debug rows: `121`; aligned shared-CP row-axis dots: `120`; min
    `alignment_dot=0.999074`; no negative shared-axis alignments.
