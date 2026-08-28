# Task log: preserve the complete winding visibility mask

- The viewer currently omits empty artifact layers, rejects partial quartets,
  and derives its rotation domain only from instantiated layers.
- `rotate_visible_winding_mask` discards an incoming bit when its destination
  key is absent, so empty/missing H/V slots corrupt arbitrary masks.
- The required domain is every integer winding between the observed minimum
  and maximum, with an independent slot for every managed state.
- Independent review confirmed physical discovery and logical placeholders
  must remain separate: present files stay strictly validated, absent files are
  not read, and placeholder Shapes layers carry the missing visibility bits.
- Initial winding selection must be computed from real nonempty H/V geometry
  before placeholder layers are materialized.
- The per-reference benchmark rows must reuse the aggregate benchmark's chosen
  gauge offsets and active-only candidate filtering rather than recalibrating
  each reference fiber independently.
- Independent review clarified that the row identity is the original selected
  JSON source, not a clipped run or piece. Rows are deterministic, include
  zero-observation sources, and their sums must exactly match the aggregate.
- The requested row schema was expanded to right/wrong/right-fraction triplets
  for every constraint class and the sum.
- The final display identifies each source by its virtual winding only and uses
  compact class suffixes so the table remains practical in a terminal.

## Implementation

- Sparse physical discovery now accepts any present subset of H/V/error/tie
  artifacts while retaining strict parsing for each present file.
- Layer creation expands the observed minimum/maximum to a complete contiguous
  winding-by-state grid. Missing, empty, and wholly absent intermediate slots
  are real empty managed Shapes layers, so live visibility is the full mask.
- Initial selection is still derived from nonempty H/V geometry before
  placeholders are added. `fiber_count` counts real paths only.
- Reference observations carry their original source JSON ID. One global gauge
  calibration now accumulates per-source, per-class counts plus sums; the CLI
  prints right/wrong/fraction triplets for every source and class before the
  aggregate table.

## Validation

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest
  vesuvius/tests/test_view_fiber_windings.py -q`: 30 passed in 1.72 seconds.
- Built `vc_fiber_trace_chunk` and `test_fiber_trace_winding_bp` with 32 jobs.
- Focused CTest: 1/1 passed in 0.20 seconds.
- A real split-fiber 1024-crop run emitted deterministic source rows in the
  requested position. With the 50-message diagnostic run, source-row sums were
  466 right and 250 wrong, exactly matching the 716-observation aggregate.
- Napari is not installed in this local environment, so real Qt viewer launch
  was not exercised; layer creation and mask behavior were tested through the
  viewer-independent fake-layer harness.
