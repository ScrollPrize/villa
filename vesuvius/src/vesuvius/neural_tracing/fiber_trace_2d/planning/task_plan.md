# Plan: preserve the complete winding visibility mask

## Discovery and representation

- Allow winding output discovery to be sparse by state and winding.
- Derive the inclusive integer winding range from the minimum and maximum
  discovered labels.
- Keep physical artifact discovery separate from the logical layer grid. Every
  present artifact retains strict header and polyline validation; missing files
  are never synthesized or passed to the reader.
- Materialize an empty Napari Shapes layer in the managed layer mapping for
  every H/V/error/tie slot in that range, while attaching geometry only from
  artifacts that exist. Count only real paths in `fiber_count` and retain the
  current all-geometry-empty launch error.

## Navigation

- Rotate the live visibility bits over the complete contiguous layer grid.
- Preserve every state bit independently, including bits whose current or
  destination layer has no geometry.
- Compute the initial selection from nonempty loaded H/V geometry before adding
  placeholders. Retain the current `All` fallback if no H/V geometry exists.
- Leave reference and unmanaged layers unchanged.

## Reference benchmark output

- Carry each usable benchmark observation's original selected reference source
  ID through gauge calibration; pieces and clipped runs from one JSON aggregate
  into the same source row, while names remain CLI-only.
- Accumulate right/wrong/total counts per reference fiber using the selected
  gauge offset from one global calibration, including when one source touches
  multiple gauges. Do not recalibrate individual sources. Exclude
  candidate-free Mixed/Defect, invalid, and unsigned-perpendicular observations.
- Print one compact row per selected reference source in source-ID order after
  gauge calibration and before aggregate constraint accuracy. Include
  only the source's virtual winding as its row identifier. Include
  right/wrong/right-fraction triplets for perpendicular, parallel-same,
  parallel-other, and sum; zero-observation fractions are `NA`. Require the
  per-source class and sum counts to equal their aggregate counterparts.

## Tests

- Add sparse-artifact discovery coverage.
- Add complete-grid materialization coverage for missing states, empty states,
  and missing winding labels.
- Verify arbitrary mask rotation is a bijection through missing-file,
  empty-file, and completely absent winding slots; Previous must invert Next,
  wraparound must work, and reference/unmanaged layers must remain untouched.
- Retain malformed-present-artifact rejection coverage.
- Run the focused Python viewer tests.
- Extend the focused winding benchmark test for shared source IDs, multiple
  gauges, candidate-free observations, zero-observation sources, and aggregate
  count invariants; rerun the C++ tests.

## Spec update

- Replace the complete-quartet requirement with sparse artifact discovery and
  a complete contiguous logical visibility grid.
- Remove the rule that missing destination layers discard incoming bits.
- Specify the per-reference-fiber benchmark table and active-only semantics.

## Docs updates

- Document placeholder layers and exact mask rotation across missing data.
- Document the per-reference benchmark rows.

## Changelog

- Record the winding viewer mask preservation fix.
