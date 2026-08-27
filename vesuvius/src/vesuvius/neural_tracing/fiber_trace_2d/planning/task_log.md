# Task Log: Fixed pre-pass orientation for winding BP

## Discovery

- Joint-grid currently skips orientation BP and solves `(H|Mixed|V,k)` in one
  model. Alternating runs orientation BP first but uses those marginals only as
  soft unaries, so winding inference can still change the class.
- Both winding implementations already use the same orientation/winding factor
  semantics. Fixed-prepass mode will use a shared winding-only piece-state
  layout and combine each integer candidate with stored fixed-class metadata;
  it will not retain impossible orientation states or pay their message cost.
- Existing component gauges implicitly use class A at winding zero. Fixed mode
  must retain the pre-pass class at the gauge while fixing only the integer
  winding origin.

## Deviations

- None.

## Plan review

- The independent review required winding-only state accounting across every
  alternating calibration/energy stage, not only message passing; fixed-class
  and orientation-mode persistence alongside retained soft pre-pass marginals;
  explicit factor/gauge cardinality tests; and a repeated performance protocol.
- It also clarified that Tie is not a fixed-class API value. Exact pre-pass MAP
  ties are converted to Mixed before entering either winding solver.

## Implementation

- Added `--winding-fixed-orientation` for both `joint-grid` and `alternating`.
  It runs the existing H/V/Mixed BP once, maps its posterior to a fixed class,
  and then represents every piece state only by an integer winding candidate.
- The fixed class is immutable metadata used while evaluating winding factors;
  it is not a separately optimized or one-hot-expanded solver state. Mixed
  retains its four-substitution factor calculation without introducing four
  latent states.
- Applied the reduced state layout to message passing, support expansion,
  calibration updates, component gauges, decoded-energy ranking, and final
  reports in both winding implementations.
- Preserved the pre-pass soft class probabilities in diagnostics and added the
  fixed class plus orientation-mode provenance to the consistency CSV.

## Validation

- Built `vc_fiber_trace_chunk`, `test_fiber_trace_winding_bp`, and
  `test_fiberlet_crop_trace` with `cmake --build volume-cartographer/build ...
  -j 32`.
- Registered focused CTest passed: 2/2 test binaries, including 18 winding-BP
  and 74 crop-trace cases.
- Joint-grid fixed-prepass and alternating fixed-prepass both completed on
  `data/workdir3/crop_traces.zarr` with the Paris 4 Lasagna normal manifest,
  1,361 pieces, and 35,673 constraints. Both retained the pre-pass assignment
  of 710 H, 630 V, and 21 Mixed pieces.
- CLI help exposes the option and using it without
  `--bp-only --bp-inference sum-product-mixed` is rejected.

## Performance

Release-build timings used the established 1,024-crop command with fixed phase
0.5, fixed scale 1.0, 32 workers, and three runs per mode. The fixed-prepass
orientation BP took 0.630/0.631/0.636 seconds.

| Mode | Candidate states | Solver seconds min/median/max | Command wall seconds min/median/max |
| --- | ---: | ---: | ---: |
| Joint H/V/Mixed+winding | 942,700 | 5.849 / 5.929 / 6.004 | 9.55 / 9.93 / 11.15 |
| Fixed-prepass winding-only | 362,663 | 1.936 / 2.050 / 2.054 | 6.45 / 6.54 / 6.56 |

The winding phase therefore uses 61.5% fewer candidate states and has a 2.9x
median solver-time speedup. Numerics intentionally differ from joint inference
because the requested mode freezes the independent H/V/Mixed result; the
default mode remains unchanged.
