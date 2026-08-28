# Plan: penalize split-piece Defect boundaries

## Implementation

- Add a nonnegative `pieceBreakCost` to both Defect-capable winding-BP
  configurations, defaulting to zero so existing runs remain unchanged. Do not
  expose it on the integer-only winding solver, which has no Defect state.
- Preserve whether a prepared factor contains the existing same-trace hard
  continuity constraint. Charge `pieceBreakCost` once per such prepared edge
  when exactly one endpoint is active and the other is Defect. Multiple
  measurements on the edge must not multiply the boundary cost.
- Apply the term in both alternating and joint-grid pair potentials before the
  existing Defect-neutral early return. Scale it by `orientationTemperature`,
  like the Defect unary rather than the winding temperature. Use one shared
  activity-boundary helper so both solvers implement the same truth table, and
  include it in decoded-energy accounting and candidate ranking. It must not
  modify active-active winding or H/V
  costs, Defect-Defect pairs, measured cross-trace factors, continuous
  initialization, constraint extraction, or hard-sign behavior.
- Expose `--piece-break-cost F` on `direction-ablation`, validate that it is
  finite and nonnegative, reject it outside the existing Defect-capable BP
  mode, and pass it through both winding solver variants. Print it in the
  winding summary and store it in the consistency CSV.

## Tests

- Add focused coverage for both solvers and fixed/joint orientation paths,
  proving the active/Defect same-trace truth table, cross-trace neutrality,
  one charge per prepared continuity edge, default-zero compatibility, and
  decoded-energy accounting.
- Cover invalid negative/non-finite configuration through both config
  validators.
- Build `vc_fiber_trace_chunk` and `test_fiber_trace_winding_bp` in the current
  Release tree with 32 jobs and run the focused tests.
- Run the exact established 1024-crop `--piece-length 512` benchmark at break costs 0,
  1, and a stronger value selected from the first result. Compare total and
  source-piece Defect rates, constraint-class tables, reference accuracy,
  convergence/residual, and wall/CPU runtime.

## Spec update

- Specify the exact activity-boundary truth table, orientation-temperature
  scaling, one-cost-per-continuity-edge accounting, default-zero
  compatibility, output provenance, and solver coverage.

## Docs updates

- Document `--piece-break-cost`, its units and intended use for discouraging
  isolated Defect spans without weakening retained hard signs.

## Changelog

- Record the new configurable split-piece activity-boundary penalty.
