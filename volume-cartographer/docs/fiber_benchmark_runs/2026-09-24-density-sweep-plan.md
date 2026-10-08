# Density sweep plan

Reuse audited inputs and native multipliers. Search
`m(d)=max(floor, scale/(scale+d))` separately for mean/median/max.

1. Coarse Cartesian grid: scale {0.05,0.5,5}, floor {0.01,0.1,0.5}.
2. Refine each best measured point with scale divided/multiplied by sqrt(10)
   and floor geometric midpoints toward adjacent tested values. Extend floor
   boundaries toward 0.001 or 0.8. Evaluate axial neighbors, then repeat around
   the new winner with scale factor 10**0.25 and new floor midpoints. This is a
   finite direct search, not proof of optimality. Report boundary winners.
3. Baseline multiplier1 remains a competing no-change candidate. Select using
   existing removal score, not independent accuracy. Parameters tuned on this
   crop are not held-out results. No default promotion automatically.
4. Existing approved crop runner, 4 concurrent jobs x32 threads. First check
   32-thread control exactly reproduces previous128-thread metrics. If not,
   retain prior thread setting. Reuse completed runs only after command, inputs,
   executable/shared-library and source identity validation.
5. All traces/constraints/preprocessing/500-message budget unchanged. No new
   inference or tracing. Record per-run outcomes and full grid, winner mapping,
   score, initial independent ladder, convergence and runtime. Persist progress.

## Spec update

Optional benchmark orchestration only. Finite positive multipliers <=1.

## Docs updates

Sweep report, current task log/status and changelog. Tests for grids, refinement,
mapping and resume validation; independent plan review before implementation.
