# Plan

1. Add one structured plot-data file containing algorithm completion date and
   revision, measurement date and revision, cohort identity, raw numerator and
   denominator, measured/assumed status, and run record for both benchmark
   families.
2. Define replay score exactly as `100 / max(failures, 1)` after complete
   evaluation. Preserve the crop benchmark's existing
   `100*problematic/retained_fulfilled` error ratio and negate it so zero is the
   ideal target and higher is better. Record direct greedy and Lasagna as
   assumed floor markers without fabricated crop metric values.
3. Add a deterministic plotting script that validates raw counts and
   provenance, orders points by algorithm completion date, renders
   percentage-versus-date stair steps, distinguishes measured and assumed
   points by both color and marker, and writes two accessible SVG artifacts.
4. Link and explain both plots in the benchmark results index, including the
   score transforms and historical-date policy.
5. Run the plot generator, validate both SVGs and their labels, and run a
   lightweight script test with the checked-in data.

## Spec Update

Document plot score definitions, assumed floor-point semantics, algorithm-date
semantics, and the requirement that future points retain provenance.

## Docs Update

Embed the generated replay and crop progress plots in the benchmark index and
document how to regenerate them.

## Changelog

Record the reproducible benchmark visualization and extensible data source.

## Validation

Run the plotting script against the checked-in data, parse both generated SVGs,
and verify the expected point count, ordering, labels, score recomputation,
numeric range, provenance, and run-record links.
