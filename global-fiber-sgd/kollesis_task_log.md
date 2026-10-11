# Archived: Kollesis candidate metrics

- User requested four non-endpoint metrics over U and U/V; no optimization or
  overlap fitting. Plan kollesis_plan.md independently reviewed and approved.
- Use both saved20 solutions, matching inspected geometry, not evolving100 runs.
- Added CPU-only approved-runner mode fiber_uv_kollesis. Shared solver coordinates;
  exact posterior distance discrepancy via cumulative sums, no100x100 event arrays.
- Existing parallel encounters only for crowding (not exhaustive neighbors).
  All hypotheses contribute; broken positions use conditional winding distribution.
  Event metrics are retention-weighted and consequently downweight rejected areas.
- Tests: first collection caught missing parenthesis; fixed. Three synthetic tests
  now pass, including brute-force all-pairs equivalence and length mass conservation.
- Full run started via run_classic_fiber_evaluation.py crop-benchmark --config
  global-fiber-sgd/run_kollesis20.json; no GPU reservation, no inference.
- Initial sequential event calculation measured~10min/convention. Interrupted
  through its execution session; geometry cache reused. Added8bounded worker
  batches with private histograms and ordered reduction (~2min/convention).
  Fixed numpy scalar grid-shape JSON serialization before completing full run.
- Added actual endpoint-map brute-force test, sorted event ID guard, equal-axis
  support plots and plotting smoke test. Five tests passed;23layout/anchor and
  diagnostic tests passed before final plot smoke addition.
- Added supplementary close encounter rate per1000retained fiber voxels, because
  fractions alone cannot show increased encounter density. Definitions and
  limitations are in kollesis_metrics.md.
- Completed both full maps:6,606,100events each;1,749,845crowding-eligible.
  Total original plot/calculation times206.36sCCW/182.09sCW; event119.86s/101.39s.
  Both length and posterior event mass conserved.29tests passed; final independent
  review approved. Eight-worker versus serial reduction is tested exactly.
- Inspected generated plots via temporary PNG previews (tool does not open TIFF),
  retained deliveredSVG/LZW TIFF, shortened overlapping profile ylabels. Added
  hash-validated accumulation-only replot, numeric support-qualified peak lists
  and descriptive profile correlations; no further event calculation.
- No convincing shared full-height joint bands established. Distance discrepancy
  correlates with uncertainty(.441/.394), not positively with close-pair density
  (-.163/-.165); largest raw peaks often low support. Exploratory diagnostics,
  not anatomy classifications. Full results kollesis_results.md.
