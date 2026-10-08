# Continuous winding measurements, without processing filters

User asks to remove winding variation/category/cutoff rejection from extraction,
retain continuous measurements and use median parallel estimates. Explain probe
connector geometry; do not remove its distance guard without a request.

- AFV extraction defaults: certifyWinding=false, enforceMaximumWindingDistance=false,
  closestParallelEstimator=false using existing shared code. Expose/record flags;
  pin all historical AFV configs to true for reproducibility. Keep BP defaults.
- When certification is explicitly false, local adaptive refinement must not
  round winding categories either; refine geometry validity/type transitions only.
- Preserve central continuous winding and continuous median fields separately,
  with finite-value validation. Do not round stored values. Median uses existing
  finite central/probe samples. No signed alignment or tracing.
- Native tests: parallel and perpendicular events with variable/high winding
  retained, median correct, legacy filtering preserved, huge finite values do
  not enter llround in unfiltered extraction.
- Approved build/tests and legacy pilot; new full moving16 extraction + DB under
  raw_winding_v1, preserving original artifact. Validate ledger and DB, report
  remaining percentages and raw/median semantics.

## Docs/spec

Update extraction overview, database semantics, task log/status/changelog and
results. Explicitly separate extraction from downstream processing; remaining
geometry/support filtering is unchanged. Existing invalid measurement guards stay.
