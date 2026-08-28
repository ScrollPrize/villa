# Task log: report central versus non-central BP states

- The previous comparison was derived from the final consistency output using
  `source_piece == 1` as the central cohort and every other source-piece index
  as non-central.
- Reference diagnostics are intentionally deferred until the end of the CLI
  run; the new summary must be deferred with the benchmark to appear directly
  before it.
- Independent review found final `pieceIndex == 1` is unstable because component
  subsetting renumbers and may split traces. Preserve the pre-filter cohort bit
  explicitly. This intentionally matches the earlier ad-hoc comparison; it is
  named as source-piece 1 in documentation rather than claimed to be a general
  geometric-center definition.
- Final-state counting must use the post-projection winding result. Invalid
  winding counts as Defect even if an orientation enum remains H or V.
- Added a shared final-state cohort aggregator with exact total invariants and
  focused coverage for selected/other cohorts, Mixed states, invalid winding,
  an empty selected cohort, and malformed inputs.
- Each BP execution now captures its original source-piece-1 mask before main
  component filtering and remaps the mask using retained old piece indices.
  With references, its table is concatenated directly before that execution's
  benchmark; without references it is emitted in deferred BP diagnostics.
- Release build used `-j 32`. The 1024 default-cost run emitted:
  central 451 pieces, 24 H, 20 V, 44 active, 407 Defect (0.902); non-central
  909 pieces, 321 H, 267 V, 588 active, 321 Defect (0.353); total 1,360 pieces,
  345 H, 287 V, 632 active, 728 Defect (0.535). The following reference
  benchmark remained 1,082 right and 126 wrong (89.570%).
- The final display renders the three Defect rates explicitly as `90.24%`,
  `35.31%`, and `53.53%` rather than decimal fractions.
