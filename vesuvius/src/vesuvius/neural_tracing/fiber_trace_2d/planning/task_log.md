# Task Log

- The current fixed-reference conflict table evaluates reference-to-BP cross
  factors, not constraints between reference fibers.
- Its factor enum collapses perpendicular 0.5/1.5+ and parallel 1/2+ classes,
  although the prepared observations retain the target distances needed to
  classify them.
- Reference-to-reference extraction and solver-prepared factor diagnostics
  already exist; the new table can evaluate those structures without another
  extraction implementation.
- A calibrated `est_w` can conflict with the separately majority-calibrated
  H/V orientation component. That makes the exact inverse land on a half
  integer after class-offset removal. This is an unavailable published layer,
  not malformed solver state, so `raw_w` must be `NA` rather than an exception.
- The current formatter always emits every reference piece-pair constraint.
  These rows are useful for targeted geometry debugging but too verbose for a
  regular benchmark run, so they will become explicitly opt-in.
- Independent review clarified that reference labels are never measurement-
  scaled; only the measured solver target is scaled before quantization.
  Ref-to-ref evaluation will consume prepared factor diagnostics, exclude
  continuity links, and share exact factor residual/sign evaluation with the
  reference-to-BP path.
- The verbose flag remains deliberately scoped to the per-reference raw
  constraint rows named by the user. Existing worst-BP-piece and group-energy
  diagnostics are left unchanged because they describe reference-to-BP solve
  behavior rather than the requested reference-to-reference listing.
- Output mapping distinguishes a valid half-step candidate on the opposite
  H/V ladder (reported as `NA`) from an arbitrary off-half-step latent value
  (still an invariant error).
- Added one shared materialized-factor evaluator for reference-to-BP and
  reference-to-reference conflicts. The compact formatter now reports the five
  magnitude bands and two sign classes independently, plus their sum.
- The reference-only path fixes the predicted canonical delta from ordered
  source IDs, excludes same-source and hard-continuity factors, and does not
  apply measurement scale to known reference labels.
- Added `--reference-constraint-details`; default output retains all compact
  summaries while omitting the long `reference fiber "..."` piece-pair blocks.
- Release focused build/test passed: `83 test case(s) passed`.
- Clang focused build/test passed: `83 test case(s) passed`.
- The approved 1024 reference diagnostic completed in 16.05 s wall time and
  322.74 s user time. It produced `NA` for unmappable output-layer estimates
  without aborting and omitted verbose reference sections by default.
- A 1%-quality smoke run completed successfully with the same nonfatal mapping
  behavior. No solver settings or inference results were changed by this task.
