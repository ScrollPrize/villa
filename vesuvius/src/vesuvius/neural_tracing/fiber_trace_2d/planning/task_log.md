# Task Log: Mixed-fiber direction-label ablation

- Mixed fibers are defects. Confidence controls only their deterministic
  admission order; their expected optimized state is Broken and they must
  never receive a tentative H/V reference. Retain the originally
  above-threshold direction groups as a separate trusted cohort.
- Gauge alignment must not optimize against the newly admitted uncertain
  labels, because that could hide degradation of the original diagnostic.
  Components are therefore aligned from trusted active pieces only; an
  unanchored component keeps the identity mapping.
- Checkpoints are cumulative but computationally independent. Canonical
  extraction and solve are rerun after every admission checkpoint so topology,
  degree-dependent Broken cost, and pruning can genuinely change.
- Independent review identified that confidence ranking must select membership
  only. Every checkpoint will retain source traces in original stored order;
  otherwise the ablation would also perturb piece IDs, spatial tie breaks,
  component canonicalization, and final all-admitted equivalence.
- Review also fixed the comparison API and accounting: pass complete reference
  states plus a trusted mask, align exact post-pruning active components
  from trusted pieces only, report explicit trusted/admitted/combined unions,
  and expose solver status/gap at every checkpoint. Intermediate execution must
  remain artifact-free; only initial and final families are written.
- Built the command and focused tests with `-j32`; all 43 test cases pass.
- The unpruned centered-384 run uses all links surviving the canonical
  exclusive 1.5-winding cutoff. Checkpoints 0 through 57 completed optimally
  with zero trusted errors, zero admitted errors, and zero Broken pieces.
  Checkpoint 57 contains 152 represented fibers, 154 pieces, and 2,010 retained
  constraints at latest admitted confidence 0.628699. Checkpoint 58 is still
  solving and demonstrates a major branch-and-bound runtime discontinuity.
- The first ablation run incorrectly treated mixed fibers as tentative H/V
  references. It was stopped at checkpoint 67 on user correction. All results
  after checkpoint zero are invalid and must not be used; checkpoint zero is
  unaffected. The implementation is being corrected so admitted mixed pieces
  are successful only when the MILP labels them Broken.
- Per-fiber checkpoints were unnecessarily expensive and obscured the trend.
  The corrected command defaults to five admitted mixed fibers per checkpoint,
  always includes the final remainder, and also solves/thresholds the matching
  LP relaxation for separate H/V-versus-defect comparison.
- Added `--ablation-limit` so every broken-cost trial uses the identical ranked
  Mixed prefix. On the centered 384 crop, costs from 0.25 through 0.5 detected
  no Mixed defects. Cost 0.1 detected 2/10 but incorrectly broke 9/95 trusted
  fibers. The narrow selected value 0.2035 keeps all 95 trusted fibers active
  and correct while detecting 1/10 and 1/15 admitted Mixed defects; 0.205 loses
  the 15-fiber detection. MILP and thresholded LP labels matched throughout
  this sweep.
- Completed the full 84-Mixed-fiber centered-384 ablation at broken cost
  0.2035. The final MILP labels 35/84 Mixed source fibers Broken and has 1/95
  trusted-fiber error; thresholded LP labels 12/84 Mixed fibers Broken and has
  8/95 trusted-fiber errors. Final piece labels are MILP H=74, V=72, Broken=44
  versus thresholded LP H=109, V=65, Broken=16. The 18 MILP checkpoints took
  22,152.536 seconds in total (6h09m), dominated by 7,245.168 seconds at 80
  admitted and 12,285.571 seconds at 84 admitted. All LP checkpoints together
  took 97.286 seconds. This demonstrates both the LP quality failure and the
  impractical branch-and-bound growth of the current full MILP ablation.
- Final verification rebuilt `vc_fiber_trace_chunk` and
  `test_fiberlet_crop_trace` with `-j32`; all 44 focused test cases pass and
  `git diff --check` is clean.
