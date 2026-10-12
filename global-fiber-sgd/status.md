# Reference-selected regional solve

## Enabled GT L1 search
- [x] Commit previous state/results:016cdf185.
- [x] Independent plan/code review, enabled-only L1, shared cached evaluator.
- [x] Configurable objective in resumable coordinate search;39tests.
- [x] Start sixGPU search; baseline exactly reproduces prior metrics and first
  changed-weight trial started successfully.
- [ ] Complete iterative grid refinement and record final incumbent.

## Corrected fresh/reference GT evaluation
- [x] Commit shared-phase correction and GT results (e2ba3415a).
- [x] Reviewed matched configs; both sixGPU solves completed (~173/174s).
- [x] Native cross extraction and frozen original GT normal sweep.
- [x] Include disabled best-nonbroken placements;30tests and independent review.
- [x] Score6735events: fresh49.6214%, reference55.0705%; exact repeat hash verified.
- [x] Persist full metrics, per-GT breakdown and event-level vote artifacts.

## Expanded GT / evaluation
- [x] Inspect seven input files and preserve previous four-fiber outputs.
- [x] Write plan and versioned runner configuration.
- [x] Independent review and intact seven-fiber solve.
- [x] Shared-phase correction resolves five coarse violations: v3 zero in both readouts.
- [x] 25 focused tests, independent review, matched v3 solve and documented comparison.
- [x] Confirm comparison artifacts and vote convention; implement/test scoring.
- [x] Report GT audit and comparison percentages with support counts.

## Four reviewed JSON fibers
- [x] Inspect inputs, plan and independent review.
- [x] AFV adapter and exact geometry roundtrip, 10 CPU tests pass.
- [x] Native extraction, orientation and features complete.
- [x] Complete normal/HV/direction and whole-fiber UV solve; report layout.
- [x] User clarification:4intact trusted fibers, rejection disabled,16/16signs satisfied.
- [x] TIFF/SVG/PDF and per-fiber/event JSON; bbox_gt_results.md;12tests pass.

## Adaptive loss-weight tuning
- [x] Previous results committed at 102d25a55.
- [x] Plan independently reviewed; controller and 13 CPU tests passing.
- [x] Implementation review and managed search launched (dba4ffa6d).
- [x] User stopped search after96 completed trials /16 accepted improvements.
- [x] GPU lease released; interrupted trial excluded, all completed trials kept.
- [x] Canonical baseline config and full compact trial records documented.
- [x] Final verification, independent baseline review and commit preparation.
- No-improvement neighborhood was NOT completed; do not claim a local optimum.

## Fresh LR follow-up
- [x] Plan and stateLR10x config.
- [x] Independentreview and stateLR10x trial; worseL1andloss.
- [x] SelectcurrentbaseLR; reviewedwidthLR20/.2 configs.
- [x] CompletewidthLR20/.2 trials; finite states and released leases.
- [x] Fulltables,validatedresults in uv_bucket_lr_results.md.

## Position weight0.5 follow-up
- [x] Create matched fresh/ref configs and suite; plan.
- [x] Independent review, tests and sixGPU fresh/ref10k runs complete.
- [x] Full checkpoint comparison saved in uv_bucket_half_results.md.

## Bucket topology / weaker position
- [x] Plan, independent review and fixed integer seam correction.
- [x] Implementation;56 focused and33 additional regression tests passed.
- [x] Launch approved sixGPU fresh/ref10k suite (manager7b42a4cb).
- [x] Both solves complete; compiled gradient checks and finite saved states.
- [x] Full tables in uv_bucket_split_results.md; manager lease released.

## Recommended LR
- [x] Record and independently review10x state LR default; explicit configs preserved.
- [x] Validate14 class-balance/default tests. Width LR remains2.
- [ ] Proposed winding-index/position loss separation: not implemented in default commit.

## Balanced classes and fixed sign charge
- [x] Plan exact graph-wide10:1 aggregate class weighting.
- [x] Independent plan/code review and implementation;47 focused tests passed.
- [x] 80 regression tests and charged compiled six-GPU validation.
- [x] Fresh10k complete; reference stopped by user after2k diagnostic. Logged.
- [x] Reviewed10x/0.1xstateLR four-trial configs and launched approved suite.
- [x] All four LR trials complete; finite states, CUDA checks and released lease.
- [x] Full tables including widths/ranges in uv_class_balance_lr_results.md.

## Positions5–11,10000totalpasses
- [x] Derive10080updates and1008updatediagnostics,unchangedsettings.
- [x] Independent config review approved; launched approved suite.
- [x] Both six-GPU runs complete; exact 1k metric reproduction and finite states.
- [x] Full 1k checkpoint tables saved in uv_reference_5_11_10000_results.md; lease released.

## Positions5–11 longer solve
- [x] Plan and derive1008update budget,101update diagnostics.
- [x] Independent config review.
- [x] Complete fresh/reference1000passes on6GPUs.
- [x] Validate/report two complete100pass metric tables and timings.
- [x] Render/verify both1000passlayout maps,TIFF+SVG/PDF+histograms.

## Shifted window follow-up
- [x] Confirm reference position1=ID10 from metadata.
- [x] Extract/audit13–19,14–20,15–21.
- [x] Independently review run configurations.
- [x] Run six100pass-equivalent solves on6GPUs.
- [x] Persist/report all checkpoint metrics and timings.

- [x] Inspect existing projection/solver and write plan.
- [x] Independent plan review.
- [x] Implement reproducible clipped subset and tests.
- [x] Independent implementation review and validation.
- [x] Extract subset and run fresh/reference100pass-equivalent solves.
- [x] Report metrics, timing, commands and limitations; persist results.
