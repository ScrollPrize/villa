# Reference-selected regional solve

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
