# Fiber surface status

## Active matched CT/fiber rendering

- [x] Plan, independent review and registration test.
- [x] Reuse shared bilinear geometry and native chunk-aware CT sampling.
- [x] Complete CT, RGBA fiber and combined TIFFs; inspect actual output.
- [x] Document output paths, coverage, settings and timing.

Result: render_deformed_central_v2,14701x6051pixels,790pieces/788source H
fibers,100%coverage,56.61seconds. Experimental folded surface unchanged.

## Previous patch repair

- [x] Plan, specification and independent review.
- [x] Sparse reconstruction, acceptance checks, saved-UV continuation and guard.
- [x] Initial synthetic tests (7 passed).
- [x] Matched original/deformed repair and continuation experiments.
- [x] Larger-first, seam-preserving/damped, stronger-guard follow-ups.
- [x] Final regression suite (59passed), results and documentation.

See repair_results.md: immediate best183/184folds, but continuation can undo
some repair. No automatic default change, no claim of completed untangling.

## Current bending/smoothness sweep

- [x] Plan and independent review.
- [x] Launch bend15/50 on both inputs with shared grants on GPUs0–3.
- [x] Inspect bending results and run increased smoothness if needed.
- [x] Audit supported/unsupported folds, regression tests and final report.

See smoothing_results.md:3×smoothness reduces folds to225/233, but57attachment
cells remain folded in each layout. No explicit repair or default promotion.

## Current joint UV experiment

- [x] Plan and independent review; opt-in specification updated.
- [x] Implicit/deformed input support and curved UV sampling.
- [x] Joint gradient-routed attachment/mesh optimization and initial tests.
- [x] Independent implementation review and full regression suite.
- [x] Both matched regional solves, diagnostics and results documentation.

See joint_uv_results.md. Both shared-GPU runs completed; remaining folds and
large-error outliers mean no production-quality/nonintersection claim.

## Completed fixed-UV baseline

- [x] Isolated worktree; task, specification and implementation plan.
- [x] Independent plan review (approved; safeguards logged).
- [x] Shared curve access and ruled initialization with extrapolation.
- [x] Fixed-UV loss integrated into existing Lasagna optimizer and ramp.
- [x] Regression tests and small measured real-data solve.
- [x] Results, limitations, reproducible commands and changelog.

Implementation pilot complete; not a production-quality surface claim. See
results.md: 181/16000 locally folded quads remain. GPU execution and whole-scroll
solve have not been measured. No upstream winding-solve changes or reruns.
