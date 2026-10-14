# Fiber surface status

## Future half-resolution rendering

- [x] Defaults/presets8baseUV per pixel,4px lines; existing outputs preserved.
- [x] Independent review and5renderer tests,including half-resolution registration.

## Tangential geometry fitting

- [x] Independent plan/code review and32regression tests.
- [x] Optional geometry tangent term, detached UV/normals; preserve UV routing.
- [x] Checkpoint persistence and common per-family positional diagnostics.
- [x] Matched0.1/0.3/1 solves; large alignment gains,more folded quads.
- [x] Render0.1 and1 for inspection; both100%coverage,~33s,paths logged.

## Stronger UV-offset regularization

- [x] Plan and independent review; reuse existing losses, preserve locks.
- [x] Matched smooth10/100,anchor0.1 solves and12regression tests.
- [x] Audit identical inputs,local offsets,slope variation,fit,folds and crossings.
- [x] Render smooth100 candidate;100%CT coverage,30.98s,output and limitations logged.

## Connected-component filtering

- [x] Confirm connected components, not cliques; independent plan/code review.
- [x] Minimum3 distinct sources, mixed H/V; whole graph components preserved.
- [x] Filter before initialization, metadata selection and legacy replay option.
- [x] 28 regression tests and real800-step managed solve completed.
- [x] Log counts, error metrics, exact crossing locks and remaining215folds.
- [x] Filtered CT/HV render: matched UV band,100%coverage,68.28s;
  lasagna/data/render_connected_hv_min3_deformed_v1.

Current filtered surface: lasagna/data/connected_hv_min3_deformed_v1/surface.tifxyz.

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
# Connected H/V solve and visualization complete

- [x] 800-step nonexclusive GPU solve; full graph preserved.
- [x] Central inspection tifxyz, CT JPG, H/V TIFF overlay and combined view.
- [x] Report final metrics, exact crossing equality and increased fold count.

## Previous implementation

- [x] Commit prior state: 9fc1bffba.
- [x] Plan and independent review.
- [x] Signed V offsets, graph component translations, exact crossing locks.
- [x] Metadata persistence and restoration; preserve H-only mode.
- [x] Synthetic regressions and full retained-input three-step smoke.
- [x] Documentation and findings recorded in task_log.md.
- The subsequent full solve is now recorded in task_log.md; folds remain.
- Legacy sample-level crop is deliberately rejected for connected mode.
