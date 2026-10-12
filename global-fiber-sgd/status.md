# Current: diagnostics and100-pass solves

- [x] Independent plan review.
- [x] Exact median fastpath and shared geometry; unit tests.
- [x] Real-state metric equality/profile/performance;4workers selected.
- [x] Reference width initialization and normalized width metrics;99tests pass.
- [ ] Low-LR100/1000 fresh and reference runs (remaininghighLRcancelled byuser).
- [ ] Fresh and reference-initialized100-pass-equivalent solves.
- [ ] Reports and comparison visualizations.

# Previous: larger batches and distributed joint solve

- [x] Independent plan review.
- [x] Larger-batch measurements and quality comparison.
- [x] Shared multi-GPU solver, correctness tests, exclusive scaling/quality runs.
- [x] Documentation/results and handoff.

# Previous: coordinate-gradient reformulation

- [x] Independent plan/code review, implementation, gradient validation.
- [x] Exclusive speed/profile comparison and matched standard/precise quality trials.
- [x] Results, derivation and limitations documented; standard division selected.

# Previous:16k LR scaling

- [x] Independent configuration review; no solver code change.
- [x] Three exclusiveGPU trials .03/.06/.12 completed exit0.
- [x] Progress and final comparison logged:uv_lr16k_results.md.

# Previous: profile and local-window trial

- [x] Independent plan review and matrix review.
- [x] Exclusive baseline quadratic pair and production phase profile.
- [x] Exact prefix sign and local windows implementation;90-test suite plus
  child-command regression pass. CUDA compiled/eager tests all4radii passed.
- [x] Matched radius trials, results and documentation; defaults unchanged.
- [x] Exclusive batch timing/memory matrix B4096/8192/16384/32768, full/N2.
- [x] Larger-batch quality trials and detailed complete-update profiler.
- [x] Both exclusive suites finished exit0;91tests passed. Results and complete
  progress in uv_local_window_results.md and uv_local_window_progress.md.

# Previous: corrected quadratic fused run

- [x] Loss implementation committed as c468354dd (81 regression tests passed).
- [x] Independent experiment-plan review.
- [x] Managed run and per-pass diagnostics; exit0,699.28s solver elapsed.
- [x] Final comparison in uv_squared_vertical_split_results.md; no meaningful
  accuracy change versus earlier quadratic,1.9% shorter nonisolated timing.

# Previous: split V equality / Z layout prior

- [x] Independent plan review and shared implementation.
- [x] Removed manufactured target_v from every production consumer/API.
- [x] Separate weak unary prior, unchanged point/probability normalization.
- [x] Regression suite:81passed; independent code review approved.
- No solve rerun requested.

# Previous: remove pairwise wrapped-phase

- [x] Independent plan review; narrow shared-helper edit.
- [x] Preserved distance phase correction, individual priors and vertical term.
- [x] Focused regression tests:54passed; docs/log updated.
- No solve rerun or moment acceleration in this task.

# Previous: four distance objectives

- [x] Independent plan review and CPU tests (44passed).
- [x] Fused implementations and four matched non-reference configurations.
- [x] Managed CUDA equivalence validation;49CPU tests passed.
- [x] Four parallel runs on GPUs0–3,16130random minibatch updates each; all exits0.
- [x] Full histories and final comparison: uv_distance_objectives_results.md.
- No large global recovery; Huber gives a modest retention/error improvement.
  No default changed; final sign error is higher than baseline for all variants.

# Previous: fused pair evaluation

- [x] Plan and independent implementation review.
- [x] Shared-objective compilation; unchanged 4096 minibatch updates.
- [x] Managed real-data, tail, boundary and production-update benchmarks.
- [x] 32 regression tests passed; timing/memory/rounding limitations documented.
- [x] Results and reproduction: uv_pair_fusion_results.md.
- Optional charge/expected-cost compiled GPU validation deferred explicitly.
- No full fused solve launched; compiled backend remains opt-in.

# Historical visualization status

## Single joint151 strong-order solve (current)
- [x] Config-only plan independently reviewed; no warm starts/stages.
- [x] CCW/CW launched with strong penalty and all parameters trainable from start.
- [ ] Ten event-pass equivalents completed and audited.

The multi-stage fresh151 experiment below was superseded by explicit user
instruction. Its categorical stage was never run.

## Fresh 151-winding solve (supersedes warm expansion)
- [x] Plan independently reviewed; no prior U/V state permitted.
- [x] Fresh initialization and joint count generalization; 24 focused tests pass.
- [x] CCW/CW fresh joint optimization (10,000 steps); both completed.
- [ ] Within-run sign refinement (4,000 steps).
- [ ] Ten categorical sweeps, final audit and maps.

## 151-winding expansion / contrasting maps
- [x] Independent plan and implementation review; 17 targeted tests pass.
- [x] Launched both ten-sweep continuations (0–150 plus broken), batch2048.
- [x] Both ten-sweep solves completed; warm expansion failed by over-rejection.
- [x] Persisted output/source/geometry audits passed for both directions.
- [x] Old100-sweep 1000px opaque2px golden-angle/OKLab maps rerendered.
- [x] Expanded151-sweep maps and final results; uv_151_results.md.


## TIFF-only correction
- [x] Plan reviewed, exporter fixed, eight regression tests passed.
- [x] Seven existing crops converted and verified; PNG copies removed.

## Reference comparison
- [x] Plan and code independently reviewed.
- [x] Original-fiber median, restricted projection, continuous common colors.
- [x]15tests including end-to-end maps, cache reuse and gap guard.
- [x] Final CCW/CW100-sweep maps and statistics (reference_comparison_results.md).

## Reference surface
- [x] Plan independently reviewed; scope download/inspect/propose only.
- [x] All6objects downloaded, hashes/sizes/headers/coordinates inspected.
- [x] Missing winding-ID sidecar checked against prefix/parent and publisher code.
- [x] Comparison/visualization proposal documented in reference_surface.md.

## 100-sweep density task
- [x] Plan independently reviewed, common-area metrics and conditional placement.
- [x] CPU implementation using existing solver/deposition helpers.
- [x] 21tests and independent implementation review.
- [x] Both conventions completed, profiles/heatmaps inspected; fiber_density100_results.md.

## H/V rejection task
- [x] Plan reviewed; final bins10kprofiles/3kheatmaps.
- [x] Separate hard length/count rejection and soft probability diagnostics.
- [x] 28tests, fullCCW/CW plots and inspection.
- [x] Independent review; results in hv_rejection_results.md.

## Kollesis candidate maps
- [x] Four non-endpoint metrics planned and independently reviewed.
- [x] CPU implementation and initial synthetic tests.
- [x] Full saved20 CCW/CW maps and profiles generated/inspected.
- [x] Independent code review and results documented in kollesis_results.md.

## H-anchored surface task complete

- [x] Both20-sweep fiber plots updated before surface implementation.
- [x] Independent plan/code review; confirmed100vx step/1000vx search.
- [x] Native normal curve export; all14.7Mendpoint normals checked.
- [x] Both full tifxyz surfaces written; no rendered surface images.
- [x] Stored-band/mask/scale/provenance audit and31tests passed.
- [x] Fragmentation/jumps/twisted quads documented in surface_results.md.

Next continuation:80additional sweeps (total100) launched on GPUs0/1 with shared
12GiB grants; no solver parameter changes. Roots ccw_sweep100/cw_sweep100.

Current continuation: both20-sweep solutions complete; final violations28CCW/22CW.
Retained arc64.8926%/64.7460%. Outputs under data/uv_categorical_v1/ccw_sweep20
and cw_sweep20. Additional10sweeps took8m16s/8m44s respectively.

- [x] Prior state committed.
- [x] Task/plan and independent plan review.
- [x] Export implementation.
- [x] Tests and saved-state export.
- [x] Visual inspection and independent review.
- [x] Results, docs and resource release.
- [x] Fitted winding boundaries added to SVG/PNG layouts; CPU rerender validated.
- [x] Superseded raster previews with1000px-high LZW TIFF, equal U/V scale,
  no SVG tooltips. Five tests passed; continuing solves unaffected.
# Current: joint full passes and reference metrics

- [x] Matched physical-Z comparison complete: reference_physical_z_results.md.

- [x] Broken-logit warmup removed; default fixed expected rejection penalty.
- [x] No new solve launched after this correction.

- [x] Follow-up CCW/reference visualization complete: uv_fullpass151_vis_results.md.
- [x] Initial/final q10/q90 recovered; intermediate states not saved (disclosed).

- [x] Plan and independent review (uv_full_pass_plan.md).
- [x] Exhaustive event/node passes; all variables trainable.
- [x] Reference metrics in pass histories and saved comparison reports.
- [x] Focused regression suite: 31 passed; independent implementation review.
- [x] Both current10-pass joint runs complete; strong follow-up cancelled.
- [x] Per-pass history and final reference quantiles: uv_fullpass151_results.md.
