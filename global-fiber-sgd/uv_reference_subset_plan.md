# Reference-selected ten-winding regional solve

## Current follow-up:10000totalpasses,reportevery1000
User clarified total10000 forbothpositions5–11 runs. Same subset,LRs,losses,
151hypotheses,sixGPUs.10080updates=10000.649146passes;logevery1008updates=
1000.064915passes. Restart fromoriginal fresh/referenceinitializations to
preserve continuousAdamstate: existingcheckpoints storeweights,notoptimizer.
This repeats first1000passes ratherthan appending10000 or resettingAdam.
Verify first1008update reference metrics exactlymatch prior1000passfinal.
Independentconfigreview,productionCUDApreflight,finitefinalstate andlease
release. Reporttwocomplete1000passcheckpointtables pluselapsedtimes in
uv_reference_5_11_10000_results.md. No code/spec/objectivechanges. Keep previous
visualizations untouched; no new rendering requested for thisfollowup.

## Current follow-up: positions5–11 for1000passes
Use the audited IDs14–20 Z30k–50k subset (2192sources,2199pieces,24771events).
Run fresh and reference/width initialized from their original starts, not warm
resume (retain complete comparable trajectories). Same sixGPUs,global24576,
stateLR0.007348469228,widthLR2,seed0,151hypotheses,all variables joint.
Only change budgets toceil(1000*24771/24576)=1008updates and reporting101updates
(~100.205passes). Finalactual1000.065passes. Report two complete checkpoint
tables inclobjective,referenceL1/within2,retention,signviolations,widthfactor,
fit/referenceq10/q90 and globaloffset; timing and peakmemory.
No code/spec/loss changes. Independent config review; reuse approved suite
runner with new output roots. Record in uv_reference_5_11_1000_results.md,
currenttasklog/status,README/changelog. Validate CUDApreflights,finite final
states,exactsubset identities,lease release,first100pass comparability.

User added visualization of both1000passsolves. Reuse existing CPU layout-only
renderer,1000pxhigh LZW map-onlyTIFF,equalU/V physicalscale,windingboundaries,
separateconfidencehistograms,SVG/vectorPDF. No new projection or solve. Verify
bothfiles/headers/sourcehashes and noPNGs. Recordpaths and dimensions.

## Follow-up: shifted seven-winding windows
User confirmed positions counted from leftmost reference winding. Metadata
leftmost column_begin1 belongs toID10. Test positions4–10,5–11,6–12, i.e.
IDs13–19,14–20,15–21, all inclusive. Keep Z30k–50k and the same extraction,
151hypotheses,6GPU globalbatch24576,stateLR0.0073484692,widthLR2,seed0.
Run fresh/reference-width-initialized for each subset under one managed suite.
Update budgets ceil(100*E/24576), reporting approximately every10passes.
Report all progression tables, input sizes, actual pass/update budgets, timings,
and losses versus previous10–19 subset. These are overlapping not independent
datasets; different event counts mean different optimizer update counts.
Do not interpret lower loss alone as locating kollesis or convergence.
No code/loss changes; independently review generated configurations and audit
extracted tensors/projections using the existing tests/audit.

## Task
Extract fibers matched to reference tifxyz windings 10–19 (the first ten
recorded IDs), clip to base Z=[30000,50000], and run 100 sampled dataset-pass
equivalents from fresh and reference/width initialization. Report every ten
passes: objective, offset-corrected reference L1/within-two, retained length,
width factor, range, and measured solve time.

## Implementation
- Reuse cached geometric reference matching, immutable earlier sign/HV/direction
  stages, and existing tensor losses/solver. No new inference or sign solve.
  Select by the existing whole-source median reference label, then clip exact
  AFV segments (not merely the sparse projection samples). Diagnostics use only
  retained geometry but the same original-source reference winding label.
- Reusable subset extraction through the approved crop-benchmark runner.
  Clip actual polylines at Z boundaries; keep disconnected reentries separate.
  Preserve source IDs/original arc coordinates, reanchor piece geometry, remap
  endpoints and continuations, remove constraints outside the selected pieces.
  Never reconnect across removed geometry. Save subset projection with matching
  identities/provenance and an extraction manifest, plus inspectable geometry.
- Keep 151 hypotheses and the full-data geometric initialization/prior for
  selected pieces (do not stretch ten windings across all 151 candidates).
  Reference mode additionally initializes assignments and widths as before.
- Keep low LR configuration, fused pair/coordinate evaluation, all parameters
  joint/trainable from update one. Set update count from subset event count,
  not original corpus size. Use a sensible smaller batch if subset warrants,
  applying the documented square-root LR scaling, identically in both runs.
- Report actual sampled pass equivalents, input counts/ranges, boundary loss
  of links, initialization differences, and cold setup vs solver timing.

Independent plan review approved. Initial reconnaissance suggests roughly2651
sources. User explicitly requests multi-GPU for speed: use six GPUs with4096
events/GPU (global24576), stateLR0.0073484692 (sqrt scaling from
196608/.02078460969), widthLR2 unchanged. This is a regional test, not a
controlled full-volume-vs-subset batch comparison.

## Spec update
Add opt-in reference-selected clipped induced subgraphs; full-volume behavior
and losses unchanged. Reference labels select the benchmark and initialize
only the reference run, never become training targets.

## Validation
Independent plan/code review; synthetic clipping/reentry/constraint/continuation
and projection identity tests; initialization subset equivalence; existing UV
tests. Production assertions on all geometry Z, winding selection, ownership,
finite arrays, reference provenance, no out-of-subset endpoints.

## Docs updates
Record commands/configs/results in uv_reference_subset_results.md; update
task/status/current task_log, README and changelog. Preserve previous results.
