# Faster diagnostics and two100-pass-equivalent solves

User requests optimize diagnostics autonomously, then fresh and reference-started
100iteration solves and report/visualize both. Preserve all objective terms,
joint optimization from update1, 151windings, CCW,32k/GPU across GPUs0–5.
Use sampled microbatch updates as current selected mode:3360updates is99.9989
dataset-equivalent passes. Diagnostics every~one pass, final fullconstraintaudit.
Both runs use same LR/globalbatch, only initialization differs.

## Implementation

1. Finish current10pass baseline timings, archive logs/results.
2. Optimize exact CPUdiagnostics first: reuse hard_geometry betweenaudit/reference;
   cache static source grouping and sample arc offsets; vectorized constant-value
   grouped median fastpath with existing weighted_median fallback for mixedgroups.
   No changed median ties,weights,retention or full/samplededge populations.
   Measure fallbackfraction; ifstilllarge, use batchwise compiled/vectorized
   segmented weightedmedian preservingtie rules. Avoid newdependency installs.
3. Add approved-runner diagnosticbenchmark mode using savedactual initial/final
   states, old/reference and optimized paths; compare allmetrics and per-source
   medians. Record repeatedwall timingsmean/p50/p95,profile. Tests for multipiece,
   empties/invalidweights,ties and sharedgeometry. No optimizationchanges.
   After exact CPU optimization, overlap diagnostics using two CPU workers with
   bounded pending immutable snapshots (two). Allrecords published in steporder;
   no metrics dropped. Mainthread captures losses/snapshot timestamps; workers
   never touch CUDA/optimizer. Finalresults drained before save/return. Reject
   asynchronous mode with adaptive/earlystop decisions which need currentaudit.
   Report aggregate CPUdiagnostic work separately from walltime (work overlaps).
4. Run same6GPU10pass timingcontrol to verify actual reduceddiagnostic overhead.
   Follow-up:user asks empirically optimizeworkercount andcommitintermediates.
   Benchmark1/2/4/8/16/128workers on same336update,12diagnostic run; measurewall,
   waits,peakCPUconcurrency and exactmetrics. Refine/repeatfastest counts. Existing
   two-worker100runs arecontrols; final100runs usechosenworker count.
   Then fresh/reference-initialized100pass solves, oneexclusivelease each (or
   sequential suite); explicitgenerator, no warmstart/frozen parameters/stages.
5. Reuse referencecomparison and layoutvisualization with cachedprojection;
   TIFF LZW1000pxheight,2pxlines,equalUVscale,fullreferenceextent, bothcolor
   transfers sidebyside in eachoutputdirectory; SVG/vector output as supported.
   Reportpassprogress,offsetcorrectedL1,within2%,retention,sign%,windingranges.

## Spec/docs/validation

Metrics are unchanged; optimized backend can be compared to reference for audit.
Independent plan/code review, unit tests + real-data equality and timings.
Update README/status/changelog/tasklog and durable results with exactcommands.
Do not hide unsupporteddistributedfullpass mode: currentrunissampled as selected.
