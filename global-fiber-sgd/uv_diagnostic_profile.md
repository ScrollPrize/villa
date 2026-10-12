# Joint-solver diagnostics overhead

## Workload and measurement

Paris4 AFV shared-anchor joint solve, 489,663 pieces / 489,197 source fibers,
151 winding hypotheses, 6,606,100 constraints. Six H100 GPUs, exclusive manager
lease, 32,768 sampled constraints per GPU (global196,608), fresh initialization,
336 updates = 9.999892 sampled pass-equivalents. All variables trainable.
This is not ten exhaustive dataset traversals. State LR .03sqrt(48), width LR2.

Approved entrypoint:
```
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distributed_32k_suite.json
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py crop-benchmark --config /home/hendrik/villa2/global-fiber-sgd/run_uv_distributed_32k_6gpu_diagnostics.json
```
The first run profiles initial/final diagnostics with cProfile. The second uses
normal per-pass-equivalent diagnostics and wall timers only; profiler overhead
therefore does not inflate its measurements. Timing starts within solve after
data/model setup, ends after solution serialization; excludes manager wait,
process launch, preflight and input preparation. Earlier history `seconds`
excludes the last reference comparison and serialization, so is not exactly the
same end-to-end boundary. Report both rather than silently conflating them.

## Initial profile results

Initial diagnostic:16.028s, reference comparison12.807s, audit3.039s,
copy/synchronization0.180s, logging0.002s. cProfile records14.19million calls;
489,197 separate weighted_median calls take8.678s cumulatively. Two hard_geometry
calls total2.173s: audit and reference comparison each reconstruct the same state.
Final diagnostic:15.529s, reference11.208s, audit4.071s, copy0.247s,
logging0.004s. These are profiled, not uninstrumented timings.

Only initial/final diagnostics:solve57.569s;diagnostic total35.191s including
initial full baseline audit3.634s. Update CUDA-event sum20.473s including
compilation; steady mean49.725ms,p5049.689,p9549.811 over331updates. Steady336
updates extrapolate16.707s. Peak allocated12.792GiB/GPU. Reduction pipeline
6.233ms/update includes coalescing/staging/NCCL/reconstruction, not network alone.

## Proposed fixes (not implemented)

Normal reporting run (no cProfile),12diagnostics including step0/final:
solve_wall_seconds152.743;legacyhistoryendpoint144.831. Diagnostics132.999s
(87.07%):reference87.270s,audit39.109s,copy/sync2.899s,logging0.026s,
plusinitialbaseline3.695s. UpdateCUDAeventsum18.020s;remaining1.724s setup/save/
bookkeeping. Exactfinalreference/retention/signmetrics match first6GPUrun.
Older16kbaselinehistory414.581s;optimizedsingleGPU16k213.333s and32k209.561s.
Comparablelegacyendpoint speedup2.862times versusolder16k,1.447times versus32k.
Rawthroughput hasimprovedmore, butdiagnosticserialization hidesmostofit.

User subsequently requested implementing these fixes and two100iterationruns;
see uv_diagnostics_100_plan.md. Below proposals describe profiledbaseline.

1. Replace the per-source Python median loop with a segmented compiled reduction.
   Precompute source/sample grouping and static arc offsets once. Preserve
   positive finite weights, retained-piece masking, stable ordering and the
   lower weighted-median convention at exactly half cumulative mass. Avoid
   full-scene resorting if per-fiber segments can be processed independently.
   Validate every per-source result against current code, including ties,
   zero weights, multiple pieces and unmatched references.
2. Compute hard_geometry once per snapshot and share it between audit and
   reference metrics. Reuse the initial full baseline audit for common step0
   geometry, while retaining the distinction between full and sampled edge stats.
3. Compute dense state diagnostics on GPU (softmax, offsets, bucket separations,
   reductions); copy compact per-fiber/readout arrays rather than the entire
   ~0.9GB state. Exact quantiles need explicit validation; do not silently sample
   them. Final full audit remains available.
4. After reducing CPU work, optionally run diagnostics asynchronously on immutable
   snapshots with a bounded queue, tagged by optimizer step. Do not mutate a
   shared state or turn per-pass metrics into stale unlabeled numbers. Final
   diagnostics must complete before declaring the run complete.

The first two are the simplest semantics-preserving fixes. Logging/file output
is negligible; reducing print volume or diagnostics frequency is not the main
answer. Existing reference metric uses cached geometric projection: it is NOT
reprojecting fibers onto the reference surface at every diagnostic.

## Caveats

Diagnostic timers exclude profile-file dump, surrounding distributed collectives
and initial projection-cache loading. copy_sync includes pending device work.
Final six-GPU sampled run retained65.0773% length,3.4741% sign violations,
offset-corrected L1=11.5217,within2=13.0682%. Compared to smaller global batches,
optimizer update count and LR differ; faster processing does not guarantee equal
solution quality. No losses, diagnostic populations or precision were changed.
