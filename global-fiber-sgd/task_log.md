# Diagnostic optimization and two100-pass-equivalent joint solves

- User authorizes diagnostic optimization then fresh/reference-initialized100
  iterations and visualizations. Plan:uv_diagnostics_100_plan.md, independently
  approved. Currentmode sampledminibatch,32k/GPU,6GPUs,3360updates~100passes.
- Follow-up: both LR levels and both initializations for100 and1000 sampled passes.
  Reference widths initialize from the reference map; radial width prior unchanged.
  Mean fitted/reference width ratio uses equal supported intervals after the same
  global winding-offset correction. Added relative absolute error too. New suite
  logs every336updates (~10passes), not34, per requested10pass reporting. Earlier
  worker benchmarks retain per-pass cadence.27targeted tests passed; plan reviewed.
  Width LR remains2; state LR .207846/.0207846. Old controls preserved.
- Previousdistributedwork archived uv_distributed_task_log.md. Normalreporting
  10pass6GPUwall152.743s:132.999sdiagnostics,18.020supdateCUDAevents. Olderhistory
  endpoint144.831s versus414.581oldbaseline;fullwall includes finalreference+save.
- Implemented exact groupedconstantmedian fastpath, canonicalfallback mixedgroups,
  staticgeometrycache and shared hard_geometry.24tests pass. Real initial/final
  per-source exactequality and3repeattimings in progress viaapprovedrunner,
  run_uv_diagnostic_benchmark.json. No optimization or metricdefinition changes.
- Realdata benchmark passed all489197source medians exact oninitial/final states,
  entiremetricdicts exact. Reference~11.5s vsoptimized4.8–5.1s fullauditCPU.
- Added bounded ordered CPUasync snapshots, noCUDAinworkers, prewarmedimmutable
  cache,mainthreadpublication,finaldrain/errorcleanup. Independentreviewapproved.
- SixGPU10pass optimizedasync2 run34.847s vs152.743s; all12checkpointdicts and
  allfinalsolutionarrays exactlyequal.97tests pass (2CUDAexcluded;actualpreflight
  includedinrun). No metricdrop; timestamp scopes explicitlytagged.
- Fresh/reference100pass suite launched using2workers before user's follow-up
  requestingworker-countoptimization. Preserve these as2workercontrols ifalready
  completed; benchmarkcounts then rerunfinal100passeswithwinner ifdifferent.
  No optimizercheckpoint available midrun, so don'tdiscard completedcontrolwork.
