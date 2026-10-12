# Diagnostic optimization and two100-pass-equivalent joint solves

- BothlowLR1000runscompleted: fresh1718.93s,loss1.468878,L1 10.585124,
  within2 14.26355%,widthfactor .882224,retained72.52136%; reference1716.47s,
  loss .670348,L1 2.216075,within2 70.36694%,widthfactor1.092490,
  retained92.48525%. All5savedstates finite,widths>100. Reference metadata SHA256
  e586740ccd35d9e4ad532f100164a62f785355c42710ed17b1acc158447e4c75.
  All10passrecords persisted; offset changes explicitlyincluded.1000layoutTIFFs
  verified1000px,LZW,2vectorPDFs,noPNG. Referencecrosscolor rendercompletedexit0;
  all6maincomparisonTIFFs verified1000px,LZW,noPNG. ManagedGPUlease finished,
  noCPU/GPUheld. Commands are the approved runner with100/1000layout/reference
  configfiles. No further solves pending. HighLR1000/referencehigh experiments
  cancelled peruser; completedfreshhigh100kept ascomparison. Unrelated untracked
  .pytest_tmp_private_hot/left untouched.

- LowLR100complete: fresh loss1.91248,referenceL1 11.59110,within2 13.25945%,
  widthfactor .866755,retained68.5867%. Referencewidth-start loss .873643,
  L1 1.060589,within2 89.64818%,widthfactor1.025152,retained91.6016%.
  Full10pass history committed2cf3fdc48. Both100passlayout and referencecomparison
  visualization commands completed; TIFF LZW1000px,SVG/PDF, separatehistograms,
  all threecolorviews perrun. Under data/uv_joint100_vis_width_v2/{layout,reference}.

- User stopped high-LR experiments after observing loss jumps. Freshhigh100 had
  justfinished (loss7.3176 overlast10passes,L1 10.9919,within2 13.2196%). Mixed
  suite interrupted with SIGINT/exit130 before furtherhighLRruns. Replacement
  run_uv_joint_low_width_suite.json runs only fresh/reference lowLR100 then1000.
  Reference widths included;10passmetrics. Oldcompleteddata preserved.

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
