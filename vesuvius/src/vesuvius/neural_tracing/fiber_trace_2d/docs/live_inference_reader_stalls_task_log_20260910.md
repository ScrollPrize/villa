# Archived task log: diagnose remaining inference stalls — 2026-09-10

- User requests continued profiling and a bottleneck report, not a speculative fix.
- Previous RAM-only task archived in docs/live_inference_ram_state_task_log_20260910.md.
- Baseline RAM-only run20260910T151437745034:177.8633 equivalent128³ tiles/s;
  admission6.518s,background19.378s,events14.358s in40.660s. RAM updates0.000275s.
- Inspect native reader allocation/GIL boundaries, coordinator queue operations,
  and budget checks. Obtain external native stack evidence during the same
  approved managed profile. Debugger-attached runs are diagnostic, not clean
  throughput comparisons. No model math, scheduling or numerical changes.
- Independent plan review approved. GDB attachment to3348406 was rejected by host
  ptrace policy even with approved escalation ("Inappropriate ioctl for device").
  No stacks obtained and no security settings changed. Run20260910T151946615783
  completed normally:188.5661equivalent128³/s,119accumulated/211received512³ tiles
  in40.3890s. This is a second unchanged-code window, not an optimization gain.
- Fallback adds opt-in reader_responsiveness_probe to existing profile runner:
  2ms independent Python heartbeat,12trials each idle sleep, no-intersection
  native128³ and512³ allocation controls (requested_chunks must be0), warmed
  native tile and full reader wrapper. Separately time return-buffer destruction,
  capture calling-threadCPU/faults, record actual native module path+SHA256.
  No-intersection controls run the same C++ output/validity vector initialization
  but no sourcechunk fetching/copying. Heartbeat delays include GIL/OS scheduling,
  not a direct GIL ownership sampler. Diagnostic cProfile also enabled temporarily.
- Targeted profiling tests:10passed1.74s; helper cleanup, error and metrics covered.
  No production reader/numerical changes. Same approved command reused.
- Diagnostic run20260910T152402140226 confirms no-I/O512³ allocationcontrol
  mean63.703ms wall/63.693msCPU (12calls),32,769minor/0majorfaults percall;
  2msheartbeat delayedmax65.354ms. Idlemax2.076ms;128³controlmax2.238ms but
  only2heartbeatintervals (not statistically meaningful). Real prewarmednative
  mean167.903ms vs fullwrapper397.865ms, heartbeatmax64.753/69.750ms.
  Returnbufferdestruction timedseparately. SourcezeroingoccursbeforeGILrelease;
  GIL-heldnative setup is confirmed blocking, not proof ofexclusivepipeline share.
- Diagnostic cProfilewindow156.8730equivalents/s,40.3893s,99accumulated/196received,
  notcomparableascleanthroughput. cProfileincludesbackgroundthreadactivity and
  overlappingcumulativetimes; annotatedreportnotetousestagecounters instead.
- Added docs/live_inference_reader_stalls_20260910.md with all controls, counts,
  percentiles, nativeextensionpath/hash, commands and limitations. No-I/Ocontrols
  guarantee0requestedchunks; prewarmedrealreadsnotassertedstorage-IOfree.
- Independentfallbackreviewpassed. Manager tasks9ebf8698-4fff-4e16-8cef-c69cbd837a3a
  and8a4dc677-7c73-417c-b832-81504db080f3 completedwithzeroheld/grantedGPUs.
  Diagnosticflagsrestoredfalse. No nativeallocation/conversionfix implemented;
  userrequesteddiagnosis. No full100-directionrun. Exactextensionbuildflagsnot
  recovered, installedbinaryreusedwithhashrecorded. No installs or securitychanges.
