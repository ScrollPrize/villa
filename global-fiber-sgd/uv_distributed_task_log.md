# Larger batches and distributed joint solve

- TwoGPU pilot passed actual-loss gradient preflight and26updates; steady26.51ms.
  CPU distributed tests including actual pair/continuation preflight:6passed.
- Started exclusive32k/GPU suite via run_uv_distributed_32k_suite.json, lease
  e90ac85f-2550-477f-a12f-25ee7bb462b5. SixGPU336updates first, then1/2/3GPU
  105update timings. User-requested diagnostics profile includes copy/audit/
  reference/logging split and cProfile files. No diagnostic metrics removed.
- Review caught benchmark_updates suppressing intermediate diagnostics. Preserve
  short-timing default but add explicit benchmark_log_every. First6GPUrun is
  initial/final-only profiling; queued second6GPU336update run every33updates,
  no cProfile, for normal walltime overhead. This distinction is documented.
- Initial profile:489197 Python weighted_median calls;reference12.807s of16.028s
  diagnostic. Finalreference11.208s. Recommend compiled segmented medians and
  shared hard_geometry; no metric removal. Detailed uv_diagnostic_profile.md.
- Tests:83passed,2CUDAcasesdeselected,1existing sparsewarning. Actual NCCL6rank
  gradient preflight passes. SixGPUsteady49.725ms/update,12.792GiB/GPU; final-only
  diagnosticrun57.569s incl35.191s diagnostic overhead under cProfile.

- Plan independently approved:uv_distributed_plan.md. Preserve SparseAdam
  touched-row semantics, global RNG sampling, reject unsynchronized active search.
- Coordinate work completed/archived in uv_coordinate_task_log.md and results.
- Larger-batch exclusive4lane matrix launched:32768/65536/131072 quality with
  sqrt-scaledLR,.widthLR2; sameGPU16384/32768/65536/131072timings.105updates,
  5warmup.72GiB/GPUreservation. Approved runner/run_uv_large_matrix.json.
- True distributed implementation next, one solve onGPUs0–5; not matrixlanes.
- Larger matrix completed all7jobs. Timingthroughput16/32/64/128k=.730/.879/
  .943/.995Mconstraints/s;peak7.39/11.95/21.07/39.30GiB. QualitywithsqrtLR
  retains65.34/65.11/65.03% for32/64/128k,sign3.316/3.524/3.535%; noqualitygain.
  Fullnumbers/commands in uv_distributed_results.md.
- Implemented shared-loop distributed reducer/spawn/suite, independentreview
  approved. CPU2rank2step SparseAdam/gradient equivalence passes. CUDA2GPU
  preflight+26updatepilot launched,configrun_uv_distributed_pilot.json.
- Timing labeled gradient-reduction pipeline (includes packing/coalescing and
  sparse reconstruction, not only NCCL). SingleGPU controls useexplicitgenerator.
- FirstCUDApilot exited in preflight (missing sigma argument in small fixture's
  distance-table call); no production optimization ran. Fixed call and rerun
  same incomplete output. Spawn cleaned other rank and manager released lease.
- User selects32k/GPU tradeoff. SixGPU globalbatch=196608;10sampledpass-equivalents
  =336updates. Upcoming scaling uses1/2/3/6GPUs at fixed32k/GPU,not largerlocal
  batches. Explain global LR scaling and compare fullruntime incldiagnostics to
  older414.581s and optimizedsingle16k213.333s; no sixGPU timing claim yet.
