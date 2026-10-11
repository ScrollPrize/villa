# Quadratic profiling and local windows

- Both exclusive suites completed exit0. Final8k/16k/32kfull elapsed429.02/
  399.82/377.11s vs519.61s4k;L1=11.737049/11.745175/11.758667 vs11.723346.
  Larger batches have lower convergence at fixed data budget, no qualitywin.
  N2B32k378.93s,L1=11.789237,retained65.1803%,no advantage except memory.
- Final docs:uv_local_window_results.md,uv_local_window_progress.md. All exact
  commands/configs persisted; exclusive timing evidence only. No defaults changed.
- Scope: no placeholders or constraint-proposed windows (intentionally first
  variant only). No optimizer LR retuning or indexed-backward acceleration.

- First exclusive suite completed exit0 (all15children). Second exclusive
  quality/profile suite running. Complete-update profiler confirms indexed
  gradient accumulation dominant:aten::_index_put_impl_10.212ms(full) vs11.345ms(N2).
- Final regression suite91passed in19.59s; existing sparse warning only.
- Radius results:fullL1=11.723346, N2=11.761666,N4=11.735520,N8=11.724982,
  N16=11.723144. No useful quality/speed improvement; defaults remain full.

- Exclusive6GPU suite granted1976dcd4-71e9-4177-9d2f-05bf906753c2.
  Timings on GPU5 sequentially; independent quality lanes GPU1-4/6.
- Batch full B4096/8192/16384/32768 meanms22.392/35.303/60.007/116.370;
  peakallocatedGiB3.985/5.137/7.430/12.027. N2 means24.241/36.224/61.789/117.173;
  peakGiB3.305/3.536/3.990/4.898.105updates,5warmup; exclusive H100.
- Queued exclusive follow-up matrix: full quality8k/16k/32k,N2quality32k,
  matched eventbudget (roundedupdates),sameLR. Additional complete-update Torch
  traces, profiling lastupdate separately and excluding it from timing stats.

- Initial matrix child launch missed the existing --unmanaged/--devices0
  internal flags; all children exited before solving. Fixed and added mocked
  subprocess regression; restarted same empty outputs. Existing parent lease
  owns all subprocesses; no nested manager. Independent review found the issue.
- Matrix env isolation/cancellation reviewed; provenance records parent/lane/GPU.
-90 tests passed before adding child-command regression; new regression pending.

- Plan: uv_local_window_plan.md. Profile before local-window experiments.
- Exclusive baseline profile B4096:22.392ms full update,1.718ms isolated pair
  forward/backward. CUDA phases:1.108 coordinates,.518 pairforward,3.939 other
  forward,15.589 allbackward,1.239 coalescing/optimizer. Larger batches under test.
- Internal matrix launch fix regression passes;90-test broader suite passed.
- Approved runner pair profile started; no exclusive GPU request.
- User subsequently required exclusive allocations for benchmarks/profiles.
  Interrupted shared quality trials after about1pass; short shared profiles
  completed but are provisional only. All uv local/batch configs now exclusive
  and fresh _exclusive_v2 output directories. GPU1-4 quality; GPU5 serial profiles.
  Manager confirms interrupted tasks terminal; no orphan grants.
