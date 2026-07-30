# Native C++ Trace2CP Parallel Runtime Task Log

## Notes

- User-reported current runtime is nearly unchanged after previous batching
  changes, so the next step is measurement from the actual workload rather than
  more code inspection.
- Representative command uses the remote fiber prediction manifest
  `s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json`, the local
  Paul test fiber glob ending in `01.json`, the Lasagna normal manifest, and
  the local remote-cache directory under the shared `data/` tree.

## Attempts

- Invalid baseline attempt: ran the user workload with
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/data/vesuvius_fiber_trace_zarr_cache`
  as `--remote-cache-dir`. That was wrong for the user's `$VES` command and
  created/filled a different remote-cache namespace. Do not use this run for
  cache or performance conclusions.
- Corrected cache root for the user's command is
  `/home/hendrik/business/aiconsulting/vesuviuschallenge/vesuvius_fiber_trace_zarr_cache`.
- Valid warm-cache baseline with corrected cache root:
  - command: `volume-cartographer/build/bin/vc_fiber_trace_metric s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --remote-cache-dir /home/hendrik/business/aiconsulting/vesuviuschallenge/vesuvius_fiber_trace_zarr_cache`
  - note: current output can still print `[lasagna] streaming uncached data into .../remote_lasagna/url_hex/...` for the remote manifest cache path; do not confuse this with using the wrong cache root.
  - `native_trace2cp_fiber err/kvx=0.3 restarts=5 segments=87`
  - `native_trace2cp_timing trace_wall_s=314.087 trace_cpu_s=2396.381`
  - `/usr/bin/time` reported `WALL_SECONDS=315.56`
- Added stage profiling to `vc_fiber_trace_metric`.
- Replaced full beam-frontier stable sort with deterministic bounded top-k
  scanning. This preserved the 5-restart result and reduced prune time
  modestly.
- Added direct per-worker Lasagna chunk resolution for normal sampling.
  This removed the per-generation normal key-map/prefetch/assign path and
  reduced wall time from about 136s to about 113s.
- Added direct per-worker chunk resolution for fiber prediction sampling.
  This removed the analogous prediction key-map path and reduced wall time to
  about 90s.
- Flattened prepared-request scalar and compact-axis sampling to reduce
  wrapper/matrix-loop overhead while preserving the compact tensor/eigen
  semantics.
- Added lightweight final-lookahead frontier records so the final generation
  only allocates path nodes for selected/reached beams. This reduced
  `frontier_s + prune_s`, but wall time remains dominated by prediction and
  normal materialization.
- Failed attempt: prediction-loss lower-bound pruning for final-generation
  normal sampling. It was stopped after early progress showed a severe
  regression because sorting/chunked batches dominated. The lower-bound path
  was removed.
- Final verification run after reverting the failed path:
  - command: `volume-cartographer/build/bin/vc_fiber_trace_metric s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json --remote-cache-dir /home/hendrik/business/aiconsulting/vesuviuschallenge/vesuvius_fiber_trace_zarr_cache`
  - `native_trace2cp_fiber err/kvx=0.3 restarts=5 segments=87`
  - `native_trace2cp_timing trace_wall_s=85.958 trace_cpu_s=2453.293`
  - profile: `prediction_batch_s=29.314 normal_batch_s=28.232 candidate_score_s=7.170 frontier_s=7.061 prune_s=6.509`

## Deviations

- The target of less than 30 seconds, or close to it, was not reached. The
  best validated run is about 86 seconds. Further improvement likely needs a
  larger redesign, such as fused prediction/normal/scoring kernels or a
  behavior-approved change to segment independence/search shape.
- Focused unit tests were not run because the user restricted approved runtime
  commands to the exact `vc_fiber_trace_metric` workload. The native target was
  rebuilt successfully and the representative workload was run.

## Validation

- Build: `cmake --build volume-cartographer/build --target vc_fiber_trace_metric -j 16`
- Workload: exact corrected-cache `vc_fiber_trace_metric` command above.
