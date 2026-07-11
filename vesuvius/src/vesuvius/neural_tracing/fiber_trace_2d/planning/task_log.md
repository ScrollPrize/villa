# Training Throughput Parallelization Task Log

## Goal

Raise actual `fiber_trace_2d` training throughput on the current
`loader_example.json` workload by parallelizing the loader/prep path while
preserving deterministic sample ordering.

## Benchmark Command

All benchmark variants must reuse this command shape and only rewrite
`/tmp/fiber_trace_p_d2_w1.json` between variants:

`PYTHONPATH=/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/volume-cartographer/build/python-bindings/python:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3/vesuvius/src:/home/hendrik/business/aiconsulting/vesuviuschallenge/villa3 python -m vesuvius.neural_tracing.fiber_trace_2d.train /tmp/fiber_trace_p_d2_w1.json --benchmark --benchmark-batches 30`

## Attempts

- Started from existing implementation where previous measured throughput was
  ~114 patches/s with `pipeline_depth=8`, `pipeline_workers=4`.
- Baseline rerun with current checkout and `pipeline_depth=8`,
  `pipeline_workers=4`: `batches=30 patches=3840 elapsed_ms=43766.6
  patches_per_second=87.74`. This is worse than the previous ~114 patches/s
  note and confirms the current pipeline is not acceptable.
- Clean baseline after GPU was free: `pipeline_depth=8`, `pipeline_workers=4`:
  `batches=30 patches=3840 elapsed_ms=33259.2 patches_per_second=115.46`.
- Diagnostic profile attempt with the same temp config plus `--profile` failed at
  loader construction with VC3D `HTTP 0 fetching .zattrs`, so it was not used as
  throughput data.
- Load-only/profile on `loader_example.json` completed: `batches=100
  patches=12800 elapsed_ms=97263.1 patches_per_second=131.60`. Warm rows were
  roughly 6.5-8.0 ms/patch wall with loader thread factor around 30. Summary:
  `loader_wall=7.581 ms/patch`, `coord_cache=72.346 ms/patch worker`,
  `line=28.984`, `coord_aug=12.707`, `loading=14.757`. This means loader work
  is already threaded but still caps throughput near 130 patches/s when isolated.
- Variant `augment_device=cpu` with otherwise same config was interrupted after
  it exceeded the normal 30-batch runtime without completion; it is worse and
  not a retained change.
- Implemented direct multi-worker load+CUDA-prep pipeline. Exact benchmark with
  `pipeline_depth=16`, `pipeline_workers=8`: `batches=30 patches=3840
  elapsed_ms=33012.7 patches_per_second=116.32`. This proves the old single
  prep worker was not the main cap.
- Added center-offset fast path that skips loading/copying `offset_axis_zyx` for
  one-offset training. Exact benchmark: `elapsed_ms=31024.6
  patches_per_second=123.77`.
- Added training-only CP+tangent line metadata path so full line coordinates are
  not generated for the loss. Exact benchmark: `elapsed_ms=30377.6
  patches_per_second=126.41`.
- Disabling `strip_coord_cache_dir` was tested and interrupted after exceeding
  the normal benchmark runtime; regenerating coords is worse than cached coords
  for this workload.
- Training-profile attempt with retained cached fast path failed at loader
  construction with VC3D `HTTP 0 fetching .zattrs`; no timing data recorded.
- Larger batch tests after parallel-path changes: `batch_size=512`, depth 16,
  workers 8 failed with CUDA OOM because queued prepared GPU batches consumed
  memory; `batch_size=256`, depth 2, workers 8 completed at
  `patches_per_second=128.26`, not meaningfully better.
- Added training-only `include_coords=False` so post-sampling coordinate arrays
  are not retained in batches/samples. Exact 128-patch benchmark:
  `elapsed_ms=30216.3 patches_per_second=127.08`.
- Oversubscribed `loader_workers=128` was worse: `elapsed_ms=30800.3
  patches_per_second=124.67`. More CP threads do not overcome the remaining
  bottleneck.
- Retested retained 8/4 queue after removing grouped-sampling regression:
  `batches=30 patches=3840 elapsed_ms=30011.0 patches_per_second=127.95`.
- Grouping all CP volume samples into one sampler call per sampler was tested
  and removed because it regressed to `patches_per_second=114.47`.

## Final Retained Changes

- Direct CUDA load+prepare pipeline submits exact training steps to concurrent
  workers instead of serializing all preparation through one prep worker.
- One-offset training cache hits skip loading/copying `offset_axis_zyx`; zero
  offset returns the cached grid directly.
- Training requests CP-local two-point tangent metadata instead of full strip
  line coordinates.
- Training does not retain post-sampling coordinate arrays in samples/batches.
- Invalid sample skip reason spam remains suppressed in the hot training loader.

## Final Measurement

- Baseline after GPU was free: `115.46 patches/s` on the approved 30-batch
  benchmark command with `batch_size=128`, `strip_z_offset_count=1`.
- Best retained result: `127.95 patches/s` on the same command/workload.
- Target `400 patches/s` was not reached. Measurements show the remaining cap is
  the per-CP strip-coordinate cache/VC3D sampling path; more loader threads,
  larger batches, disabled strip cache, and grouped VC3D sampling did not improve
  throughput.

## Validation

- `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/train.py` passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py` passed: 106 tests.
