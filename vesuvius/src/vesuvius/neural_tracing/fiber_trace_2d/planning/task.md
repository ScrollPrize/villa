# Native C++ Trace2CP Parallel Runtime

Bring the native `vc_fiber_trace_metric` whole-fiber runtime on the remote
precomputed fiber manifest workload close to the Python tracer runtime target.

Representative workload:

```bash
volume-cartographer/build/bin/vc_fiber_trace_metric \
  s3://philodemos/hendrik/fiber_vols/fiber_s1_001.lasagna.json \
  /home/hendrik/business/aiconsulting/vesuviuschallenge/data/train_fibers/fibers_test_paul_4/kb_20260605T150824406_000001.json \
  --normal-manifest /home/hendrik/business/aiconsulting/vesuviuschallenge/data/lasagna3d_inf/las008_s1_full/las_008.lasagna.json \
  --remote-cache-dir /home/hendrik/business/aiconsulting/vesuviuschallenge/vesuvius_fiber_trace_zarr_cache
```

Measure the actual command, add profiling/logging as needed, and iterate on
implementation changes. Log each successful and failed attempt. The expected
target is less than 30 seconds, or close to it, while preserving current trace
semantics and deterministic beam ordering.
