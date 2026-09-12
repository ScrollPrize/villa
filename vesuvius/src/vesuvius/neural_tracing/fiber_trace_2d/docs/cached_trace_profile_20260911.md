# Cached tracer comparison — 2026-09-11

Final measured evidence: `/ephemeral/hendrik/fiber3d_train2/evaluations/classic_cached_readiness_20260911_v3`.
Exclusive manager task: `a8b6866d-81e0-473e-8162-8e2672f23f8a`; runner exited0.
Original full evaluation completed100/100 in2316.144s before the requested stop; nothing was killed.

## Workload and command

Four shortest complete locked fibers, eight directed cases, PHerc0139/PHerc0332/Paris4.
Unchanged original annotation files; classic best91_5k checkpoint's published inference.
GCC13 RelWithDebInfo, beam8/lookahead2/threads1/step4, 201.54GiB shared decoded cache.
Order off/on/on/off; fresh native process and decoded cache per run, existing disk/page-cache inputs.
No inference model/worker/accumulator initialized, zero inferred tiles and published chunks in every pass.
Readiness is still checked on first access through the same authenticated native coordinator.

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile --config /home/hendrik/villa2/vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/configs/eval_classic_collection_cached_trace_profile.json
```

Set a fresh `profile.output_root` before another run; existing evidence cannot be overwritten.

## Results

| Metric | Readiness cache off | Readiness cache on |
|---|---:|---:|
| Native wall mean / median (2 runs) | 71.968 s | 65.703 s |
| Native wall min–max | 70.471–73.466 s | 65.525–65.880 s |
| Coordinator wall mean | 72.093 s | 65.826 s |
| Native CPU mean | 451.480 CPU-s | 418.120 CPU-s |
| Mean within-run case p50 / p95 | 61.611 / 71.713 s | 55.900 / 65.160 s |
| Bridge entries per run | 8986980 | 243 |
| Bridge+permit time, sum over cases (mean) | 26.584 s | 0.259 s |
| Explicit bridge-lock acquisition time, sum (mean) | 10.028212 s | 0.000070 s |
| Local proof lookup time, sum (mean) | 0.200 s | 6.520 s |
| Ready wait time, sum (mean) | 0.173 s | 0.252 s |

Native wall reduction: **8.71%**.
ABBA native wall times: 73.466, 65.525, 65.880, 70.471s.
All eight replay artifacts match both across all four runs and their original full-run counterparts.
Replay hash: `b0284864d66fbbb53509fa8f000a16b38e3d58d31aa5ee44cce9cf1496065872`.
Each pass evaluates8986950 candidates/5558 generations.
Enabled proofs bypass8986737 of8986980 callbacks (243 bridge entries,798 submitted dependencies; no proof-cache clears).

Times summed across cases are thread elapsed time, not sequential wall time.
Bridge/permit contains lock and wait time; candidate evaluation includes readiness, sampling and scoring.
Do not add these overlapping columns.

## Bottleneck and limits

Candidate evaluation occupies99.74–99.80% of summed per-case elapsed time.
Actual readiness waits are negligible on these cached inputs.
The fast path removes redundant shared synchronization, but dependency construction,
scalar prediction sampling, scalar normal sampling and scoring still run per candidate.
The existing threads1 branch bypasses batched corner sampling. This test does not
separately measure those inner subparts and does not enable nested parallelism or
change numeric behavior. Their relative costs require another finer profiling step.

This is an eight-case short pilot, not a100-case scaling result or full inference throughput measurement.
Only two repetitions per mode; Linux amd64 tested, not macOS/arm64.
The old full-run total includes inference and a different workload length, so it is not a timing baseline here.

## Rejected preliminary pilots and validation

The cropped128-voxel pilot (v1) failed closed on an uncached reverse-start chunk.
v2 copied whole references through a float32 Python loader; off/on results agreed,
but original-full-run comparison detected altered reference arcs. Its preliminary12% gain is superseded.
v3 passes original paths unchanged and requires exact original replay equality.

The shared coordinator extraction and readiness paths passed318 native cases and76 Python tests
(Python34.35s) using the existing command:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
```

The script explicitly selects this checkout's Python source path to avoid a stale villa editable installation.
Independent plan/code review covered fail-closed cache reuse, source validation,
cancellation, artifact membership and runner exit status.
