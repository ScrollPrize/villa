# Archived manager merge and inference profiling — 2026-09-10

- User paused queue instrumentation; partial implementation and remaining work in docs/live_inference_concurrency_20260910.md.51 precommit tests pass.
- Checkpoint commits:f31412abc native tracing,2c88b8ffd shared demand/accumulator,ef16a59e5 Fiber runners/results/configs. Working tree clean before merge.
- Merging local fiber-training-manager e7bde77d5 into fiber-training-retrain; common ancestor6009bb7f4. No rebase needed.7 conflicts:5 planning files,train.py,volume-cartographer/README.md. C++ implementations merged automatically.
- train.py combines sample-axis progress with manager handoff/shared snapshot helpers. Preserve sample metadata in both durable snapshots and RAM handoff; retain resumable-test dedup guards. README retains HiGHS docs and test diagnostics; changelog preserves both histories.
- Manager protocol v1 and capability amendment read; no protocol changes authorized/needed. Fixed evaluation uses existing low-level Client with pausable=false/adaptable=false and explicit priority, exclusive=true. Cooperative registered training can yield; arbitrary GPU processes cannot be safely evicted.
- Independent merge/plan review approved; corrected stale draft/pending wording
  in merged specs to match locked protocol and scoped reference validation.
- Tests explicitly force villa2 import roots:25 training-state/gradient/sample
  progress tests pass (12CUDA cases not run),54 manager/scheduling tests pass.
  Initial unqualified invocation resolved one module from the environment's
  villa checkout; discarded that result and reran with explicit villa2 paths.
  Gloo local sockets required the approved pytest command outside sandbox.

- Merge committed as fed6dc24a; exclusive wrapper is still uncommitted/incomplete.
  Its independent review identified a controller/grandchild crash tracking gap;
  do not present it as production-ready or bypass manager protection for profiling.
- User resumed concurrency work. Fast training config actually uses128 shared
  process workers,87 load-affinity,2048 slots,128 initial credits; inference had
  only8 readers/8 input slots and32 trace cases. Earlier "fast loader" wording
  incorrectly implied reuse of the whole training pipeline: only native sampling
  is shared. Ordered training sample/augmentation scheduling is not reused and
  remains explicitly outside this bounded admission change.
- Memory read-only check: about1.9TiB available and1008GiB /dev/shm free.256
  uint16 512-cubed input slots cost64GiB plus8GiB for128 current result slots;
  native read temporaries/caches are additional, bounded by reader concurrency.

- Independent review corrected earlier tracing-limit explanation: native live
  runner already starts one thread per direction and releases CPU permits during
  inference waits. workers32 capped active CPU only; changed classic config to
  workers0 (automatic, bounded by case count/hardware), NOT256 newly enabled cases.
- Implemented input_slots independent of devices; classic256inputs/128readers/
  128results, full evaluation chunk-major; shared native reader options for
  profiler/full runner, buffered live1s queueJSONL, bounded16admissions/8copies
  per pump and reader-sized executor admission. Deeper not-yet-admitted work
  remains priority-selectable; active reads are not preemptible.
- Manager watchdog independently reported before execution, survives launcher
  SIGKILL and cleans non-detaching worker group before exiting; shared cleanup
  with launcher. SIGTERM/HUP cancellation now goes through release lifecycle.
-64 CPU tests pass in29.18s (exact command below). New128reader test proves
  capacity and216tile exact equality to existing serialnumerics, not native/GPU
  performance. Initial test expected original85 but shared fp16/truncating
  identity path produces84; corrected to serialoracle, no numerical changes.
- No manager was running. User approved starting it for all8idleGPUs. Initial
  startup rejected224 logicalcores; corrected to112physicalcores per protocol.
- Profile20260910T141534179391 failed before inference: native I/O128 exceeds
  fixed64capacity. Profile20260910T141617317064 similarly rejected32decode
  workers above8capacity. Both tasks cleaned/released through manager. Set
  supported64I/O+8decode; retain128tile readers/256slots. Native capacity tuning
  beyond these requires separate startup configuration/binding work, not done.

CPU validation command (explicit villa2 import roots):
```
/home/hendrik/.venv_las/bin/python -m pytest --noconftest -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' -q gpu_manager/tests/test_exclusive.py lasagna/tests/test_demand_predict3d.py vesuvius/tests/neural_tracing/test_live_inference_profile.py vesuvius/tests/neural_tracing/test_managed_evaluation.py vesuvius/tests/scripts/test_classic_fiber_evaluation.py
```

## Completed managed profiles / current status

Same approved classic runner `profile` command, PHerc0139 reference-intersection
workload,8H100,512-cubed inputs, unchanged checkpoint/bf16/f16raw/f32weights.
One approximately40s measured window each after all8workers warmed; isolated
inference profiles, NOT full100direction trace evaluations.

| Timestamp under inference_profiles | Variant | Completed128-cubed equivalents/s |
|---|---|---:|
|20260910T130746099412|Historical8input/8reader chunk-major+cProfile|322.899|
|20260910T141653965159|256inputs/128readers, coordinator copies+cProfile|87.848|
|20260910T141944960160|Same, cProfile disabled|83.210|
|20260910T142327806621|Reader-side shared-slot copies, cProfile disabled|169.581|

- Removing cProfile did not resolve regression. Moved input copy into reader
  thread, keeping reserved shared-slot ownership through H2D; future now returns
  dtype/copy timing only. No raw-array retention in ready futures, no byte changes.
  Independent review passed;44 demand regressions passed26.61s after this change.
- Latest106fully accumulated512-cubed tiles in40.004525s;197GPUresults received.
  Accumulation backlog25→116tiles; final124readfutures,10GPUsubmitted,
  2free result slots,82queued+4active publications. Do not equate received with
  complete/consumer throughput; this is not yet a gain over historical323.
- Latest coordinator stage walltime: admission32.785s, result handling5.886s,
  accumulation/publication pumping0.997s, GPUdispatch0.0047s. These are stage
  elapsed times including descheduling/contention, NOT CPU attribution. Median
  sampling gap1.112s,p952.210s. Current measured choke point is coordinator
  admission/control responsiveness; exact admission suboperation not yet isolated.
- Model CUDA mean0.800s,p500.771s,p950.995s per512tile. Accumulation taskmean
  2.541ms,p500.888ms,p957.086ms; queued+ack mean0.870s. Native readermean0.998s,
  p500.785s,p952.397s. Shared-copy summed worker elapsed37.792s overlaps across
  readers; it is not37.792s of serial coordinator work. read_done now means shared
  slot populated awaiting dispatch; load_seconds includes read/copy/dispatchwait.
- GPUmanager confirms all5attempts terminal with zero held/granted resources,
  including latest69568cea-fffe-4ae9-a695-b4486d280767. No training started.
  Manager service remains running; profiling command finished. GPUutil snapshots
  showed alternating busy/idle, not sustained saturation; no averaged hardware
  utilization metric is claimed. Changes remain uncommitted pending remaining work.
