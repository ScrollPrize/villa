# Archived task log: remove demand-inference SQLite — 2026-09-10

- User explicitly requires RAM-only bookkeeping, no persistent database.
- Earlier manager/profiling log archived in docs/managed_inference_profiling_history_20260910.md.
- Baseline instrumented run20260910T145617665960:177.9999 equivalent128-cubed
  tiles/s,40.2697s; per-tile SQLite updates27.5349s elapsed but0.0354s calling-thread
  CPU. Priorityselection0.0892s and executor.submit0.0057s. This locates the wait
  around SQLite, not its underlying I/O/GIL mechanism.52profiling/regression tests
  passed before the final optional-platform-counter guard. Follow-up diagnostic
  run20260910T145855340044 completed; discussion interrupted the planned stack
  inspection, and no successful native stack inspection was obtained. No evidence supports claiming a proven
  database-disk bottleneck.
- Preserve all unrelated uncommitted manager/concurrency/profiling work. Remove
  only live-inference database dependency; keep existing dict/set/deque states.
- Removed SQLite imports, schema, reads, updates, commits and per-tile timers.
  Existing coordinator dict/set/deque structures are authoritative. One static
  cache_identity.json replaces only provenance, not scheduling persistence.
- Independent plan/code review identified unidentified-cache adoption and nested
  chunk-path symlink hazards; both fixed with preservation regressions. No full
  volume scan. All incomplete sibling paths validated before any replacement.
- Legacy DB-only caches are refused untouched, not migrated. Incomplete work is
  recomputed after restart. Removed persistent cryptographic fingerprints: decodable
  same-size corruption is no longer detectable. Process interruption is covered;
  arbitrary power-loss transactional publication is not promised.
- Regression command: /home/hendrik/.venv_las/bin/python -m pytest --noconftest
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2'
  -q lasagna/tests/test_demand_predict3d.py lasagna/tests/test_inference_profiling.py
  vesuvius/tests/neural_tracing/test_live_inference_profile.py
  Result: 60 passed in24.79s. Includes forbidden SQLite connection, warm zero/edge
  chunks, interrupted siblings, unsafe scratch, priorities and multiworker slots.
- Final independent review passed. Managed benchmark command reused unchanged:
  /home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
  Run20260910T151437745034 report under
  /ephemeral/hendrik/fiber3d_train2/evaluations/inference_profiles/.
  40.6604s window,113 fully accumulated512³ tiles,210 received,20 published chunks;
  177.8633 equivalent128³ tiles/s versus177.9999 before: no measured speedup.
  State updates286calls0.000275s total; coordinator admission6.518s,
  background19.378s,events14.358s. Executor.submit6.439s wall/0.0054s CPU;
  budget checks10.431s inclusive; cache completeness0.225s over11,386calls.
  Contention remains elsewhere; neither disk-I/O nor GIL cause is proven.
  Accumulation mean/p50/p95=2.244/0.826/5.247ms. Singlewindow, unchanged8H100,
  PHerc0139 workload,512³ inputs,128readers,256inputs/128results,chunkmajor,
  bf16model,f16raw/f32weight,no native rebuild. Detailed comparison and exact
  commands in docs/live_inference_ram_state_20260910.md.
- Manager task da7721dd-2ab3-40bc-bb22-6b188705aa57 terminal/completed, zero held
  and granted GPUs. No training started. Full100-direction replay not run;
  no kernel/conversion/executor or budget-policy change in this task.
- Final native wrapper passed62 tracer cases,1 global-cache case and66 Python
  replay/shared inference tests in29.98s. Independent review passed. Working
  changes remain uncommitted.
