# Demand-scoped GPU ownership (2026-09-14)

- Recorded first remaining family's one-coordinate sensitivity: plain mean
  LRU K4/512/w.30 stays68; denser insertion and more anchors worsen it locally.
  This is not a Cartesian optimum. Corrected stale next-step paragraph now that
  softmin67 exact verification has completed. Live session14738 continues the
  transferred seed search with zero inferred tiles; final repeats pending.

- Continuation verified live exec14738 (not restarted): remaining CPU refinement
  report has21 completed variants and first plain-mean-LRU family minimum68.
  Family repeats are performed at search end and remain pending; overall prior
  verified best64 unchanged. All21 inferred_tiles counts zero. Startup follow-up
  committed separately as eb6f749db.

- Startup follow-up: defer manager bootstrap and first Supervisor tick until
  the first input is ready. No GPU grant during initial CPU reads, and unused
  close never registers. Review caught idle-manager exit before delayed first
  registration; deferring bootstrap resolves it. Protocol unchanged.
- Approved cooperative test caller, output `20260914_092511_225918`:
  CPU lifecycle tests 92 passed/1 opt-in skip in52.95s. Separate GPU smoke
  verified pre-demand no-registration/no-workers assertions, then skipped after
  90s without manager capacity (91.80s total). No GPU execution claimed for this
  follow-up. Independent final review passed; diff check clean.

- Stopped boundary sweep via approved experiment launcher. Task 9cf987a9-ef04-441e-aabf-1a8e26fea75a finished, zero held CPU/GPU confirmed. Partial report preserved at evaluations/embedding_ema99_anchor_boundaries_20260914_v1/report.json.
- Independent plan review approved with drain/join, retained CPU state, separate administrative/self pause and heartbeat invariants.
- Added deferred/gated GPU lifetime to shared DemandWorkerPool; existing CPU queues and numerical paths retained. Supervisor owns GPU lifecycle, no manager protocol changes.
- CPU tracing deliberately continues outside GPU reservation accounting while paused, as requested. Slow unfinished reads do not trigger idle reacquisition.
- Build instructions referenced by root AGENTS (volume-cartographer/scripts/build_dependencies.sh) are absent. Used existing configured optimized CMake build through approved test caller; no dependency installation.
- First validation: 88 lifecycle/pool/Supervisor tests passed; existing native tests passed. 118/119 embedding tests passed; updated obsolete test expecting whole-sweep reservation to new demand-scoped contract. Added native heartbeat integration coverage; rerun underway.
- Review fixes: unconditional nested cleanup after worker errors, idempotent pool close, nonblocking GPU join lifecycle, heartbeat ready/deadline race fixed. Transport queues recreated after terminating idle Queue readers (old read locks must not be reused).
- Real nonexclusive4GiB GPU smoke passed two idle release/reacquire cycles with exact uint8 output in7.93s. It used its own manager task and finished; no training/sweep restarted. Added administrative pause case for final run.
- Exact compatibility map permits the two audited scheduling-only old/new source SHA pairs, with all remaining fingerprint fields strict; logs acceptance and does not rewrite old cache metadata. New arbitrary edits fail closed.
- Native heartbeat timeout test initially used unavailable doctest compatibility macro; switched to supported CHECK_THROWS_WITH_AS and rerunning.
- Native compatibility wrapper requires a matcher for that macro, so replaced assertion with explicit try/catch and exact message check. Native heartbeat success and absent-heartbeat timeout tests now pass.
- Final review found no GPU ownership blocker. Fixed paused CPU read exceptions to surface immediately, with regression test. Existing CPU executor/uninterruptible I/O shutdown can still block; this task does not claim universal bounded OS process termination.
- Managed smoke with administrative pause passed in9.27s: reads become ready without GPU workers, administrative pause remains set, then grant/restart processes retained data exactly. Two idle release/reacquire cycles verified zero held GPU allocations. No sweep/training restart.
- Final approved command: `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests`. Results at `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260914_090635_610805`: lifecycle90passed/1GPU-opt-in skip50.99s; separate actual GPU smoke1passed9.27s; native2passed4.83s; embedding/inference120passed34.98s (includes new paused-read failure regression). Optimized GCC QuickBuild-O3 native build, tiny synthetic identity-volume inputs; no performance claim from this correctness smoke.
- Manager status confirms all three smoke tasks terminal with zero held CPU/GPU. Original sweep remains stopped. `git diff --check` clean. No full sweep or training rerun requested/performed.
- Goal continuation, read-only evidence audit: interrupted report now contains141 variant records/eight completed families. Added exact partial family table to docs/embedding_refinement_summary_20260914.md. Best remains64; support-softmin67 requires final repeat and complementary-failure analysis before proposing a combination. Kept GPU sweep stopped per user instruction.
- CPU-only saved-failure comparison: LRU64 versus softmin67 has60 one-to-one matched failure locations within64 base voxels (4LRU-only/7softmin-only); at128,61matched/3/6. Both direction reference ranges equal. Most failures shared, so broad mixture is not justified by aggregate scores; unmatched regions need path inspection, not an oracle claim. Full method/counts documented in refinement summary; no GPU restart.
- CPU follow-up: local±32-base-voxel inspection finds clean softmin neighborhoods at reverse-reference arcs9676.86,12656.47,12792.75 (max threshold ratios0.829/0.343/0.858); forward2102.14 intersects a restart. These are not causal recoveries because histories differ. Started bounded cached-only verification via approved embedding-trace-sweep caller, `fill_missing=false`, output embedding_ema99_boundary_cpu_verify_20260914_v1. Controls + LRU64 + two identical softmin67 candidates + repeat-best; no GPU requests permitted.
- Cached-only verification finished(exit0): counts87/82/64/67/67/64. Both controls, LRU candidate/repeat and both softmin candidates match the earlier exact replay hashes. Zero inferred tiles throughout, no manager/GPU reservation. Independent read-only review verified candidate equality and no-GPU routing. Softmin67 is now repeat-verified; overall best remains64.
- Started seven remaining family refinements in embedding_ema99_remaining_cpu_refine_20260914_v1, cached-only/inference disabled. Independent review verified exact equality to original plan indices8 onward, no completed-family overlap, fail-closed missing coverage. Parameter-cache reuse is within invocation only, so some individual trials can repeat old partial trials; no numerical or search-axis changes.
- Preserved validated implementation as focused commits: e71527d5f(native heartbeat), eb7c4af4b(shared GPU lifetime), a6bd32fa9(demand manager integration/tests/docs). Remaining CPU-only refinement passed off87/original82 exact hashes and entered first pending family. Active exec handle14738; output remaining_cpu_refine_20260914_v1. No GPU grant requested.
