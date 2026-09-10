# Task log: cooperative resource manager and elastic Fiber3D training

## 2026-09-10 — reference validation

- Final independent review found no remaining blocker under revised scope.
  Post-lock metadata checks: manager61passed (12.33s), reference harness7passed
  (1.55s), `20260910_101540_260997`. Commits: short tests/timer `d46718e4d`,
  protocol lock `2d1ab122f`; no runtime protocol shape/semantic change at lock.
- Revised short validation completed: CPU344passed/18GPUskips in71.31s and
  manager61passed in12.29s (`20260910_100827_372654`); two-update profile passed
  in161.41s (`20260910_100958_644509`), owned peaks8.03/7.78GiB. Protocol v1 locked
  with explicit no-change-without-user-authorization instructions. Long timing
  repeats and pause-RAM profiling omitted per user direction; no eight-GPU or
  convergence claim. Default restored to CPU/manager, invocation cap300s.
- User stopped the excessive timing sweep and requires tests lasting minutes.
  Interrupted only launcher3041489 and its test group; verified no test processes
  remain and GPUs0/1 returned to zero MiB. First complete repetition retained;
  second fixed case intentionally incomplete. Normal launcher now rejects the
  long sweep and enforces a 300-second total invocation budget. Removed the
  unrun pause-RAM study; queue bounds remain structurally tested. This overrides
  the earlier three-repeat performance/protocol-lock gate, not functional scope.
- First long repetition completed: unmanaged/fixed states byte-exact at110
  updates; dynamic lifecycle/frontier complete. Dynamic BF16 trajectory drift
  amplified: model relativeL2 0.010365, optimizer moments0.114254; final held-out
  test1.606135 versus fixed1.582692. This is recorded explicitly, not called a
  numerical-equivalence or convergence pass. Earlier frozen-gradient and exact
  same-allocation restart checks remain distinct evidence.
- Final failure-path review added real spawned CPU/Gloo forward, reduction,
  handoff and release failures plus abort/join-failure reservation retention.
  Manager suite now 61 passed (13.30s), `20260910_093836_893334`. No production
  changes needed. A separate short pause-memory observation is planned; bounded
  queue tests alone are not labelled as measured host-RAM settling.
- Extended benchmark caught a test-only stale pause timestamp at dynamic step20:
  the step1 resume had not reset it, causing an immediate false 20-second timeout.
  Deadline now resets for each newly constructed cohort, with a two-cycle timing
  regression retaining the real timeout. Production unchanged. Failed run
  `20260910_085311_079168` retained; its completed unmanaged/fixed cases are reused
  explicitly by the new run `20260910_093209_505125`, not silently overwritten.
  Harness tests 6 passed (1.61s). First global two-GPU throughput: unmanaged
  15.9802 samples/s, fixed16.1110 samples/s; repeated results still pending.
- Final CPU/manager rerun `20260910_092145_189992`: 341 passed/18 GPU skips
  (75.32s), manager 56 passed (5.25s). Additional independent-review regression
  checks applied grants plus completed updates, not merely issued requests;
  reference-unit suite 5 passed (1.49s), `20260910_092350_114722`. Saved lifecycle
  evidence can be revalidated without repeating GPU work.
- Final manager audit fixed strict integer validation for held allocations and
  typed retry equality (Python true/1/1.0 equality must not bypass the schema),
  plus validation on registration reconnect. Thirteen new regressions; full
  manager suite now 56 passed (5.42s), native 3/3 passed (2.51s), run
  `20260910_090759_469557`. Independent reviewer found no further concrete blocker.
- Commits on current `fiber-training-manager`: native prerequisites `72f7fccb4`,
  standalone manager/protocol draft `2127da6f5`, Fiber integration `96b2b5b5e`.
  Final validation/docs remain separate pending commits.

- Continuing after user requested completion: implement/run counterbalanced
  unmanaged-cohort / managed-fixed / dynamic reference benchmark, 10 warmup +
  100 measured B128 BF16 updates, three repeats, GPUs0/1. The unmanaged baseline
  is the identical bounded production trainer/cohort with manager RPC/admission
  removed, not the old unbounded graph-retaining algorithm. Repeated dynamic
  transitions include GPU-free pauses, two-GPU reduced VRAM and one-GPU shrink.
  Sharing is allowed, not proof of contention; measurements are recorded without
  attributing a cause to timing differences from that permission alone.
  Fixed approved launcher timeout expanded to bounded six hours for this suite.
- GPU inventory during first benchmark showed only approximately 8 GiB used on
  GPUs0/1 and all other GPUs idle: the earlier external-training assumption may
  no longer hold. Do not claim contention as a measured cause; report the run's
  allowance for sharing and observed state separately. GPU permission stays0/1,
  max20GiB, regardless of other idle devices.
- Profiler repeated stages now sum microbatch wall time instead of retaining
  only the last call; added regression. Separate short CUDA trace planned after
  throughput measurements, not mixed into their wall-time sample.

- BF16 same-allocation reference restart also passed byte-exact model/optimizer
  and final test metric after six B128 updates, two GPU-free pauses (including
  test interruption): `20260910_083716_992318`, owned peak 8.23 GiB/GPU.
  Production BF16 unchanged. Fixed approved launcher used for every test;
  default suite selection restored to CPU/manager after GPU tests.
- Remaining controlled performance gate cannot be attributed cleanly while the
  authorized GPUs share unrelated active training. No isolation or interruption
  of that job is authorized; an uncontended GPU window is needed for the planned
  overhead comparison. This is not a protocol-lock or full-completion claim.

- Frozen S1A FP32/B128 gradient oracle passed both allocations: all 612
  component/layer pairs within calculated summation error bounds (maximum bound
  fraction 0.059152 for one GPU, 0.032104 for two). FP64 shadow sums agree;
  sample IDs/counts exact. Run `20260910_082534_937046` under cooperative-tests.
  These shadow sums are test-only, not a change to production computation.
- Same-allocation real S1A pause/restart passed byte-exact model/optimizer and
  metric equality after six updates, including interruption during test;
  `20260910_083025_287860`. Driver checks confirm GPU-free pauses.
- Validation refinement: resized final trajectories are reported diagnostically
  (including failures of the original pointwise criterion), not accepted by
  widening epsilon. Analytical gradient tests and frozen-gradient roundoff bounds
  are separate correctness checks; exact restart remains zero-tolerance. This
  does not prove BF16 bit-identical behavior across different batch shapes.
- Independent review found no production blocker; fixed a diagnostics-only
  reused-reference file lookup by comparing already-loaded states, with regression.
- Regression run `20260910_083405_608714`: manager 43 passed, CPU 337 passed/18
  GPU skips, CUDA AMP 12 passed, CUDA cohort 1 passed. Final CPU rerun including
  reused-state regression: 338 passed/18 GPU skips (73.40s), run
  `20260910_083635_795365`. Controlled throughput
  (100 updates x3 per mode) and protocol lock remain uncompleted, explicitly.

- User clarified production should keep BF16 compute. The FP32 override lives
  only in generated reference test configs; production autocast unchanged,
  component gradient banks remain explicitly FP32. CUDA exact-state restore
  checks now pass FP32/FP16/BF16 (12 AMP tests total); CUDA cohort restart passes.
- FP32 constant-micro1 lifecycle also has small partition-dependent differences:
  model relative L2 1.39e-5/max 0.000267, optimizer moments relative L2 2.04e-5
  and all pass the original pointwise threshold. Final test 2.242254823 versus
  2.242277861. A separate identical fixed FP32 run is byte-identical in model,
  optimizer and metric. Therefore not baseline run noise; summing 64+64 versus
  128 contributions remains a candidate. No tolerance has been relaxed.
- Added test-only frozen-update FP64 shadow sums observing actual contributions
  to the shared FP32 banks, exact sample/count checks and per-layer roundoff
  bounds. Optional observer defaults to None and cannot alter default math.
  One-GPU B128 passed all 612 component/layer bounds (max fraction 0.059152);
  two-GPU and cross-partition FP64 comparison running. Probe requires two
  updates, so only its first frozen-model update is audited; second is unaudited.

- BF16 lifecycle completed all transitions and six updates, frontier 768, no
  pending test and matching scaler state; strict elementwise oracle failed.
  Saved states show model relative L2 5.90e-5, max absolute 0.001178; final test
  2.239006564 fixed vs 2.239104435 dynamic. Not declaring an equivalence pass or
  silently relaxing tolerance. Starting FP32 constant-microbatch-1 control to
  distinguish handoff/reduction correctness from BF16 batch-shape rounding.
  The reference harness now permits explicit test precision; production unchanged.
- The real lifecycle reached pause and a one-GPU regrant, then failed in shared
  CUDA batch preparation: index-less `cuda` was passed to set_device in the
  background thread. Resolve the initiating thread's device index at pipeline
  construction; added CPU-mocked regression for implicit and explicit indices.
  Strengthened reference oracle with driver-zero pause checks, an initial-test
  pause/resume, scaler comparison and persisted diagnostic states. Rerunning.
- Rerun CPU suite passed 336 tests (15 GPU-only skips, 99.41s). Fixed real
  reference six updates completed in 204.1s with identical logged losses to the
  prior fixed run. Dynamic reference successfully paused during initial test:
  pending rows retained and driver reported zero owned GPU processes; initial
  test resumed to the same 2.640352 loss. Remaining resize/numerical checks active.

- Native suites passed (3/3, 2.75s). Real S1A B128 BF16 calibration passed
  microbatches 1 and 2 on GPUs 0/1, highest owned driver VRAM 8.02 GiB/GPU.
  Results: cooperative-tests/20260910_073826_465551 under fiber3d_train2.
  Started fixed-vs-pause/shrink/expand lifecycle through the same approved caller:
  cooperative-tests/20260910_074431_128557. No throughput claim from shared-GPU
  calibration, and no protocol lock before full validation.

## 2026-09-09 — implementation in progress

- User raised the current-session process/thread limit; verified 65536. Resumed
  approved launcher. Added opt-in VC_TEST_TRACE case names to the small native
  test runner and a 120s CTest timeout so a hanging case is identifiable and
  cannot silently consume another long wait. Default test output is unchanged.
- Native rerun passed all three suites in 2.75s (including full chunk cache);
  earlier hang cause is not proven by that rerun. Capped reference first found
  scikit-build's installed editable finder overrides PYTHONPATH; added an isolated
  test-only vc/vc.volume finder, leaving venv untouched. Real S1A BF16 B128
  microbatch1 now completed initial evaluation (26.12s) and update1 (58.63s
  fw/bw/step). Calibration remains running; no performance conclusion from this
  deliberately small microbatch on shared GPUs.

- 2026-09-10 continuation: HiGHS optional build now succeeds. GCC LTO subprocess
  failed with Operation not permitted even under limited affinity; existing
  QuickBuild with VC_QUICKBUILD_OPT_LEVEL=3 avoids LTO and built all requested
  targets successfully. No source numerical changes or venv installation.
  Native test_fiber_json passed, test_fiberlet_crop_trace passed (1.15s).
  test_chunk_cache did not complete; stopped ONLY that test PID 2879526 after
  1164.89s. Its hang predates the later eight-GPU training startup, cause unknown;
  do not label it a resource-limit diagnosis or a pass.
  Results: /ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260910_063925_528265/.
- Host now reports 16,290 threads for hendrik against RLIMIT_NPROC 16,384;
  read-only process creation also failed with EAGAIN. The unrelated villa2
  training (root 2883519, started 06:47:46) has many 117-thread workers. No such
  processes were changed/killed. Full GPU validation is blocked by host headroom;
  the sequential launcher stopped on native-test failure before CUDA calibration.
- Added (not yet executed) real-reference fixed-vs-pause/shrink/expand lifecycle
  harness using production broker/cohort/supervisor, B128 and 16GiB grants.
  Independent review identified cleanup exceptions skipping server shutdown;
  corrected with ExitStack. Pause currently verifies joined workers/no parent
  CUDA context, not a separate driver-memory-disappearance oracle. Calibration,
  reference lifecycle/performance results and protocol lock remain outstanding.

- User approved making HiGHS optional: guarded only LP/MILP solver and its tests,
  kept non-HiGHS tracing/preprocessing helpers and explicit unavailable error.
  Native rebuild passed HiGHS discovery, then found an unrelated existing
  registered-but-absent test_fiber_json.cpp; supplied a minimal parser regression
  rather than silently removing that registered test. Continuing native build
  via the fixed approved caller, no install or GPU activity in this build.

- Added capped/shared-GPU reference calibration to the approved fixed launcher
  (16 GiB allowance, 12 GiB allocator, 4 GiB reserve); no successful reference
  training yet. Fixed initial reference path lookup. Real S1A startup then failed
  in VC3D cache initialization: setters limited admissions only after the default
  64 fetch / 32 probe / 8 decode physical pools were already constructed. Added
  explicit bounded native initialization reused by cooperative loading/calibration.
  This native change is UNBUILT: existing CMake now requires HiGHS and cannot find
  highsConfig.cmake. No install authority; asked user for an existing prefix.
  The documented build_dependencies.sh is absent; inspected the actual
  scripts/install_build_deps.sh instead without executing it. Native build was
  attempted only through the approved launcher, with no install/venv mutation.
- Added request/response schema conformance and real socket/crash/liveness tests:
  manager suite 43 passed (6.88s) via the approved caller, output
  /ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260909_202809_825544/.
  Latest CPU rerun: 334 passed, 15 skipped (CUDA opt-ins), two existing Pillow
  warnings, 94.48s. Independent read-only review found no new correctness blocker
  in conformance/schema or bounded cache initialization; native validation remains
  blocked. `git diff --check` passes. Protocol
  remains DRAFT. Full reference lifecycle/performance harness and validation,
  final review and lock remain unfinished; tiny CUDA tests do not replace them.

- Continued implementation/review: fixed constructor-failure child cleanup and
  ensured remaining supervisor cleanup runs if one close fails. Calibration now
  requires target rank-count coverage at every admissible microbatch. Added
  owned-driver-memory failure tests. Manager/supervisor: 33 pass; integration:
  4 pass; broader CPU suite: 330 pass, 5 CUDA skips (63.69s). Earlier Zarr hang
  reproduced at event-loop wakeup in sandbox; the same suite succeeds outside
  its socket restrictions. No production training processes were touched.
- User now permits GPU testing with <=20 GiB/GPU. Added tiny explicitly opted-in
  CUDA AMP oracle and shared 2->1 cohort tests with lower allocator caps. Interrupted
  approval left no matching test process running; no result claimed for that attempt.
- User requested a reusable test caller: fixed no-argument
  `vesuvius/scripts/run_fiber3d_cooperative_tests`, adjacent JSON controls named
  suites/devices/timeouts. Logs/control/results go to fiber3d_train2. Awaiting
  approval of that fixed prefix before running the CUDA suites; no per-run
  argument changes or arbitrary shell evaluation. Approval granted; first run:
  CUDA oracle 9 pass in 33.88s, CUDA 2->1 restart 1 pass in 28.01s. Tiny models,
  not reference throughput. Added shared owned-PID VRAM monitoring to the caller;
  verifying through the same approved command.
- Monitored launcher rerun passed: 9 CUDA oracle tests (7.29s), 1 CUDA cohort
  test (28.29s). Observed owned driver peaks: oracle 845152256 bytes; cohort
  GPU0 1579155456 / GPU1 1587544064 bytes (<1.5 GiB each). Results:
  /ephemeral/hendrik/fiber3d_train2/cooperative-tests/20260909_195322_632607/results.json.
  Launcher shell syntax/Python compilation and git diff --check pass. This
  completes reusable caller setup, not full reference-model validation or the
  protocol lock. Reference calibration still needs a capped/shared-GPU probe
  before it may run under the current 20 GiB permission; do not run the existing
  uncapped calibration entry point on this occupied host.

- User clarified production training is CUDA-only; CPU execution is solely a
  test harness for lifecycle/numerics. No production CPU fallback is being added.

- Connected opt-in managed launcher, persistent CPU broker, single writer and
  shared trainer. CPU integration passes actual two-process Gloo -> one-process
  restart with changed microbatch, comparing model/Adam state to fixed baseline;
  scalar steps and config text are not duplicated. No CUDA activity.
- Dense test waves preserve completed logical rows across handoff; CPU test
  interruption/resume changes physical batch 2->1 without replaying an update.
  Trace metric now shares an iterator yielding CP segments. Targeted CPU tests:
  77 pass (215 deselected); training state/pool suite 26 pass; memory/state 12 pass.
- Added calibration probe/config checks, not executed on GPUs. Review caught
  single-rank calibration missing NCCL overhead: probe now supports target rank
  count and rejects larger requested counts, checks CUDA identity too. Added own
  process driver-memory monitoring. Full CUDA lifecycle/performance gate remains.
- Direct manager-suite invocation exposed missing native limits in its nested
  Torch-CPU test (OpenBLAS thread creation failures); added CPU-only/native-limit
  autouse fixture. Retesting. Production Fiber spawn already uses shared limits.
- Explicit visualization detail: managed train sheets reload canonical samples
  with existing CPU augmentation; small CPU-vs-CUDA numerical differences are
  possible in visualization only. Test sheets retain original logical selection.

- Continued after user explicitly rejected premature stopping; GPU prohibition
  remains. Added reusable CPU supervisor and gated spawned cohorts. Independent
  review caught blocking large pipe handoffs, sender-lifetime and lost-finish
  reply issues: switched private controls to bounded nonblocking JSON sockets,
  opaque shared-memory CPU state and receipt acknowledgement; child identities
  are reported before GPU initialization is authorized. No manager wire changes.
- Extracted model/optimizer snapshot payload/restore for reuse by RAM handoff;
  CPU state includes optimizer/scaler, RNG and logical progress. Initial tests
  caught optimizer CPU-storage aliasing on restore; fixed with owned copies.
- Extending the existing shared CPU sample pool (not copying it): optional
  active-worker limit, park acknowledgement and nonblocking ordered polling.
  Elastic one-worker scheduling can execute both stages rather than depend on
  parked augmenters. Default fixed-pool behavior is unchanged. Tests in progress;
  full Fiber supervisor/loader/logging integration remains unfinished.
- CPU validation: supervisor lifecycle 11 pass; shared pool + state 20 pass;
  canonical broker/state/partition tests 11 pass. The actual trainer's update
  iterator matches four uninterrupted CPU updates when interrupted after two
  and resumed with physical microbatch 1 instead of 2 (effective B=5, including
  partial microbatches). No GPU measurement. Review caught a one-worker shrink
  race (publishing then blocking-getting one's own item); changed it to keep
  ownership and augment directly. Default numerical formulas are unchanged.

- User approved implementation. Added standalone draft manager protocol, pure
  scheduling/ownership engine, journalled Unix service, client and CLI. CPU tests
  cover release-before-regrant, partial CPU drain, exclusive/share/override policy,
  request changes during initialization, reconnect/restart and process liveness.
- Extracted shared LossStatistics (old trainer imports it); added bounded
  component-gradient banks and opt-in bounded_gradient_accumulation to the
  existing training loop. No change to default training. Banks reduce in three
  flat collectives rather than one collective per parameter.
- CPU tests: 17 manager tests pass; Fiber3D + accumulation suite 289 pass.
  Initial test invocation from repo root could not resolve tests.zarr_utils;
  reran successfully from vesuvius/. Commands: venv python -m pytest -q
  gpu_manager/tests (root), and python -m pytest -q
  tests/neural_tracing/test_gradient_accumulation.py
  tests/neural_tracing/test_fiber_trace_3d.py (vesuvius/).
- Independent implementation review caught lost registration-response recovery,
  malformed override type, terminal history capacity, and shutdown gaps. Added
  client-retained registration secret, strict validation, terminal compaction,
  paged status, secret redaction and graceful CLI drain. Review still ongoing.
- Not yet complete: live inventory monitoring, protocol conformance/docs and
  lifecycle hardening; full Fiber supervisor/persistent pool/relaunch integration;
  calibration and reference GPU numerical/performance validation. Protocol is
  not locked. No installation or commit has been performed.
- Follow-up: added nonblocking NVIDIA inventory monitor, draft protocol docs and
  schema, operator CLI/reference example, sticky group directory and owned/private
  recovery files, fresh process-tree discovery at registration. Manager tests:22
  pass. Real 3-rank CPU Gloo test passes, including a zero-sample rank; ran outside
  sandbox because loopback sockets are restricted. Native thread env set to1;
  unconstrained initial test startup hit OpenBLAS thread creation failures.
- GPU smoke test via existing approved caller on devices0,1, B128, microbatch16,
  40 requested updates failed BEFORE model training in VC3D set_chunk_cache_budget
  with Resource temporarily unavailable. Evidence:
  /ephemeral/hendrik/fiber3d_train2/profile_2gpu_off_group_os4_b16_sample_pipeline_async_batch_gpu_all_20260909_164044_037920/run.log.
  No throughput result. Benchmark argument file restored afterwards.
- User then prohibited all GPU use because training is running. No further GPU
  runs until explicit go-ahead; existing training is untouched. CPU-only work
  continues. Added conservative measured-memory planning/unequal-VRAM canonical
  partition helpers, not yet connected to a live Fiber supervisor.
- Independent loss review caught FP16 underflow from dense-count scaling. Replaced
  it with actual local-count normalization plus reversible FP16 local VJP retry.
  This refines the plan's scaling paragraph; no global scaler change on local
  retries. Added CPU FP16 regressions for tiny sparse gradients and local
  numerator overflow with finite normalized full-batch oracle. Actual legacy
  one-/two-branch and conditioned Fiber loss gradient tests pass (3 tests).
- A later CPU-hidden-GPU full-suite rerun stopped making progress and was
  interrupted (only this task's test session); it is NOT reported as a pass.
  Earlier full suite completed 289 tests. Focused tests are used for new logic;
  no changes/kills to the user's training or unrelated old test processes.

## 2026-09-09 — plan only

- Reviewed clean merged baseline `6009bb7f4`, local workflow, broader plan,
  relevant Fiber3D/transport/augmentation specs, loss statistics and batching.
  Replaced transient integration-task docs as required; history remains in Git.
- Planned standalone `gpu_manager/`, polled JSON Unix protocol, explicit held/
  target/granted ownership, release-before-regrant and conservative PID recovery.
- Current virtual batching retains graphs. Proposed separate component gradient
  sums bound activation memory; extra gradient storage/backward cost must be measured.
- Corrected earlier blanket GT-only claim: legacy multi-branch positive-presence
  nonempty group counts can depend on prediction-selected branches. Proposed
  accumulation preserves this without microbatch communication/guessed counts.
- Policy assumptions: exclusive reserves managed pool; otherwise one feasible
  minimum share per task, spare capacity follows priority; impossible minima queue.
  Manual map supersedes automatic/exclusive policy; pause remains authoritative.
- All implementation phases included. No runtime changes, installation, GPU tests,
  service startup, protocol lock or commit; user explicitly requested plan only.
- Independent review required by the local AGENTS.md identified five details:
  preserve canonical test groups, journal complete control state, advertise full
  cohort release, invalidate conflicting overrides atomically, and validate AMP
  component scaling against normalized gradients. Incorporated all into the plan.
- Independent re-review found no remaining planning blocker. `git diff --check`
  passes; only the four current-task planning Markdown files changed.
