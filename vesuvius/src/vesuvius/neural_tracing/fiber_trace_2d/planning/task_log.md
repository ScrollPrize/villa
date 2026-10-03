# Resident training trace evaluation

- Committed prior evaluation/config changes as e5245a30c.
- Root cause: old live supervisor launched updated Python evaluator expecting
  a new job field. Training stopped after saving step 51300. No restart yet.
- Current architecture unnecessarily drains/reconstructs the whole GPU cohort
  for each trace. Replace that with callable evaluation and resident replicas.
- Independent plan review completed; addressed borrowed-worker ownership,
  cancellation-before-buffer-release, preserved training state and VRAM limits.
- Extracted shared preparation/audit; resident tile inference reuses the existing
  kernel and buffer scheduler. Training model/optimizer remain resident.
- Explicit limitations: unmanaged debug training must disable reviewed trace
  (removed its broken extra-model subprocess fallback); resident tests cannot
  load a second, different embedding checkpoint. Native C++ tracing and CPU
  multiprocessing remain unchanged, not ported/duplicated into Python.
- First CPU suite: 139 passed, 4 skipped; two failures in unrelated existing
  cache tests (batch-size cache compatibility and legacy scratch cleanup).
- Single-GPU resident state test passed (4.68s) through manager on GPU0/1 grant;
  two-rank lifecycle validation in progress. No training restarted yet.
- Native cold/warm runner passed through both owned and resident executors
  (2 tests, 26.19s). Actual two-rank training → resident tile inference → same-PID
  training → one-rank restore matched the uninterrupted reference (21.77s).
- Final review caught and fixed pre-attempt revocation (no evaluator exists to
  finish a drain) and pipe startup ordering. Tiny command precedes the work
  unit; large config/slot descriptors follow only after workers can read.
  Tests now send >512 KiB descriptors, exceeding pipe capacity.
- Native/reference/accumulator CPU children are registered with the existing
  stop guard, so removing the old enclosing process does not orphan them on
  interrupted cleanup. Final suites rerunning with these lifecycle checks.
- The >512 KiB test exposed a second startup dependency: cohort commands flush
  only during supervisor polling. Made descriptor sends asynchronous and made
  start-unit a supervisor-thread request (rather than touching its channel from
  the evaluation thread). The failed fixture is bounded by the test runner's
  300s timeout; no production training was running during this regression.
- Final validation command: `/home/hendrik/villa/vesuvius/scripts/run_fiber3d_cooperative_tests`.
  Results: `/ephemeral/hendrik/fiber3d_train2/cooperative-tests/20261003_202623_196625/results.json`.
  CPU/API: 45 passed, 3 skipped (28.29s); native cold/warm replay: 2 passed
  (25.75s); resident CUDA state: 1 passed (3.90s); two-GPU lifecycle: 1 passed
  (21.90s). GPUs 0–1 only via manager; peak owned VRAM 1.47 GiB/GPU.
- Fixtures: deterministic 8³ training patches and 32³ synthetic CT/normal/fiber
  replay, existing optimized native binary in this checkout. Resident products
  matched expected values; optimizer/model/EMA after continuation matched the
  uninterrupted reference (existing rank-change tolerances 1e-5/1e-6).
  No production-32-fiber performance measurement or training restart is claimed.
- Broader suite separately ran 139 passed/4 skipped with two failures in
  untouched cache semantics: `test_cache_extension_validates_parameters_not_code_hashes`
  expects batch-size rejection although current implementation allows it;
  `test_legacy_stale_scratch_is_cleaned_on_resume` expects unknown files inside
  abandoned sessions retained although current cleanup deletes the session.
  Those unrelated behaviors were not changed as part of this task.
