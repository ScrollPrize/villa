# Training-integrated reviewed failure benchmark

- Merged the committed `fiber-exp5` reviewed-benchmark base (`00eb345bd`),
  including native v4/editor-hint parsing, rather than maintaining a local port.
- The first periodic integrated trace reached update 35400 but failed against
  the stale pre-merge native binary. Rebuilt from the merged source and advanced
  the resume pointer to the already-written update-35400 checkpoint.
- The next periodic trace exposed a second integration defect: it launched the
  standalone evaluator while the training model and optimizer remained GPU-
  resident, exhausting the shared allocation. The initial two-task handoff was
  also wrong: overlapping process trees forced unnecessary release/re-register
  transitions. Cooperative training now retains one task, generation, grant,
  and canonical CPU broker; at the trace boundary it unloads only GPU training
  workers, runs the four-tile benchmark as a descendant under the held grant,
  and restores GPU workers from the in-memory handoff.
- Live update-35500 validation kept task
  `1704364f-b428-41e9-a075-8648c3d76216`, generation 1, and the six-GPU grant
  applied throughout. The trace completed with 22 combined restarts / 24.816457
  implied mm, then training advanced through update 35600. That trace also
  completed under the same task (17 combined restarts / 31.709917 implied mm)
  and began reinitializing training without releasing the allocation.

- Reviewed-v3 contains 91 truncated eligible pieces / 182 directions, derived
  from 46 source-run identifiers and 36 original source fibers.
- Canonical union over `lowres`, `fullres`, `gaussian`, `halfsigma_raw`, and
  `halfsigma_ema99` contains exactly 32 pieces. PHerc0332 and PHerc1451 have no
  members in this selected hard subset.
- The subset is intentionally failure-enriched and must not be reported as an
  unbiased model-quality estimate.
- Plan self-review: preserve existing dense metric semantics, reuse the shared
  live-inference/native runner, keep exact hash-locked membership, maintain DDP
  responsiveness during child execution, and fail on incomplete traces.
- Process deviation: independent sub-agent review is unavailable because the
  active agent policy disallows delegation without explicit user instruction.
- Materialized `/home/hendrik/data/2026-10-01_reviewed_failure_union_v1`:
  11 PHerc0139 + 6 PHerc1299 + 15 PHercParis4 = 32 pieces. Manifest validation
  checks exact membership and SHA-256 before dense loading and native tracing.
- The first production launch exposed that a merge had lost the existing v4
  fiber/editor-hint parser support. Restored the shared parser behavior rather
  than rewriting the cohort; the v1 dataset remains byte-identical to source.
- Resume keeps exact training-refinement membership locked, while permitting a
  replacement deterministic test cohort to add independent test parents to a
  restored store. Historical test entries may remain but cannot affect current
  test sampling or metrics.
- Added the reviewed32 training config. Its ordinary raw/EMA/refinement/cross-
  section tests use only the locked subset without changing metric semantics.
- Native tracing reuses `run_classic_fiber_evaluation.py` and the shared live
  inference/native replay implementation through its restricted
  `training-full` inherited-allocation mode. It never registers another manager
  task; the supervisor's existing grant covers the child process. It evaluates
  current raw weights, 64 directions, at the configured test interval.
- Startup and periodic evaluation now use the same trace-test lifecycle.
  Startup traces use the just-written raw `best.pt`; periodic traces use
  `current.pt`, so TensorBoard metrics exist from the initial test onward.
- DDP ranks remain in a one-second collective poll while the rank-zero child
  uses all manager-assigned GPU UUIDs. Startup/completion failures are
  broadcast before raising, avoiding a stranded rank. The dedicated transient
  prediction namespace is removed after every success or failure.
- Metrics retained: forward/reverse/combined restart episodes, reference mm,
  and `reference_mm/(restarts+1)` implied mean, plus wall time and all ordinary
  runner evidence.
- Validation: reviewed/script and cooperative lifecycle tests `17 passed, 2
  skipped`; gradient-accumulation and
  training-state tests `20 passed, 12 skipped`. A first sandboxed invocation
  could not create Gloo/forkserver sockets; the approved unsandboxed rerun
  passed. `py_compile` and `git diff --check` passed.
