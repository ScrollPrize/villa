# Task log

## 2026-09-18

- Began an inventory of current benchmark families, canonical configs, approved
  runner modes and detailed result reports. This task is documentation-only and
  will not launch evaluation workloads.
- Independent plan review identified missing taxonomy, immutable-evidence,
  lifecycle and completion-check details. Expanded the plan before writing docs.
- Added `docs/benchmark_runbook.md` as the central index. It separates the
  canonical five-scroll scientific evaluation, frozen failure-enriched short200,
  current beam/fiberlet comparison, anchor/mini5/contrast diagnostics,
  performance profiles and historical model/embedding evidence.
- Explicitly retained short200 v3 as immutable evidence and marked the later
  ground-truth review directory as an unpromoted annotation-audit draft.
- Linked the runbook from the live-inference, short-failure, fiberlet and anchor
  entry docs; added durable specification and changelog entries.
- Validation:
  - runner `--help` succeeded and exposed every documented mode;
  - all 40 Markdown links resolve in the worktree;
  - all 21 absolute JSON configs named by commands exist and parse;
  - initial focused pytest collection lacked the repository `PYTHONPATH` and
    failed before running tests;
  - rerun with `PYTHONPATH=/home/hendrik/villa2/vesuvius/src:/home/hendrik/villa2`
    passed 21 tests in 2.89 seconds.
- No benchmark, inference, rendering or GPU workload was launched.
