# Integration and exclusive profiling plan

1. Archive bbox task; inspect branch topology and all dirty files. Checkpoint known task changes in subproject-separated commits before merge. Preserve incoming planning history in docs.
2. Independently review plan and incoming manager lifecycle/ownership changes. Merge fiber-training-embedding without rebasing or discarding history. Resolve explicit conflicts preserving inference cleanup/watchdog safety, new auto-start/socket/logical-CPU behavior and embedding/training features.
3. Use one shared exclusive manager interface; adapt both existing callers rather than retaining duplicate launchers. Verify grant covers all requested GPUs and required CPU budget, worker ownership excludes manager infrastructure, external GPU occupancy prevents contaminated admission, and release happens only after workers exit.
4. Run manager/supervisor/launcher CPU regressions, native bbox and inference regressions, and embedding/training affected tests. Revalidate installed extension identity; use the corrected binding without silently overwriting unrelated work. Update binding only by normal approved project rebuild if needed.
5. Run existing approved short profile in exclusive manager context. Record exact grant, workload/build/module identity, stage timing, backlogs and observed GPU occupancy. Do not bypass manager or mutate journals to force admission. Report throughput and current bottleneck; distinguish reference-prefetch profile from full native-trace run. Restore diagnostic flags.
6. Independent integration review, logs/specs/docs/changelog/status and merge commit. Preserve pending/performance limitations explicitly.

## Spec update

Document unified manager launch semantics, logical CPU inventory, infrastructure exclusion, exclusive ownership/cleanup and no unmanaged benchmark fallback. No model precision, predictor or native bbox semantic changes.

## Docs updates

Archive incoming and current task histories; record merge conflicts/resolutions, manager review, exact commands, tests and measured profile in a new integration report.

## Validation/changelog

Tests for both exclusive launcher call paths and orphan/idle/bootstrap handling; bbox/regression tests; managed benchmark with resource-release check. Changelog records merge and measured outcome, not assumed speedup.
