# Plan

1. Record branch tips and inspect incoming code/features and overlap.
2. Merge without committing; keep HEAD versions of task.md,task_plan.md,status.md,task_log.md. Combine changelog and specs semantically; retain other incoming plans/docs/features.
3. Review demand inference/cache changes, replay/embedding and training-span changes together. Resolve conflicts preserving existing aligned visualization/metrics and incoming GPU sweeps/embedding/span eligibility.
4. Independent plan/integration review; existing native build/inspection tests and relevant incoming tests. Do not run training or evaluation. Inspect staged diff and commit merge with results.

## Spec update
Union both specifications except contradictory details resolved against actual retained behavior; no protocol/default changes beyond incoming committed features.

## Docs updates
Fused changelog. Merge-specific conflict/validation log here, keeping current task planning unchanged as requested.

## Validation
No conflict markers/unmerged paths; preserved task files byte-identical to original HEAD. Native build/replay and Python integration tests for changed paths. Check both branch tips are ancestors of final merge.
