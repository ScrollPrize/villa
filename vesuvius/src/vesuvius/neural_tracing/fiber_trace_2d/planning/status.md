# Status: explicit Z-band shared inference pipeline

- [x] Capture the user request in `task.md`.
- [x] Inspect the current coordinator, accumulator, flush, ring, and tests.
- [x] Draft implementation, validation, spec, docs, and changelog plan.
- [x] Independently review the plan against task/spec/current implementation.
- [x] Implement explicit Z-band scheduling and slot invariants.
- [x] Implement immediate completed-flush finalization and release.
- [x] Replace misleading idle polling/watchdog diagnostics.
- [x] Remove the incorrect elapsed-time/process-limit diagnosis while retaining
      the established per-worker Zarr/native thread cap.
- [x] Add deterministic regression and quiescence tests.
- [x] Update specs, docs, changelog, and task log.
- [x] Run focused validation and inspect the final diff.
- [x] Capture and diagnose the production rerun's mid-band all-idle deadlock.
- [x] Independently review and reject the unsupported pipe-transport diagnosis.
- [x] Restore atomic normal-input read submission while keeping live fetch upstream.
- [x] Keep queue transport and add exact worker/task/slot ownership checks.
- [x] Add local/live multi-window exact-output regression coverage.
- [x] Re-run focused Lasagna and Fiber validation.
- [x] Finish docs/spec/changelog reconciliation for the scheduler repair.
- [x] Capture the production coordinator with Python-aware stacks and locals.
- [x] Prove the deadlock is frontier capacity inversion, not lost IPC.
- [x] Remove the unproven completion ACK/retry experiment.
- [x] Update the plan with a canonical-frontier capacity invariant.
- [x] Independently review the capacity-invariant amendment.
- [x] Implement frontier slot reservation and immediate invariant failure.
- [x] Add delayed-frontier/full-window regression coverage.
- [x] Re-run focused shared and live-cache validation; record broader-suite blockers.
- [x] Reconcile specs, docs, changelog, and task log with the proven cause.
