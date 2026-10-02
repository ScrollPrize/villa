# Plan: graceful cooperative maintenance under allocation changes

## Lifecycle

1. Store the durable pending maintenance request in a supervisor-owned mailbox
   passed to every replaceable GPU cohort. Publish synchronously from the
   completed worker boundary, before the supervisor can observe a release.
   Deduplicate by `(kind, step, samples_seen)`; conflicting requests fail.
2. Make cohort maintenance admission idempotent and deferrable: attach the
   request immediately to block new optimizer units, but begin GPU draining
   only when the cohort is ready and is not already allocation-draining.
3. Permit an allocation-draining cohort to close even when it carried a
   maintenance request; the supervisor's handoff state remains authoritative.
4. Replacement cohorts consult the same mailbox and start held while work is
   pending. Execute only under an exact currently-applied
   `(server epoch, generation, grant)` token.
5. Bind the trace child to the exact granted GPU UUIDs, count, and VRAM rather
   than the configured maximum or assumed physical ordinals.
6. Track the cohort/token owning an active reviewed-trace child. If ownership
   is revoked (including epoch change or expired disconnect grace), stop and
   verify-reap its complete setsid process group before release acknowledgement,
   clean its transient cache, retain the request, and retry after reallocation.
   Explicit shutdown cancels without retry. A genuine child failure remains
   fatal; only manager revocation retries.
7. Write each attempt to a staging directory and atomically promote only a
   validated complete result to the canonical step directory. Preserve or
   safely clean incomplete attempts and never overwrite a completed result.
8. Reconstruct directly after success only if the exact owning token remains
   applied and is not draining; otherwise let normal supervisor release/regrant
   reconstruct from its handoff state.

## Validation

9. Add CPU-only interleaving tests covering release before discovery, during
   handoff, before child spawn, during the child, concurrent successful exit,
   changed GPU UUID/count, pause/unpause, epoch change, grace expiry, explicit
   stop, genuine child failure, retry-output collision, exact-once metrics, and
   absence of duplicated optimizer boundaries. Assert release is never reported
   while a trace GPU descendant remains alive.
10. Run the focused cooperative integration and manager lifecycle tests plus
   `git diff --check`.
11. Resume the failed production run from its update-42600 checkpoint. Force a
   pause/reallocation during a trace attempt, then verify retry, trace completion,
   and later training progress.

## Spec update

- Specify that application maintenance is pending durable state across manager
  allocation changes, must never make release unacknowledgeable, and may run
  only while its owning grant remains applied.

## Documentation updates

- Update the cooperative validation/runbook lifecycle notes to describe
  deferred maintenance and retry under pause/reallocation.

## Changelog

- Record the cooperative maintenance/reallocation race fix and production
  validation.
