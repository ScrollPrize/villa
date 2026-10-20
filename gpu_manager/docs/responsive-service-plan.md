# Responsive service implementation plan

## Task
Keep process scans and full-state/history work out of the socket-serving process;
use real OS processes, not Python threads. Preserve locked wire semantics and
durable release barriers. Do not restart the live manager during development.

## Design
- Collect one OS process-table snapshot per scan; reuse PID/parent/identity/UID
  lookups for every task, including orphan descendants and manager exclusions.
- Run collection asynchronously in a dedicated process. Missing/inaccessible or
  stale evidence cannot retire newer registrations or release live reservations.
- Put Engine dispatch, policy/history traversal and journal writes into a separate
  state-owner process. The public socket process only multiplexes bounded frames
  and forwards requests/replies; it never scans process tables or iterates tasks.
- Share the existing Engine and framing/authorization implementation rather than
  introduce a second protocol implementation. Keep per-connection ordering and
  bounded backpressure. Child death fails closed without erasing the journal.
- Process discovery is separate from the state owner too, so a slow scan does not
  prevent it replying to status/poll. Synchronous durable mutation acknowledgements
  remain ordered after journal writes; do not acknowledge unpersisted grants.
- Remove unnecessary full-state transaction copies/reconciliation for read-only
  poll/status. Schedule observation-based policy changes on observation/mutation,
  not on every client poll.

## Validation
CPU-only synthetic inventory tests: slow collector while status/poll continue,
PID reuse, inaccessible processes, child discovery, death after root exit,
stale snapshots, worker failure, restart/journal retention, request ordering and
bounded queues. Run existing manager conformance suite. Measure single-snapshot
discovery against recorded152-process baseline5.21/5.41/5.73s. No GPU workloads.

## Docs/spec/changelog
Document process architecture, failure behavior and measured results in manager
README/report. No wire fields or ownership semantics change. Any unsupported or
deferred item must be recorded explicitly before handoff.

## Completion
Implemented and independently reviewed. CPU suite: 122 passed in 31.16s.
Full collection/ownership scan: 67.79–74.36ms on 153 tracked identities versus
5.21–5.73s previously on 152. See the validation report for methodology and
the remaining serialized policy/durability limitation. No live restart performed.
