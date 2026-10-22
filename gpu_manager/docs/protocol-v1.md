# Protocol 1 — LOCKED, 2026-09-10

Validated by the manager conformance/recovery suite and real Fiber S1A reference
lifecycle, calibration and gradient checks. The user replaced the extended
repeated timing gate with short tests; one full timing/lifecycle comparison had
already completed. See the [validation scope](validation.md).
Changes to wire shapes **or semantics**, even additive ones, require explicit
user authorization. A version bump is not a substitute for that authorization.

## Transport and types

Filesystem AF_UNIX/SOCK_STREAM, newline-delimited UTF-8 JSON objects. Maximum
frame size 1,048,576 bytes excluding the newline. Partial frames are supported;
at most 16 coalesced requests per read event. Maximum 128 open connections;
connections idle for 120 seconds close without releasing reservations. The
reference client sends one request at a time. Slow peers cannot block other peers.

Reject duplicate keys, NaN/infinity (including float overflow), invalid UTF-8,
unknown fields/operations, and noninteger resource quantities. Booleans are not
integers. UUIDs for tasks/epochs/requests use canonical lowercase hyphenated
strings; GPU identities are NVIDIA `GPU-...` identifiers, never local ordinals.
Bytes, milliseconds and integer logical CPUs are the wire units. Logical CPU
accounting replaces physical-core accounting by explicit user instruction on
2026-09-10; field names and framing remain unchanged.

Every request:

```json
{"version":1,"id":"REQUEST_UUID","server_epoch":"EPOCH_UUID","op":"poll","body":{"task_id":"TASK_UUID","token":"SECRET","last_generation":3}}
```

`server_epoch` is omitted for `hello`. Every connection must first complete
`hello`. Response:

```json
{"version":1,"id":"REQUEST_UUID","server_epoch":"EPOCH_UUID","ok":true,"body":{}}
```

Failure replaces `body` with `error: {code,message,details}` and sets `ok:false`.
Codes include invalid_request, invalid_json, frame_too_large, version_mismatch,
handshake_required, stale_epoch, unknown_operation, forbidden, not_found,
conflict, terminal, capacity and unavailable. Unrecoverable I/O/journal failures
stop the service; they must not continue with an unpersisted ownership ledger.

## Operations and bodies

All task operations include `task_id,token`. Controller calls to set_request and
set_paused may omit the token; the kernel-authenticated UID must have permission.

| Operation | Additional body fields | Result |
| --- | --- | --- |
| hello | versions: integer array; required_capabilities: string array | capabilities, inventory, peer_uid |
| register | task_id, token, label, root, request, reconfigure_mode:"full_release"; optional pausable, adaptable (default true) | task, token |
| poll | last_generation: integer | task |
| set_request | request (complete replacement), expected_request_revision | task |
| report | generation, phase, held, processes, detail | task |
| set_paused | paused:boolean, scope:"self" or "controller" | task |
| finish | outcome:string | task |
| status | optional after_order (default -1), limit (1–16, default16) | inventory, policy_revision, override, draining, blocked_gpu_uuids, tasks, next_after_order |
| set_override | expected_policy_revision, allocations, bound_task_id (UUID or null) | policy_revision |
| clear_override | expected_policy_revision | policy_revision |

Registration token is **client-generated, retained before sending**, 32 random
bytes encoded as 64 lowercase hexadecimal characters. This permits recovery if
the registration response is lost, including across server restart. Protect it
like a credential. The server also checks the OS peer UID; a token alone is not
sufficient. Tokens and raw completion reports are never included in status.

Process identity: `{pid,start_us,boot_us}`; timestamps derive from process creation
and host boot time in microseconds. Root ownership must be verifiable. Child
identities are kept through reparenting; process disappearance/PID reuse is not
confused with a live original process. Unverifiable identities are not reclaimed.

Request: `max_cpu_cores,max_gpus,min_vram_bytes_per_gpu,max_vram_bytes_per_gpu`
are positive integers; `allowed_gpu_uuids` is a unique list (empty means managed
pool); `priority` is a signed integer; `exclusive` is boolean.

Allocation: `{cpu_cores:N,gpus:{GPU_UUID:BYTES}}`. Zero allocation is exactly
`{cpu_cores:0,gpus:{}}`. Automatic cores are the ceiling of
`max_cpu_cores * sum(granted_vram/max_vram) / max_gpus`. Manual cores are explicit,
bounded by the request and host inventory. Partial release may have zero GPU
memory but positive CPU cores; the full old reservation stays charged until
all resources are released.

## Task state and lifecycle

Public task state includes identity/label/root/uid/order, request/revision,
generation, phase/reason, independent self/controller pause latches, target,
granted, held, terminal and last_contact_ms. `held` is the client's reported
use; `granted` reserves capacity even before initialization finishes. Neither
changing target nor closing a socket frees a grant.

Phases: waiting → initialize → applied → release → waiting; terminal finished.
Initialize/applied grants authorize use. Release permits only completing existing
work to a safe boundary and draining. No new update should begin once a release
is observed. Every changed full-release allocation requires cohort exit, even an
expansion or a change that retains some/all old UUIDs.

Reports: `applied` must match the exact current grant. `released` reports drained
resources; only the zero allocation fully relinquishes ownership. `failed` also
self-pauses, and may report remaining resources so failures cannot fictitiously
free them. Finish requires zero held/granted resources. Send pause, drain and
report release before finish to avoid automatic regrant.

Request changes while initializing update only the target. Initialization must
first report applied/failed; then a new generation requests complete release.
Changes during release coalesce targets but do not overwrite the release
generation. All donors release before a dependent grant; free unrelated capacity
can progress independently. No client needs another grant to release its old one.

Self and controller pause are independent latches. Manual splits supersede
automatic/exclusive policy but never unpause a task. An incompatible request or
pause clears the split atomically. Exclusive demand reserves the whole pool;
nonexclusive tasks first get one feasible minimum share, distinct GPUs preferred,
then priority-ordered VRAM/GPU growth. Impossible minima wait rather than overcommit.

## Authorized scheduling amendment (2026-09-10)

The user explicitly authorized this amendment after the initial v1 lock.
`task_scheduling` in hello capabilities identifies the amended semantics; the
reference client requires it and fails closed against older servers. Upgrade
client/server together: older strict response validators may reject new task
fields. Omitted registration flags and older journal tasks default to true.
No further protocol changes are authorized by this amendment.

- Immutable boolean `pausable` and `adaptable` flags are returned in task state.
  Non-pausable requires non-adaptable: full-release resizing necessarily pauses.
- Non-pausable grants are protected from initialization onwards, including server
  restart and graceful drain. Owner self-pause/release, failure recovery and
  verified process-tree death can release them. Controller pause of a live
  non-pausable task fails with `not_pausable`; pausing its startup is permitted.
- Non-adaptable tasks retain their first exact CPU/GPU/VRAM allocation across
  pauses, including GPU UUIDs. Incompatible request changes/overrides are rejected.
  Overrides cannot revoke non-pausable live grants. Release before changing the
  request's exclusivity. Explicit owner self-pause permits finishing/cancelling.
- All candidates use descending explicit priority, then FIFO registration order.
  Exclusivity has no priority bonus. Exclusive requests wait behind admitted
  higher-priority ordinary work. A higher-priority exclusive pauses lower-priority
  work only when non-pausable blockers are absent. While blocked, other pausable
  work continues, but new lower-priority non-pausable jobs cannot prolong blockage.
- Both entering and leaving exclusivity wait for outgoing release acknowledgements,
  even if unused capacity exists. Manual splits retain policy precedence subject
  to capability constraints.
- Waiting reasons are refreshed on reconciliation:
  `blocked_by_nonpausable_task:<comma-separated-task-UUIDs>`,
  `blocked_by_higher_priority_task:<comma-separated-task-UUIDs>`,
  `waiting_for_higher_priority_exclusive`, `insufficient_resources`,
  `waiting_for_release`, `waiting_for_resources`, `inventory_unavailable`, `paused`.
  Polling automatically observes admission after blockers leave.

Non-pausable clients can keep working under their existing grant on server loss;
reconnect/register/poll preserves its generation and phase. Never initialize a
duplicate workload on reconnect or interpret disconnection as release. The
reference `Supervisor` supports both capability modes, including exclusive
non-pausable startup calibration. It never creates or authorizes new workers
offline, so a calibration cohort restart waits for restored communication.

## Retries, recovery and bounds

Keep a request UUID stable when retrying an uncertain mutation. Last 256 successful
mutation requests/responses are journalled; same ID/different content conflicts.
Beyond that window use current revisions/generation from poll/status instead of
blind replay. Request and policy writes compare-and-set revisions. Report
duplicates for the same current generation/content are idempotent. After a
generation advances a stale report conflicts: poll the complete state, do not
guess whether a grant exists. Never reuse a completed task UUID.

On restart the epoch changes, dedup replies are discarded, and active reservations
are retained with a new release directive. Reconnect via hello and register using
the retained token. A task with both journaled granted and held allocations empty
has no active reservation and recovers to waiting, retaining its pause flags,
without requiring a release report. Old-epoch mutations fail. Before regrant, the client releases
old workers or the server verifies the entire known process tree is dead. EOF and
missing heartbeat are not death. A root that dies while descendants survive stays
reserved. PID discovery defaults to every 60 seconds; clients must not detach
unreported GPU workers between scans.

At most 512 active tasks. Process lists are pruned of verified-dead descendants
on acknowledged release; terminal records compact to the last32 at process scans.
The journal retains compact retired UUID tombstones to prevent resurrection;
their storage grows with lifetime task count (roughly 40 bytes/task), not GPU data.
Status pages are ordered by registration sequence and are a live view, not an
atomic multi-page snapshot. The reference client's status() collects all pages.

### User-authorized communication grace (2026-09-11)

No wire fields or allocation/release rules change. Pausable clients may continue
already-authorized work for a bounded communication grace period (Fiber:60s from
the first failed task exchange), logging actual exceptions and retrying (Fiber:5s
after a failed attempt). Successful hello alone does not reset the deadline.
At expiry they stop admitting work, unload at the next safe boundary, retain CPU
state and reconnect. They must not continue indefinitely under an unverifiable
grant. The generic Supervisor default remains immediate safe-boundary drain.
Exclusive non-pausable calibration may finish its current authorized cohort
without an outage deadline. Stop, known release directives and observed epoch
changes override grace immediately. No new workers are initialized or authorized
offline. Reconnection uses the same task/token and validates the allocation;
after a drain, join and release/regrant precede automatic resume. Transient
transport failure is not task termination and never releases reservations.

Socket locks prevent concurrent servers; missing/corrupt journal with an existing
initialized lock fails closed. Inventory changes require explicit reconciliation,
not replacing the journal with an empty one.
