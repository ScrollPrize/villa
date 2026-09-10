# Protocol 1 — LOCKED, 2026-09-10

Validated by the manager conformance/recovery suite and real Fiber S1A reference
lifecycle, calibration and gradient checks. The user replaced the extended
repeated timing gate with short tests; one full timing/lifecycle comparison had
already completed. See the [validation report](../../vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/cooperative_validation.md).
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
Bytes, milliseconds and integer physical cores are the wire units.

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
| register | task_id, token, label, root, request, reconfigure_mode:"full_release" | task, token |
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
the retained token. Old-epoch mutations fail. Before regrant, the client releases
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

On server loss clients unload at the next safe boundary, retain CPU state and
reconnect; they must not continue indefinitely under an unverifiable grant.
Socket locks prevent concurrent servers; missing/corrupt journal with an existing
initialized lock fails closed. Inventory changes require explicit reconciliation,
not replacing the journal with an empty one.
