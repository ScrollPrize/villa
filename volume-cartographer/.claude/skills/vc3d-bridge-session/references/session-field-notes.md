# Session field notes

## Discovery and build identity

`vc3d_ping` is authoritative for the responding pid, executable, revision,
application version, and protocol. A build SHA is stamped at CMake configure
time, so reconfigure before making revision-sensitive evidence.

Discovery records under `~/.vc3d/agent_bridge` contain pid, server name,
resolved socket path, and start time. Stale records are ignored after their pid
is proven dead. Do not remove another process's record.

## MCP and RPC names

Skills and agents use MCP names such as `vc3d_get_state`; bridge tests use RPC
names such as `state.get`. Do not transliterate. The generated contract owns
the mapping, parameter renames, defaults, enums, and bounds.

`rpc.describe` is the only bridge method without an MCP tool.
`vc3d_wait_job` is the only MCP-only convenience.

## Live objects

Viewer ids are monotonic and process-local. Match on `surfName`, then use the
fresh `viewerId`. Opening data and rebuilding workspace panes can invalidate a
cached id.

`vc3d_get_state` uses `vpkg`, `volume`, `activeSurface`, `viewers`, `job`, and
`jobs`. Inspect returned keys instead of guessing aliases.

## Jobs

Concurrency is per source, not global. A second job for the same source fails
`-32004`; different sources may overlap. Poll the original id. Cancelling an
MCP wait does not cancel the underlying application job.

Some deferred calls, including manifest attachment and Spiral file transfers,
wait for application signals but are not jobs. Their RPC response is the
completion record.

## Mutation order

Resolve every selector before mutating. For editing, confirm the active segment
and editing session, ensure no conflicting growth job is running, then perform
the mutation and save. Disabling editing may itself trigger persistence.

Path-based calls see the VC3D process's filesystem. Record absolute input and
output paths in evidence.
