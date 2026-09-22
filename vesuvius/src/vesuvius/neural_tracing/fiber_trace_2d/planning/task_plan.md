# Task plan: unlinkable inference scratch

## Implementation

1. Add a bounded cleanup helper that scans only known channel `.live_tmp`
   directories and removes direct `session-*` directories after the cache
   writer lock is held.
2. Make `_FileMappedSlotPool` use Linux `O_TMPFILE` and a parent-owned
   `/proc/<pid>/fd/<fd>` descriptor path for spawned workers. Close the
   descriptor only after the pool and workers are finished. Retain the named
   tempfile fallback elsewhere.
3. Update focused regression tests and document the lifecycle.

## Spec update

The cache lifecycle now requires stale-session cleanup under writer ownership
and anonymous scratch backing for new Linux slot pools.

## Docs updates

Record the unlink handoff, fallback behavior, and the fact that old sessions
are still explicitly cleaned in the task log and inference scratch docs.

## Testing

- compile the changed Python modules;
- run the file-mapped spawn/unlink test;
- run the stale-scratch cleanup regression.
