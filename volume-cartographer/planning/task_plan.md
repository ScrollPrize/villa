# Plan

1. Use the existing cache mutex for every derived-cache reset. Snapshot normal
   and validity matrices under that mutex, including the validity fast-path
   flag, before reading outside the lock. Do not serialize whole renders.
   Use a private validity snapshot helper shared with validMask(); never nest
   acquisitions of the non-recursive cache mutex. ensureLoaded() precedes cache
   locking; any nested locks retain load-mutex then cache-mutex ordering.
2. Keep point storage, geometry mutation, channel I/O, scheduling and numerical
   algorithms unchanged. Concurrent point edits/unloading are not made safe by
   this cache-only fix.
3. Add a focused concurrent invalidation/render test with exact output checks,
   including repository TIFFXYZ fixtures. Exercise cache rebuilds and retained
   snapshots; test the unpatched implementation as a negative control.
4. Build with 32 jobs and existing dependencies only. Run focused QuadSurface
   tests and repeated offscreen RPC smoke runs, including four-CPU runs.
5. Obtain independent plan and code review. Present the PR draft for approval;
   do not create the PR before approval.

## Specification Updates

Document derived-cache snapshot lifetime and invalidation requirements in
specs/rendering.md. No render cancellation or geometry-edit contract change.

## Documentation Updates

Correct cache ownership and synchronization comments in QuadSurface.hpp.
Record reproduction, validation and remaining limitations in task_log.md.

## Testing And Validation

Compare coordinates, normals and masks against a serial baseline; include
concurrent renders and cache clears. Reuse the actual smoke test and record
failures rather than increasing its timeout. A stress test is not a proof of
all schedules; source-level locking review is also required.
Preload fixtures, clone serial render baselines (gen returns TLS views), and
compare exact finite values and NaN positions. Include all-valid components,
invalid points, strict validity, normals omitted/requested and depth offsets.
Only cache-only operations run concurrently. Geometry invalidation, lazy loading,
channel access and point eviction remain excluded from the concurrent contract.

## Changelog Update

Add one dated summary line after validation.
