# VC3D render attribution and lookup performance repair

Repair the synthetic rendering performance gate before re-enabling it, then
recover the lookup performance lost around the render-order changes.

Priorities:

1. Correct and stabilize the passive Valgrind attribution. Collect periodic
   Callgrind cost slices and a DRD dependency graph from separate executions
   with identical Valgrind scheduling parameters. Reconstruct logical worker
   traces canonically instead of equating raw worker IDs, trim DRD to the same
   existing timed render boundary, and reject any pair whose logical pattern is
   ambiguous or not reproducible. The measured binary must remain unchanged:
   no markers, affinity mode, deterministic executor, or added instructions.
2. Measure and repair renderer lookup speed using the commit immediately before
   `Vc3d renderorder` and the current implementation as an A/B case. Preserve
   rendered bytes, fallback behavior, request priority, and scheduling
   semantics.

Speed investigation requirements:

- Treat `ChunkRequestContext` as render-job-constant; verify and exploit that
  without changing request publication semantics.
- Investigate a prepared/source-bound key lookup path so level, source, and
  request information are not repeatedly assembled in the hot lookup.
- Avoid rebuilding and probing a full key when a correlated sample remains in
  the same successfully resolved chunk.
- Account for the existing `LocalChunkCache`: it already retains the last key
  and result and up to eight pinned chunks. Do not duplicate that cache under a
  different name; improve the work that happens before it can identify a hit.
- Establish repeatable before/after Release measurements and verify exact output
  checksums before accepting any optimization.

The Valgrind attribution repair is first. Lookup optimization remains planned
until the gate can produce stable, semantically valid scores.
