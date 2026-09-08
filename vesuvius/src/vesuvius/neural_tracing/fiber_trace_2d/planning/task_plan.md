# Plan: integrate current main

## Integration

1. Fetch and identify the current `origin/main` and branch divergence.
2. Merge `origin/main` without rewriting the branch's published history.
3. Resolve conflicts by preserving the Fiberlet branch behavior while
   incorporating upstream fixes and current interfaces.
4. Audit every path changed on both sides, with focused semantic review of
   FiberTrace, ChunkCache, Lasagna sampling/optimization, CMake registrations,
   tests, Python entry points, and durable planning/specification files.
5. Compare the merge result against both `origin/main` and the pre-merge branch
   for overlapping files, then check unmerged entries, conflict markers,
   whitespace errors, accidental deletions, and unrelated-file inclusion.
6. Preserve durable spec/changelog additions from both histories without
   importing unrelated upstream active task/status records into this task.
7. Commit the completed merge.

## Tests

1. Regenerate the existing Release CMake build and build `vc_fiber_trace_chunk`,
   `vc_fiberlets`, and the directly affected test targets.
2. Run `test_fiber_trace3d`, `test_chunk_cache`, Lasagna normal/line-optimizer
   tests, `test_fiberlet_paths`, `test_fiberlet_storage`,
   `test_fiberlet_crop_trace`, and available reference-replay tests.
3. Run an existing deterministic Fiberlet smoke fixture covering selection,
   cost, and termination behavior if one is available in the built tests.
4. Report any unrelated or pre-existing failures without relaxing them.

## Spec update

- No behavior change is intended by the integration itself. Update specs only
  if conflict resolution changes or clarifies an existing contract.

## Docs updates

- Preserve compatible documentation from both histories. No new user-facing
  documentation is expected unless a conflict requires reconciliation.

## Changelog update

- The merge history records the integration; no separate changelog entry is
  expected unless conflict resolution introduces new behavior.
