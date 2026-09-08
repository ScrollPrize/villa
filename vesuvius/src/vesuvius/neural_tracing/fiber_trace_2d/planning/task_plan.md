# Plan: integrate current main

## Integration

1. Fetch and identify the current `origin/main` and branch divergence.
2. Merge `origin/main` without rewriting the branch's published history.
3. Resolve conflicts by preserving the Fiberlet branch behavior while
   incorporating upstream fixes and current interfaces.
4. Review the resulting diff for unresolved markers, accidental deletions,
   and incompatible build/test registrations.
5. Commit the completed merge.

## Tests

1. Build the directly affected Volume Cartographer targets in the existing
   Release build.
2. Run focused FiberTrace/Fiberlet tests selected from the actual merged diff.
3. Report any unrelated or pre-existing failures without relaxing them.

## Spec update

- No behavior change is intended by the integration itself. Update specs only
  if conflict resolution changes or clarifies an existing contract.

## Docs updates

- Preserve compatible documentation from both histories. No new user-facing
  documentation is expected unless a conflict requires reconciliation.

## Changelog update

- The merge history records the integration; no separate changelog entry is
  expected unless conflict resolution introduces new behavior.
