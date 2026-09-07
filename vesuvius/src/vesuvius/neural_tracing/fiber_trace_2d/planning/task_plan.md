# Plan: resolve Fiberlet path-cost merge

## Resolution

1. Preserve both independent additions in `planning/todo.md` and remove the
   conflict markers.
2. Review the incoming DP transition-cost retention against the current path
   reconstruction and memory-accounting code.
3. Confirm that exact per-transition costs reproduce the selected DP objective
   without recomputation or numerical relaxation.
4. Stage the resolution and complete the merge commit.

## Tests

1. Build the focused `test_fiberlet_paths` target in the existing optimized
   build.
2. Run the focused test binary and report any remaining test gaps.

## Spec update

- Specify that reconstruction retains the exact five-component transition cost
  alongside every predecessor and include that array in state-memory
  accounting.

## Docs updates

- Update `docs/fiberlets.md` with the retained-cost reconstruction and memory
  contract.

## Changelog update

- No additional changelog entry is required beyond the merged commit history.
