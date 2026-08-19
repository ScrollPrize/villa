# Task log: Native Trace2CP benchmark record

## Findings

- Historical task records at commit `dea2a8ebb` state that the benchmark file
  was created, but it is absent from that commit and repository history.
- The exact missing file remains in the sibling `villa2` worktree at the
  expected path; `villa` has no copy. Its content matches the user-supplied
  Markdown and replaces the incomplete reconstruction verbatim before table
  extension.
- The supplied multi-fiber lines provide the new authoritative aggregate row.
- Both runs used code revision
  `9095ba351b6fea1c37253190cedddab5c97373f6`, the `fiber-lets` head before the
  benchmark-document commit and subsequent `fiber-lets2` merge.
- The new row uses the existing document's aggregate-only schema; final-fiber
  diagnostics and the malformed carriage-return block token are not added.
- The second supplied run uses the same snapshot and inference scaledown as the
  first new run, with `--step-voxels 32` instead of `--step-voxels 8`.

## Deviations

- None.

## Validation

- Copied the exact historical content from the sibling `villa2` worktree,
  extended its result/snapshot/settings tables, and added a discoverability
  link from `docs/code_structure.md`.
- Verified aggregate shell arithmetic: `real=7904.174s` and
  `user+sys=4556.251s` (`75m56.251s`).
- Verified the step-32 aggregate shell arithmetic: `real=8171.426s` and
  `user+sys=5833.624s` (`97m13.624s`).
- Verified the new scale-0 config parses as JSON.
- `git diff --check` passed.
- No runtime code changed and no benchmark was rerun.
