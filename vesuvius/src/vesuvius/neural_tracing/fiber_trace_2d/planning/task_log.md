# Task log: resolve Fiberlet path-cost merge

## 2026-09-07

- Merge `a880b7c76bf5ff0140afee3425f6527389431ffb` was already in progress.
- The only textual conflict is in `planning/todo.md`; the two sides added
  independent todo sections, so both are retained.
- The incoming source change retains the exact cost chosen for every DP
  transition and uses those retained values when constructing the selected
  route's segment-cost decomposition.
- Independent review identified that zero-initialized unwritten cost slots were
  indistinguishable from legitimate zero-cost transitions. Source and relaxed
  states now mark the existing predecessor table when their corresponding cost
  is written, and reconstruction rejects an unmarked slot before reading it.
- The retained cost array substantially increases DP state memory. Updated the
  specification, implementation documentation, and focused assertions for DP,
  concurrent transient, and total owned-memory accounting.
- Release build command:
  `cmake --build volume-cartographer/build --target test_fiberlet_paths -j 16`
  completed successfully.
- The focused binary completed all merge-specific checks without a failure at
  their source lines, but the overall executable remains nonzero because 295
  checks fail at the unchanged line 414 legacy prepared-scoring bitwise
  equivalence helper. Neither that helper nor the compared scoring
  implementation is changed by this merge; all reported failures have that
  single pre-existing site. This unrelated test debt was not silently relaxed.
