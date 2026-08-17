# VC3D short line-segment task log

## Findings

- Current strip subdivision already retains every original endpoint and keeps a
  short span as one interval.
- Current strip scale is derived from total arclength divided by total interval
  count, so mixed short/long spans distort one another's displayed pitch.
- Current generated-click replacement uses `abs(linePosition difference) <=
  0.5`, which depends on producer point density rather than physical arclength.
- The local control update API supports one inserted/moved control, not several
  simultaneous removals.
- Branch links are indexed by the live control vector and therefore require an
  explicit old-to-new index remap during a multi-control collapse.
- Independent review identified that strip supports are optimized line points,
  the maximum-control-distance gate also needs physical arclength conversion,
  and multi-collapse needs explicit metadata ownership, branch remapping,
  adjacent dirty spans, and asynchronous rollback state.

## Deviations

- The private Qt controller has no isolated interaction-test target. The pure
  collapse operation and its old-to-new mapping are unit tested, reciprocal
  branch synchronization was reviewed and compiled in the full VC3D target,
  but modal confirmation and asynchronous UI orchestration are not directly
  exercised by an automated test.

## Validation

- Built with all 32 cores:
  `cmake --build volume-cartographer/build --parallel 32 --target test_lasagna_line_view_surfaces test_fiber_slice_geometry test_line_annotation_generated_views VC3D`
- `test_lasagna_line_view_surfaces`: 22 test cases passed.
- `test_fiber_slice_geometry`: 10 test cases passed.
- `test_line_annotation_generated_views`: 76 test cases passed.
- `git diff --check`: passed.
