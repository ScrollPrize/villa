# Task Log: bounded fiberlet replay comparison

## Findings

- The CLI currently selects a full interval for tube extraction, while greedy
  and graph replay independently derive their own full-reference ends.
- Limiting extraction alone would be incorrect: both engines would continue
  matching and reporting fractions against the unbounded tail.
- `--along` is already reserved for optional failure-visualization half-windows
  and must remain independent from the comparison interval.

## Plan Review

- The selected interval must be represented once at the CLI boundary and its
  effective absolute end passed to extraction and both engines.
- A selected end can fall inside a graph edge. Such a result needs an explicit
  partial-terminal-edge state; otherwise retained full-edge metadata would
  falsely imply completion at an anchor.
- At the boundary, distance failure takes precedence over successful interval
  completion. No later samples from that fiberlet may be evaluated.
- Artifact publication must validate both engines against the selected interval
  and persist only the exact sliced reference geometry.
- Focused tests must cover bounded preprocessing, both boundary outcomes,
  publication, and unchanged full-tail behavior when `--length` is omitted.

## Deviations

- None.

## Validation

- Built with 32 jobs:
  `cmake --build volume-cartographer/build -j32 --target vc_fiberlets test_fiber_trace3d test_fiberlet_paths test_fiber_replay`.
- `test_fiber_trace3d`: 51 cases passed.
- `test_fiberlet_paths`: 40 cases passed.
- `test_fiber_replay`: 4 cases passed.
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src python -m pytest -q vesuvius/tests/test_view_fiber_presence.py`:
  55 tests passed. Plugin autoload was disabled because the host's unrelated
  global pytest plugin imports the absent `zarr.testing` module before test
  collection.
- Confirmed `vc_fiberlets --help` documents replay-only `--length N` in base
  voxels with a full-reference default.
