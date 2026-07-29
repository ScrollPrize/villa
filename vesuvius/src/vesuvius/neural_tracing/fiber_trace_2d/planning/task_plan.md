# Plan: Python Native 3D Trace2CP Whole-Fiber Start CP

## Implementation

1. Add `--whole-fiber-start-cp-index` to the Python native 3D Trace2CP CLI.
2. Thread the value through `run_native_trace2cp(...)`,
   `run_native_trace2cp_many(...)`, and `trace_native_3d_whole_fiber(...)`.
3. In the whole-fiber trace core:
   - validate that the selected start CP is in `[0, cp_count - 2]`;
   - initialize current point/direction from that CP;
   - trace consecutive CP pairs from the selected CP to the final CP;
   - compute total and partial reference lengths relative to the selected CP.
4. Reject `--whole-fiber-start-cp-index` outside whole-fiber `--fiber-json`
   mode so it is not silently ignored.
5. Include selected start/final CP indices in stdout/summary metadata.

## Spec Update

- Document the new whole-fiber start CP argument and suffix-denominator
  semantics.

## Docs Update

- Update `docs/code_structure.md` with the new CLI usage.

## Validation

- Add tests for CLI parsing, suffix tracing, denominator behavior, and invalid
  start CP rejection.
- Run the 3D neural tracing test file and `git diff --check`.
