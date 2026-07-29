# Python Native 3D Trace2CP Whole-Fiber Start CP

Allow the Python native 3D Trace2CP whole-fiber metric to start at a selected
control point so smaller failing suffixes of a fiber can be tested quickly.

Requirements:

- add a CLI argument for whole-fiber start CP selection;
- keep existing `--start-cp-index` / `--target-cp-index` as explicit
  single-segment selection, not whole-fiber suffix selection;
- trace from the selected CP through the final CP;
- compute the restart-rate denominator over the original line length from the
  selected CP to the final CP;
- reject invalid start CP values that do not leave at least one segment;
- preserve default CP0 behavior.
