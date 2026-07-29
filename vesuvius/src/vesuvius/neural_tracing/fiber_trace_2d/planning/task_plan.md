# Plan: Python Native 3D Trace2CP Multi-Fiber Input

## Implementation

1. Change the CLI parser so `--fiber-json` accepts one or more paths while
   preserving no-fiber and single-fiber behavior.
2. Add an output-stem argument to the existing single-fiber runner. Default
   remains `trace2cp_native_3d` so current output filenames do not change.
3. Add a multi-fiber wrapper that:
   - rejects sample-index and explicit CP selectors for multiple JSON files;
   - loads the checkpoint/model once;
   - runs the existing whole-fiber path once per JSON;
   - writes `trace2cp_native_3d_000_summary.json`,
     `trace2cp_native_3d_001_summary.json`, etc.;
   - writes matching indexed JPGs when `--vis` is enabled;
   - writes `trace2cp_native_3d_summary_all.json`.
4. Compute the accumulated score from summed restarts and summed original-line
   reference length, not by averaging per-fiber rates.

## Spec Update

- Document multiple `--fiber-json` paths, whole-fiber-only semantics,
  indexed per-fiber outputs, shared model reuse, and accumulated restart-rate
  reporting.

## Docs Update

- Update `docs/code_structure.md` for the multi-fiber CLI behavior and output
  filenames.

## Validation

- Add focused tests for CLI parsing, indexed output stems, and aggregate metric
  math.
- Run the focused 3D Trace2CP test subset.
