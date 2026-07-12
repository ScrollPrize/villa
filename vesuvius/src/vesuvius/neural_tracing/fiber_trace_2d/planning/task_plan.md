# Trace2CP Iterative Fused-Trace Refinement Plan

## Scope

- Add opt-in iterative refinement to Trace2CP runner exports.
- Preserve the existing default single-pass behavior.
- Keep the implementation local to Trace2CP segment tracing and the shared
  volume-sampling loader path.

## Implementation

- Add a CLI integer flag `--trace2cp-refine-iterations`, default `0`.
- Treat the initial Trace2CP evaluation as iteration `0`; each requested extra
  pass becomes `it1`, `it2`, and so on.
- Add a loader helper that builds a Trace2CP segment source from an explicit
  refined patch-space trace:
  - sample the previous source grid at the smoothed fused trace `(x,y,z)` to
    recover volume-space centerline points;
  - sample the previous source offset axis at the same points for Lasagna-style
    normal/frame input;
  - build a fresh side-strip grid from those centerline points and normals;
  - preserve original start/target CP metadata and x positions;
  - keep endpoint context before the start CP and after the target CP so the
    refined pass behaves like Trace2CP on an independent line source in both
    directions.
- Add a runner helper that smooths fused trace points with a finite Gaussian
  kernel without moving the first or last endpoint or changing x columns.
- Reuse the existing `_evaluate_trace2cp_pair` scoring logic by allowing it to
  accept a prebuilt segment source.
- Write extra single-pair outputs as `trace2cp_vis_it1.jpg`,
  `trace2cp_summary_it1.txt`, etc.; keep the current un-suffixed initial
  output for compatibility.
- For whole-fiber Trace2CP, evaluate the same refinement chain per pair and use
  the final iteration for the long-strip aggregate, while writing optional
  pair-local iteration images/debug summaries under a refinement subdirectory.

## Spec Update

- Document iterative Trace2CP refinement, the CLI flag, output naming, the
  Gaussian smoothing semantics, and the requirement that refined passes
  resample from volume coordinates rather than warping previous images.

## Docs Updates

- Update `docs/code_structure.md` Trace2CP runner/loader sections.
- Update changelog, status, and task log for this task.

## Tests

- Add unit tests for:
  - fused-trace Gaussian smoothing preserving endpoints and x columns;
  - explicit-source refinement using the supplied source instead of rebuilding
    from the dataset descriptor;
  - output naming for iteration suffixes where practical.
  - refined source endpoint context keeping both forward and reverse trace
    starts valid.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/loader.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
