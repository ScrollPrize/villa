# Task Plan: Tool1 Patch Line Tracing

## Scope

Add a runner-only inspection mode for V0.1 line tracing. This does not alter
training, prefetch, augmentation, sampling order, or model architecture.

## Plan

- Add direction-field tracing helpers in `runner.py`.
  - Prepare the center side-strip patch with the same loader path used by
    `--augment-vis`.
  - Load a `FiberStripDirectionNet` checkpoint and run inference on the patch.
  - Decode model output to strip-image direction vectors with the existing
    Lasagna ambiguous two-cos-channel decoder.
  - Bilinearly sample the decoded direction field at floating-point trace
    positions.
  - Align each sampled direction to the current tracing direction so the
    ambiguous sign is continuous.
  - Step in both directions from the transformed CP and stop at the configured
    receptive-field border margin or invalid/non-finite direction samples.
- Add CLI flags to `runner.py`.
  - `--line-trace-vis` enables the tool.
  - `--checkpoint <path>` supplies the trained snapshot.
  - `--line-trace-step <px>` controls trace step length.
  - `--line-trace-rf-margin <px>` optionally overrides the default margin
    derived from the configured model depth.
- Export inspection artifacts under `--export-dir`.
  - `line_trace_vis.jpg`: raw patch with original line and traced line.
  - `line_trace_summary.txt`: sample/checkpoint metadata and trace stats.
- Keep the tool deterministic for a given config/checkpoint/sample index.

## Spec Update

Update `planning/specs.md` with the V0.1 runner tool behavior, checkpoint
requirement, line tracing stop rule, and exported files.

## Docs Updates

Update `docs/code_structure.md` to document the new runner mode and CLI flags.

## Testing

- Add focused helper tests using synthetic direction fields:
  - straight horizontal tracing from the CP;
  - sign-continuity through the ambiguous decoded directions;
  - border margin stops before the patch edge.
- Run:
  - `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/runner.py`
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`

## Changelog

Add a changelog entry because this adds a public runner inspection mode.
