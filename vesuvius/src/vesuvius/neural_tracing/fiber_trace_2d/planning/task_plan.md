# Trace2CP Similarity Debug Column Plan

## Scope

- Add embedding-similarity maps to the existing single-pair
  `--trace2cp-vis` JPG.
- Keep Trace2CP metric, refinement, TTA, and candidate scoring unchanged.
- Use the existing contrastive embedding field and combined-mode CP embedding
  bank; do not invent a replacement global embedding definition when no bank
  exists.

## Implementation

1. Add Trace2CP similarity-debug data structures and helpers in `runner.py`.
   - Compute per-pixel cosine similarity maps from the normalized embedding
     field.
   - Build maps for start CP, target CP, same-fiber CP bank mean similarity,
     forward trace last sampled embedding, and reverse trace last sampled
     embedding.
   - Treat missing embedding channels as "no debug column".

2. Attach similarity debug to pair evaluation.
   - After the selected Trace2CP result is known, build debug maps from the
     selected forward/reverse traces.
   - Use the existing combined fiber embedding bank for the global map when it
     is available.

3. Render the debug column.
   - Add one optional right-side column to `_draw_trace2cp_overlay`.
   - Render similarity maps on a fixed cosine scale `-1..1 -> 0..255`.
   - Overlay the fiber line, relevant trace, and CP markers for orientation.

4. Add tests.
   - Verify similarity debug maps are computed with the expected cosine values.
   - Verify `_draw_trace2cp_overlay` adds the debug column.

## Spec Update

- Document that Trace2CP visualization includes an optional embedding-debug
  column when embedding output is present.
- Document that the global map uses the same-fiber CP embedding bank from
  combined Trace2CP mode.

## Docs Updates

- Update `docs/code_structure.md` runner/Trace2CP documentation.
- Update `planning/changelog.md`, `status.md`, and `task_log.md`.

## Validation

- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
- Run `git diff --check` on touched files.
