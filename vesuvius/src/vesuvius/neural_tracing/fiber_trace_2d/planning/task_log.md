# Combined Direction And Contrastive Embedding Tracer Task Log

- Planned optional Trace2CP combined direction+embedding tracer for visualization and score tuning.
- Added runner helpers for:
  - predicting decoded direction plus appended embedding channels from one model pass;
  - bilinear normalized embedding sampling;
  - symmetric angular Trace2CP candidate fans;
  - greedy combined candidate scoring with direction, previous-step embedding, enclosing-CP embedding, and same-fiber CP-bank embedding terms.
- Added `--trace2cp-combined` plus candidate fan and score-weight CLI knobs.
- Single-pair and whole-fiber Trace2CP now build/use an in-memory same-fiber CP embedding bank when combined mode is enabled.
- Combined mode keeps the existing public Trace2CP metric unchanged; it only changes the selected trace path when explicitly requested.
- With `--med-tta --trace2cp-combined`, median TTA supplies the direction source while embedding scoring samples the reference segment patch.
- Updated `planning/specs.md`, `docs/code_structure.md`, and `planning/changelog.md`.
- Added synthetic tests for candidate fan generation, embedding sampling, combined direction-only scoring, embedding-dominant candidate choice, missing embedding failure, and updated Trace2CP test fixtures for the optional combined summary field.
- Validation:
  - `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
  - Result: 157 passed in 6.09s.
  - `git diff --check` on changed code/docs returned clean.
