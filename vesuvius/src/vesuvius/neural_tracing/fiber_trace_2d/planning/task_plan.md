# Native 3D Trace2CP Beam Lookahead Plan

## Interpretation

- Current native beam tracing expands one step from each live beam and prunes
  immediately back to `beam_width`.
- For this task, keep the same candidate scoring and branch semantics, but
  expand a short future tree before pruning. With lookahead depth `3`, each
  live beam can survive two locally worse steps if the third-step continuation
  recovers enough cumulative score.
- `beam_width = 1` remains the greedy compatibility mode and should not use
  lookahead.

## Implementation

- Extend `NativeTrace2CpConfig` with `beam_lookahead_steps: int = 3`.
- Add CLI flag `--beam-lookahead-steps`, default `3`.
- In the beam path, replace the single-step expand/prune cycle with:
  - Start from the current live beam set.
  - Expand all active frontier states for up to `beam_lookahead_steps` trace
    steps or until a target-plane candidate is reached.
  - Do not prune between those internal lookahead steps.
  - If target-plane candidates are found at any lookahead depth, choose the
    reached state with lowest cumulative score and stop.
  - Otherwise prune the full lookahead frontier back to `beam_width`, using the
    existing near-duplicate pruning.
- Keep the existing step guard semantics. The lookahead expansion must not run
  beyond `step_limit`; if fewer than `beam_lookahead_steps` remain, expand only
  the remaining steps.
- Keep progress reporting based on committed beam frontiers, and include the
  lookahead setting in the progress text and summary JSON.
- Preserve multi-direction branch coupling, sign ambiguity handling, smoothness
  scoring, target-plane interpolation, and inferred block caching.

## Spec Update

- Update native 3D Trace2CP specs to state that beam pruning happens after
  `beam_lookahead_steps` future expansion steps, default `3`.
- State that `beam_width=1` bypasses lookahead and preserves greedy tracing.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section with lookahead
  behavior and the new CLI flag.

## Tests

- Unit-test default config includes `beam_lookahead_steps == 3`.
- Unit-test that `beam_width=1` keeps the old greedy behavior.
- Unit-test a fake-cache path where one-step beam pruning drops the necessary
  locally worse branch but `beam_lookahead_steps=3` keeps it and reaches the
  target plane.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Add a 2026-07-17 changelog entry for native 3D Trace2CP beam lookahead.

## Non-Goals

- Do not add dynamic programming or global optimal tracing.
- Do not change candidate scoring, direction/presence weighting, or
  forward/reverse fusion in this task.
