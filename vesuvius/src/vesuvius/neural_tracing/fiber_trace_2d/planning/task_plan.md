# Native 3D Trace2CP Cumulative Tangent Smoothness Plan

## Task Summary

Native 3D Trace2CP currently has only local step-to-step smoothness. This can
allow several small tangent-plane turns to accumulate into a 90 degree bend.
Add a short-history cumulative smoothness term that is tangent-plane only.

## Target Semantics

- Keep the existing local smoothness unchanged.
- Keep first-step CP-tangent relaxation unchanged.
- Add a history direction per trace state.
- Compare each candidate step direction against that history direction in the
  tangent plane defined by the candidate-point Lasagna normal.
- Add the cumulative term to the existing smoothness loss.
- The cumulative term must not include a normal/elevation component.
- Invalid/unavailable candidate normals fall back to no cumulative penalty for
  that candidate; do not invent a normal or run a search.
- The term is configurable:
  - `cumulative_smoothness_steps`: short history window length, default `4`;
  - `cumulative_smoothness_tangent_weight`: tangent-only cumulative weight,
    default active and tunable from CLI.

## Implementation Plan

1. Add config fields and CLI flags for cumulative tangent smoothness.
2. Add a tensor helper that computes tangent-plane angular loss between a
   history direction `[N,3]` and candidates `[N,M,3]` using candidate normals.
3. Add optional `history_directions` input to the batched scorer.
4. Add cumulative tangent loss into `smoothness_loss`.
5. Preserve the first-step behavior by passing no effective history or zeroing
   cumulative loss for first-step states.
6. Extend greedy state with an EMA/finite-window approximation:
   - use aligned average direction update;
   - history factor is derived from `cumulative_smoothness_steps`;
   - no extra normal component is stored.
7. Extend beam tensor state with `history_directions_zyx` so every beam carries
   its own history direction.
8. Include history direction in node reconstruction and pruning transitions.

## Spec Update

Update native 3D Trace2CP scoring in `planning/specs.md`:

- local smoothness remains;
- cumulative smoothness is tangent-plane only;
- history length and weight are configurable;
- invalid candidate normals skip/fallback this cumulative term.

## Docs Update

Update `docs/code_structure.md` native 3D Trace2CP section with:

- where history is stored;
- that the cumulative term is additive smoothness, not direction/presence
  gating;
- that it is tangent-plane only.

## Tests

Add focused tests to `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`:

1. cumulative tangent loss penalizes a candidate that turns away from history;
2. cumulative tangent loss does not penalize equal tangent direction;
3. cumulative tangent loss is zero for first-step states;
4. invalid normals skip the cumulative tangent loss;
5. beam tracing carries separate history directions across states.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
```

Also run `git diff --check` on touched files.

## Changelog

Add a 2026-07-17 changelog entry for native 3D Trace2CP cumulative tangent
smoothness.

## Independent Plan Review

- Preserves the existing first-step CP tangent relaxation.
- Does not alter candidate generation, target-plane stopping, or fusion.
- Does not add cumulative normal/elevation penalties, matching the user
  clarification.
- Does not reimplement Lasagna normal decoding.

## Deviations / Deferred

None planned.
