# Native 3D Trace2CP All-Pairs Direction Product Plan

## Task Summary

Native 3D Trace2CP currently scores candidate direction agreement with two
terms:

```text
dot(last_sampled_dir, curr_step_dir) * dot(cand_sampled_dir, curr_step_dir)
```

Add a new default scoring mode that multiplies all pairwise direction
agreements among:

1. last step direction;
2. last sampled/current-point model direction;
3. current candidate step direction;
4. candidate-point model direction.

## Target Semantics

- Align all sign-ambiguous directions to the candidate step direction before
  comparing.
- Use six pairwise dot products:
  - last step vs current sampled;
  - last step vs candidate step;
  - last step vs candidate sampled;
  - current sampled vs candidate step;
  - current sampled vs candidate sampled;
  - candidate step vs candidate sampled.
- Clamp direction dot products to `[0, 1]` before multiplying.
- Candidate score becomes:

```text
presence * product(all six pairwise dots)
```

- Existing two-dot score remains available with a switch for comparison.
- Default mode is all-pairs enabled.
- First-step CP-root relaxation remains:
  - no smoothness;
  - current sampled/current gate uses normal-only CP tangent agreement where
    applicable;
  - last-step/history terms should not over-constrain the first root step.

## Implementation Plan

1. Add `all_pairs_direction_product: bool = True` to `NativeTrace2CpConfig`.
2. Add CLI flag `--no-all-pairs-direction-product`.
3. Add `all_pairs_direction_product` arg to candidate scorer wrappers.
4. In `_score_candidate_loss_tensors_batched(...)`:
   - compute `previous_dot = dot(previous_step_dir, candidate_step_dir)`;
   - align candidate-point model directions to candidate step as now;
   - compute pairwise dots among previous/current/candidate-step/candidate-sampled;
   - use all-pairs product when enabled;
   - keep the existing two-dot product when disabled.
5. For first-step states:
   - replace previous-dot and previous/current sampled constraints with neutral
     `1` where they would otherwise reintroduce the CP tangent-plane penalty;
   - preserve the existing normal-only `current_dot` override.
6. Apply analogous logic in the candidate-substeps path:
   - all substep candidate-sampled directions compare against the same previous
     and current/candidate step directions.
7. Update summary output with the scoring mode.

## Spec Update

Update `planning/specs.md` native 3D Trace2CP scoring section:

- document the four-direction all-pairs score;
- document default enabled;
- document fallback/toggle to the previous two-dot score;
- document first-step neutralization of previous/history pair terms.

## Docs Update

Update `docs/code_structure.md` native 3D Trace2CP section with the scoring
formula and CLI switch.

## Tests

Add focused tests in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`:

1. all-pairs mode penalizes an outlier previous/last-step direction that the
   old two-dot score would ignore;
2. disabling all-pairs preserves the old two-dot choice;
3. first-step mode does not reintroduce CP tangent-plane penalty through the
   previous direction terms;
4. substep scoring uses all-pairs products.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
```

Also run `git diff --check` on touched files.

## Changelog

Add a 2026-07-17 changelog entry for native 3D Trace2CP all-pairs direction
product scoring.

## Independent Plan Review

- Keeps model inference and candidate generation unchanged.
- Keeps recent cumulative tangent smoothness unchanged.
- Keeps first-step CP-tangent relaxation by neutralizing previous/current
  tangent-plane pair terms for root candidates.
- Does not change target-plane stopping, fusion, or visualization.

## Deviations / Deferred

None planned.
