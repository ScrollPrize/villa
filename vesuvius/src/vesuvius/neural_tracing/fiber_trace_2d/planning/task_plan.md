# Native 3D Trace Fusion Pairwise Meeting Plan

## Implementation

- Remove the native 3D fusion search's dependence on `closest_progress` over
  the straight CP axis as the primary meeting search.
- Keep `closest_progress` in the result as a diagnostic only, computed after
  selecting the pair by projecting the chosen midpoint onto the CP axis.
- Add helpers in `fiber_trace_3d/trace2cp_tool.py`:
  - cumulative traced arc length for a polyline;
  - resample a traced polyline by arc length at about one selected voxel
    spacing, while preserving original endpoints;
  - extract a partial trace up to a selected resampled point index;
  - warp a partial trace to a midpoint using traced arc-length fraction, not
    CP-axis progress.
- In `fuse_forward_reverse_traces(...)`:
  - create dense/resampled forward and reverse point sets using trace arc
    length;
  - compute pairwise Euclidean gaps in vectorized NumPy chunks;
  - compute pair score:
    `gap_factor * pair_gap + forward_arc_to_i + reverse_arc_to_j`;
  - use `gap_factor = 1.0` for now, matching the request;
  - choose the pair with minimum score, tie-breaking by smaller gap and then by
    larger combined traveled distance so exact ties prefer a more central
    meeting over immediate endpoints;
  - compute the midpoint and warp both partial traces to it;
  - concatenate `forward(start->mid)` plus `reverse(mid->target)` and
    arc-length resample as before.
- Preserve result fields:
  - `raw_gap_voxels` = Euclidean gap between chosen pair;
  - `considered_gap_voxels` = full pair score above;
  - `center_penalty` = `1.0` because the old center penalty no longer defines
    the selection;
  - `closest_forward_zyx`, `closest_reverse_zyx`, and
    `closest_midpoint_zyx` = chosen pair diagnostics;
  - `reason = "pairwise_arc_length_meeting"`.
- Remove the accidental startup-print change from the prior misread image
  issue if it is still present; that was not requested and should not remain as
  part of this task.

## Spec Update

- Update native 3D Trace2CP fusion specs to state that meeting selection is now
  pairwise over traced arc length, not straight-axis overlap progress.
- Document the exact score formula and current `gap_factor = 1.0`.
- Note that `closest_progress` is diagnostic only after this change.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP fusion description to
  match pairwise arc-length meeting and midpoint warping.

## Tests

- Update existing fusion tests that assert center-progress semantics.
- Add a regression where two traces bend/backtrack relative to the CP axis and
  the axis-progress method would choose the wrong meeting, while pairwise
  arc-length scoring chooses the intended closest pair.
- Add a tie-break regression where equally scored endpoint and middle meetings
  prefer the more central/later pair.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Add a `2026-07-17` changelog entry for native 3D Trace2CP pairwise
  arc-length fusion.

## Non-Goals / Risks

- Do not change tracing candidate search, model inference, strip rendering, or
  whole-fiber restart semantics.
- Do not add a user-facing CLI knob for `gap_factor` yet.
- The requested `gap_factor = 1.0` may still prefer shorter traced partials in
  some geometries; tests should lock the requested behavior, and later tuning
  can expose/raise this factor if needed.
