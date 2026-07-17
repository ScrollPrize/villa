# Native 3D Trace2CP Beam Search Plan

## Interpretation

- Current native 3D tracing is greedy: at each step it evaluates all candidate
  directions once and commits to the single best next point.
- Current candidate density is not literally one-degree angles. It is
  `cone_grid_size x cone_grid_size`, currently `25x25 = 625`, mapped to the
  cone disk.
- For this task, replace that with an explicit angular-step generator:
  `cone_angle_step_degrees = 5.0`. With `cone_angle_degrees = 25.0`, generate
  tangent-plane angular offsets in `[-25, 25]` degrees at 5-degree spacing and
  keep offsets inside the cone disk. This gives roughly 80 candidates rather
  than 625, depending on disk boundary inclusion. The center direction is
  always included and evaluated first for stable tie behavior.

## Implementation

- Extend `NativeTrace2CpConfig` with:
  - `beam_width: int = 8`
  - `beam_prune_distance_voxels: float = 1.0`
  - `cone_angle_step_degrees: float = 5.0`
- Keep `beam_width = 1` as the exact greedy compatibility mode.
- Replace or bypass `generate_cone_candidates(..., grid_size=...)` in native
  Trace2CP with a new angular-step candidate generator. Keep any old
  `--cone-grid-size` CLI compatibility only if required by existing tests, but
  native default behavior should be driven by the 5-degree step.
- Add a beam-state structure containing:
  - current point
  - previous accepted step direction
  - cumulative score/loss
  - trace points and per-step diagnostics
  - reached-target-plane state
- At each trace step:
  - Generate candidate directions around each beam state's current inferred
    direction.
  - Evaluate candidates with the existing branch-aware native scoring:
    `1 - dot(current_dir, step_dir) * dot(branch_dir, step_dir) * branch_presence`
    plus the configured smoothness penalty.
  - Add local loss to cumulative beam loss.
  - Interpolate target-plane crossings exactly as the greedy tracer does.
  - Merge/prune near-duplicate live states within
    `beam_prune_distance_voxels`, keeping the lower cumulative loss.
  - Keep the best `beam_width` live states.
- Stop when target-plane candidates are found and choose the reached state with
  lowest cumulative score. If no beam reaches the target plane before the step
  guard, return the best live state and the same failure reason semantics as
  greedy tracing.
- Preserve multi-direction semantics: branch choices are always coupled
  `(6 direction channels + 1 presence channel)` options, and branch choice is
  local to each candidate score.
- Preserve ambiguous direction semantics: inferred branch axes are sign-aligned
  against the candidate/previous direction before dot products; smoothing uses
  the directed proposed step because candidate motion itself is not ambiguous.
- Update progress reporting to show beam width and active beam count, while
  retaining target-plane progress/ETA/block count.

## CLI / Config

- Add CLI flags:
  - `--beam-width`
  - `--beam-prune-distance-voxels`
  - `--cone-angle-step-degrees`
- Default native Trace2CP runs should use beam search with width `8` and
  5-degree cone steps.
- Keep `--cone-angle-degrees` unchanged at `25.0`.
- If `--cone-grid-size` remains accepted for backward CLI compatibility,
  document it as legacy/ignored when `--cone-angle-step-degrees` is positive.

## Spec Update

- Update native 3D Trace2CP specs from greedy one-step tracing to beam search.
- Specify the explicit 5-degree angular-step cone generator and state that the
  center direction is always included.
- Specify that beam search still uses the existing branch-aware scoring and
  ambiguous-direction alignment.
- Specify `beam_width = 1` as greedy compatibility mode.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section with beam-state
  search, new CLI flags, and candidate-count behavior.

## Tests

- Unit-test angular-step candidate generation: with `25 deg` cone and
  `5 deg` step, candidates are finite/unit length, include the center, and all
  lie inside the configured cone.
- Unit-test greedy compatibility: `beam_width=1` reproduces existing one-way
  trace behavior on a simple fake cache.
- Unit-test beam advantage: construct a fake cache where the locally best first
  step leads to a dead/worse continuation, and verify `beam_width>1` selects
  the globally better continuation.
- Unit-test CLI/config defaults for `beam_width=8` and
  `cone_angle_step_degrees=5.0`.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog

- Add a 2026-07-17 changelog entry for native 3D Trace2CP beam search and
  5-degree cone candidate steps.

## Non-Goals

- Do not change training losses, model output layout, branch routing in
  training, or snapshot format.
- Do not add a full dynamic-programming solver for native 3D tracing.
- Do not change forward/reverse fusion scoring in this task.
