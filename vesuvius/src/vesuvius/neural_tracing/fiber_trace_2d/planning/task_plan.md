# Native 3D Trace2CP First-Step CP-Tangent Relaxation Plan

## Task Summary

Native 3D Trace2CP should keep using the CP-local fiber-line tangent to choose
the forward/backward trace orientation, but the first accepted step should not
be forced to match that tangent in the tangent plane. For the first step from
each CP:

- smoothness loss is zero;
- CP tangent direction agreement is evaluated only by its Lasagna-normal
  component;
- tangent-plane disagreement against the CP tangent is ignored;
- normal direction ambiguity remains handled by absolute/squared normal
  component comparison;
- all later steps keep the existing full candidate scoring.

## Current Relevant Behavior

- The first trace direction is seeded from the adjacent CP-local fiber-line
  tangent toward the target CP. This is required and should remain.
- Candidate scoring currently uses:
  `1 - dot(current_dir, step_dir) * dot(candidate_dir, step_dir) * presence`,
  plus smoothness against the previous accepted step direction.
- With normal-aware smoothness active, smoothness is split into tangent-plane
  and normal-tilt components using Lasagna normals sampled at candidate trace
  coordinates.
- Beam mode tracks state depth in `frontier_gen.depth`; root states have
  `depth == 0`.
- Greedy mode can identify the first accepted step by loop iteration or an
  explicit step counter.

## Target Scoring Semantics

### First Step From CP

For candidates expanded from a root CP state:

1. Keep generating candidate directions around the current CP tangent as today.
2. Keep sampling candidate endpoint model direction/presence as today.
3. Set smoothness loss to zero for these candidates.
4. Replace the `current_dot = dot(current_dir, step_dir)` gate with a
   normal-only gate:
   - sample/use the Lasagna normal at the candidate point;
   - compute the signed/elevation component of the CP tangent along the normal;
   - compute the signed/elevation component of the candidate step direction
     along the same normal;
   - compare those components without considering tangent-plane rotation.
5. Preserve normal sign ambiguity:
   - flipping `n -> -n` must not change this first-step gate;
   - use an absolute or squared component difference formulation.
6. If the candidate Lasagna normal is invalid, fall back to the existing full
   `dot(current_dir, step_dir)` gate for that candidate. Do not invent a normal
   score from invalid normals.

One suitable first-step normal gate:

```text
current_elev = asin(clamp(dot(current_dir, n), -1, 1))
candidate_elev = asin(clamp(dot(step_dir, n), -1, 1))
normal_angle = abs(candidate_elev - current_elev)
normal_gate = clamp(cos(normal_angle), 0, 1)
```

This is invariant to `n -> -n` because both elevations flip sign and the
absolute difference is unchanged. It also matches the existing direction-gate
scale: `1` for equal normal/elevation component and lower values for larger
normal-component mismatch.

### Later Steps

For all non-root expansions:

- keep the existing full current direction gate;
- keep candidate endpoint/substep direction and presence scoring;
- keep normal-aware smoothness with configured tangent/normal weights;
- keep beam lookahead and pruning behavior unchanged.

## Implementation Plan

### 1. Add First-Step Mask To Candidate Scoring

- Extend `_score_candidate_loss_tensors_batched(...)` with optional
  `first_step_mask: torch.Tensor | None`.
- Shape: `[N]`, where `N` is the active state count.
- Default `None` means all false to preserve current behavior for tests and
  helper callers.
- Validate shape when provided.
- Expand to `[N, M]` when applying candidate-level overrides.

### 2. Implement Normal-Only First-Step Gate

- Add a small helper near the existing smoothness helper, for example:
  `_native_first_step_normal_gate_torch(current, candidates, normals, valid)`.
- Inputs:
  - `current`: `[N,3]`;
  - `candidates`: `[N,M,3]`;
  - `normals`: `[N,M,3]` or `[N,M,S,3]`;
  - `valid`: `[N,M]` or `[N,M,S]`.
- For `[N,M,S,3]`, use the final substep normal for the first-step gate, to
  match the current one-per-step smoothness behavior.
- Return:
  - `gate`: `[N,M]`;
  - `gate_valid`: `[N,M]`.
- For invalid normals, `gate_valid` is false so scoring falls back to the
  regular full `current_dot`.

### 3. Override `current_dot` For First-Step States

- Compute regular `current_dot` as today.
- If `first_step_mask` has true states and candidate normals are available:
  - compute `normal_gate`;
  - replace `current_dot` with `normal_gate` only where:
    - state is first step;
    - candidate normal is valid.
- If candidate normals are unavailable for first-step scoring, keep regular
  `current_dot`. This preserves fake-cache/tests without Lasagna normal
  sampling, and avoids silent invalid normal semantics.

### 4. Zero First-Step Smoothness

- After computing `smoothness_loss`, zero it where `first_step_mask` is true:
  `smoothness_loss = where(first_step_mask[:,None], 0, smoothness_loss)`.
- This must apply to both endpoint-only and candidate-substep scoring, since
  smoothness remains one term per accepted candidate step.

### 5. Wire Beam Tracing

- In `_trace_native_3d_one_way_beam(...)`, derive:
  `first_step_mask = frontier_gen.depth[valid_state_indices] == 0`.
- Pass it to `_score_candidate_loss_tensors_batched(...)`.
- Keep the existing root-state direction override:
  current direction for root states remains the CP-local tangent, not sampled
  model direction.

### 6. Wire Greedy Tracing

- In `_trace_native_3d_one_way_greedy(...)`, pass a first-step flag on the
  first accepted candidate scoring call only.
- Use the same scorer path as beam, not a separate formula.
- Preserve existing behavior if greedy is selected with `--beam-width 1`.

### 7. Diagnostics / Summaries

- No new CLI flag is required. This is the new native 3D Trace2CP behavior.
- Optional: include a boolean summary field such as
  `first_step_cp_tangent_relaxed: true` in the native config summary if the
  surrounding summary structure is already centralized and low-risk.
- Do not add extra stdout noise.

## Spec Update

Update `planning/specs.md` native 3D Trace2CP candidate scoring section:

- First step from each CP keeps CP-local tangent orientation.
- First-step smoothness is disabled.
- First-step CP tangent gate is normal/elevation-only using Lasagna normals
  sampled at candidate points.
- First-step tangent-plane mismatch against the CP tangent is ignored.
- If candidate normal is invalid or unavailable, first-step current-direction
  gate falls back to full dot product for that candidate.
- All later steps use the regular full scoring and normal-aware smoothness.

## Docs Updates

Update `docs/code_structure.md` native 3D Trace2CP section:

- Document that root CP expansion has special first-step scoring.
- Clarify that the CP tangent is still used for orientation, but not for
  tangent-plane first-step penalty.
- Clarify that this uses the same candidate-point Lasagna normals as
  normal-aware smoothness.

## Tests

Add focused tests in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`:

1. First-step smoothness zero:
   - high tangent smoothness weight;
   - first-step mask true;
   - candidate with large tangent turn;
   - smoothness term returns zero for that state/candidate.
2. Later-step smoothness still active:
   - same candidate setup;
   - first-step mask false;
   - tangent smoothness is non-zero.
3. First-step tangent-plane mismatch ignored:
   - two candidates with equal normal/elevation component but different
     tangent-plane angle against CP tangent;
   - with first-step mask true and valid normals, current-direction gate should
     not distinguish them by tangent-plane rotation.
4. First-step normal/elevation mismatch still matters:
   - candidate with different normal component scores worse than one with
     matching normal component.
5. Invalid first-step normal fallback:
   - invalid normal mask false for a candidate;
   - scorer uses regular full `current_dot`, preserving current behavior.
6. Beam wiring:
   - fake cache and fake normal sampler show root-depth candidates receive the
     first-step override and second-depth candidates do not.

Run:

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py
```

Also run:

```bash
git diff --check -- vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/planning/specs.md vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/docs/code_structure.md
```

## Changelog

Add a 2026-07-17 changelog bullet after implementation:

- Native 3D Trace2CP now relaxes first-step CP tangent scoring: smoothness is
  disabled for the root step and CP tangent agreement is normal/elevation-only
  for that step.

## Independent Plan Review

Checked against the current spec excerpt:

- Keeps CP-local tangent seeding and does not use straight CP-to-target chord.
- Keeps candidate-point Lasagna normal sampling; no reference-line normal
  interpolation or new normal decoder.
- Keeps normal-aware smoothness for later steps.
- Keeps direction/presence scoring and branch choice unchanged except for the
  first-step current-direction gate.
- Keeps CLI defaults unless explicitly stated; no new user-facing knob is
  required.

## Non-Goals

- Do not change candidate generation, cone angle, beam width, or lookahead.
- Do not change target-plane stopping or trace fusion.
- Do not change model inference, strip rendering, or visualization panels.
- Do not use sampled model direction at the starting CP.
- Do not add nearest-normal search or interpolate normals from reference-line
  progress.

## Deviations / Deferred

None planned.
