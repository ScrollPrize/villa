# Native 3D Trace2CP Normal-Aware Smoothness Plan

## Availability Answer

Yes, the Lasagna normal is available and can be sampled directly at native
3D trace coordinates:

- `fiber_trace_2d.loader.FiberStrip2DLoader` already has the established
  Lasagna normal sampling/decoding implementation:
  `_lasagna_normals_at_zyx_batch(...)`.
- That helper samples `grad_mag`, `nx`, and `ny`, decodes the Lasagna ambiguous
  normal encoding with the existing code, and resolves the principal axis from
  the local interpolation cube.
- Native 3D candidate points are in selected-level ZYX coordinates. They can be
  converted back to base ZYX by multiplying by `record.volume_spacing_base`,
  then passed to the existing batched Lasagna normal sampler.
- The implementation should not pass around interpolated reference-line
  normals for this smoothness term.

## Target Behavior

- Replace the single isotropic smoothness turn penalty with an optional
  normal-aware split:
  - **tangent-plane turn**: the turn around the Lasagna normal axis, i.e. the
    angle between previous and candidate directions after projection into the
    plane perpendicular to the Lasagna normal.
  - **normal tilt turn**: the turn into/out of the normal direction, i.e. the
    difference in elevation against the Lasagna normal.
- The normal sign ambiguity must not affect the loss:
  - tangent-plane projected angle is sign-invariant;
  - normal tilt should be squared/absolute, so flipping `normal -> -normal`
    produces the same penalty.
- Add separate weights:
  - `smoothness_tangent_weight`
  - `smoothness_normal_weight`
- Keep `smoothness_free_angle_degrees` as the shared dead zone for both
  components in the first implementation.
- Keep the existing isotropic smoothness path available for tests, fake caches,
  and any future non-Lasagna/native use. Real native 3D Trace2CP with a normal
  sampler should use the split penalty.

## Candidate Normal Sampling

- Add a native normal sampler wrapper around the existing 2D loader Lasagna
  batch code, for example:
  `sample_lasagna_normals_selected_zyx(record, points_zyx_selected)`.
- The wrapper converts selected-level ZYX to base ZYX:
  `points_zyx_base = points_zyx_selected * record.volume_spacing_base`.
- It calls the existing Lasagna batch normal sampler rather than reimplementing
  normal decode logic.
- `_lasagna_normals_at_zyx_batch(...)` currently requires `line_indices` for
  diagnostics. For candidate points, pass synthetic candidate indices or add a
  small sibling helper that shares the implementation but reports
  `candidate_index`/`trace_point_index` instead of pretending they are fiber
  line indices.
- Candidate normals are sampled at the actual candidate endpoint for
  endpoint-only scoring.
- With `candidate_substeps > 1`, sample normals at the same substep points
  already built for direction/presence scoring, then use the final substep
  normal for the per-step smoothness term in the first implementation. A later
  refinement can average normal-aware smoothness across substeps if needed.
- Normal sampling should be batched over all active beam states and all
  candidates, matching the direction/presence candidate batch.
- If the Lasagna normal is invalid at a candidate point, fall back to the
  existing isotropic smoothness term for that candidate rather than failing the
  trace. The trace candidate can still be rejected by the model-output validity
  rules independently.

## Smoothness Formula

Given:

- previous accepted step direction `p`;
- candidate step direction `c`;
- Lasagna normal sampled at the candidate point, `n`;

Normalize all vectors and align `c` to the current trace direction as today.

Tangent-plane turn:

```text
p_t = normalize(p - dot(p,n) n)
c_t = normalize(c - dot(c,n) n)
tangent_angle = acos(clamp(dot(p_t,c_t), -1, 1))
```

Normal tilt turn:

```text
p_elev = asin(clamp(dot(p,n), -1, 1))
c_elev = asin(clamp(dot(c,n), -1, 1))
normal_angle = abs(c_elev - p_elev)
```

Penalty:

```text
tangent_loss = tangent_weight * max(0, tangent_angle - free_angle)^2
normal_loss = normal_weight * max(0, normal_angle - free_angle)^2
smoothness_loss = tangent_loss + normal_loss
```

Degenerate projection handling:

- If either tangent-plane projection is too small, fall back to the isotropic
  angle for the tangent component at that state.
- The normal tilt component remains valid as long as `n` is finite/non-zero.

## Config / CLI

- Extend `NativeTrace2CpConfig`:
  - `smoothness_tangent_weight: float | None = None`
  - `smoothness_normal_weight: float | None = None`
- Add CLI flags:
  - `--smoothness-tangent-weight`
  - `--smoothness-normal-weight`
- Interpretation:
  - if both are omitted and no normal sampler is passed, keep current
    isotropic `smoothness_weight` behavior;
  - if a normal sampler is passed and split weights are omitted, initialize both
    split weights from `smoothness_weight` for a conservative first default;
  - if either split weight is provided, require a valid normal sampler and use
    `smoothness_weight` only as backward-compatible default for the omitted
    split weight.
- Record all smoothness weights and whether normal-aware smoothness was active
  in single-pair and whole-fiber summary JSON.

## Scorer Integration

- Extend `_score_candidate_loss_tensors_batched(...)` with an optional
  `candidate_normals` tensor:
  - endpoint mode: `[N,M,3]`;
  - substep mode may either pass `[N,M,3]` for the final substep normal or
    `[N,M,S,3]` if the implementation keeps all sampled normals.
- Keep the existing isotropic smoothness computation when `candidate_normals`
  is absent or invalid for a candidate.
- When valid `candidate_normals` are provided:
  - compute tangent/normal component smoothness vectorized over `[N,M]`;
  - return the total smoothness loss in the existing `smoothness_loss` tensor;
  - optionally expose component diagnostics later, but do not expand
    `NativeTraceStep` in this first pass unless needed for debugging.
- Greedy tracing samples normals for its candidate endpoints in one batch.
- Beam tracing samples normals for all `[frontier,candidate]` endpoints or
  substeps in one batch.
- Candidate substep scoring remains unchanged: substeps affect direction/
  presence segment score, while smoothness is still one term per accepted
  candidate step in the first implementation.

## Trace API Wiring

- Add a normal sampler object/callback to native tracing, not a reference-line
  normal context.
- The sampler owns:
  - the `FiberStrip2DLoader`/geometry loader instance with Lasagna channels;
  - the active `record`;
  - conversion between selected-level and base ZYX.
- Pass the sampler to:
  - `trace_native_3d_one_way`
  - `_trace_native_3d_one_way_greedy`
  - `_trace_native_3d_one_way_beam`
  - `trace_native_3d_pair`
  - whole-fiber segment tracing
- Single-pair and whole-fiber native CLI both create this sampler once after
  constructing the existing geometry loader and record.

## Tests

- Add unit tests for the split smoothness formula:
  - pure tangent-plane turn with `normal_weight=0` is penalized and
    `tangent_weight=0` is not;
  - pure normal-tilt turn with `tangent_weight=0` is penalized and
    `normal_weight=0` is not;
  - flipping `n -> -n` gives the same smoothness value.
- Add scorer test with fixed candidate directions and `candidate_normals` to
  verify the split penalty changes candidate choice relative to isotropic or
  single-weight smoothing.
- Add trace wiring test with a fake normal sampler to verify greedy and beam
  pass candidate endpoint batches through the sampler without changing
  endpoint/substep direction-presence scoring.
- Run:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Spec Update

- Add native 3D Trace2CP normal-aware smoothness requirements:
  - samples Lasagna normals directly at candidate trace coordinates;
  - uses the existing Lasagna normal sampling/decode implementation;
  - does not interpolate normals from reference-line progress;
  - splits smoothness into tangent-plane and normal-tilt components;
  - normal sign ambiguity does not affect the penalty;
  - substep candidate scoring remains independent of smoothness component
    splitting.

## Docs Updates

- Update `docs/code_structure.md` native 3D Trace2CP section:
  - document direct candidate-point Lasagna normal sampling;
  - document the two smoothness weights and geometric interpretation.

## Changelog

- Add a 2026-07-17 entry after implementation:
  native 3D Trace2CP smoothness can be split into tangent-plane and normal-tilt
  penalties using direct Lasagna normals sampled at trace candidate points.

## Non-Goals

- Do not add new Lasagna nx/ny sampling inside the candidate scorer.
- Do not pass/interpolate reference-line normals by progress for this term.
- Do not use normal information to change candidate generation yet.
- Do not change direction/presence candidate scoring or candidate-substep
  semantics.
- Do not add nearest-normal searches in the first pass.
