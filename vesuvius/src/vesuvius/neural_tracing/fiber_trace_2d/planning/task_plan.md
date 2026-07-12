# Trace2CP Short Z-Search Plan

## Scope

- Add an experimental short z-search to the combined Trace2CP inspection path.
- Keep direction-only Trace2CP, median-TTA direction tracing, training loss,
  best-checkpoint selection, and the public `trace2cp_error` metric unchanged
  unless a later task explicitly changes them.
- Require checkpoint embedding channels for z-search, matching the existing
  `--trace2cp-combined` embedding requirement.
- Do not add image-space geometric warps. All extra z planes must be sampled
  from explicit volume coordinates through the existing Trace2CP segment
  coordinate path.

## CLI And Config

1. Add a Trace2CP CLI flag for the new behavior.
   - Preferred name: `--trace2cp-z-search`.
   - Require `--trace2cp-combined`, because the search is defined by combined
     direction plus embedding evidence.
   - Add optional tuning flags only if needed by the implementation:
     `--trace2cp-z-step-voxels` defaulting to `2.0` selected-scale voxels, and
     `--trace2cp-z-max-layer` defaulting to a small bound such as `4` layers.

2. Keep defaults stable.
   - Existing Trace2CP commands without `--trace2cp-z-search` should produce
     the same traces, summaries, metrics, and visual layout as today.

## Loader And Plane Sampling

1. Extend Trace2CP segment loading to support center-strip z offsets.
   - Reuse the already-built Trace2CP segment coordinate grid and its strip
     offset axis.
   - A z layer index `k` maps to selected-scale voxel offset
     `k * trace2cp_z_step_voxels`.
   - Convert selected-scale voxel offsets to base-coordinate offsets using the
     selected `base_volume_scale`, following existing strip-z offset semantics.
   - Sample each requested plane through the existing VC3D blocking coordinate
     sampler.

2. Add a z-plane prediction cache in `runner.py`.
   - Start by loading/infering layers `-1, 0, +1`.
   - Store per-layer image, valid mask, decoded direction field, and embedding
     field.
   - At every trace step, before candidate evaluation, ensure layers
     `current_layer - 1`, `current_layer`, and `current_layer + 1` exist.
   - If a trace reaches a new outer layer, lazily sample/infer the next needed
     layer and append it to the cache.
   - Enforce the configured z-layer bound so tracing cannot expand without
     limit.

3. Preserve deterministic behavior.
   - No random choices are introduced.
   - Plane cache expansion order must be deterministic from trace history.

## Z-Aware Combined Tracing

1. Add a z-aware trace point representation.
   - Existing trace points are `x, y`.
   - Z-search trace points carry `x, y, z_layer`, plus derived
     `z_voxels = z_layer * trace2cp_z_step_voxels`.
   - Existing 2D trace helpers should remain available for non-z-search modes.

2. Extend candidate generation.
   - Keep the existing 2D angular fan around the current oriented 2D direction.
   - For each 2D angular candidate, evaluate it at candidate z layers
     `current_layer - 1`, `current_layer`, and `current_layer + 1`, filtered to
     layers present or creatable inside the layer bound.
   - The xy step remains the same `--line-trace-step`; z changes discretely by
     the selected candidate layer.
   - Ignore out-of-plane angle effects for now: z does not change the 2D angle
     decoding, and there is no 3D direction vector regression.

3. Extend the combined candidate score.
   - Keep the existing weighted terms:
     direction disagreement, last-point embedding distance, enclosing-CP
     embedding distance, and same-fiber CP-bank embedding distance.
   - Evaluate direction and embedding terms from the candidate z layer's
     predicted fields at the candidate xy point.
   - Use the existing weights initially. Do not add a separate z regularizer
     unless the first implementation shows unstable z jumps.
   - The selected candidate updates both xy and z layer; the selected
     embedding becomes the next last-point embedding.

4. Handle CP embeddings.
   - Start and target CP embeddings should come from the center layer `z=0`.
   - Same-fiber CP-bank embeddings should remain center-layer embeddings for
     the first version, so the bank stays comparable with existing combined
     tracing.

## Z-Aware Connection And Refinement

1. Keep public target-column metric stable for this task.
   - The current `trace2cp_error` remains y-error at the opposite CP column
     divided by horizontal span.
   - Add z-aware closest/connection diagnostics only for fused-line
     construction and visualization.

2. Extend closest-approach selection.
   - Interpolate forward and reverse traces at common x positions.
   - Compute raw connection distance as
     `abs(forward_y - reverse_y) + abs(forward_z_voxels - reverse_z_voxels)`.
   - Apply the same center-distance magnification currently used for y-only
     closest-approach selection.
   - Keep the existing deterministic tie-break behavior as much as possible.

3. Extend fused line construction.
   - CP z positions are fixed at `0`.
   - Warp each partial trace linearly toward the closest-approach midpoint in
     both y and z.
   - The fused/resampled line carries fractional z voxel positions.
   - Existing 2D optimized refinement may remain y-only for the first version,
     but it must preserve and pass through the fused fractional z positions if
     it renders z-aware output. If that is too awkward, explicitly skip
     optimized refinement for z-search and report that in the summary.

## Visualization

1. Keep the existing Trace2CP rows.
   - Full bidirectional traces.
   - Partial traces.
   - Fused CP-to-CP line.
   - Optimized/refined line where supported.

2. Split z-search visualization by trace direction.
   - Add separate forward and backward z-corrected views because each trace can
     choose a different z layer per x column.
   - Do not replace the base center-strip view; show z-corrected views as
     additional columns or clearly labeled panels.

3. Build z-corrected images column by column.
   - For each x column, derive a z voxel value from the selected trace/fused
     z path.
   - Round that column's z value to the nearest inferred z layer and copy that
     layer's already sampled image content for the whole image column.
   - Do not re-sample the volume during reconstruction and do not interpolate
     image values between z layers for this visualization.
   - If the rounded target layer is missing, mark that column invalid/black and
     report it in debug output.
   - Overlay the corresponding z-aware trace/line in the same visual style as
     existing Trace2CP output.

4. Update summary/debug text.
   - Print whether z-search was enabled.
   - Report z step in selected-scale voxels, layer range touched, layers
     inferred, and any columns whose z-corrected visualization lacked a layer.
   - Add per-direction z min/max/mean absolute z movement.

## Spec Update

- Add `--trace2cp-z-search` to Trace2CP specs as an experimental combined
  tracing mode.
- Specify that z-search samples additional Trace2CP segment planes from
  coordinate offsets along the strip offset axis, not by warping images.
- Define layer spacing in selected-scale voxels, default `2.0`.
- Specify lazy z-plane inference and bounded expansion.
- Specify combined candidate scoring over 2D angular candidates and neighbor z
  layers while ignoring out-of-plane angle effects.
- Specify z-aware closest-approach distance as
  `abs(dy) + abs(dz_voxels)`, with the existing center-distance magnification.
- Specify fractional z output for fused lines.
- Specify separate forward/backward column-wise z-corrected visualizations
  assembled by nearest-layer copy from already inferred images, with no
  reconstruction-time volume sampling.
- State explicitly that public `trace2cp_error`, training test metric, and
  best-checkpoint selection remain unchanged for this task.

## Docs Updates

- Update `docs/code_structure.md`:
  - Trace2CP runner z-search mode and CLI usage.
  - Loader/runner plane-cache data flow.
  - Visualization output layout and summary fields.
- Update `planning/changelog.md` after implementation.
- Replace `planning/task_log.md` with current-task implementation notes and
  validation results during implementation.

## Tests

1. Unit-test z-plane offset construction.
   - With a fake Trace2CP segment source and fake sampler, request layers
     `-1, 0, +1`.
   - Assert offsets are exactly `-2, 0, +2` selected-scale voxels by default
     and use the existing strip offset axis.

2. Unit-test lazy plane expansion.
   - Start with layers `-1, 0, +1`.
   - Force a trace to select layer `+1`.
   - Assert layer `+2` is inferred before the next candidate step.
   - Assert expansion stops clearly at the configured layer bound.

3. Unit-test z-aware candidate selection.
   - Fake equal direction evidence across layers.
   - Make embedding evidence strongest in `+1`.
   - Assert the selected candidate chooses the `+1` layer.
   - Also test that an invalid candidate layer is skipped.

4. Unit-test z-aware closest/fused connection.
   - Provide forward/reverse traces with identical y but different z and assert
     the Manhattan distance includes z voxels.
   - Assert linear correction produces fractional z values and keeps CP z at
     zero.

5. Unit-test z-corrected visualization assembly.
   - Use three synthetic layer images with known constant values.
   - Provide a trace whose z changes by column.
   - Assert output columns come from the nearest rounded layer, with no
     interpolation between layers.

6. Regression tests.
   - Existing Trace2CP and loader tests should pass without
     `--trace2cp-z-search`.
   - Run:
     `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py`
   - Run `git diff --check` on touched files.

## Review Notes

- The plan intentionally keeps the public metric unchanged. The requested
  Manhattan `y + z` distance is applied to the connection/closest-approach
  logic where the request mentions "for the connection". If the public
  `trace2cp_error` should also become z-aware, that should be a separate
  explicit spec change because it affects training best-model selection.
- The plan treats "z" as the existing Trace2CP strip offset axis, measured in
  selected-scale voxels. It does not mean raw volume z-axis unless the strip
  offset axis happens to align with raw z for a given local frame.
