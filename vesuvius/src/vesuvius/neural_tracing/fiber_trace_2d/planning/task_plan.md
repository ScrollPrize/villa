# 3D Trace2CP Metric Wiring Plan

## Current State

The previous 3D follow-up implemented the low-level bridge in
`fiber_trace_3d.trace2cp_bridge`, but stopped before the requested wiring:

- `fiber_trace_3d/train.py` still evaluates only dense 3D direction/presence
  loss in `evaluate_dense_loss(...)`.
- `best.pt` is selected from dense test loss, not public Trace2CP error.
- The checked-in 3D configs have `test_datasets`, but no keys that tell 3D
  training how to build the 2D Trace2CP metric geometry.
- There is no 3D CLI path equivalent to the 2D `--trace2cp-vis` command.
- The bridge is currently covered only by a synthetic projection/scoring test,
  not by train/CLI integration.
- `fiber_trace_3d/projection.py` currently decodes 3D Lasagna directions with a
  unit-sphere candidate grid. That must be removed. The repo already contains
  analytic two-channel angle decoders and Lasagna 3-plane reconstruction logic.

That was a miss: full Trace2CP metric wiring was explicitly in the task and
should not have been deferred.

## Goal

Add minimal, explicit 3D Trace2CP evaluation support:

1. During 3D test evaluation, project dense 3D checkpoint outputs onto the
   existing 2D Trace2CP side-strip geometry.
2. Compute the public bidirectional `trace2cp_error` over fixed test CP pairs.
3. Use this metric for TensorBoard, stdout, current snapshots, and best
   checkpoint selection when enabled.
4. Add a 3D CLI visualization/inspection mode for one CP pair or one fiber.

Training samples remain ordinary CP-centered 3D blocks. Only evaluation uses
the 2D Trace2CP geometry.

## Config/API Changes

Add these 3D training keys:

- `training.test_trace2cp_enabled`: default `true` when `test_datasets` exists,
  otherwise `false`.
- `training.test_trace2cp_loader_config`: optional path to a 2D loader config
  used only for Trace2CP geometry. If omitted, build a minimal in-memory 2D
  metric config from the 3D config and `test_datasets`.
- `training.test_trace2cp_control_points`: optional override. If omitted, reuse
  `training.test_control_points`. `0` means all held-out CP samples.
- `training.test_trace2cp_start_sample_index`: optional override. If omitted,
  reuse `training.test_start_sample_index`.
- `training.test_trace2cp_step_px`: default `4.0`.
- `training.test_trace2cp_rf_margin_px`: default to model depth / receptive
  margin, matching the 2D training behavior.
- `training.test_trace2cp_presence_enabled`: default `true` when the 3D output
  has a presence channel.

Add these 3D CLI flags to `fiber_trace_3d.train` or a small 3D runner module:

- `--trace2cp-vis`
- `--checkpoint <path>`
- `--sample-index <idx>`
- `--fiber-json <path>`
- `--export-dir <path>`
- `--trace2cp-step-px <float>`
- `--trace2cp-rf-margin-px <float>`

The CLI should print `trace2cp_error=...` for single-pair runs and
`trace2cp_error_mean=...` for whole-fiber runs.

## Implementation Plan

0. Enforce no-silent-skip reporting for this task.
   - Any missing, simplified, unsupported, or postponed item must be written in
     `planning/task_log.md` during implementation.
   - The final response must include those items plainly, even when tests pass.
   - Do not convert a requested feature into a "boundary" without user-visible
     reporting and a concrete follow-up plan.

1. Add a 3D Trace2CP config parser/helper.
   - Read explicit training keys.
   - Build or load a 2D `FiberStrip2DLoader` for metric geometry only.
   - Fail loudly if required geometry keys cannot be derived.

2. Add a 3D dense inference helper for arbitrary Trace2CP strip coordinates.
   - Accept `coords_xyz` and valid mask from the 2D loader.
   - Convert base-volume XYZ coordinates to selected-level ZYX coordinates
     using the same `base_volume_scale`/spacing convention as the 3D loader.
   - Compute one or more axis-aligned 3D blocks that cover the strip coords
     plus a small interpolation margin.
   - Run the 3D model on those blocks.
   - Use `project_3d_output_to_trace2cp_fields(...)` to sample/project
     direction and presence onto the 2D strip.

3. Remove grid-search 3D Lasagna direction decoding.
   - Delete or stop exposing `decode_lasagna_direction_3x2_grid_search(...)`
     and the Fibonacci/unit-sphere candidate table path.
   - Add `decode_lasagna_direction_3x2_analytic(...)` in
     `fiber_trace_3d.direction` or `fiber_trace_3d.projection`.
   - Reuse the established analytic two-channel decode:
     - `fiber_trace_3d.direction.decode_lasagna_direction_2d(...)`;
     - `fiber_trace_2d.direction.decode_lasagna_direction_xy(...)`;
     - Lasagna `_decode_dir_angle(...)` / `_estimate_normal(...)` logic in
       `lasagna/preprocess_cos_omezarr.py`.
   - The 3D decoder should:
     - analytically decode each two-channel projection to a 2D ambiguous unit
       direction using `theta = 0.5 * atan2(sin2theta, cos2theta)`;
     - reconstruct the 3D ambiguous axis from the three projection directions
       using the established Lasagna 3-plane reconstruction/sign-alignment
       logic, not a sampled sphere;
     - be torch-vectorized so projected Trace2CP inference can stay on GPU;
     - preserve sign ambiguity: `v` and `-v` remain equivalent.
   - Add tests that compare the analytic decoder round-trip against
     `encode_lasagna_direction_3x2(...)` over representative 3D axes,
     including near-axis/degenerate projections.

4. Add `_evaluate_trace2cp_metric_fixed_set_3d(...)`.
   - Mirror the 2D fixed-set evaluator.
   - For each selected held-out sample, use the 2D loader to build the CP-to
     next-CP Trace2CP segment.
   - Run 3D block inference/projection.
   - Score with `score_trace2cp_projected_fields(...)`.
   - Skip invalid segments with counted skip diagnostics; fail only if no valid
     segment remains.

5. Wire 3D training evaluation.
   - At each configured test interval, keep dense test loss as diagnostics.
   - If `test_trace2cp_enabled`, also compute `test/trace2cp_error`,
     `test/trace2cp_raw_y_error_mean_px`, segment count, and skipped count.
   - Use `test/trace2cp_error` as the checkpoint metric when available.
   - Log the metric to TensorBoard and stdout.
   - Store `metric_name="test/trace2cp_error"` in `current.pt` and `best.pt`.

6. Add 3D CLI inspection.
   - Reuse the same evaluator and projection code path as training.
   - For single-pair mode, export a compact JPG with:
     - source side-strip image;
     - projected 3D direction overlay;
     - projected 3D presence image;
     - trace overlay and metric text.
   - For `--fiber-json`, evaluate each adjacent CP pair and stitch a compact
     whole-fiber visualization, skipping invalid pairs but reporting counts.

7. Add integration coverage for the real wiring.
   - Do not rely only on the synthetic bridge unit test.
   - Add a small fake-model/fake-loader integration test for metric evaluation
     and best-metric selection, or a minimal real-loader smoke test if cheap.
   - Cover `test_trace2cp_control_points: 0` full-held-out-set behavior.

8. Profile/report obvious performance risks.
   - Report dense block count/size for Trace2CP inference.
   - Assert/report that Trace2CP projection is using analytic 3x2 decode, not
     grid search.
   - If tiling is incomplete or constrained, fail loudly or log the exact
     limitation; do not silently assume one block is enough.

9. Update checked-in 3D configs.
   - Add explicit Trace2CP keys to `loader_example.json`.
   - Add explicit Trace2CP keys to `train_s1a_nml_all.json`.
   - Keep values minimal and deterministic; no augmentations are used by this
     metric path.

## Spec Update

Update `planning/specs.md` to state:

- 3D test evaluation supports public Trace2CP by projecting 3D direction and
  presence outputs onto 2D Trace2CP side-strip coordinates.
- 3D Trace2CP uses existing 2D loader geometry only for evaluation; 3D training
  samples remain CP-centered volume patches.
- When enabled, 3D best checkpoint selection uses averaged
  `test/trace2cp_error`.
- `test_trace2cp_control_points: 0` means evaluate the full held-out CP set.
- The metric path performs no training augmentations.
- The 3D projection decoder must use analytic Lasagna 3x2 decoding. Unit-sphere
  candidate/grid-search decoding is not allowed in Trace2CP projection.
- Any simplified/deferred/unsupported Trace2CP behavior must be reported in the
  task log and final response.

## Docs Updates

Update `docs/code_structure.md` with:

- where the 3D Trace2CP train/CLI evaluator lives;
- how the 2D geometry loader and 3D dense block inference interact;
- which config keys are required.
- replace the current `fiber_trace_3d/projection.py` grid-search description
  with the analytic Lasagna 3x2 decode/reconstruction path once implemented.

## Tests

Add/extend focused tests:

- 3D Trace2CP config parsing and defaults.
- Coordinate conversion from 2D Trace2CP `coords_xyz` to 3D selected-level
  inference coordinates.
- Synthetic block projection still scores expected `trace2cp_error`.
- A small fake model/fake loader integration confirms training metric selection
  prefers `test/trace2cp_error` over dense loss when enabled.
- CLI smoke test for single-pair projection visualization can run with a small
  synthetic fixture or mocked loader/model path.

Regression command:

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py vesuvius/tests/neural_tracing/test_fiber_trace_2d_loader.py vesuvius/tests/neural_tracing/test_fiber_trace.py`

## Other Deferred Or Ignored Items From The Previous Follow-Up

- Full 3D Trace2CP train-loop wiring: not implemented; this plan fixes it.
- 3D Trace2CP CLI visualization: not implemented; this plan fixes it.
- Real-loader integration test for 3D Trace2CP: not implemented; this plan adds
  at least a focused integration or mocked smoke test.
- Analytic/direct 3D Lasagna direction decode: not implemented in the 3D bridge
  yet. Current projection uses an inappropriate deterministic grid-search
  decoder. This plan requires removing that path and using the established
  analytic Lasagna projection decoding/reconstruction logic.
- Anisotropic blur remains opt-in and default off. It was implemented but not
  profiled enough to enable by default.
- Shear/skew and ringing remain intentionally unsupported and should continue
  to fail loudly.

## Risks

- Long CP-pair strips may require tiling more than one 3D inference block.
  Implement this explicitly rather than assuming one block always covers the
  segment.
- The 2D Trace2CP loader returns base-volume XYZ coordinates; the 3D bridge
  must convert to the selected zarr level consistently with `base_volume_scale`.
- Whole-fiber CLI visualization should skip bad CP pairs, but training metric
  should fail if all selected pairs are invalid.
