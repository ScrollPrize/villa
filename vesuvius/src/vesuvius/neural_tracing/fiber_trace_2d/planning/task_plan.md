# Correct Fiber 3D Tiled Inference Output Plan

## Scope

Targets:

- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`
- `vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py`
- shared helpers in `lasagna/tiled_predict3d.py` only where the generic
  adapter boundary is missing pyramid/manifest hooks
- tests in `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- docs/specs under `vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/`

Do not change Lasagna `preprocess_cos_omezarr.py predict3d` behavior except
for reusable helper exports required by fiber inference.

## Current Code Facts

- Current Lasagna `predict3d` stores normals as separate `grad_mag`, `nx`, and
  `ny` OME-Zarr groups and records them in `.lasagna.json`.
- Current Lasagna `predict3d` encodes the hemisphere vector by flipping the
  estimated vector when `z < 0`, then writing:
  `nx_u8 = round(nx * 127 + 128)` and
  `ny_u8 = round(ny * 127 + 128)`, clipped to `uint8`.
- Current Lasagna `predict3d` builds scalar OME-Zarr pyramids for scalar
  channels and uses `build_normal_omezarr_pyramid(nx_path, ny_path, ...)` for
  the paired normal/angle channels.
- Current fiber inference adapter writes one raw seven-channel product per
  option:
  `dir0_z`, `dir1_z`, `dir0_y`, `dir1_y`, `dir0_x`, `dir1_x`, `presence`.
  That is useful internally but is not the desired persisted output format.

## Design

- Keep the model-facing schema unchanged: the 3D fiber model still emits six
  Lasagna 3x2 ambiguous direction channels plus one presence channel per
  option.
- Change the persisted fiber product schema to Lasagna-style channels:
  `presence`, `nx`, and `ny` per output option.
- Convert each option's six direction channels to one ambiguous 3D axis with
  `decode_lasagna_direction_3x2_analytic(...)`, resolve the sign by `z >= 0`,
  and encode `nx/ny` exactly like Lasagna predict3d.
- Encode presence as `round(clamp(presence, 0, 1) * 255)` uint8.
- Treat each option's `presence/nx/ny` triplet as one coherent product. This
  is only an adapter schema change: the existing shared tiled runner must keep
  doing resume/skipping/writing exactly as it already does for Lasagna.
- Write each channel as an OME-Zarr group with levels from the data level to
  `--levels - 1`, just like Lasagna `_open_or_create_omezarr(...)`.
- Build coarser pyramids after data-level inference:
  - scalar mean-pool pyramid for `presence`;
  - paired normal pyramid for `nx/ny`.
- Write a `.lasagna.json` manifest using `LasagnaVolume` and `ChannelGroup`.
  For single-option output, use group names `presence`, `nx`, `ny`.
  For multi-option output, use stable option-prefixed group/channel names such
  as `option_000_presence`, `option_000_nx`, `option_000_ny`.
- Keep a small fiber-specific metadata JSON only if needed for non-Lasagna
  fields such as model checkpoint/config/options. The authoritative spatial
  data manifest must be `.lasagna.json`.

## Implementation Steps

1. **Add Lasagna-style fiber product schema**
   - Replace the persisted `FIBER_TRACE_3D_OPTION_CHANNELS` product layout in
     the inference adapter with `presence`, `nx`, and `ny` output channels per
     option.
   - Keep the model-output splitting as seven internal channels.
   - Add explicit internal-to-output conversion helpers:
     `fiber_option_to_presence_nx_ny_uint8(...)`.

2. **Implement direction conversion**
   - Decode six-channel model direction output with
     `decode_lasagna_direction_3x2_analytic(...)`.
   - Flip sign where decoded `z < 0`.
   - Encode `nx/ny` using the current Lasagna formula:
     `clip(round(component * 127.0 + 128.0), 0, 255).astype(uint8)`.
   - Add tests for both equivalent signs and edge cases near horizontal axes.

3. **Create OME-Zarr groups for fiber channels**
   - Reuse `_open_or_create_omezarr(...)` from `lasagna.tiled_predict3d` or
     promote it to the public shared helper set if needed.
   - Ensure `--scaledown` and input zarr level determine the data level exactly
     as today: `effective_output_sd = input_scaledown * scaledown`;
     `data_level = log2(effective_output_sd)`.
   - The data-level shape must be the OME-Zarr shape for that effective
     scaledown. Coarser pyramid levels are derived from the data level.

4. **Add fiber `.lasagna.json` manifest writing**
   - Use `LasagnaVolume` with the resolved `source_to_base` and
     `base_shape_zyx`.
   - Add configured crop metadata when `--crop` is used.
   - Record channel groups pointing at the data-level arrays for every
     option/channel.
   - Use backup behavior compatible with existing Lasagna manifest writes.

5. **Wire pyramid construction**
   - After tiled inference, call scalar pyramid builder for each presence
     group.
   - Call `build_normal_omezarr_pyramid(nx_path, ny_path, data_level, ...)`
     for each option's `nx/ny` pair.
   - Use the same crop-aware pyramid rebuild policy as `predict3d`, including
     `scan_existing_source_chunks=True`.

6. **Use the existing shared runner resume path unchanged**
   - Do not implement a new fiber resume loop, skip loop, atomic writer, done
     marker, or resume policy.
   - The only fiber-specific change is the adapter product schema:
     `product_chunk_complete(...)` for a fiber option checks the three
     persisted sibling chunks `presence`, `nx`, and `ny`.
   - Once that schema is set, the shared tiled runner must handle chunk
     skipping, recomputation of incomplete chunks, and atomic writes through
     the same code path already used by Lasagna.

7. **Update CLI and metadata wording**
   - Keep existing fiber CLI arguments compatible where possible.
   - Rename or document `--output-prefix` as the Lasagna channel-name prefix,
     not a raw seven-channel bundle prefix.
   - Remove/replace `data_level_only` manifest wording.
   - Update startup output to report data level and pyramid levels.

8. **Tests**
   - Update existing fiber adapter tests from seven persisted channels to
     three persisted channels per option.
   - Add conversion tests proving:
     - six-channel direction outputs are decoded and encoded as Lasagna
       `nx/ny`;
     - `v` and `-v` persist identically after the `z >= 0` sign fold;
     - presence fixed-point encoding maps `0.0 -> 0`, `1.0 -> 255`.
   - Add end-to-end CPU inference test verifying:
     - `.lasagna.json` exists;
     - expected OME-Zarr groups/levels exist;
     - presence, nx, ny data-level chunks are written;
     - coarser pyramid metadata/arrays exist when `--levels > data_level + 1`.
   - Add resume test where one of `presence/nx/ny` is missing and the option
     product is treated incomplete.

## Spec Update

Update `planning/specs.md`:

- Replace the fiber V0 seven-channel persisted output spec with:
  model-internal seven channels, persisted Lasagna-style
  `presence/nx/ny` channels.
- State that fiber inference writes `.lasagna.json` and OME-Zarr scale-space
  pyramids like Lasagna `predict3d`.
- Document presence fixed-point encoding.
- Document the `nx/ny` sign fold and uint8 formula.
- Reclassify existence-based resume and cubic tiled inference as required
  behavior, not limitations. State explicitly that fiber inference must reuse
  the shared runner resume path rather than reimplementing it.
- Clarify output resolution:
  the data level is the configured inference output resolution; pyramid levels
  are derived from it.

## Docs Updates

- Update `fiber_trace_2d/docs/code_structure.md` for the new persisted fiber
  inference output layout.
- Update Lasagna/shared inference docs only where they describe adapter
  behavior shared by both products.
- Add a CLI example that points users to the `.lasagna.json` output.
- Update `planning/changelog.md` when implementation lands.

## Validation Commands

- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
- `python -m py_compile lasagna/tiled_predict3d.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py`

## Non-Goals

- Do not add fiber tracing or Trace2CP inference output.
- Do not change fiber training output/loss semantics.
- Do not change Lasagna cos/normal `predict3d` numeric behavior.
- Do not add non-cubic tiles.
- Do not add any fiber-specific resume implementation, done markers, or
  config-hash resume gating.
- Do not collapse multi-option model outputs unless an explicit later
  postprocessing mode is specified.

## Review Checklist

- Fiber output can be opened through `.lasagna.json`.
- Fiber persisted output is only presence plus compact `nx/ny` angle channels.
- Presence and `nx/ny` have OME-Zarr pyramids.
- Existing Lasagna `predict3d` commands still work.
- Resume remains chunk-existence based.
- Output data level is exactly the configured inference resolution.
