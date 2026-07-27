# Thin Lasagna Port For Fiber 3D Inference Plan

## Goal

Make `vesuvius.neural_tracing.fiber_trace_3d.infer` a thin Lasagna-style
`predict3d` port. Remove the intermediate V0 fiber inference output layer
instead of preserving aliases or compatibility shims for it.

Fiber-specific code may only define:

- fiber model loading;
- raw fiber model output splitting;
- conversion from raw accumulated fiber channels to persisted channels;
- product completeness in terms of the persisted fiber channels.

Everything else must be shared with Lasagna `predict3d` or moved into a shared
Lasagna inference helper.

## Reference Behavior

Use `lasagna/preprocess_cos_omezarr.py predict3d` as the behavior reference:

- output is a `.lasagna.json` manifest plus per-channel OME-Zarr groups;
- output groups are created with `_open_or_create_omezarr(...)`;
- raw model channels are accumulated in rolling z-band accumulators;
- persisted chunks are encoded only at product chunk finalization;
- scalar pyramids use `_build_omezarr_pyramid(...)`;
- paired normal pyramids use `build_normal_omezarr_pyramid(...)`;
- resume is based only on durable output chunk existence;
- temp cleanup and atomic chunk writes are handled by the shared predict3d
  code path.

Lasagna `nx/ny` encoding is the exact reference:

- estimate the 3D ambiguous axis from raw 3x2 direction channels;
- flip equivalent sign so `z >= 0`;
- persist `round(component * 127 + 128)` clipped to `uint8`.

## Current Leftovers To Remove

The current tree still contains V0/intermediate fiber inference artifacts. They
must be removed, not aliased:

- `FiberTrace3DOmeZarrOutputAdapter` in
  `fiber_trace_3d/inference_adapter.py`: duplicate output writer/completeness
  adapter. Replace all use with the shared Lasagna/OME-Zarr output adapter,
  then delete the class and its exports/imports/tests.
- `FIBER_TRACE_3D_OPTION_CHANNELS` as a persisted-output constant:
  currently names the raw seven-channel bundle. Replace with explicit names:
  `FIBER_TRACE_3D_INTERNAL_CHANNELS` for raw model/accumulator channels and
  `FIBER_TRACE_3D_PERSISTED_CHANNELS = ("presence", "nx", "ny")`.
- `product_channel_arrays_from_output(...)` if it remains a direct
  raw-output-to-seven-uint8-bundle conversion. Replace with a finalizer that
  accepts the averaged raw accumulator slab and returns only
  `presence/nx/ny`.
- `fiber_trace_3d/infer.py` custom manifest functions:
  `_atomic_json_write(...)`, `_write_fiber_inference_manifest(...)`,
  `_manifest_relative_path(...)`, `_interval_dict_zyx(...)`, and
  `_output_region_dict_zyx(...)`. Delete them.
- `fiber_trace_3d_inference.json`: stop writing it. No optional provenance
  sidecar in this task.
- Fiber output layout under `fiber/option_000/<raw-channel>`: remove.
- CLI behavior where `--output` is an output directory: remove. Match Lasagna:
  `--output` must be a `.lasagna.json` path. The zarr group names derive from
  the manifest stem, as in Lasagna.
- V0 docs/spec/tests that mention raw seven persisted channels,
  `data_level_only`, `not_lasagna_normal_products`, or the custom manifest:
  update or delete.
- `fiber_trace_3d/__init__.py` exports for removed V0 symbols: delete, not
  alias.

## Shared Boundary

Shared code should own mechanics:

- tile iteration;
- crop handling and output chunk lattice;
- S3/input download handling;
- rolling z-band accumulation;
- output chunk skip/resume;
- temp cleanup;
- atomic chunk writes;
- OME-Zarr group creation;
- pyramid generation;
- `.lasagna.json` manifest writing.

Fiber code should own product semantics:

- load configured fiber checkpoint/model;
- normalize/preprocess tiles in the same way training expects;
- split model output into raw option tensors;
- state the raw accumulator channel count;
- finalize averaged raw option slabs to persisted `presence/nx/ny`;
- define completeness as all three persisted chunks existing.

If the existing shared runner lacks this boundary, add the minimum generic hook
to `lasagna/tiled_predict3d.py`:

- `OutputProductSpec` can have an accumulator channel count that differs from
  persisted channel count;
- model adapters can finalize an averaged raw product slab into a
  `dict[channel_name, np.ndarray]`;
- default finalization remains current direct `clamp(x * 255)` behavior.

Do not create another fiber-specific runner, output adapter, skip policy,
manifest writer, or pyramid writer.

## Fiber Product Mapping

For each output option:

- raw accumulated channels:
  `dir0_z`, `dir1_z`, `dir0_y`, `dir1_y`, `dir0_x`, `dir1_x`, `presence`;
- persisted channels:
  `presence`, `nx`, `ny`;
- persisted channel paths:
  - single option: `<manifest_stem>_presence.ome.zarr`,
    `<manifest_stem>_nx.ome.zarr`, `<manifest_stem>_ny.ome.zarr`;
  - multi-option: `<manifest_stem>_option_000_presence.ome.zarr`,
    `<manifest_stem>_option_000_nx.ome.zarr`,
    `<manifest_stem>_option_000_ny.ome.zarr`, etc.

Finalization:

- `presence_u8 = round(clamp(raw_presence, 0, 1) * 255)`;
- use the established Lasagna analytic 3x2 direction implementation; do not
  add a grid search or a new decoder;
- sign-fold by `z >= 0`;
- encode `nx/ny` with the Lasagna formula above.

Do not collapse branch/recurrent options. Each option remains its own
`presence/nx/ny` product triplet.

## Implementation Steps

1. **Make the shared runner support raw-finalize-persist products**
   - Add the accumulator-channel-count/finalizer hook in
     `lasagna/tiled_predict3d.py`.
   - Keep default behavior byte-compatible for existing direct products.
   - Do not change Lasagna `predict3d` numeric behavior.

2. **Move generic OME-Zarr output adapter to shared code**
   - Promote the Lasagna OME-Zarr output adapter behavior to the shared
     predict3d module, or expose an equivalent shared helper there.
   - Update Lasagna `preprocess_cos_omezarr.py` to use the shared adapter if
     needed.
   - Delete `FiberTrace3DOmeZarrOutputAdapter`; do not keep an alias.

3. **Replace fiber adapter output schema**
   - Rename constants to distinguish raw internal vs persisted output channels.
   - Product specs expose only persisted `presence/nx/ny`.
   - Product specs set raw accumulator channel count to seven.
   - Product specs use scalar pyramid policy for `presence` and custom/normal
     policy for `nx/ny` as required by shared pyramid dispatch.

4. **Implement fiber finalizer**
   - Accept the averaged raw seven-channel slab.
   - Return only `presence`, `nx`, `ny` arrays.
   - Match Lasagna's 3x2 normal estimation/sign fold/uint8 encoding.
   - Remove any old helper that writes raw seven output channels.

5. **Rewrite fiber inference output setup around `.lasagna.json`**
   - Require `--output` to end in `.lasagna.json`.
   - Derive output directory and zarr name prefix from the manifest path,
     matching Lasagna.
   - Create OME-Zarr groups with `_open_or_create_omezarr(...)`.
   - Write `LasagnaVolume`/`ChannelGroup` as the authoritative manifest.
   - Delete the custom `fiber_trace_3d_inference.json` writer.

6. **Build pyramids through existing Lasagna helpers**
   - Build scalar pyramids for every `presence` group.
   - Build paired normal pyramids for each `nx/ny` pair.
   - Use crop-aware `scan_existing_source_chunks=True`.
   - Remove any V0 `data_level_only` behavior.

7. **Update imports and public surface**
   - Remove V0 exports from `fiber_trace_3d/__init__.py`.
   - Update tests and internal imports to use new names.
   - Do not leave compatibility aliases for removed V0 names.

8. **Update docs/specs/tests**
   - Replace V0 raw bundle docs and specs.
   - Delete or rewrite tests that assert V0 behavior.
   - Add tests for raw internal split, persisted schema, finalization,
     completeness, `.lasagna.json`, and pyramids.

## Spec Update

Update `planning/specs.md`:

- Fiber 3D inference is a thin Lasagna `predict3d` port.
- The authoritative output is `.lasagna.json`; no custom primary manifest.
- `--output` is a `.lasagna.json` path.
- Raw seven-channel fiber model output is internal only.
- Persisted fiber output is only `presence`, `nx`, `ny`.
- Raw channels are accumulated before final encoding.
- Presence is uint8 fixed point with `0 == 0.0`, `255 == 1.0`.
- `nx/ny` use Lasagna's compact ambiguous hemisphere encoding.
- Coarser pyramids are required.
- Shared predict3d owns resume/temp/atomic/chunk/pyramid mechanics.
- No legacy V0 aliases or output compatibility shims should remain.

## Docs Updates

Update `fiber_trace_2d/docs/code_structure.md`:

- Replace the V0 fiber inference description.
- Document the thin Lasagna port and shared ownership boundary.
- Show a CLI example writing a `.lasagna.json` output.
- Document the resulting output groups.

Update `planning/changelog.md` after implementation.

## Tests

Update or add focused tests:

- fiber adapter product schema persists only `presence/nx/ny`;
- raw seven channels still split correctly per option internally;
- finalizer maps presence `0.0 -> 0` and `1.0 -> 255`;
- finalizer maps equivalent `v` and `-v` directions to identical `nx/ny`;
- product completeness requires all three persisted chunks;
- V0 symbols/classes are absent from the public fiber package;
- end-to-end CPU inference writes:
  - `.lasagna.json`;
  - OME-Zarr groups for `presence`, `nx`, `ny`;
  - data-level chunks;
  - coarser pyramid levels;
- Lasagna `predict3d` tests still pass.

Validation commands:

- `python -m py_compile lasagna/tiled_predict3d.py lasagna/preprocess_cos_omezarr.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/infer.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/inference_adapter.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/__init__.py`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "fiber_infer or fiber_inference_adapter or fiber_output_adapter"`
- `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`

## Non-Goals

- No fiber tracing output.
- No Trace2CP output.
- No done markers.
- No fiber-specific pyramid implementation.
- No encoded `nx/ny` accumulation.
- No change to Lasagna `predict3d` outputs or numerics.
- No branch collapse or option averaging.
- No compatibility shim for the V0 raw-bundle output.

## Review Checklist

- No `FiberTrace3DOmeZarrOutputAdapter` class or export remains.
- No custom `fiber_trace_3d_inference.json` writer remains.
- No persisted raw seven-channel fiber output remains.
- No `data_level_only` fiber inference wording remains.
- Fiber output opens through `.lasagna.json`.
- Existing Lasagna `predict3d` remains compatible.
- Any remaining fiber-specific code is tied to model loading, raw channel
  interpretation, finalization, or product completeness.
