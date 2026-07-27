# Shared 3D Tiled Inference Plan

## Scope

Targets:

- `lasagna/preprocess_cos_omezarr.py`
- new shared 3D tiled inference module, likely under `lasagna/`
- new 3D fiber inference entry point, likely
  `vesuvius.neural_tracing.fiber_trace_3d.infer`
- tests under `lasagna/tests/` and
  `vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

This is a refactor-plus-extension task. Current Lasagna `predict3d` behavior is
the reference behavior and must remain compatible.

## Current State After Main Merge

- `preprocess_cos_omezarr.py predict3d` already has the rolling z-band
  accumulator and related resume fixes:
  - `_RollingZBand` with per-channel mmap files;
  - startup and finish cleanup of `.predict3d_*` and temp OME-Zarr paths;
  - `_atomic_zarr_write(...)` temp-write then rename;
  - canonical tile-position helpers;
  - per-output-channel chunk completeness checks;
  - model-derived outputs independent from derived `pred_dt`;
  - manifest update and OME-Zarr pyramid rebuild/update.
- `lasagna/tests/test_preprocess_cos_omezarr.py` already covers key pieces of
  that behavior, including rolling-band discard, canonical tile origins,
  temp cleanup, grouped chunk completeness, and `pred_dt` manifest behavior.
- `vesuvius.neural_tracing.fiber_trace_3d` now has real source files:
  `model.py`, `loader.py`, `targets.py`, `direction.py`, `train.py`,
  `trace2cp_bridge.py`, and `trace2cp_tool.py`.
- The fiber 3D model output schema is seven channels per option:
  six Lasagna 3x2 direction channels plus one presence channel.
  Legacy/free-branch and conditioned/recurrent grouped outputs both exist.

## Design Invariants

- Lasagna `predict3d` output values, channel names, manifests, scale handling,
  normalization, model loading, and CLI compatibility must not change.
- The shared runner owns only generic mechanics:
  input/crop/level resolution, canonical global tile lattice, rolling
  accumulation, chunk completeness, resume, temp cleanup, atomic writes,
  progress, missing-input skip, and common CLI argument definitions.
- Product-specific adapters own all model and output semantics.
- Resume state is durable output chunks only. Accumulator mmap files are
  scratch.
- Output products are independent. Missing `pred_dt` schedules only derived
  `pred_dt` generation. Missing fiber option chunks schedule only that fiber
  product/option, not unrelated products.
- Global chunk and tile origins are crop-independent. Crops select which global
  chunks to produce; they do not change the lattice.
- Existing chunk files are never modified unless their product is incomplete or
  explicitly incompatible.
- Every channel chunk write remains atomic on the same filesystem.
- Fiber multi-option inference keeps each option coherent. A seven-channel
  option is indivisible for completeness and for branch/conditioned output
  semantics.

## Implementation Plan

1. **Extract shared data structures without behavior changes**
   - Introduce a shared module such as `lasagna/tiled_predict3d.py`.
   - Move or wrap generic helpers from `preprocess_cos_omezarr.py`:
     - OME-Zarr chunk path/existence helpers;
     - temp cleanup;
     - atomic chunk writes;
     - canonical tile/grid helpers;
     - rolling z-band accumulator;
     - tiled inference loop;
     - common progress formatting.
   - Keep old function names imported/re-exported where tests or scripts rely
     on them.

2. **Define adapter interfaces**
   - Add small, explicit adapter dataclasses/protocols:
     - `OutputProductSpec`: name, level/scaledown, channels, chunk size,
       dtype/range, pyramid policy;
     - `ModelAdapter`: load model, run tile inference, convert raw tile output
       to logical product accumulators;
     - `OutputAdapter`: product completeness, chunk postprocess/write,
       manifest/group update.
   - Keep the interface concrete and minimal; avoid a large framework.

3. **Port Lasagna cos predict3d to the adapter**
   - Implement a `LasagnaCosPredict3DAdapter`.
   - Preserve current products:
     - fine `cos`;
     - coarse `grad_mag`, `nx`, `ny` bundle;
     - optional derived `pred_dt`.
   - Preserve `grad_mag_factor`, normal estimation, manifest groups, crop
     recording, and pyramid generation.
   - Ensure the old `preprocess_cos_omezarr.py predict3d` CLI remains a thin
     compatibility wrapper with the same arguments and defaults.

4. **Add fiber 3D inference adapter**
   - Implement a `FiberTrace3DPredictAdapter` using:
     - `build_fiber_trace_3d_model(...)`;
     - `direction_outputs(...)` / `presence_outputs(...)`;
     - `encode/decode_lasagna_direction_3x2` helpers where needed;
     - training checkpoint/config metadata to infer architecture and channel
       layout.
   - Support single-option output and grouped multi-option output.
   - In conditioned mode, define the requested inference grouping explicitly:
     default compatibility `forward(volume)` output, plus an optional recurrent
     grouped mode if configured by CLI/config.
   - Write each option as a coherent product containing seven logical channels:
     `dir0_z`, `dir1_z`, `dir0_y`, `dir1_y`, `dir0_x`, `dir1_x`, `presence`.
   - Product completeness requires all seven channel chunks for the option.

5. **Add fiber CLI**
   - Add `python -m vesuvius.neural_tracing.fiber_trace_3d.infer`.
   - Reuse common arguments from the shared runner where applicable.
   - Add fiber-specific arguments only for model config/checkpoint semantics:
     e.g. `--config`, `--recurrent-steps`, `--output-prefix`, and explicit
     product selection if needed.
   - Use the same input zarr, crop, S3 download, temp cleanup, resume, atomic
     write, and progress behavior as Lasagna predict3d.

6. **Preserve crop-composable output**
   - Keep output chunk selection global and chunk-aligned.
   - Compute contributing tile origins from the global lattice for both
     Lasagna and fiber adapters.
   - Add tests that the same output chunk selected through two crops uses the
     same contributing tile origins and produces byte-identical output with a
     deterministic fake model.

7. **Pyramid and manifest behavior**
   - Keep Lasagna manifest output exactly compatible for cos predict3d.
   - Define fiber output metadata explicitly:
     either a lightweight JSON manifest or a Lasagna-style manifest with fiber
     group names. Do not pretend fiber outputs are Lasagna normals.
   - Build OME-Zarr pyramids according to product type:
     scalar pyramid for presence;
     direction-channel pyramid only if the product semantics are defined and
     tested. Otherwise document data-level-only output as a non-goal for V0.

8. **Backwards-compatible migration**
   - Update existing Lasagna tests to import shared helpers through the old
     compatibility names first, then add direct shared-module tests.
   - Keep `preprocess_cos_omezarr.py` runnable as before.
   - Avoid changing any current 3D training or Trace2CP code paths.

## Spec Update

Add a shared inference section documenting:

- `predict3d` rolling z-band behavior is current baseline;
- shared tiled inference mechanics and adapter boundaries;
- global/crop-independent tile and output chunk lattices;
- output chunks as the only resume/completeness state;
- product-independent completeness and resume;
- atomic temp-write then rename semantics;
- no done markers;
- missing-input skip semantics;
- fiber inference output schema: seven channels per option, with Lasagna 3x2
  direction channels plus presence;
- multi-option fiber output preservation;
- fiber outputs are not `grad_mag/nx/ny`;
- compatibility requirement for `preprocess_cos_omezarr.py predict3d`.

## Docs Updates

- Update Lasagna predict3d docs to describe it as a wrapper around shared
  tiled inference, while keeping current user-facing CLI examples.
- Add fiber 3D inference docs with:
  - CLI command;
  - required config/checkpoint/input/output arguments;
  - output channel/group names;
  - resume behavior;
  - crop-composable output rule;
  - how multi-option outputs are stored.
- Update developer docs/code-structure notes to point shared mechanics to the
  new module and product-specific logic to adapters.
- Update `planning/changelog.md` when implementation lands.

## Tests

- Existing Lasagna tests must continue passing:
  `PYTHONPATH=lasagna:vesuvius/src:. pytest -q lasagna/tests/test_preprocess_cos_omezarr.py`
- Add shared-runner unit tests:
  - canonical tile support is crop-independent;
  - product completeness handles missing sibling channel chunks;
  - atomic write cleanup/install behavior;
  - temp cleanup removes stale artifacts but not live-process temp files.
- Add Lasagna adapter regression tests:
  - existing CLI wrapper reaches the same manifest/product setup;
  - missing `pred_dt` does not schedule neural inference when model outputs
    are complete;
  - incomplete `grad_mag/nx/ny` bundle recomputes only the coarse product.
- Add fiber adapter tests:
  - single-option seven-channel mapping;
  - multi-option grouped channel mapping;
  - conditioned/recurrent grouped output mapping when enabled;
  - product completeness requires all seven option chunks;
  - resume with one missing fiber channel recomputes the option chunk;
  - overlapping crops produce byte-identical chunks with a deterministic fake
    fiber model.
- Run focused fiber 3D tests after adding the CLI:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`

## Changelog Update

- Record that shared tiled 3D inference was extracted from Lasagna predict3d
  and reused by fiber 3D inference while preserving predict3d resume/crop
  semantics.

## Non-Goals

- Do not change Lasagna normal encoding.
- Do not change cos predict3d numeric output semantics.
- Do not change 3D fiber training, target generation, or Trace2CP tracing.
- Do not collapse fiber branches/options.
- Do not introduce done markers.
- Do not implement a new independent resume/temp/chunk writer in the fiber CLI.

## Review Checklist

- Existing predict3d CLI remains compatible.
- Existing predict3d tests pass.
- Rolling accumulator behavior is preserved, not redesigned again.
- Shared runner has no Lasagna-specific output assumptions.
- Fiber adapter has no duplicated resume/temp/chunk lattice implementation.
- Fiber outputs preserve seven-channel option grouping.
- Crop-composable output invariants are tested for both adapters.
