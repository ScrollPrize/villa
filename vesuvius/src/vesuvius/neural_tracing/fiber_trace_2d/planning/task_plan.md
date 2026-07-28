# Plan: Native 3D Trace2CP Hot-Path Acceleration

## Goals

- Remove the largest avoidable cost in native 3D Trace2CP:
  per-block dense coordinate-grid construction plus generic coordinate sampling.
- Keep the trained 128-cube / 48-core-margin setup intact; do not optimize by
  changing patch size, margin, trace scoring, model resolution, or volume scale.
- Batch source-block construction and model inference so small repeated forwards
  are not serialized one block at a time.
- Audit remaining tracing stages for batching/vectorization opportunities and
  implement only the low-risk ones in this task.

## Non-Goals

- Do not change trace metrics, restart criteria, fusion logic, candidate scoring
  weights, beam parameters, normalization, or checkpoint/model selection.
- Do not add a fiber-only direct zarr/raw reader. Volume reads must still go
  through the shared VC3D-backed sampler/volume abstraction.
- Do not relax strict requested-level blocking. Scale fallback stays forbidden
  for native 3D Trace2CP inference blocks and visualization reads.

## Current Diagnosis

- `NativeTraceFieldCache._sample_block_volume(...)` builds a dense
  `[D,H,W,3]` coordinate grid for every axis-aligned inference block, multiplies
  it into base coordinates, computes validity, and calls
  `sampler.sample_coord_batch(...)`.
- This path is correct for arbitrary strip/coordinate sampling but wasteful for
  regular 3D inference blocks. The profile shows `src_read + src_coords` is
  about two thirds of measured wall time.
- `sample_point_choices_torch(...)` is partly batched: it groups queried points
  by inferred block and samples each block with `grid_sample`. It still loops
  over unique blocks and moves each cached CPU block to GPU one at a time.
- Candidate scoring is already substantially vectorized in the beam path, but
  cache misses inside point lookup still trigger one block read/inference at a
  time.
- Lasagna normal sampling is already batched per scoring call, but it converts
  candidate points to NumPy and calls the 2D geometry loader once per trace
  step. It is a smaller hotspot and should be treated after source/inference
  batching.

## Implementation Steps

### 1. Add A Shared Axis-Aligned Block Read API

- Extend `CoordinateSampler` with a method for regular ZYX selected-level block
  reads, for example:
  `sample_block_zyx(start_zyx, shape_zyx) -> CoordinateSampleResult`.
- Implement it for `Vc3dCoordinateSampler` using a VC3D `Volume` binding that
  reads an axis-aligned requested-level region through the same chunk cache,
  blocking download/decode, and strict no-fallback behavior.
- Reuse or add a VC3D binding around the existing C++ `readZYX(...)`/blocking
  chunk-copy logic rather than reconstructing zarr paths in Python.
- Preserve the same result contract as coordinate sampling:
  image array, valid mask, and stats including requested-level-only/fallback
  indicators.
- For `NumpyZarrCoordinateSampler`, either implement an equivalent direct slice
  path for local tests or raise clearly if the test surface cannot support it.

### 2. Switch Native Trace Field Blocks To Direct Block Reads

- Update `NativeTraceFieldCache._sample_block_volume(...)` to use the new
  block-read method for axis-aligned inference blocks.
- Keep `_block_origin_for_point(...)`, trusted-core routing, and the 128/48
  patch-margin behavior unchanged.
- Keep selected-level semantics unchanged: `origin_zyx` and `patch_shape_zyx`
  are selected-level voxel coordinates, and the sampler reads the configured
  zarr level.
- Keep validity handling strict:
  out-of-volume samples become invalid/zero; chunk errors raise; true missing
  chunks may be invalid/zero; requested-level fallback remains an error.
- Retain profiler stages, but split block-read timing into useful rows such as
  `src_read_block` and `src_tensor`; `src_coords` should disappear or fall near
  zero for this path.

### 3. Batch Missing Inference Blocks

- Add a cache method that accepts many query points, computes all missing block
  origins, and materializes those missing blocks in batches.
- Batch source block reads where the sampler/VC3D API permits it. If VC3D only
  exposes one block read per call, still collect missing origins first and keep
  the loop isolated so later C++ batching can replace it.
- Batch model forwards by stacking multiple raw blocks into `[B,1,D,H,W]` before
  preprocessing/inference/decode/cache-crop.
- Add a config/CLI-controlled maximum inference block batch size if needed to
  avoid OOM with 128³ blocks. Default conservatively, then measure.
- Keep cached inferred blocks CPU-resident after decode. GPU tensors are
  temporary for the batched forward only.

### 4. Vectorize/Batch Point Lookup Where It Is Still Cheap

- In `sample_point_choices_torch(...)`, keep grouping points by inferred block,
  but first call the new batch-materialization method so cache misses do not
  occur inside the per-block lookup loop.
- Reuse per-block GPU transfer within one lookup call. If multiple point groups
  hit the same block in one call, transfer that block once.
- Keep decoded direction/presence semantics unchanged: grouped 7-channel options
  stay selected jointly by valid/presence/reference alignment.
- Do not attempt a complex global packed-3D texture cache in this task unless
  profiling after steps 1-3 shows lookup now dominates.

### 5. Lasagna Normal Sampling Audit

- Confirm that candidate normals are sampled once per flattened candidate batch,
  not per candidate.
- If the current loader call is the only remaining issue, add a small
  selected-level spatial normal cache keyed by integer/nearby candidate voxel or
  by inference block origin, whichever is simpler and deterministic.
- Do not interpolate normals from the reference line or substitute approximate
  normals. Candidate smoothness normals must still be sampled at candidate
  coordinates when available.

### 6. Scoring/Beam Audit

- Check whether `trace_current_sample` and `field_sample_lookup` can be fused for
  each lookahead expansion: current-point samples and candidate/substep samples
  should be gathered into one point lookup call where semantics permit.
- Keep candidate-score math in torch. Avoid moving candidate losses or branch
  decisions through NumPy except for final beam-node reconstruction.
- Keep the profiler table after changes and add any new stage names clearly.

## Spec Update

Add/modify `planning/specs.md` to state:

- Native 3D Trace2CP axis-aligned inference blocks must use a shared
  VC3D-backed requested-level block-read API, not generic dense coordinate
  sampling.
- Generic `sample_coords` remains the correct boundary for arbitrary strip and
  surface coordinates; direct block reads are only for regular axis-aligned
  selected-level inference blocks.
- Native 3D Trace2CP may batch missing block reads and model forwards, but must
  keep requested-level blocking, no scale fallback, configured normalization,
  CPU-resident inferred-block cache, and trusted-core routing unchanged.
- Candidate normal sampling must remain coordinate-local; batching/caching is
  allowed only when it preserves sampled-at-candidate semantics.

## Docs Updates

- Update the native 3D Trace2CP docs/spec notes to describe the two sampler
  paths:
  axis-aligned inference block reads vs arbitrary coordinate strip sampling.
- Document any new CLI/config key for inference block batch size.
- Update local development notes if a VC3D binding rebuild is required.

## Validation

Functional validation:

- Python compile:
  `python -m py_compile vesuvius/src/vesuvius/neural_tracing/fiber_trace_3d/trace2cp_tool.py vesuvius/src/vesuvius/neural_tracing/fiber_trace_2d/sampling.py`
- Existing Python tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:lasagna:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k "native_3d or whole_fiber_trace"`
- Add or update a small sampler test that verifies direct block read and dense
  coordinate sampling agree on an axis-aligned block for a fixture/local volume.

Performance validation:

- Re-run the same native Trace2CP metric command used to produce the current
  profile, with the same checkpoint, fiber JSON, 128 patch, and 48 margin.
- Compare the profiler table before/after. Expected first-order improvement:
  `src_coords` should become near-zero, `src_read` should drop materially, and
  `inference_forward` count should remain equal or lower while wall time drops
  if batched inference is active.
- Report total wall/cpu time, `src_read`, `src_coords` or replacement stage,
  `inference_forward`, `field_sample_lookup`, and total inferred block count.

## Changelog Update

- Add one concise changelog entry if implementation changes sampler or native
  Trace2CP behavior.

## Risks / Checks

- VC3D direct block reads must honor the configured zarr level, not full-res
  level zero unless the config selected level zero.
- If the direct block API returns integer dtype while coordinate sampling
  returned float, conversion to float must happen at the same boundary as before.
- Batched inference can increase transient GPU memory; batch size needs a safe
  cap for 128³ blocks.
- The direct block path must not be used for side/top strip rendering or other
  arbitrary coordinate surfaces.
