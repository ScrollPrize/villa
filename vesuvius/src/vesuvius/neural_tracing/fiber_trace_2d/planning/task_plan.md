# VC3D BBox Dependency Metadata For 3D Prefetch Plan

## Current State

- `volume-cartographer/python/vc/volume.cpp` already has:
  - `collectChunkKeys(volume, offset, shape, level)`, which converts a
    selected-level ZYX bbox to chunk keys;
  - `collectCoordsDependencies(...)`, which converts chunk keys to the
    metadata dictionaries Python prefetch needs.
- `fiber_trace_3d.loader` now computes a CP-centered augmentation-envelope bbox
  but still creates representative chunk-center coordinates so it can call the
  coordinate dependency API.
- That representative-coordinate layer is unnecessary and leaves chunk-range
  logic in Python.

## Implementation

### VC3D Binding

- Factor the metadata-emission logic in `collectCoordsDependencies(...)` into a
  shared helper, for example:
  - input: `Volume&`, `std::vector<vc::render::ChunkKey>`;
  - output: `nb::list` with the same dict fields as today:
    `level`, `iz`, `iy`, `ix`, `key`, `valid`, `remote_chunk_key`,
    `remote_url`, `cache_path`, `empty_path`, `persistent_extension`,
    `cache_payload_format`, and `source_payload_matches_cache`.
- Add a new binding in `volume.cpp`, preferably:
  - `collect_bbox_dependencies(offset, shape, level=0)`
  - `offset` and `shape` are selected-level ZYX coordinates, matching existing
    `prefetch_zyx`, `read_zyx`, and `collectChunkKeys(...)` semantics.
- The binding should:
  - validate non-negative `level` and present scale level through the existing
    `collectChunkKeys(...)` path;
  - clamp boxes against the selected-level volume shape exactly as
    `prefetch_zyx` does;
  - return an empty list for empty/out-of-volume boxes;
  - release the GIL only around the chunk-key collection if needed, then build
    Python dicts after reacquiring it;
  - return the same metadata schema and cache-path authority as
    `collect_coords_dependencies`.
- Do not add Python-side cache path reconstruction or remote key generation.

### Python Sampler

- Add `chunk_requests_for_bbox(start_zyx, end_zyx)` to the coordinate sampler
  abstraction.
- For `Vc3dCoordinateSampler`:
  - convert half-open `start/end` selected-level ZYX bbox to `offset/shape`;
  - call `volume.collect_bbox_dependencies(offset, shape, level)`;
  - reuse the same dependency-dict-to-`ZarrChunkRequest` conversion as
    `chunk_requests_for_coords`;
  - if the binding is missing, fail clearly with a “rebuild
    volume-cartographer” message.
- For `NumpyZarrCoordinateSampler`:
  - keep local prefetch as a no-op returning `[]`, matching current local-array
    behavior.
- Keep `chunk_requests_for_coords(...)` for 2D strip prefetch and other
  coordinate-surface use cases.

### 3D Prefetch Loader

- Replace `_prefetch_envelope_chunk_center_coords_base(...)` with an envelope
  bbox path:
  - compute the CP-centered selected-level augmentation-envelope bbox;
  - clamp it to the selected-level volume shape;
  - compute `valid_voxels` from the clamped selected-level bbox volume;
  - call `record.sampler.chunk_requests_for_bbox(start, end)`.
- Remove Python-side chunk conversion from 3D prefetch:
  - no representative chunk-center coordinate materialization;
  - no `record.volume.chunks` / `volume.chunk_shape(...)` dependency in the
    3D loader;
  - no `chunk_requests_for_coords(...)` call for 3D prefetch.
- Remove the intermediate `prefetch_sampler_device` config key and docs/tests,
  because direct bbox dependency generation does not materialize torch
  coordinate tensors.
- Keep:
  - deterministic sample ordering and `idx` semantics;
  - augmentation-envelope semantics based on configured extrema;
  - `prefetch_sampler_workers` for parallel bbox dependency producers;
  - Python cache-hit / `.empty` / atomic download handling;
  - no Lasagna-channel prefetch.
- Update prefetch startup output to keep `mode=augmentation_envelope` but drop
  sampler-device reporting.

## Spec Update

- Update `planning/specs.md` to say 3D prefetch passes selected-level
  augmentation-envelope bboxes to VC3D dependency metadata collection.
- Clarify that VC3D owns bbox-to-chunk conversion for 3D prefetch.
- Remove the spec text that allows/mentions `prefetch_sampler_device` for 3D
  dependency generation.
- Keep the augmentation-sample-independent requirement: configured augmentation
  extrema define the bbox; one random draw must not decide chunks.

## Docs Updates

- Update `planning/local_development.md`:
  - document that 3D prefetch uses `collect_bbox_dependencies`;
  - remove `prefetch_sampler_device` from the documented workflow;
  - keep `prefetch_sampler_workers` and `prefetch_workers` descriptions.
- Update `docs/code_structure.md` where 3D prefetch internals are described.
- Add a changelog entry for the VC3D bbox dependency binding and 3D prefetch
  cleanup.
- If volume-cartographer has local binding/development notes near the changed
  code, update only if a relevant doc already exists.

## Tests

### Vesuvius Python Tests

- Add/adjust tests so 3D prefetch:
  - calls `chunk_requests_for_bbox(...)`, not `chunk_requests_for_coords(...)`;
  - does not call `_sample_augment_params(...)`;
  - returns stable requests for different raw augmentation indices mapping to
    the same bounded data CP;
  - expands the bbox when augmentation extrema expand;
  - treats local `NumpyZarrCoordinateSampler` prefetch as no-op.
- Remove tests for `prefetch_sampler_device`.
- Add sampler wrapper tests that fake a VC3D volume exposing
  `collect_bbox_dependencies(...)` and assert returned metadata converts into
  `ZarrChunkRequest` correctly.

### Volume-Cartographer Tests

- Add a binding-level test if the existing Python binding test harness supports
  it. At minimum, add a focused C++/binding smoke that checks:
  - empty/out-of-bounds bbox returns an empty list;
  - an in-bounds bbox returns the same metadata keys as coordinate dependency
    collection for equivalent chunk coverage;
  - returned dicts include cache and remote metadata fields.
- If no lightweight binding test harness exists, document that limitation in
  the task log and cover the wrapper path from Vesuvius with a fake binding.

### Commands

- Run the focused 3D prefetch/dependency tests:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py -k 'prefetch or dependency'`
- Run the full 3D fiber test file:
  `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 PYTHONPATH=vesuvius/src:. pytest -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py`
- Rebuild volume-cartographer bindings after the C++ change using the documented
  local-development command, then run the smallest available VC3D test target
  that covers the binding/chunk dependency path.
- Run `python -m py_compile` for touched Python modules and `git diff --check`.

## Non-Goals

- Do not change augmentation semantics or training sample ordering.
- Do not change 2D prefetch, which still needs coordinate-surface dependency
  discovery.
- Do not reconstruct VC3D cache paths in Python.
- Do not add GPU dependency-generation support for this bbox path; it is no
  longer relevant once coords are not materialized.
