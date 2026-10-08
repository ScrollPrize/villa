# Unified native bbox reads

`Volume.read_zyx` is the single Python-facing C++ bounding-box reader.
`sample_zyx_block` has been removed entirely, without a compatibility alias.
`read_xyz` only reorders its XYZ offset/shape arguments; it returns a ZYX array
through the same implementation. Volume slicing delegates to `read_zyx` at level0.
Arbitrary-coordinate `sample_coords` and affine `sample_planes` are separate
sampling operations and remain unchanged.

## API and exact behavior

```python
values = volume.read_zyx(offset_zyx, shape_zyx, level=1)
values, valid, stats = volume.read_zyx(
    offset_zyx, shape_zyx, level=1, with_mask=True, with_stats=True)
```

Default output is a C-contiguous array in the source's uint8/uint16 dtype. No
interpolation, normalization or float conversion. Both auxiliary flags are
keyword-only and default false. When either is true, the return is a three-tuple;
an unrequested mask or statistics object is `None`.

| Condition | Unified behavior |
|---|---|
| Requested scale group absent | Throw, even for an empty or entirely outside bbox |
| Negative scale | Throw |
| Genuinely missing chunk | Declared Zarr fill value; valid inside logical volume |
| Known all-fill chunk | Declared fill; valid inside logical volume |
| Outside logical volume | Declared fill; invalid mask |
| Fetch/decode error | Throw |
| Unexpected unresolved result from blocking cache read | Throw |
| Data result with null or incorrect-sized decoded payload | Throw |
| Physically padded edge chunk | Copy only logical in-volume voxels; never expose padding |
| Zero-size bbox at an existing level | Correctly shaped empty array |
| Unsupported dtype or oversized shape | Throw |

There is no Python bbox `missing_policy` argument anymore: these reads do not
silently synthesize a missing scale. The general C++ `Volume` resampling policies
are separate APIs and were not changed in this task.

Large allocation/fill, dependency enumeration, prefetch and block copies run
with the GIL released. Python array/tuple/dictionary creation retains it. The
returned array owns the native vector without another bulk copy. With masks
disabled no mask buffer is allocated or written; with statistics disabled no
per-row coverage counters are accumulated. Returned-array destruction still
uses the existing shared capsule destructor; this task does not redesign ownership
or promise zero GIL time during deallocation.

## Callers and intentional behavior changes

- `NativeTileReader` calls the native bbox API directly for the full inference
  tile, starting at `tile_origin - border`. No mask/statistics and no float32
  adapter or integer recast. It checks shape/dtype/contiguity and returns the same
  native array. GPU-side normalization, including uint16 handling, is unchanged.
- Partial inference boundary tiles previously used reflection through
  `_finalize_tile_read`; they now use declared fill, as requested. Boundary outputs
  can therefore differ. This is intentional, not an exact-numerical speedup claim.
- Training's `CoordinateSampler.sample_block_zyx` abstraction now delegates to
  `read_zyx` with auxiliaries. It retains the training-facing float image and
  boolean mask interface; training consumers still handle invalid voxels/masks.
  Separate training augmentation padding is unchanged.
- The row/column inference wrapper's explicit missing-level fill fallback was
  removed. It now requires its configured level to exist.
- Profiling uses the new API, including a same-buffer-size allocation-only control
  with mask enabled. Raw-native and inference-wrapper probes use identical full
  bboxes so boundary clipping cannot distort that comparison.

## Build and validation

Existing native build, GCC13/RelWithDebInfo, no downloaded dependencies:

```bash
cmake -S /home/hendrik/villa2/volume-cartographer -B /tmp/vc-live-native-cmake \
  -DVC_BUILD_PYTHON=ON -DPython_EXECUTABLE=/home/hendrik/.venv_las/bin/python \
  -Dnanobind_DIR=/home/hendrik/.cache/uv/archive-v0/rcrHRTTfYN8hOlRV/nanobind/cmake
cmake --build /tmp/vc-live-native-cmake --target vc_volume -j 8
```

The editable loader selected the old installed binary despite a build-tree
`pythonpath` override. The first launch consequently failed12tests against the
old signature; this was not a run of the new implementation. Updated the actual
environment using the previously approved local-package command:

```bash
CC=/usr/bin/gcc-13 CXX=/usr/bin/g++-13 AGENTS_AGENT_MODE=1 AGENTS_ALLOW_INSTALL=1 \
  uv pip install --python /home/hendrik/.venv_las/bin/python --no-deps --reinstall -e ./volume-cartographer
```

Package rebuild completed in43.89s. Editable project origin changed from the old
`/home/hendrik/villa/volume-cartographer` to this checkout's `villa2` project.
The installed binding confirms the new signature and no `sample_zyx_block`.

```bash
/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/gpu_manager/src /home/hendrik/villa2/vesuvius/src /home/hendrik/villa2' \
  -q vesuvius/tests/neural_tracing/test_native_bbox_read.py \
  lasagna/tests/test_demand_predict3d.py lasagna/tests/test_inference_profiling.py \
  vesuvius/tests/neural_tracing/test_live_inference_profile.py

/home/hendrik/.venv_las/bin/python -m pytest --noconftest \
  -o 'pythonpath=/home/hendrik/villa2/vesuvius/src /home/hendrik/villa2/vesuvius /home/hendrik/villa2' \
  -q vesuvius/tests/neural_tracing/test_fiber_trace_3d.py \
  -k 'coordinate_sampler or native_block or direct_block'
```

Initial full targeted suite76passed25.83s; training-adapter selection3passed,
281deselected4.44s. Expanded native/profiling suite24passed1.77s, including64
concurrent reads from one native Volume, INT_MIN/INT_MAX offsets, and oversized
as well as short corrupt payloads. Independent implementation review passed.
Null-data and unexpected-MissQueued guards are code-reviewed but lack direct
injection coverage through the real-file fixtures. No macOS/arm64 run performed;
no platform-specific SIMD or numeric approximation introduced.

The existing native replay wrapper also passed62tracer cases,1cache-root case,
and66Python replay/shared-inference tests in29.89s:

```bash
bash /home/hendrik/villa2/vesuvius/scripts/run_fiber3d_native_replay_tests /tmp/vc-live-native-cmake /home/hendrik/.venv_las/bin/python
```

The usual managed profile command remains:

```bash
/home/hendrik/.venv_las/bin/python /home/hendrik/villa2/vesuvius/scripts/run_classic_fiber_evaluation.py profile
```

Post-change performance measurement pending manager availability: service absent,
saved inventory224cores conflicts with112physical-core validation. All saved jobs
are terminal. No journal rewrite or unmanaged GPU fallback performed without
approval; no post-change throughput claim yet.
