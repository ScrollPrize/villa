# Native sampling and shared caches

CT and Lasagna channel readers share the process-wide decoded chunk cache.
There is no separate normal-sampler RAM budget. Sources are keyed by identity;
concurrent readers coalesce chunk requests and retain pinned data while sampling.
Eviction removes cache ownership even if a caller still holds the decoded data;
that caller's shared reference remains valid. The cache budget therefore bounds
cache-owned data, not all application memory (active batches and region pins can
hold additional data). Releasing a caller's last reference frees evicted data
without requiring another cache operation.
Separate processes still have separate decoded caches.

## Python

Configure physical worker capacities **before** opening volumes:

```python
import numpy as np
from vc.volume import initialize_chunk_cache, Volume, NormalSampler

initialize_chunk_cache(
    bytes=512 * 1024**2,
    fetch_workers=4,
    probe_workers=4,
    decode_workers=2,
)
volume = Volume.open("/path/to/volume.zarr")
block = volume.read_zyx((0, 0, 0), (128, 128, 128), level=0)

normal_sampler = NormalSampler("/path/to/dataset.lasagna.json")
points = np.array([[128, 256, 512]], dtype=np.float32)
normals, valid = normal_sampler.sample(points, threads=1)
```

Normal points are base-volume **XYZ**, and returned normals are unit XYZ axes
(sign-ambiguous), with a boolean validity array. Interpolation uses the native
axis-tensor decoder, not interpolation of encoded nx/ny bytes. Zero grad_mag
marks invalid samples. The opt-in manifest setting `mask_invalid_normal_corners`
also excludes invalid corners from interpolation; the default remains unchanged.
The binding owns its dataset and releases the GIL for native sampling.

`read_zyx` preserves the volume's native dtype and declared fill value; optional
`with_mask=True` / `with_stats=True` return auxiliary data.
`sample_coords` instead follows rendering semantics: float32 `[H,W,3]` XYZ
coordinates and a boolean `[H,W]` mask produce uint8 images, validity and stats.
Coordinates use the opened volume's level-zero frame.

`sample_coords(..., sampler_threads=1)` executes the sampling stage serially.
It does **not** disable background source-read, probe or decode workers.
`set_chunk_cache_io_threads` adjusts remote-download admission;
`set_chunk_cache_decode_threads` adjusts decode admission on the existing pools.
Local source reads use the independent pool introduced upstream (16 workers by
default, configured through C++ `ChunkCacheService::Options.localReadWorkers`),
not the remote-download limit.
`set_chunk_cache_budget` changes decoded-cache capacity.
Repeated initialization must preserve physical pool capacities.
Legacy C++ `LasagnaNormalSamplerOptions.maxCachedBytes` remains accepted, but
sets the shared process budget rather than creating a second channel cache.

Native requested-level dependency discovery and gathers reuse the last resolved
chunk and a small local lookup cache. `chunk_locality_fast_path=False` selects
the scalar reference for exact comparisons. No fast-math flags are required.
The optional AVX2/FMA normal interpolation path is compile-time guarded and has
a scalar fallback with the same corner order and fused operations.

`PinnedChunkRegion` supports readers that need a stable resolved region.
Opaque decoded payloads and float32 typed channels are available to native
consumers; they do not add these types to the CT rendering API.

## Cache location and authentication

`global_cache_root()` resolves without creating directories:

1. Explicit `{"version": 1, "root": "/absolute/path"}` in
   `$VC3D_CONFIG_DIR/cache.json` (default `$HOME/.VC3D/cache.json`).
2. Existing `/volpkgs`, then `/ephemeral`, using their `remote_cache` child.
3. Caller suggestion, then `$HOME/.VC3D/remote_cache`.

Malformed explicit configuration fails rather than silently falling back.
Public HTTP reads do not eagerly invoke AWS credential discovery. Each deferred
credential resolver caches either its result or its exception; a failed provider
is not rerun for every subsequent request. See
[remote_file_cache.md](remote_file_cache.md) for authentication behavior.

## Validation

Configure a Release native build with `VC_TESTING=ON` and `VC_BUILD_PYTHON=ON`
using the project's documented prerequisites. From the repository root:

```bash
cmake --build volume-cartographer/build/pr-native --target \
  vc_volume vc_fiber_trace test_chunk_cache test_chunked_plane_sampler \
  test_chunked_plane_sampler_fallback test_lasagna_normal_sampler \
  test_http_fetch_errors test_global_cache_root test_remote_file_cache \
  test_lasagna_manifest test_volume_extras test_persistent_zarr_cache_budget -j 8
ctest --test-dir volume-cartographer/build/pr-native --output-on-failure \
  -R '^test_(chunk_cache|chunked_plane_sampler|chunked_plane_sampler_fallback|lasagna_normal_sampler|http_fetch_errors|global_cache_root|remote_file_cache|lasagna_manifest|volume_extras|persistent_zarr_cache_budget)$'
python -m pytest -q \
  volume-cartographer/python/tests/test_native_sampling.py \
  volume-cartographer/python/tests/test_fiber_trace.py \
  --vc-build-python=volume-cartographer/build/pr-native/python
```

The explicit build selector prevents an installed editable `vc` package from
silently redirecting tests to an older binary.

For an optional real-volume comparison, add `-s`, `--vc-real-volume=PATH`,
`--vc-origin-xyz=X,Y,Z` and `--vc-level=N` to the native-sampling pytest command.
The test warms both paths, alternates their order across ten iterations each,
asserts exact image/mask equality and zero timed chunk decodes, and emits JSON
with mean, p50/p95, native-stage times and throughput.

Measured on Paris4 `20260411134726-2.400um-0.2m-78keV-masked.zarr`,
opening its physical level `1` array directly as level zero, at XYZ
`4096,11008,5120`, with 128³ quarter-voxel-offset coordinates:

| Path | Mean ms | p50 ms | p95 ms | 128³ samples/s | Million voxels/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scalar reference | 575.12 | 573.30 | 582.13 | 1.74 | 3.65 |
| Locality optimization | 99.16 | 98.65 | 102.88 | 10.09 | 21.15 |

Ubuntu amd64, GCC 13 Release, one sampling worker, one admitted decode/read
worker, warm 256 MiB decoded cache, ten iterations/path. All 2,097,152 output
voxels and masks matched exactly. This is a **5.8× resident-sampling speedup
against the same build's scalar reference**, not end-to-end training throughput
or an upstream-release benchmark. Cold I/O and multi-process scaling are not
measured here. macOS/arm64 validation remains outstanding.
