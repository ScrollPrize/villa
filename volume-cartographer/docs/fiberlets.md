# Fiberlet preprocessing and storage

`vc_fiberlets` builds local anchors and short directed routes from a fiber
prediction manifest and a Lasagna normal manifest. It is CPU-only. Graph replay,
tracing benchmarks, experiment configurations, and datasets are not part of this
component.

## Build and run

With the normal volume-cartographer dependencies already installed:

```sh
cmake -S volume-cartographer -B build -DVC_BUILD_FIBERLETS=ON
cmake --build build --target vc_fiberlets
build/bin/vc_fiberlets --help
```

The Python wheel also installs a `vc_fiberlets` launcher for its bundled binary.
`VC_BUILD_FIBERLETS` defaults to the value of `VC_BUILD_APPS`; it can be enabled
independently when building a headless wheel.

```sh
vc_fiberlets anchors fibers.lasagna.json anchors --crop X,Y,Z,W,H,D
vc_fiberlets paths fibers.lasagna.json anchors/anchors.json paths \
  --normal-manifest normals.lasagna.json
vc_fiberlets preprocess-volume fibers.lasagna.json final.zarr \
  --normal-manifest normals.lasagna.json --anchor-cache anchors.zarr --threads 8
```

Crop coordinates and CLI distances are base-volume voxels, except `--cell-size`,
which is in stored prediction voxels. Manifest scale determines the conversion;
the canonical grid requires matching presence/direction shapes and scales.
Normals are interpolated through the shared normal sampler, not as independent
encoded scalar channels. `--cache-gib` controls decoded cache capacity.

`anchors` writes inspection artifacts; `paths` reads their manifest identity and
geometry before calculating routes. `preprocess-volume` uses the chunk pipeline
directly and can be rerun with identical settings to reuse completed work.
It scans stored presence chunks to select work; absent and all-zero input chunks
do not require dense volume processing. This presence scan is repeated on resume.

## Storage contract

The final sparse Zarr group contains `anchors/`, `prefix/`, and `routes/` arrays.
These form a **logical chunk tuple**, not one physical file. Owner anchors are
copied into the final group, so reading final output does not require the
temporary anchor cache. Chunk keys use slash-separated coordinates.

Metadata is created before generation. It records source manifest hashes,
effective processing settings, grid shape, prediction-to-base scale, codec
profile, and generation contract version. `--source-context` supplies additional
source-volume/model identities. Fingerprints exclude runtime paths and toolchain
diagnostics. Incompatible metadata is rejected, not silently reused.

Chunk publication uses atomic replacement and file locks. Readers validate codec
headers, identity and tuple completeness. Missing sparse tuples represent empty
output; partially present tuples are errors. No separate expected-chunk index or
completion ledger is required. Do not interpret metadata alone as proof that an
interrupted whole-volume run finished.

## Library integration

`FiberletPipeline.hpp` provides shared metadata construction and canonical
on-demand settings. `FiberletOnDemandPreprocessor` exposes anchor and route caches;
cache misses generate dependency anchors before routes. `cachedOnly` instead
fails on missing materialized data. Call `shutdown()` to drain generation and
writes before releasing the preprocessor. CPU admission is acquired for
computation, not while waiting for dependency chunks.

`FiberletPreprocessSchedule` dispatches dependency-ready work and prioritizes
ready output in the current Z slab. A blocked route must not occupy the workers
needed to produce its dependencies. Storage and cache APIs are shared by
whole-volume and on-demand generation.

## Validation

```sh
cmake --build build --target test_fiber_anchors test_fiberlet_paths test_fiberlet_storage test_fiber_trace3d
ctest --test-dir build --output-on-failure -R 'test_fiber_(anchors|trace3d)|test_fiberlet_(paths|storage)'
python -m pytest -q volume-cartographer/python/test_fiberlets_cli.py
```

Tests cover anchor fits, DP objectives, route codecs, malformed data, atomic
publication, sparse tuples, cache reuse and dependency scheduling. The inline
scalar/prepared scoring comparison disables FMA contraction **in its test target
only** so bitwise operation-order assertions are meaningful; production compiler
settings and numerical tolerances are unchanged.

Known inherited limitation: at large absolute coordinates, the inspection
`anchors` → `paths` JSON round trip can fail the strict rotating-plane residual
check after float32 coordinate rounding. Both the source executable and extracted
executable reject the same Paris4 artifact. This extraction deliberately does not
relax the geometry validation. The direct chunk pipeline does not load that JSON
artifact. Linux Release was validated locally; other platform validation remains
for CI.
