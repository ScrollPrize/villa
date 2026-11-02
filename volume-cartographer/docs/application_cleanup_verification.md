# Application cleanup verification

Implemented 2026-11-01 from baseline `5afe15609`.

Changes are committed in focused groups starting at `cc78bb9f7`. No user data,
existing generated outputs, or old build-tree executables were deleted. A fresh
runtime staging prefix contains no retired tools. SLIM/flatboi remains built and
installed. The full build caught and restored `normalGridPaths()`, which belongs
to explicit open workflows rather than directory watching.

### Inventory evidence and retained boundaries

| Item | Final disposition and evidence |
| --- | --- |
| PostProcess | Reusable code is `vc_reader/display.hpp` and CPU/CUDA kernels. VC3D `DisplayPalette`/`Colormaps` retain palette selection and palette generation only. LUT endpoints, conversion, coverage, premultiplied blending and CUDA parity pass. |
| Sampling | Duplicate enum deleted. Viewer requests carry native order integers; settings/RPC boundary values remain. |
| ArcHermite | Retained unchanged: `LineOptimizer` uses `sampleLine` and `sampleField`; `hermite` and `arcDerivative` support those operations. Chord-arclength interpolation is distinct from the reader's surface patches. Analytic line/field tests pass. |
| AtomicFile | Publication, recovery and snapshot mechanics moved to reader `publication.hpp`; `OutputDirectory` retains application locks and abandoned-temporary cleanup. Snapshot count/throttle remain application policy. |
| Compositing | Reader `slab.hpp` implements opacity, masked bilinear sampling, lighting and front-to-back compositing on CPU/CUDA. VC3D retains camera projection, margins, settings and OpenCV output ownership. Python binds the native kernels. |
| GlobalCacheRoot | Retained config/environment/mount selection in `core/include/vc/core/util/GlobalCacheRoot.hpp`; native open options receive the selected root. Cache-policy tests pass. |
| InpaintSurface | Retained Ceres geometry optimization used by tifxyz2obj; it is not an image interpolation replacement. |
| MemMap | Renamed to `AnonymousPages`, with lazy allocation and platform branches preserved. |
| PlaneSurface | Retained editable orientation/frame and pointer coordinates. Viewer consumers construct native `grid::Plane`. |
| PointIndex | Deleted from VC; resident object-store `VertexIndex` owns immutable vertices and original row-major IDs. Bounds and candidate comparisons use the same precision; large-coordinate and brute-force fixtures pass. |
| SurfaceArea | Duplicate wrappers deleted. Callers invoke `surfaceGridArea`; `SurfaceArea.hpp` retains unit conversion and metadata policy only. |
| SurfacePatchIndex | Deleted from VC. Store `SurfaceIndex` uses shared bounds/triangle code. `SurfaceQuerySet` associates application identity and converts pointer coordinates; it contains no spatial algorithm or database writes. |
| SurfaceTexture | Deleted. Four production call sites use native grid sampling and byte conversion. The plan's initial count of three omitted a second replay-visualization call. |
| TexturedMesh | OBJ/MTL text retained. TIFF/PNG/JPEG publication uses reader writers. `ImageMat` only converts OpenCV buffers and color order. |
| Umbilicus | Retained rounded/clamped-Z centerline vector calculation; import/persistence stays in object-store adapters. |
| utils | Target and Json wrapper deleted; signatures use nlohmann JSON. Stable hash/thread pool live in core. Numeric-string, reference and config tests pass. |
| Surface/QuadSurface | Editable geometry, identity, metadata, channels and pointer bookkeeping retained. Native immutable snapshots feed workers; old sampling methods and custom affine interpolation removed. Native tifxyz writing preserves channel depth/color/resolution, metadata, sibling artifacts and snapshot history. |

Retained OpenCV operations are application edits or display construction:
`GrowPatch`'s weighted geometry/area resizing and channel transfer, `QuadSurface`'s
editable channel resizing/rotation, and replay-presence coordinate-raster construction.
These implement domain editing rules, including invalid-point weights, label-channel
selection and border behavior. They are not file decoding or parallel reader APIs.
Drawing, text/layout and BGR/RGB buffer conversion remain application concerns.
Tests intentionally keep OpenCV/Pillow/tifffile as independent output decoders.
Normal-grid image resizing now uses existing native transform kernels.

`Surface::coord`/`normal` retain the polymorphic application pointer/offset contract
for editable planes and grids. They translate application coordinates before querying
native geometry; rendering and cursor queries use native calls directly. Native
snapshot construction and coordinate conversion are ownership/bookkeeping boundaries,
not a second sampling interface. Channel/component changes and geometry edits invalidate
cached snapshots. Retained snapshots survive edits and destruction of their owner;
invalid-to-valid reconstruction (undo semantics) and store-edit tests pass.

Catalog parsing, resolution, browsing, listing, fetching and file caching already used
native reader APIs. The final forwarding cache helpers were removed. Normal-grid
batch orchestration retains UI progress/cancellation and application completion markers;
`EncodedFiles` owns fetch/cache behavior. Service-job RPCs and attachment selection remain.

### Intentional semantics

- Surface-index triangles use the opposite-corner diagonal through top-right and
  bottom-left; warped triangle hits differ from bilinear reader patch positions.
  Tests verify this and filters, tolerance, validity and original vertex IDs.
- In-memory explicit validity permits finite negative coordinates and z=0. File
  validity remains the format's rule. Nonfinite coordinates never become valid.
- Native point queries reject positions outside the grid, including the former
  application's small out-of-bounds tolerance. The edge itself belongs to the final
  patch. Analytic tests cover the edge, just-outside positions, holes and scales.
- Overwrite retains unrelated regular files and subdirectories. Forced publication
  failure restores the preceding generation. Existing cache marker artifacts are
  preserved; readers no longer need compatibility cleanup.
- Pathfinding uses float32 entered-pixel costs with eight neighbors, no diagonal
  multiplier, and deterministic cost/hop/predecessor tie rules. Annotation acceptance
  still requires the resulting route to stay on the skeleton.

### Baseline and verification

Logs are retained locally in `/tmp/vc-cleanup-baseline/`. All builds use Release and
`-j16`; existing VC and CUDA reader trees were reused. New CPU-only trees live in
`/tmp`, using the already-present delta3d source without downloads or provisioning.

Baseline commands:

```sh
env AGENTS_AGENT_MODE=1 QT_QPA_PLATFORM=offscreen \
  VC3D_CONFIG_DIR=/tmp/vc-cleanup-baseline/config \
  XDG_CACHE_HOME=/tmp/vc-cleanup-baseline/cache \
  ctest --test-dir build -j16 --output-on-failure
.venv/bin/python -m pytest python/tests --vc-build-python=build/python -q
```

The sandboxed baseline had 12 socket-related CTest failures out of 175 and two CUDA
skips. Python had 99 passes, 15 socket-related failures and four skips. With local
socket/GPU access, Python passed 114 with four skips. The unrestricted baseline
CTest run passed 174/175; its tool-store test overlapped CMake regeneration and lost
its test environment. That contaminated run is not presented as a clean baseline.
The earlier 176/176 result above belongs to the historical review.

Final results:

| Check | Result |
| --- | --- |
| `CCACHE_DIR=/tmp/vc-cleanup-ccache cmake --build build -j16` | Full GUI, surviving CLIs, tests and Python targets build successfully. |
| Offscreen CTest command above, with local socket/GPU access | **177/177 passed**, 10.93 s (`final-ctest2.log`). |
| VC Python command above, with local socket access | **114 passed, 4 skipped** (`final-python.log`). |
| `ctest --test-dir ../vc_reader/build-native-cuda -j16 --output-on-failure` | **25/25 passed**, including CUDA (`reader-cuda-tests.log`). |
| `ctest --test-dir /tmp/vc-cleanup-reader-cpu -j16 --output-on-failure` | **23/23 passed** (`cpu-tests.log`). The CPU-only build exposed and fixed unguarded CUDA dispatch references. |
| `ctest --test-dir /tmp/vc-cleanup-store -j16 --output-on-failure` | **17/17 passed**, adapters/linking enabled (`store-tests.log`). VC's embedded build also verifies resident indices with linking disabled. |
| Store Python tests | **67 passed** (`store-python.log`); nanobind reports shutdown reference-leak warnings in existing geometry bindings. |
| Affected reader Python suites | **163 passed, 88 subtests passed** (`reader-python-final.log`); two multiprocessing-fork deprecation warnings. |
| Additional display CPU/CUDA tests | **6 passed**, including device palette application and alpha blending (`reader-display-final.log`). |
| `python3 scripts/test_install_build_deps.py` | All three test groups pass, including mocked Debian/Ubuntu profiles and install guards. |
| Shell and CI syntax | `bash -n scripts/install_build_deps.sh`; `bash -n build_from_src_debian.sh`; both edited workflow files parse as YAML. No dependencies installed. |
| `cmake --install build --prefix /tmp/vc-cleanup-install --component vc_runtime` | Successful staged install; VC3D, tifxyz2obj and flatboi present, retired executables absent. |

Reader Python command (existing environment; no installation):

```sh
VC_READER_BUILD_DIR=/home/sean/Documents/villa/vc_reader/build-native-cuda \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest \
  ../vc_reader/tests/test_display.py ../vc_reader/tests/test_tifxyz.py \
  ../vc_reader/tests/test_images.py ../vc_reader/tests/test_surface.py \
  ../vc_reader/tests/test_sampling.py ../vc_reader/tests/test_geometry.py \
  ../vc_reader/tests/test_source_readers.py -q
VC_OBJECT_STORE_BUILD_PYTHON_PATH=/tmp/vc-cleanup-store/python \
  /home/sean/Documents/villa/vesuvius/.venv/bin/python -m pytest ../vc_object_store/tests/python -q
```

CPU reader configuration:

```sh
cmake -S ../vc_reader -B /tmp/vc-cleanup-reader-cpu -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVC_READER_ENABLE_CUDA=OFF \
  -DVC_READER_BUILD_PYTHON=OFF -DVC_READER_BUILD_NATIVE_TESTS=ON \
  -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D=/home/sean/Documents/villa/vc_reader/build-native-cuda/_deps/vc_delta3d-src
cmake --build /tmp/vc-cleanup-reader-cpu -j16
cmake -S ../vc_object_store -B /tmp/vc-cleanup-store -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DVC_READER_ENABLE_CUDA=OFF \
  -DVC_OBJECT_STORE_BUILD_TESTS=ON -DVC_OBJECT_STORE_BUILD_PYTHON=ON \
  -DPython_EXECUTABLE=/home/sean/Documents/villa/vesuvius/.venv/bin/python \
  -DFETCHCONTENT_SOURCE_DIR_VC_DELTA3D=/home/sean/Documents/villa/vc_reader/build-native-cuda/_deps/vc_delta3d-src
cmake --build /tmp/vc-cleanup-store -j16
```

### Timing and memory

Host: Intel Core Ultra 7 270K Plus, 24 logical CPUs; NVIDIA RTX 5090. Release,
float32 path costs, `mt19937(42)`, integer costs 1..20, opposite-corner endpoints.
One warm-up plus five measured runs; timings include host/device transfer and path
reconstruction. Median milliseconds:

| Grid | CPU | Initial CUDA | Batched CUDA |
| --- | ---: | ---: | ---: |
| 32² | 0.042 | 0.365 | 0.143 |
| 128² | 0.920 | 1.482 | 0.592 |
| 512² | 17.437 | 13.095 | 7.908 |

The initial GPU implementation synchronized after every wavefront round. An active
frontier experiment did not improve small-grid timings and was discarded. Batching
eight verified rounds per synchronization improves all measured sizes without changing
arithmetic or tie rules. CUDA launch/transfer overhead still dominates at 32²; the
annotation caller uses CPU. No universal GPU speedup is claimed.

The spatial fixture has 65,536 warped vertices and 10,000 seeded queries. A repeat run
measured 47.06 ms construction and 17.86 ms total queries. Input grid plus resident index
increased process peak RSS by 7,568 KiB. This is process-level measurement, not an exact
allocator accounting. Native 512² surface resampling (one warm-up, ten measurements)
measured min/median/max 1.769/1.792/2.673 ms for an unowned view and
1.761/1.778/1.985 ms for an owned snapshot. An earlier run overlapped compilation and
had scheduling outliers; the repeat is recorded in `spatial-sampling-benchmark2.log`.
These are current fixture measurements, not a reconstructed pre-cleanup spatial baseline.

Reproduce with `scripts/benchmark_cleanup.cpp`:

```sh
c++ -O3 -std=c++20 -DVC_READER_CUDA=1 \
  -I../vc_reader/include -I../vc_reader/build-native-cuda/include \
  -I../vc_object_store/include -I/usr/include/opencv4 \
  scripts/benchmark_cleanup.cpp build/object_store/src/libvc_resident_geometry.a \
  -L../vc_reader/build-native-cuda -lvc_reader_native -lvc_reader_reader_cuda \
  -L/usr/local/cuda/lib64 -lcudart -lopencv_core \
  -Wl,-rpath,/home/sean/Documents/villa/vc_reader/build-native-cuda \
  -Wl,-rpath,/usr/local/cuda/lib64 -o /tmp/vc-cleanup-benchmark
/tmp/vc-cleanup-benchmark
/tmp/vc-cleanup-benchmark spatial
```

### Verification limits / input needed

- MCP collection fails because `mcp` is absent from both available Python environments.
  Command: `PYTHONPATH=tools/vc3d-mcp .venv/bin/python -m pytest tools/vc3d-mcp/tests -q`.
  Dependency installation requires approval under the monorepo `AGENTS.md`; none was
  performed. Provide an environment with the package installed, or authorize installing
  the MCP project's dependencies, to run this remaining suite. GUI/RPC contract tests pass.
- macOS/Windows CI was not available on this Linux host. Platform guards are retained,
  and portable staging/CPU dispatch were checked in source; those platforms still need
  their CI executions. Linux CPU and CUDA builds/tests passed.
- No implementation ambiguity remains. The exceptions above concern verification access,
  not deferred feature work.
