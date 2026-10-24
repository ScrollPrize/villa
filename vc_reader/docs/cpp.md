# C++ API

The native APIs expose the same reader, sampling, geometry, and transform
implementations used by Python. They do not depend on Python, nanobind, Qt,
Torch, or volume-cartographer. CPU and CUDA builds use the same public headers.

## Build and consume

From `villa/vc_reader`:

```sh
cmake -S . -B build-native -GNinja -DCMAKE_BUILD_TYPE=Release \
  -DVC_READER_BUILD_PYTHON=OFF -DVC_READER_ENABLE_CUDA=OFF \
  -DVC_READER_BUILD_NATIVE_TESTS=ON -DVC_READER_BUILD_EXAMPLES=ON
ninja -C build-native
ctest --test-dir build-native --output-on-failure
cmake --install build-native --prefix /path/to/prefix
```

For CUDA, set `VC_READER_ENABLE_CUDA=ON` and choose
`CMAKE_CUDA_ARCHITECTURES` for the deployment GPU. CUDA discovery defaults to
auto-detection when the option is omitted. The toolkit supplies CUDA Runtime
and cuFFT. No Python development files are needed when Python is disabled.

Reader dependencies are libcurl, c-blosc, zstd, SQLite, nlohmann-json, and
VC-Delta3D. CMake uses an installed VC-Delta3D package or fetches its pinned
revision. A transforms-only build uses `VC_READER_BUILD_READER=OFF` and needs
none of these reader dependencies.

An installed consumer needs only C++20 and the exported targets:

```cmake
cmake_minimum_required(VERSION 3.22)
project(example LANGUAGES CXX)
find_package(vc_reader CONFIG REQUIRED)
add_executable(example main.cpp)
target_link_libraries(example PRIVATE vc_reader::reader vc_reader::transforms)
```

Set `CMAKE_PREFIX_PATH=/path/to/prefix` when configuring that consumer. The
same targets are available with `add_subdirectory`. `vc_reader::reader` and
`vc_reader::transforms` propagate their dependencies, include paths, and CUDA
compile definitions. Link the target for every API you use. Do not define
`VC_READER_CUDA` manually. CUDA-enabled builds additionally export
`vc_reader::reader_cuda` and `vc_reader::transforms_cuda`; the primary targets
already link them. The Python extensions link those same CUDA libraries.

## API coverage

| Header | API | CPU and CUDA coverage |
| --- | --- | --- |
| `vc_reader/reader.hpp` | `vcr::CtReader`: open a CT array or pyramid level, read regions, sample sheet normals; Zarr metadata normalization | Convenience methods use CPU; `array()` exposes the same array to general and CUDA sampling |
| `vc_reader/array.hpp` | `vcr::Array`, `ArraySpec`, `MemoryCache`, `DiskCache`, HTTP/auth options, region reads, point reads, nearest/linear sampling, affine reads | Full existing CPU reader; connect the same array to `reader_cuda::Reader` for CUDA |
| `vc_reader/sampling.hpp` | `aligned::Plan` and `PointPlan`, `surface::Plan`, `geometry::Curves` and `Voxels`, sheet-normal kernels | Existing CPU algorithms and CUDA launch APIs, including surfaces, curves and voxel membership |
| `vc_reader/transforms.hpp` | All `transforms::Op` alternatives, parameter types, `Pipeline`, `Sample`, `Plan`, `save_plan`, `load_plan` | Full transform catalog; `cpu`, `cuda` and `auto` pipeline backends |
| `vc_reader/cuda.hpp` | Device storage, borrowed storage, stream scope, completion, resident transform input/output sessions | CUDA only |
| `vc_reader/cuda_reader.hpp` | `reader_cuda::Reader`, region/affine grids, point plans, resident sampling, cache statistics; CUDA sampling declarations | CUDA only |

The `vc_reader/native/` include tree contains the underlying typed APIs,
including individual transform operators and CPU/CUDA kernels. These are
published from the implementation headers, so there is only one implementation
to maintain. Python binding headers are not installed. This is a source-level
C++ interface: rebuild consumers when updating the package.

The transform catalog and parameter meanings are documented in the main
[README](../README.md). Native operators are the structs under
[`transforms/ops`](../src/cpp/transforms/ops), and `transforms::Op` in
[`pipeline.hpp`](../src/cpp/transforms/pipeline.hpp) enumerates all of them.
CUDA dispatch checks every operator at compile/link time.

The general array API accepts an `ArraySpec`. This exposes the existing cache,
authentication, storage-layer and sampling controls directly. Catalog lookup,
Python DLPack wrappers and Python object orchestration are not C++ types.
`CtReader` is a convenient opener for three-dimensional uint8 local/anonymous
HTTP/S3 arrays; other dtypes and authenticated sources use `ArraySpec`.
The set of supported codecs is unchanged.

## Transform pipelines and replay

```cpp
#include <vc_reader/transforms.hpp>

using namespace vcr::transforms;
Target image;
Sample input;
input["image"] = {Array({1, 16, 32, 32}, 'f', 4), image}; // CZYX float32
// Fill input.at("image").array.data() here.
BrightnessMultiply brightness;
brightness.factor = Uniform{0.8, 1.2};
Pipeline pipeline({brightness, Flip{"x", 0.5, 1.0}}, {{"image", image}},
                  true, "auto");
auto result = pipeline.run(input, 1234, true, false); // seed, record, profile
const auto saved = save_plan(result.plan);
auto replay = pipeline.replay(input, load_plan(pipeline.program(), saved));
```

`Array` is an owning, dtype-tagged channel-first buffer. It can also borrow
caller storage through its constructor. Copies own their data; writable
borrowed inputs require the caller to keep the memory alive through completion.
Roles include images, labels, points, vectors, validity, and directional fields.
Parameters and replay use the same seeds and serialization as Python. Treat
saved replay plans as trusted artifacts.

Host-input CUDA pipelines handle upload/download. Resident pipelines accept
`cuda::DeviceInput` maps through `cuda::begin_device`, then pass the returned
session to `Pipeline::run` or `replay`. Retain that session while accessing
`cuda::device_outputs` and `device_pyramids`. Caller-owned device buffers must
remain alive until the work completes.

## CT reads and normals

```cpp
#include <vc_reader/reader.hpp>

vcr::CtReader volume("/data/scroll.zarr", 0);
std::array<double, 3> point{3000.25, 4000.5, 5000.75}; // XYZ in selected level
const auto normal = volume.sheetNormals(std::span(&point, 1)).front();
// normal = XYZ unit vector, then relative eigengap. Invalid support is zero.
```

The estimator needs complete 109³ context. Image arithmetic and normal outputs
use float32; global coordinates retain float64 until their integer lattice
origin has been removed. The recipe and numerical behavior match the Python
sheet-normal API. Normals are unsigned.

[`examples/ct_normals.cpp`](../examples/ct_normals.cpp) is a complete native
client: it reads the same real CT through CPU and CUDA readers, runs both normal
kernels, and compares their output. Build it with `VC_READER_BUILD_EXAMPLES=ON`:

```sh
build-native/vc_reader_ct_normals /data/scroll.zarr 0 3000.25 4000.5 5000.75
```

`CtReader::array()` returns a borrowed `Array&`. Both a `reader_cuda::Reader`
constructed from it and any work in progress must finish before the `CtReader`
is destroyed. CUDA reader results own their device storage independently.

Low-level sampling APIs take explicit pointer/shape/dtype descriptors; callers
must supply buffers of the declared sizes and the correct host/device placement.
Array/grid coordinates are ZYX; sheet normals and geometry coordinates are XYZ.
CUDA operations use the calling thread's `device::StreamScope`. Use
`device::complete()` before reading asynchronous outputs on the host or passing
them to work on another stream without an event dependency. CUDA objects must
be created after a process fork.
