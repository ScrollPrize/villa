# Volume Class API

The `Volume` class is the main API for opening, creating, reading, and writing
3D zarr volumes. It supports local OME-Zarr-like pyramids, remote HTTP/S3-backed
zarr stores, cached sampling, typed region reads/writes, raw chunk I/O, and
metadata/root attribute updates.

**Header:** `core/include/vc/core/types/Volume.hpp`
**Implementation:** `core/src/Volume.cpp`

## Overview

A `Volume` represents a 3D image dataset with:

- Volume metadata such as name, UUID, dimensions, voxel size, dtype, and format.
- Local or remote zarr pyramid access.
- Cached sampling through `vc::render::ChunkCache`.
- Typed region reads/writes in `uint8_t` and `uint16_t`.
- Raw local zarr chunk reads/writes for tools that need chunk-level control.
- File-based persistence through `meta.json` and root `.zattrs` for local volumes.

Volumes are typically managed through a `VolumePkg` container, but can also be
opened or created directly.

## Coordinate Ordering

The storage order is **ZYX**:

- `shape[0]` = Z = slices
- `shape[1]` = Y = height
- `shape[2]` = X = width

The `meta.json` dimension fields are:

- `slices` = Z
- `height` = Y
- `width` = X

Use `shape()` and `shape(level)` for storage order `{z, y, x}`. Use
`shapeXyz()` when UI or coordinate-order code needs `{x, y, z}`.

`Array3D<T>` is also indexed and shaped as `{z, y, x}`. The `readXYZ()` and
`writeXYZ()` helpers accept offsets in `{x, y, z}` order, but the `Array3D`
payload remains `{z, y, x}`.

## Factory Methods

```cpp
// Open an existing local volume.
static std::shared_ptr<Volume> New(std::filesystem::path path);

// Open an existing local volume, or create one when no zarr array exists.
static std::shared_ptr<Volume> New(std::filesystem::path path,
                                   const ZarrCreateOptions& options);

// Open a remote HTTP or s3:// zarr volume. s3:// URLs are resolved to HTTPS.
static std::shared_ptr<Volume> NewFromUrl(
    const std::string& url,
    const std::filesystem::path& cacheRoot = {},
    const vc::HttpAuth& auth = {});
```

`New(path, options)` creates a local zarr pyramid when `path` has no existing
zarr array, or when `options.overwriteExisting` is true. Existing volumes are
opened as-is by default.

## Creation Options

```cpp
struct PyramidPolicy {
    enum class Reduction {
        Mean,
        Max,
        BinaryOr,
    };

    std::array<double, 3> downsampleZYX{2.0, 2.0, 2.0};
    Reduction reduction = Reduction::Mean;
};

struct ZarrCreateOptions {
    std::array<size_t, 3> shapeZYX{};
    std::array<size_t, 3> chunkShapeZYX{64, 64, 64};
    vc::render::ChunkDtype dtype = vc::render::ChunkDtype::UInt8;
    size_t numLevels = 1;
    PyramidPolicy pyramid;
    double fillValue = 0.0;
    double voxelSize = 1.0;
    std::string voxelUnit;
    std::string uuid;
    std::string name;
    std::string compressor = "blosc";
    int compressionLevel = 3;
    bool overwriteExisting = false;
};
```

Creation writes `meta.json`, `.zattrs`, `.zgroup`, and one zarr array per scale
level named `0`, `1`, `2`, and so on. Only `uint8_t` and `uint16_t` volume data
are currently supported by this API.

## API Reference

### Metadata and Identity

| Method | Returns | Description |
|--------|---------|-------------|
| `id()` | `std::string` | Volume UUID from metadata |
| `name()` | `std::string` | Display name from metadata |
| `metadata()` | `const utils::Json&` | Current volume metadata |
| `writeMetadata(const utils::Json&)` | `void` | Replace local `meta.json` and update cached dimensions |
| `updateMetadata(const utils::Json&)` | `void` | Merge fields into existing metadata and persist |
| `rootAttributes()` | `utils::Json` | Read local root `.zattrs`, or `{}` if absent |
| `writeRootAttributes(const utils::Json&)` | `void` | Replace local root `.zattrs` |
| `updateRootAttributes(const utils::Json&)` | `void` | Merge fields into local root `.zattrs` |
| `path()` | `std::filesystem::path` | Local volume path; empty for remote volumes |
| `isRemote()` | `bool` | Whether the volume is backed by a remote zarr store |
| `remoteUrl()` | `const std::string&` | Normalized remote URL for remote volumes |
| `remoteAuth()` | `const vc::HttpAuth&` | Auth used for remote chunk reads |

`writeMetadata()`, `updateMetadata()`, `writeRootAttributes()`, and
`updateRootAttributes()` are local-only operations and throw for remote volumes.

### Dimensions, Dtype, and Scales

| Method | Returns | Description |
|--------|---------|-------------|
| `sliceWidth()` | `int` | X dimension at base scale |
| `sliceHeight()` | `int` | Y dimension at base scale |
| `numSlices()` | `int` | Z dimension at base scale |
| `shape()` | `std::array<int, 3>` | Base shape as `{z, y, x}` |
| `shape(int level)` | `std::array<int, 3>` | Present scale shape as `{z, y, x}` |
| `shapeXyz()` | `std::array<int, 3>` | Base shape as `{x, y, z}` |
| `voxelSize()` | `double` | Physical voxel size from metadata |
| `dtype()` | `vc::render::ChunkDtype` | Volume dtype |
| `numScales()` | `size_t` | Number of scale slots known to the volume |
| `baseScaleLevel()` | `int` | Always `0` |
| `hasScaleLevel(int level)` | `bool` | Whether a physical zarr array exists for the level |
| `presentScaleLevels()` | `std::vector<int>` | Physical scale levels |
| `firstPresentScaleLevel()` | `int` | Finest available physical scale |
| `finestPresentScaleLevelAtOrBelow(int level)` | `int` | Finer source level for virtual downsampling |
| `pyramidReduction()` | `PyramidPolicy::Reduction` | Reduction method used for pyramid writes |

`numScales()` can include missing scale slots. Check `hasScaleLevel()` or
`presentScaleLevels()` before reading or writing a specific level.

### Cache and Sampling

| Method | Returns | Description |
|--------|---------|-------------|
| `chunkedCache()` | `vc::render::IChunkedArray*` | Shared chunked cache for local or remote reads |
| `createChunkCache(Options)` | `std::shared_ptr<ChunkCache>` | Create an independent cache |
| `setCacheBudget(size_t hotBytes)` | `void` | Set decoded cache capacity and reset the shared cache |
| `setIOThreads(int count)` | `void` | Set background chunk read concurrency and reset the shared cache |
| `invalidateCache()` | `void` | Drop decoded/read cache state |
| `sample(Mat_<uint8_t>&, coords, params)` | `void` | Sample into an 8-bit image |
| `sample(Mat_<uint16_t>&, coords, params)` | `void` | Sample into a 16-bit image |

`sample()` reads through the chunked cache and supports `vc::Sampling::Nearest`,
`Trilinear`, `Tricubic`, and `Lanczos` through `vc::SampleParams`.

### Typed Region Reads

| Method | Returns | Description |
|--------|---------|-------------|
| `readZYX(Array3D<uint8_t>&, offsetZYX, level, policy)` | `bool` | Read an 8-bit region using `{z, y, x}` offset |
| `readZYX(Array3D<uint16_t>&, offsetZYX, level, policy)` | `bool` | Read a 16-bit region using `{z, y, x}` offset |
| `readXYZ(Array3D<uint8_t>&, offsetXYZ, level, policy)` | `bool` | Read an 8-bit region using `{x, y, z}` offset |
| `readXYZ(Array3D<uint16_t>&, offsetXYZ, level, policy)` | `bool` | Read a 16-bit region using `{x, y, z}` offset |
| `static readZYX(..., IChunkedArray&, level)` | `void` | Read from an explicit chunked array/cache |

Region reads are available for local and remote volumes. Out-of-bounds portions
are filled with the zarr fill value. The output array shape determines the read
size.

When a requested level is missing, `MissingScaleLevelPolicy` controls behavior:

| Policy | Behavior |
|--------|----------|
| `Error` | Throw for the missing scale level |
| `AllFill` | Fill the output with the zarr fill value and return `true` |
| `Empty` | Leave the output unchanged and return `false` |
| `VirtualDownsample` | Read the finest available lower level and downsample into the output |

### Typed Region Writes

| Method | Returns | Description |
|--------|---------|-------------|
| `writeZYX(const Array3D<uint8_t>&, offsetZYX, level)` | `void` | Write an 8-bit local region using `{z, y, x}` offset |
| `writeZYX(const Array3D<uint16_t>&, offsetZYX, level)` | `void` | Write a 16-bit local region using `{z, y, x}` offset |
| `writeXYZ(const Array3D<uint8_t>&, offsetXYZ, level)` | `void` | Write an 8-bit local region using `{x, y, z}` offset |
| `writeXYZ(const Array3D<uint16_t>&, offsetXYZ, level)` | `void` | Write a 16-bit local region using `{x, y, z}` offset |

Writes are local-only. They update the target level, propagate the affected
region into coarser present pyramid levels using the configured reduction, and
invalidate the shared cache.

### Raw Chunk I/O

| Method | Returns | Description |
|--------|---------|-------------|
| `readChunk(level, chunkZYX)` | `std::optional<std::vector<std::byte>>` | Read a local raw chunk, or `std::nullopt` when absent |
| `readChunkOrFill(level, chunkZYX)` | `std::vector<std::byte>` | Read a local raw chunk, or synthesize fill bytes |
| `chunkExists(level, chunkZYX)` | `bool` | Check local chunk presence |
| `writeChunk(level, chunkZYX, data)` | `void` | Write a local raw chunk |
| `writeChunk(level, chunkZYX, data, options)` | `void` | Write a local raw chunk with options |
| `removeChunk(level, chunkZYX)` | `bool` | Remove or mark a local chunk empty |

Chunk coordinates are `{z, y, x}` chunk indices. `writeChunk()` expects the raw,
uncompressed chunk payload to have exactly `chunk_elems * dtype_size` bytes.
When `ChunkWriteOptions::writeEmptyChunks` is false, chunks containing only the
zarr fill value are removed instead of stored.

### Utilities

| Method | Returns | Description |
|--------|---------|-------------|
| `static checkDir(const std::filesystem::path&)` | `bool` | Accepts directories with `meta.json`, `metadata.json`, or any local zarr array |
| `skipShapeCheck` | `thread_local bool` | Debug/import escape hatch to skip zarr shape validation |

## Metadata Files

The preferred local metadata file is `meta.json`:

```json
{
  "type": "vol",
  "uuid": "unique-identifier",
  "name": "Volume Name",
  "width": 1024,
  "height": 1024,
  "slices": 256,
  "voxelsize": 5.0,
  "min": 0.0,
  "max": 65535.0,
  "format": "zarr"
}
```

`metadata.json` is also accepted when it contains a `scan` object. If no
metadata file exists but a zarr array is present, metadata is synthesized from
the zarr path and physical level shapes.

## Usage Examples

### Opening a Local Volume

```cpp
#include "vc/core/types/Volume.hpp"

auto volume = Volume::New("/path/to/volume");

auto shapeXYZ = volume->shapeXyz();

std::cout << volume->name() << ": "
          << shapeXYZ[0] << "x" << shapeXYZ[1] << "x" << shapeXYZ[2]
          << std::endl;
```

### Creating a Local Volume

```cpp
#include "vc/core/types/Volume.hpp"

Volume::ZarrCreateOptions opts;
opts.shapeZYX = {256, 1024, 1024};
opts.chunkShapeZYX = {64, 64, 64};
opts.dtype = vc::render::ChunkDtype::UInt16;
opts.numLevels = 4;
opts.voxelSize = 5.0;
opts.voxelUnit = "nm";
opts.name = "scan";

auto volume = Volume::New("/path/to/output.zarr", opts);
```

### Opening a Remote Volume

```cpp
#include "vc/core/types/Volume.hpp"

auto volume = Volume::NewFromUrl("s3://bucket/path/to/volume.zarr");

if (volume->isRemote()) {
    std::cout << volume->remoteUrl() << std::endl;
}
```

Remote volumes are read-only through `Volume`: metadata/root attribute writes,
region writes, and raw chunk writes throw.

### Reading a Region

```cpp
#include "vc/core/types/Array3D.hpp"
#include "vc/core/types/Volume.hpp"

auto volume = Volume::New("/path/to/volume");

Array3D<uint16_t> block({64, 128, 128});  // {z, y, x}
volume->readZYX(block, {100, 200, 300});  // offset {z, y, x}, level 0

uint16_t value = block(10, 20, 30);
```

### Reading with XYZ Offsets

```cpp
Array3D<uint8_t> block({32, 64, 64});     // data is still {z, y, x}
volume->readXYZ(block, {300, 200, 100});  // offset is {x, y, z}
```

### Handling Missing Scale Levels

```cpp
Array3D<uint8_t> tile({64, 64, 64});
bool ok = volume->readZYX(
    tile,
    {0, 0, 0},
    3,
    Volume::MissingScaleLevelPolicy::VirtualDownsample);

if (!ok) {
    // Only possible with MissingScaleLevelPolicy::Empty.
}
```

### Writing a Region

```cpp
Array3D<uint8_t> mask({32, 128, 128}, uint8_t{1});

volume->writeZYX(mask, {50, 100, 100});
```

The write updates level 0 and any coarser present levels affected by the region.

### Raw Chunk Read/Write

```cpp
const int level = 0;
const std::array<size_t, 3> chunkZYX{0, 0, 0};

auto bytes = volume->readChunkOrFill(level, chunkZYX);

Volume::ChunkWriteOptions options;
options.writeEmptyChunks = false;
volume->writeChunk(level, chunkZYX, bytes, options);
```

### Sampling a Slice

```cpp
#include "vc/core/types/SampleParams.hpp"
#include "vc/core/types/Volume.hpp"

cv::Mat_<cv::Vec3f> coords(1000, 1000);
for (int y = 0; y < coords.rows; ++y) {
    for (int x = 0; x < coords.cols; ++x) {
        coords(y, x) = cv::Vec3f(500.0f, y + 100.0f, x + 100.0f);
    }
}

vc::SampleParams params;
params.level = 0;
params.method = vc::Sampling::Trilinear;

cv::Mat_<uint8_t> slice;
volume->sample(slice, coords, params);
```

Coordinates passed to `sample()` are base-scale coordinates in zarr order
`(z, y, x)`. When `params.level > 0`, `Volume` scales them to the requested
level before sampling.

### Updating Metadata

```cpp
utils::Json patch;
patch["name"] = "renamed-volume";
volume->updateMetadata(patch);
```

## Implementation Notes

- Shape validation checks physical zarr level shapes against metadata dimensions
  using power-of-two level indices, with limited per-level padding tolerance.
- Local zarr arrays may be stored as scale directories (`0`, `1`, `2`, ...)
  or as a root array for level 0.
- Remote `s3://` URLs are resolved to HTTPS. AWS SigV4 credentials are used
  when available, with anonymous fallback for public buckets if stale
  credentials are rejected.
- Cache settings are applied by resetting the shared cache; existing independent
  caches created by `createChunkCache()` are not modified.
- Region writes and raw chunk writes are local-only and invalidate the shared
  cache after successful mutation.
