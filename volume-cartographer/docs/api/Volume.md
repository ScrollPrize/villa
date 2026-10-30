# Volume Class API

The `Volume` class is the main API for opening and reading 3D zarr volumes. It
supports local OME-Zarr-like pyramids, remote HTTP/S3-backed zarr stores,
cached sampling, typed region reads and raw chunk reads. Volumes are read-only.

**Header:** `core/include/vc/core/types/Volume.hpp`
**Implementation:** `core/src/Volume.cpp`

## Overview

A `Volume` represents a 3D image dataset with:

- Volume metadata such as name, UUID, dimensions, voxel size, dtype, and format.
- Local or remote zarr pyramid access.
- Typed region reads in `uint8_t` and `uint16_t`.
- Raw local zarr chunk reads for tools that need chunk-level control.

Volumes are typically managed through a `VolumePkg` container, but can also be
opened directly.

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

## Factory Methods

```cpp
// Open an existing local volume.
static std::shared_ptr<Volume> New(std::filesystem::path path);

// Open an existing local volume and apply metadata over its meta.json.
static std::shared_ptr<Volume> New(std::filesystem::path path,
                                   const utils::Json& metadata);

// Open a remote HTTP or s3:// zarr volume. s3:// URLs are resolved to HTTPS.
static std::shared_ptr<Volume> NewFromUrl(
    const std::string& url,
    const vc::HttpAuth& auth = {});
```

Remote volumes always use the process-wide remote cache root configured by
VC3D.

## API Reference

### Metadata and Identity

| Method | Returns | Description |
|--------|---------|-------------|
| `id()` | `std::string` | Volume UUID from metadata |
| `name()` | `std::string` | Display name from metadata |
| `metadata()` | `const utils::Json&` | Current volume metadata |
| `path()` | `std::filesystem::path` | Local volume path; empty for remote volumes |
| `isRemote()` | `bool` | Whether the volume is backed by a remote zarr store |
| `remoteUrl()` | `const std::string&` | Normalized remote URL for remote volumes |
| `remoteAuth()` | `const vc::HttpAuth&` | Auth used for remote chunk reads |

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
| `dtype()` | `const std::string&` | NumPy typestr of the source (`\|u1`, `<u2`, `<f4`) |
| `numScales()` | `size_t` | Number of scale slots known to the volume |
| `baseScaleLevel()` | `int` | Working level: the source pyramid level opened as level 0 (remote `#vc-base-scale=N`), else `0`; the ID does not change with it |
| `hasScaleLevel(int level)` | `bool` | Whether a physical zarr array exists for the level |
| `presentScaleLevels()` | `std::vector<int>` | Physical scale levels |

`numScales()` can include missing scale slots. Check `hasScaleLevel()` or
`presentScaleLevels()` before reading a specific level.

### Opening the Source

`Volume` holds project metadata, not a reader. To read or sample, open the source
with vc_reader. Working level `L` is physical pyramid level `L + baseScaleLevel()`.

```cpp
auto options = volume->sourceOpenOptions();          // credentials, cache directory and limits
options.memoryCache = std::make_shared<vcr::MemoryCache>(bytes);
auto level = vcr::zarr::openPyramidLevel(volume->sourceLocation(),
                                         volume->baseScaleLevel() + L, options);
level.array->read_many(...);                         // or sample_linear, vcr::grid::sample
```

VC3D tools and viewers share opened levels and one memory cache through
`OpenedVolumes`.

### Utilities

| Method | Returns | Description |
|--------|---------|-------------|
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

### Opening a Remote Volume

```cpp
#include "vc/core/types/Volume.hpp"

auto volume = Volume::NewFromUrl("s3://bucket/path/to/volume.zarr");

if (volume->isRemote()) {
    std::cout << volume->remoteUrl() << std::endl;
}
```
