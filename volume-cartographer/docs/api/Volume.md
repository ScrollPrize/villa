# Volume Class API

The `Volume` class provides access to 3D volumetric image data stored in zarr format with multi-resolution support.

**Header:** `core/include/vc/core/types/Volume.hpp`
**Implementation:** `core/src/Volume.cpp`

## Overview

A Volume represents a 3D image dataset with:
- Metadata (name, UUID, dimensions, voxel size)
- Multi-scale zarr dataset access (pyramid levels)
- File-based persistence via `meta.json`

Volumes are typically managed through a `VolumePkg` container but can be loaded standalone.

## Coordinate Ordering

Zarr datasets use **ZYX ordering** internally:
- `shape[0]` = Z (slices)
- `shape[1]` = Y (height)
- `shape[2]` = X (width)

The `meta.json` maps dimensions as:
- `slices` = Z
- `height` = Y
- `width` = X

## Factory Methods

```cpp
// Load existing volume
static std::shared_ptr<Volume> Volume::New(std::filesystem::path path);

// Create new volume
static std::shared_ptr<Volume> Volume::New(
    std::filesystem::path path,
    std::string uuid,
    std::string name
);
```

## API Reference

### Metadata

| Method | Returns | Description |
|--------|---------|-------------|
| `id()` | `std::string` | Volume UUID |
| `name()` | `std::string` | Display name |
| `path()` | `std::filesystem::path` | Volume directory |
| `setName(const std::string& n)` | `void` | Update name (call `saveMetadata()` to persist) |
| `saveMetadata()` | `void` | Write metadata to `meta.json` |

### Dimensions

| Method | Returns | Description |
|--------|---------|-------------|
| `sliceWidth()` | `int` | X dimension (width) |
| `sliceHeight()` | `int` | Y dimension (height) |
| `numSlices()` | `int` | Z dimension (depth) |
| `shape()` | `std::array<int, 3>` | Dimensions as {X, Y, Z} |
| `voxelSize()` | `double` | Physical voxel size (typically nm) |

### Zarr Access (use sparingly, chunkcache should be preferred in nearly all cases)

| Method | Returns | Description |
|--------|---------|-------------|
| `zarrDataset(int level = 0)` | `z5::Dataset*` | Dataset at scale level (0 = full res) |
| `numScales()` | `size_t` | Number of pyramid levels |

### Utilities

| Method | Returns | Description |
|--------|---------|-------------|
| `static checkDir(path)` | `bool` | Check if directory is a valid volume |

## meta.json Structure

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

## Usage Examples

### Loading a Volume

```cpp
auto volume = Volume::New("/path/to/volume");

int w = volume->sliceWidth();   // X
int h = volume->sliceHeight();  // Y
int d = volume->numSlices();    // Z

std::cout << volume->name() << ": " << w << "x" << h << "x" << d << std::endl;
```

### Reading Voxel Data

Use `ChunkCache` for reading voxel data. It provides thread-safe LRU caching which is essential for interactive use and any access pattern with repeated or nearby reads:

```cpp
#include "vc/core/util/ChunkCache.hpp"
#include "vc/core/util/Slicing.hpp"

ChunkCache<uint8_t> cache(1ULL * 1024 * 1024 * 1024);  // 1GB cache
z5::Dataset* ds = volume->zarrDataset(0);

// Read interpolated 3D data at arbitrary points
readInterpolated3D(outputMat, ds, points3d, &cache);

// Read an interpolated slice
readInterpolatedSlice(outputMat, ds, origin, basis_u, basis_v, &cache);
```

> **Note:** Direct z5 reads (`z5::multiarray::readSubarray`) should only be used in batch processing tools that read slices sequentially without re-reads. For all other cases, use ChunkCache.

### Accessing Dataset Metadata

```cpp
// Full resolution
z5::Dataset* ds = volume->zarrDataset(0);

// Downsampled levels (2x, 4x, 8x, ...)
z5::Dataset* ds_2x = volume->zarrDataset(1);
z5::Dataset* ds_4x = volume->zarrDataset(2);

// Check available scales
for (size_t i = 0; i < volume->numScales(); ++i) {
    auto* ds = volume->zarrDataset(i);
    const auto& shape = ds->shape();
    std::cout << "Level " << i << ": "
              << shape[2] << "x" << shape[1] << "x" << shape[0] << std::endl;
}
```

### Via VolumePkg

```cpp
auto vpkg = VolumePkg::New("/path/to/volpkg");

// Get volume by ID
auto volume = vpkg->volume("volume-uuid");

// List all volumes
for (const auto& id : vpkg->volumeIDs()) {
    auto vol = vpkg->volume(id);
    std::cout << vol->name() << std::endl;
}
```

### Modifying Metadata

```cpp
auto volume = Volume::New(path);
volume->setName("New Name");
volume->saveMetadata();  // Persist to disk
```

## Implementation Notes

- Scale levels assume power-of-2 downsampling (2x, 4x, 8x, ...)
- Zarr datasets are opened lazily when `format: "zarr"` is in metadata
- Only uint8 and uint16 data types are currently supported
- Metadata changes are only persisted when `saveMetadata()` is called