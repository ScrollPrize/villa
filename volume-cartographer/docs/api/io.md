# I/O API

This document covers file I/O utilities for image and volume data.

## Tiff API

Header: `vc/core/util/Tiff.hpp`

The Tiff module provides utilities for writing single-channel tiled TIFF files with LZW compression.

### writeTiff

Write a complete image to a tiled TIFF file in one call.

```cpp
void writeTiff(const std::filesystem::path& outPath,
               const cv::Mat& img,
               int cvType = -1,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f);
```

**Parameters:**
- `outPath` - Output file path
- `img` - Single-channel OpenCV Mat (CV_8UC1, CV_16UC1, or CV_32FC1)
- `cvType` - Output type. Use -1 to preserve input type, or specify CV_8UC1/CV_16UC1/CV_32FC1 to convert
- `tileW` - Tile width in pixels (default 1024)
- `tileH` - Tile height in pixels (default 1024)
- `padValue` - Value for padding partial edge tiles (used for float; int types use 0)

**Type Conversion:**
When converting between types, values are scaled to preserve full dynamic range:
- 8U ↔ 16U: scale by 257 (255 × 257 = 65535)
- 8U ↔ 32F: scale by 1/255 (maps 0-255 to 0.0-1.0)
- 16U ↔ 32F: scale by 1/65535 (maps 0-65535 to 0.0-1.0)

**Example:**
```cpp
#include "vc/core/util/Tiff.hpp"

// Write a 16-bit image
cv::Mat img = cv::imread("input.tif", cv::IMREAD_UNCHANGED);
writeTiff("output.tif", img);

// Convert 16-bit to 8-bit while writing
writeTiff("output_8bit.tif", img, CV_8UC1);

// Write 32-bit float with custom tile size
cv::Mat floatImg;
img.convertTo(floatImg, CV_32FC1, 1.0/65535.0);
writeTiff("output_float.tif", floatImg, -1, 512, 512, -1.0f);
```

### TiffWriter

Class for incremental tiled TIFF writing. Useful for parallel tile generation or streaming data.

```cpp
class TiffWriter {
public:
    TiffWriter(const std::filesystem::path& path,
               uint32_t width, uint32_t height,
               int cvType,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f);

    void writeTile(uint32_t x0, uint32_t y0, const cv::Mat& tile);
    void close();

    bool isOpen() const;
    uint32_t width() const;
    uint32_t height() const;
    uint32_t tileWidth() const;
    uint32_t tileHeight() const;
    int cvType() const;
};
```

**Constructor Parameters:**
- `path` - Output file path
- `width`, `height` - Full image dimensions
- `cvType` - Pixel type (CV_8UC1, CV_16UC1, or CV_32FC1)
- `tileW`, `tileH` - Tile dimensions (default 1024x1024)
- `padValue` - Padding value for partial tiles

**Methods:**
- `writeTile(x0, y0, tile)` - Write a tile at position (x0, y0). Position should be tile-aligned. Tile can be smaller than tile size for edge tiles.
- `close()` - Explicitly close the file (also called by destructor)

**Example:**
```cpp
#include "vc/core/util/Tiff.hpp"

// Create a writer for a 4096x4096 16-bit image
TiffWriter writer("large_output.tif", 4096, 4096, CV_16UC1);

// Write tiles (can be done in parallel with proper synchronization)
for (uint32_t y = 0; y < 4096; y += 1024) {
    for (uint32_t x = 0; x < 4096; x += 1024) {
        cv::Mat tile = generateTile(x, y);  // Your tile generation
        writer.writeTile(x, y, tile);
    }
}

writer.close();
```

---

## ChunkCache

Header: `vc/core/util/ChunkCache.hpp`

Thread-safe LRU cache for volume chunks. Uses generation-based eviction - when full, removes the oldest 10% of entries.

```cpp
template<typename T>  // T = uint8_t or uint16_t
class ChunkCache {
public:
    explicit ChunkCache(size_t byteSize);  // Maximum cache size in bytes
    // ...
};
```

### Methods

#### groupIdx

```cpp
int groupIdx(const std::string& name);
```

Get or create a unique group index for a dataset. Used to distinguish chunks from different datasets in the same cache.

- `name` - Unique identifier (typically the dataset path)
- Returns an integer index starting from 1

#### put

```cpp
void put(const cv::Vec4i& key, xt::xarray<T>* ar);
```

Store a chunk in the cache. Transfers ownership of the array to the cache.

- `key` - Cache key as `{group_idx, chunk_z, chunk_y, chunk_x}`
- `ar` - Pointer to chunk data (ownership transferred)

When the cache is full, the oldest 10% of entries are evicted.

#### get

```cpp
std::shared_ptr<xt::xarray<T>> get(const cv::Vec4i& key);
```

Retrieve a chunk from the cache. Updates the entry's generation for LRU tracking.

- `key` - Cache key
- Returns shared pointer to the chunk, or nullptr if not found

#### has

```cpp
bool has(const cv::Vec4i& idx);
```

Check if a chunk exists in the cache.

#### reset

```cpp
void reset();
```

Clear all cached data and reset internal state.

---

## Zarr Volume Access

Header: `vc/core/util/Slicing.hpp`

High-level functions for reading data from zarr volumes using ChunkCache.

### readInterpolated3D

Read values from a zarr volume at arbitrary floating-point coordinates with trilinear interpolation.

```cpp
void readInterpolated3D(cv::Mat_<uint8_t>& out,
                        z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords,
                        ChunkCache<uint8_t>* cache,
                        bool nearest_neighbor = false);

void readInterpolated3D(cv::Mat_<uint16_t>& out,
                        z5::Dataset* ds,
                        const cv::Mat_<cv::Vec3f>& coords,
                        ChunkCache<uint16_t>* cache,
                        bool nearest_neighbor = false);
```

**Parameters:**
- `out` - Output image (same size as coords)
- `ds` - Zarr dataset pointer (from `Volume::zarrDataset()`)
- `coords` - Coordinates as Vec3f containing (z, y, x) for each output pixel. Negative values indicate invalid/skip.
- `cache` - Chunk cache (required)
- `nearest_neighbor` - Use nearest-neighbor instead of trilinear interpolation

**Example:**
```cpp
#include "vc/core/util/Slicing.hpp"
#include "vc/core/types/Volume.hpp"

auto vol = Volume::New("/path/to/volume");
z5::Dataset* ds = vol->zarrDataset(0);

ChunkCache<uint8_t> cache(4ULL * 1024 * 1024 * 1024);

// Create coordinate grid for a slice
cv::Mat_<cv::Vec3f> coords(1000, 1000);
float z = 500.0f;
for (int y = 0; y < 1000; y++) {
    for (int x = 0; x < 1000; x++) {
        coords(y, x) = cv::Vec3f(z, y + 100.0f, x + 100.0f);
    }
}

// Read interpolated slice
cv::Mat_<uint8_t> slice;
readInterpolated3D(slice, ds, coords, &cache);

// Or use nearest-neighbor for speed
cv::Mat_<uint8_t> sliceNN;
readInterpolated3D(sliceNN, ds, coords, &cache, true);
```

### readArea3D

Read a contiguous 3D block from a zarr volume.

```cpp
void readArea3D(xt::xtensor<uint8_t, 3, xt::layout_type::column_major>& out,
                const cv::Vec3i& offset,
                z5::Dataset* ds,
                ChunkCache<uint8_t>* cache);

void readArea3D(xt::xtensor<uint16_t, 3, xt::layout_type::column_major>& out,
                const cv::Vec3i& offset,
                z5::Dataset* ds,
                ChunkCache<uint16_t>* cache);
```

**Parameters:**
- `out` - Pre-sized output tensor (determines the region size)
- `offset` - Starting position as (z, y, x)
- `ds` - Zarr dataset pointer
- `cache` - Chunk cache

**Example:**
```cpp
#include "vc/core/util/Slicing.hpp"

auto vol = Volume::New("/path/to/volume");
z5::Dataset* ds = vol->zarrDataset(0);

ChunkCache<uint8_t> cache(4ULL * 1024 * 1024 * 1024);

// Read a 128x128x128 block starting at (100, 200, 300)
xt::xtensor<uint8_t, 3, xt::layout_type::column_major> block;
block.resize({128, 128, 128});

cv::Vec3i offset = {100, 200, 300};  // z, y, x
readArea3D(block, offset, ds, &cache);

// Access values: block(z, y, x)
uint8_t val = block(64, 64, 64);
```

### Thread Safety

When accessing the cache from multiple threads, use the public mutex:

```cpp
ChunkCache<uint8_t> cache(4ULL * 1024 * 1024 * 1024);

// Reading (shared lock)
{
    std::shared_lock<std::shared_mutex> lock(cache.mutex);
    if (cache.has(key)) {
        auto chunk = cache.get(key);
        // use chunk...
    }
}

// Writing (exclusive lock)
{
    std::unique_lock<std::shared_mutex> lock(cache.mutex);
    if (!cache.has(key)) {
        cache.put(key, newChunk);
    }
}
```

The `readInterpolated3D` and `readArea3D` functions handle thread safety internally using OpenMP parallelization.
