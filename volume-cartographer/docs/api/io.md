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

## Zarr volumes

volume-cartographer has no Zarr implementation of its own: open, read, sample and
write volumes through vc_reader (`vc_reader/zarr.hpp`, `vc_reader/array.hpp`,
`vc_reader/grid.hpp`). `vc/core/types/Volume.hpp` says where a volume's source is;
see [Volume](Volume.md).
