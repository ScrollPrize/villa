#pragma once

#include <opencv2/core.hpp>
#include <filesystem>
#include <vector>
#include <cstdint>
#include <tiffio.h>

// Write single-channel image (8U, 16U, 32F) as tiled TIFF with LZW compression
// cvType: output type (-1 = same as input). If different, values are scaled:
//         8U↔16U: scale by 257, 8U↔32F: scale by 1/255, 16U↔32F: scale by 1/65535
// padValue: value for padding partial tiles (default -1.0f, used for float; int types use 0)
void writeTiff(const std::filesystem::path& outPath,
               const cv::Mat& img,
               int cvType = -1,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f);

// Class for incremental tiled TIFF writing
// Useful for writing tiles in parallel or from streaming data
class TiffWriter {
public:
    // Open a new TIFF file for tiled writing
    // cvType: CV_8UC1, CV_16UC1, or CV_32FC1
    // padValue: value for padding partial tiles (used for float; int types use 0)
    TiffWriter(const std::filesystem::path& path,
               uint32_t width, uint32_t height,
               int cvType,
               uint32_t tileW = 1024,
               uint32_t tileH = 1024,
               float padValue = -1.0f);

    ~TiffWriter();

    // Non-copyable
    TiffWriter(const TiffWriter&) = delete;
    TiffWriter& operator=(const TiffWriter&) = delete;

    // Movable
    TiffWriter(TiffWriter&& other) noexcept;
    TiffWriter& operator=(TiffWriter&& other) noexcept;

    // Write a tile at the given position (should be tile-aligned)
    // tile: cv::Mat of appropriate type, can be smaller than tile size for edge tiles
    void writeTile(uint32_t x0, uint32_t y0, const cv::Mat& tile);

    // Explicitly close the file (also called by destructor)
    void close();

    // Check if file is open
    bool isOpen() const { return _tiff != nullptr; }

    // Accessors
    uint32_t width() const { return _width; }
    uint32_t height() const { return _height; }
    uint32_t tileWidth() const { return _tileW; }
    uint32_t tileHeight() const { return _tileH; }
    int cvType() const { return _cvType; }

private:
    TIFF* _tiff = nullptr;
    uint32_t _width = 0;
    uint32_t _height = 0;
    uint32_t _tileW = 0;
    uint32_t _tileH = 0;
    int _cvType = 0;
    int _elemSize = 0;
    float _padValue = -1.0f;
    std::vector<uint8_t> _tileBuf;  // Reusable tile buffer
    std::filesystem::path _path;     // For error messages
};
